module AtomAttentionModule

using Random
using ConcreteStructs
using Flux: @layer
import ..Primitives: LinearNoBias, LayerNormFirst
import ..TransformerBlocks: DiffusionTransformer

export AtomAttentionEncoder, AtomAttentionDecoder

_has_feat(input::AbstractDict{<:AbstractString, <:Any}, key::String) = haskey(input, key)
_has_feat(input::NamedTuple, key::String) = hasproperty(input, Symbol(key))
_feat(input::AbstractDict{<:AbstractString, <:Any}, key::String) = input[key]
_feat(input::NamedTuple, key::String) = getproperty(input, Symbol(key))

# Convert to features-first: (features, N_atom)
function _matrix_f32(x, name::String)
    if name == "ref_atom_name_chars" && x isa AbstractArray && ndims(x) == 3 && size(x, 2) == 4 && size(x, 3) == 64
        # Match Python flatten order: (pos-1)*64 + bucket.
        xp = permutedims(Float32.(x), (1, 3, 2))
        m = reshape(xp, size(x, 1), 256)
        return permutedims(m)  # (256, N_atom)
    end
    x isa AbstractMatrix || error("$name must be a matrix, got $(typeof(x))")
    return Float32.(x)  # caller provides features-first: (features, N)
end

function _vector_i(x, name::String, n::Int)
    x isa AbstractVector || error("$name must be a vector, got $(typeof(x))")
    length(x) == n || error("$name length mismatch: expected $n, got $(length(x))")
    return Int.(x)
end

# Convert vector to features-first: (1, N)
function _column_f32(x, name::String, n::Int)
    if x isa AbstractVector
        length(x) == n || error("$name length mismatch: expected $n, got $(length(x))")
        return reshape(Float32.(x), 1, n)  # (1, N) — features-first
    elseif x isa AbstractMatrix && size(x, 1) == n && size(x, 2) == 1
        return permutedims(Float32.(x))  # (1, N) — features-first
    end
    error("$name must be a vector or [N,1], got $(typeof(x))")
end

# Features-first: x_token (c, N_token) → (c, N_atom) via indexing
function _broadcast_token_to_atom(
    x_token::AbstractMatrix{<:Real},
    atom_to_token_idx::AbstractVector{<:Integer},
)
    idx = atom_to_token_idx .+ 1
    return Float32.(x_token[:, idx])  # (c, N_atom) — GPU-compatible advanced indexing
end

function _validate_local_window(n_queries::Int, n_keys::Int)
    n_queries > 0 || error("n_queries must be positive.")
    n_keys >= n_queries || error("n_keys must be >= n_queries.")
    n_queries % 2 == 0 || error("n_queries must be even.")
    n_keys % 2 == 0 || error("n_keys must be even.")
    return nothing
end

# Features-first: z_token (c, N_token, N_token) → (c, n_q, n_k, n_trunks)
function _broadcast_token_pair_to_local_trunks(
    z_token::AbstractArray{<:Real,3},
    atom_to_token_idx::AbstractVector{<:Integer},
    n_queries::Int,
    n_keys::Int,
)
    _validate_local_window(n_queries, n_keys)
    n_atom = length(atom_to_token_idx)
    c = size(z_token, 1)
    n_trunks = cld(n_atom, n_queries)
    z = Array(Float32.(z_token))
    ati = Array(atom_to_token_idx)
    out = zeros(Float32, c, n_queries, n_keys, n_trunks)

    left = div(n_keys - n_queries, 2)
    q_pad = n_trunks * n_queries - n_atom
    k_pad_right = floor(Int, (n_trunks - 0.5) * n_queries + n_keys / 2 - n_atom + 0.5)
    q_padded_len = n_atom + q_pad
    k_padded_len = n_atom + left + k_pad_right

    for b in 1:n_trunks
        q_start = (b - 1) * n_queries + 1
        k_start = (b - 1) * n_queries + 1
        for ql in 1:n_queries
            q_pad_idx = q_start + ql - 1
            (1 <= q_pad_idx <= q_padded_len) || error("query padded index out of range.")
            q_atom_idx = q_pad_idx <= n_atom ? q_pad_idx : 0
            ti = q_atom_idx == 0 ? 1 : Int(ati[q_atom_idx]) + 1
            (1 <= ti <= size(z, 2)) || error("atom_to_token_idx query out of range.")
            for kl in 1:n_keys
                k_pad_idx = k_start + kl - 1
                (1 <= k_pad_idx <= k_padded_len) || error("key padded index out of range.")
                k_atom_idx = k_pad_idx - left
                k_atom_idx = (1 <= k_atom_idx <= n_atom) ? k_atom_idx : 0
                tj = k_atom_idx == 0 ? 1 : Int(ati[k_atom_idx]) + 1
                (1 <= tj <= size(z, 3)) || error("atom_to_token_idx key out of range.")
                @inbounds out[:, ql, kl, b] .= z[:, ti, tj]
            end
        end
    end
    return copyto!(similar(z_token, Float32, size(out)...), out)
end

# Features-first: x_atom (c, N_atom) → (c, N_token)
# Uses one-hot matmul for GPU compatibility (following BoltzGen pattern).
function _aggregate_atom_to_token_mean(
    x_atom::AbstractMatrix{<:Real},
    atom_to_token_idx::AbstractVector{<:Integer},
    n_token::Int,
)
    c, n_atom = size(x_atom)
    length(atom_to_token_idx) == n_atom || error("atom_to_token_idx length mismatch.")
    x_f = Float32.(x_atom)
    idx_cpu = Array(Int.(atom_to_token_idx)) .+ 1  # ensure CPU, 0→1 indexed
    # Build one-hot mapping on CPU: (N_atom, N_token)
    oh_cpu = zeros(Float32, n_atom, n_token)
    @inbounds for i in 1:n_atom
        1 <= idx_cpu[i] <= n_token || error("atom_to_token_idx[$i] out of range")
        oh_cpu[i, idx_cpu[i]] = 1f0
    end
    # Transfer to same device as x_atom
    oh = copyto!(similar(x_f, Float32, n_atom, n_token), oh_cpu)
    # Matrix multiply: (c, N_atom) × (N_atom, N_token) = (c, N_token)
    sums = x_f * oh
    # Counts per token: (1, N_token)
    counts = max.(sum(oh; dims=1), 1f0)
    return sums ./ counts
end

function _small_mlp!(
    p::AbstractArray{Float32},
    linear1,
    linear2,
    linear3,
)
    x = max.(p, 0f0)
    x = linear1(x)
    x = max.(x, 0f0)
    x = linear2(x)
    x = max.(x, 0f0)
    p .+= linear3(x)
    return p
end

# Features-first: ref_pos (3, N_atom), ref_charge (1, N_atom), etc.
# Returns c_l (c_atom, N_atom), p_local (c_atompair, n_q, n_k, n_trunks)
function _base_atom_features(
    ref_pos::AbstractMatrix{Float32},
    ref_charge::AbstractMatrix{Float32},
    ref_mask::AbstractMatrix{Float32},
    ref_element::AbstractMatrix{Float32},
    ref_atom_name_chars::AbstractMatrix{Float32},
    ref_space_uid::AbstractVector{<:Integer},
    linear_no_bias_ref_pos,
    linear_no_bias_ref_charge,
    linear_no_bias_f,
    linear_no_bias_d,
    linear_no_bias_invd,
    linear_no_bias_v,
    n_queries::Int,
    n_keys::Int,
)
    _validate_local_window(n_queries, n_keys)
    n_atom = size(ref_pos, 2)  # features-first: (3, N_atom)

    c_l = linear_no_bias_ref_pos(ref_pos) + linear_no_bias_ref_charge(asinh.(ref_charge))
    # cat along dim 1 (features): ref_mask (1,N), ref_element (128,N), ref_atom_name_chars (256,N) → (385,N)
    c_l .= (c_l + linear_no_bias_f(cat(ref_mask, ref_element, ref_atom_name_chars; dims = 1))) .* ref_mask

    n_trunks = cld(n_atom, n_queries)
    rp = Array(ref_pos)  # (3, N_atom)
    rsu = Array(ref_space_uid)
    # Features-first: (c, n_q, n_k, n_trunks)
    d_local = zeros(Float32, 3, n_queries, n_keys, n_trunks)
    v_local = zeros(Float32, 1, n_queries, n_keys, n_trunks)
    inv_local = zeros(Float32, 1, n_queries, n_keys, n_trunks)
    mask_local = zeros(Float32, 1, n_queries, n_keys, n_trunks)
    left = div(n_keys - n_queries, 2)
    q_pad = n_trunks * n_queries - n_atom
    k_pad_right = floor(Int, (n_trunks - 0.5) * n_queries + n_keys / 2 - n_atom + 0.5)
    q_padded_len = n_atom + q_pad
    k_padded_len = n_atom + left + k_pad_right

    for b in 1:n_trunks
        q_start = (b - 1) * n_queries + 1
        k_start = (b - 1) * n_queries + 1
        for ql in 1:n_queries
            q_pad_idx = q_start + ql - 1
            (1 <= q_pad_idx <= q_padded_len) || error("query padded index out of range.")
            q_idx = q_pad_idx <= n_atom ? q_pad_idx : 0
            xi1 = q_idx == 0 ? 0f0 : rp[1, q_idx]
            xi2 = q_idx == 0 ? 0f0 : rp[2, q_idx]
            xi3 = q_idx == 0 ? 0f0 : rp[3, q_idx]
            ui = q_idx == 0 ? 0 : rsu[q_idx]
            for kl in 1:n_keys
                k_pad_idx = k_start + kl - 1
                (1 <= k_pad_idx <= k_padded_len) || error("key padded index out of range.")
                k_idx = k_pad_idx - left
                k_idx = (1 <= k_idx <= n_atom) ? k_idx : 0
                xk1 = k_idx == 0 ? 0f0 : rp[1, k_idx]
                xk2 = k_idx == 0 ? 0f0 : rp[2, k_idx]
                xk3 = k_idx == 0 ? 0f0 : rp[3, k_idx]
                uk = k_idx == 0 ? 0 : rsu[k_idx]
                dx = xi1 - xk1
                dy = xi2 - xk2
                dz = xi3 - xk3
                same = ui == uk ? 1f0 : 0f0
                valid = (q_idx != 0 && k_idx != 0) ? 1f0 : 0f0
                @inbounds d_local[1, ql, kl, b] = dx
                @inbounds d_local[2, ql, kl, b] = dy
                @inbounds d_local[3, ql, kl, b] = dz
                @inbounds v_local[1, ql, kl, b] = same
                @inbounds inv_local[1, ql, kl, b] = inv(1f0 + dx * dx + dy * dy + dz * dz)
                @inbounds mask_local[1, ql, kl, b] = valid
            end
        end
    end

    _ref = linear_no_bias_d.weight
    d_dev = copyto!(similar(_ref, Float32, size(d_local)...), d_local)
    v_dev = copyto!(similar(_ref, Float32, size(v_local)...), v_local)
    inv_dev = copyto!(similar(_ref, Float32, size(inv_local)...), inv_local)
    mask_dev = copyto!(similar(_ref, Float32, size(mask_local)...), mask_local)

    p_local = (linear_no_bias_d(d_dev) .* v_dev) .* mask_dev
    p_local .+= linear_no_bias_invd(inv_dev) .* v_dev
    p_local .+= linear_no_bias_v(v_dev)
    return c_l, p_local
end

# Features-first: p_local (c_atompair, n_q, n_k, n_trunks), c_l (c_atom, N_atom)
function _mix_pair_with_single!(
    p_local::AbstractArray{Float32,4},
    c_l::AbstractMatrix{Float32},
    linear_no_bias_cl,
    linear_no_bias_cm,
    small_mlp_1,
    small_mlp_2,
    small_mlp_3,
    n_queries::Int,
    n_keys::Int,
)
    _validate_local_window(n_queries, n_keys)
    n_atom = size(c_l, 2)  # features-first: (c_atom, N_atom)
    c_relu = max.(c_l, 0f0)
    c_q = Array(linear_no_bias_cl(c_relu))  # (c_atompair, N_atom)
    c_k = Array(linear_no_bias_cm(c_relu))  # (c_atompair, N_atom)
    n_trunks = size(p_local, 4)
    size(p_local, 2) == n_queries || error("p_local query window mismatch.")
    size(p_local, 3) == n_keys || error("p_local key window mismatch.")
    left = div(n_keys - n_queries, 2)
    q_pad = n_trunks * n_queries - n_atom
    k_pad_right = floor(Int, (n_trunks - 0.5) * n_queries + n_keys / 2 - n_atom + 0.5)
    q_padded_len = n_atom + q_pad
    k_padded_len = n_atom + left + k_pad_right

    p_cpu = Array(p_local)
    for b in 1:n_trunks
        q_start = (b - 1) * n_queries + 1
        k_start = (b - 1) * n_queries + 1
        for ql in 1:n_queries
            q_pad_idx = q_start + ql - 1
            (1 <= q_pad_idx <= q_padded_len) || error("query padded index out of range.")
            q_idx = q_pad_idx <= n_atom ? q_pad_idx : 0
            for kl in 1:n_keys
                k_pad_idx = k_start + kl - 1
                (1 <= k_pad_idx <= k_padded_len) || error("key padded index out of range.")
                k_idx = k_pad_idx - left
                k_idx = (1 <= k_idx <= n_atom) ? k_idx : 0

                if q_idx == 0 && k_idx == 0
                    continue
                elseif q_idx == 0
                    @inbounds p_cpu[:, ql, kl, b] .+= c_k[:, k_idx]
                elseif k_idx == 0
                    @inbounds p_cpu[:, ql, kl, b] .+= c_q[:, q_idx]
                else
                    @inbounds p_cpu[:, ql, kl, b] .+= c_q[:, q_idx] .+ c_k[:, k_idx]
                end
            end
        end
    end

    copyto!(p_local, p_cpu)
    _small_mlp!(p_local, small_mlp_1, small_mlp_2, small_mlp_3)
    return p_local
end

# ---------------------------------------------------------------------------
# AtomAttentionEncoder
# ---------------------------------------------------------------------------
@concrete struct AtomAttentionEncoder
    has_coords
    c_atom
    c_atompair
    c_token
    c_s
    c_z
    n_queries
    n_keys
    linear_no_bias_ref_pos
    linear_no_bias_ref_charge
    linear_no_bias_f
    linear_no_bias_d
    linear_no_bias_invd
    linear_no_bias_v
    layernorm_s
    linear_no_bias_s
    layernorm_z
    linear_no_bias_z
    linear_no_bias_r
    linear_no_bias_cl
    linear_no_bias_cm
    small_mlp_1
    small_mlp_2
    small_mlp_3
    atom_transformer
    linear_no_bias_q
end
@layer AtomAttentionEncoder

function AtomAttentionEncoder(
    c_token::Int;
    has_coords::Bool,
    c_atom::Int = 128,
    c_atompair::Int = 16,
    c_s::Int = 384,
    c_z::Int = 128,
    n_blocks::Int = 3,
    n_heads::Int = 4,
    n_queries::Int = 32,
    n_keys::Int = 128,
    rng::AbstractRNG = Random.default_rng(),
)
    ln_s = has_coords ? LayerNormFirst(c_s) : nothing
    lin_s = has_coords ? LinearNoBias(c_s, c_atom) : nothing
    ln_z = has_coords ? LayerNormFirst(c_z) : nothing
    lin_z = has_coords ? LinearNoBias(c_z, c_atompair) : nothing
    lin_r = has_coords ? LinearNoBias(3, c_atom) : nothing
    return AtomAttentionEncoder(
        has_coords,
        c_atom,
        c_atompair,
        c_token,
        c_s,
        c_z,
        n_queries,
        n_keys,
        LinearNoBias(3, c_atom),              # ref_pos: (3, N) → (c_atom, N)
        LinearNoBias(1, c_atom),              # ref_charge: (1, N) → (c_atom, N)
        LinearNoBias(1 + 128 + 256, c_atom),  # f: (385, N) → (c_atom, N)
        LinearNoBias(3, c_atompair),           # d: (3, q, k, t) → (c_atompair, q, k, t)
        LinearNoBias(1, c_atompair),           # invd
        LinearNoBias(1, c_atompair),           # v
        ln_s,
        lin_s,
        ln_z,
        lin_z,
        lin_r,
        LinearNoBias(c_atom, c_atompair),      # cl
        LinearNoBias(c_atom, c_atompair),      # cm
        LinearNoBias(c_atompair, c_atompair),  # small_mlp layers
        LinearNoBias(c_atompair, c_atompair),
        LinearNoBias(c_atompair, c_atompair),
        DiffusionTransformer(
            c_atom,
            c_atom,
            c_atompair;
            n_blocks = n_blocks,
            n_heads = n_heads,
            cross_attention_mode = true,
            rng = rng,
        ),
        LinearNoBias(c_atom, c_token),         # q: (c_atom, N) → (c_token, N)
    )
end

function (encoder::AtomAttentionEncoder)(
    input_feature_dict;
    r_l = nothing,
    s = nothing,
    z = nothing,
)
    for k in ("ref_pos", "ref_mask", "ref_element", "ref_atom_name_chars", "ref_space_uid", "atom_to_token_idx")
        _has_feat(input_feature_dict, k) || error("Missing input feature '$k' for AtomAttentionEncoder.")
    end
    ref_pos = _matrix_f32(_feat(input_feature_dict, "ref_pos"), "ref_pos")  # (3, N_atom)
    n_atom = size(ref_pos, 2)
    size(ref_pos, 1) == 3 || error("ref_pos must be (3, N_atom)")
    ref_charge = _has_feat(input_feature_dict, "ref_charge") ?
                 _column_f32(_feat(input_feature_dict, "ref_charge"), "ref_charge", n_atom) :
                 fill!(similar(ref_pos, Float32, 1, n_atom), 0f0)
    ref_mask = _column_f32(_feat(input_feature_dict, "ref_mask"), "ref_mask", n_atom)
    ref_element = _matrix_f32(_feat(input_feature_dict, "ref_element"), "ref_element")
    size(ref_element) == (128, n_atom) || error("ref_element must be (128, N_atom)")
    ref_atom_name_chars = _matrix_f32(_feat(input_feature_dict, "ref_atom_name_chars"), "ref_atom_name_chars")
    size(ref_atom_name_chars) == (256, n_atom) || error("ref_atom_name_chars must be (256, N_atom)")
    ref_space_uid = _vector_i(_feat(input_feature_dict, "ref_space_uid"), "ref_space_uid", n_atom)
    atom_to_token_idx = _vector_i(_feat(input_feature_dict, "atom_to_token_idx"), "atom_to_token_idx", n_atom)

    # Transfer matrix features to model device (GPU if model is on GPU).
    # Int vectors (ref_space_uid, atom_to_token_idx) stay CPU — used in scalar loops.
    _dev_ref = encoder.linear_no_bias_ref_pos.weight
    _to_dev(x::AbstractArray{Float32}) = copyto!(similar(_dev_ref, Float32, size(x)...), x)
    ref_pos = _to_dev(ref_pos)
    ref_charge = _to_dev(ref_charge)
    ref_mask = _to_dev(ref_mask)
    ref_element = _to_dev(ref_element)
    ref_atom_name_chars = _to_dev(ref_atom_name_chars)

    n_token = maximum(atom_to_token_idx) + 1

    c_base, p_base = _base_atom_features(
        ref_pos,
        ref_charge,
        ref_mask,
        ref_element,
        ref_atom_name_chars,
        ref_space_uid,
        encoder.linear_no_bias_ref_pos,
        encoder.linear_no_bias_ref_charge,
        encoder.linear_no_bias_f,
        encoder.linear_no_bias_d,
        encoder.linear_no_bias_invd,
        encoder.linear_no_bias_v,
        encoder.n_queries,
        encoder.n_keys,
    )

    if !encoder.has_coords
        c_l = copy(c_base)      # (c_atom, N_atom)
        p_trunk = copy(p_base)  # (c_atompair, n_q, n_k, n_trunks)
        _mix_pair_with_single!(
            p_trunk,
            c_l,
            encoder.linear_no_bias_cl,
            encoder.linear_no_bias_cm,
            encoder.small_mlp_1,
            encoder.small_mlp_2,
            encoder.small_mlp_3,
            encoder.n_queries,
            encoder.n_keys,
        )
        q_l = encoder.atom_transformer(c_l, c_l, p_trunk; n_queries = encoder.n_queries, n_keys = encoder.n_keys)
        a = _aggregate_atom_to_token_mean(max.(encoder.linear_no_bias_q(q_l), 0f0), atom_to_token_idx, n_token)
        return a, q_l, c_l, p_trunk
    end

    # has_coords = true path
    # r_l: (3, N_atom, N_sample), s: (c_s, N_token, N_sample), z: (c_z, N_token, N_token, N_sample)
    r_l isa AbstractArray{<:Real,3} || error("AtomAttentionEncoder(has_coords=true) requires r_l with shape (3, N_atom, N_sample).")
    s isa AbstractArray{<:Real,3} || error("AtomAttentionEncoder(has_coords=true) requires s with shape (c_s, N_token, N_sample).")
    z isa AbstractArray{<:Real,4} || error("AtomAttentionEncoder(has_coords=true) requires z with shape (c_z, N_token, N_token, N_sample).")

    r_f = Float32.(r_l)
    s_f = Float32.(s)
    z_f = Float32.(z)
    n_sample = size(r_f, 3)
    size(r_f, 1) == 3 || error("r_l first dim must be 3.")
    size(r_f, 2) == n_atom || error("r_l atom axis mismatch.")
    size(s_f, 2) == n_token || error("s token axis mismatch.")
    size(s_f, 3) == n_sample || error("s sample axis mismatch.")
    size(z_f, 2) == n_token || error("z token axis mismatch.")
    size(z_f, 3) == n_token || error("z token axis mismatch.")
    size(z_f, 4) == n_sample || error("z sample axis mismatch.")

    # Features-first, sample-last output buffers
    a_out = similar(c_base, Float32, encoder.c_token, n_token, n_sample)
    q_skip = similar(c_base, Float32, encoder.c_atom, n_atom, n_sample)
    c_skip = similar(c_base, Float32, encoder.c_atom, n_atom, n_sample)
    n_trunks = cld(n_atom, encoder.n_queries)
    p_skip = similar(c_base, Float32, encoder.c_atompair, encoder.n_queries, encoder.n_keys, n_trunks, n_sample)

    for i in 1:n_sample
        c_l = copy(c_base)      # (c_atom, N_atom)
        p_trunk = copy(p_base)  # (c_atompair, n_q, n_k, n_trunks)
        encoder.layernorm_s === nothing && error("layernorm_s missing for has_coords=true")
        encoder.linear_no_bias_s === nothing && error("linear_no_bias_s missing for has_coords=true")
        c_l .+= _broadcast_token_to_atom(
            encoder.linear_no_bias_s(encoder.layernorm_s(s_f[:, :, i])),
            atom_to_token_idx,
        )

        encoder.layernorm_z === nothing && error("layernorm_z missing for has_coords=true")
        encoder.linear_no_bias_z === nothing && error("linear_no_bias_z missing for has_coords=true")
        p_trunk .+= _broadcast_token_pair_to_local_trunks(
            encoder.linear_no_bias_z(encoder.layernorm_z(z_f[:, :, :, i])),
            atom_to_token_idx,
            encoder.n_queries,
            encoder.n_keys,
        )

        _mix_pair_with_single!(
            p_trunk,
            c_l,
            encoder.linear_no_bias_cl,
            encoder.linear_no_bias_cm,
            encoder.small_mlp_1,
            encoder.small_mlp_2,
            encoder.small_mlp_3,
            encoder.n_queries,
            encoder.n_keys,
        )

        encoder.linear_no_bias_r === nothing && error("linear_no_bias_r missing for has_coords=true")
        q_l = c_l + encoder.linear_no_bias_r(r_f[:, :, i])
        q_l = encoder.atom_transformer(q_l, c_l, p_trunk; n_queries = encoder.n_queries, n_keys = encoder.n_keys)
        a_i = _aggregate_atom_to_token_mean(max.(encoder.linear_no_bias_q(q_l), 0f0), atom_to_token_idx, n_token)
        @inbounds a_out[:, :, i] = a_i
        @inbounds q_skip[:, :, i] = q_l
        @inbounds c_skip[:, :, i] = c_l
        @inbounds p_skip[:, :, :, :, i] = p_trunk
    end

    return a_out, q_skip, c_skip, p_skip
end

# ---------------------------------------------------------------------------
# AtomAttentionDecoder
# ---------------------------------------------------------------------------
@concrete struct AtomAttentionDecoder
    c_token
    c_atom
    c_atompair
    n_queries
    n_keys
    linear_no_bias_a
    layernorm_q
    linear_no_bias_out
    atom_transformer
end
@layer AtomAttentionDecoder

function AtomAttentionDecoder(
    c_token::Int;
    c_atom::Int = 128,
    c_atompair::Int = 16,
    n_blocks::Int = 3,
    n_heads::Int = 4,
    n_queries::Int = 32,
    n_keys::Int = 128,
    rng::AbstractRNG = Random.default_rng(),
)
    return AtomAttentionDecoder(
        c_token,
        c_atom,
        c_atompair,
        n_queries,
        n_keys,
        LinearNoBias(c_token, c_atom),
        LayerNormFirst(c_atom),
        LinearNoBias(c_atom, 3),
        DiffusionTransformer(
            c_atom,
            c_atom,
            c_atompair;
            n_blocks = n_blocks,
            n_heads = n_heads,
            cross_attention_mode = true,
            rng = rng,
        ),
    )
end

function (decoder::AtomAttentionDecoder)(
    input_feature_dict,
    a::AbstractArray{<:Real,3},
    q_skip::AbstractArray{<:Real,3},
    c_skip::AbstractArray{<:Real,3},
    p_skip::AbstractArray{<:Real,5},
)
    _has_feat(input_feature_dict, "atom_to_token_idx") || error("Missing input feature 'atom_to_token_idx'.")
    atom_to_token_idx = Int.(_feat(input_feature_dict, "atom_to_token_idx"))
    # a: (c_token, N_token, N_sample)
    n_sample = size(a, 3)
    n_token = size(a, 2)
    c_tok = size(a, 1)
    c_tok == decoder.c_token || error("Decoder token dim mismatch: expected $(decoder.c_token), got $c_tok")
    size(q_skip, 3) == n_sample || error("q_skip sample axis mismatch.")
    size(c_skip, 3) == n_sample || error("c_skip sample axis mismatch.")
    size(p_skip, 5) == n_sample || error("p_skip sample axis mismatch.")
    # p_skip: (c_atompair, n_q, n_k, n_trunks, N_sample)
    size(p_skip, 4) == cld(size(q_skip, 2), decoder.n_queries) || error("p_skip trunk axis mismatch.")
    size(p_skip, 2) == decoder.n_queries || error("p_skip query window mismatch.")
    size(p_skip, 3) == decoder.n_keys || error("p_skip key window mismatch.")
    size(p_skip, 1) == decoder.c_atompair || error("p_skip pair channel mismatch.")
    n_atom = size(q_skip, 2)

    out = similar(a, Float32, 3, n_atom, n_sample)
    for i in 1:n_sample
        a_i = Float32.(a[:, :, i])  # (c_token, N_token)
        q = _broadcast_token_to_atom(decoder.linear_no_bias_a(a_i), atom_to_token_idx) + Float32.(q_skip[:, :, i])
        q = decoder.atom_transformer(
            q,
            Float32.(c_skip[:, :, i]),
            Float32.(p_skip[:, :, :, :, i]);
            n_queries = decoder.n_queries,
            n_keys = decoder.n_keys,
        )
        @inbounds out[:, :, i] = decoder.linear_no_bias_out(decoder.layernorm_q(q))
    end
    return out
end

end

module AtomAttentionModule

using Random

import ..Primitives: LinearNoBias, LayerNormNoOffset
import ..TransformerBlocks: DiffusionTransformer

export AtomAttentionEncoder, AtomAttentionDecoder

_has_feat(input::AbstractDict{<:AbstractString, <:Any}, key::String) = haskey(input, key)
_has_feat(input::NamedTuple, key::String) = hasproperty(input, Symbol(key))
_feat(input::AbstractDict{<:AbstractString, <:Any}, key::String) = input[key]
_feat(input::NamedTuple, key::String) = getproperty(input, Symbol(key))

function _matrix_f32(x, name::String)
    x isa AbstractMatrix || error("$name must be a matrix, got $(typeof(x))")
    return Float32.(x)
end

function _vector_i(x, name::String, n::Int)
    x isa AbstractVector || error("$name must be a vector, got $(typeof(x))")
    length(x) == n || error("$name length mismatch: expected $n, got $(length(x))")
    return Int.(x)
end

function _column_f32(x, name::String, n::Int)
    if x isa AbstractVector
        length(x) == n || error("$name length mismatch: expected $n, got $(length(x))")
        return reshape(Float32.(x), n, 1)
    elseif x isa AbstractMatrix && size(x, 1) == n && size(x, 2) == 1
        return Float32.(x)
    end
    error("$name must be a vector or [N,1], got $(typeof(x))")
end

function _broadcast_token_to_atom(
    x_token::AbstractMatrix{<:Real},
    atom_to_token_idx::AbstractVector{<:Integer},
)
    n_atom = length(atom_to_token_idx)
    c = size(x_token, 2)
    out = Array{Float32}(undef, n_atom, c)
    for a in 1:n_atom
        t = Int(atom_to_token_idx[a]) + 1
        (1 <= t <= size(x_token, 1)) || error("atom_to_token_idx[$a] out of range.")
        @inbounds out[a, :] = Float32.(x_token[t, :])
    end
    return out
end

function _validate_local_window(n_queries::Int, n_keys::Int)
    n_queries > 0 || error("n_queries must be positive.")
    n_keys >= n_queries || error("n_keys must be >= n_queries.")
    n_queries % 2 == 0 || error("n_queries must be even.")
    n_keys % 2 == 0 || error("n_keys must be even.")
    return nothing
end

function _broadcast_token_pair_to_local_trunks(
    z_token::AbstractArray{<:Real,3},
    atom_to_token_idx::AbstractVector{<:Integer},
    n_queries::Int,
    n_keys::Int,
)
    _validate_local_window(n_queries, n_keys)
    n_atom = length(atom_to_token_idx)
    c = size(z_token, 3)
    n_trunks = cld(n_atom, n_queries)
    out = zeros(Float32, n_trunks, n_queries, n_keys, c)
    z = Float32.(z_token)

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
            ti = q_atom_idx == 0 ? 1 : Int(atom_to_token_idx[q_atom_idx]) + 1
            (1 <= ti <= size(z, 1)) || error("atom_to_token_idx query out of range.")
            for kl in 1:n_keys
                k_pad_idx = k_start + kl - 1
                (1 <= k_pad_idx <= k_padded_len) || error("key padded index out of range.")
                k_atom_idx = k_pad_idx - left
                k_atom_idx = (1 <= k_atom_idx <= n_atom) ? k_atom_idx : 0
                tj = k_atom_idx == 0 ? 1 : Int(atom_to_token_idx[k_atom_idx]) + 1
                (1 <= tj <= size(z, 2)) || error("atom_to_token_idx key out of range.")
                @inbounds out[b, ql, kl, :] .= z[ti, tj, :]
            end
        end
    end
    return out
end

function _aggregate_atom_to_token_mean(
    x_atom::AbstractMatrix{<:Real},
    atom_to_token_idx::AbstractVector{<:Integer},
    n_token::Int,
)
    n_atom, c = size(x_atom)
    length(atom_to_token_idx) == n_atom || error("atom_to_token_idx length mismatch.")
    out = zeros(Float32, n_token, c)
    counts = zeros(Int, n_token)
    for i in 1:n_atom
        t = Int(atom_to_token_idx[i]) + 1
        (1 <= t <= n_token) || error("atom_to_token_idx[$i] out of range.")
        @inbounds out[t, :] .+= Float32.(x_atom[i, :])
        counts[t] += 1
    end
    for t in 1:n_token
        if counts[t] > 0
            @inbounds out[t, :] ./= counts[t]
        end
    end
    return out
end

function _small_mlp!(
    p::AbstractArray{Float32},
    linear1::LinearNoBias,
    linear2::LinearNoBias,
    linear3::LinearNoBias,
)
    x = max.(p, 0f0)
    x = linear1(x)
    x = max.(x, 0f0)
    x = linear2(x)
    x = max.(x, 0f0)
    p .+= linear3(x)
    return p
end

function _base_atom_features(
    ref_pos::Matrix{Float32},
    ref_charge::Matrix{Float32},
    ref_mask::Matrix{Float32},
    ref_element::Matrix{Float32},
    ref_atom_name_chars::Matrix{Float32},
    ref_space_uid::Vector{Int},
    linear_no_bias_ref_pos::LinearNoBias,
    linear_no_bias_ref_charge::LinearNoBias,
    linear_no_bias_f::LinearNoBias,
    linear_no_bias_d::LinearNoBias,
    linear_no_bias_invd::LinearNoBias,
    linear_no_bias_v::LinearNoBias,
    n_queries::Int,
    n_keys::Int,
)
    _validate_local_window(n_queries, n_keys)
    n_atom = size(ref_pos, 1)
    c_l = linear_no_bias_ref_pos(ref_pos) + linear_no_bias_ref_charge(asinh.(ref_charge))
    c_l .= (c_l + linear_no_bias_f(hcat(ref_mask, ref_element, ref_atom_name_chars))) .* ref_mask

    n_trunks = cld(n_atom, n_queries)
    d_local = zeros(Float32, n_trunks, n_queries, n_keys, 3)
    v_local = zeros(Float32, n_trunks, n_queries, n_keys, 1)
    inv_local = zeros(Float32, n_trunks, n_queries, n_keys, 1)
    mask_local = zeros(Float32, n_trunks, n_queries, n_keys, 1)
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
            xi1 = q_idx == 0 ? 0f0 : ref_pos[q_idx, 1]
            xi2 = q_idx == 0 ? 0f0 : ref_pos[q_idx, 2]
            xi3 = q_idx == 0 ? 0f0 : ref_pos[q_idx, 3]
            ui = q_idx == 0 ? 0 : ref_space_uid[q_idx]
            for kl in 1:n_keys
                k_pad_idx = k_start + kl - 1
                (1 <= k_pad_idx <= k_padded_len) || error("key padded index out of range.")
                k_idx = k_pad_idx - left
                k_idx = (1 <= k_idx <= n_atom) ? k_idx : 0
                xk1 = k_idx == 0 ? 0f0 : ref_pos[k_idx, 1]
                xk2 = k_idx == 0 ? 0f0 : ref_pos[k_idx, 2]
                xk3 = k_idx == 0 ? 0f0 : ref_pos[k_idx, 3]
                uk = k_idx == 0 ? 0 : ref_space_uid[k_idx]
                dx = xi1 - xk1
                dy = xi2 - xk2
                dz = xi3 - xk3
                same = ui == uk ? 1f0 : 0f0
                valid = (q_idx != 0 && k_idx != 0) ? 1f0 : 0f0
                @inbounds d_local[b, ql, kl, 1] = dx
                @inbounds d_local[b, ql, kl, 2] = dy
                @inbounds d_local[b, ql, kl, 3] = dz
                @inbounds v_local[b, ql, kl, 1] = same
                @inbounds inv_local[b, ql, kl, 1] = inv(1f0 + dx * dx + dy * dy + dz * dz)
                @inbounds mask_local[b, ql, kl, 1] = valid
            end
        end
    end

    p_local = (linear_no_bias_d(d_local) .* v_local) .* mask_local
    p_local .+= linear_no_bias_invd(inv_local) .* v_local
    p_local .+= linear_no_bias_v(v_local)
    return c_l, p_local
end

function _mix_pair_with_single!(
    p_local::Array{Float32,4},
    c_l::Matrix{Float32},
    linear_no_bias_cl::LinearNoBias,
    linear_no_bias_cm::LinearNoBias,
    small_mlp_1::LinearNoBias,
    small_mlp_2::LinearNoBias,
    small_mlp_3::LinearNoBias,
    n_queries::Int,
    n_keys::Int,
)
    _validate_local_window(n_queries, n_keys)
    n_atom = size(c_l, 1)
    c_atompair = size(p_local, 4)
    c_relu = max.(c_l, 0f0)
    c_q = linear_no_bias_cl(c_relu)
    c_k = linear_no_bias_cm(c_relu)
    n_trunks = size(p_local, 1)
    size(p_local, 2) == n_queries || error("p_local query window mismatch.")
    size(p_local, 3) == n_keys || error("p_local key window mismatch.")
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
            for kl in 1:n_keys
                k_pad_idx = k_start + kl - 1
                (1 <= k_pad_idx <= k_padded_len) || error("key padded index out of range.")
                k_idx = k_pad_idx - left
                k_idx = (1 <= k_idx <= n_atom) ? k_idx : 0

                if q_idx == 0 && k_idx == 0
                    continue
                elseif q_idx == 0
                    @inbounds p_local[b, ql, kl, :] .+= c_k[k_idx, :]
                elseif k_idx == 0
                    @inbounds p_local[b, ql, kl, :] .+= c_q[q_idx, :]
                else
                    @inbounds p_local[b, ql, kl, :] .+= c_q[q_idx, :] .+ c_k[k_idx, :]
                end
            end
        end
    end

    _small_mlp!(p_local, small_mlp_1, small_mlp_2, small_mlp_3)
    return p_local
end

struct AtomAttentionEncoder
    has_coords::Bool
    c_atom::Int
    c_atompair::Int
    c_token::Int
    c_s::Int
    c_z::Int
    n_queries::Int
    n_keys::Int
    linear_no_bias_ref_pos::LinearNoBias
    linear_no_bias_ref_charge::LinearNoBias
    linear_no_bias_f::LinearNoBias
    linear_no_bias_d::LinearNoBias
    linear_no_bias_invd::LinearNoBias
    linear_no_bias_v::LinearNoBias
    layernorm_s::Union{LayerNormNoOffset, Nothing}
    linear_no_bias_s::Union{LinearNoBias, Nothing}
    layernorm_z::Union{LayerNormNoOffset, Nothing}
    linear_no_bias_z::Union{LinearNoBias, Nothing}
    linear_no_bias_r::Union{LinearNoBias, Nothing}
    linear_no_bias_cl::LinearNoBias
    linear_no_bias_cm::LinearNoBias
    small_mlp_1::LinearNoBias
    small_mlp_2::LinearNoBias
    small_mlp_3::LinearNoBias
    atom_transformer::DiffusionTransformer
    linear_no_bias_q::LinearNoBias
end

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
    ln_s = has_coords ? LayerNormNoOffset(c_s) : nothing
    lin_s = has_coords ? LinearNoBias(c_atom, c_s; rng = rng) : nothing
    ln_z = has_coords ? LayerNormNoOffset(c_z) : nothing
    lin_z = has_coords ? LinearNoBias(c_atompair, c_z; rng = rng) : nothing
    lin_r = has_coords ? LinearNoBias(c_atom, 3; rng = rng) : nothing
    return AtomAttentionEncoder(
        has_coords,
        c_atom,
        c_atompair,
        c_token,
        c_s,
        c_z,
        n_queries,
        n_keys,
        LinearNoBias(c_atom, 3; rng = rng),
        LinearNoBias(c_atom, 1; rng = rng),
        LinearNoBias(c_atom, 1 + 128 + 256; rng = rng),
        LinearNoBias(c_atompair, 3; rng = rng),
        LinearNoBias(c_atompair, 1; rng = rng),
        LinearNoBias(c_atompair, 1; rng = rng),
        ln_s,
        lin_s,
        ln_z,
        lin_z,
        lin_r,
        LinearNoBias(c_atompair, c_atom; rng = rng),
        LinearNoBias(c_atompair, c_atom; rng = rng),
        LinearNoBias(c_atompair, c_atompair; rng = rng),
        LinearNoBias(c_atompair, c_atompair; rng = rng),
        LinearNoBias(c_atompair, c_atompair; rng = rng),
        DiffusionTransformer(
            c_atom,
            c_atom,
            c_atompair;
            n_blocks = n_blocks,
            n_heads = n_heads,
            cross_attention_mode = true,
            rng = rng,
        ),
        LinearNoBias(c_token, c_atom; rng = rng),
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
    ref_pos = _matrix_f32(_feat(input_feature_dict, "ref_pos"), "ref_pos")
    n_atom = size(ref_pos, 1)
    size(ref_pos, 2) == 3 || error("ref_pos must be [N_atom, 3]")
    ref_charge = _has_feat(input_feature_dict, "ref_charge") ?
                 _column_f32(_feat(input_feature_dict, "ref_charge"), "ref_charge", n_atom) :
                 zeros(Float32, n_atom, 1)
    ref_mask = _column_f32(_feat(input_feature_dict, "ref_mask"), "ref_mask", n_atom)
    ref_element = _matrix_f32(_feat(input_feature_dict, "ref_element"), "ref_element")
    size(ref_element) == (n_atom, 128) || error("ref_element must be [N_atom, 128]")
    ref_atom_name_chars = _matrix_f32(_feat(input_feature_dict, "ref_atom_name_chars"), "ref_atom_name_chars")
    size(ref_atom_name_chars) == (n_atom, 256) || error("ref_atom_name_chars must be [N_atom, 256]")
    ref_space_uid = _vector_i(_feat(input_feature_dict, "ref_space_uid"), "ref_space_uid", n_atom)
    atom_to_token_idx = _vector_i(_feat(input_feature_dict, "atom_to_token_idx"), "atom_to_token_idx", n_atom)
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
        c_l = copy(c_base)
        p_trunk = copy(p_base)
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
        q_l = encoder.atom_transformer(c_l, c_l, p_trunk, encoder.n_queries, encoder.n_keys)
        a = _aggregate_atom_to_token_mean(max.(encoder.linear_no_bias_q(q_l), 0f0), atom_to_token_idx, n_token)
        return a, q_l, c_l, p_trunk
    end

    r_l isa AbstractArray{<:Real,3} || error("AtomAttentionEncoder(has_coords=true) requires r_l with shape [N_sample, N_atom, 3].")
    s isa AbstractArray{<:Real,3} || error("AtomAttentionEncoder(has_coords=true) requires s with shape [N_sample, N_token, c_s].")
    z isa AbstractArray{<:Real,4} || error("AtomAttentionEncoder(has_coords=true) requires z with shape [N_sample, N_token, N_token, c_z].")

    r_f = Float32.(r_l)
    s_f = Float32.(s)
    z_f = Float32.(z)
    n_sample = size(r_f, 1)
    size(r_f, 2) == n_atom || error("r_l atom axis mismatch.")
    size(r_f, 3) == 3 || error("r_l last dim must be 3.")
    size(s_f, 1) == n_sample || error("s sample axis mismatch.")
    size(s_f, 2) == n_token || error("s token axis mismatch.")
    size(z_f, 1) == n_sample || error("z sample axis mismatch.")
    size(z_f, 2) == n_token || error("z token axis mismatch.")
    size(z_f, 3) == n_token || error("z token axis mismatch.")

    a_out = Array{Float32}(undef, n_sample, n_token, encoder.c_token)
    q_skip = Array{Float32}(undef, n_sample, n_atom, encoder.c_atom)
    c_skip = Array{Float32}(undef, n_sample, n_atom, encoder.c_atom)
    n_trunks = cld(n_atom, encoder.n_queries)
    p_skip = Array{Float32}(undef, n_sample, n_trunks, encoder.n_queries, encoder.n_keys, encoder.c_atompair)

    for i in 1:n_sample
        c_l = copy(c_base)
        p_trunk = copy(p_base)
        encoder.layernorm_s === nothing && error("layernorm_s missing for has_coords=true")
        encoder.linear_no_bias_s === nothing && error("linear_no_bias_s missing for has_coords=true")
        c_l .+= _broadcast_token_to_atom(
            encoder.linear_no_bias_s(encoder.layernorm_s(Array{Float32,2}(s_f[i, :, :]))),
            atom_to_token_idx,
        )

        encoder.layernorm_z === nothing && error("layernorm_z missing for has_coords=true")
        encoder.linear_no_bias_z === nothing && error("linear_no_bias_z missing for has_coords=true")
        p_trunk .+= _broadcast_token_pair_to_local_trunks(
            encoder.linear_no_bias_z(encoder.layernorm_z(Array{Float32,3}(z_f[i, :, :, :]))),
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
        q_l = c_l + encoder.linear_no_bias_r(Array{Float32,2}(r_f[i, :, :]))
        q_l = encoder.atom_transformer(q_l, c_l, p_trunk, encoder.n_queries, encoder.n_keys)
        a_i = _aggregate_atom_to_token_mean(max.(encoder.linear_no_bias_q(q_l), 0f0), atom_to_token_idx, n_token)
        @inbounds a_out[i, :, :] = a_i
        @inbounds q_skip[i, :, :] = q_l
        @inbounds c_skip[i, :, :] = c_l
        @inbounds p_skip[i, :, :, :, :] = p_trunk
    end

    return a_out, q_skip, c_skip, p_skip
end

struct AtomAttentionDecoder
    c_token::Int
    c_atom::Int
    c_atompair::Int
    n_queries::Int
    n_keys::Int
    linear_no_bias_a::LinearNoBias
    layernorm_q::LayerNormNoOffset
    linear_no_bias_out::LinearNoBias
    atom_transformer::DiffusionTransformer
end

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
        LinearNoBias(c_atom, c_token; rng = rng),
        LayerNormNoOffset(c_atom),
        LinearNoBias(3, c_atom; rng = rng),
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
    n_sample = size(a, 1)
    n_token = size(a, 2)
    c_tok = size(a, 3)
    c_tok == decoder.c_token || error("Decoder token dim mismatch: expected $(decoder.c_token), got $c_tok")
    size(q_skip, 1) == n_sample || error("q_skip sample axis mismatch.")
    size(c_skip, 1) == n_sample || error("c_skip sample axis mismatch.")
    size(p_skip, 1) == n_sample || error("p_skip sample axis mismatch.")
    size(p_skip, 2) == cld(size(q_skip, 2), decoder.n_queries) || error("p_skip trunk axis mismatch.")
    size(p_skip, 3) == decoder.n_queries || error("p_skip query window mismatch.")
    size(p_skip, 4) == decoder.n_keys || error("p_skip key window mismatch.")
    size(p_skip, 5) == decoder.c_atompair || error("p_skip pair channel mismatch.")
    n_atom = size(q_skip, 2)

    out = Array{Float32}(undef, n_sample, n_atom, 3)
    for i in 1:n_sample
        a_i = Array{Float32,2}(a[i, :, :])
        size(a_i, 1) == n_token || error("a token axis mismatch.")
        q = _broadcast_token_to_atom(decoder.linear_no_bias_a(a_i), atom_to_token_idx) + Array{Float32,2}(q_skip[i, :, :])
        q = decoder.atom_transformer(
            q,
            Array{Float32,2}(c_skip[i, :, :]),
            Array{Float32,4}(p_skip[i, :, :, :, :]),
            decoder.n_queries,
            decoder.n_keys,
        )
        @inbounds out[i, :, :] = decoder.linear_no_bias_out(decoder.layernorm_q(q))
    end
    return out
end

end

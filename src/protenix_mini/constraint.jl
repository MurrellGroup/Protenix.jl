module Constraint

using Random
using ConcreteStructs
using Flux: @layer
import Onion: flash_attention_forward

import ..Primitives: Linear, LinearNoBias, LayerNorm

export AbstractSubstructureEmbedder,
    SubstructureLinearEmbedder,
    SubstructureMLPEmbedder,
    SubstructureSelfAttention,
    SubstructureTransformerLayer,
    SubstructureTransformerEmbedder,
    ConstraintEmbedder

_as_f32_array(x::AbstractArray{<:Real}) = x isa AbstractArray{Float32} ? x : Float32.(x)

abstract type AbstractSubstructureEmbedder end

# ─── SubstructureLinearEmbedder ───────────────────────────────────────────────

@concrete struct SubstructureLinearEmbedder <: AbstractSubstructureEmbedder
    proj
end
@layer SubstructureLinearEmbedder

SubstructureLinearEmbedder(n_classes::Int, c_pair_dim::Int; rng::AbstractRNG = Random.default_rng()) =
    SubstructureLinearEmbedder(LinearNoBias(n_classes, c_pair_dim; rng = rng))

function (m::SubstructureLinearEmbedder)(x::AbstractArray{<:Real})
    return m.proj(x)
end

# ─── SubstructureMLPEmbedder ─────────────────────────────────────────────────

@concrete struct SubstructureMLPEmbedder <: AbstractSubstructureEmbedder
    layers
end
@layer SubstructureMLPEmbedder

function SubstructureMLPEmbedder(
    n_classes::Int,
    c_pair_dim::Int;
    hidden_dim::Int = 256,
    n_layers::Int = 3,
    rng::AbstractRNG = Random.default_rng(),
)
    hidden_dim > 0 || error("hidden_dim must be positive")
    linears = Linear[]
    push!(linears, LinearNoBias(n_classes, hidden_dim; rng = rng))      # n_classes → hidden
    for _ in 1:max(n_layers - 2, 0)
        push!(linears, LinearNoBias(hidden_dim, hidden_dim; rng = rng)) # hidden → hidden
    end
    push!(linears, LinearNoBias(hidden_dim, c_pair_dim; rng = rng))     # hidden → c_pair
    return SubstructureMLPEmbedder(linears)
end

function (m::SubstructureMLPEmbedder)(x::AbstractArray{<:Real})
    h = _as_f32_array(x)
    for i in eachindex(m.layers)
        h = m.layers[i](h)
        i < length(m.layers) && (h = max.(h, 0f0))
    end
    return h
end

# ─── SubstructureSelfAttention ───────────────────────────────────────────────
# Features-first: input (hidden, seq_len, bsz)

@concrete struct SubstructureSelfAttention
    in_proj_weight # [3H, H]
    in_proj_bias # [3H]
    out_proj
    n_heads
end
@layer SubstructureSelfAttention

function SubstructureSelfAttention(
    hidden_dim::Int;
    n_heads::Int = 4,
    rng::AbstractRNG = Random.default_rng(),
)
    hidden_dim > 0 || error("hidden_dim must be positive")
    n_heads > 0 || error("n_heads must be positive")
    hidden_dim % n_heads == 0 || error("hidden_dim must be divisible by n_heads")
    in_proj_weight = 0.02f0 .* randn(rng, Float32, hidden_dim * 3, hidden_dim)
    in_proj_bias = zeros(Float32, hidden_dim * 3)
    out_proj = Linear(hidden_dim, hidden_dim; bias = true)
    return SubstructureSelfAttention(in_proj_weight, in_proj_bias, out_proj, n_heads)
end

function (m::SubstructureSelfAttention)(x::AbstractArray{<:Real,3})
    # x: (hidden, seq_len, bsz) features-first
    h = _as_f32_array(x)
    hidden, seq_len, bsz = size(h)
    size(m.in_proj_weight) == (hidden * 3, hidden) || error("SubstructureSelfAttention in_proj_weight shape mismatch")
    length(m.in_proj_bias) == hidden * 3 || error("SubstructureSelfAttention in_proj_bias shape mismatch")

    # Project: in_proj_weight * flat → (3*hidden, batch_flat)
    flat = reshape(h, hidden, :)  # (hidden, seq_len*bsz)
    qkv = m.in_proj_weight * flat  # (3*hidden, seq_len*bsz)
    qkv .+= reshape(m.in_proj_bias, :, 1)
    qkv = reshape(qkv, hidden * 3, seq_len, bsz)

    # Copy slices (needed for contiguous reshape)
    q = qkv[1:hidden, :, :]
    k = qkv[hidden + 1:2*hidden, :, :]
    v = qkv[2*hidden + 1:3*hidden, :, :]

    H = m.n_heads
    d = fld(hidden, H)

    # Reshape (d*H, seq, bsz) → (d, seq, H, bsz) for flash attention
    q4 = permutedims(reshape(q, d, H, seq_len, bsz), (1, 3, 2, 4))  # (d, seq, H, bsz)
    k4 = permutedims(reshape(k, d, H, seq_len, bsz), (1, 3, 2, 4))
    v4 = permutedims(reshape(v, d, H, seq_len, bsz), (1, 3, 2, 4))

    # Flash attention: no bias, handles scale internally
    ctx4 = flash_attention_forward(q4, k4, v4)  # (d, seq, H, bsz)

    # Reshape back: (d, seq, H, bsz) → (d, H, seq, bsz) → (hidden, seq, bsz)
    merged = reshape(permutedims(ctx4, (1, 3, 2, 4)), hidden, seq_len, bsz)

    return m.out_proj(merged)  # (hidden, seq_len, bsz)
end

# ─── SubstructureTransformerLayer ────────────────────────────────────────────
# Features-first: input (hidden, seq_len, bsz)

@concrete struct SubstructureTransformerLayer
    self_attn
    linear1
    linear2
    norm1
    norm2
end
@layer SubstructureTransformerLayer

function SubstructureTransformerLayer(
    hidden_dim::Int;
    n_heads::Int = 4,
    rng::AbstractRNG = Random.default_rng(),
)
    return SubstructureTransformerLayer(
        SubstructureSelfAttention(hidden_dim; n_heads = n_heads, rng = rng),
        Linear(hidden_dim, hidden_dim * 4; bias = true),   # hidden → 4*hidden
        Linear(hidden_dim * 4, hidden_dim; bias = true),   # 4*hidden → hidden
        LayerNorm(hidden_dim),
        LayerNorm(hidden_dim),
    )
end

function (m::SubstructureTransformerLayer)(x::AbstractArray{<:Real,3})
    x0 = _as_f32_array(x)
    a = m.norm1(x0 .+ m.self_attn(x0))
    b = m.norm2(a .+ m.linear2(max.(m.linear1(a), 0f0)))
    return b
end

# ─── SubstructureTransformerEmbedder ─────────────────────────────────────────
# Features-first: input (C, N_tok, N_tok)

@concrete struct SubstructureTransformerEmbedder <: AbstractSubstructureEmbedder
    input_proj
    layers
    output_proj
end
@layer SubstructureTransformerEmbedder

function SubstructureTransformerEmbedder(
    n_classes::Int,
    c_pair_dim::Int;
    hidden_dim::Int = 128,
    n_layers::Int = 1,
    n_heads::Int = 4,
    rng::AbstractRNG = Random.default_rng(),
)
    n_layers > 0 || error("n_layers must be positive")
    input_proj = LinearNoBias(n_classes, hidden_dim; rng = rng)   # n_classes → hidden
    layers = [SubstructureTransformerLayer(hidden_dim; n_heads = n_heads, rng = rng) for _ in 1:n_layers]
    output_proj = LinearNoBias(hidden_dim, c_pair_dim; rng = rng)  # hidden → c_pair
    return SubstructureTransformerEmbedder(input_proj, layers, output_proj)
end

function (m::SubstructureTransformerEmbedder)(x::AbstractArray{<:Real})
    n = ndims(x)
    n >= 3 || error("SubstructureTransformerEmbedder expects rank >= 3")
    # Features-first: dim=1 is features/channels
    in_dim = size(x, 1)
    n_tok1 = size(x, 2)
    n_tok2 = size(x, 3)
    n_tok1 == n_tok2 || error("SubstructureTransformerEmbedder expects square token axes")
    size(m.input_proj.weight, 2) == in_dim || error(
        "substructure channel mismatch: expected $(size(m.input_proj.weight, 2)), got $in_dim",
    )

    trail_dims = n > 3 ? Tuple(size(x)[4:n]) : ()
    batch = n > 3 ? prod(trail_dims) : 1

    # (C, N1, N2, batch) → project features → (hidden, N1, N2, batch)
    x4 = reshape(_as_f32_array(x), in_dim, n_tok1, n_tok2, batch)
    x4 = m.input_proj(x4)  # (hidden, N1, N2, batch)
    h = reshape(x4, size(x4, 1), n_tok1 * n_tok2, batch)  # (hidden, N1*N2, batch)

    for layer in m.layers
        h = layer(h)
    end
    y = m.output_proj(h)  # (c_pair, N1*N2, batch)
    y4 = reshape(y, size(y, 1), n_tok1, n_tok2, batch)
    if isempty(trail_dims)
        return reshape(y4, size(y, 1), n_tok1, n_tok2)
    end
    return reshape(y4, (size(y, 1), n_tok1, n_tok2, trail_dims...))
end

# ─── Helpers ─────────────────────────────────────────────────────────────────

_substructure_channels(m::SubstructureLinearEmbedder) = size(m.proj.weight, 2)
_substructure_channels(m::SubstructureMLPEmbedder) = size(first(m.layers).weight, 2)
_substructure_channels(m::SubstructureTransformerEmbedder) = size(m.input_proj.weight, 2)

function _constraint_arr3(x, key::String)
    x isa AbstractArray || error("constraint_feature.$key must be array-like")
    ndims(x) == 3 || error("constraint_feature.$key must be rank-3 (C, N, N) features-first")
    return _as_f32_array(x)
end

function _constraint_get(constraint_feature, key::String)
    if constraint_feature isa AbstractDict
        haskey(constraint_feature, key) || return nothing
        return constraint_feature[key]
    elseif constraint_feature isa NamedTuple || hasproperty(constraint_feature, Symbol(key))
        hasproperty(constraint_feature, Symbol(key)) || return nothing
        return getproperty(constraint_feature, Symbol(key))
    end
    error("constraint_feature must expose fields by key (Dict/NamedTuple/struct).")
end

"""
Convert substructure index matrix to features-first one-hot: (n_classes, n1, n2).
"""
function _substructure_from_index(sub_raw::AbstractMatrix{<:Integer}, n_classes::Int)
    n1, n2 = size(sub_raw)
    cls = clamp.(Int.(sub_raw) .+ 1, 1, n_classes)
    out = zeros(Float32, n_classes, n1, n2)  # features-first
    @inbounds for i in 1:n1, j in 1:n2
        out[cls[i, j], i, j] = 1f0
    end
    return out
end

function _prepare_substructure_feature(sub_raw, n_classes::Int)
    if sub_raw isa AbstractArray && ndims(sub_raw) == 2
        return _substructure_from_index(Int.(sub_raw), n_classes)
    end
    sub = _constraint_arr3(sub_raw, "substructure")
    size(sub, 1) == n_classes || error("constraint_feature.substructure channel mismatch: expected $n_classes, got $(size(sub, 1))")
    return sub
end

# ─── ConstraintEmbedder ─────────────────────────────────────────────────────

@concrete struct ConstraintEmbedder
    pocket_z_embedder
    contact_z_embedder
    contact_atom_z_embedder
    substructure_z_embedder
end
@layer ConstraintEmbedder

function ConstraintEmbedder(
    c_constraint_z::Int;
    pocket_enable::Bool = false,
    pocket_c_z_input::Int = 1,
    contact_enable::Bool = false,
    contact_c_z_input::Int = 2,
    contact_atom_enable::Bool = false,
    contact_atom_c_z_input::Int = 2,
    substructure_enable::Bool = false,
    substructure_n_classes::Int = 4,
    substructure_architecture::Symbol = :linear,
    substructure_hidden_dim::Int = 128,
    substructure_n_layers::Int = 1,
    substructure_n_heads::Int = 4,
    initialize_method::Symbol = :zero,
    rng::AbstractRNG = Random.default_rng(),
)
    p = pocket_enable ? LinearNoBias(pocket_c_z_input, c_constraint_z; rng = rng) : nothing
    c = contact_enable ? LinearNoBias(contact_c_z_input, c_constraint_z; rng = rng) : nothing
    ca = contact_atom_enable ? LinearNoBias(contact_atom_c_z_input, c_constraint_z; rng = rng) : nothing

    s = nothing
    if substructure_enable
        arch = lowercase(String(substructure_architecture))
        if arch == "linear"
            s = SubstructureLinearEmbedder(substructure_n_classes, c_constraint_z; rng = rng)
        elseif arch == "mlp"
            s = SubstructureMLPEmbedder(
                substructure_n_classes,
                c_constraint_z;
                hidden_dim = substructure_hidden_dim,
                n_layers = substructure_n_layers,
                rng = rng,
            )
        elseif arch == "transformer"
            s = SubstructureTransformerEmbedder(
                substructure_n_classes,
                c_constraint_z;
                hidden_dim = substructure_hidden_dim,
                n_layers = substructure_n_layers,
                n_heads = substructure_n_heads,
                rng = rng,
            )
        else
            error("Unsupported substructure architecture: $substructure_architecture")
        end
    end

    if initialize_method == :zero
        p !== nothing && fill!(p.weight, 0f0)
        c !== nothing && fill!(c.weight, 0f0)
        ca !== nothing && fill!(ca.weight, 0f0)
        if s isa SubstructureLinearEmbedder
            fill!(s.proj.weight, 0f0)
        elseif s isa SubstructureMLPEmbedder
            for lin in s.layers
                fill!(lin.weight, 0f0)
            end
        elseif s isa SubstructureTransformerEmbedder
            fill!(s.input_proj.weight, 0f0)
            fill!(s.output_proj.weight, 0f0)
            for layer in s.layers
                fill!(layer.self_attn.in_proj_weight, 0f0)
                fill!(layer.self_attn.in_proj_bias, 0f0)
                fill!(layer.self_attn.out_proj.weight, 0f0)
                layer.self_attn.out_proj.bias !== nothing && fill!(layer.self_attn.out_proj.bias, 0f0)
                fill!(layer.linear1.weight, 0f0)
                layer.linear1.bias !== nothing && fill!(layer.linear1.bias, 0f0)
                fill!(layer.linear2.weight, 0f0)
                layer.linear2.bias !== nothing && fill!(layer.linear2.bias, 0f0)
            end
        end
    end

    return ConstraintEmbedder(p, c, ca, s)
end

function (cemb::ConstraintEmbedder)(constraint_feature)
    z_constraint = nothing

    if cemb.pocket_z_embedder !== nothing
        pocket_raw = _constraint_get(constraint_feature, "pocket")
        pocket_raw === nothing || begin
            pocket = _constraint_arr3(pocket_raw, "pocket")
            # Features-first: dim=1 is input channels, weight dim=2 is input size
            size(pocket, 1) == size(cemb.pocket_z_embedder.weight, 2) ||
                error("constraint_feature.pocket channel mismatch")
            zp = cemb.pocket_z_embedder(pocket)
            z_constraint = z_constraint === nothing ? zp : z_constraint .+ zp
        end
    end

    if cemb.contact_z_embedder !== nothing
        contact_raw = _constraint_get(constraint_feature, "contact")
        contact_raw === nothing || begin
            contact = _constraint_arr3(contact_raw, "contact")
            size(contact, 1) == size(cemb.contact_z_embedder.weight, 2) ||
                error("constraint_feature.contact channel mismatch")
            zc = cemb.contact_z_embedder(contact)
            z_constraint = z_constraint === nothing ? zc : z_constraint .+ zc
        end
    end

    if cemb.contact_atom_z_embedder !== nothing
        contact_atom_raw = _constraint_get(constraint_feature, "contact_atom")
        contact_atom_raw === nothing || begin
            contact_atom = _constraint_arr3(contact_atom_raw, "contact_atom")
            size(contact_atom, 1) == size(cemb.contact_atom_z_embedder.weight, 2) ||
                error("constraint_feature.contact_atom channel mismatch")
            zca = cemb.contact_atom_z_embedder(contact_atom)
            z_constraint = z_constraint === nothing ? zca : z_constraint .+ zca
        end
    end

    if cemb.substructure_z_embedder !== nothing
        sub_raw = _constraint_get(constraint_feature, "substructure")
        sub_raw === nothing || begin
            n_classes = _substructure_channels(cemb.substructure_z_embedder)
            sub = _prepare_substructure_feature(sub_raw, n_classes)
            zs = cemb.substructure_z_embedder(sub)
            z_constraint = z_constraint === nothing ? zs : z_constraint .+ zs
        end
    end

    return z_constraint
end

end

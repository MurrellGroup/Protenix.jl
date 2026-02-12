module Constraint

using Random

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

struct SubstructureLinearEmbedder <: AbstractSubstructureEmbedder
    proj::LinearNoBias
end

SubstructureLinearEmbedder(n_classes::Int, c_pair_dim::Int; rng::AbstractRNG = Random.default_rng()) =
    SubstructureLinearEmbedder(LinearNoBias(c_pair_dim, n_classes; rng = rng))

function (m::SubstructureLinearEmbedder)(x::AbstractArray{<:Real})
    return m.proj(x)
end

struct SubstructureMLPEmbedder <: AbstractSubstructureEmbedder
    layers::Vector{LinearNoBias}
end

function SubstructureMLPEmbedder(
    n_classes::Int,
    c_pair_dim::Int;
    hidden_dim::Int = 256,
    n_layers::Int = 3,
    rng::AbstractRNG = Random.default_rng(),
)
    hidden_dim > 0 || error("hidden_dim must be positive")
    linears = LinearNoBias[]
    push!(linears, LinearNoBias(hidden_dim, n_classes; rng = rng))
    for _ in 1:max(n_layers - 2, 0)
        push!(linears, LinearNoBias(hidden_dim, hidden_dim; rng = rng))
    end
    push!(linears, LinearNoBias(c_pair_dim, hidden_dim; rng = rng))
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

function _row_softmax!(scores::Matrix{Float32})
    @inbounds for i in axes(scores, 1)
        row = @view scores[i, :]
        m = maximum(row)
        row .= exp.(row .- m)
        s = sum(row)
        row ./= s
    end
    return scores
end

struct SubstructureSelfAttention
    in_proj_weight::Matrix{Float32} # [3H, H]
    in_proj_bias::Vector{Float32} # [3H]
    out_proj::Linear
    n_heads::Int
end

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
    out_proj = Linear(hidden_dim, hidden_dim; bias = true, rng = rng)
    return SubstructureSelfAttention(in_proj_weight, in_proj_bias, out_proj, n_heads)
end

function (m::SubstructureSelfAttention)(x::AbstractArray{<:Real,3})
    h = _as_f32_array(x)
    bsz, seq_len, hidden = size(h)
    size(m.in_proj_weight) == (hidden * 3, hidden) || error("SubstructureSelfAttention in_proj_weight shape mismatch")
    length(m.in_proj_bias) == hidden * 3 || error("SubstructureSelfAttention in_proj_bias shape mismatch")

    flat = reshape(h, :, hidden)
    qkv = flat * transpose(m.in_proj_weight)
    qkv .+= reshape(m.in_proj_bias, 1, :)
    qkv = reshape(qkv, bsz, seq_len, hidden * 3)

    q = @view qkv[:, :, 1:hidden]
    k = @view qkv[:, :, hidden + 1:2*hidden]
    v = @view qkv[:, :, 2*hidden + 1:3*hidden]

    n_heads = m.n_heads
    head_dim = fld(hidden, n_heads)
    qh = permutedims(reshape(q, bsz, seq_len, n_heads, head_dim), (1, 3, 2, 4)) # [B,H,S,D]
    kh = permutedims(reshape(k, bsz, seq_len, n_heads, head_dim), (1, 3, 2, 4))
    vh = permutedims(reshape(v, bsz, seq_len, n_heads, head_dim), (1, 3, 2, 4))

    out = Array{Float32}(undef, bsz, n_heads, seq_len, head_dim)
    scale = inv(sqrt(Float32(head_dim)))
    @inbounds for b in 1:bsz, hidx in 1:n_heads
        qmat = reshape(@view(qh[b, hidx, :, :]), seq_len, head_dim)
        kmat = reshape(@view(kh[b, hidx, :, :]), seq_len, head_dim)
        vmat = reshape(@view(vh[b, hidx, :, :]), seq_len, head_dim)
        scores = (qmat * transpose(kmat)) .* scale
        _row_softmax!(scores)
        @view(out[b, hidx, :, :]) .= scores * vmat
    end

    merged = permutedims(out, (1, 3, 2, 4))
    merged = reshape(merged, bsz, seq_len, hidden)
    return m.out_proj(merged)
end

struct SubstructureTransformerLayer
    self_attn::SubstructureSelfAttention
    linear1::Linear
    linear2::Linear
    norm1::LayerNorm
    norm2::LayerNorm
end

function SubstructureTransformerLayer(
    hidden_dim::Int;
    n_heads::Int = 4,
    rng::AbstractRNG = Random.default_rng(),
)
    return SubstructureTransformerLayer(
        SubstructureSelfAttention(hidden_dim; n_heads = n_heads, rng = rng),
        Linear(hidden_dim * 4, hidden_dim; bias = true, rng = rng),
        Linear(hidden_dim, hidden_dim * 4; bias = true, rng = rng),
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

struct SubstructureTransformerEmbedder <: AbstractSubstructureEmbedder
    input_proj::LinearNoBias
    layers::Vector{SubstructureTransformerLayer}
    output_proj::LinearNoBias
end

function SubstructureTransformerEmbedder(
    n_classes::Int,
    c_pair_dim::Int;
    hidden_dim::Int = 128,
    n_layers::Int = 1,
    n_heads::Int = 4,
    rng::AbstractRNG = Random.default_rng(),
)
    n_layers > 0 || error("n_layers must be positive")
    input_proj = LinearNoBias(hidden_dim, n_classes; rng = rng)
    layers = [SubstructureTransformerLayer(hidden_dim; n_heads = n_heads, rng = rng) for _ in 1:n_layers]
    output_proj = LinearNoBias(c_pair_dim, hidden_dim; rng = rng)
    return SubstructureTransformerEmbedder(input_proj, layers, output_proj)
end

function (m::SubstructureTransformerEmbedder)(x::AbstractArray{<:Real})
    n = ndims(x)
    n >= 3 || error("SubstructureTransformerEmbedder expects rank >= 3")
    n_tok1 = size(x, n - 2)
    n_tok2 = size(x, n - 1)
    n_tok1 == n_tok2 || error("SubstructureTransformerEmbedder expects square token axes")
    in_dim = size(x, n)
    size(m.input_proj.weight, 2) == in_dim || error(
        "substructure channel mismatch: expected $(size(m.input_proj.weight, 2)), got $in_dim",
    )

    lead_dims = n > 3 ? Tuple(size(x)[1:(n - 3)]) : ()
    batch = n > 3 ? prod(lead_dims) : 1
    x4 = reshape(_as_f32_array(x), batch, n_tok1, n_tok2, in_dim)
    x4 = m.input_proj(x4)
    h = Array{Float32}(undef, batch, n_tok1 * n_tok2, size(x4, 4))
    @inbounds for b in 1:batch, i in 1:n_tok1, j in 1:n_tok2
        seq_idx = (i - 1) * n_tok2 + j
        @views h[b, seq_idx, :] .= x4[b, i, j, :]
    end

    for layer in m.layers
        h = layer(h)
    end
    y = m.output_proj(h)
    y4 = Array{Float32}(undef, batch, n_tok1, n_tok2, size(y, 3))
    @inbounds for b in 1:batch, i in 1:n_tok1, j in 1:n_tok2
        seq_idx = (i - 1) * n_tok2 + j
        @views y4[b, i, j, :] .= y[b, seq_idx, :]
    end
    if isempty(lead_dims)
        return reshape(y4, n_tok1, n_tok2, size(y4, 4))
    end
    return reshape(y4, (lead_dims..., n_tok1, n_tok2, size(y4, 4)))
end

_substructure_channels(m::SubstructureLinearEmbedder) = size(m.proj.weight, 2)
_substructure_channels(m::SubstructureMLPEmbedder) = size(first(m.layers).weight, 2)
_substructure_channels(m::SubstructureTransformerEmbedder) = size(m.input_proj.weight, 2)

function _constraint_arr3(x, key::String)
    x isa AbstractArray || error("constraint_feature.$key must be array-like")
    ndims(x) == 3 || error("constraint_feature.$key must be rank-3 [N_token,N_token,C]")
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

function _substructure_from_index(sub_raw::AbstractMatrix{<:Integer}, n_classes::Int)
    out = zeros(Float32, size(sub_raw, 1), size(sub_raw, 2), n_classes)
    @inbounds for i in axes(sub_raw, 1), j in axes(sub_raw, 2)
        cls = clamp(Int(sub_raw[i, j]) + 1, 1, n_classes)
        out[i, j, cls] = 1f0
    end
    return out
end

function _prepare_substructure_feature(sub_raw, n_classes::Int)
    if sub_raw isa AbstractArray && ndims(sub_raw) == 2
        return _substructure_from_index(Int.(sub_raw), n_classes)
    end
    sub = _constraint_arr3(sub_raw, "substructure")
    size(sub, 3) == n_classes || error("constraint_feature.substructure channel mismatch")
    return sub
end

struct ConstraintEmbedder{P, C, CA, S}
    pocket_z_embedder::P
    contact_z_embedder::C
    contact_atom_z_embedder::CA
    substructure_z_embedder::S
end

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
    p = pocket_enable ? LinearNoBias(c_constraint_z, pocket_c_z_input; rng = rng) : nothing
    c = contact_enable ? LinearNoBias(c_constraint_z, contact_c_z_input; rng = rng) : nothing
    ca = contact_atom_enable ? LinearNoBias(c_constraint_z, contact_atom_c_z_input; rng = rng) : nothing

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
            size(pocket, 3) == size(cemb.pocket_z_embedder.weight, 2) ||
                error("constraint_feature.pocket channel mismatch")
            zp = cemb.pocket_z_embedder(pocket)
            z_constraint = z_constraint === nothing ? zp : z_constraint .+ zp
        end
    end

    if cemb.contact_z_embedder !== nothing
        contact_raw = _constraint_get(constraint_feature, "contact")
        contact_raw === nothing || begin
            contact = _constraint_arr3(contact_raw, "contact")
            size(contact, 3) == size(cemb.contact_z_embedder.weight, 2) ||
                error("constraint_feature.contact channel mismatch")
            zc = cemb.contact_z_embedder(contact)
            z_constraint = z_constraint === nothing ? zc : z_constraint .+ zc
        end
    end

    if cemb.contact_atom_z_embedder !== nothing
        contact_atom_raw = _constraint_get(constraint_feature, "contact_atom")
        contact_atom_raw === nothing || begin
            contact_atom = _constraint_arr3(contact_atom_raw, "contact_atom")
            size(contact_atom, 3) == size(cemb.contact_atom_z_embedder.weight, 2) ||
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

module Heads

using Random
using ConcreteStructs
using Flux: @layer

import ..Primitives: Linear, LinearNoBias, LayerNorm
import ..Features: ProtenixFeatures, as_protenix_features
import ..Pairformer: PairformerStack
import ..Utils: one_hot_interval, broadcast_token_to_atom, pairwise_distances

export DistogramHead, ConfidenceHead

_as_f32_array(x::AbstractArray{<:Real}) = x isa AbstractArray{Float32} ? x : Float32.(x)
_as_f32_copy(x::AbstractArray{<:Real}) = x isa AbstractArray{Float32} ? copy(x) : Float32.(x)

# ─── DistogramHead ───────────────────────────────────────────────────────────
# Features-first: z (c_z, N, N) → logits (no_bins, N, N)

@concrete struct DistogramHead
    c_z
    no_bins
    linear
end
@layer DistogramHead

function DistogramHead(c_z::Int; no_bins::Int = 64, rng::AbstractRNG = Random.default_rng())
    return DistogramHead(c_z, no_bins, Linear(c_z, no_bins; bias = true))
end

function (m::DistogramHead)(z::AbstractArray{<:Real,3})
    # z: (c_z, N, N) features-first
    size(z, 1) == m.c_z || error("DistogramHead c_z mismatch")
    logits = m.linear(z)  # (no_bins, N, N)
    # Symmetrize: swap spatial dims, keep features
    return logits .+ permutedims(logits, (1, 3, 2))
end

# ─── ConfidenceHead ──────────────────────────────────────────────────────────
# Features-first throughout: s (c_s, N), z (c_z, N, N), coords (3, N_atom, N_sample)

@concrete struct ConfidenceHead
    c_s
    c_z
    c_s_inputs
    b_pae
    b_pde
    b_plddt
    b_resolved
    max_atoms_per_token
    stop_gradient
    lower_bins
    upper_bins
    linear_no_bias_s1
    linear_no_bias_s2
    linear_no_bias_d
    linear_no_bias_d_wo_onehot
    pairformer_stack
    linear_no_bias_pae
    linear_no_bias_pde
    plddt_weight # [max_atoms_per_token, c_s, b_plddt] — matches Python checkpoint layout
    resolved_weight # [max_atoms_per_token, c_s, b_resolved]
    input_strunk_ln
    pae_ln
    pde_ln
    plddt_ln
    resolved_ln
end
@layer ConfidenceHead

function ConfidenceHead(
    c_s::Int,
    c_z::Int,
    c_s_inputs::Int;
    n_blocks::Int = 4,
    b_pae::Int = 64,
    b_pde::Int = 64,
    b_plddt::Int = 50,
    b_resolved::Int = 2,
    max_atoms_per_token::Int = 20,
    distance_bin_start::Real = 3.25,
    distance_bin_end::Real = 52.0,
    distance_bin_step::Real = 1.25,
    stop_gradient::Bool = true,
    rng::AbstractRNG = Random.default_rng(),
)
    lower = collect(Float32(distance_bin_start):Float32(distance_bin_step):Float32(distance_bin_end - distance_bin_step))
    upper = vcat(lower[2:end], Float32[1f6])

    return ConfidenceHead(
        c_s,
        c_z,
        c_s_inputs,
        b_pae,
        b_pde,
        b_plddt,
        b_resolved,
        max_atoms_per_token,
        stop_gradient,
        lower,
        upper,
        LinearNoBias(c_s_inputs, c_z; rng = rng),   # c_s_inputs → c_z (s1)
        LinearNoBias(c_s_inputs, c_z; rng = rng),   # c_s_inputs → c_z (s2)
        LinearNoBias(length(lower), c_z; rng = rng), # n_bins → c_z (d)
        LinearNoBias(1, c_z; rng = rng),             # 1 → c_z (d_wo_onehot)
        PairformerStack(c_z, c_s; n_blocks = n_blocks, n_heads = 16, rng = rng),
        LinearNoBias(c_z, b_pae; rng = rng),         # c_z → b_pae
        LinearNoBias(c_z, b_pde; rng = rng),         # c_z → b_pde
        zeros(Float32, max_atoms_per_token, c_s, b_plddt),
        zeros(Float32, max_atoms_per_token, c_s, b_resolved),
        LayerNorm(c_s),
        LayerNorm(c_z),
        LayerNorm(c_z),
        LayerNorm(c_s),
        LayerNorm(c_s),
    )
end

"""
Select representative atom coordinates.
Features-first: x_pred (3, N_atom, N_sample) → (3, N_rep, N_sample).
"""
function _select_rep_coords(
    x_pred_coords::AbstractArray{<:Real,3},
    rep_mask::AbstractVector{Bool},
)
    idx = findall(identity, rep_mask)
    return Float32.(x_pred_coords[:, idx, :])
end

"""
Per-atom head using gathered weight slices.
Features-first: a (c_s, N_atom), output (out_bins, N_atom).
Weight w stays in Python layout (max_atoms_per_token, c_s, out_bins).
"""
function _atom_head(
    a::AbstractMatrix{<:Real},
    atom_to_tokatom_idx::AbstractVector{<:Integer},
    w::AbstractArray{Float32,3},
)
    c_s, n_atom = size(a)
    size(w, 2) == c_s || error("atom head weight c_s mismatch")
    out_bins = size(w, 3)
    idx = Int.(atom_to_tokatom_idx) .+ 1
    all(i -> 1 <= i <= size(w, 1), idx) || error("atom_to_tokatom_idx out of range")
    w_sel = w[idx, :, :]  # (n_atom, c_s, out_bins)
    a_f = Float32.(a)     # (c_s, n_atom)
    out = fill!(similar(a_f, Float32, out_bins, n_atom), 0f0)
    for b in 1:out_bins
        for c in 1:c_s
            out[b, :] .+= a_f[c, :] .* w_sel[:, c, b]
        end
    end
    return out  # (out_bins, n_atom) features-first
end

"""
Augment pair representation with distance features.
Features-first: z (c_z, N, N), rep_coords (3, N_rep).
"""
function _distance_augmented_z(
    m::ConfidenceHead,
    z_pair::AbstractArray{<:Real,3},
    rep_coords::AbstractMatrix{<:Real},
)
    d = pairwise_distances(rep_coords)  # (N_rep, N_rep) from (3, N_rep)
    z = _as_f32_copy(z_pair)
    # one_hot_interval: (N, N) → (bins, N, N), linear: (bins, N, N) → (c_z, N, N)
    z .+= m.linear_no_bias_d(one_hot_interval(d, m.lower_bins, m.upper_bins))
    # Reshape d to (1, N, N) for linear: (1, N, N) → (c_z, N, N)
    z .+= m.linear_no_bias_d_wo_onehot(reshape(d, 1, size(d, 1), size(d, 2)))
    return z
end

function _symmetrize_pair_firstdim(x::AbstractArray{<:Real,3})
    return Float32.(x) .+ permutedims(Float32.(x), (1, 3, 2))
end

function (m::ConfidenceHead)(;
    input_feature_dict,
    s_inputs::AbstractMatrix{<:Real},
    s_trunk::AbstractMatrix{<:Real},
    z_trunk::AbstractArray{<:Real,3},
    pair_mask::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    x_pred_coords::AbstractArray{<:Real},
    use_embedding::Bool = true,
)
    # x_pred_coords: (3, N_atom, N_sample) or (3, N_atom)
    x_pred = if ndims(x_pred_coords) == 2
        x_f = _as_f32_array(x_pred_coords)
        reshape(x_f, 3, size(x_f, 2), 1)
    elseif ndims(x_pred_coords) == 3
        _as_f32_array(x_pred_coords)
    else
        error("x_pred_coords must be rank-2 or rank-3")
    end

    features = input_feature_dict isa ProtenixFeatures ?
        input_feature_dict : as_protenix_features(input_feature_dict)

    rep_mask = features.distogram_rep_atom_mask
    atom_to_token_idx = features.atom_to_token_idx
    atom_to_tokatom_idx = features.atom_to_tokatom_idx

    # Features-first: s_inputs (c_s_inputs, N_tok), s_trunk (c_s, N_tok), z_trunk (c_z, N, N)
    n_tok = size(s_inputs, 2)
    size(s_inputs, 1) == m.c_s_inputs || error("s_inputs channel mismatch")
    size(s_trunk) == (m.c_s, n_tok) || error("s_trunk shape mismatch")
    size(z_trunk) == (m.c_z, n_tok, n_tok) || error("z_trunk shape mismatch")

    s_base = m.input_strunk_ln(clamp.(_as_f32_array(s_trunk), -512f0, 512f0))
    z_base = _as_f32_copy(z_trunk)
    if !use_embedding
        z_base .= 0f0
    end

    s_inputs_f = _as_f32_array(s_inputs)
    s1 = m.linear_no_bias_s1(s_inputs_f)  # (c_z, N_tok)
    s2 = m.linear_no_bias_s2(s_inputs_f)  # (c_z, N_tok)
    # Outer sum: (c_z, N, 1) + (c_z, 1, N) → (c_z, N, N)
    z_init = reshape(s1, m.c_z, n_tok, 1) .+ reshape(s2, m.c_z, 1, n_tok)
    z_base .+= z_init

    x_rep = _select_rep_coords(x_pred, rep_mask)  # (3, N_rep, N_sample)
    n_sample = size(x_rep, 3)

    plddt_preds = Vector{AbstractMatrix{Float32}}(undef, n_sample)
    pae_preds = Vector{AbstractArray{Float32,3}}(undef, n_sample)
    pde_preds = Vector{AbstractArray{Float32,3}}(undef, n_sample)
    resolved_preds = Vector{AbstractMatrix{Float32}}(undef, n_sample)

    for i in 1:n_sample
        z_i = _distance_augmented_z(m, z_base, @view(x_rep[:, :, i]))  # (c_z, N, N)
        s_i, z_i = m.pairformer_stack(s_base, z_i; pair_mask = pair_mask)
        s_i === nothing && error("Confidence pairformer produced no single features")

        z_f = _as_f32_array(z_i)
        s_f = _as_f32_array(s_i)

        pae = m.linear_no_bias_pae(m.pae_ln(z_f))              # (b_pae, N, N)
        pde = m.linear_no_bias_pde(m.pde_ln(_symmetrize_pair_firstdim(z_f)))  # (b_pde, N, N)

        a_atom = broadcast_token_to_atom(s_f, atom_to_token_idx)  # (c_s, N_atom)
        plddt = _atom_head(m.plddt_ln(a_atom), atom_to_tokatom_idx, m.plddt_weight)      # (b_plddt, N_atom)
        resolved = _atom_head(m.resolved_ln(a_atom), atom_to_tokatom_idx, m.resolved_weight)  # (b_resolved, N_atom)

        plddt_preds[i] = plddt
        pae_preds[i] = pae
        pde_preds[i] = pde
        resolved_preds[i] = resolved
    end

    # Stack along last dim (sample dimension)
    plddt_out = cat([reshape(x, size(x, 1), size(x, 2), 1) for x in plddt_preds]...; dims = 3)
    pae_out = cat([reshape(x, size(x, 1), size(x, 2), size(x, 3), 1) for x in pae_preds]...; dims = 4)
    pde_out = cat([reshape(x, size(x, 1), size(x, 2), size(x, 3), 1) for x in pde_preds]...; dims = 4)
    resolved_out = cat([reshape(x, size(x, 1), size(x, 2), 1) for x in resolved_preds]...; dims = 3)

    return plddt_out, pae_out, pde_out, resolved_out
end

end

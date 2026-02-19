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

@concrete struct DistogramHead
    c_z
    no_bins
    linear
end
@layer DistogramHead

function DistogramHead(c_z::Int; no_bins::Int = 64, rng::AbstractRNG = Random.default_rng())
    return DistogramHead(c_z, no_bins, Linear(no_bins, c_z; bias = true, rng = rng))
end

function (m::DistogramHead)(z::AbstractArray{<:Real,3})
    size(z, 3) == m.c_z || error("DistogramHead c_z mismatch")
    logits = m.linear(z)
    # Symmetrize: out[i,j,:] = logits[i,j,:] + logits[j,i,:]
    return logits .+ permutedims(logits, (2, 1, 3))
end

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
    plddt_weight # [max_atoms_per_token, c_s, b_plddt]
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
        LinearNoBias(c_z, c_s_inputs; rng = rng),
        LinearNoBias(c_z, c_s_inputs; rng = rng),
        LinearNoBias(c_z, length(lower); rng = rng),
        LinearNoBias(c_z, 1; rng = rng),
        PairformerStack(c_z, c_s; n_blocks = n_blocks, n_heads = 16, rng = rng),
        LinearNoBias(b_pae, c_z; rng = rng),
        LinearNoBias(b_pde, c_z; rng = rng),
        zeros(Float32, max_atoms_per_token, c_s, b_plddt),
        zeros(Float32, max_atoms_per_token, c_s, b_resolved),
        LayerNorm(c_s),
        LayerNorm(c_z),
        LayerNorm(c_z),
        LayerNorm(c_s),
        LayerNorm(c_s),
    )
end

function _select_rep_coords(
    x_pred_coords::AbstractArray{<:Real,3},
    rep_mask::AbstractVector{Bool},
)
    idx = findall(identity, rep_mask)
    return Float32.(x_pred_coords[:, idx, :])
end

function _atom_head(
    a::AbstractMatrix{<:Real},
    atom_to_tokatom_idx::AbstractVector{<:Integer},
    w::AbstractArray{Float32,3},
)
    n_atom, c_s = size(a)
    size(w, 2) == c_s || error("atom head weight c_s mismatch")
    out_bins = size(w, 3)
    # Gather per-atom weight slices: w_sel[n, c, b] = w[tokatom[n]+1, c, b]
    idx = Int.(atom_to_tokatom_idx) .+ 1
    all(i -> 1 <= i <= size(w, 1), idx) || error("atom_to_tokatom_idx out of range")
    w_sel = w[idx, :, :] # [n_atom, c_s, out_bins]
    # out[n, b] = sum_c a[n, c] * w_sel[n, c, b]
    a_f = Float32.(a)
    out = fill!(similar(a_f, Float32, n_atom, out_bins), 0f0)
    for b in 1:out_bins
        for c in 1:c_s
            out[:, b] .+= a_f[:, c] .* w_sel[:, c, b]
        end
    end
    return out
end

function _distance_augmented_z(
    m::ConfidenceHead,
    z_pair::AbstractArray{<:Real,3},
    rep_coords::AbstractMatrix{<:Real},
)
    d = pairwise_distances(rep_coords)
    z = _as_f32_copy(z_pair)
    z .+= m.linear_no_bias_d(one_hot_interval(d, m.lower_bins, m.upper_bins))
    z .+= m.linear_no_bias_d_wo_onehot(reshape(d, size(d, 1), size(d, 2), 1))
    return z
end

function _symmetrize_pair_lastdim(x::AbstractArray{<:Real,3})
    return Float32.(x) .+ permutedims(Float32.(x), (2, 1, 3))
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
    # x_pred_coords expected [N_sample, N_atom, 3] or [N_atom, 3]
    x_pred = if ndims(x_pred_coords) == 2
        x_f = _as_f32_array(x_pred_coords)
        reshape(x_f, 1, size(x_f, 1), 3)
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

    n_tok = size(s_inputs, 1)
    size(s_inputs, 2) == m.c_s_inputs || error("s_inputs channel mismatch")
    size(s_trunk) == (n_tok, m.c_s) || error("s_trunk shape mismatch")
    size(z_trunk) == (n_tok, n_tok, m.c_z) || error("z_trunk shape mismatch")

    s_base = m.input_strunk_ln(clamp.(_as_f32_array(s_trunk), -512f0, 512f0))
    z_base = _as_f32_copy(z_trunk)
    if !use_embedding
        z_base .= 0f0
    end

    s_inputs_f = _as_f32_array(s_inputs)
    s1 = m.linear_no_bias_s1(s_inputs_f)
    s2 = m.linear_no_bias_s2(s_inputs_f)
    z_init = reshape(s1, 1, n_tok, m.c_z) .+ reshape(s2, n_tok, 1, m.c_z)
    z_base .+= z_init

    x_rep = _select_rep_coords(x_pred, rep_mask)
    n_sample = size(x_rep, 1)

    plddt_preds = Vector{AbstractArray{Float32,2}}(undef, n_sample)
    pae_preds = Vector{AbstractArray{Float32,3}}(undef, n_sample)
    pde_preds = Vector{AbstractArray{Float32,3}}(undef, n_sample)
    resolved_preds = Vector{AbstractArray{Float32,2}}(undef, n_sample)

    for i in 1:n_sample
        z_i = _distance_augmented_z(m, z_base, @view(x_rep[i, :, :]))
        s_i, z_i = m.pairformer_stack(s_base, z_i; pair_mask = pair_mask)
        s_i === nothing && error("Confidence pairformer produced no single features")

        z_f = _as_f32_array(z_i)
        s_f = _as_f32_array(s_i)

        pae = m.linear_no_bias_pae(m.pae_ln(z_f))
        pde = m.linear_no_bias_pde(m.pde_ln(_symmetrize_pair_lastdim(z_f)))

        a_atom = broadcast_token_to_atom(s_f, atom_to_token_idx)
        plddt = _atom_head(m.plddt_ln(a_atom), atom_to_tokatom_idx, m.plddt_weight)
        resolved = _atom_head(m.resolved_ln(a_atom), atom_to_tokatom_idx, m.resolved_weight)

        plddt_preds[i] = plddt
        pae_preds[i] = pae
        pde_preds[i] = pde
        resolved_preds[i] = resolved
    end

    plddt_out = cat([reshape(x, 1, size(x, 1), size(x, 2)) for x in plddt_preds]...; dims = 1)
    pae_out = cat([reshape(x, 1, size(x, 1), size(x, 2), size(x, 3)) for x in pae_preds]...; dims = 1)
    pde_out = cat([reshape(x, 1, size(x, 1), size(x, 2), size(x, 3)) for x in pde_preds]...; dims = 1)
    resolved_out = cat([reshape(x, 1, size(x, 1), size(x, 2)) for x in resolved_preds]...; dims = 1)

    return plddt_out, pae_out, pde_out, resolved_out
end

end

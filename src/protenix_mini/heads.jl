module Heads

using Random

import ..Primitives: Linear, LinearNoBias, LayerNorm
import ..Pairformer: PairformerStack
import ..Utils: one_hot_interval, broadcast_token_to_atom, pairwise_distances

export DistogramHead, ConfidenceHead

struct DistogramHead
    c_z::Int
    no_bins::Int
    linear::Linear
end

function DistogramHead(c_z::Int; no_bins::Int = 64, rng::AbstractRNG = Random.default_rng())
    return DistogramHead(c_z, no_bins, Linear(no_bins, c_z; bias = true, rng = rng))
end

function (m::DistogramHead)(z::AbstractArray{<:Real,3})
    size(z, 3) == m.c_z || error("DistogramHead c_z mismatch")
    logits = m.linear(z)
    return logits .+ permutedims(logits, (2, 1, 3))
end

struct ConfidenceHead
    c_s::Int
    c_z::Int
    c_s_inputs::Int
    b_pae::Int
    b_pde::Int
    b_plddt::Int
    b_resolved::Int
    max_atoms_per_token::Int
    stop_gradient::Bool
    lower_bins::Vector{Float32}
    upper_bins::Vector{Float32}
    linear_no_bias_s1::LinearNoBias
    linear_no_bias_s2::LinearNoBias
    linear_no_bias_d::LinearNoBias
    linear_no_bias_d_wo_onehot::LinearNoBias
    pairformer_stack::PairformerStack
    linear_no_bias_pae::LinearNoBias
    linear_no_bias_pde::LinearNoBias
    plddt_weight::Array{Float32,3} # [max_atoms_per_token, c_s, b_plddt]
    resolved_weight::Array{Float32,3} # [max_atoms_per_token, c_s, b_resolved]
    input_strunk_ln::LayerNorm
    pae_ln::LayerNorm
    pde_ln::LayerNorm
    plddt_ln::LayerNorm
    resolved_ln::LayerNorm
end

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
    n_sample = size(x_pred_coords, 1)
    n_rep = count(identity, rep_mask)
    out = zeros(Float32, n_sample, n_rep, 3)
    idx = findall(identity, rep_mask)
    @inbounds for s in 1:n_sample, r in 1:n_rep
        out[s, r, 1] = Float32(x_pred_coords[s, idx[r], 1])
        out[s, r, 2] = Float32(x_pred_coords[s, idx[r], 2])
        out[s, r, 3] = Float32(x_pred_coords[s, idx[r], 3])
    end
    return out
end

function _atom_head(
    a::AbstractMatrix{<:Real},
    atom_to_tokatom_idx::AbstractVector{<:Integer},
    w::Array{Float32,3},
)
    n_atom, c_s = size(a)
    size(w, 2) == c_s || error("atom head weight c_s mismatch")
    out_bins = size(w, 3)
    out = zeros(Float32, n_atom, out_bins)
    @inbounds for n in 1:n_atom
        t = Int(atom_to_tokatom_idx[n]) + 1
        1 <= t <= size(w, 1) || error("atom_to_tokatom_idx[$n] out of range")
        out[n, :] .= vec(transpose(Float32.(a[n, :])) * @view(w[t, :, :]))
    end
    return out
end

function _distance_augmented_z(
    m::ConfidenceHead,
    z_pair::AbstractArray{<:Real,3},
    rep_coords::AbstractMatrix{<:Real},
)
    d = pairwise_distances(rep_coords)
    z = Float32.(z_pair)
    z .+= m.linear_no_bias_d(one_hot_interval(d, m.lower_bins, m.upper_bins))
    z .+= m.linear_no_bias_d_wo_onehot(reshape(d, size(d, 1), size(d, 2), 1))
    return z
end

function (m::ConfidenceHead)(;
    input_feature_dict::AbstractDict{<:AbstractString, <:Any},
    s_inputs::AbstractMatrix{<:Real},
    s_trunk::AbstractMatrix{<:Real},
    z_trunk::AbstractArray{<:Real,3},
    pair_mask::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    x_pred_coords::AbstractArray{<:Real},
    use_embedding::Bool = true,
)
    # x_pred_coords expected [N_sample, N_atom, 3] or [N_atom, 3]
    x_pred = if ndims(x_pred_coords) == 2
        reshape(Float32.(x_pred_coords), 1, size(x_pred_coords, 1), 3)
    elseif ndims(x_pred_coords) == 3
        Float32.(x_pred_coords)
    else
        error("x_pred_coords must be rank-2 or rank-3")
    end

    haskey(input_feature_dict, "distogram_rep_atom_mask") ||
        error("Missing distogram_rep_atom_mask for ConfidenceHead")
    haskey(input_feature_dict, "atom_to_token_idx") ||
        error("Missing atom_to_token_idx for ConfidenceHead")
    haskey(input_feature_dict, "atom_to_tokatom_idx") ||
        error("Missing atom_to_tokatom_idx for ConfidenceHead")

    rep_mask = Bool.(input_feature_dict["distogram_rep_atom_mask"])
    atom_to_token_idx = Int.(input_feature_dict["atom_to_token_idx"])
    atom_to_tokatom_idx = Int.(input_feature_dict["atom_to_tokatom_idx"])

    n_tok = size(s_inputs, 1)
    size(s_inputs, 2) == m.c_s_inputs || error("s_inputs channel mismatch")
    size(s_trunk) == (n_tok, m.c_s) || error("s_trunk shape mismatch")
    size(z_trunk) == (n_tok, n_tok, m.c_z) || error("z_trunk shape mismatch")

    s_base = m.input_strunk_ln(clamp.(Float32.(s_trunk), -512f0, 512f0))
    z_base = Float32.(z_trunk)
    if !use_embedding
        z_base .= 0f0
    end

    s1 = m.linear_no_bias_s1(Float32.(s_inputs))
    s2 = m.linear_no_bias_s2(Float32.(s_inputs))
    z_init = reshape(s1, 1, n_tok, m.c_z) .+ reshape(s2, n_tok, 1, m.c_z)
    z_base .+= z_init

    x_rep = _select_rep_coords(x_pred, rep_mask)
    n_sample = size(x_rep, 1)

    plddt_preds = Vector{Array{Float32,2}}(undef, n_sample)
    pae_preds = Vector{Array{Float32,3}}(undef, n_sample)
    pde_preds = Vector{Array{Float32,3}}(undef, n_sample)
    resolved_preds = Vector{Array{Float32,2}}(undef, n_sample)

    for i in 1:n_sample
        z_i = _distance_augmented_z(m, z_base, @view(x_rep[i, :, :]))
        s_i, z_i = m.pairformer_stack(s_base, z_i; pair_mask = pair_mask)
        s_i === nothing && error("Confidence pairformer produced no single features")

        z_f = Float32.(z_i)
        s_f = Float32.(s_i)

        pae = m.linear_no_bias_pae(m.pae_ln(z_f))
        pde = m.linear_no_bias_pde(m.pde_ln(z_f .+ permutedims(z_f, (2, 1, 3))))

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

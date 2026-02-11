module StateLoad

using Random

import ..Primitives: Linear, LinearNoBias, LayerNorm
import ..OpenFoldBlocks: PairAttentionNoS, TriangleMultiplication, TriangleAttention, OuterProductMean
import ..Embedders: InputFeatureEmbedder, RelativePositionEncoding
import ..Pairformer:
    TransitionBlock,
    PairformerBlock,
    PairformerStack,
    MSAPairWeightedAveraging,
    MSAStack,
    MSABlock,
    MSAModule,
    TemplateEmbedder,
    NoisyStructureEmbedder
import ..Heads: DistogramHead, ConfidenceHead
import ..Model: ProtenixMiniModel
import ...Model: load_diffusion_module!, infer_model_scaffold_dims

export infer_protenix_mini_dims, build_protenix_mini_model, load_protenix_mini_model!

function _key(weights::AbstractDict{<:AbstractString, <:Any}, key::String; strict::Bool = true)
    if haskey(weights, key)
        return weights[key]
    end
    strict && error("Missing checkpoint tensor: $key")
    return nothing
end

function _load_vector!(dst::Vector{Float32}, src, key::String)
    src isa AbstractVector || error("Checkpoint key $key must be vector")
    length(dst) == length(src) || error("Shape mismatch for $key")
    dst .= Float32.(src)
    return dst
end

function _load_matrix!(dst::Matrix{Float32}, src, key::String)
    src isa AbstractMatrix || error("Checkpoint key $key must be matrix")
    size(dst) == size(src) || error("Shape mismatch for $key: expected $(size(dst)) got $(size(src))")
    dst .= Float32.(src)
    return dst
end

function _load_linear_nobias!(
    lin,
    weights::AbstractDict{<:AbstractString, <:Any},
    key::String;
    strict::Bool = true,
)
    hasproperty(lin, :weight) || error("LinearNoBias-like object missing `weight` field for key $key")
    src = _key(weights, key; strict = strict)
    src === nothing && return lin
    _load_matrix!(getfield(lin, :weight), src, key)
    return lin
end

function _load_linear!(
    lin::Linear,
    weights::AbstractDict{<:AbstractString, <:Any},
    w_key::String;
    b_key::Union{Nothing, String} = nothing,
    strict::Bool = true,
)
    src = _key(weights, w_key; strict = strict)
    src === nothing && return lin
    _load_matrix!(lin.weight, src, w_key)
    if lin.bias !== nothing
        bname = b_key === nothing ? replace(w_key, ".weight" => ".bias") : b_key
        b = _key(weights, bname; strict = false)
        b !== nothing && _load_vector!(lin.bias, b, bname)
    end
    return lin
end

function _load_layernorm!(
    ln,
    weights::AbstractDict{<:AbstractString, <:Any},
    prefix::String;
    strict::Bool = true,
)
    hasproperty(ln, :weight) || error("LayerNorm-like object missing `weight` field for prefix $prefix")
    hasproperty(ln, :bias) || error("LayerNorm-like object missing `bias` field for prefix $prefix")
    w = _key(weights, "$prefix.weight"; strict = strict)
    w !== nothing && _load_vector!(getfield(ln, :weight), w, "$prefix.weight")
    b = _key(weights, "$prefix.bias"; strict = false)
    b !== nothing && _load_vector!(getfield(ln, :bias), b, "$prefix.bias")
    return ln
end

function _load_transition!(
    tr::TransitionBlock,
    weights::AbstractDict{<:AbstractString, <:Any},
    prefix::String;
    strict::Bool = true,
)
    _load_layernorm!(tr.layernorm1, weights, "$prefix.layernorm1"; strict = strict)
    _load_linear_nobias!(tr.linear_no_bias_a, weights, "$prefix.linear_no_bias_a.weight"; strict = strict)
    _load_linear_nobias!(tr.linear_no_bias_b, weights, "$prefix.linear_no_bias_b.weight"; strict = strict)
    _load_linear_nobias!(tr.linear_no_bias, weights, "$prefix.linear_no_bias.weight"; strict = strict)
    return tr
end

function _load_pair_attention_no_s!(
    apb::PairAttentionNoS,
    weights::AbstractDict{<:AbstractString, <:Any},
    prefix::String;
    strict::Bool = true,
)
    _load_layernorm!(apb.layernorm_a, weights, "$prefix.layernorm_a"; strict = strict)
    _load_layernorm!(apb.layernorm_z, weights, "$prefix.layernorm_z"; strict = strict)
    _load_linear_nobias!(apb.linear_nobias_z, weights, "$prefix.linear_nobias_z.weight"; strict = strict)
    _load_linear!(apb.linear_q, weights, "$prefix.attention.linear_q.weight"; b_key = "$prefix.attention.linear_q.bias", strict = strict)
    _load_linear_nobias!(apb.linear_k, weights, "$prefix.attention.linear_k.weight"; strict = strict)
    _load_linear_nobias!(apb.linear_v, weights, "$prefix.attention.linear_v.weight"; strict = strict)
    _load_linear_nobias!(apb.linear_o, weights, "$prefix.attention.linear_o.weight"; strict = strict)
    _load_linear_nobias!(apb.linear_g, weights, "$prefix.attention.linear_g.weight"; strict = strict)
    return apb
end

function _load_triangle_mul!(
    tm::TriangleMultiplication,
    weights::AbstractDict{<:AbstractString, <:Any},
    prefix::String;
    strict::Bool = true,
)
    _load_layernorm!(tm.layer_norm_in, weights, "$prefix.layer_norm_in"; strict = strict)
    _load_layernorm!(tm.layer_norm_out, weights, "$prefix.layer_norm_out"; strict = strict)
    _load_linear_nobias!(tm.linear_a_p, weights, "$prefix.linear_a_p.weight"; strict = strict)
    _load_linear_nobias!(tm.linear_a_g, weights, "$prefix.linear_a_g.weight"; strict = strict)
    _load_linear_nobias!(tm.linear_b_p, weights, "$prefix.linear_b_p.weight"; strict = strict)
    _load_linear_nobias!(tm.linear_b_g, weights, "$prefix.linear_b_g.weight"; strict = strict)
    _load_linear_nobias!(tm.linear_z, weights, "$prefix.linear_z.weight"; strict = strict)
    _load_linear_nobias!(tm.linear_g, weights, "$prefix.linear_g.weight"; strict = strict)
    return tm
end

function _load_triangle_att!(
    ta::TriangleAttention,
    weights::AbstractDict{<:AbstractString, <:Any},
    prefix::String;
    strict::Bool = true,
)
    _load_layernorm!(ta.layer_norm, weights, "$prefix.layer_norm"; strict = strict)
    _load_linear_nobias!(ta.linear, weights, "$prefix.linear.weight"; strict = strict)
    _load_linear_nobias!(ta.linear_q, weights, "$prefix.mha.linear_q.weight"; strict = strict)
    _load_linear_nobias!(ta.linear_k, weights, "$prefix.mha.linear_k.weight"; strict = strict)
    _load_linear_nobias!(ta.linear_v, weights, "$prefix.mha.linear_v.weight"; strict = strict)
    _load_linear_nobias!(ta.linear_o, weights, "$prefix.mha.linear_o.weight"; strict = strict)
    _load_linear_nobias!(ta.linear_g, weights, "$prefix.mha.linear_g.weight"; strict = strict)
    return ta
end

function _load_outer_product_mean!(
    opm::OuterProductMean,
    weights::AbstractDict{<:AbstractString, <:Any},
    prefix::String;
    strict::Bool = true,
)
    _load_layernorm!(opm.layer_norm, weights, "$prefix.layer_norm"; strict = strict)
    _load_linear_nobias!(opm.linear_1, weights, "$prefix.linear_1.weight"; strict = strict)
    _load_linear_nobias!(opm.linear_2, weights, "$prefix.linear_2.weight"; strict = strict)
    _load_linear!(opm.linear_out, weights, "$prefix.linear_out.weight"; b_key = "$prefix.linear_out.bias", strict = strict)
    return opm
end

function _load_pairformer_block!(
    blk::PairformerBlock,
    weights::AbstractDict{<:AbstractString, <:Any},
    prefix::String;
    strict::Bool = true,
)
    _load_triangle_mul!(blk.tri_mul_out, weights, "$prefix.tri_mul_out"; strict = strict)
    _load_triangle_mul!(blk.tri_mul_in, weights, "$prefix.tri_mul_in"; strict = strict)
    _load_triangle_att!(blk.tri_att_start, weights, "$prefix.tri_att_start"; strict = strict)
    _load_triangle_att!(blk.tri_att_end, weights, "$prefix.tri_att_end"; strict = strict)
    _load_transition!(blk.pair_transition, weights, "$prefix.pair_transition"; strict = strict)
    if blk.c_s > 0
        blk.attention_pair_bias === nothing && error("Missing attention_pair_bias")
        blk.single_transition === nothing && error("Missing single_transition")
        _load_pair_attention_no_s!(blk.attention_pair_bias, weights, "$prefix.attention_pair_bias"; strict = strict)
        _load_transition!(blk.single_transition, weights, "$prefix.single_transition"; strict = strict)
    end
    return blk
end

function _load_pairformer_stack!(
    stk::PairformerStack,
    weights::AbstractDict{<:AbstractString, <:Any},
    prefix::String;
    strict::Bool = true,
)
    for (i, blk) in enumerate(stk.blocks)
        _load_pairformer_block!(blk, weights, "$prefix.blocks.$(i - 1)"; strict = strict)
    end
    return stk
end

function _load_msa_pair_weighted!(
    msa::MSAPairWeightedAveraging,
    weights::AbstractDict{<:AbstractString, <:Any},
    prefix::String;
    strict::Bool = true,
)
    _load_layernorm!(msa.layernorm_m, weights, "$prefix.layernorm_m"; strict = strict)
    _load_linear_nobias!(msa.linear_no_bias_mv, weights, "$prefix.linear_no_bias_mv.weight"; strict = strict)
    _load_layernorm!(msa.layernorm_z, weights, "$prefix.layernorm_z"; strict = strict)
    _load_linear_nobias!(msa.linear_no_bias_z, weights, "$prefix.linear_no_bias_z.weight"; strict = strict)
    _load_linear_nobias!(msa.linear_no_bias_mg, weights, "$prefix.linear_no_bias_mg.weight"; strict = strict)
    _load_linear_nobias!(msa.linear_no_bias_out, weights, "$prefix.linear_no_bias_out.weight"; strict = strict)
    return msa
end

function _load_msa_stack!(
    stk::MSAStack,
    weights::AbstractDict{<:AbstractString, <:Any},
    prefix::String;
    strict::Bool = true,
)
    _load_msa_pair_weighted!(stk.msa_pair_weighted_averaging, weights, "$prefix.msa_pair_weighted_averaging"; strict = strict)
    _load_transition!(stk.transition_m, weights, "$prefix.transition_m"; strict = strict)
    return stk
end

function _load_msa_block!(
    blk::MSABlock,
    weights::AbstractDict{<:AbstractString, <:Any},
    prefix::String;
    strict::Bool = true,
)
    _load_outer_product_mean!(blk.outer_product_mean_msa, weights, "$prefix.outer_product_mean_msa"; strict = strict)
    if blk.msa_stack !== nothing
        _load_msa_stack!(blk.msa_stack, weights, "$prefix.msa_stack"; strict = strict)
    end
    _load_pairformer_block!(blk.pair_stack, weights, "$prefix.pair_stack"; strict = strict)
    return blk
end

function _load_msa_module!(
    msa::MSAModule,
    weights::AbstractDict{<:AbstractString, <:Any},
    prefix::String;
    strict::Bool = true,
)
    _load_linear_nobias!(msa.linear_no_bias_m, weights, "$prefix.linear_no_bias_m.weight"; strict = strict)
    _load_linear_nobias!(msa.linear_no_bias_s, weights, "$prefix.linear_no_bias_s.weight"; strict = strict)
    for (i, blk) in enumerate(msa.blocks)
        _load_msa_block!(blk, weights, "$prefix.blocks.$(i - 1)"; strict = strict)
    end
    return msa
end

function _load_template_embedder!(
    t::TemplateEmbedder,
    weights::AbstractDict{<:AbstractString, <:Any},
    prefix::String;
    strict::Bool = true,
)
    _load_layernorm!(t.layernorm_z, weights, "$prefix.layernorm_z"; strict = strict)
    _load_linear_nobias!(t.linear_no_bias_z, weights, "$prefix.linear_no_bias_z.weight"; strict = strict)
    _load_linear_nobias!(t.linear_no_bias_a, weights, "$prefix.linear_no_bias_a.weight"; strict = strict)
    _load_layernorm!(t.layernorm_v, weights, "$prefix.layernorm_v"; strict = strict)
    _load_linear_nobias!(t.linear_no_bias_u, weights, "$prefix.linear_no_bias_u.weight"; strict = strict)
    # pairformer_stack exists but template path is disabled in reference; load only if keys exist.
    return t
end

function _load_noisy_structure_embedder!(
    nse::NoisyStructureEmbedder,
    weights::AbstractDict{<:AbstractString, <:Any},
    prefix::String;
    strict::Bool = true,
)
    _load_linear_nobias!(nse.linear_struct, weights, "$prefix.linear_struct.weight"; strict = strict)
    _load_layernorm!(nse.layernorm_z, weights, "$prefix.layernorm_z"; strict = strict)
    _load_linear_nobias!(nse.linear_z, weights, "$prefix.linear_z.weight"; strict = strict)
    _load_transition!(nse.transition_out, weights, "$prefix.transition_out"; strict = strict)
    return nse
end

function _load_input_feature_embedder!(
    inp::InputFeatureEmbedder,
    weights::AbstractDict{<:AbstractString, <:Any},
    prefix::String;
    strict::Bool = true,
)
    # Reuse existing atom-attention loader shape by matching keys from diffusion loader logic.
    enc = inp.atom_attention_encoder
    _load_linear_nobias!(enc.linear_no_bias_ref_pos, weights, "$prefix.atom_attention_encoder.linear_no_bias_ref_pos.weight"; strict = strict)
    _load_linear_nobias!(enc.linear_no_bias_ref_charge, weights, "$prefix.atom_attention_encoder.linear_no_bias_ref_charge.weight"; strict = strict)
    _load_linear_nobias!(enc.linear_no_bias_f, weights, "$prefix.atom_attention_encoder.linear_no_bias_f.weight"; strict = strict)
    _load_linear_nobias!(enc.linear_no_bias_d, weights, "$prefix.atom_attention_encoder.linear_no_bias_d.weight"; strict = strict)
    _load_linear_nobias!(enc.linear_no_bias_invd, weights, "$prefix.atom_attention_encoder.linear_no_bias_invd.weight"; strict = strict)
    _load_linear_nobias!(enc.linear_no_bias_v, weights, "$prefix.atom_attention_encoder.linear_no_bias_v.weight"; strict = strict)
    _load_linear_nobias!(enc.linear_no_bias_cl, weights, "$prefix.atom_attention_encoder.linear_no_bias_cl.weight"; strict = strict)
    _load_linear_nobias!(enc.linear_no_bias_cm, weights, "$prefix.atom_attention_encoder.linear_no_bias_cm.weight"; strict = strict)
    _load_linear_nobias!(enc.small_mlp_1, weights, "$prefix.atom_attention_encoder.small_mlp.1.weight"; strict = strict)
    _load_linear_nobias!(enc.small_mlp_2, weights, "$prefix.atom_attention_encoder.small_mlp.3.weight"; strict = strict)
    _load_linear_nobias!(enc.small_mlp_3, weights, "$prefix.atom_attention_encoder.small_mlp.5.weight"; strict = strict)

    # atom transformer blocks
    for (i, blk) in enumerate(enc.atom_transformer.blocks)
        b = "$prefix.atom_attention_encoder.atom_transformer.diffusion_transformer.blocks.$(i - 1)"
        apb = blk.attention_pair_bias
        _load_layernorm!(apb.adaln_a.layernorm_s, weights, "$b.attention_pair_bias.layernorm_a.layernorm_s"; strict = strict)
        _load_linear_nobias!(apb.adaln_a.linear_nobias_s, weights, "$b.attention_pair_bias.layernorm_a.linear_nobias_s.weight"; strict = strict)
        _load_linear_nobias!(apb.adaln_a.linear_s, weights, "$b.attention_pair_bias.layernorm_a.linear_s.weight"; strict = strict)
        b_s = _key(weights, "$b.attention_pair_bias.layernorm_a.linear_s.bias"; strict = false)
        b_s !== nothing && _load_vector!(apb.adaln_a.bias_s, b_s, "$b.attention_pair_bias.layernorm_a.linear_s.bias")

        apb.adaln_kv !== nothing || error("Expected cross-attention adaptive LN in input embedder atom transformer")
        _load_layernorm!(apb.adaln_kv.layernorm_s, weights, "$b.attention_pair_bias.layernorm_kv.layernorm_s"; strict = strict)
        _load_linear_nobias!(apb.adaln_kv.linear_nobias_s, weights, "$b.attention_pair_bias.layernorm_kv.linear_nobias_s.weight"; strict = strict)
        _load_linear_nobias!(apb.adaln_kv.linear_s, weights, "$b.attention_pair_bias.layernorm_kv.linear_s.weight"; strict = strict)
        b_kv = _key(weights, "$b.attention_pair_bias.layernorm_kv.linear_s.bias"; strict = false)
        b_kv !== nothing && _load_vector!(apb.adaln_kv.bias_s, b_kv, "$b.attention_pair_bias.layernorm_kv.linear_s.bias")

        _load_layernorm!(apb.layernorm_z, weights, "$b.attention_pair_bias.layernorm_z"; strict = strict)
        _load_linear_nobias!(apb.linear_bias_z, weights, "$b.attention_pair_bias.linear_nobias_z.weight"; strict = strict)
        _load_linear_nobias!(apb.linear_q, weights, "$b.attention_pair_bias.attention.linear_q.weight"; strict = strict)
        qbias = _key(weights, "$b.attention_pair_bias.attention.linear_q.bias"; strict = false)
        qbias !== nothing && _load_vector!(apb.bias_q, qbias, "$b.attention_pair_bias.attention.linear_q.bias")
        _load_linear_nobias!(apb.linear_k, weights, "$b.attention_pair_bias.attention.linear_k.weight"; strict = strict)
        _load_linear_nobias!(apb.linear_v, weights, "$b.attention_pair_bias.attention.linear_v.weight"; strict = strict)
        _load_linear_nobias!(apb.linear_o, weights, "$b.attention_pair_bias.attention.linear_o.weight"; strict = strict)
        _load_linear_nobias!(apb.linear_g, weights, "$b.attention_pair_bias.attention.linear_g.weight"; strict = strict)
        _load_linear_nobias!(apb.linear_a_last, weights, "$b.attention_pair_bias.linear_a_last.weight"; strict = strict)
        alast_bias = _key(weights, "$b.attention_pair_bias.linear_a_last.bias"; strict = false)
        alast_bias !== nothing && _load_vector!(apb.bias_a_last, alast_bias, "$b.attention_pair_bias.linear_a_last.bias")

        ctb = blk.conditioned_transition_block
        _load_layernorm!(ctb.adaln.layernorm_s, weights, "$b.conditioned_transition_block.adaln.layernorm_s"; strict = strict)
        _load_linear_nobias!(ctb.adaln.linear_nobias_s, weights, "$b.conditioned_transition_block.adaln.linear_nobias_s.weight"; strict = strict)
        _load_linear_nobias!(ctb.adaln.linear_s, weights, "$b.conditioned_transition_block.adaln.linear_s.weight"; strict = strict)
        ctbias = _key(weights, "$b.conditioned_transition_block.adaln.linear_s.bias"; strict = false)
        ctbias !== nothing && _load_vector!(ctb.adaln.bias_s, ctbias, "$b.conditioned_transition_block.adaln.linear_s.bias")
        _load_linear_nobias!(ctb.linear_a1, weights, "$b.conditioned_transition_block.linear_nobias_a1.weight"; strict = strict)
        _load_linear_nobias!(ctb.linear_a2, weights, "$b.conditioned_transition_block.linear_nobias_a2.weight"; strict = strict)
        _load_linear_nobias!(ctb.linear_b, weights, "$b.conditioned_transition_block.linear_nobias_b.weight"; strict = strict)
        _load_linear_nobias!(ctb.linear_s, weights, "$b.conditioned_transition_block.linear_s.weight"; strict = strict)
        sbias = _key(weights, "$b.conditioned_transition_block.linear_s.bias"; strict = false)
        sbias !== nothing && _load_vector!(ctb.bias_s, sbias, "$b.conditioned_transition_block.linear_s.bias")
    end

    _load_linear_nobias!(enc.linear_no_bias_q, weights, "$prefix.atom_attention_encoder.linear_no_bias_q.weight"; strict = strict)

    if inp.esm_enable && inp.linear_esm !== nothing
        _load_linear_nobias!(inp.linear_esm, weights, "$prefix.linear_esm.weight"; strict = strict)
    end

    return inp
end

function _load_confidence_head!(
    conf::ConfidenceHead,
    weights::AbstractDict{<:AbstractString, <:Any},
    prefix::String;
    strict::Bool = true,
)
    _load_linear_nobias!(conf.linear_no_bias_s1, weights, "$prefix.linear_no_bias_s1.weight"; strict = strict)
    _load_linear_nobias!(conf.linear_no_bias_s2, weights, "$prefix.linear_no_bias_s2.weight"; strict = strict)
    _load_linear_nobias!(conf.linear_no_bias_d, weights, "$prefix.linear_no_bias_d.weight"; strict = strict)
    _load_linear_nobias!(conf.linear_no_bias_d_wo_onehot, weights, "$prefix.linear_no_bias_d_wo_onehot.weight"; strict = strict)
    _load_pairformer_stack!(conf.pairformer_stack, weights, "$prefix.pairformer_stack"; strict = strict)
    _load_linear_nobias!(conf.linear_no_bias_pae, weights, "$prefix.linear_no_bias_pae.weight"; strict = strict)
    _load_linear_nobias!(conf.linear_no_bias_pde, weights, "$prefix.linear_no_bias_pde.weight"; strict = strict)

    plddt_w = _key(weights, "$prefix.plddt_weight"; strict = strict)
    plddt_w !== nothing || return conf
    size(conf.plddt_weight) == size(plddt_w) || error("Shape mismatch for $prefix.plddt_weight")
    conf.plddt_weight .= Float32.(plddt_w)

    resolved_w = _key(weights, "$prefix.resolved_weight"; strict = strict)
    resolved_w !== nothing && (size(conf.resolved_weight) == size(resolved_w) || error("Shape mismatch for $prefix.resolved_weight"))
    resolved_w !== nothing && (conf.resolved_weight .= Float32.(resolved_w))

    lower = _key(weights, "$prefix.lower_bins"; strict = false)
    if lower !== nothing
        length(conf.lower_bins) == length(lower) || error("Shape mismatch for $prefix.lower_bins")
        conf.lower_bins .= Float32.(lower)
        conf.upper_bins .= vcat(conf.lower_bins[2:end], Float32[1f6])
    end

    _load_layernorm!(conf.input_strunk_ln, weights, "$prefix.input_strunk_ln"; strict = strict)
    _load_layernorm!(conf.pae_ln, weights, "$prefix.pae_ln"; strict = strict)
    _load_layernorm!(conf.pde_ln, weights, "$prefix.pde_ln"; strict = strict)
    _load_layernorm!(conf.plddt_ln, weights, "$prefix.plddt_ln"; strict = strict)
    _load_layernorm!(conf.resolved_ln, weights, "$prefix.resolved_ln"; strict = strict)
    return conf
end

function infer_protenix_mini_dims(weights::AbstractDict{<:AbstractString, <:Any})
    dm = infer_model_scaffold_dims(weights)

    # c_s_inputs from trunk init projection
    k_sinit = "linear_no_bias_sinit.weight"
    haskey(weights, k_sinit) || error("Missing $k_sinit")
    sinit = weights[k_sinit]
    sinit isa AbstractMatrix || error("$k_sinit must be matrix")

    k_rel = "relative_position_encoding.linear_no_bias.weight"
    haskey(weights, k_rel) || error("Missing $k_rel")
    rel = weights[k_rel]
    rel isa AbstractMatrix || error("$k_rel must be matrix")

    k_pair_att = "pairformer_stack.blocks.0.attention_pair_bias.linear_nobias_z.weight"
    haskey(weights, k_pair_att) || error("Missing $k_pair_att")
    pair_att = weights[k_pair_att]
    pair_att isa AbstractMatrix || error("$k_pair_att must be matrix")

    function _infer_blocks(prefix::String)
        max_block = -1
        for key_any in keys(weights)
            key = String(key_any)
            if startswith(key, prefix)
                parts = split(key, '.')
                pos = findfirst(==("blocks"), parts)
                if pos !== nothing && pos < length(parts)
                    idx = tryparse(Int, parts[pos + 1])
                    idx !== nothing && (max_block = max(max_block, idx))
                end
            end
        end
        return max_block + 1
    end

    pairformer_blocks = _infer_blocks("pairformer_stack.blocks.")
    msa_blocks = _infer_blocks("msa_module.blocks.")
    conf_blocks = _infer_blocks("confidence_head.pairformer_stack.blocks.")

    plddt_w = _key(weights, "confidence_head.plddt_weight"; strict = true)
    plddt_w isa AbstractArray || error("confidence_head.plddt_weight must be array")

    k_input_token = "input_embedder.atom_attention_encoder.linear_no_bias_q.weight"
    haskey(weights, k_input_token) || error("Missing $k_input_token")
    w_in_tok = weights[k_input_token]
    w_in_tok isa AbstractMatrix || error("$k_input_token must be matrix")

    k_esm = "input_embedder.linear_esm.weight"
    has_esm = haskey(weights, k_esm)
    esm_embedding_dim = 2560
    if has_esm
        w_esm = weights[k_esm]
        w_esm isa AbstractMatrix || error("$k_esm must be matrix")
        esm_embedding_dim = size(w_esm, 2)
    end

    return (
        c_token_diffusion = dm.c_token,
        c_token_input = size(w_in_tok, 1),
        c_s = size(sinit, 1),
        c_z = size(rel, 1),
        c_s_inputs = size(sinit, 2),
        c_atom = dm.c_atom,
        c_atompair = dm.c_atompair,
        diffusion_blocks = dm.n_blocks,
        diffusion_heads = dm.n_heads,
        diffusion_atom_encoder_blocks = dm.atom_encoder_blocks,
        diffusion_atom_decoder_blocks = dm.atom_decoder_blocks,
        pairformer_blocks = pairformer_blocks,
        msa_blocks = msa_blocks,
        confidence_blocks = conf_blocks,
        pairformer_heads = size(pair_att, 1),
        max_atoms_per_token = size(plddt_w, 1),
        b_plddt = size(plddt_w, 3),
        input_esm_enable = has_esm,
        input_esm_embedding_dim = esm_embedding_dim,
    )
end

function build_protenix_mini_model(
    weights::AbstractDict{<:AbstractString, <:Any};
    rng::AbstractRNG = MersenneTwister(0),
)
    d = infer_protenix_mini_dims(weights)
    return ProtenixMiniModel(
        d.c_token_diffusion,
        d.c_token_input,
        d.c_s,
        d.c_z,
        d.c_s_inputs;
        c_atom = d.c_atom,
        c_atompair = d.c_atompair,
        n_cycle = 4,
        pairformer_blocks = d.pairformer_blocks,
        msa_blocks = max(d.msa_blocks, 1),
        diffusion_transformer_blocks = d.diffusion_blocks,
        diffusion_atom_encoder_blocks = d.diffusion_atom_encoder_blocks,
        diffusion_atom_decoder_blocks = d.diffusion_atom_decoder_blocks,
        confidence_max_atoms_per_token = d.max_atoms_per_token,
        sample_n_step = 5,
        sample_n_sample = 1,
        input_esm_enable = d.input_esm_enable,
        input_esm_embedding_dim = d.input_esm_embedding_dim,
        rng = rng,
    )
end

function load_protenix_mini_model!(
    model::ProtenixMiniModel,
    weights::AbstractDict{<:AbstractString, <:Any};
    strict::Bool = true,
)
    _load_input_feature_embedder!(model.input_embedder, weights, "input_embedder"; strict = strict)

    rel = _key(weights, "relative_position_encoding.linear_no_bias.weight"; strict = strict)
    rel !== nothing && _load_matrix!(model.relative_position_encoding.weight, rel, "relative_position_encoding.linear_no_bias.weight")

    _load_template_embedder!(model.template_embedder, weights, "template_embedder"; strict = false)
    if model.noisy_structure_embedder !== nothing
        _load_noisy_structure_embedder!(model.noisy_structure_embedder, weights, "noisy_structure_embedder"; strict = false)
    end

    _load_msa_module!(model.msa_module, weights, "msa_module"; strict = strict)
    _load_pairformer_stack!(model.pairformer_stack, weights, "pairformer_stack"; strict = strict)
    load_diffusion_module!(model.diffusion_module, weights, "diffusion_module"; strict = strict)

    _load_linear!(model.distogram_head.linear, weights, "distogram_head.linear.weight"; b_key = "distogram_head.linear.bias", strict = strict)
    _load_confidence_head!(model.confidence_head, weights, "confidence_head"; strict = strict)

    _load_linear_nobias!(model.linear_no_bias_sinit, weights, "linear_no_bias_sinit.weight"; strict = strict)
    _load_linear_nobias!(model.linear_no_bias_zinit1, weights, "linear_no_bias_zinit1.weight"; strict = strict)
    _load_linear_nobias!(model.linear_no_bias_zinit2, weights, "linear_no_bias_zinit2.weight"; strict = strict)
    _load_linear_nobias!(model.linear_no_bias_token_bond, weights, "linear_no_bias_token_bond.weight"; strict = strict)
    _load_linear_nobias!(model.linear_no_bias_z_cycle, weights, "linear_no_bias_z_cycle.weight"; strict = strict)
    _load_linear_nobias!(model.linear_no_bias_s, weights, "linear_no_bias_s.weight"; strict = strict)
    _load_layernorm!(model.layernorm_z_cycle, weights, "layernorm_z_cycle"; strict = strict)
    _load_layernorm!(model.layernorm_s, weights, "layernorm_s"; strict = strict)

    return model
end

end

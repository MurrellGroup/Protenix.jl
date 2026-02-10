module StateLoad

import ..Embedders: ConditionTemplateEmbedder, RelativePositionEncoding
import ..DesignConditionEmbedderModule: DesignConditionEmbedder, InputFeatureEmbedderDesign
import ..AtomAttentionModule: AtomAttentionEncoder, AtomAttentionDecoder
import ..Primitives: AdaptiveLayerNorm, LayerNormNoOffset, LinearNoBias, Transition
import ..DiffusionConditioningModule: DiffusionConditioning
import ..TransformerBlocks:
    AttentionPairBias,
    ConditionedTransitionBlock,
    DiffusionTransformer,
    DiffusionTransformerBlock
import ..DiffusionModuleModule: DiffusionModule

export load_condition_template_embedder!,
    load_design_condition_embedder!,
    load_relative_position_encoding!,
    load_diffusion_conditioning!,
    load_diffusion_transformer!,
    load_diffusion_module!,
    infer_model_scaffold_dims,
    infer_design_condition_embedder_dims,
    expected_diffusion_module_keys,
    expected_design_condition_embedder_keys,
    checkpoint_coverage_report

function _key(weights::AbstractDict{<:AbstractString, <:Any}, key::String; strict::Bool = true)
    if haskey(weights, key)
        return weights[key]
    end
    strict && error("Missing checkpoint tensor: $key")
    return nothing
end

function _load_vector!(dst::Vector{Float32}, src, key::String)
    src isa AbstractVector || error("Checkpoint key $key must be a vector, got $(typeof(src))")
    length(dst) == length(src) || error("Shape mismatch for $key: expected $(length(dst)), got $(length(src))")
    dst .= Float32.(src)
    return dst
end

function _load_matrix!(dst::Matrix{Float32}, src, key::String)
    src isa AbstractMatrix || error("Checkpoint key $key must be a matrix, got $(typeof(src))")
    size(dst) == size(src) || error("Shape mismatch for $key: expected $(size(dst)), got $(size(src))")
    dst .= Float32.(src)
    return dst
end

function _load_linear!(
    linear::LinearNoBias,
    weights::AbstractDict{<:AbstractString, <:Any},
    key::String;
    strict::Bool = true,
)
    src = _key(weights, key; strict = strict)
    src === nothing && return linear
    _load_matrix!(linear.weight, src, key)
    return linear
end

function _load_layernorm!(
    ln::LayerNormNoOffset,
    weights::AbstractDict{<:AbstractString, <:Any},
    prefix::String;
    strict::Bool = true,
)
    w = _key(weights, "$prefix.weight"; strict = strict)
    w !== nothing && _load_vector!(ln.weight, w, "$prefix.weight")
    b = _key(weights, "$prefix.bias"; strict = false)
    b !== nothing && _load_vector!(ln.bias, b, "$prefix.bias")
    return ln
end

function _load_adaptive_layernorm!(
    ada::AdaptiveLayerNorm,
    weights::AbstractDict{<:AbstractString, <:Any},
    prefix::String;
    strict::Bool = true,
)
    _load_layernorm!(ada.layernorm_s, weights, "$prefix.layernorm_s"; strict = strict)
    _load_linear!(ada.linear_nobias_s, weights, "$prefix.linear_nobias_s.weight"; strict = strict)
    _load_linear!(ada.linear_s, weights, "$prefix.linear_s.weight"; strict = strict)
    b = _key(weights, "$prefix.linear_s.bias"; strict = false)
    b !== nothing && _load_vector!(ada.bias_s, b, "$prefix.linear_s.bias")
    return ada
end

function _load_transition!(
    tr::Transition,
    weights::AbstractDict{<:AbstractString, <:Any},
    prefix::String;
    strict::Bool = true,
)
    _load_layernorm!(tr.layernorm, weights, "$prefix.layernorm1"; strict = strict)
    _load_linear!(tr.linear_a, weights, "$prefix.linear_no_bias_a.weight"; strict = strict)
    _load_linear!(tr.linear_b, weights, "$prefix.linear_no_bias_b.weight"; strict = strict)
    _load_linear!(tr.linear_out, weights, "$prefix.linear_no_bias.weight"; strict = strict)
    return tr
end

function load_condition_template_embedder!(
    cte::ConditionTemplateEmbedder,
    weights::AbstractDict{<:AbstractString, <:Any},
    prefix::String = "design_condition_embedder.condition_template_embedder";
    strict::Bool = true,
)
    w = _key(weights, "$prefix.embedder.weight"; strict = strict)
    w === nothing && return cte
    _load_matrix!(cte.weight, w, "$prefix.embedder.weight")
    return cte
end

function _load_input_feature_embedder_design!(
    emb::InputFeatureEmbedderDesign,
    weights::AbstractDict{<:AbstractString, <:Any},
    prefix::String;
    strict::Bool = true,
)
    enc = emb.atom_attention_encoder
    _load_linear!(enc.linear_no_bias_ref_pos, weights, "$prefix.atom_attention_encoder.linear_no_bias_ref_pos.weight"; strict = strict)
    _load_linear!(
        enc.linear_no_bias_ref_charge,
        weights,
        "$prefix.atom_attention_encoder.linear_no_bias_ref_charge.weight";
        strict = strict,
    )
    _load_linear!(enc.linear_no_bias_f, weights, "$prefix.atom_attention_encoder.linear_no_bias_f.weight"; strict = strict)
    _load_linear!(enc.linear_no_bias_d, weights, "$prefix.atom_attention_encoder.linear_no_bias_d.weight"; strict = strict)
    _load_linear!(
        enc.linear_no_bias_invd,
        weights,
        "$prefix.atom_attention_encoder.linear_no_bias_invd.weight";
        strict = strict,
    )
    _load_linear!(enc.linear_no_bias_v, weights, "$prefix.atom_attention_encoder.linear_no_bias_v.weight"; strict = strict)
    _load_linear!(enc.linear_no_bias_cl, weights, "$prefix.atom_attention_encoder.linear_no_bias_cl.weight"; strict = strict)
    _load_linear!(enc.linear_no_bias_cm, weights, "$prefix.atom_attention_encoder.linear_no_bias_cm.weight"; strict = strict)
    _load_linear!(enc.small_mlp_1, weights, "$prefix.atom_attention_encoder.small_mlp.1.weight"; strict = strict)
    _load_linear!(enc.small_mlp_2, weights, "$prefix.atom_attention_encoder.small_mlp.3.weight"; strict = strict)
    _load_linear!(enc.small_mlp_3, weights, "$prefix.atom_attention_encoder.small_mlp.5.weight"; strict = strict)
    _load_linear!(enc.linear_no_bias_q, weights, "$prefix.atom_attention_encoder.linear_no_bias_q.weight"; strict = strict)
    load_diffusion_transformer!(
        enc.atom_transformer,
        weights,
        "$prefix.atom_attention_encoder.atom_transformer.diffusion_transformer";
        strict = strict,
    )

    w = _key(weights, "$prefix.input_map.weight"; strict = strict)
    w !== nothing && _load_matrix!(emb.input_map_weight, w, "$prefix.input_map.weight")
    b = _key(weights, "$prefix.input_map.bias"; strict = strict)
    b !== nothing && _load_vector!(emb.input_map_bias, b, "$prefix.input_map.bias")
    return emb
end

function load_design_condition_embedder!(
    dce::DesignConditionEmbedder,
    weights::AbstractDict{<:AbstractString, <:Any},
    prefix::String = "design_condition_embedder";
    strict::Bool = true,
)
    load_condition_template_embedder!(
        dce.condition_template_embedder,
        weights,
        "$prefix.condition_template_embedder";
        strict = strict,
    )
    _load_input_feature_embedder_design!(
        dce.input_embedder,
        weights,
        "$prefix.input_embedder";
        strict = strict,
    )
    return dce
end

function _load_atom_attention_encoder!(
    enc::AtomAttentionEncoder,
    weights::AbstractDict{<:AbstractString, <:Any},
    prefix::String;
    strict::Bool = true,
)
    _load_linear!(enc.linear_no_bias_ref_pos, weights, "$prefix.linear_no_bias_ref_pos.weight"; strict = strict)
    _load_linear!(enc.linear_no_bias_ref_charge, weights, "$prefix.linear_no_bias_ref_charge.weight"; strict = strict)
    _load_linear!(enc.linear_no_bias_f, weights, "$prefix.linear_no_bias_f.weight"; strict = strict)
    _load_linear!(enc.linear_no_bias_d, weights, "$prefix.linear_no_bias_d.weight"; strict = strict)
    _load_linear!(enc.linear_no_bias_invd, weights, "$prefix.linear_no_bias_invd.weight"; strict = strict)
    _load_linear!(enc.linear_no_bias_v, weights, "$prefix.linear_no_bias_v.weight"; strict = strict)
    if enc.layernorm_s !== nothing
        _load_layernorm!(enc.layernorm_s, weights, "$prefix.layernorm_s"; strict = strict)
    end
    if enc.linear_no_bias_s !== nothing
        _load_linear!(enc.linear_no_bias_s, weights, "$prefix.linear_no_bias_s.weight"; strict = strict)
    end
    if enc.layernorm_z !== nothing
        _load_layernorm!(enc.layernorm_z, weights, "$prefix.layernorm_z"; strict = strict)
    end
    if enc.linear_no_bias_z !== nothing
        _load_linear!(enc.linear_no_bias_z, weights, "$prefix.linear_no_bias_z.weight"; strict = strict)
    end
    if enc.linear_no_bias_r !== nothing
        _load_linear!(enc.linear_no_bias_r, weights, "$prefix.linear_no_bias_r.weight"; strict = strict)
    end
    _load_linear!(enc.linear_no_bias_cl, weights, "$prefix.linear_no_bias_cl.weight"; strict = strict)
    _load_linear!(enc.linear_no_bias_cm, weights, "$prefix.linear_no_bias_cm.weight"; strict = strict)
    _load_linear!(enc.small_mlp_1, weights, "$prefix.small_mlp.1.weight"; strict = strict)
    _load_linear!(enc.small_mlp_2, weights, "$prefix.small_mlp.3.weight"; strict = strict)
    _load_linear!(enc.small_mlp_3, weights, "$prefix.small_mlp.5.weight"; strict = strict)
    load_diffusion_transformer!(
        enc.atom_transformer,
        weights,
        "$prefix.atom_transformer.diffusion_transformer";
        strict = strict,
    )
    _load_linear!(enc.linear_no_bias_q, weights, "$prefix.linear_no_bias_q.weight"; strict = strict)
    return enc
end

function _load_atom_attention_decoder!(
    dec::AtomAttentionDecoder,
    weights::AbstractDict{<:AbstractString, <:Any},
    prefix::String;
    strict::Bool = true,
)
    _load_linear!(dec.linear_no_bias_a, weights, "$prefix.linear_no_bias_a.weight"; strict = strict)
    _load_layernorm!(dec.layernorm_q, weights, "$prefix.layernorm_q"; strict = strict)
    _load_linear!(dec.linear_no_bias_out, weights, "$prefix.linear_no_bias_out.weight"; strict = strict)
    load_diffusion_transformer!(
        dec.atom_transformer,
        weights,
        "$prefix.atom_transformer.diffusion_transformer";
        strict = strict,
    )
    return dec
end

function load_relative_position_encoding!(
    relpe::RelativePositionEncoding,
    weights::AbstractDict{<:AbstractString, <:Any},
    prefix::String;
    strict::Bool = true,
)
    _load_linear!(LinearNoBias(relpe.weight), weights, "$prefix.linear_no_bias.weight"; strict = strict)
    return relpe
end

function load_diffusion_conditioning!(
    cond::DiffusionConditioning,
    weights::AbstractDict{<:AbstractString, <:Any},
    prefix::String = "diffusion_module.diffusion_conditioning";
    strict::Bool = true,
)
    load_relative_position_encoding!(cond.relpe, weights, "$prefix.relpe"; strict = strict)
    _load_layernorm!(cond.layernorm_z, weights, "$prefix.layernorm_z"; strict = strict)
    _load_linear!(cond.linear_no_bias_z, weights, "$prefix.linear_no_bias_z.weight"; strict = strict)
    _load_transition!(cond.transition_z1, weights, "$prefix.transition_z1"; strict = strict)
    _load_transition!(cond.transition_z2, weights, "$prefix.transition_z2"; strict = strict)

    _load_layernorm!(cond.layernorm_s, weights, "$prefix.layernorm_s"; strict = strict)
    _load_linear!(cond.linear_no_bias_s, weights, "$prefix.linear_no_bias_s.weight"; strict = strict)

    w = _key(weights, "$prefix.fourier_embedding.w"; strict = strict)
    w !== nothing && _load_vector!(cond.w_noise, w, "$prefix.fourier_embedding.w")
    b = _key(weights, "$prefix.fourier_embedding.b"; strict = strict)
    b !== nothing && _load_vector!(cond.b_noise, b, "$prefix.fourier_embedding.b")

    _load_layernorm!(cond.layernorm_n, weights, "$prefix.layernorm_n"; strict = strict)
    _load_linear!(cond.linear_no_bias_n, weights, "$prefix.linear_no_bias_n.weight"; strict = strict)
    _load_transition!(cond.transition_s1, weights, "$prefix.transition_s1"; strict = strict)
    _load_transition!(cond.transition_s2, weights, "$prefix.transition_s2"; strict = strict)
    return cond
end

function _load_attention_pair_bias!(
    blk::AttentionPairBias,
    weights::AbstractDict{<:AbstractString, <:Any},
    prefix::String;
    strict::Bool = true,
)
    _load_adaptive_layernorm!(blk.adaln_a, weights, "$prefix.layernorm_a"; strict = strict)
    if blk.adaln_kv !== nothing
        _load_adaptive_layernorm!(blk.adaln_kv, weights, "$prefix.layernorm_kv"; strict = strict)
    end
    _load_layernorm!(blk.layernorm_z, weights, "$prefix.layernorm_z"; strict = strict)
    _load_linear!(blk.linear_bias_z, weights, "$prefix.linear_nobias_z.weight"; strict = strict)

    _load_linear!(blk.linear_q, weights, "$prefix.attention.linear_q.weight"; strict = strict)
    q_bias = _key(weights, "$prefix.attention.linear_q.bias"; strict = false)
    q_bias !== nothing && _load_vector!(blk.bias_q, q_bias, "$prefix.attention.linear_q.bias")
    _load_linear!(blk.linear_k, weights, "$prefix.attention.linear_k.weight"; strict = strict)
    _load_linear!(blk.linear_v, weights, "$prefix.attention.linear_v.weight"; strict = strict)
    _load_linear!(blk.linear_o, weights, "$prefix.attention.linear_o.weight"; strict = strict)
    _load_linear!(blk.linear_g, weights, "$prefix.attention.linear_g.weight"; strict = strict)

    _load_linear!(blk.linear_a_last, weights, "$prefix.linear_a_last.weight"; strict = strict)
    b_last = _key(weights, "$prefix.linear_a_last.bias"; strict = false)
    b_last !== nothing && _load_vector!(blk.bias_a_last, b_last, "$prefix.linear_a_last.bias")
    return blk
end

function _load_conditioned_transition_block!(
    blk::ConditionedTransitionBlock,
    weights::AbstractDict{<:AbstractString, <:Any},
    prefix::String;
    strict::Bool = true,
)
    _load_adaptive_layernorm!(blk.adaln, weights, "$prefix.adaln"; strict = strict)
    _load_linear!(blk.linear_a1, weights, "$prefix.linear_nobias_a1.weight"; strict = strict)
    _load_linear!(blk.linear_a2, weights, "$prefix.linear_nobias_a2.weight"; strict = strict)
    _load_linear!(blk.linear_b, weights, "$prefix.linear_nobias_b.weight"; strict = strict)
    _load_linear!(blk.linear_s, weights, "$prefix.linear_s.weight"; strict = strict)
    b = _key(weights, "$prefix.linear_s.bias"; strict = false)
    b !== nothing && _load_vector!(blk.bias_s, b, "$prefix.linear_s.bias")
    return blk
end

function _load_diffusion_transformer_block!(
    blk::DiffusionTransformerBlock,
    weights::AbstractDict{<:AbstractString, <:Any},
    prefix::String;
    strict::Bool = true,
)
    _load_attention_pair_bias!(blk.attention_pair_bias, weights, "$prefix.attention_pair_bias"; strict = strict)
    _load_conditioned_transition_block!(
        blk.conditioned_transition_block,
        weights,
        "$prefix.conditioned_transition_block";
        strict = strict,
    )
    return blk
end

function load_diffusion_transformer!(
    tr::DiffusionTransformer,
    weights::AbstractDict{<:AbstractString, <:Any},
    prefix::String = "diffusion_module.diffusion_transformer";
    strict::Bool = true,
)
    for (i, blk) in enumerate(tr.blocks)
        _load_diffusion_transformer_block!(blk, weights, "$prefix.blocks.$(i - 1)"; strict = strict)
    end
    return tr
end

function load_diffusion_module!(
    dm::DiffusionModule,
    weights::AbstractDict{<:AbstractString, <:Any},
    prefix::String = "diffusion_module";
    strict::Bool = true,
)
    load_diffusion_conditioning!(dm.diffusion_conditioning, weights, "$prefix.diffusion_conditioning"; strict = strict)
    _load_atom_attention_encoder!(dm.atom_attention_encoder, weights, "$prefix.atom_attention_encoder"; strict = strict)
    _load_layernorm!(dm.layernorm_s, weights, "$prefix.layernorm_s"; strict = strict)
    _load_linear!(dm.linear_no_bias_s, weights, "$prefix.linear_no_bias_s.weight"; strict = strict)
    load_diffusion_transformer!(dm.diffusion_transformer, weights, "$prefix.diffusion_transformer"; strict = strict)
    _load_layernorm!(dm.layernorm_a, weights, "$prefix.layernorm_a"; strict = strict)
    _load_atom_attention_decoder!(dm.atom_attention_decoder, weights, "$prefix.atom_attention_decoder"; strict = strict)
    return dm
end

function _append_transition_keys!(keys::Vector{String}, prefix::String)
    push!(keys, "$prefix.layernorm1.weight")
    push!(keys, "$prefix.layernorm1.bias")
    push!(keys, "$prefix.linear_no_bias_a.weight")
    push!(keys, "$prefix.linear_no_bias_b.weight")
    push!(keys, "$prefix.linear_no_bias.weight")
    return keys
end

function _append_adaln_keys!(keys::Vector{String}, prefix::String)
    push!(keys, "$prefix.layernorm_s.weight")
    push!(keys, "$prefix.linear_nobias_s.weight")
    push!(keys, "$prefix.linear_s.weight")
    push!(keys, "$prefix.linear_s.bias")
    return keys
end

function _append_attention_pair_bias_keys!(keys::Vector{String}, prefix::String; has_layernorm_kv::Bool)
    _append_adaln_keys!(keys, "$prefix.layernorm_a")
    if has_layernorm_kv
        _append_adaln_keys!(keys, "$prefix.layernorm_kv")
    end
    push!(keys, "$prefix.layernorm_z.weight")
    push!(keys, "$prefix.linear_nobias_z.weight")
    push!(keys, "$prefix.attention.linear_q.weight")
    push!(keys, "$prefix.attention.linear_q.bias")
    push!(keys, "$prefix.attention.linear_k.weight")
    push!(keys, "$prefix.attention.linear_v.weight")
    push!(keys, "$prefix.attention.linear_o.weight")
    push!(keys, "$prefix.attention.linear_g.weight")
    push!(keys, "$prefix.linear_a_last.weight")
    push!(keys, "$prefix.linear_a_last.bias")
    return keys
end

function _append_conditioned_transition_block_keys!(keys::Vector{String}, prefix::String)
    _append_adaln_keys!(keys, "$prefix.adaln")
    push!(keys, "$prefix.linear_nobias_a1.weight")
    push!(keys, "$prefix.linear_nobias_a2.weight")
    push!(keys, "$prefix.linear_nobias_b.weight")
    push!(keys, "$prefix.linear_s.weight")
    push!(keys, "$prefix.linear_s.bias")
    return keys
end

function _append_diffusion_transformer_keys!(
    keys::Vector{String},
    tr::DiffusionTransformer,
    prefix::String,
)
    for (i, blk) in enumerate(tr.blocks)
        b = "$prefix.blocks.$(i - 1)"
        _append_attention_pair_bias_keys!(
            keys,
            "$b.attention_pair_bias";
            has_layernorm_kv = blk.attention_pair_bias.adaln_kv !== nothing,
        )
        _append_conditioned_transition_block_keys!(keys, "$b.conditioned_transition_block")
    end
    return keys
end

function _expected_atom_attention_encoder_keys(enc::AtomAttentionEncoder, prefix::String)
    keys = String[]
    push!(keys, "$prefix.linear_no_bias_ref_pos.weight")
    push!(keys, "$prefix.linear_no_bias_ref_charge.weight")
    push!(keys, "$prefix.linear_no_bias_f.weight")
    push!(keys, "$prefix.linear_no_bias_d.weight")
    push!(keys, "$prefix.linear_no_bias_invd.weight")
    push!(keys, "$prefix.linear_no_bias_v.weight")
    if enc.has_coords
        push!(keys, "$prefix.layernorm_s.weight")
        push!(keys, "$prefix.linear_no_bias_s.weight")
        push!(keys, "$prefix.layernorm_z.weight")
        push!(keys, "$prefix.linear_no_bias_z.weight")
        push!(keys, "$prefix.linear_no_bias_r.weight")
    end
    push!(keys, "$prefix.linear_no_bias_cl.weight")
    push!(keys, "$prefix.linear_no_bias_cm.weight")
    push!(keys, "$prefix.small_mlp.1.weight")
    push!(keys, "$prefix.small_mlp.3.weight")
    push!(keys, "$prefix.small_mlp.5.weight")
    _append_diffusion_transformer_keys!(keys, enc.atom_transformer, "$prefix.atom_transformer.diffusion_transformer")
    push!(keys, "$prefix.linear_no_bias_q.weight")
    return keys
end

function _expected_atom_attention_decoder_keys(dec::AtomAttentionDecoder, prefix::String)
    keys = String[]
    push!(keys, "$prefix.linear_no_bias_a.weight")
    push!(keys, "$prefix.layernorm_q.weight")
    push!(keys, "$prefix.linear_no_bias_out.weight")
    _append_diffusion_transformer_keys!(keys, dec.atom_transformer, "$prefix.atom_transformer.diffusion_transformer")
    return keys
end

function expected_design_condition_embedder_keys(
    dce::DesignConditionEmbedder;
    prefix::String = "design_condition_embedder",
)
    keys = String[]
    push!(keys, "$prefix.condition_template_embedder.embedder.weight")
    append!(keys, _expected_atom_attention_encoder_keys(dce.input_embedder.atom_attention_encoder, "$prefix.input_embedder.atom_attention_encoder"))
    push!(keys, "$prefix.input_embedder.input_map.weight")
    push!(keys, "$prefix.input_embedder.input_map.bias")
    return sort!(unique!(keys))
end

function expected_diffusion_module_keys(
    dm::DiffusionModule;
    prefix::String = "diffusion_module",
)
    keys = String[]
    push!(keys, "$prefix.diffusion_conditioning.relpe.linear_no_bias.weight")
    push!(keys, "$prefix.diffusion_conditioning.layernorm_z.weight")
    push!(keys, "$prefix.diffusion_conditioning.linear_no_bias_z.weight")
    _append_transition_keys!(keys, "$prefix.diffusion_conditioning.transition_z1")
    _append_transition_keys!(keys, "$prefix.diffusion_conditioning.transition_z2")
    push!(keys, "$prefix.diffusion_conditioning.layernorm_s.weight")
    push!(keys, "$prefix.diffusion_conditioning.linear_no_bias_s.weight")
    push!(keys, "$prefix.diffusion_conditioning.fourier_embedding.w")
    push!(keys, "$prefix.diffusion_conditioning.fourier_embedding.b")
    push!(keys, "$prefix.diffusion_conditioning.layernorm_n.weight")
    push!(keys, "$prefix.diffusion_conditioning.linear_no_bias_n.weight")
    _append_transition_keys!(keys, "$prefix.diffusion_conditioning.transition_s1")
    _append_transition_keys!(keys, "$prefix.diffusion_conditioning.transition_s2")

    append!(keys, _expected_atom_attention_encoder_keys(dm.atom_attention_encoder, "$prefix.atom_attention_encoder"))
    push!(keys, "$prefix.layernorm_s.weight")
    push!(keys, "$prefix.linear_no_bias_s.weight")
    _append_diffusion_transformer_keys!(keys, dm.diffusion_transformer, "$prefix.diffusion_transformer")
    push!(keys, "$prefix.layernorm_a.weight")
    append!(keys, _expected_atom_attention_decoder_keys(dm.atom_attention_decoder, "$prefix.atom_attention_decoder"))

    return sort!(unique!(keys))
end

function checkpoint_coverage_report(
    dm::Union{DiffusionModule, Nothing},
    dce::Union{DesignConditionEmbedder, Nothing},
    weights::AbstractDict{<:AbstractString, <:Any},
)
    expected = String[]
    prefixes = String[]
    if dm !== nothing
        append!(expected, expected_diffusion_module_keys(dm))
        push!(prefixes, "diffusion_module.")
    end
    if dce !== nothing
        append!(expected, expected_design_condition_embedder_keys(dce))
        push!(prefixes, "design_condition_embedder.")
    end
    expected_set = Set(expected)

    considered_present = String[]
    for key_any in keys(weights)
        key = String(key_any)
        if any(startswith(key, p) for p in prefixes)
            push!(considered_present, key)
        end
    end
    present_set = Set(considered_present)
    missing = sort!(collect(setdiff(expected_set, present_set)))
    unused = sort!(collect(setdiff(present_set, expected_set)))
    return (
        missing = missing,
        unused = unused,
        n_expected = length(expected_set),
        n_present = length(present_set),
    )
end

function infer_model_scaffold_dims(weights::AbstractDict{<:AbstractString, <:Any})
    k_rel = "diffusion_module.diffusion_conditioning.relpe.linear_no_bias.weight"
    k_mod_s = "diffusion_module.linear_no_bias_s.weight"
    k_cond_s = "diffusion_module.diffusion_conditioning.linear_no_bias_s.weight"
    k_heads = "diffusion_module.diffusion_transformer.blocks.0.attention_pair_bias.linear_nobias_z.weight"
    k_c_atom = "diffusion_module.atom_attention_encoder.linear_no_bias_ref_pos.weight"
    k_c_atompair = "diffusion_module.atom_attention_encoder.linear_no_bias_d.weight"
    k_enc_heads = "diffusion_module.atom_attention_encoder.atom_transformer.diffusion_transformer.blocks.0.attention_pair_bias.linear_nobias_z.weight"
    k_dec_heads = "diffusion_module.atom_attention_decoder.atom_transformer.diffusion_transformer.blocks.0.attention_pair_bias.linear_nobias_z.weight"

    haskey(weights, k_rel) || error("Missing key for c_z inference: $k_rel")
    haskey(weights, k_mod_s) || error("Missing key for c_token/c_s inference: $k_mod_s")
    haskey(weights, k_cond_s) || error("Missing key for c_s_inputs inference: $k_cond_s")
    haskey(weights, k_heads) || error("Missing key for n_heads inference: $k_heads")
    haskey(weights, k_c_atom) || error("Missing key for c_atom inference: $k_c_atom")
    haskey(weights, k_c_atompair) || error("Missing key for c_atompair inference: $k_c_atompair")
    haskey(weights, k_enc_heads) || error("Missing key for atom encoder heads inference: $k_enc_heads")
    haskey(weights, k_dec_heads) || error("Missing key for atom decoder heads inference: $k_dec_heads")

    rel = weights[k_rel]
    mod_s = weights[k_mod_s]
    cond_s = weights[k_cond_s]
    heads = weights[k_heads]
    atom_w = weights[k_c_atom]
    atompair_w = weights[k_c_atompair]
    enc_heads_w = weights[k_enc_heads]
    dec_heads_w = weights[k_dec_heads]
    rel isa AbstractMatrix || error("$k_rel must be matrix.")
    mod_s isa AbstractMatrix || error("$k_mod_s must be matrix.")
    cond_s isa AbstractMatrix || error("$k_cond_s must be matrix.")
    heads isa AbstractMatrix || error("$k_heads must be matrix.")
    atom_w isa AbstractMatrix || error("$k_c_atom must be matrix.")
    atompair_w isa AbstractMatrix || error("$k_c_atompair must be matrix.")
    enc_heads_w isa AbstractMatrix || error("$k_enc_heads must be matrix.")
    dec_heads_w isa AbstractMatrix || error("$k_dec_heads must be matrix.")

    c_z = size(rel, 1)
    c_token = size(mod_s, 1)
    c_s = size(mod_s, 2)
    cond_in = size(cond_s, 2)
    c_s_inputs = cond_in - c_s
    c_s_inputs >= 0 || error("Inferred negative c_s_inputs from $k_cond_s")
    n_heads = size(heads, 1)
    c_atom = size(atom_w, 1)
    c_atompair = size(atompair_w, 1)
    atom_encoder_heads = size(enc_heads_w, 1)
    atom_decoder_heads = size(dec_heads_w, 1)

    function _infer_block_count(block_prefix::String)
        max_block = -1
        for key_any in keys(weights)
            key = String(key_any)
            if startswith(key, block_prefix)
                parts = split(key, '.')
                pos = findfirst(==("blocks"), parts)
                if pos !== nothing && pos < length(parts)
                    idx = tryparse(Int, parts[pos + 1])
                    if idx !== nothing
                        max_block = max(max_block, idx)
                    end
                end
            end
        end
        n = max_block + 1
        n > 0 || error("Could not infer n_blocks from prefix: $block_prefix")
        return n
    end

    n_blocks = _infer_block_count("diffusion_module.diffusion_transformer.blocks.")
    atom_encoder_blocks = _infer_block_count("diffusion_module.atom_attention_encoder.atom_transformer.diffusion_transformer.blocks.")
    atom_decoder_blocks = _infer_block_count("diffusion_module.atom_attention_decoder.atom_transformer.diffusion_transformer.blocks.")

    return (
        c_token = c_token,
        c_s = c_s,
        c_z = c_z,
        c_s_inputs = c_s_inputs,
        n_blocks = n_blocks,
        n_heads = n_heads,
        c_atom = c_atom,
        c_atompair = c_atompair,
        atom_encoder_blocks = atom_encoder_blocks,
        atom_encoder_heads = atom_encoder_heads,
        atom_decoder_blocks = atom_decoder_blocks,
        atom_decoder_heads = atom_decoder_heads,
    )
end

function infer_design_condition_embedder_dims(weights::AbstractDict{<:AbstractString, <:Any})
    k_input_map = "design_condition_embedder.input_embedder.input_map.weight"
    k_templ = "design_condition_embedder.condition_template_embedder.embedder.weight"
    k_heads = "design_condition_embedder.input_embedder.atom_attention_encoder.atom_transformer.diffusion_transformer.blocks.0.attention_pair_bias.linear_nobias_z.weight"

    haskey(weights, k_input_map) || error("Missing key for design c_token/c_s_inputs inference: $k_input_map")
    haskey(weights, k_templ) || error("Missing key for design c_z inference: $k_templ")
    haskey(weights, k_heads) || error("Missing key for design n_heads inference: $k_heads")

    input_map = weights[k_input_map]
    templ = weights[k_templ]
    heads = weights[k_heads]
    input_map isa AbstractMatrix || error("$k_input_map must be matrix.")
    templ isa AbstractMatrix || error("$k_templ must be matrix.")
    heads isa AbstractMatrix || error("$k_heads must be matrix.")

    c_s_inputs = size(input_map, 1)
    c_token = size(input_map, 2) - 46
    c_token > 0 || error("Could not infer positive design c_token from $k_input_map")
    c_z = size(templ, 2)
    n_heads = size(heads, 1)

    max_block = -1
    for key_any in keys(weights)
        key = String(key_any)
        if startswith(
            key,
            "design_condition_embedder.input_embedder.atom_attention_encoder.atom_transformer.diffusion_transformer.blocks.",
        )
            parts = split(key, '.')
            if length(parts) >= 7
                idx = tryparse(Int, parts[7])
                if idx !== nothing
                    max_block = max(max_block, idx)
                end
            end
        end
    end
    n_blocks = max_block + 1
    n_blocks > 0 || error("Could not infer design n_blocks from atom_transformer block keys.")

    return (
        c_token = c_token,
        c_s_inputs = c_s_inputs,
        c_z = c_z,
        n_blocks = n_blocks,
        n_heads = n_heads,
    )
end

end

module LayerRegressionReference

using Random
using PXDesign

function _deepcopy_feature_dict(x)
    if x isa AbstractDict
        out = Dict{String, Any}()
        for (k, v) in x
            out[String(k)] = _deepcopy_feature_dict(v)
        end
        return out
    elseif x isa AbstractArray
        return deepcopy(x)
    else
        return deepcopy(x)
    end
end

function _as_array_f32(x)
    return Float32.(Array(x))
end

function _build_feature_bundles()
    atoms = PXDesign.ProtenixMini.build_sequence_atoms("ACDEFG"; chain_id = "A0")
    bundle = PXDesign.Data.build_feature_bundle_from_atoms(
        atoms;
        task_name = "layer_regression",
        rng = MersenneTwister(20260211),
    )
    feat_design = _deepcopy_feature_dict(bundle["input_feature_dict"])
    feat_mini = _deepcopy_feature_dict(bundle["input_feature_dict"])
    PXDesign.ProtenixAPI._normalize_protenix_feature_dict!(feat_mini)
    return (atoms = atoms, feat_design = feat_design, feat_mini = feat_mini)
end

function compute_layer_regression_outputs()
    # Seed the global RNG so that Onion sub-layers (AttentionPairBias, Transition,
    # BGLinear) which use Random.default_rng() for weight initialization produce
    # deterministic results regardless of prior RNG state in the process.
    Random.seed!(42)

    base = _build_feature_bundles()
    atoms = base.atoms
    feat_design = base.feat_design
    feat_mini = base.feat_mini

    out = Dict{String, Any}()

    n_tok = length(feat_design["token_index"])
    n_atom = length(atoms)
    relpos_input = PXDesign.Model.as_relpos_input(feat_design)
    atom_input = PXDesign.Model.as_atom_attention_input(feat_design)

    # PXDesign model layers
    cte = PXDesign.Model.ConditionTemplateEmbedder(65, 16; rng = MersenneTwister(1))
    out["pxdesign.condition_template_embedding"] = PXDesign.Model.condition_template_embedding(
        cte,
        Int.(feat_design["conditional_templ"]),
        Int.(feat_design["conditional_templ_mask"]),
    )

    relpe = PXDesign.Model.RelativePositionEncoding(4, 2, 8; rng = MersenneTwister(2))
    out["pxdesign.relative_position_encoding"] = relpe(relpos_input)

    ife_design = PXDesign.Model.InputFeatureEmbedderDesign(
        24;
        c_s_inputs = 32,
        c_atom = 16,
        c_atompair = 8,
        n_blocks = 1,
        n_heads = 2,
        rng = MersenneTwister(3),
    )
    s_inputs_design = ife_design(feat_design)
    out["pxdesign.input_feature_embedder_design"] = s_inputs_design

    dce = PXDesign.Model.DesignConditionEmbedder(
        24;
        c_s_inputs = 32,
        c_z = 8,
        c_atom = 16,
        c_atompair = 8,
        n_blocks = 1,
        n_heads = 2,
        rng = MersenneTwister(4),
    )
    s_inputs_dce, z_dce = dce(feat_design)
    out["pxdesign.design_condition_embedder.s_inputs"] = s_inputs_dce
    out["pxdesign.design_condition_embedder.z"] = z_dce

    cond = PXDesign.Model.DiffusionConditioning(
        16.0;
        c_z = 8,
        c_s = 16,
        c_s_inputs = 32,
        c_noise_embedding = 16,
        r_max = 4,
        s_max = 2,
        rng = MersenneTwister(5),
    )
    s_trunk = reshape(Float32.(collect(1:(n_tok * 16))), 16, n_tok) ./ 100f0
    z_trunk = reshape(Float32.(collect(1:(n_tok * n_tok * 8))), 8, n_tok, n_tok) ./ 100f0
    pair_cache = PXDesign.Model.prepare_pair_cache(cond, relpos_input, z_trunk)
    single_s, pair_z = cond(
        Float32[16f0, 16f0],
        relpos_input,
        s_inputs_design,
        s_trunk,
        z_trunk;
        pair_z = pair_cache,
        use_conditioning = true,
    )
    out["pxdesign.diffusion_conditioning.single_s"] = single_s
    out["pxdesign.diffusion_conditioning.pair_z"] = pair_z

    a_test = reshape(Float32.(collect(1:(n_tok * 12))), 12, n_tok) ./ 50f0
    s_test = reshape(Float32.(collect(1:(n_tok * 16))), 16, n_tok) ./ 60f0
    z_test = reshape(Float32.(collect(1:(n_tok * n_tok * 8))), 8, n_tok, n_tok) ./ 70f0
    ctb = PXDesign.Model.ConditionedTransitionBlock(12, 16; n = 2, rng = MersenneTwister(6))
    out["pxdesign.conditioned_transition_block"] = ctb(a_test, s_test)
    mask_test = ones(Float32, n_tok, 1)

    apb = PXDesign.Model.AttentionPairBias(12, 16, 8; n_heads = 4, rng = MersenneTwister(7))
    out["pxdesign.attention_pair_bias"] = apb(a_test, s_test, z_test)

    dtb = PXDesign.Model.DiffusionTransformerBlock(12, 16, 8; n_heads = 4, rng = MersenneTwister(8))
    out["pxdesign.diffusion_transformer_block"] = dtb(a_test, s_test, z_test, mask_test)

    dt = PXDesign.Model.DiffusionTransformer(12, 16, 8; n_blocks = 2, n_heads = 4, rng = MersenneTwister(9))
    out["pxdesign.diffusion_transformer"] = dt(a_test, s_test, z_test)

    aae_no_coords = PXDesign.Model.AtomAttentionEncoder(
        24;
        has_coords = false,
        c_atom = 16,
        c_atompair = 8,
        n_blocks = 1,
        n_heads = 2,
        n_queries = 8,
        n_keys = 16,
        rng = MersenneTwister(10),
    )
    a_tok0, q_skip0, c_skip0, p_skip0 = aae_no_coords(atom_input)
    out["pxdesign.atom_attention_encoder_no_coords.a"] = a_tok0
    out["pxdesign.atom_attention_encoder_no_coords.q_skip"] = q_skip0
    out["pxdesign.atom_attention_encoder_no_coords.c_skip"] = c_skip0
    out["pxdesign.atom_attention_encoder_no_coords.p_skip"] = p_skip0

    aae_coords = PXDesign.Model.AtomAttentionEncoder(
        24;
        has_coords = true,
        c_atom = 16,
        c_atompair = 8,
        c_s = 16,
        c_z = 8,
        n_blocks = 1,
        n_heads = 2,
        n_queries = 8,
        n_keys = 16,
        rng = MersenneTwister(11),
    )
    r_l = zeros(Float32, 3, n_atom, 2)
    s_batch = reshape(Float32.(collect(1:(2 * n_tok * 16))), 16, n_tok, 2) ./ 80f0
    z_batch = reshape(Float32.(collect(1:(2 * n_tok * n_tok * 8))), 8, n_tok, n_tok, 2) ./ 90f0
    a_tok1, q_skip1, c_skip1, p_skip1 = aae_coords(atom_input; r_l = r_l, s = s_batch, z = z_batch)
    out["pxdesign.atom_attention_encoder_with_coords.a"] = a_tok1
    out["pxdesign.atom_attention_encoder_with_coords.q_skip"] = q_skip1
    out["pxdesign.atom_attention_encoder_with_coords.c_skip"] = c_skip1
    out["pxdesign.atom_attention_encoder_with_coords.p_skip"] = p_skip1

    aad = PXDesign.Model.AtomAttentionDecoder(
        24;
        c_atom = 16,
        c_atompair = 8,
        n_blocks = 1,
        n_heads = 2,
        n_queries = 8,
        n_keys = 16,
        rng = MersenneTwister(12),
    )
    out["pxdesign.atom_attention_decoder"] = aad(atom_input, a_tok1, q_skip1, c_skip1, p_skip1)

    dm = PXDesign.Model.DiffusionModule(
        24,
        16,
        8,
        32;
        c_atom = 16,
        c_atompair = 8,
        atom_encoder_blocks = 1,
        atom_encoder_heads = 2,
        n_blocks = 1,
        n_heads = 2,
        atom_decoder_blocks = 1,
        atom_decoder_heads = 2,
        rng = MersenneTwister(13),
    )
    x_noisy = zeros(Float32, 3, n_atom, 2)
    out["pxdesign.diffusion_module"] = dm(
        x_noisy,
        Float32[16f0, 16f0];
        relpos_input = relpos_input,
        s_inputs = s_inputs_design,
        s_trunk = s_trunk,
        z_trunk = z_trunk,
        atom_to_token_idx = Int.(feat_design["atom_to_token_idx"]),
        input_feature_dict = atom_input,
    )

    # Protenix-mini/v0.5 layers
    ife_mini = PXDesign.ProtenixMini.InputFeatureEmbedder(16, 8, 24; rng = MersenneTwister(21))
    s_inputs_mini = ife_mini(feat_mini)
    out["protenix_mini.input_feature_embedder"] = s_inputs_mini

    relpe_mini = PXDesign.ProtenixMini.RelativePositionEncoding(4, 2, 8; rng = MersenneTwister(22))
    out["protenix_mini.relative_position_encoding"] = relpe_mini(feat_mini)

    a_pair = reshape(Float32.(collect(1:(n_tok * 12))), 12, n_tok) ./ 100f0
    z_pair = reshape(Float32.(collect(1:(n_tok * n_tok * 8))), 8, n_tok, n_tok) ./ 120f0
    pair_att = PXDesign.ProtenixMini.PairAttentionNoS(12, 8; n_heads = 4, rng = MersenneTwister(23))
    out["protenix_mini.pair_attention_no_s"] = pair_att(a_pair, z_pair)

    tri_mul = PXDesign.ProtenixMini.TriangleMultiplication(8, 4; outgoing = true, rng = MersenneTwister(24))
    out["protenix_mini.triangle_multiplication"] = tri_mul(z_pair)

    tri_att = PXDesign.ProtenixMini.TriangleAttention(8, 8, 2; starting = true, rng = MersenneTwister(25))
    out["protenix_mini.triangle_attention"] = tri_att(z_pair)

    msa_small = reshape(Float32.(collect(1:(2 * n_tok * 6))), 6, 2, n_tok) ./ 130f0
    opm = PXDesign.ProtenixMini.OuterProductMean(6, 8, 4; rng = MersenneTwister(26))
    out["protenix_mini.outer_product_mean"] = opm(msa_small)

    pfb = PXDesign.ProtenixMini.PairformerBlock(
        8,
        12;
        n_heads = 4,
        c_hidden_mul = 4,
        c_hidden_pair_att = 2,
        no_heads_pair = 2,
        rng = MersenneTwister(27),
    )
    s_pfb, z_pfb = pfb(a_pair, z_pair)
    out["protenix_mini.pairformer_block.s"] = s_pfb
    out["protenix_mini.pairformer_block.z"] = z_pfb

    pfs = PXDesign.ProtenixMini.PairformerStack(8, 12; n_blocks = 1, n_heads = 4, rng = MersenneTwister(28))
    s_pfs, z_pfs = pfs(a_pair, z_pair)
    out["protenix_mini.pairformer_stack.s"] = s_pfs
    out["protenix_mini.pairformer_stack.z"] = z_pfs

    msa_module = PXDesign.ProtenixMini.MSAModule(
        8,
        89;
        n_blocks = 1,
        c_m = 8,
        sample_cutoff = 4,
        sample_lower_bound = 1,
        rng = MersenneTwister(29),
    )
    z_msa = reshape(Float32.(collect(1:(n_tok * n_tok * 8))), 8, n_tok, n_tok) ./ 140f0
    s_msa = reshape(Float32.(collect(1:(n_tok * 89))), 89, n_tok) ./ 150f0
    out["protenix_mini.msa_module"] = msa_module(feat_mini, z_msa, s_msa; rng = MersenneTwister(291))

    dist_head = PXDesign.ProtenixMini.DistogramHead(8; no_bins = 16, rng = MersenneTwister(30))
    out["protenix_mini.distogram_head"] = dist_head(z_pair)

    conf_head = PXDesign.ProtenixMini.ConfidenceHead(
        16,
        8,
        89;
        n_blocks = 1,
        b_pae = 8,
        b_pde = 8,
        b_plddt = 10,
        max_atoms_per_token = 20,
        rng = MersenneTwister(31),
    )
    conf_s_inputs = reshape(Float32.(collect(1:(n_tok * 89))), 89, n_tok) ./ 160f0
    conf_s_trunk = reshape(Float32.(collect(1:(n_tok * 16))), 16, n_tok) ./ 170f0
    conf_z = reshape(Float32.(collect(1:(n_tok * n_tok * 8))), 8, n_tok, n_tok) ./ 180f0
    plddt, pae, pde, resolved = conf_head(
        input_feature_dict = feat_mini,
        s_inputs = conf_s_inputs,
        s_trunk = conf_s_trunk,
        z_trunk = conf_z,
        pair_mask = nothing,
        x_pred_coords = zeros(Float32, 3, n_atom, 1),
    )
    out["protenix_mini.confidence_head.plddt"] = plddt
    out["protenix_mini.confidence_head.pae"] = pae
    out["protenix_mini.confidence_head.pde"] = pde
    out["protenix_mini.confidence_head.resolved"] = resolved

    pm = PXDesign.ProtenixMini.ProtenixMiniModel(
        32,
        32,
        16,
        8,
        97;
        c_atom = 16,
        c_atompair = 8,
        n_cycle = 1,
        pairformer_blocks = 1,
        msa_blocks = 1,
        diffusion_transformer_blocks = 1,
        diffusion_atom_encoder_blocks = 1,
        diffusion_atom_decoder_blocks = 1,
        confidence_max_atoms_per_token = 20,
        sample_gamma0 = 0.0,
        sample_gamma_min = 1.0,
        sample_noise_scale_lambda = 1.0,
        sample_step_scale_eta = 0.0,
        sample_n_step = 1,
        sample_n_sample = 1,
        rng = MersenneTwister(32),
    )
    trunk = PXDesign.ProtenixMini.get_pairformer_output(pm, feat_mini; n_cycle = 1, rng = MersenneTwister(321))
    out["protenix_mini.model_trunk.s_inputs"] = trunk.s_inputs
    out["protenix_mini.model_trunk.s"] = trunk.s
    out["protenix_mini.model_trunk.z"] = trunk.z

    x_noisy_pm = zeros(Float32, 3, n_atom, 1)
    # Trunk outputs are already features-first
    s_inputs_ff = Float32.(trunk.s_inputs)  # (c_s_inputs, N) features-first
    s_trunk_ff = Float32.(trunk.s)          # (c_s, N) features-first
    z_trunk_ff = Float32.(trunk.z)          # (c_z, N, N) features-first
    x_denoised_pm = pm.diffusion_module(
        x_noisy_pm,
        Float32[16f0];
        relpos_input = PXDesign.Model.as_relpos_input(feat_mini),
        s_inputs = s_inputs_ff,
        s_trunk = s_trunk_ff,
        z_trunk = z_trunk_ff,
        atom_to_token_idx = Int.(feat_mini["atom_to_token_idx"]),
        input_feature_dict = PXDesign.Model.as_atom_attention_input(feat_mini),
    )
    out["protenix_mini.model_diffusion_module"] = x_denoised_pm
    out["protenix_v0_5.model_diffusion_module"] = x_denoised_pm

    # Ensure plain arrays for stable serialization
    flat_out = Dict{String, Any}()
    for (k, v) in out
        if v isa AbstractArray
            flat_out[k] = _as_array_f32(v)
        else
            flat_out[k] = v
        end
    end
    return flat_out
end

end

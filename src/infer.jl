module Infer

using Dates
using Random
using SHA
using TOML

import ..Cache: ensure_inference_cache!
import ..Data: build_basic_feature_bundle
import ..Inputs: load_input_tasks, process_input_file, snapshot_input
import ..Model:
    InferenceNoiseScheduler,
    sample_diffusion,
    DesignConditionEmbedder,
    DiffusionModule,
    as_atom_attention_input,
    as_relpos_input,
    checkpoint_coverage_report,
    infer_design_condition_embedder_dims,
    infer_model_scaffold_dims,
    load_design_condition_embedder!,
    load_raw_weights,
    load_safetensors_weights,
    load_diffusion_module!
import ..Output: dump_prediction_bundle
import ..Schema: parse_tasks

export run_infer, derive_seed

function derive_seed(base_seed::Integer, rank::Integer = 0; digits::Int = 6)
    mod_base = 10^digits
    payload = codeunits("pxdesign|$(base_seed)|$(rank)")
    digest = sha256(payload)
    acc = UInt64(0)
    for i in 1:8
        acc = (acc << 8) | UInt64(digest[i])
    end
    return Int(acc % UInt64(mod_base))
end

function _setup_runtime_env!(cfg::Dict{String, Any}; io::IO = stdout)
    if get(cfg, "use_fast_ln", false)
        ENV["LAYERNORM_TYPE"] = "fast_layernorm"
    end

    use_deepspeed_evo = get(cfg, "use_deepspeed_evo_attention", false)
    ENV["DEEPSPEED_EVO"] = use_deepspeed_evo ? "true" : "false"

    if use_deepspeed_evo
        cutlass_path = get(ENV, "CUTLASS_PATH", joinpath(homedir(), "cutlass"))
        ENV["CUTLASS_PATH"] = cutlass_path
        if !isdir(cutlass_path)
            println(io, "[warn] CUTLASS not found at $cutlass_path")
            println(io, "[warn] DeepSpeed Evo kernels may fail until CUTLASS v3.5.1 is installed.")
        end
    end
end

function _persist_config(cfg::Dict{String, Any}, path::AbstractString)
    open(path, "w") do io
        TOML.print(io, cfg)
    end
end

function _persist_task_manifest(tasks, path::AbstractString)
    manifest = Dict(
        "tasks" => [
            Dict(
                "name" => task.name,
                "structure_file" => task.structure_file,
                "num_chains" => length(task.chain_ids),
                "generation_count" => length(task.generation),
            ) for task in tasks
        ],
    )
    open(path, "w") do io
        TOML.print(io, manifest)
    end
end

function _ensure_seed_list(cfg::Dict{String, Any})
    seeds = get(cfg, "seeds", Int[])
    if seeds isa AbstractVector && !isempty(seeds)
        return Int.(seeds)
    end
    return [derive_seed(time_ns())]
end

function _to_bool_vec(x)
    if x isa AbstractVector{Bool}
        return x
    elseif x isa AbstractVector
        return [xi != 0 for xi in x]
    end
    error("Expected vector mask, got $(typeof(x))")
end

function _to_matrix_f32(x)
    x isa AbstractMatrix || error("Expected matrix feature, got $(typeof(x))")
    return Float32.(x)
end

function _make_scheduler(cfg::Dict{String, Any})
    s_cfg = get(cfg, "inference_noise_scheduler", Dict{String, Any}())
    return InferenceNoiseScheduler(
        s_max = Float64(get(s_cfg, "s_max", 160.0)),
        s_min = Float64(get(s_cfg, "s_min", 4e-4)),
        rho = Float64(get(s_cfg, "rho", 7.0)),
        sigma_data = Float64(get(s_cfg, "sigma_data", 16.0)),
    )
end

function _pad_or_truncate_columns(x::AbstractMatrix{<:Real}, width::Int)
    n, d = size(x)
    if d == width
        return Float32.(x)
    elseif d > width
        return Float32.(x[:, 1:width])
    end
    out = zeros(Float32, n, width)
    out[:, 1:d] .= Float32.(x)
    return out
end

function _build_scaffold_model_inputs(
    feat::Dict{String, Any},
    c_s_inputs::Int,
    c_s::Int,
    c_z::Int;
    design_condition_embedder = nothing,
)
    relpos_input = as_relpos_input(feat)
    n_token = length(relpos_input.token_index)

    s_trunk = zeros(Float32, n_token, c_s)
    if design_condition_embedder === nothing
        restype = _to_matrix_f32(feat["restype"])
        profile = _to_matrix_f32(feat["profile"])
        deletion_mean = Float32.(feat["deletion_mean"])
        plddt = Float32.(feat["plddt"])
        hotspot = Float32.(feat["hotspot"])
        token_features = hcat(
            restype,
            profile,
            reshape(deletion_mean, :, 1),
            reshape(plddt, :, 1),
            reshape(hotspot, :, 1),
        )

        s_inputs = _pad_or_truncate_columns(token_features, c_s_inputs)
        z_trunk = zeros(Float32, n_token, n_token, c_z)
        if c_z > 0
            templ_mask = _to_matrix_f32(feat["conditional_templ_mask"])
            z_trunk[:, :, 1] .= templ_mask
        end
        if c_z > 1
            templ_bins = _to_matrix_f32(feat["conditional_templ"])
            z_trunk[:, :, 2] .= templ_bins ./ 63f0
        end
        atom_to_token_idx = Int.(feat["atom_to_token_idx"])
        return (
            relpos_input = relpos_input,
            atom_input = as_atom_attention_input(feat),
            s_inputs = s_inputs,
            s_trunk = s_trunk,
            z_trunk = z_trunk,
            atom_to_token_idx = atom_to_token_idx,
        )
    end

    s_inputs, z_trunk = design_condition_embedder(feat)
    size(s_inputs, 1) == n_token || error("DesignConditionEmbedder returned mismatched token count.")
    size(s_inputs, 2) == c_s_inputs || error("DesignConditionEmbedder returned c_s_inputs=$(size(s_inputs, 2)) expected $c_s_inputs.")
    size(z_trunk, 1) == n_token || error("DesignConditionEmbedder returned mismatched pair token count.")
    size(z_trunk, 2) == n_token || error("DesignConditionEmbedder returned mismatched pair token count.")
    size(z_trunk, 3) == c_z || error("DesignConditionEmbedder returned c_z=$(size(z_trunk, 3)) expected $c_z.")

    atom_to_token_idx = Int.(feat["atom_to_token_idx"])
    return (
        relpos_input = relpos_input,
        atom_input = as_atom_attention_input(feat),
        s_inputs = s_inputs,
        s_trunk = s_trunk,
        z_trunk = z_trunk,
        atom_to_token_idx = atom_to_token_idx,
    )
end

function _run_diffusion_coordinates(feature_bundle::Dict{String, Any}, cfg::Dict{String, Any}, seed::Int)
    feat = feature_bundle["input_feature_dict"]
    dims = feature_bundle["dims"]

    n_atom = Int(dims["N_atom"])
    diff_cfg = cfg["sample_diffusion"]

    n_sample = Int(get(diff_cfg, "N_sample", 1))
    n_step = Int(get(diff_cfg, "N_step", 40))
    gamma0 = Float64(get(diff_cfg, "gamma0", 1.0))
    gamma_min = Float64(get(diff_cfg, "gamma_min", 0.01))
    noise_scale_lambda = Float64(get(diff_cfg, "noise_scale_lambda", 1.003))
    eta = get(diff_cfg, "eta_schedule", Dict("type" => "const", "min" => 1.5, "max" => 1.5))
    infer_setting = get(cfg, "infer_setting", Dict{String, Any}())
    diffusion_chunk_size = Int(get(infer_setting, "sample_diffusion_chunk_size", 0))

    model_scaffold_cfg = get(cfg, "model_scaffold", Dict{String, Any}())
    use_model_scaffold = Bool(get(model_scaffold_cfg, "enabled", false))

    typed_model = nothing
    model_inputs = nothing
    if use_model_scaffold
        raw_weights_dir = String(get(cfg, "raw_weights_dir", ""))
        safetensors_weights_path = String(get(cfg, "safetensors_weights_path", ""))
        weights = nothing
        if !isempty(raw_weights_dir) && !isempty(safetensors_weights_path)
            error("Set only one of raw_weights_dir or safetensors_weights_path.")
        end
        if !isempty(safetensors_weights_path)
            weights = load_safetensors_weights(safetensors_weights_path)
        elseif !isempty(raw_weights_dir)
            weights = load_raw_weights(raw_weights_dir)
        end

        auto_dims_from_weights = Bool(get(model_scaffold_cfg, "auto_dims_from_weights", false))
        if auto_dims_from_weights
            weights === nothing && error(
                "model_scaffold.auto_dims_from_weights=true requires raw_weights_dir or safetensors_weights_path to be set.",
            )
            inferred = infer_model_scaffold_dims(weights)
            c_token = inferred.c_token
            c_s = inferred.c_s
            c_z = inferred.c_z
            c_s_inputs = inferred.c_s_inputs
            n_blocks = inferred.n_blocks
            n_heads = inferred.n_heads
            c_atom = inferred.c_atom
            c_atompair = inferred.c_atompair
            atom_encoder_blocks = inferred.atom_encoder_blocks
            atom_encoder_heads = inferred.atom_encoder_heads
            atom_decoder_blocks = inferred.atom_decoder_blocks
            atom_decoder_heads = inferred.atom_decoder_heads
        else
            c_token = Int(get(model_scaffold_cfg, "c_token", 64))
            c_s = Int(get(model_scaffold_cfg, "c_s", 64))
            c_z = Int(get(model_scaffold_cfg, "c_z", 32))
            c_s_inputs = Int(get(model_scaffold_cfg, "c_s_inputs", 128))
            n_blocks = Int(get(model_scaffold_cfg, "n_blocks", 2))
            n_heads = Int(get(model_scaffold_cfg, "n_heads", 4))
            c_atom = Int(get(model_scaffold_cfg, "c_atom", 128))
            c_atompair = Int(get(model_scaffold_cfg, "c_atompair", 16))
            atom_encoder_blocks = Int(get(model_scaffold_cfg, "atom_encoder_blocks", 3))
            atom_encoder_heads = Int(get(model_scaffold_cfg, "atom_encoder_heads", 4))
            atom_decoder_blocks = Int(get(model_scaffold_cfg, "atom_decoder_blocks", 3))
            atom_decoder_heads = Int(get(model_scaffold_cfg, "atom_decoder_heads", 4))
        end

        typed_model = DiffusionModule(
            c_token,
            c_s,
            c_z,
            c_s_inputs;
            c_atom = c_atom,
            c_atompair = c_atompair,
            atom_encoder_blocks = atom_encoder_blocks,
            atom_encoder_heads = atom_encoder_heads,
            n_blocks = n_blocks,
            n_heads = n_heads,
            atom_decoder_blocks = atom_decoder_blocks,
            atom_decoder_heads = atom_decoder_heads,
            rng = MersenneTwister(seed + 17),
        )
        if weights !== nothing
            strict_weight_load = Bool(get(cfg, "strict_weight_load", false))
            load_diffusion_module!(typed_model, weights; strict = strict_weight_load)
        end
        use_design_condition_embedder = Bool(get(model_scaffold_cfg, "use_design_condition_embedder", true))
        design_condition_embedder = nothing
        if use_design_condition_embedder
            design_c_token = c_token
            design_n_blocks = 3
            design_n_heads = 4
            if weights !== nothing &&
               haskey(weights, "design_condition_embedder.input_embedder.input_map.weight") &&
               haskey(weights, "design_condition_embedder.condition_template_embedder.embedder.weight")
                inferred_design = infer_design_condition_embedder_dims(weights)
                inferred_design.c_s_inputs == c_s_inputs ||
                    error("DesignConditionEmbedder c_s_inputs mismatch: diffusion inferred $c_s_inputs, design inferred $(inferred_design.c_s_inputs)")
                inferred_design.c_z == c_z ||
                    error("DesignConditionEmbedder c_z mismatch: diffusion inferred $c_z, design inferred $(inferred_design.c_z)")
                design_c_token = inferred_design.c_token
                design_n_blocks = inferred_design.n_blocks
                design_n_heads = inferred_design.n_heads
            end
            design_condition_embedder = DesignConditionEmbedder(
                design_c_token;
                c_s_inputs = c_s_inputs,
                c_z = c_z,
                c_atom = c_atom,
                c_atompair = c_atompair,
                n_blocks = design_n_blocks,
                n_heads = design_n_heads,
                rng = MersenneTwister(seed + 31),
            )
            if weights !== nothing
                strict_weight_load = Bool(get(cfg, "strict_weight_load", false))
                load_design_condition_embedder!(
                    design_condition_embedder,
                    weights;
                    strict = strict_weight_load,
                )
            end
        end
        if weights !== nothing && Bool(get(cfg, "strict_weight_load", false))
            report = checkpoint_coverage_report(typed_model, design_condition_embedder, weights)
            if !isempty(report.missing) || !isempty(report.unused)
                error(
                    "Checkpoint key coverage mismatch: missing=$(length(report.missing)) unused=$(length(report.unused)). " *
                    "Set strict_weight_load=false to allow partial loads.",
                )
            end
        end
        model_inputs = _build_scaffold_model_inputs(
            feat,
            c_s_inputs,
            c_s,
            c_z;
            design_condition_embedder = design_condition_embedder,
        )
    end

    denoise_net = function (x_noisy, t_hat; kwargs...)
        x = if use_model_scaffold
            typed_model(
                x_noisy,
                t_hat;
                relpos_input = model_inputs.relpos_input,
                s_inputs = model_inputs.s_inputs,
                s_trunk = model_inputs.s_trunk,
                z_trunk = model_inputs.z_trunk,
                atom_to_token_idx = model_inputs.atom_to_token_idx,
                input_feature_dict = model_inputs.atom_input,
            )
        else
            copy(x_noisy)
        end
        return x
    end

    scheduler = _make_scheduler(cfg)
    noise_schedule = scheduler(n_step; dtype = Float32)
    rng = MersenneTwister(seed)
    return sample_diffusion(
        denoise_net;
        noise_schedule = noise_schedule,
        N_sample = n_sample,
        N_atom = n_atom,
        gamma0 = gamma0,
        gamma_min = gamma_min,
        noise_scale_lambda = noise_scale_lambda,
        step_scale_eta = eta,
        diffusion_chunk_size = diffusion_chunk_size,
        rng = rng,
    )
end

function _write_stub_prediction(task_dump_dir::AbstractString, task_name::AbstractString, seed::Int)
    predictions_dir = joinpath(task_dump_dir, "predictions")
    mkpath(predictions_dir)
    placeholder = Dict(
        "status" => "dry_run",
        "task_name" => task_name,
        "seed" => seed,
        "message" => "Dry-run mode: feature pipeline executed, model inference skipped.",
        "timestamp_utc" => string(now(UTC)),
    )
    open(joinpath(predictions_dir, "PXDesign_stub_result.toml"), "w") do io
        TOML.print(io, placeholder)
    end
end

function _write_feature_manifest(task_dump_dir::AbstractString, feature_bundle::Dict{String, Any})
    preds = joinpath(task_dump_dir, "predictions")
    mkpath(preds)
    dims = feature_bundle["dims"]
    feat = feature_bundle["input_feature_dict"]
    condition_token_count = count(identity, _to_bool_vec(feat["condition_token_mask"]))
    design_token_count = count(identity, _to_bool_vec(feat["design_token_mask"]))
    open(joinpath(preds, "feature_manifest.toml"), "w") do io
        TOML.print(
            io,
            Dict(
                "task_name" => feature_bundle["task_name"],
                "N_token" => dims["N_token"],
                "N_atom" => dims["N_atom"],
                "N_msa" => dims["N_msa"],
                "N_condition_token" => condition_token_count,
                "N_design_token" => design_token_count,
            ),
        )
    end
end

function run_infer(cfg::Dict{String, Any}; dry_run::Bool = false, io::IO = stdout)
    dump_dir = String(cfg["dump_dir"])
    raw_input_path = String(cfg["input_json_path"])
    isempty(raw_input_path) && error("input_json_path is required.")

    mkpath(dump_dir)
    _setup_runtime_env!(cfg; io = io)

    input_path = process_input_file(raw_input_path; out_dir = dump_dir)
    cfg["input_json_path"] = input_path
    raw_tasks = load_input_tasks(input_path)
    tasks = parse_tasks(raw_tasks)
    _persist_config(cfg, joinpath(dump_dir, "config.toml"))
    snapshot_input(input_path, joinpath(dump_dir, "input_source_snapshot" * splitext(input_path)[2]))
    _persist_task_manifest(tasks, joinpath(dump_dir, "task_manifest.toml"))

    if get(cfg, "download_cache", false)
        ensure_inference_cache!(cfg; dry_run = dry_run, io = io)
    else
        println(io, "[cache] skipped (download_cache=false)")
    end

    seeds = _ensure_seed_list(cfg)
    println(io, "[infer] tasks=$(length(tasks)) seeds=$(join(seeds, ','))")

    for (run_idx, seed) in enumerate(seeds)
        global_run_dir = joinpath(dump_dir, "global_run_$(run_idx - 1)")
        feature_rng = MersenneTwister(seed)
        for task in tasks
            task_name = task.name
            task_dump_dir = joinpath(global_run_dir, task_name, "seed_$(seed)")
            feature_bundle = build_basic_feature_bundle(task; rng = feature_rng)
            _write_feature_manifest(task_dump_dir, feature_bundle)

            if dry_run
                _write_stub_prediction(task_dump_dir, task_name, seed)
                continue
            end

            coordinates = _run_diffusion_coordinates(feature_bundle, cfg, seed)
            dump_prediction_bundle(task_dump_dir, task_name, feature_bundle["atoms"], coordinates)
            pred_dir = joinpath(task_dump_dir, "predictions")
            println(
                io,
                "[infer] wrote $(size(coordinates, 1)) sample(s) to $pred_dir",
            )
        end
    end

    if dry_run
        println(io, "[infer] dry-run complete (input + feature manifests validated).")
        return Dict("status" => "dry_run", "dump_dir" => dump_dir)
    end

    return Dict(
        "status" => "ok_scaffold_model",
        "dump_dir" => dump_dir,
        "num_tasks" => length(tasks),
        "num_runs" => length(seeds),
    )
end

end

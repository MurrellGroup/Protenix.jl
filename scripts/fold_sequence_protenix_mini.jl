#!/usr/bin/env julia

using Random
using PXDesign

function _usage()
    println("Usage: julia --project=. scripts/fold_sequence_protenix_mini.jl <sequence> [out_dir] [n_step] [n_sample] [seed]")
end

function main(args::Vector{String})
    isempty(args) && (_usage(); error("Missing sequence argument"))

    sequence = strip(args[1])
    isempty(sequence) && error("Sequence must be non-empty")

    out_dir = length(args) >= 2 ? args[2] : joinpath(pwd(), "output", "protenix_mini_sequence")
    n_step = length(args) >= 3 ? parse(Int, args[3]) : 5
    n_sample = length(args) >= 4 ? parse(Int, args[4]) : 1
    seed = length(args) >= 5 ? parse(Int, args[5]) : 0

    weights_dir = joinpath(pwd(), "weights_safetensors_protenix_mini_default_v0.5.0")
    isdir(weights_dir) || error("Missing weights directory: $weights_dir")

    weights = PXDesign.Model.load_safetensors_weights(weights_dir)
    model = PXDesign.ProtenixMini.build_protenix_mini_model(weights)
    PXDesign.ProtenixMini.load_protenix_mini_model!(model, weights; strict = true)

    rng = MersenneTwister(seed)
    folded = PXDesign.fold_sequence(
        model,
        sequence;
        n_step = n_step,
        n_sample = n_sample,
        rng = rng,
    )

    task_dir = joinpath(out_dir, "seed_$(seed)")
    pred_dir = PXDesign.Output.dump_prediction_bundle(
        task_dir,
        "protenix_mini_sequence",
        folded.atoms,
        folded.prediction.coordinate,
    )
    cif_path = joinpath(pred_dir, "protenix_mini_sequence_sample_0.cif")
    isfile(cif_path) || error("Missing CIF output: $cif_path")

    println("protenix_mini_sequence_fold_ok")
    println("pred_dir=$(pred_dir)")
    println("cif_path=$(cif_path)")
    println("bytes=$(filesize(cif_path))")
end

main(ARGS)

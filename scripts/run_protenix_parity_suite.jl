#!/usr/bin/env julia

using Dates

const ROOT = normpath(joinpath(@__DIR__, ".."))
const PYTHON = joinpath(ROOT, ".venv_pyref", "bin", "python")

function _run(cmd::Cmd)
    println("[run] ", cmd)
    run(cmd)
end

function _require_file(path::AbstractString, label::AbstractString)
    isfile(path) || error("$label not found: $path")
end

function main()
    mini_ckpt = joinpath(ROOT, "release_data", "checkpoint", "protenix_mini_default_v0.5.0.pt")
    base_ckpt = joinpath(ROOT, "release_data", "checkpoint", "protenix_base_default_v0.5.0.pt")
    _require_file(PYTHON, "Python interpreter")
    _require_file(mini_ckpt, "Protenix-mini checkpoint")
    _require_file(base_ckpt, "Protenix-base checkpoint")

    diag_msa = "/tmp/py_msa_diag.json"
    diag_pair = "/tmp/py_pairformer_diag.json"
    diag_base = "/tmp/py_protenix_base_trunk_denoise_diag.json"

    t0 = now()

    _run(`$PYTHON $(joinpath(ROOT, "scripts", "dump_python_msa_parity.py")) --checkpoint $mini_ckpt --out $diag_msa`)
    _run(`$PYTHON $(joinpath(ROOT, "scripts", "dump_python_pairformer_parity.py")) --checkpoint $mini_ckpt --out $diag_pair`)
    _run(`$PYTHON $(joinpath(ROOT, "scripts", "dump_python_protenix_base_trunk_denoise_parity.py")) --checkpoint $base_ckpt --out $diag_base`)

    julia = Base.julia_cmd()
    depot = get(ENV, "JULIA_DEPOT_PATH", joinpath(ROOT, ".julia_depot") * ":" * joinpath(homedir(), ".julia"))
    juliaup_depot = get(ENV, "JULIAUP_DEPOT_PATH", joinpath(ROOT, ".julia_depot"))
    _run(Cmd(
        `$julia --project=$ROOT $(joinpath(ROOT, "scripts", "compare_msa_parity.jl"))`;
        env = Dict("MSA_DIAG" => diag_msa, "JULIA_DEPOT_PATH" => depot, "JULIAUP_DEPOT_PATH" => juliaup_depot),
    ))
    _run(Cmd(
        `$julia --project=$ROOT $(joinpath(ROOT, "scripts", "compare_pairformer_parity.jl"))`;
        env = Dict("PAIRFORMER_DIAG" => diag_pair, "JULIA_DEPOT_PATH" => depot, "JULIAUP_DEPOT_PATH" => juliaup_depot),
    ))
    _run(Cmd(
        `$julia --project=$ROOT $(joinpath(ROOT, "scripts", "compare_protenix_base_trunk_denoise_parity.jl"))`;
        env = Dict("PBASE_TRUNK_DENOISE_DIAG" => diag_base, "JULIA_DEPOT_PATH" => depot, "JULIAUP_DEPOT_PATH" => juliaup_depot),
    ))

    dt = now() - t0
    println("[done] Protenix parity suite completed in ", dt)
end

main()

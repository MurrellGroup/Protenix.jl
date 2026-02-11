#!/usr/bin/env julia

using Dates
using Serialization
using SHA

const _REGEN_ENV = "PXDESIGN_LAYER_FIXTURE_REGEN"
const _REGEN_MAGIC = "do-not-set-this"

if get(ENV, _REGEN_ENV, "") != _REGEN_MAGIC
    error(
        "Refusing to regenerate layer regression fixtures.\n" *
        "Set $_REGEN_ENV=$_REGEN_MAGIC only when you intentionally want to overwrite goldens."
    )
end

include(joinpath(@__DIR__, "..", "test", "layer_regression_reference.jl"))
using .LayerRegressionReference

fixture_path = length(ARGS) >= 1 ? abspath(ARGS[1]) : abspath(joinpath(@__DIR__, "..", "test", "regression_fixtures", "layer_regression_v1.bin"))
mkpath(dirname(fixture_path))

outputs = LayerRegressionReference.compute_layer_regression_outputs()
payload = Dict{String, Any}(
    "format_version" => "layer_regression_v1",
    "generated_at_utc" => Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"),
    "julia_version" => string(VERSION),
    "regen_guard_env" => _REGEN_ENV,
    "regen_guard_magic" => _REGEN_MAGIC,
    "keys" => sort(collect(keys(outputs))),
    "outputs" => outputs,
)

open(fixture_path, "w") do io
    serialize(io, payload)
end

blob = read(fixture_path)
println("wrote fixture: $fixture_path")
println("sha1: ", bytes2hex(sha1(blob)))
println("entries: ", length(outputs))

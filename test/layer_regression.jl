using Test
using Serialization
using SHA

include("layer_regression_reference.jl")
using .LayerRegressionReference

const _FIXTURE_PATH = abspath(joinpath(@__DIR__, "regression_fixtures", "layer_regression_v1.bin"))
const _KEY_TOLERANCES = Dict(
    # Pairformer trunk accumulation can shift slightly under layout-preserving refactors.
    "protenix_mini.model_trunk.z" => (atol = 3e-4, rtol = 3e-4),
)

function _tolerances_for_key(k::String)
    return get(_KEY_TOLERANCES, k, (atol = 1e-5, rtol = 1e-5))
end

function _compare_values(k::String, expected, actual; atol::Float64 = 1e-5, rtol::Float64 = 1e-5)
    if expected isa AbstractArray
        @test actual isa AbstractArray
        @test size(actual) == size(expected)
        @test isapprox(Float32.(actual), Float32.(expected); atol = atol, rtol = rtol)
        return
    elseif expected isa Number
        @test actual isa Number
        @test isapprox(float(actual), float(expected); atol = atol, rtol = rtol)
        return
    elseif expected isa AbstractDict
        @test actual isa AbstractDict
        @test Set(keys(actual)) == Set(keys(expected))
        for kk in sort!(collect(keys(expected)))
            _compare_values("$(k).$(kk)", expected[kk], actual[kk]; atol = atol, rtol = rtol)
        end
        return
    end
    @test actual == expected
end

@testset "Layer regression fixtures" begin
    isfile(_FIXTURE_PATH) || error(
        "Missing layer regression fixture at $_FIXTURE_PATH.\n" *
        "Generate it intentionally with:\n" *
        "  PROTENIX_LAYER_FIXTURE_REGEN=do-not-set-this julia --project=. scripts/generate_layer_regression_fixtures.jl"
    )

    payload = open(_FIXTURE_PATH, "r") do io
        deserialize(io)
    end
    @test payload isa AbstractDict
    @test haskey(payload, "format_version")
    @test payload["format_version"] == "layer_regression_v1"
    @test haskey(payload, "outputs")
    expected_outputs = payload["outputs"]
    @test expected_outputs isa AbstractDict

    # Optional safety visibility in test logs for fixture provenance.
    fixture_sha1 = bytes2hex(sha1(read(_FIXTURE_PATH)))
    @test length(fixture_sha1) == 40

    actual_outputs = LayerRegressionReference.compute_layer_regression_outputs()
    @test Set(keys(actual_outputs)) == Set(keys(expected_outputs))
    for k in sort!(collect(keys(expected_outputs)))
        tol = _tolerances_for_key(k)
        _compare_values(k, expected_outputs[k], actual_outputs[k]; atol = tol.atol, rtol = tol.rtol)
    end
end

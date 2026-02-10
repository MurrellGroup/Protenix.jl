#!/usr/bin/env julia

using LinearAlgebra
using Statistics
using PXDesign

function _nested_dims(x)
    dims = Int[]
    cur = x
    while cur isa AbstractVector
        push!(dims, length(cur))
        isempty(cur) && break
        cur = cur[1]
    end
    return dims
end

function _to_array_f32(x)
    x isa AbstractVector || return Float32(x)
    dims = _nested_dims(x)
    out = Array{Float32}(undef, Tuple(dims)...)
    idx = Int[]
    function walk(y)
        if y isa AbstractVector
            for (i, z) in enumerate(y)
                push!(idx, i)
                walk(z)
                pop!(idx)
            end
        else
            out[Tuple(idx)...] = Float32(y)
        end
    end
    walk(x)
    return out
end

function _to_array_i(x)
    x isa AbstractVector || return Int(x)
    dims = _nested_dims(x)
    out = Array{Int}(undef, Tuple(dims)...)
    idx = Int[]
    function walk(y)
        if y isa AbstractVector
            for (i, z) in enumerate(y)
                push!(idx, i)
                walk(z)
                pop!(idx)
            end
        else
            out[Tuple(idx)...] = Int(round(Float64(y)))
        end
    end
    walk(x)
    return out
end

function _to_bool_vec(x)
    x isa AbstractVector || error("expected vector")
    return [Bool(y) for y in x]
end

function _load_diag(path::AbstractString)
    raw = PXDesign.JSONLite.parse_json(read(path, String))
    feat = Dict{String, Any}()
    required_int = (
        "token_index",
        "residue_index",
        "asym_id",
        "entity_id",
        "sym_id",
        "atom_to_token_idx",
        "ref_space_uid",
        "ref_mask",
    )
    required_f32 = ("ref_pos", "ref_charge", "ref_element", "ref_atom_name_chars")
    optional_int = ("conditional_templ", "conditional_templ_mask")
    optional_f32 = ("restype", "profile", "deletion_mean", "plddt", "hotspot")

    for k in required_int
        key = "feat_" * k
        haskey(raw, key) || error("missing key $key in diag")
        feat[k] = _to_array_i(raw[key])
    end
    for k in required_f32
        key = "feat_" * k
        haskey(raw, key) || error("missing key $key in diag")
        feat[k] = _to_array_f32(raw[key])
    end
    for k in optional_int
        key = "feat_" * k
        if haskey(raw, key)
            feat[k] = _to_array_i(raw[key])
        end
    end
    for k in optional_f32
        key = "feat_" * k
        if haskey(raw, key)
            feat[k] = _to_array_f32(raw[key])
        end
    end
    if haskey(raw, "feat_condition_atom_mask")
        feat["condition_atom_mask"] = _to_bool_vec(raw["feat_condition_atom_mask"])
    end
    # Python may store ref_atom_name_chars as [N_atom, 4, 64]; Julia path expects [N_atom, 256].
    rac = feat["ref_atom_name_chars"]
    if rac isa AbstractArray && ndims(rac) == 3 && size(rac, 2) * size(rac, 3) == 256
        n_atom = size(rac, 1)
        n_char = size(rac, 2)
        n_vocab = size(rac, 3)
        flat = Array{Float32}(undef, n_atom, n_char * n_vocab)
        @inbounds for a in 1:n_atom, c in 1:n_char, v in 1:n_vocab
            flat[a, (c - 1) * n_vocab + v] = Float32(rac[a, c, v])
        end
        feat["ref_atom_name_chars"] = flat
    end

    return (
        t_hat = vec(_to_array_f32(raw["t_hat"])),
        x_noisy = _to_array_f32(raw["x_noisy"]),
        x_denoised_py = _to_array_f32(raw["x_denoised"]),
        s_inputs = _to_array_f32(raw["s_inputs"]),
        s_trunk = _to_array_f32(raw["s_trunk"]),
        z_trunk = _to_array_f32(raw["z_trunk"]),
        feat = feat,
    )
end

function main()
    diag = _load_diag("/tmp/py_step_diag.json")
    weights = PXDesign.Model.load_raw_weights(joinpath(pwd(), "weights_raw"))

    dims = PXDesign.Model.infer_model_scaffold_dims(weights)
    dm = PXDesign.Model.DiffusionModule(
        dims.c_token,
        dims.c_s,
        dims.c_z,
        dims.c_s_inputs;
        c_atom = dims.c_atom,
        c_atompair = dims.c_atompair,
        atom_encoder_blocks = dims.atom_encoder_blocks,
        atom_encoder_heads = dims.atom_encoder_heads,
        n_blocks = dims.n_blocks,
        n_heads = dims.n_heads,
        atom_decoder_blocks = dims.atom_decoder_blocks,
        atom_decoder_heads = dims.atom_decoder_heads,
    )
    PXDesign.Model.load_diffusion_module!(dm, weights; strict = true)

    relpos = PXDesign.Model.as_relpos_input(diag.feat)
    atom_input = PXDesign.Model.as_atom_attention_input(diag.feat)
    x_denoised_jl = dm(
        diag.x_noisy,
        diag.t_hat;
        relpos_input = relpos,
        s_inputs = diag.s_inputs,
        s_trunk = diag.s_trunk,
        z_trunk = diag.z_trunk,
        atom_to_token_idx = vec(Int.(diag.feat["atom_to_token_idx"])),
        input_feature_dict = atom_input,
    )

    py = Float32.(diag.x_denoised_py)
    jl = Float32.(x_denoised_jl)
    size(py) == size(jl) || error("shape mismatch: py=$(size(py)) jl=$(size(jl))")
    absdiff = abs.(jl .- py)
    denom = max.(abs.(py), 1f-6)
    reldiff = absdiff ./ denom

    println("x_denoised step parity:")
    println("  py_norm=", norm(py))
    println("  jl_norm=", norm(jl))
    println("  max_abs=", maximum(absdiff))
    println("  mean_abs=", mean(absdiff))
    println("  max_rel=", maximum(reldiff))
end

main()

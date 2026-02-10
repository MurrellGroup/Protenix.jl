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

function _load_diag(path::AbstractString)
    raw = PXDesign.JSONLite.parse_json(read(path, String))
    feat = Dict{String, Any}()
    for k in (
        "token_index",
        "residue_index",
        "asym_id",
        "entity_id",
        "sym_id",
        "atom_to_token_idx",
        "ref_pos",
        "ref_mask",
        "ref_charge",
        "ref_element",
        "ref_atom_name_chars",
        "ref_space_uid",
    )
        v = raw["feat_" * k]
        feat[k] = k in ("token_index", "residue_index", "asym_id", "entity_id", "sym_id", "atom_to_token_idx", "ref_space_uid") ? _to_array_i(v) : _to_array_f32(v)
    end

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
        s_inputs = _to_array_f32(raw["s_inputs"]),
        s_trunk = _to_array_f32(raw["s_trunk"]),
        z_trunk = _to_array_f32(raw["z_trunk"]),
        single_s_py = _to_array_f32(raw["single_s"]),
        pair_z_py = _to_array_f32(raw["pair_z"]),
        a_enc_py = _to_array_f32(raw["a_enc"]),
        q_skip_py = _to_array_f32(raw["q_skip"]),
        c_skip_py = _to_array_f32(raw["c_skip"]),
        p_skip_py = _to_array_f32(raw["p_skip"]),
        r_update_py = _to_array_f32(raw["r_update"]),
        x_denoised_py = _to_array_f32(raw["x_denoised"]),
        feat = feat,
    )
end

function _report(name::AbstractString, py, jl)
    a = Float32.(py)
    b = Float32.(jl)
    size(a) == size(b) || error("$name shape mismatch: py=$(size(a)) jl=$(size(b))")
    d = abs.(a .- b)
    rel = d ./ max.(abs.(a), 1f-6)
    println(name)
    println("  py_norm=", norm(a))
    println("  jl_norm=", norm(b))
    println("  max_abs=", maximum(d), " mean_abs=", mean(d), " max_rel=", maximum(rel))
end

function main()
    diag = _load_diag("/tmp/py_step_blocks.json")
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

    single_s, pair_z = dm.diffusion_conditioning(
        diag.t_hat,
        relpos,
        diag.s_inputs,
        diag.s_trunk,
        diag.z_trunk,
    )
    _report("single_s", diag.single_s_py, single_s)
    _report("pair_z", diag.pair_z_py, pair_z)

    sigma_data = dm.diffusion_conditioning.sigma_data
    n_sample = size(diag.x_noisy, 1)
    n_atom = size(diag.x_noisy, 2)
    t_scale = sqrt.(sigma_data^2 .+ diag.t_hat .^ 2)
    r_noisy = Array{Float32}(undef, n_sample, n_atom, 3)
    for i in 1:n_sample
        @inbounds r_noisy[i, :, :] = diag.x_noisy[i, :, :] ./ t_scale[i]
    end

    n_token = size(single_s, 2)
    s_tok = Array{Float32}(undef, n_sample, n_token, dm.c_s)
    z_tok = Array{Float32}(undef, n_sample, n_token, n_token, dm.c_z)
    for i in 1:n_sample
        @inbounds s_tok[i, :, :] = diag.s_trunk
        @inbounds z_tok[i, :, :, :] = pair_z
    end

    a_enc, q_skip, c_skip, p_skip = dm.atom_attention_encoder(
        atom_input;
        r_l = r_noisy,
        s = s_tok,
        z = z_tok,
    )
    _report("a_enc", diag.a_enc_py, a_enc)
    _report("q_skip", diag.q_skip_py, q_skip)
    _report("c_skip", diag.c_skip_py, c_skip)
    _report("p_skip", diag.p_skip_py, p_skip)

    a_post = a_enc .+ dm.linear_no_bias_s(dm.layernorm_s(single_s))
    a_ln = Array{Float32}(undef, size(a_post))
    for i in 1:n_sample
        a_i = Array{Float32,2}(a_post[i, :, :])
        s_i = Array{Float32,2}(single_s[i, :, :])
        a_i = dm.diffusion_transformer(a_i, s_i, pair_z)
        @inbounds a_ln[i, :, :] = dm.layernorm_a(a_i)
    end

    r_update = dm.atom_attention_decoder(atom_input, a_ln, q_skip, c_skip, p_skip)
    _report("r_update", diag.r_update_py, r_update)

    s_ratio = reshape(diag.t_hat ./ sigma_data, n_sample, 1, 1)
    t_expanded = reshape(diag.t_hat, n_sample, 1, 1)
    x_denoised = (1f0 ./ (1f0 .+ s_ratio .^ 2)) .* diag.x_noisy .+
                 (t_expanded ./ sqrt.(1f0 .+ s_ratio .^ 2)) .* r_update
    _report("x_denoised", diag.x_denoised_py, x_denoised)
end

main()

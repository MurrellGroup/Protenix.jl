module ESMProvider

export embed_sequence, embed_sequences, clear_esm_cache!, set_sequence_embedder_override!, hf_hub_download_file

using SafeTensors

const _MODEL_CACHE = Dict{Tuple{Symbol, String, String, String, Bool, Symbol}, Any}()
const _SEQUENCE_CACHE = Dict{Tuple{Symbol, String, String, String, Bool, Symbol, String}, Matrix{Float32}}()
const _CACHE_LOCK = ReentrantLock()
const _SEQUENCE_EMBEDDER_OVERRIDE = Ref{Union{Nothing, Function}}(nothing)
const _ESMFOLD_MODULE = Ref{Union{Nothing, Module}}(nothing)

function _env_bool(key::AbstractString, default::Bool = false)
    raw = get(ENV, String(key), "")
    isempty(raw) && return default
    s = lowercase(strip(raw))
    return s in ("1", "true", "yes", "y", "on")
end

function _resolve_source(variant::Symbol)
    local_files_only = _env_bool("PXDESIGN_ESM_LOCAL_FILES_ONLY", false)
    if variant == :esm2_3b
        repo_id = get(ENV, "PXDESIGN_ESM_REPO_ID", "facebook/esmfold_v1")
        filename = get(ENV, "PXDESIGN_ESM_FILENAME", "model.safetensors")
        revision = get(ENV, "PXDESIGN_ESM_REVISION", "ba837a3")
        loader_kind = :esmfold_embed
        return (
            repo_id = repo_id,
            filename = filename,
            revision = revision,
            local_files_only = local_files_only,
            loader_kind = loader_kind,
        )
    elseif variant == :esm2_3b_ism
        repo_id = get(ENV, "PXDESIGN_ESM_ISM_REPO_ID", get(ENV, "PXDESIGN_WEIGHTS_REPO_ID", "MurrellLab/PXDesign.jl"))
        filename = get(
            ENV,
            "PXDESIGN_ESM_ISM_FILENAME",
            "weights_safetensors_esm2_t36_3B_UR50D_ism/esm2_t36_3B_UR50D_ism.safetensors",
        )
        revision = get(ENV, "PXDESIGN_ESM_ISM_REVISION", get(ENV, "PXDESIGN_WEIGHTS_REVISION", "main"))
        loader_kind_raw = lowercase(strip(get(ENV, "PXDESIGN_ESM_ISM_LOADER", "fair_esm2")))
        loader_kind = loader_kind_raw == "esmfold_embed" ? :esmfold_embed : :fair_esm2
        return (
            repo_id = repo_id,
            filename = filename,
            revision = revision,
            local_files_only = local_files_only,
            loader_kind = loader_kind,
        )
    end
    error("Unsupported ESM variant: $variant")
end

function _esmfold_module()
    mod = _ESMFOLD_MODULE[]
    mod !== nothing && return mod
    try
        Base.eval(@__MODULE__, :(import ESMFold))
        mod = getfield(@__MODULE__, :ESMFold)
        _ESMFOLD_MODULE[] = mod
        return mod
    catch err
        msg = sprint(showerror, err)
        error(
            "ESMFold.jl is required for automatic ESM embedding generation. " *
            "Install project deps (including .external/ESMFold.jl) and retry. " *
            "Underlying error: $msg",
        )
    end
end

function _load_model(variant::Symbol)
    src = _resolve_source(variant)
    key = (variant, src.repo_id, src.filename, src.revision, src.local_files_only, src.loader_kind)
    lock(_CACHE_LOCK) do
        if haskey(_MODEL_CACHE, key)
            return _MODEL_CACHE[key]
        end
    end

    ESMFold = _esmfold_module()
    model = if src.loader_kind == :esmfold_embed
        Base.invokelatest(
            ESMFold.load_ESM;
            repo_id = src.repo_id,
            filename = src.filename,
            revision = src.revision,
            local_files_only = src.local_files_only,
        )
    elseif src.loader_kind == :fair_esm2
        _load_fair_esm2_model(src)
    else
        error("Unsupported ESM loader kind: $(src.loader_kind)")
    end

    lock(_CACHE_LOCK) do
        _MODEL_CACHE[key] = model
    end
    return model
end

function hf_hub_download_file(
    repo_id::AbstractString,
    filename::AbstractString;
    revision::AbstractString = "main",
    cache::Bool = true,
    local_files_only::Bool = false,
)
    ESMFold = _esmfold_module()
    return Base.invokelatest(
        ESMFold.hf_hub_download,
        String(repo_id),
        String(filename);
        revision = String(revision),
        cache = cache,
        local_files_only = local_files_only,
    )
end

function _find_num_layers(state::AbstractDict{<:AbstractString, <:Any})
    ids = Int[]
    for key in keys(state)
        startswith(key, "encoder.sentence_encoder.layers.") || continue
        parts = split(String(key), '.')
        length(parts) >= 4 || continue
        idx = tryparse(Int, parts[4])
        idx === nothing || push!(ids, idx)
    end
    isempty(ids) && error("Unable to infer ESM2 layer count from ISM checkpoint keys.")
    return maximum(ids) + 1
end

function _assign_transpose!(dest, src)
    dest .= permutedims(Float32.(src), (2, 1))
    return dest
end

function _assign_direct!(dest, src)
    dest .= Float32.(src)
    return dest
end

function _load_fair_esm2_model(src::NamedTuple)
    ESMFold = _esmfold_module()
    path = hf_hub_download_file(
        src.repo_id,
        src.filename;
        revision = src.revision,
        local_files_only = src.local_files_only,
    )

    state_any = SafeTensors.load_safetensors(path; mmap = true)
    state = Dict{String, Any}(String(k) => v for (k, v) in state_any)

    embed_key = "encoder.sentence_encoder.embed_tokens.weight"
    haskey(state, embed_key) || error("ISM checkpoint missing '$embed_key': $path")
    embed_w = state[embed_key]
    ndims(embed_w) == 2 || error("Unexpected embed_tokens rank in ISM checkpoint: $(ndims(embed_w))")
    embed_dim = Int(size(embed_w, 2))

    inv_key = "encoder.sentence_encoder.layers.0.self_attn.rot_emb.inv_freq"
    haskey(state, inv_key) || error("ISM checkpoint missing '$inv_key': $path")
    head_dim = Int(length(state[inv_key]) * 2)
    attention_heads = embed_dim ÷ head_dim

    num_layers = _find_num_layers(state)
    alphabet = Base.invokelatest(ESMFold.Alphabet_from_architecture, "ESM-1b")
    esm = Base.invokelatest(
        ESMFold.ESM2,
        num_layers,
        embed_dim,
        attention_heads;
        alphabet = alphabet,
        token_dropout = true,
    )

    _assign_transpose!(esm.embed_tokens.weight, state[embed_key])
    _assign_direct!(esm.emb_layer_norm_after.w, state["encoder.sentence_encoder.emb_layer_norm_after.weight"])
    _assign_direct!(esm.emb_layer_norm_after.b, state["encoder.sentence_encoder.emb_layer_norm_after.bias"])

    for i in 0:(num_layers - 1)
        layer = esm.layers[i + 1]
        prefix = "encoder.sentence_encoder.layers.$i"

        _assign_direct!(layer.self_attn.q_proj.weight, state["$prefix.self_attn.q_proj.weight"])
        _assign_direct!(layer.self_attn.k_proj.weight, state["$prefix.self_attn.k_proj.weight"])
        _assign_direct!(layer.self_attn.v_proj.weight, state["$prefix.self_attn.v_proj.weight"])
        _assign_direct!(layer.self_attn.out_proj.weight, state["$prefix.self_attn.out_proj.weight"])

        _assign_direct!(layer.self_attn.q_proj.bias, state["$prefix.self_attn.q_proj.bias"])
        _assign_direct!(layer.self_attn.k_proj.bias, state["$prefix.self_attn.k_proj.bias"])
        _assign_direct!(layer.self_attn.v_proj.bias, state["$prefix.self_attn.v_proj.bias"])
        _assign_direct!(layer.self_attn.out_proj.bias, state["$prefix.self_attn.out_proj.bias"])

        _assign_direct!(layer.self_attn_layer_norm.w, state["$prefix.self_attn_layer_norm.weight"])
        _assign_direct!(layer.self_attn_layer_norm.b, state["$prefix.self_attn_layer_norm.bias"])

        _assign_direct!(layer.fc1.weight, state["$prefix.fc1.weight"])
        _assign_direct!(layer.fc1.bias, state["$prefix.fc1.bias"])
        _assign_direct!(layer.fc2.weight, state["$prefix.fc2.weight"])
        _assign_direct!(layer.fc2.bias, state["$prefix.fc2.bias"])

        _assign_direct!(layer.final_layer_norm.w, state["$prefix.final_layer_norm.weight"])
        _assign_direct!(layer.final_layer_norm.b, state["$prefix.final_layer_norm.bias"])
    end

    if haskey(state, "encoder.lm_head.dense.weight")
        _assign_direct!(esm.lm_head.dense.weight, state["encoder.lm_head.dense.weight"])
        _assign_direct!(esm.lm_head.dense.bias, state["encoder.lm_head.dense.bias"])
        _assign_direct!(esm.lm_head.layer_norm.w, state["encoder.lm_head.layer_norm.weight"])
        _assign_direct!(esm.lm_head.layer_norm.b, state["encoder.lm_head.layer_norm.bias"])
    end
    haskey(state, "encoder.lm_head.bias") &&
        _assign_direct!(esm.lm_head.bias, state["encoder.lm_head.bias"])

    # Keep lm_head weight tied to token embeddings.
    esm.lm_head.weight .= esm.embed_tokens.weight

    return (kind = :fair_esm2, esm = esm, alphabet = alphabet)
end

function _compute_last_layer_embedding_esmfold_embed(model, sequence::AbstractString)
    ESMFold = _esmfold_module()
    seq = uppercase(strip(String(sequence)))
    isempty(seq) && error("Cannot compute ESM embedding for an empty sequence.")

    aa_af2_idx = Base.invokelatest(ESMFold.sequence_to_af2_indices, seq)
    aa_af2 = reshape(Int.(aa_af2_idx), 1, :)
    aa_mask = ones(Int, size(aa_af2))
    esmaa = Base.invokelatest(ESMFold._af2_idx_to_esm_idx, model, aa_af2, aa_mask)

    bosi = model.esm_dict.cls_idx
    eosi = model.esm_dict.eos_idx
    pad = model.esm_dict.padding_idx
    bos = fill(bosi, 1, 1)
    eos = fill(pad, 1, 1)
    esmaa2 = hcat(bos, esmaa, eos)
    lengths = sum(esmaa2 .!= pad, dims = 2)
    positions = reshape(1:size(esmaa2, 2), 1, :)
    eos_mask = positions .== (lengths .+ 1)
    esmaa2 = ifelse.(eos_mask, eosi, esmaa2)

    out = Base.invokelatest(
        (m, toks) -> m.esm(
            toks;
            repr_layers = [m.esm.num_layers],
            need_head_weights = false,
        ),
        model,
        esmaa2,
    )
    rep = out.representations[model.esm.num_layers]
    rep_cpu = Float32.(Array(rep[:, 2:(end - 1), :]))
    size(rep_cpu, 1) == 1 || error("Unexpected ESM output batch size: $(size(rep_cpu, 1)).")
    return dropdims(rep_cpu; dims = 1)
end

function _encode_esm_tokens(alphabet, sequence::AbstractString)
    ESMFold = _esmfold_module()
    toks = Int[]
    if alphabet.prepend_bos
        push!(toks, alphabet.cls_idx)
    end
    get_idx = getfield(ESMFold, :get_idx)
    for aa in sequence
        push!(toks, Base.invokelatest(get_idx, alphabet, string(aa)))
    end
    if alphabet.append_eos
        push!(toks, alphabet.eos_idx)
    end
    return toks
end

function _compute_last_layer_embedding_fair_esm2(model_bundle::NamedTuple, sequence::AbstractString)
    seq = uppercase(strip(String(sequence)))
    isempty(seq) && error("Cannot compute ESM embedding for an empty sequence.")

    alphabet = model_bundle.alphabet
    esm = model_bundle.esm
    toks_vec = _encode_esm_tokens(alphabet, seq)
    toks = reshape(toks_vec, 1, :)
    out = Base.invokelatest(
        esm,
        toks;
        repr_layers = [esm.num_layers],
        need_head_weights = false,
        return_contacts = false,
    )
    rep = out.representations[esm.num_layers]
    start_idx = alphabet.prepend_bos ? 2 : 1
    stop_idx = start_idx + length(seq) - 1
    rep_cpu = Float32.(Array(rep[:, start_idx:stop_idx, :]))
    size(rep_cpu, 1) == 1 || error("Unexpected ESM output batch size: $(size(rep_cpu, 1)).")
    return dropdims(rep_cpu; dims = 1)
end

function _compute_last_layer_embedding(model, sequence::AbstractString)
    if model isa NamedTuple && haskey(model, :kind) && model.kind == :fair_esm2
        return _compute_last_layer_embedding_fair_esm2(model, sequence)
    end
    return _compute_last_layer_embedding_esmfold_embed(model, sequence)
end

function embed_sequence(sequence::AbstractString; variant::Symbol = :esm2_3b)
    seq = uppercase(strip(String(sequence)))
    isempty(seq) && error("ESM sequence must be non-empty.")

    override = _SEQUENCE_EMBEDDER_OVERRIDE[]
    if override !== nothing
        out = override(seq, variant)
        out isa AbstractMatrix || error("ESM embedder override must return a rank-2 matrix.")
        return Float32.(out)
    end

    src = _resolve_source(variant)
    cache_key = (
        variant,
        src.repo_id,
        src.filename,
        src.revision,
        src.local_files_only,
        src.loader_kind,
        seq,
    )
    lock(_CACHE_LOCK) do
        if haskey(_SEQUENCE_CACHE, cache_key)
            return _SEQUENCE_CACHE[cache_key]
        end
    end

    model = _load_model(variant)
    emb = _compute_last_layer_embedding(model, seq)
    size(emb, 1) == length(seq) || error(
        "ESM embedding length mismatch: expected $(length(seq)), got $(size(emb, 1)).",
    )

    lock(_CACHE_LOCK) do
        _SEQUENCE_CACHE[cache_key] = emb
    end
    return emb
end

function embed_sequences(sequences::AbstractVector{<:AbstractString}; variant::Symbol = :esm2_3b)
    return [embed_sequence(seq; variant = variant) for seq in sequences]
end

function clear_esm_cache!()
    lock(_CACHE_LOCK) do
        empty!(_MODEL_CACHE)
        empty!(_SEQUENCE_CACHE)
    end
    return nothing
end

function set_sequence_embedder_override!(f::Union{Nothing, Function})
    lock(_CACHE_LOCK) do
        _SEQUENCE_EMBEDDER_OVERRIDE[] = f
        empty!(_SEQUENCE_CACHE)
    end
    return nothing
end

end # module ESMProvider

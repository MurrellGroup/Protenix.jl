module ProtenixAPI

using Random
using Statistics

import ..Data: AtomRecord, build_feature_bundle_from_atoms, load_structure_atoms
import ..Data.Constants: PROT_STD_RESIDUES_ONE_TO_THREE, STD_RESIDUES_PROTENIX
import ..JSONLite: parse_json, write_json
import ..Model: load_safetensors_weights
import ..Output: dump_prediction_bundle
import ..ProtenixBase
import ..ProtenixMini

export ProtenixModelSpec,
    MODEL_SPECS,
    resolve_model_spec,
    recommended_params,
    default_weights_path,
    predict_json,
    predict_sequence,
    convert_structure_to_infer_json,
    add_precomputed_msa_to_json

struct ProtenixModelSpec
    model_name::String
    family::Symbol
    default_cycle::Int
    default_step::Int
    default_sample::Int
    default_use_msa::Bool
    needs_esm_embedding::Bool
end

const MODEL_SPECS = Dict{String, ProtenixModelSpec}(
    "protenix_base_default_v0.5.0" => ProtenixModelSpec(
        "protenix_base_default_v0.5.0",
        :base,
        10,
        200,
        5,
        true,
        false,
    ),
    "protenix_base_constraint_v0.5.0" => ProtenixModelSpec(
        "protenix_base_constraint_v0.5.0",
        :base,
        10,
        200,
        5,
        true,
        false,
    ),
    "protenix_mini_default_v0.5.0" => ProtenixModelSpec(
        "protenix_mini_default_v0.5.0",
        :mini,
        4,
        5,
        5,
        true,
        false,
    ),
    "protenix_mini_tmpl_v0.5.0" => ProtenixModelSpec(
        "protenix_mini_tmpl_v0.5.0",
        :mini,
        4,
        5,
        5,
        true,
        false,
    ),
    "protenix_tiny_default_v0.5.0" => ProtenixModelSpec(
        "protenix_tiny_default_v0.5.0",
        :mini,
        4,
        5,
        5,
        true,
        false,
    ),
    "protenix_mini_esm_v0.5.0" => ProtenixModelSpec(
        "protenix_mini_esm_v0.5.0",
        :mini,
        4,
        5,
        5,
        false,
        true,
    ),
    "protenix_mini_ism_v0.5.0" => ProtenixModelSpec(
        "protenix_mini_ism_v0.5.0",
        :mini,
        4,
        5,
        5,
        false,
        true,
    ),
)

const _MODEL_DEFAULT_SAFETENSORS_DIR = Dict(
    "protenix_base_default_v0.5.0" => "weights_safetensors_protenix_base_default_v0.5.0",
    "protenix_mini_default_v0.5.0" => "weights_safetensors_protenix_mini_default_v0.5.0",
    "protenix_mini_tmpl_v0.5.0" => "weights_safetensors_protenix_mini_tmpl_v0.5.0",
)

const _AA3_TO_1 = let
    out = Dict{String, String}()
    for (aa1, aa3) in PROT_STD_RESIDUES_ONE_TO_THREE
        aa3 == "xpb" && continue
        out[aa3] = aa1
    end
    out["UNK"] = "X"
    out["MSE"] = "M"
    out
end

const _HHBLITS_AA_TO_ID = Dict{Char, Int}(
    'A' => 0,
    'B' => 2,
    'C' => 1,
    'D' => 2,
    'E' => 3,
    'F' => 4,
    'G' => 5,
    'H' => 6,
    'I' => 7,
    'J' => 20,
    'K' => 8,
    'L' => 9,
    'M' => 10,
    'N' => 11,
    'O' => 20,
    'P' => 12,
    'Q' => 13,
    'R' => 14,
    'S' => 15,
    'T' => 16,
    'U' => 1,
    'V' => 17,
    'W' => 18,
    'X' => 20,
    'Y' => 19,
    'Z' => 3,
    '-' => 21,
)

const _HHBLITS_ID_TO_AA = [
    'A',
    'C',
    'D',
    'E',
    'F',
    'G',
    'H',
    'I',
    'K',
    'L',
    'M',
    'N',
    'P',
    'Q',
    'R',
    'S',
    'T',
    'V',
    'W',
    'Y',
    'X',
    '-',
]

const _HHBLITS_TO_PROTENIX = let
    out = Vector{Int}(undef, length(_HHBLITS_ID_TO_AA))
    for i in eachindex(_HHBLITS_ID_TO_AA)
        aa = _HHBLITS_ID_TO_AA[i]
        if aa == '-'
            out[i] = 31
        else
            aa3 = get(PROT_STD_RESIDUES_ONE_TO_THREE, string(aa), "UNK")
            out[i] = get(STD_RESIDUES_PROTENIX, aa3, STD_RESIDUES_PROTENIX["UNK"])
        end
    end
    out
end

function resolve_model_spec(model_name::AbstractString)
    key = String(model_name)
    haskey(MODEL_SPECS, key) && return MODEL_SPECS[key]
    supported = join(sort(collect(keys(MODEL_SPECS))), ", ")
    error("Unsupported model_name '$key'. Supported models: $supported")
end

function recommended_params(
    model_name::AbstractString;
    use_default_params::Bool = true,
    cycle::Union{Nothing, Int} = nothing,
    step::Union{Nothing, Int} = nothing,
    sample::Union{Nothing, Int} = nothing,
    use_msa::Union{Nothing, Bool} = nothing,
)
    spec = resolve_model_spec(model_name)
    n_cycle = use_default_params ? spec.default_cycle : (cycle === nothing ? spec.default_cycle : cycle)
    n_step = use_default_params ? spec.default_step : (step === nothing ? spec.default_step : step)
    n_sample = sample === nothing ? spec.default_sample : sample
    use_msa_eff = use_default_params ? spec.default_use_msa : (use_msa === nothing ? spec.default_use_msa : use_msa)

    n_cycle > 0 || error("cycle must be positive")
    n_step > 0 || error("step must be positive")
    n_sample > 0 || error("sample must be positive")

    return (
        model_name = spec.model_name,
        family = spec.family,
        cycle = n_cycle,
        step = n_step,
        sample = n_sample,
        use_msa = use_msa_eff,
        needs_esm_embedding = spec.needs_esm_embedding,
    )
end

function default_weights_path(model_name::AbstractString; project_root::AbstractString = normpath(joinpath(@__DIR__, "..")))
    key = String(model_name)
    haskey(_MODEL_DEFAULT_SAFETENSORS_DIR, key) || error(
        "No default safetensors path for model '$key'. Pass weights_path explicitly.",
    )
    return joinpath(project_root, _MODEL_DEFAULT_SAFETENSORS_DIR[key])
end

function _ensure_json_tasks(path::AbstractString)
    value = parse_json(read(path, String))
    if value isa AbstractVector
        return Any[value...]
    elseif value isa AbstractDict
        return Any[value]
    end
    error("Input JSON must be an object or array of objects: $path")
end

function _chain_letters(n::Int)
    n > 0 || error("chain index must be positive")
    chars = Char[]
    x = n
    while x > 0
        x -= 1
        pushfirst!(chars, Char(Int('A') + (x % 26)))
        x = fld(x, 26)
    end
    return String(chars)
end

function _chain_id_from_index(i::Int)
    return _chain_letters(i) * "0"
end

function _as_string_dict(x)
    if x isa AbstractDict
        out = Dict{String, Any}()
        for (k, v) in x
            out[String(k)] = _as_string_dict(v)
        end
        return out
    elseif x isa AbstractVector
        return Any[_as_string_dict(v) for v in x]
    end
    return x
end

function _extract_protein_chain_specs(task::AbstractDict{<:Any, <:Any})
    haskey(task, "sequences") || error("Task is missing required field: sequences")
    sequences = task["sequences"]
    sequences isa AbstractVector || error("Task.sequences must be an array")

    specs = NamedTuple{(:chain_id, :sequence, :msa_cfg)}[]
    chain_idx = 1
    for (i, entity_any) in enumerate(sequences)
        entity_any isa AbstractDict || error("Task.sequences[$i] must be an object")
        entity = entity_any
        if haskey(entity, "proteinChain")
            pc_any = entity["proteinChain"]
            pc_any isa AbstractDict || error("Task.sequences[$i].proteinChain must be an object")
            pc = pc_any
            haskey(pc, "sequence") || error("Task.sequences[$i].proteinChain.sequence is required")
            seq = uppercase(strip(String(pc["sequence"])))
            isempty(seq) && error("Task.sequences[$i].proteinChain.sequence must be non-empty")
            count = Int(get(pc, "count", 1))
            count > 0 || error("Task.sequences[$i].proteinChain.count must be positive")
            msa_cfg = haskey(pc, "msa") && pc["msa"] isa AbstractDict ? _as_string_dict(pc["msa"]) : Dict{String, Any}()
            for _ in 1:count
                chain_id = _chain_id_from_index(chain_idx)
                chain_idx += 1
                push!(specs, (chain_id = chain_id, sequence = seq, msa_cfg = copy(msa_cfg)))
            end
        elseif haskey(entity, "dnaSequence")
            error("dnaSequence is not yet supported in Julia Protenix inference")
        elseif haskey(entity, "rnaSequence")
            error("rnaSequence is not yet supported in Julia Protenix inference")
        elseif haskey(entity, "ligand")
            error("ligand is not yet supported in Julia Protenix inference")
        elseif haskey(entity, "ion")
            error("ion is not yet supported in Julia Protenix inference")
        else
            keys_str = join(string.(collect(keys(entity))), ", ")
            error("Unsupported sequence entry keys at sequences[$i]: [$keys_str]")
        end
    end

    return specs
end

function _build_atoms_from_infer_task(task::AbstractDict{<:Any, <:Any})
    specs = _extract_protein_chain_specs(task)
    atoms = AtomRecord[]
    for spec in specs
        append!(atoms, ProtenixMini.build_sequence_atoms(spec.sequence; chain_id = spec.chain_id))
    end
    isempty(atoms) && error("No supported proteinChain entries found in infer task")
    return atoms
end

function _normalize_protenix_feature_dict!(feat::Dict{String, Any})
    haskey(feat, "restype") || error("Missing restype in feature dict")
    restype_full = Float32.(feat["restype"])
    size(restype_full, 2) >= 32 || error("Expected restype depth >= 32")
    restype = restype_full[:, 1:32]
    n_tok = size(restype, 1)

    restype_idx = Vector{Int}(undef, n_tok)
    @inbounds for i in 1:n_tok
        _, idx = findmax(@view restype[i, :])
        restype_idx[i] = idx - 1
    end

    feat["restype"] = restype
    feat["profile"] = copy(restype)
    feat["msa"] = reshape(restype_idx, 1, n_tok)
    feat["has_deletion"] = zeros(Float32, 1, n_tok)
    feat["deletion_value"] = zeros(Float32, 1, n_tok)
    feat["deletion_mean"] = zeros(Float32, n_tok)
    return feat
end

function _hhblits_idx(c::Char)
    c == '.' && return _HHBLITS_AA_TO_ID['-']
    return get(_HHBLITS_AA_TO_ID, uppercase(c), _HHBLITS_AA_TO_ID['X'])
end

function _sequence_to_protenix_indices(sequence::AbstractString)
    seq = uppercase(strip(sequence))
    idx = Int[]
    for c in seq
        isspace(c) && continue
        aa3 = get(PROT_STD_RESIDUES_ONE_TO_THREE, string(c), "UNK")
        push!(idx, get(STD_RESIDUES_PROTENIX, aa3, STD_RESIDUES_PROTENIX["UNK"]))
    end
    return idx
end

function _parse_a3m(path::AbstractString; seq_limit::Int = -1)
    isfile(path) || error("MSA file not found: $path")
    sequences = String[]
    descriptions = String[]
    idx = 0
    for line in eachline(path)
        ln = strip(line)
        isempty(ln) && continue
        startswith(ln, "#") && continue
        if startswith(ln, ">")
            if seq_limit > 0 && length(sequences) > seq_limit
                break
            end
            idx += 1
            push!(descriptions, ln[2:end])
            push!(sequences, "")
            continue
        end
        idx > 0 || continue
        sequences[idx] *= ln
    end
    return sequences, descriptions
end

function _aligned_and_deletions_from_a3m(sequences::Vector{String})
    aligned = String[]
    deletion_matrix = Vector{Vector{Float32}}()
    for raw in sequences
        deletion_vec = Float32[]
        deletion_count = 0
        io = IOBuffer()
        for c in raw
            if islowercase(c)
                deletion_count += 1
            else
                push!(deletion_vec, Float32(deletion_count))
                deletion_count = 0
                uc = c == '.' ? '-' : uppercase(c)
                print(io, uc)
            end
        end
        aln = String(take!(io))
        isempty(aln) && continue
        length(aln) == length(deletion_vec) || error("Invalid A3M row: aligned/deletion length mismatch.")
        push!(aligned, aln)
        push!(deletion_matrix, deletion_vec)
    end
    return aligned, deletion_matrix
end

function _build_chain_msa_features(
    sequence::AbstractString,
    aligned::Vector{String},
    deletion_matrix::Vector{Vector{Float32}},
)
    if isempty(aligned)
        query = uppercase(strip(sequence))
        push!(aligned, query)
        push!(deletion_matrix, zeros(Float32, length(query)))
    end

    seen = Set{String}()
    dedup_aligned = String[]
    dedup_del = Vector{Vector{Float32}}()
    for (seq, del) in zip(aligned, deletion_matrix)
        seq in seen && continue
        push!(seen, seq)
        push!(dedup_aligned, seq)
        push!(dedup_del, del)
    end

    n_row = length(dedup_aligned)
    n_col = length(dedup_aligned[1])
    n_row > 0 || error("MSA has zero rows after deduplication.")
    all(length(s) == n_col for s in dedup_aligned) || error("MSA rows must have uniform length.")
    all(length(d) == n_col for d in dedup_del) || error("Deletion rows must have uniform length.")

    msa_hh = Matrix{Int}(undef, n_row, n_col)
    deletion_mat = Matrix{Float32}(undef, n_row, n_col)
    for i in 1:n_row
        seq = dedup_aligned[i]
        del = dedup_del[i]
        for j in 1:n_col
            msa_hh[i, j] = _hhblits_idx(seq[j])
            deletion_mat[i, j] = del[j]
        end
    end

    profile22 = zeros(Float32, n_col, 22)
    @inbounds for i in 1:n_row, j in 1:n_col
        profile22[j, msa_hh[i, j] + 1] += 1f0
    end
    profile22 ./= Float32(n_row)

    msa = _HHBLITS_TO_PROTENIX[msa_hh .+ 1]
    profile = zeros(Float32, n_col, 32)
    for hh in 1:22
        dst = _HHBLITS_TO_PROTENIX[hh] + 1
        @views profile[:, dst] .+= profile22[:, hh]
    end

    has_deletion = clamp.(deletion_mat, 0f0, 1f0)
    deletion_value = (2f0 / Float32(pi)) .* atan.(deletion_mat ./ 3f0)
    deletion_mean = vec(mean(deletion_mat; dims = 1))

    return (
        msa = Int.(msa),
        has_deletion = Float32.(has_deletion),
        deletion_value = Float32.(deletion_value),
        deletion_mean = Float32.(deletion_mean),
        profile = Float32.(profile),
    )
end

function _chain_msa_features(
    sequence::AbstractString,
    msa_cfg::AbstractDict{<:AbstractString, <:Any},
    json_dir::AbstractString;
    require_pairing::Bool,
)
    msa_dir_any = get(msa_cfg, "precomputed_msa_dir", nothing)
    if msa_dir_any === nothing || isempty(strip(String(msa_dir_any)))
        query_idx = _sequence_to_protenix_indices(sequence)
        profile = zeros(Float32, length(query_idx), 32)
        for (i, x) in enumerate(query_idx)
            profile[i, x + 1] = 1f0
        end
        return (
            msa = reshape(query_idx, 1, :),
            has_deletion = zeros(Float32, 1, length(query_idx)),
            deletion_value = zeros(Float32, 1, length(query_idx)),
            deletion_mean = zeros(Float32, length(query_idx)),
            profile = profile,
        )
    end

    msa_dir_raw = String(msa_dir_any)
    msa_dir = isabspath(msa_dir_raw) ? msa_dir_raw : normpath(joinpath(json_dir, msa_dir_raw))
    isdir(msa_dir) || error("The provided precomputed_msa_dir does not exist: $msa_dir")

    aligned = String[]
    deletion_matrix = Vector{Vector{Float32}}()

    non_pair_path = joinpath(msa_dir, "non_pairing.a3m")
    if isfile(non_pair_path)
        seqs, _ = _parse_a3m(non_pair_path; seq_limit = -1)
        aln, del = _aligned_and_deletions_from_a3m(seqs)
        append!(aligned, aln)
        append!(deletion_matrix, del)
    end

    if require_pairing
        pair_path = joinpath(msa_dir, "pairing.a3m")
        isfile(pair_path) || error("No pairing-MSA found at $pair_path for multi-chain assembly.")
        seqs, _ = _parse_a3m(pair_path; seq_limit = -1)
        aln, del = _aligned_and_deletions_from_a3m(seqs)
        append!(aligned, aln)
        append!(deletion_matrix, del)
    end

    return _build_chain_msa_features(sequence, aligned, deletion_matrix)
end

function _inject_task_msa_features!(
    feat::Dict{String, Any},
    task::AbstractDict{<:Any, <:Any},
    json_path::AbstractString;
    use_msa::Bool,
)
    use_msa || return feat
    chain_specs = _extract_protein_chain_specs(task)
    isempty(chain_specs) && return feat

    restype = Float32.(feat["restype"])
    n_tok = size(restype, 1)
    restype_idx = Vector{Int}(undef, n_tok)
    @inbounds for i in 1:n_tok
        _, idx = findmax(@view restype[i, :])
        restype_idx[i] = idx - 1
    end

    asym = Int.(feat["asym_id"])
    chain_token_cols = [findall(==(i - 1), asym) for i in 1:length(chain_specs)]
    all(!isempty(cols) for cols in chain_token_cols) || error("Failed to map task chains to token columns.")

    chain_sequences = [s.sequence for s in chain_specs]
    is_homomer_or_monomer = length(Set(chain_sequences)) == 1
    json_dir = dirname(abspath(json_path))
    chain_features = NamedTuple[]
    for spec in chain_specs
        push!(
            chain_features,
            _chain_msa_features(
                spec.sequence,
                spec.msa_cfg,
                json_dir;
                require_pairing = !is_homomer_or_monomer,
            ),
        )
    end

    for (i, cols) in enumerate(chain_token_cols)
        size(chain_features[i].msa, 2) == length(cols) || error(
            "MSA/token length mismatch on chain $(chain_specs[i].chain_id): MSA has $(size(chain_features[i].msa, 2)) columns, tokens have $(length(cols)).",
        )
    end

    total_rows = sum(size(cf.msa, 1) for cf in chain_features)
    total_rows > 0 || (total_rows = 1)

    msa = repeat(reshape(restype_idx, 1, :), total_rows, 1)
    has_deletion = zeros(Float32, total_rows, n_tok)
    deletion_value = zeros(Float32, total_rows, n_tok)
    profile = copy(restype)
    deletion_mean = zeros(Float32, n_tok)

    row_start = 1
    for (cf, cols) in zip(chain_features, chain_token_cols)
        n_row = size(cf.msa, 1)
        row_stop = row_start + n_row - 1
        rows = row_start:row_stop
        msa[rows, cols] .= cf.msa
        has_deletion[rows, cols] .= cf.has_deletion
        deletion_value[rows, cols] .= cf.deletion_value
        profile[cols, :] .= cf.profile
        deletion_mean[cols] .= cf.deletion_mean
        row_start = row_stop + 1
    end

    feat["msa"] = msa
    feat["has_deletion"] = has_deletion
    feat["deletion_value"] = deletion_value
    feat["deletion_mean"] = deletion_mean
    feat["profile"] = profile
    return feat
end

function _nested_shape(x)
    x isa AbstractArray || return ()
    x isa AbstractVector || return size(x)
    if isempty(x)
        return (0,)
    end
    s0 = _nested_shape(x[1])
    for i in 2:length(x)
        _nested_shape(x[i]) == s0 || error("Template feature arrays must be rectangular.")
    end
    return (length(x), s0...)
end

function _flatten_nested!(dst::Vector{Any}, x)
    if x isa AbstractArray
        for y in x
            _flatten_nested!(dst, y)
        end
    else
        push!(dst, x)
    end
    return dst
end

function _to_dense_array(x, ::Type{T}) where {T}
    x isa AbstractArray || error("Expected array-like template feature, got $(typeof(x)).")
    if !(x isa AbstractVector)
        return T.(x)
    end
    shape = _nested_shape(x)
    flat = Any[]
    _flatten_nested!(flat, x)
    isempty(shape) && return T.(flat)
    return reshape(T.(flat), shape...)
end

function _to_int_array(x)
    return _to_dense_array(x, Int)
end

function _to_float_array(x)
    return _to_dense_array(x, Float32)
end

function _inject_task_template_features!(
    feat::Dict{String, Any},
    task::AbstractDict{<:Any, <:Any},
)
    haskey(task, "template_features") || return feat
    tf_any = task["template_features"]
    tf_any isa AbstractDict || error("task.template_features must be an object when provided.")
    tf = _as_string_dict(tf_any)

    required = ("template_restype", "template_all_atom_mask", "template_all_atom_positions")
    all(haskey(tf, k) for k in required) || error(
        "template_features must include template_restype/template_all_atom_mask/template_all_atom_positions.",
    )

    n_tok = size(feat["restype"], 1)
    template_restype = _to_int_array(tf["template_restype"])
    template_mask = _to_float_array(tf["template_all_atom_mask"])
    template_pos = _to_float_array(tf["template_all_atom_positions"])

    ndims(template_restype) == 2 || error("template_restype must be rank-2 [N_template, N_token].")
    ndims(template_mask) == 3 || error("template_all_atom_mask must be rank-3 [N_template, N_token, 37].")
    ndims(template_pos) == 4 || error("template_all_atom_positions must be rank-4 [N_template, N_token, 37, 3].")

    size(template_restype, 2) == n_tok || error("template_restype token length mismatch.")
    size(template_mask, 2) == n_tok || error("template_all_atom_mask token length mismatch.")
    size(template_pos, 2) == n_tok || error("template_all_atom_positions token length mismatch.")
    size(template_mask, 3) == 37 || error("template_all_atom_mask must have 37 atom slots.")
    size(template_pos, 3) == 37 || error("template_all_atom_positions must have 37 atom slots.")
    size(template_pos, 4) == 3 || error("template_all_atom_positions final dimension must be xyz=3.")
    size(template_restype, 1) == size(template_mask, 1) == size(template_pos, 1) || error(
        "template feature N_template dimensions must match.",
    )

    feat["template_restype"] = template_restype
    feat["template_all_atom_mask"] = template_mask
    feat["template_all_atom_positions"] = template_pos
    return feat
end

function _protein_chain_sequences(atoms::Vector{AtomRecord})
    chain_ids = String[]
    seen_chain = Set{String}()
    for a in atoms
        if a.mol_type == "protein" && !(a.chain_id in seen_chain)
            push!(chain_ids, a.chain_id)
            push!(seen_chain, a.chain_id)
        end
    end

    out = NamedTuple{(:chain_id, :sequence)}[]
    for chain_id in chain_ids
        residues = NamedTuple{(:res_id, :res_name)}[]
        seen_res = Set{Int}()
        for a in atoms
            a.chain_id == chain_id || continue
            a.mol_type == "protein" || continue
            if !(a.res_id in seen_res)
                push!(residues, (res_id = a.res_id, res_name = a.res_name))
                push!(seen_res, a.res_id)
            end
        end
        sort!(residues; by = r -> r.res_id)
        seq = IOBuffer()
        for r in residues
            print(seq, get(_AA3_TO_1, uppercase(r.res_name), "X"))
        end
        sequence = String(take!(seq))
        isempty(sequence) && continue
        push!(out, (chain_id = chain_id, sequence = sequence))
    end
    return out
end

function _collect_input_paths(input::AbstractString; exts::Tuple{Vararg{String}})
    path = abspath(input)
    if isfile(path)
        ext = lowercase(splitext(path)[2])
        ext in exts || error("Unsupported input extension '$ext' for $path")
        return [path]
    elseif isdir(path)
        files = String[]
        for (root, _, names) in walkdir(path)
            for name in names
                f = joinpath(root, name)
                ext = lowercase(splitext(f)[2])
                ext in exts || continue
                push!(files, f)
            end
        end
        sort!(files)
        isempty(files) && error("No files with extensions $(join(exts, ", ")) found in directory: $path")
        return files
    end
    error("Input path does not exist: $path")
end

function _next_available_json_path(out_dir::AbstractString, stem::AbstractString)
    p0 = joinpath(out_dir, "$(stem).json")
    !isfile(p0) && return p0
    i = 1
    while true
        p = joinpath(out_dir, "$(stem)_$(i).json")
        !isfile(p) && return p
        i += 1
    end
end

function _load_model(model_name::AbstractString, weights_path::AbstractString; strict::Bool = true)
    params = recommended_params(model_name)
    params.needs_esm_embedding && error(
        "Model $(model_name) needs ESM2/ISM embeddings; this Julia path does not yet inject them.",
    )

    w = load_safetensors_weights(weights_path)
    if params.family == :mini
        m = ProtenixMini.build_protenix_mini_model(w)
        ProtenixMini.load_protenix_mini_model!(m, w; strict = strict)
        return (model = m, family = :mini)
    elseif params.family == :base
        m = ProtenixBase.build_protenix_base_model(w)
        ProtenixBase.load_protenix_base_model!(m, w; strict = strict)
        return (model = m, family = :base)
    end

    error("Unsupported model family: $(params.family)")
end

function _run_model(
    loaded,
    feature_dict::AbstractDict{<:AbstractString, <:Any};
    cycle::Int,
    step::Int,
    sample::Int,
    rng::AbstractRNG,
)
    if loaded.family == :mini
        return ProtenixMini.run_inference(
            loaded.model,
            feature_dict;
            n_cycle = cycle,
            n_step = step,
            n_sample = sample,
            rng = rng,
        )
    elseif loaded.family == :base
        return ProtenixBase.run_inference(
            loaded.model,
            feature_dict;
            n_cycle = cycle,
            n_step = step,
            n_sample = sample,
            rng = rng,
        )
    end
    error("Unsupported model family: $(loaded.family)")
end

function _softmax(v::AbstractVector{<:Real})
    m = maximum(v)
    ex = exp.(Float64.(v) .- Float64(m))
    s = sum(ex)
    return ex ./ s
end

function _confidence_proxy(logits::AbstractArray{<:Real, 2})
    size(logits, 1) > 0 || return 0.0
    acc = 0.0
    for i in 1:size(logits, 1)
        p = _softmax(@view logits[i, :])
        acc += maximum(p)
    end
    return acc / size(logits, 1)
end

function _write_confidence_summaries(
    pred_dir::AbstractString,
    task_name::AbstractString,
    seed::Int,
    pred,
)
    n_sample = size(pred.coordinate, 1)
    for sample_idx in 1:n_sample
        plddt_i = Array{Float32, 2}(pred.plddt[sample_idx, :, :])
        pae_i = Array{Float32, 3}(pred.pae[sample_idx, :, :, :])
        pde_i = Array{Float32, 3}(pred.pde[sample_idx, :, :, :])
        resolved_i = Array{Float32, 2}(pred.resolved[sample_idx, :, :])

        summary = Dict{String, Any}(
            "model_output" => "julia_protenix",
            "sample_name" => String(task_name),
            "seed" => seed,
            "sample_idx" => sample_idx - 1,
            "plddt_logits_shape" => [size(plddt_i, 1), size(plddt_i, 2)],
            "pae_logits_shape" => [size(pae_i, 1), size(pae_i, 2), size(pae_i, 3)],
            "pde_logits_shape" => [size(pde_i, 1), size(pde_i, 2), size(pde_i, 3)],
            "resolved_logits_shape" => [size(resolved_i, 1), size(resolved_i, 2)],
            "plddt_logits_maxprob_mean" => _confidence_proxy(plddt_i),
            "resolved_logits_maxprob_mean" => _confidence_proxy(resolved_i),
        )

        summary_path = joinpath(
            pred_dir,
            "$(task_name)_$(seed)_summary_confidence_sample_$(sample_idx - 1).json",
        )
        write_json(summary_path, summary)
    end
    return nothing
end

function _default_task_name(input_path::AbstractString)
    return splitext(basename(input_path))[1]
end

function predict_json(
    input::AbstractString;
    out_dir::AbstractString = "./output",
    model_name::String = "protenix_base_default_v0.5.0",
    weights_path::AbstractString = "",
    seeds::Vector{Int} = [101],
    use_default_params::Bool = true,
    cycle::Union{Nothing, Int} = nothing,
    step::Union{Nothing, Int} = nothing,
    sample::Union{Nothing, Int} = nothing,
    use_msa::Union{Nothing, Bool} = nothing,
    strict::Bool = true,
)
    params = recommended_params(
        model_name;
        use_default_params = use_default_params,
        cycle = cycle,
        step = step,
        sample = sample,
        use_msa = use_msa,
    )
    params.needs_esm_embedding && error(
        "Model $(model_name) needs ESM2/ISM embeddings; this Julia path does not yet inject them.",
    )

    mkpath(out_dir)
    length(seeds) > 0 || error("seeds must be non-empty")
    all(s -> s >= 0, seeds) || error("seeds must be non-negative")

    local_weights = isempty(weights_path) ? default_weights_path(model_name) : abspath(weights_path)
    isdir(local_weights) || error("weights_path does not exist: $local_weights")

    loaded = _load_model(model_name, local_weights; strict = strict)
    json_paths = _collect_input_paths(input; exts = (".json",))
    records = NamedTuple[]

    for json_path in json_paths
        tasks = _ensure_json_tasks(json_path)
        for (task_idx, task_any) in enumerate(tasks)
            task_any isa AbstractDict || error("Task $(task_idx) in $json_path is not an object")
            task = task_any
            task_name = haskey(task, "name") ? String(task["name"]) : "$(_default_task_name(json_path))_$(task_idx - 1)"
            atoms = _build_atoms_from_infer_task(task)

            for seed in seeds
                rng = MersenneTwister(seed)
                bundle = build_feature_bundle_from_atoms(atoms; task_name = task_name, rng = rng)
                _normalize_protenix_feature_dict!(bundle["input_feature_dict"])
                _inject_task_msa_features!(
                    bundle["input_feature_dict"],
                    task,
                    json_path;
                    use_msa = params.use_msa,
                )
                _inject_task_template_features!(bundle["input_feature_dict"], task)
                pred = _run_model(
                    loaded,
                    bundle["input_feature_dict"];
                    cycle = params.cycle,
                    step = params.step,
                    sample = params.sample,
                    rng = rng,
                )

                task_dump_dir = joinpath(out_dir, task_name, "seed_$(seed)")
                pred_dir = dump_prediction_bundle(task_dump_dir, task_name, bundle["atoms"], pred.coordinate)
                _write_confidence_summaries(pred_dir, task_name, seed, pred)
                cif_paths = sort(filter(endswith(".cif"), readdir(pred_dir; join = true)))

                push!(
                    records,
                    (
                        input_json = json_path,
                        task_name = task_name,
                        seed = seed,
                        prediction_dir = pred_dir,
                        cif_paths = cif_paths,
                    ),
                )
            end
        end
    end

    return records
end

function predict_sequence(
    sequence::AbstractString;
    out_dir::AbstractString = "./output",
    model_name::String = "protenix_base_default_v0.5.0",
    weights_path::AbstractString = "",
    task_name::String = "protenix_sequence",
    chain_id::String = "A0",
    seeds::Vector{Int} = [101],
    use_default_params::Bool = true,
    cycle::Union{Nothing, Int} = nothing,
    step::Union{Nothing, Int} = nothing,
    sample::Union{Nothing, Int} = nothing,
    use_msa::Union{Nothing, Bool} = nothing,
    strict::Bool = true,
)
    seq = uppercase(strip(sequence))
    isempty(seq) && error("sequence must be non-empty")

    params = recommended_params(
        model_name;
        use_default_params = use_default_params,
        cycle = cycle,
        step = step,
        sample = sample,
        use_msa = use_msa,
    )
    params.needs_esm_embedding && error(
        "Model $(model_name) needs ESM2/ISM embeddings; this Julia path does not yet inject them.",
    )

    mkpath(out_dir)
    local_weights = isempty(weights_path) ? default_weights_path(model_name) : abspath(weights_path)
    isdir(local_weights) || error("weights_path does not exist: $local_weights")

    loaded = _load_model(model_name, local_weights; strict = strict)
    records = NamedTuple[]

    for seed in seeds
        rng = MersenneTwister(seed)
        atoms = ProtenixMini.build_sequence_atoms(seq; chain_id = chain_id)
        bundle = build_feature_bundle_from_atoms(atoms; task_name = task_name, rng = rng)
        _normalize_protenix_feature_dict!(bundle["input_feature_dict"])
        pred = _run_model(
            loaded,
            bundle["input_feature_dict"];
            cycle = params.cycle,
            step = params.step,
            sample = params.sample,
            rng = rng,
        )

        task_dump_dir = joinpath(out_dir, task_name, "seed_$(seed)")
        pred_dir = dump_prediction_bundle(task_dump_dir, task_name, bundle["atoms"], pred.coordinate)
        _write_confidence_summaries(pred_dir, task_name, seed, pred)
        cif_paths = sort(filter(endswith(".cif"), readdir(pred_dir; join = true)))

        push!(
            records,
            (
                task_name = task_name,
                seed = seed,
                prediction_dir = pred_dir,
                cif_paths = cif_paths,
            ),
        )
    end

    return records
end

function convert_structure_to_infer_json(
    input::AbstractString;
    out_dir::AbstractString = "./output",
    altloc::String = "first",
    assembly_id::Union{Nothing, String} = nothing,
)
    lowercase(strip(altloc)) == "first" || error("Only altloc='first' is currently supported.")
    assembly_id === nothing || error("assembly_id expansion is not yet supported in Julia tojson.")

    mkpath(out_dir)
    paths = _collect_input_paths(input; exts = (".pdb", ".cif", ".mmcif"))
    out_paths = String[]

    for p in paths
        atoms = load_structure_atoms(p)
        chains = _protein_chain_sequences(atoms)
        isempty(chains) && error("No protein chains parsed from structure: $p")

        sequences = Any[]
        for chain in chains
            push!(
                sequences,
                Dict(
                    "proteinChain" => Dict(
                        "sequence" => chain.sequence,
                        "count" => 1,
                    ),
                ),
            )
        end

        payload = Any[
            Dict(
                "name" => _default_task_name(p),
                "sequences" => sequences,
            ),
        ]

        out_path = _next_available_json_path(out_dir, _default_task_name(p))
        write_json(out_path, payload)
        push!(out_paths, out_path)
    end

    return out_paths
end

function add_precomputed_msa_to_json(
    input_json::AbstractString;
    out_dir::AbstractString = "./output",
    precomputed_msa_dir::AbstractString,
    pairing_db::String = "uniref100",
)
    json_path = abspath(input_json)
    isfile(json_path) || error("input_json not found: $json_path")

    tasks = _ensure_json_tasks(json_path)
    for (task_idx, task_any) in enumerate(tasks)
        task_any isa AbstractDict || error("Task $(task_idx) in $json_path is not an object")
        task = task_any
        haskey(task, "sequences") || error("Task $(task_idx) is missing sequences")
        seqs = task["sequences"]
        seqs isa AbstractVector || error("Task $(task_idx).sequences must be an array")
        for (i, entity_any) in enumerate(seqs)
            entity_any isa AbstractDict || error("Task $(task_idx).sequences[$i] must be an object")
            entity = entity_any
            haskey(entity, "proteinChain") || continue
            pc_any = entity["proteinChain"]
            pc_any isa AbstractDict || error("Task $(task_idx).sequences[$i].proteinChain must be an object")
            pc = pc_any
            pc["msa"] = Dict(
                "precomputed_msa_dir" => String(precomputed_msa_dir),
                "pairing_db" => pairing_db,
            )
        end
    end

    mkpath(out_dir)
    stem = splitext(basename(json_path))[1] * "_with_msa"
    out_path = _next_available_json_path(out_dir, stem)
    write_json(out_path, tasks)
    return out_path
end

end

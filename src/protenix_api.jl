module ProtenixAPI

using Random
using Statistics

import ..Data: AtomRecord, build_feature_bundle_from_atoms, load_structure_atoms
import ..Data.Constants: PROT_STD_RESIDUES_ONE_TO_THREE, PROTEIN_HEAVY_ATOMS, STD_RESIDUES_PROTENIX
import ..JSONLite: parse_json, write_json
import ..Model: load_safetensors_weights
import ..Output: dump_prediction_bundle
import ..ProtenixBase
import ..ProtenixMini

export ProtenixModelSpec,
    ProtenixPredictOptions,
    ProtenixSequenceOptions,
    MODEL_SPECS,
    resolve_model_spec,
    recommended_params,
    list_supported_models,
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

"""
Shared predict options for JSON/sequence inference.

Use this struct to keep runtime options typed and reusable across call-sites.
"""
struct ProtenixPredictOptions
    out_dir::String
    model_name::String
    weights_path::String
    seeds::Vector{Int}
    use_default_params::Bool
    cycle::Union{Nothing, Int}
    step::Union{Nothing, Int}
    sample::Union{Nothing, Int}
    use_msa::Union{Nothing, Bool}
    strict::Bool
end

function ProtenixPredictOptions(;
    out_dir::AbstractString = "./output",
    model_name::AbstractString = "protenix_base_default_v0.5.0",
    weights_path::AbstractString = "",
    seeds::AbstractVector{<:Integer} = [101],
    use_default_params::Bool = true,
    cycle::Union{Nothing, Integer} = nothing,
    step::Union{Nothing, Integer} = nothing,
    sample::Union{Nothing, Integer} = nothing,
    use_msa::Union{Nothing, Bool} = nothing,
    strict::Bool = true,
)
    s = Int.(collect(seeds))
    isempty(s) && error("seeds must be non-empty")
    all(x -> x >= 0, s) || error("seeds must be non-negative")
    cycle !== nothing && cycle <= 0 && error("cycle must be positive")
    step !== nothing && step <= 0 && error("step must be positive")
    sample !== nothing && sample <= 0 && error("sample must be positive")
    return ProtenixPredictOptions(
        String(out_dir),
        String(model_name),
        String(weights_path),
        s,
        use_default_params,
        cycle === nothing ? nothing : Int(cycle),
        step === nothing ? nothing : Int(step),
        sample === nothing ? nothing : Int(sample),
        use_msa,
        strict,
    )
end

"""
Sequence-only options layered on top of `ProtenixPredictOptions`.
"""
struct ProtenixSequenceOptions
    common::ProtenixPredictOptions
    task_name::String
    chain_id::String
    esm_token_embedding::Union{Nothing, Matrix{Float32}}
end

function ProtenixSequenceOptions(;
    common::ProtenixPredictOptions = ProtenixPredictOptions(),
    task_name::AbstractString = "protenix_sequence",
    chain_id::AbstractString = "A0",
    esm_token_embedding::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
)
    emb = esm_token_embedding === nothing ? nothing : Float32.(esm_token_embedding)
    return ProtenixSequenceOptions(common, String(task_name), String(chain_id), emb)
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

const _DNA_1TO3 = Dict{Char, String}(
    'A' => "DA",
    'G' => "DG",
    'C' => "DC",
    'T' => "DT",
    'X' => "DN",
    'I' => "DI",
    'N' => "DN",
    'U' => "DU",
)

const _RNA_1TO3 = Dict{Char, String}(
    'A' => "A",
    'G' => "G",
    'C' => "C",
    'U' => "U",
    'X' => "N",
    'I' => "I",
    'N' => "N",
)

const _DNA_BACKBONE_ATOMS = ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"]
const _RNA_BACKBONE_ATOMS = ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"]

const _DNA_BASE_ATOMS = Dict{String, Vector{String}}(
    "DA" => ["N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
    "DG" => ["N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
    "DC" => ["N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
    "DT" => ["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C7", "C6"],
    "DN" => ["N1", "C2", "N3", "C4", "C5", "C6"],
)

const _RNA_BASE_ATOMS = Dict{String, Vector{String}}(
    "A" => ["N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
    "G" => ["N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
    "C" => ["N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
    "U" => ["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"],
    "N" => ["N1", "C2", "N3", "C4", "C5", "C6"],
)

const _CCD_COMPONENT_CACHE = Dict{
    String,
    Vector{NamedTuple{(:atom_name, :element, :x, :y, :z, :charge, :has_coord), Tuple{String, String, Float32, Float32, Float32, Float32, Bool}}},
}()

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

"""
Return sorted model metadata for discoverability in CLIs/UIs.
"""
function list_supported_models()
    names = sort!(collect(keys(MODEL_SPECS)))
    return NamedTuple[
        (
            model_name = n,
            family = MODEL_SPECS[n].family,
            default_cycle = MODEL_SPECS[n].default_cycle,
            default_step = MODEL_SPECS[n].default_step,
            default_sample = MODEL_SPECS[n].default_sample,
            default_use_msa = MODEL_SPECS[n].default_use_msa,
            needs_esm_embedding = MODEL_SPECS[n].needs_esm_embedding,
        ) for n in names
    ]
end

function default_weights_path(model_name::AbstractString; project_root::AbstractString = normpath(joinpath(@__DIR__, "..")))
    key = String(model_name)
    direct = joinpath(project_root, "weights_safetensors_" * key)
    isdir(direct) && return direct
    if haskey(_MODEL_DEFAULT_SAFETENSORS_DIR, key)
        mapped = joinpath(project_root, _MODEL_DEFAULT_SAFETENSORS_DIR[key])
        isdir(mapped) && return mapped
    end
    error(
        "No default safetensors path found for model '$key'. " *
        "Looked for: $(abspath(direct)). Pass weights_path explicitly or convert/checkpoint this model first.",
    )
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

struct ProteinChainSpec
    chain_id::String
    sequence::String
    msa_cfg::Dict{String, Any}
end

struct TaskEntityParseResult
    atoms::Vector{AtomRecord}
    protein_specs::Vector{ProteinChainSpec}
    entity_chain_ids::Dict{Int, Vector{String}}
    entity_atom_map::Dict{Int, Dict{Int, String}}
end

function _split_cif_tokens(line::AbstractString)
    tokens = String[]
    n = lastindex(line)
    i = firstindex(line)
    while i <= n
        while i <= n && isspace(line[i])
            i = nextind(line, i)
        end
        i > n && break
        c = line[i]
        if c == '\'' || c == '"'
            quote_char = c
            i = nextind(line, i)
            start = i
            while i <= n && line[i] != quote_char
                i = nextind(line, i)
            end
            if i <= n
                stop = prevind(line, i)
                push!(tokens, start <= stop ? String(line[start:stop]) : "")
                i = nextind(line, i)
            else
                push!(tokens, String(line[start:n]))
            end
        else
            start = i
            while i <= n && !isspace(line[i])
                i = nextind(line, i)
            end
            stop = prevind(line, i)
            push!(tokens, String(line[start:stop]))
        end
    end
    return tokens
end

function _try_parse_f32(x)
    s = strip(String(x))
    s == "?" && return nothing
    s == "." && return nothing
    return tryparse(Float32, s)
end

function _try_parse_i(x)
    s = strip(String(x))
    s == "?" && return nothing
    s == "." && return nothing
    return tryparse(Int, s)
end

function _default_ccd_components_path()
    if haskey(ENV, "PROTENIX_DATA_ROOT_DIR")
        root = ENV["PROTENIX_DATA_ROOT_DIR"]
        p1 = joinpath(root, "components.v20240608.cif")
        isfile(p1) && return p1
        p2 = joinpath(root, "components.cif")
        isfile(p2) && return p2
    end
    project_root = normpath(joinpath(@__DIR__, ".."))
    p1 = joinpath(project_root, "release_data", "ccd_cache", "components.v20240608.cif")
    isfile(p1) && return p1
    p2 = joinpath(project_root, "release_data", "ccd_cache", "components.cif")
    isfile(p2) && return p2
    return ""
end

function _infer_element_from_atom_name(atom_name::AbstractString)
    for c in uppercase(strip(String(atom_name)))
        isletter(c) && return string(c)
    end
    return "C"
end

function _pseudo_atom_xyz(res_idx::Int, atom_name::AbstractString)
    x0 = Float32((res_idx - 1) * 3.8)
    if atom_name == "N"
        return (x0 - 1.2f0, 0f0, 0f0)
    elseif atom_name == "CA"
        return (x0, 0f0, 0f0)
    elseif atom_name == "C"
        return (x0 + 1.4f0, 0.1f0, 0f0)
    elseif atom_name == "O"
        return (x0 + 2.2f0, -0.4f0, 0f0)
    elseif atom_name == "OXT"
        return (x0 + 2.4f0, 0.5f0, 0f0)
    end
    h = abs(hash(atom_name))
    dx = 0.8f0 + 0.1f0 * Float32(h % 11)
    dy = -0.9f0 + 0.3f0 * Float32((h ÷ 11) % 7)
    dz = -0.9f0 + 0.3f0 * Float32((h ÷ 77) % 7)
    return (x0 + dx, dy, dz)
end

function _ensure_ccd_component_entries!(codes::Set{String})
    needed = Set{String}()
    for c in codes
        uc = uppercase(strip(c))
        isempty(uc) && continue
        haskey(_CCD_COMPONENT_CACHE, uc) || push!(needed, uc)
    end
    isempty(needed) && return

    ccd_path = _default_ccd_components_path()
    isempty(ccd_path) && return

    current_code = ""
    active = false
    pending = nothing

    open(ccd_path, "r") do io
        while true
            line = if pending === nothing
                eof(io) ? nothing : readline(io)
            else
                x = pending
                pending = nothing
                x
            end
            line === nothing && break
            s = strip(line)
            isempty(s) && continue

            if startswith(s, "data_")
                current_code = uppercase(String(s[6:end]))
                active = current_code in needed
                continue
            end
            active || continue

            if s == "loop_"
                headers = String[]
                while !eof(io)
                    h = strip(readline(io))
                    startswith(h, "_") || (pending = h; break)
                    push!(headers, h)
                end
                isempty(headers) && continue
                all(startswith(h, "_chem_comp_atom.") for h in headers) || continue

                findidx(name) = findfirst(==(name), headers)
                idx_atom = findidx("_chem_comp_atom.atom_id")
                idx_type = findidx("_chem_comp_atom.type_symbol")
                idx_charge = findidx("_chem_comp_atom.charge")
                idx_x = findidx("_chem_comp_atom.model_Cartn_x")
                idx_y = findidx("_chem_comp_atom.model_Cartn_y")
                idx_z = findidx("_chem_comp_atom.model_Cartn_z")
                idx_xi = findidx("_chem_comp_atom.pdbx_model_Cartn_x_ideal")
                idx_yi = findidx("_chem_comp_atom.pdbx_model_Cartn_y_ideal")
                idx_zi = findidx("_chem_comp_atom.pdbx_model_Cartn_z_ideal")
                if any(
                    x -> x === nothing,
                    (idx_atom, idx_type, idx_charge, idx_x, idx_y, idx_z, idx_xi, idx_yi, idx_zi),
                )
                    continue
                end
                idx_atom = idx_atom::Int
                idx_type = idx_type::Int
                idx_charge = idx_charge::Int
                idx_x = idx_x::Int
                idx_y = idx_y::Int
                idx_z = idx_z::Int
                idx_xi = idx_xi::Int
                idx_yi = idx_yi::Int
                idx_zi = idx_zi::Int

                rows = Vector{Vector{String}}()
                while true
                    row_line = if pending === nothing
                        eof(io) ? nothing : readline(io)
                    else
                        x = pending
                        pending = nothing
                        x
                    end
                    row_line === nothing && break
                    rs = strip(row_line)
                    if isempty(rs)
                        continue
                    end
                    if rs == "#" || rs == "loop_" || startswith(rs, "_") || startswith(rs, "data_")
                        pending = row_line
                        break
                    end
                    cols = _split_cif_tokens(row_line)
                    length(cols) < length(headers) && continue
                    push!(rows, cols)
                end

                if !isempty(rows)
                    atoms = NamedTuple{(:atom_name, :element, :x, :y, :z, :charge, :has_coord), Tuple{String, String, Float32, Float32, Float32, Float32, Bool}}[]
                    for r in rows
                        atom_name = String(r[idx_atom])
                        element = uppercase(strip(String(r[idx_type])))
                        isempty(element) && (element = _infer_element_from_atom_name(atom_name))
                        element in ("H", "D", "T") && continue
                        q = _try_parse_i(r[idx_charge])
                        charge = q === nothing ? 0f0 : Float32(q)

                        x = _try_parse_f32(r[idx_x])
                        y = _try_parse_f32(r[idx_y])
                        z = _try_parse_f32(r[idx_z])
                        if x === nothing || y === nothing || z === nothing
                            x = _try_parse_f32(r[idx_xi])
                            y = _try_parse_f32(r[idx_yi])
                            z = _try_parse_f32(r[idx_zi])
                        end
                        if x === nothing || y === nothing || z === nothing
                            x = 0f0
                            y = 0f0
                            z = 0f0
                            has_coord = false
                        else
                            has_coord = true
                        end
                        push!(
                            atoms,
                            (atom_name = atom_name, element = element, x = Float32(x), y = Float32(y), z = Float32(z), charge = charge, has_coord = has_coord),
                        )
                    end
                    _CCD_COMPONENT_CACHE[current_code] = atoms
                    delete!(needed, current_code)
                    isempty(needed) && return
                end
            end
        end
    end
end

function _ccd_component_atoms(code::AbstractString)
    uc = uppercase(strip(String(code)))
    isempty(uc) && return NamedTuple{(:atom_name, :element, :x, :y, :z, :charge, :has_coord), Tuple{String, String, Float32, Float32, Float32, Float32, Bool}}[]
    _ensure_ccd_component_entries!(Set([uc]))
    return get(
        _CCD_COMPONENT_CACHE,
        uc,
        NamedTuple{(:atom_name, :element, :x, :y, :z, :charge, :has_coord), Tuple{String, String, Float32, Float32, Float32, Float32, Bool}}[],
    )
end

function _polymer_default_atom_names(mol_type::String, code::String)
    if mol_type == "protein"
        return get(PROTEIN_HEAVY_ATOMS, uppercase(code), PROTEIN_HEAVY_ATOMS["UNK"])
    elseif mol_type == "dna"
        return vcat(_DNA_BACKBONE_ATOMS, get(_DNA_BASE_ATOMS, uppercase(code), _DNA_BASE_ATOMS["DN"]))
    elseif mol_type == "rna"
        return vcat(_RNA_BACKBONE_ATOMS, get(_RNA_BASE_ATOMS, uppercase(code), _RNA_BASE_ATOMS["N"]))
    end
    return ["C1"]
end

_polymer_centre_atom(mol_type::String, atom_name::String) = mol_type == "protein" ? atom_name == "CA" : (atom_name == "C1'")

function _build_polymer_atoms_from_codes(
    ccd_codes::AbstractVector{<:AbstractString},
    mol_type::String;
    chain_id::String,
    is_resolved::Bool = false,
)
    atoms = AtomRecord[]
    n_res = length(ccd_codes)
    for (res_id, ccd_code_raw) in enumerate(ccd_codes)
        ccd_code = uppercase(strip(ccd_code_raw))
        comp_atoms = _ccd_component_atoms(ccd_code)
        xoff = Float32((res_id - 1) * 4.0)

        if !isempty(comp_atoms)
            for a in comp_atoms
                atom_name = String(a.atom_name)
                atom_name == "OXT" && mol_type == "protein" && res_id < n_res && continue
                element = isempty(a.element) ? _infer_element_from_atom_name(atom_name) : String(a.element)
                xyz = a.has_coord ? (Float32(a.x + xoff), Float32(a.y), Float32(a.z)) : _pseudo_atom_xyz(res_id, atom_name)
                centre = _polymer_centre_atom(mol_type, atom_name)
                push!(
                    atoms,
                    AtomRecord(
                        atom_name,
                        ccd_code,
                        mol_type,
                        element,
                        chain_id,
                        res_id,
                        centre,
                        xyz[1],
                        xyz[2],
                        xyz[3],
                        is_resolved,
                    ),
                )
            end
            continue
        end

        for atom_name in _polymer_default_atom_names(mol_type, ccd_code)
            atom_name == "OXT" && mol_type == "protein" && res_id < n_res && continue
            element = _infer_element_from_atom_name(atom_name)
            x, y, z = _pseudo_atom_xyz(res_id, atom_name)
            centre = _polymer_centre_atom(mol_type, atom_name)
            push!(
                atoms,
                AtomRecord(atom_name, ccd_code, mol_type, element, chain_id, res_id, centre, x, y, z, is_resolved),
            )
        end
    end
    return atoms
end

function _build_ligand_atoms_from_codes(ccd_codes::AbstractVector{<:AbstractString}; chain_id::String, is_resolved::Bool = false)
    atoms = AtomRecord[]
    for (res_id, code_raw) in enumerate(ccd_codes)
        code = uppercase(strip(code_raw))
        comp_atoms = _ccd_component_atoms(code)
        xoff = Float32((res_id - 1) * 5.0)
        if !isempty(comp_atoms)
            for (k, a) in enumerate(comp_atoms)
                atom_name = String(a.atom_name)
                element = isempty(a.element) ? _infer_element_from_atom_name(atom_name) : String(a.element)
                if a.has_coord
                    x = Float32(a.x + xoff)
                    y = Float32(a.y)
                    z = Float32(a.z)
                else
                    x = Float32(xoff + 0.2f0 * k)
                    y = 0f0
                    z = 0f0
                end
                push!(
                    atoms,
                    AtomRecord(atom_name, code, "ligand", element, chain_id, res_id, true, x, y, z, is_resolved),
                )
            end
            continue
        end

        element = length(code) <= 2 ? code : _infer_element_from_atom_name(code)
        atom_name = length(code) <= 4 ? code : "C1"
        push!(
            atoms,
            AtomRecord(atom_name, code, "ligand", uppercase(element), chain_id, res_id, true, xoff, 0f0, 0f0, is_resolved),
        )
    end
    return atoms
end

function _normalize_ligand_atom_map(atom_map_any, context::String)
    atom_map_any isa AbstractDict || error("$context.atom_map_to_atom_name must be an object")
    out = Dict{Int, String}()
    for (k, v) in atom_map_any
        idx = k isa Integer ? Int(k) : parse(Int, strip(String(k)))
        out[idx] = uppercase(strip(String(v)))
    end
    return out
end

function _smiles_atom_name(element::String, i::Int)
    elem = uppercase(strip(element))
    stem = isempty(elem) ? "X" : elem
    return "$(stem)$(i)"
end

function _parse_smiles_element(body::AbstractString)
    s = String(body)
    isempty(s) && return "C"
    i = firstindex(s)
    n = lastindex(s)
    while i <= n && isdigit(s[i])
        i = nextind(s, i)
    end
    i > n && return "C"
    c = s[i]
    if c == '*'
        return "C"
    elseif isuppercase(c)
        if i < n && islowercase(s[nextind(s, i)])
            return uppercase(String(s[i:nextind(s, i)]))
        end
        return uppercase(string(c))
    elseif islowercase(c)
        return uppercase(string(c))
    end
    return "C"
end

function _extract_smiles_atommap(body::AbstractString)
    s = String(body)
    pos = findlast(==(':'), s)
    pos === nothing && return nothing
    pos >= lastindex(s) && return nothing
    raw = strip(s[nextind(s, pos):end])
    isempty(raw) && return nothing
    all(isdigit, raw) || return nothing
    return parse(Int, raw)
end

function _build_ligand_atoms_from_smiles(smiles::AbstractString; chain_id::String, is_resolved::Bool = false)
    s = strip(String(smiles))
    isempty(s) && error("SMILES ligand string cannot be empty")

    elems = String[]
    atom_map = Dict{Int, String}()
    i = firstindex(s)
    n = lastindex(s)
    atom_idx = 0
    while i <= n
        c = s[i]
        if c == '['
            j = nextind(s, i)
            while j <= n && s[j] != ']'
                j = nextind(s, j)
            end
            body = if j <= n
                String(s[nextind(s, i):prevind(s, j)])
            else
                String(s[nextind(s, i):n])
            end
            atom_idx += 1
            elem = _parse_smiles_element(body)
            atom_name = _smiles_atom_name(elem, atom_idx)
            push!(elems, elem)
            map_idx = _extract_smiles_atommap(body)
            map_idx !== nothing && (atom_map[map_idx] = atom_name)
            i = j <= n ? nextind(s, j) : n + 1
            continue
        end

        if isuppercase(c)
            elem = string(c)
            if i < n && islowercase(s[nextind(s, i)])
                nxt = s[nextind(s, i)]
                cand = string(c, nxt)
                if cand in ("Cl", "Br", "Si", "Na", "Mg", "Ca", "Fe", "Zn", "Cu", "Mn", "Co", "Ni")
                    elem = cand
                    i = nextind(s, i)
                end
            end
            atom_idx += 1
            push!(elems, uppercase(elem))
            i = nextind(s, i)
            continue
        end

        if c in ('b', 'c', 'n', 'o', 'p', 's')
            atom_idx += 1
            push!(elems, uppercase(string(c)))
            i = nextind(s, i)
            continue
        end

        i = nextind(s, i)
    end

    isempty(elems) && error("Failed to parse any atoms from SMILES ligand: $smiles")

    atoms = AtomRecord[]
    for (k, elem) in enumerate(elems)
        θ = Float32(2 * pi * (k - 1) / max(length(elems), 3))
        r = 1.6f0 + 0.1f0 * Float32((k - 1) ÷ 6)
        x = r * cos(θ)
        y = r * sin(θ)
        z = 0.15f0 * Float32(k - 1)
        atom_name = _smiles_atom_name(elem, k)
        push!(
            atoms,
            AtomRecord(atom_name, "UNL", "ligand", elem, chain_id, 1, true, x, y, z, is_resolved),
        )
    end

    if isempty(atom_map)
        for k in 1:length(atoms)
            atom_map[k] = atoms[k].atom_name
        end
    end
    return atoms, atom_map
end

function _build_ligand_atoms_from_file(
    file_spec::AbstractString;
    chain_id::String,
    json_dir::AbstractString = ".",
    is_resolved::Bool = true,
)
    raw = strip(String(file_spec))
    isempty(raw) && error("FILE_ ligand path cannot be empty")
    path = if isabspath(raw)
        raw
    else
        normpath(joinpath(json_dir, raw))
    end
    isfile(path) || error("Ligand FILE_ path does not exist: $path")

    loaded = load_structure_atoms(path)
    atoms = AtomRecord[]
    atom_map = Dict{Int, String}()
    atom_counter = 0
    for a in loaded
        atom_counter += 1
        atom_name = uppercase(strip(a.atom_name))
        atom_map[atom_counter] = atom_name
        push!(
            atoms,
            AtomRecord(
                atom_name,
                a.res_name,
                "ligand",
                uppercase(a.element),
                chain_id,
                a.res_id,
                true,
                a.x,
                a.y,
                a.z,
                is_resolved,
            ),
        )
    end
    isempty(atoms) && error("No atoms parsed from FILE_ ligand: $path")
    return atoms, atom_map
end

function _compact_sequence(sequence::AbstractString)
    io = IOBuffer()
    for c in uppercase(sequence)
        isspace(c) && continue
        print(io, c)
    end
    return String(take!(io))
end

function _apply_polymer_modifications!(
    ccd_codes::Vector{String},
    mods_any,
    pos_key::String,
    type_key::String,
    context::String,
)
    mods_any isa AbstractVector || error("$context.modifications must be an array")
    for (i, m_any) in enumerate(mods_any)
        m_any isa AbstractDict || error("$context.modifications[$i] must be an object")
        mod = _as_string_dict(m_any)
        haskey(mod, type_key) || error("$context.modifications[$i].$type_key is required")
        haskey(mod, pos_key) || error("$context.modifications[$i].$pos_key is required")
        pos = Int(mod[pos_key])
        (1 <= pos <= length(ccd_codes)) || error("$context.modifications[$i].$pos_key is out of range")
        mtype = uppercase(strip(String(mod[type_key])))
        startswith(mtype, "CCD_") && (mtype = mtype[5:end])
        isempty(mtype) && error("$context.modifications[$i].$type_key cannot be empty")
        ccd_codes[pos] = mtype
    end
    return ccd_codes
end

function _extract_protein_chain_specs(task::AbstractDict{<:Any, <:Any})
    haskey(task, "sequences") || error("Task is missing required field: sequences")
    sequences = task["sequences"]
    sequences isa AbstractVector || error("Task.sequences must be an array")

    specs = ProteinChainSpec[]
    chain_idx = 1
    for (i, entity_any) in enumerate(sequences)
        entity_any isa AbstractDict || error("Task.sequences[$i] must be an object")
        entity = _as_string_dict(entity_any)
        if haskey(entity, "proteinChain")
            pc_any = entity["proteinChain"]
            pc_any isa AbstractDict || error("Task.sequences[$i].proteinChain must be an object")
            pc = _as_string_dict(pc_any)
            haskey(pc, "sequence") || error("Task.sequences[$i].proteinChain.sequence is required")
            seq = _compact_sequence(String(pc["sequence"]))
            isempty(seq) && error("Task.sequences[$i].proteinChain.sequence must be non-empty")
            count = Int(get(pc, "count", 1))
            count > 0 || error("Task.sequences[$i].proteinChain.count must be positive")
            msa_cfg = haskey(pc, "msa") && pc["msa"] isa AbstractDict ? _as_string_dict(pc["msa"]) : Dict{String, Any}()
            for _ in 1:count
                chain_id = _chain_id_from_index(chain_idx)
                chain_idx += 1
                push!(specs, ProteinChainSpec(chain_id, seq, copy(msa_cfg)))
            end
            continue
        end

        local count::Int
        if haskey(entity, "dnaSequence")
            count = Int(get(_as_string_dict(entity["dnaSequence"]), "count", 1))
        elseif haskey(entity, "rnaSequence")
            count = Int(get(_as_string_dict(entity["rnaSequence"]), "count", 1))
        elseif haskey(entity, "ligand")
            count = Int(get(_as_string_dict(entity["ligand"]), "count", 1))
        elseif haskey(entity, "condition_ligand")
            count = Int(get(_as_string_dict(entity["condition_ligand"]), "count", 1))
        elseif haskey(entity, "ion")
            count = Int(get(_as_string_dict(entity["ion"]), "count", 1))
        else
            keys_str = join(string.(collect(keys(entity))), ", ")
            error("Unsupported sequence entry keys at sequences[$i]: [$keys_str]")
        end
        count > 0 || error("Task.sequences[$i].count must be positive")
        chain_idx += count
    end
    return specs
end

function _parse_task_entities(task::AbstractDict{<:Any, <:Any}; json_dir::AbstractString = ".")
    haskey(task, "sequences") || error("Task is missing required field: sequences")
    sequences = task["sequences"]
    sequences isa AbstractVector || error("Task.sequences must be an array")

    atoms = AtomRecord[]
    protein_specs = ProteinChainSpec[]
    entity_chain_ids = Dict{Int, Vector{String}}()
    entity_atom_map = Dict{Int, Dict{Int, String}}()
    chain_idx = 1

    for (entity_idx, entity_any) in enumerate(sequences)
        entity_any isa AbstractDict || error("Task.sequences[$entity_idx] must be an object")
        entity = _as_string_dict(entity_any)
        chain_ids = String[]

        if haskey(entity, "proteinChain")
            pc_any = entity["proteinChain"]
            pc_any isa AbstractDict || error("Task.sequences[$entity_idx].proteinChain must be an object")
            pc = _as_string_dict(pc_any)
            haskey(pc, "sequence") || error("Task.sequences[$entity_idx].proteinChain.sequence is required")
            seq = _compact_sequence(String(pc["sequence"]))
            isempty(seq) && error("Task.sequences[$entity_idx].proteinChain.sequence must be non-empty")
            count = Int(get(pc, "count", 1))
            count > 0 || error("Task.sequences[$entity_idx].proteinChain.count must be positive")
            msa_cfg = haskey(pc, "msa") && pc["msa"] isa AbstractDict ? _as_string_dict(pc["msa"]) : Dict{String, Any}()

            ccd_codes = String[get(PROT_STD_RESIDUES_ONE_TO_THREE, string(c), "UNK") for c in seq]
            has_mods = haskey(pc, "modifications") && (pc["modifications"] isa AbstractVector) && !isempty(pc["modifications"])
            if has_mods
                _apply_polymer_modifications!(
                    ccd_codes,
                    pc["modifications"],
                    "ptmPosition",
                    "ptmType",
                    "Task.sequences[$entity_idx].proteinChain",
                )
            end

            for _ in 1:count
                chain_id = _chain_id_from_index(chain_idx)
                chain_idx += 1
                push!(chain_ids, chain_id)
                if has_mods
                    append!(atoms, _build_polymer_atoms_from_codes(ccd_codes, "protein"; chain_id = chain_id))
                else
                    append!(atoms, ProtenixMini.build_sequence_atoms(seq; chain_id = chain_id))
                end
                push!(protein_specs, ProteinChainSpec(chain_id, seq, copy(msa_cfg)))
            end
        elseif haskey(entity, "dnaSequence")
            dna_any = entity["dnaSequence"]
            dna_any isa AbstractDict || error("Task.sequences[$entity_idx].dnaSequence must be an object")
            dna = _as_string_dict(dna_any)
            haskey(dna, "sequence") || error("Task.sequences[$entity_idx].dnaSequence.sequence is required")
            seq = _compact_sequence(String(dna["sequence"]))
            isempty(seq) && error("Task.sequences[$entity_idx].dnaSequence.sequence must be non-empty")
            count = Int(get(dna, "count", 1))
            count > 0 || error("Task.sequences[$entity_idx].dnaSequence.count must be positive")
            ccd_codes = String[get(_DNA_1TO3, c, "DN") for c in seq]
            if haskey(dna, "modifications")
                _apply_polymer_modifications!(
                    ccd_codes,
                    dna["modifications"],
                    "basePosition",
                    "modificationType",
                    "Task.sequences[$entity_idx].dnaSequence",
                )
            end
            for _ in 1:count
                chain_id = _chain_id_from_index(chain_idx)
                chain_idx += 1
                push!(chain_ids, chain_id)
                append!(atoms, _build_polymer_atoms_from_codes(ccd_codes, "dna"; chain_id = chain_id))
            end
        elseif haskey(entity, "rnaSequence")
            rna_any = entity["rnaSequence"]
            rna_any isa AbstractDict || error("Task.sequences[$entity_idx].rnaSequence must be an object")
            rna = _as_string_dict(rna_any)
            haskey(rna, "sequence") || error("Task.sequences[$entity_idx].rnaSequence.sequence is required")
            seq = _compact_sequence(String(rna["sequence"]))
            isempty(seq) && error("Task.sequences[$entity_idx].rnaSequence.sequence must be non-empty")
            count = Int(get(rna, "count", 1))
            count > 0 || error("Task.sequences[$entity_idx].rnaSequence.count must be positive")
            ccd_codes = String[get(_RNA_1TO3, c, "N") for c in seq]
            if haskey(rna, "modifications")
                _apply_polymer_modifications!(
                    ccd_codes,
                    rna["modifications"],
                    "basePosition",
                    "modificationType",
                    "Task.sequences[$entity_idx].rnaSequence",
                )
            end
            for _ in 1:count
                chain_id = _chain_id_from_index(chain_idx)
                chain_idx += 1
                push!(chain_ids, chain_id)
                append!(atoms, _build_polymer_atoms_from_codes(ccd_codes, "rna"; chain_id = chain_id))
            end
        elseif haskey(entity, "ligand") || haskey(entity, "condition_ligand")
            lig_key = haskey(entity, "ligand") ? "ligand" : "condition_ligand"
            lig_any = entity[lig_key]
            lig_any isa AbstractDict || error("Task.sequences[$entity_idx].$lig_key must be an object")
            lig = _as_string_dict(lig_any)
            haskey(lig, "ligand") || error("Task.sequences[$entity_idx].$lig_key.ligand is required")
            ligand_str = strip(String(lig["ligand"]))
            isempty(ligand_str) && error("Task.sequences[$entity_idx].$lig_key.ligand must be non-empty")
            count = Int(get(lig, "count", 1))
            count > 0 || error("Task.sequences[$entity_idx].$lig_key.count must be positive")
            ligand_uc = uppercase(ligand_str)
            if startswith(ligand_uc, "CCD_")
                ccd_codes = split(ligand_uc[5:end], "_")
                isempty(ccd_codes) && error("Task.sequences[$entity_idx].$lig_key.ligand has no CCD codes")
                for _ in 1:count
                    chain_id = _chain_id_from_index(chain_idx)
                    chain_idx += 1
                    push!(chain_ids, chain_id)
                    append!(atoms, _build_ligand_atoms_from_codes(ccd_codes; chain_id = chain_id))
                end
            elseif startswith(ligand_uc, "FILE_")
                file_spec = strip(ligand_str[6:end])
                provided_map = haskey(lig, "atom_map_to_atom_name") ? _normalize_ligand_atom_map(
                    lig["atom_map_to_atom_name"],
                    "Task.sequences[$entity_idx].$lig_key",
                ) : Dict{Int, String}()
                inferred_map = Dict{Int, String}()
                for _ in 1:count
                    chain_id = _chain_id_from_index(chain_idx)
                    chain_idx += 1
                    push!(chain_ids, chain_id)
                    lig_atoms, atom_map = _build_ligand_atoms_from_file(
                        file_spec;
                        chain_id = chain_id,
                        json_dir = json_dir,
                    )
                    append!(atoms, lig_atoms)
                    isempty(inferred_map) && (inferred_map = atom_map)
                end
                entity_atom_map[entity_idx] = isempty(provided_map) ? inferred_map : provided_map
            else
                provided_map = haskey(lig, "atom_map_to_atom_name") ? _normalize_ligand_atom_map(
                    lig["atom_map_to_atom_name"],
                    "Task.sequences[$entity_idx].$lig_key",
                ) : Dict{Int, String}()
                inferred_map = Dict{Int, String}()
                for _ in 1:count
                    chain_id = _chain_id_from_index(chain_idx)
                    chain_idx += 1
                    push!(chain_ids, chain_id)
                    lig_atoms, atom_map = _build_ligand_atoms_from_smiles(ligand_str; chain_id = chain_id)
                    append!(atoms, lig_atoms)
                    isempty(inferred_map) && (inferred_map = atom_map)
                end
                entity_atom_map[entity_idx] = isempty(provided_map) ? inferred_map : provided_map
            end
        elseif haskey(entity, "ion")
            ion_any = entity["ion"]
            ion_any isa AbstractDict || error("Task.sequences[$entity_idx].ion must be an object")
            ion = _as_string_dict(ion_any)
            haskey(ion, "ion") || error("Task.sequences[$entity_idx].ion.ion is required")
            ion_code = uppercase(strip(String(ion["ion"])))
            startswith(ion_code, "CCD_") && (ion_code = ion_code[5:end])
            isempty(ion_code) && error("Task.sequences[$entity_idx].ion.ion must be non-empty")
            count = Int(get(ion, "count", 1))
            count > 0 || error("Task.sequences[$entity_idx].ion.count must be positive")
            for _ in 1:count
                chain_id = _chain_id_from_index(chain_idx)
                chain_idx += 1
                push!(chain_ids, chain_id)
                append!(atoms, _build_ligand_atoms_from_codes([ion_code]; chain_id = chain_id))
            end
        else
            keys_str = join(string.(collect(keys(entity))), ", ")
            error("Unsupported sequence entry keys at sequences[$entity_idx]: [$keys_str]")
        end

        entity_chain_ids[entity_idx] = chain_ids
    end

    isempty(atoms) && error("No supported sequence entities found in infer task")
    return TaskEntityParseResult(atoms, protein_specs, entity_chain_ids, entity_atom_map)
end

function _build_atoms_from_infer_task(task::AbstractDict{<:Any, <:Any})
    return _parse_task_entities(task).atoms
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
    chain_specs::Union{Nothing, Vector{ProteinChainSpec}} = nothing,
    token_chain_ids::Union{Nothing, Vector{String}} = nothing,
)
    use_msa || return feat
    local_specs = chain_specs === nothing ? _extract_protein_chain_specs(task) : chain_specs
    isempty(local_specs) && return feat

    restype = Float32.(feat["restype"])
    n_tok = size(restype, 1)
    restype_idx = Vector{Int}(undef, n_tok)
    @inbounds for i in 1:n_tok
        _, idx = findmax(@view restype[i, :])
        restype_idx[i] = idx - 1
    end

    chain_token_cols = if token_chain_ids === nothing
        asym = Int.(feat["asym_id"])
        [findall(==(i - 1), asym) for i in 1:length(local_specs)]
    else
        [findall(==(spec.chain_id), token_chain_ids) for spec in local_specs]
    end
    all(!isempty(cols) for cols in chain_token_cols) || error("Failed to map task chains to token columns.")

    chain_sequences = [s.sequence for s in local_specs]
    is_homomer_or_monomer = length(Set(chain_sequences)) == 1
    json_dir = dirname(abspath(json_path))
    chain_features = NamedTuple[]
    for spec in local_specs
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
            "MSA/token length mismatch on chain $(local_specs[i].chain_id): MSA has $(size(chain_features[i].msa, 2)) columns, tokens have $(length(cols)).",
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

function _bond_field(bond::Dict{String, Any}, side::String, idx::Int, field::String)
    k_side = string(side, "_", field)
    k_num = string(field, idx)
    haskey(bond, k_side) && return bond[k_side]
    return get(bond, k_num, nothing)
end

function _resolve_bond_chains(
    entity_chain_ids::Dict{Int, Vector{String}},
    entity_id::Int,
    copy_any,
    context::String,
)
    haskey(entity_chain_ids, entity_id) || error("$context references unknown entity $entity_id")
    chains = entity_chain_ids[entity_id]
    if copy_any === nothing
        return chains
    end
    copy_idx = copy_any isa Integer ? Int(copy_any) : parse(Int, strip(String(copy_any)))
    (1 <= copy_idx <= length(chains)) || error("$context copy index $copy_idx is out of range for entity $entity_id")
    return [chains[copy_idx]]
end

function _resolve_bond_atoms(
    atom_lookup::Dict{Tuple{String, Int, String}, Vector{Int}},
    chains::Vector{String},
    position::Int,
    atom_name::String,
    context::String,
)
    idxs = Int[]
    atom_name_u = uppercase(strip(atom_name))
    for chain_id in chains
        key = (chain_id, position, atom_name_u)
        haskey(atom_lookup, key) || error("$context did not match atom '$atom_name' at position $position on chain $chain_id")
        append!(idxs, atom_lookup[key])
    end
    return sort(idxs)
end

function _resolve_bond_atom_name(atom_any, entity_id::Int, entity_atom_map::Dict{Int, Dict{Int, String}}, context::String)
    if atom_any isa Integer
        idx = Int(atom_any)
        haskey(entity_atom_map, entity_id) || error(
            "$context uses numeric atom index $idx for entity $entity_id, but no atom_map_to_atom_name is available.",
        )
        amap = entity_atom_map[entity_id]
        haskey(amap, idx) || error("$context atom index $idx not found in atom_map_to_atom_name for entity $entity_id.")
        return String(amap[idx])
    end

    atom_name = String(strip(String(atom_any)))
    all(isdigit, atom_name) || return atom_name
    idx = parse(Int, atom_name)
    haskey(entity_atom_map, entity_id) || error(
        "$context uses numeric atom index $idx for entity $entity_id, but no atom_map_to_atom_name is available.",
    )
    amap = entity_atom_map[entity_id]
    haskey(amap, idx) || error("$context atom index $idx not found in atom_map_to_atom_name for entity $entity_id.")
    return String(amap[idx])
end

function _inject_task_covalent_token_bonds!(
    feat::Dict{String, Any},
    atoms::Vector{AtomRecord},
    task::AbstractDict{<:Any, <:Any},
    entity_chain_ids::Dict{Int, Vector{String}},
    entity_atom_map::Dict{Int, Dict{Int, String}} = Dict{Int, Dict{Int, String}}(),
)
    haskey(task, "covalent_bonds") || return feat
    bonds_any = task["covalent_bonds"]
    bonds_any isa AbstractVector || error("task.covalent_bonds must be an array when provided.")
    isempty(bonds_any) && return feat

    n_tok = size(feat["restype"], 1)
    token_bonds = haskey(feat, "token_bonds") ? Int.(feat["token_bonds"]) : zeros(Int, n_tok, n_tok)
    atom_to_token_idx = Int.(feat["atom_to_token_idx"])

    atom_lookup = Dict{Tuple{String, Int, String}, Vector{Int}}()
    for (atom_idx, atom) in enumerate(atoms)
        key = (atom.chain_id, atom.res_id, uppercase(atom.atom_name))
        if haskey(atom_lookup, key)
            push!(atom_lookup[key], atom_idx)
        else
            atom_lookup[key] = [atom_idx]
        end
    end

    for (bond_i, bond_any) in enumerate(bonds_any)
        bond_any isa AbstractDict || error("task.covalent_bonds[$bond_i] must be an object.")
        bond = _as_string_dict(bond_any)
        ctx = "task.covalent_bonds[$bond_i]"

        entity1_any = _bond_field(bond, "left", 1, "entity")
        entity2_any = _bond_field(bond, "right", 2, "entity")
        position1_any = _bond_field(bond, "left", 1, "position")
        position2_any = _bond_field(bond, "right", 2, "position")
        atom1_any = _bond_field(bond, "left", 1, "atom")
        atom2_any = _bond_field(bond, "right", 2, "atom")
        copy1_any = _bond_field(bond, "left", 1, "copy")
        copy2_any = _bond_field(bond, "right", 2, "copy")

        entity1_any === nothing && error("$ctx missing entity1/left_entity")
        entity2_any === nothing && error("$ctx missing entity2/right_entity")
        position1_any === nothing && error("$ctx missing position1/left_position")
        position2_any === nothing && error("$ctx missing position2/right_position")
        atom1_any === nothing && error("$ctx missing atom1/left_atom")
        atom2_any === nothing && error("$ctx missing atom2/right_atom")

        entity1 = entity1_any isa Integer ? Int(entity1_any) : parse(Int, strip(String(entity1_any)))
        entity2 = entity2_any isa Integer ? Int(entity2_any) : parse(Int, strip(String(entity2_any)))
        position1 = position1_any isa Integer ? Int(position1_any) : parse(Int, strip(String(position1_any)))
        position2 = position2_any isa Integer ? Int(position2_any) : parse(Int, strip(String(position2_any)))
        atom1 = _resolve_bond_atom_name(atom1_any, entity1, entity_atom_map, "$ctx side1")
        atom2 = _resolve_bond_atom_name(atom2_any, entity2, entity_atom_map, "$ctx side2")

        chains1 = _resolve_bond_chains(entity_chain_ids, entity1, copy1_any, "$ctx side1")
        chains2 = _resolve_bond_chains(entity_chain_ids, entity2, copy2_any, "$ctx side2")
        if copy1_any === nothing && copy2_any === nothing
            length(chains1) == length(chains2) || error(
                "$ctx omits copy indices, but entity counts differ ($(length(chains1)) vs $(length(chains2))).",
            )
        end
        length(chains1) == length(chains2) || error("$ctx resolved unequal copy multiplicity ($(length(chains1)) vs $(length(chains2))).")

        atoms1 = _resolve_bond_atoms(atom_lookup, chains1, position1, atom1, "$ctx side1")
        atoms2 = _resolve_bond_atoms(atom_lookup, chains2, position2, atom2, "$ctx side2")
        length(atoms1) == length(atoms2) || error("$ctx resolved unequal atom matches ($(length(atoms1)) vs $(length(atoms2))).")

        for (a1, a2) in zip(atoms1, atoms2)
            t1 = atom_to_token_idx[a1] + 1
            t2 = atom_to_token_idx[a2] + 1
            (1 <= t1 <= n_tok && 1 <= t2 <= n_tok) || error("$ctx resolved token index out of range.")
            token_bonds[t1, t2] = 1
            token_bonds[t2, t1] = 1
        end
    end

    feat["token_bonds"] = token_bonds
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

function _inject_task_esm_token_embedding!(
    feat::Dict{String, Any},
    task::AbstractDict{<:Any, <:Any},
)
    haskey(task, "esm_token_embedding") || return feat
    emb = _to_float_array(task["esm_token_embedding"])
    ndims(emb) == 2 || error("esm_token_embedding must be rank-2 [N_token, D].")
    size(emb, 1) == size(feat["restype"], 1) || error(
        "esm_token_embedding token length mismatch: expected $(size(feat["restype"], 1)), got $(size(emb, 1)).",
    )
    feat["esm_token_embedding"] = emb
    return feat
end

function _validate_required_model_inputs!(
    params::NamedTuple,
    feat::AbstractDict{<:AbstractString, <:Any},
    context::AbstractString,
)
    if params.needs_esm_embedding && !haskey(feat, "esm_token_embedding")
        error(
            "Model $(params.model_name) requires esm_token_embedding for $context. " *
            "Provide task.esm_token_embedding [N_token,D] in JSON, or pass esm_token_embedding directly in sequence mode.",
        )
    end
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
    feature_dict;
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

function _resolve_predict_runtime(opts::ProtenixPredictOptions)
    params = recommended_params(
        opts.model_name;
        use_default_params = opts.use_default_params,
        cycle = opts.cycle,
        step = opts.step,
        sample = opts.sample,
        use_msa = opts.use_msa,
    )
    mkpath(opts.out_dir)
    local_weights = isempty(opts.weights_path) ? default_weights_path(opts.model_name) : abspath(opts.weights_path)
    isdir(local_weights) || error("weights_path does not exist: $local_weights")
    loaded = _load_model(opts.model_name, local_weights; strict = opts.strict)
    return (params = params, loaded = loaded)
end

function predict_json(input::AbstractString, opts::ProtenixPredictOptions)
    runtime = _resolve_predict_runtime(opts)
    params = runtime.params
    loaded = runtime.loaded
    json_paths = _collect_input_paths(input; exts = (".json",))
    records = NamedTuple[]

    for json_path in json_paths
        tasks = _ensure_json_tasks(json_path)
        for (task_idx, task_any) in enumerate(tasks)
            task_any isa AbstractDict || error("Task $(task_idx) in $json_path is not an object")
            task = _as_string_dict(task_any)
            task_name = haskey(task, "name") ? String(task["name"]) : "$(_default_task_name(json_path))_$(task_idx - 1)"
            parsed_task = _parse_task_entities(task; json_dir = dirname(abspath(json_path)))
            atoms = parsed_task.atoms

            for seed in opts.seeds
                rng = MersenneTwister(seed)
                bundle = build_feature_bundle_from_atoms(atoms; task_name = task_name, rng = rng)
                token_chain_ids = [bundle["atoms"][tok.centre_atom_index].chain_id for tok in bundle["tokens"]]
                _normalize_protenix_feature_dict!(bundle["input_feature_dict"])
                _inject_task_msa_features!(
                    bundle["input_feature_dict"],
                    task,
                    json_path;
                    use_msa = params.use_msa,
                    chain_specs = parsed_task.protein_specs,
                    token_chain_ids = token_chain_ids,
                )
                _inject_task_covalent_token_bonds!(
                    bundle["input_feature_dict"],
                    bundle["atoms"],
                    task,
                    parsed_task.entity_chain_ids,
                    parsed_task.entity_atom_map,
                )
                _inject_task_template_features!(bundle["input_feature_dict"], task)
                _inject_task_esm_token_embedding!(bundle["input_feature_dict"], task)
                _validate_required_model_inputs!(
                    params,
                    bundle["input_feature_dict"],
                    "task '$task_name' in $(basename(json_path))",
                )
                typed_feat = ProtenixMini.as_protenix_features(bundle["input_feature_dict"])
                pred = _run_model(
                    loaded,
                    typed_feat;
                    cycle = params.cycle,
                    step = params.step,
                    sample = params.sample,
                    rng = rng,
                )

                task_dump_dir = joinpath(opts.out_dir, task_name, "seed_$(seed)")
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
    opts = ProtenixPredictOptions(
        out_dir = out_dir,
        model_name = model_name,
        weights_path = weights_path,
        seeds = seeds,
        use_default_params = use_default_params,
        cycle = cycle,
        step = step,
        sample = sample,
        use_msa = use_msa,
        strict = strict,
    )
    return predict_json(input, opts)
end

function predict_sequence(sequence::AbstractString, opts::ProtenixSequenceOptions)
    seq = uppercase(strip(sequence))
    isempty(seq) && error("sequence must be non-empty")

    runtime = _resolve_predict_runtime(opts.common)
    params = runtime.params
    loaded = runtime.loaded
    records = NamedTuple[]

    for seed in opts.common.seeds
        rng = MersenneTwister(seed)
        atoms = ProtenixMini.build_sequence_atoms(seq; chain_id = opts.chain_id)
        bundle = build_feature_bundle_from_atoms(atoms; task_name = opts.task_name, rng = rng)
        _normalize_protenix_feature_dict!(bundle["input_feature_dict"])
        if opts.esm_token_embedding !== nothing
            emb = opts.esm_token_embedding
            size(emb, 1) == size(bundle["input_feature_dict"]["restype"], 1) || error(
                "esm_token_embedding token length mismatch: expected $(size(bundle["input_feature_dict"]["restype"], 1)), got $(size(emb, 1)).",
            )
            bundle["input_feature_dict"]["esm_token_embedding"] = emb
        end
        _validate_required_model_inputs!(
            params,
            bundle["input_feature_dict"],
            "sequence task '$(opts.task_name)'",
        )
        typed_feat = ProtenixMini.as_protenix_features(bundle["input_feature_dict"])
        pred = _run_model(
            loaded,
            typed_feat;
            cycle = params.cycle,
            step = params.step,
            sample = params.sample,
            rng = rng,
        )

        task_dump_dir = joinpath(opts.common.out_dir, opts.task_name, "seed_$(seed)")
        pred_dir = dump_prediction_bundle(task_dump_dir, opts.task_name, bundle["atoms"], pred.coordinate)
        _write_confidence_summaries(pred_dir, opts.task_name, seed, pred)
        cif_paths = sort(filter(endswith(".cif"), readdir(pred_dir; join = true)))

        push!(
            records,
            (
                task_name = opts.task_name,
                seed = seed,
                prediction_dir = pred_dir,
                cif_paths = cif_paths,
            ),
        )
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
    esm_token_embedding::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    strict::Bool = true,
)
    common = ProtenixPredictOptions(
        out_dir = out_dir,
        model_name = model_name,
        weights_path = weights_path,
        seeds = seeds,
        use_default_params = use_default_params,
        cycle = cycle,
        step = step,
        sample = sample,
        use_msa = use_msa,
        strict = strict,
    )
    seq_opts = ProtenixSequenceOptions(
        common = common,
        task_name = task_name,
        chain_id = chain_id,
        esm_token_embedding = esm_token_embedding,
    )
    return predict_sequence(sequence, seq_opts)
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

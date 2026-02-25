module ProtenixAPI

using Random
using Statistics
using Flux: gpu, cpu
using kalign_jll: kalign_jll
using HuggingFaceApi: hf_hub_download

import ..Device: feats_to_device, feats_to_cpu, device_ref
import ..Data: AtomRecord, build_feature_bundle_from_atoms, load_structure_atoms
import ..Data.Structure: _split_cif_row, _normalize_mse
import ..Data.Constants: PROT_STD_RESIDUES_ONE_TO_THREE, PROTEIN_HEAVY_ATOMS, STD_RESIDUES_PROTENIX, DNA_STD_RESIDUES, RNA_STD_RESIDUES_NATURAL, STD_RESIDUES_WITH_GAP
import ..Data.Design: PROT_THREE_TO_ONE
import ..Data.Features
import ..JSONLite: parse_json, write_json
import ..Model: load_safetensors_weights
import ..Model:
    DiffusionModule, DesignConditionEmbedder,
    load_diffusion_module!, load_design_condition_embedder!,
    infer_model_scaffold_dims, infer_design_condition_embedder_dims,
    checkpoint_coverage_report,
    as_relpos_input, as_atom_attention_input,
    InferenceNoiseScheduler, sample_diffusion
import ..Data: build_basic_feature_bundle
import ..Output: dump_prediction_bundle
import ..ProtenixBase
import ..ProtenixMini
import ..ESMProvider
import ..Schema: InputTask, GenerationSpec, MSAChainOptions
import ..WeightsHub: download_model_weights

export ProtenixModelSpec,
    ProtenixPredictOptions,
    ProtenixSequenceOptions,
    PredictJSONRecord,
    PredictSequenceRecord,
    ProtenixHandle,
    MODEL_SPECS,
    resolve_model_spec,
    recommended_params,
    list_supported_models,
    default_weights_path,
    predict_json,
    predict_sequence,
    convert_structure_to_infer_json,
    add_precomputed_msa_to_json,
    load_protenix,
    fold,
    confidence_metrics,
    PXDesignHandle,
    load_pxdesign,
    design,
    design_task,
    design_target,
    template_structure

struct ProtenixModelSpec
    model_name::String
    family::Symbol
    default_cycle::Int
    default_step::Int
    default_sample::Int
    default_use_msa::Bool
    needs_esm_embedding::Bool
    msa_pair_as_unpair::Bool
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
    gpu::Bool
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
    gpu::Bool = false,
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
        gpu,
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

const PredictJSONRecord = NamedTuple{
    (:input_json, :task_name, :seed, :prediction_dir, :cif_paths),
    Tuple{String, String, Int, String, Vector{String}},
}

const PredictSequenceRecord = NamedTuple{
    (:task_name, :seed, :prediction_dir, :cif_paths),
    Tuple{String, Int, String, Vector{String}},
}

function ProtenixSequenceOptions(;
    common::ProtenixPredictOptions = ProtenixPredictOptions(),
    task_name::AbstractString = "protenix_sequence",
    chain_id::AbstractString = "A",
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
        false,
    ),
    "protenix_mini_ism_v0.5.0" => ProtenixModelSpec(
        "protenix_mini_ism_v0.5.0",
        :mini,
        4,
        5,
        5,
        false,
        true,
        false,
    ),
    "protenix_base_default_v1.0.0" => ProtenixModelSpec(
        "protenix_base_default_v1.0.0",
        :base,
        10,
        200,
        5,
        true,
        false,
        true,
    ),
    "protenix_base_20250630_v1.0.0" => ProtenixModelSpec(
        "protenix_base_20250630_v1.0.0",
        :base,
        10,
        200,
        5,
        true,
        false,
        true,
    ),
)

# Short aliases → canonical model names.
# "protenix_v1" resolves to the latest v1.0 model (20250630), not the original default.
const _MODEL_ALIASES = Dict{String, String}(
    "protenix_v1" => "protenix_base_20250630_v1.0.0",
)

function _resolve_model_alias(model_name::AbstractString)
    key = lowercase(strip(String(model_name)))
    return get(_MODEL_ALIASES, key, key)
end

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
    Vector{NamedTuple{(:atom_name, :element, :x, :y, :z, :charge, :has_coord, :leaving), Tuple{String, String, Float32, Float32, Float32, Float32, Bool, Bool}}},
}()

# Per-component bond connectivity (atom_name1, atom_name2) for leaving-group graph traversal.
const _CCD_BOND_CACHE = Dict{String, Vector{Tuple{String, String}}}()

# Per-component CCD _chem_comp.type (e.g., "L-PEPTIDE LINKING", "NON-POLYMER").
const _CCD_TYPE_CACHE = Dict{String, String}()
# Per-component CCD _chem_comp.one_letter_code (e.g., "S" for SEP → serine).
const _CCD_ONE_LETTER_CACHE = Dict{String, String}()

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
            # MSA gap is index 31 in the 32-class one-hot space.
            # Python msa_utils.py: HHBLITS_INDEX_TO_OUR_INDEX maps '-' → 31
            # (0-19 = amino acids, 20 = UNK, 21-30 = unused, 31 = gap)
            out[i] = 31
        else
            aa3 = get(PROT_STD_RESIDUES_ONE_TO_THREE, string(aa), "UNK")
            out[i] = get(STD_RESIDUES_PROTENIX, aa3, STD_RESIDUES_PROTENIX["UNK"])
        end
    end
    out
end

# RNA MSA character mapping: A→21, G→22, C→23, U→24, unknown→25, gap→31
# Matches ProteinxV1.jl _MSA_RNA_SEQ_TO_ID exactly.
const _MSA_RNA_SEQ_TO_ID = let d = Dict{Char, Int}()
    for c in 'A':'Z'
        d[c] = 25   # default unknown RNA nucleotide
    end
    d['-'] = 31      # gap
    d['A'] = 21
    d['G'] = 22
    d['C'] = 23
    d['U'] = 24
    d
end

const ChainMSABlock = NamedTuple{
    (:msa, :has_deletion, :deletion_value, :deletion_mean, :profile),
    Tuple{Matrix{Int}, Matrix{Float32}, Matrix{Float32}, Vector{Float32}, Matrix{Float32}},
}

const ChainMSAFeatures = NamedTuple{
    (:combined, :non_pairing, :pairing, :pairing_keys),
    Tuple{
        ChainMSABlock,
        ChainMSABlock,
        Union{Nothing, ChainMSABlock},
        Union{Nothing, Vector{String}},
    },
}

const ChainSequenceRecord = NamedTuple{
    (:chain_id, :sequence),
    Tuple{String, String},
}

const ChainResidueRecord = NamedTuple{
    (:res_id, :res_name),
    Tuple{Int, String},
}

function _chain_msa_block(
    msa::AbstractMatrix{<:Integer},
    has_deletion::AbstractMatrix{<:Real},
    deletion_value::AbstractMatrix{<:Real},
    deletion_mean::AbstractVector{<:Real},
    profile::AbstractMatrix{<:Real},
)::ChainMSABlock
    return (
        msa = Int.(msa),
        has_deletion = Float32.(has_deletion),
        deletion_value = Float32.(deletion_value),
        deletion_mean = Float32.(deletion_mean),
        profile = Float32.(profile),
    )
end

"""
    _broadcast_msa_block_to_tokens(block, token_res_ids) → ChainMSABlock

Broadcast a ChainMSABlock from sequence-level (N_residues columns) to token-level
(N_tokens columns) using per-token residue IDs. This matches Python Protenix's
`expand_msa_features()` which broadcasts MSA features when modified residues
create multiple per-atom tokens for a single sequence position.

All tokens sharing the same res_id receive identical MSA feature values
(same MSA column, same profile row, same deletion statistics).
"""
function _broadcast_msa_block_to_tokens(block::ChainMSABlock, token_res_ids::Vector{Int})::ChainMSABlock
    seq_len = size(block.msa, 2)
    n_tokens = length(token_res_ids)
    seq_len == n_tokens && return block

    # Map unique sorted res_ids to 1-based MSA column indices.
    unique_rids = sort(unique(token_res_ids))
    n_unique = length(unique_rids)

    if n_unique > seq_len
        error(
            "MSA broadcast: expected $seq_len unique residue IDs for sequence-level MSA, got $n_unique " *
            "(more tokens than MSA columns — likely a bug)",
        )
    elseif n_unique < seq_len
        # MSA has more columns than the chain has sequence positions.
        # This happens when precomputed MSA is from a different (longer) protein.
        # Truncate MSA to match, using positional mapping (same as Python's sparse join).
        @warn("MSA has $seq_len columns but chain has $n_unique sequence positions; " *
              "truncating MSA to match (precomputed MSA may not correspond to this chain)")
        block = _chain_msa_block(
            block.msa[:, 1:n_unique],
            block.has_deletion[:, 1:n_unique],
            block.deletion_value[:, 1:n_unique],
            block.deletion_mean[1:n_unique],
            block.profile[1:n_unique, :],
        )
        seq_len = n_unique
    end

    rid_to_col = Dict(rid => i for (i, rid) in enumerate(unique_rids))
    col_map = [rid_to_col[rid] for rid in token_res_ids]

    n_rows = size(block.msa, 1)
    new_msa = Matrix{Int}(undef, n_rows, n_tokens)
    new_has_del = Matrix{Float32}(undef, n_rows, n_tokens)
    new_del_val = Matrix{Float32}(undef, n_rows, n_tokens)
    @inbounds for j in 1:n_tokens
        c = col_map[j]
        for r in 1:n_rows
            new_msa[r, j] = block.msa[r, c]
            new_has_del[r, j] = block.has_deletion[r, c]
            new_del_val[r, j] = block.deletion_value[r, c]
        end
    end

    new_del_mean = Float32[block.deletion_mean[col_map[j]] for j in 1:n_tokens]
    n_classes = size(block.profile, 2)
    new_profile = Matrix{Float32}(undef, n_tokens, n_classes)
    @inbounds for j in 1:n_tokens
        c = col_map[j]
        for k in 1:n_classes
            new_profile[j, k] = block.profile[c, k]
        end
    end

    return _chain_msa_block(new_msa, new_has_del, new_del_val, new_del_mean, new_profile)
end

"""
    _broadcast_msa_features_to_tokens(features, token_res_ids) → ChainMSAFeatures

Broadcast all MSA blocks (combined, non_pairing, pairing) from sequence-level
to token-level using per-token residue IDs.
"""
function _broadcast_msa_features_to_tokens(features::ChainMSAFeatures, token_res_ids::Vector{Int})::ChainMSAFeatures
    new_combined = _broadcast_msa_block_to_tokens(features.combined, token_res_ids)
    new_non_pairing = _broadcast_msa_block_to_tokens(features.non_pairing, token_res_ids)
    new_pairing = features.pairing === nothing ? nothing : _broadcast_msa_block_to_tokens(features.pairing, token_res_ids)
    return (
        combined = new_combined,
        non_pairing = new_non_pairing,
        pairing = new_pairing,
        pairing_keys = features.pairing_keys,
    )
end

function resolve_model_spec(model_name::AbstractString)
    key = _resolve_model_alias(model_name)
    haskey(MODEL_SPECS, key) && return MODEL_SPECS[key]
    supported = join(sort(collect(keys(MODEL_SPECS))), ", ")
    aliases = join(sort(collect(keys(_MODEL_ALIASES))), ", ")
    error("Unsupported model_name '$model_name'. Supported models: $supported. Aliases: $aliases")
end

"""
    _is_v1_model(model_name) → Bool

Return `true` if the model name indicates Protenix v1.0 or later.
v1.0 models use a different featurization path (e.g., CCD mol_type override
applies to all entities including ligands, not just polymer entities).
"""
_is_v1_model(model_name::AbstractString) = occursin("v1.0", String(model_name))

"""
    recommended_params(model_name; use_default_params=true, cycle=nothing, step=nothing,
                       sample=nothing, use_msa=nothing) → NamedTuple

Return recommended inference parameters for `model_name`. When `use_default_params=true`,
the model's registered defaults are used and individual overrides are ignored. Set
`use_default_params=false` to selectively override `cycle`, `step`, `sample`, or `use_msa`.

Returns a NamedTuple with fields: `model_name`, `family`, `cycle`, `step`, `sample`,
`use_msa`, `needs_esm_embedding`.
"""
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
        msa_pair_as_unpair = spec.msa_pair_as_unpair,
    )
end

"""
    list_supported_models() → Vector{NamedTuple}

Return sorted metadata for all registered Protenix model variants.

Each entry is a NamedTuple with fields: `model_name`, `family`, `default_cycle`,
`default_step`, `default_sample`, `default_use_msa`, `needs_esm_embedding`.

# Example

```julia
julia> list_supported_models()
7-element Vector{NamedTuple}:
 (model_name = "protenix_base_constraint_v0.5.0", family = :base, ...)
 (model_name = "protenix_base_default_v0.5.0", family = :base, ...)
 (model_name = "protenix_mini_default_v0.5.0", family = :mini, ...)
 ...
```
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
            msa_pair_as_unpair = MODEL_SPECS[n].msa_pair_as_unpair,
        ) for n in names
    ]
end

function default_weights_path(model_name::AbstractString; project_root::AbstractString = normpath(joinpath(@__DIR__, "..")))
    _ = project_root
    return download_model_weights(model_name)
end

function _load_json_task_payload(path::AbstractString)
    value = _as_string_dict(parse_json(read(path, String)))
    if value isa AbstractVector
        return (tasks = Any[value...], shape = :array, root = nothing)
    elseif value isa AbstractDict
        if haskey(value, "tasks")
            tasks = value["tasks"]
            tasks isa AbstractVector || error("Input JSON field 'tasks' must be an array of task objects: $path")
            return (tasks = Any[tasks...], shape = :tasks_wrapper, root = value)
        end
        return (tasks = Any[value], shape = :object, root = nothing)
    end
    error("Input JSON must be an object or array of objects: $path")
end

function _ensure_json_tasks(path::AbstractString)
    return _load_json_task_payload(path).tasks
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
    return _chain_letters(i)
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
    msa_cfg::NamedTuple{(:precomputed_msa_dir, :pairing_db), Tuple{Union{Nothing, String}, String}}
end

const _DEFAULT_MSA_CFG = (precomputed_msa_dir = nothing, pairing_db = "uniref100")

# RNA chain spec for RNA MSA support.
# RNA chains have unpairedMsa (inline text) or unpairedMsaPath (file path).
# No paired MSA for RNA — only non-pairing.
struct RNAChainSpec
    chain_id::String
    sequence::String
    unpaired_msa::Union{Nothing, String}        # inline A3M text
    unpaired_msa_path::Union{Nothing, String}    # path to A3M file
end

function _parse_msa_cfg(x)
    if !(x isa AbstractDict)
        return _DEFAULT_MSA_CFG
    end
    cfg = _as_string_dict(x)
    precomputed_msa_dir = if haskey(cfg, "precomputed_msa_dir")
        raw = cfg["precomputed_msa_dir"]
        raw === nothing ? nothing : String(raw)
    else
        nothing
    end
    pairing_db = haskey(cfg, "pairing_db") ? String(cfg["pairing_db"]) : "uniref100"
    return (precomputed_msa_dir = precomputed_msa_dir, pairing_db = pairing_db)
end

struct DNAChainSpec
    chain_id::String
    sequence::String           # original one-letter DNA sequence (e.g. "ATGC")
    ccd_codes::Vector{String}  # per-position CCD codes after modifications (e.g. ["DA", "DT", "5MC", "DC"])
end

struct TaskEntityParseResult
    atoms::Vector{AtomRecord}
    protein_specs::Vector{ProteinChainSpec}
    rna_specs::Vector{RNAChainSpec}
    dna_specs::Vector{DNAChainSpec}
    entity_chain_ids::Vector{Vector{String}}
    entity_atom_map::Vector{Dict{Int, String}}
    polymer_chain_ids::Set{String}   # chain IDs from proteinChain/dnaSequence/rnaSequence entities
    ion_chain_ids::Set{String}       # chain IDs from ion entities (Fix 19)
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
    # Check local paths first (fast, no network)
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
    # Download from HuggingFace (cached after first download)
    return hf_hub_download("MurrellLab/PXDesign.jl", "components.v20240608.cif")
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
        # Also re-scan if CCD metadata (type, one_letter_code) is missing.
        if !haskey(_CCD_COMPONENT_CACHE, uc) || !haskey(_CCD_TYPE_CACHE, uc)
            push!(needed, uc)
        end
    end
    isempty(needed) && return

    ccd_path = _default_ccd_components_path()

    current_code = ""
    active = false
    pending = nothing
    # Components whose atoms are cached but bonds haven't been parsed yet.
    pending_bonds = Set{String}()

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
                # Leaving the previous block: if it was pending bonds but had none, clear it.
                if !isempty(current_code) && current_code in pending_bonds
                    delete!(pending_bonds, current_code)
                end
                isempty(needed) && isempty(pending_bonds) && return
                current_code = uppercase(String(s[6:end]))
                active = current_code in needed || current_code in pending_bonds
                continue
            end
            active || continue

            # ─── Non-loop _chem_comp.* key-value entries (type, one_letter_code) ───
            if startswith(s, "_chem_comp.type ")
                toks = _split_cif_tokens(s)
                if length(toks) >= 2
                    val = String(toks[2])
                    if val != "?" && val != "."
                        _CCD_TYPE_CACHE[current_code] = val
                    end
                end
                continue
            elseif startswith(s, "_chem_comp.one_letter_code ")
                toks = _split_cif_tokens(s)
                if length(toks) >= 2
                    val = String(toks[2])
                    if val != "?" && val != "."
                        _CCD_ONE_LETTER_CACHE[current_code] = val
                    end
                end
                continue
            end

            if s == "loop_"
                headers = String[]
                while !eof(io)
                    h = strip(readline(io))
                    startswith(h, "_") || (pending = h; break)
                    push!(headers, h)
                end
                isempty(headers) && continue

                # ─── _chem_comp_atom table ───
                if all(startswith(h, "_chem_comp_atom.") for h in headers)
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
                    idx_leaving = findidx("_chem_comp_atom.pdbx_leaving_atom_flag")
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
                        atoms = NamedTuple{(:atom_name, :element, :x, :y, :z, :charge, :has_coord, :leaving), Tuple{String, String, Float32, Float32, Float32, Float32, Bool, Bool}}[]
                        for r in rows
                            atom_name = String(r[idx_atom])
                            element = uppercase(strip(String(r[idx_type])))
                            isempty(element) && (element = _infer_element_from_atom_name(atom_name))
                            element in ("H", "D", "T") && continue
                            q = _try_parse_i(r[idx_charge])
                            charge = q === nothing ? 0f0 : Float32(q)
                            is_leaving = idx_leaving !== nothing && uppercase(strip(r[idx_leaving])) == "Y"

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
                                (atom_name = atom_name, element = element, x = Float32(x), y = Float32(y), z = Float32(z), charge = charge, has_coord = has_coord, leaving = is_leaving),
                            )
                        end
                        if !haskey(_CCD_COMPONENT_CACHE, current_code)
                            _CCD_COMPONENT_CACHE[current_code] = atoms
                            push!(pending_bonds, current_code)
                        end
                        delete!(needed, current_code)
                    end

                # ─── _chem_comp_bond table ───
                elseif all(startswith(h, "_chem_comp_bond.") for h in headers)
                    findidx_b(name) = findfirst(==(name), headers)
                    idx_a1 = findidx_b("_chem_comp_bond.atom_id_1")
                    idx_a2 = findidx_b("_chem_comp_bond.atom_id_2")
                    if idx_a1 === nothing || idx_a2 === nothing
                        continue
                    end
                    idx_a1 = idx_a1::Int
                    idx_a2 = idx_a2::Int

                    bond_pairs = Tuple{String, String}[]
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
                        isempty(rs) && continue
                        if rs == "#" || rs == "loop_" || startswith(rs, "_") || startswith(rs, "data_")
                            pending = row_line
                            break
                        end
                        cols = _split_cif_tokens(row_line)
                        length(cols) < length(headers) && continue
                        push!(bond_pairs, (String(cols[idx_a1]), String(cols[idx_a2])))
                    end
                    if !isempty(bond_pairs)
                        _CCD_BOND_CACHE[current_code] = bond_pairs
                    end
                    delete!(pending_bonds, current_code)
                end
            end
        end
    end
end

function _ccd_component_atoms(code::AbstractString)
    uc = uppercase(strip(String(code)))
    isempty(uc) && return NamedTuple{(:atom_name, :element, :x, :y, :z, :charge, :has_coord, :leaving), Tuple{String, String, Float32, Float32, Float32, Float32, Bool, Bool}}[]
    _ensure_ccd_component_entries!(Set([uc]))
    return get(
        _CCD_COMPONENT_CACHE,
        uc,
        NamedTuple{(:atom_name, :element, :x, :y, :z, :charge, :has_coord, :leaving), Tuple{String, String, Float32, Float32, Float32, Float32, Bool, Bool}}[],
    )
end

"""
    _ccd_component_type(code) → String

Return the CCD `_chem_comp.type` for a component (e.g., "L-PEPTIDE LINKING", "NON-POLYMER").
Returns "" if the component is not found or the type was not parsed.
"""
function _ccd_component_type(code::AbstractString)::String
    uc = uppercase(strip(String(code)))
    isempty(uc) && return ""
    _ensure_ccd_component_entries!(Set([uc]))
    return get(_CCD_TYPE_CACHE, uc, "")
end

"""
    _ccd_one_letter_code(code) → String

Return the CCD `_chem_comp.one_letter_code` for a component (e.g., "S" for SEP → serine).
Returns "" if not found or missing.
"""
function _ccd_one_letter_code(code::AbstractString)::String
    uc = uppercase(strip(String(code)))
    isempty(uc) && return ""
    _ensure_ccd_component_entries!(Set([uc]))
    return get(_CCD_ONE_LETTER_CACHE, uc, "")
end

"""
    _ccd_mol_type(code) → String

Classify a CCD component as "protein", "dna", "rna", or "ligand" based on its
`_chem_comp.type` field, matching Python Protenix's `get_mol_type()`.

Rules (from http://mmcif.rcsb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_chem_comp.type.html):
- Contains "PEPTIDE" and is NOT "PEPTIDE-LIKE" → "protein"
- Contains "DNA" → "dna"
- Contains "RNA" → "rna"
- Otherwise (or not found) → "ligand"
"""
function _ccd_mol_type(code::AbstractString)::String
    comp_type = uppercase(_ccd_component_type(code))
    isempty(comp_type) && return "ligand"
    if occursin("PEPTIDE", comp_type) && comp_type != "PEPTIDE-LIKE"
        return "protein"
    elseif occursin("DNA", comp_type)
        return "dna"
    elseif occursin("RNA", comp_type)
        return "rna"
    end
    return "ligand"
end

"""
    _ccd_canonical_resname(mol_type, res_name) → String

Map a CCD component to its canonical standard residue name using the CCD
`one_letter_code` field. This matches Python Protenix's `add_cano_seq_resname()`.

For modified residues (SEP, TPO, etc.), the CCD one_letter_code provides the
parent amino acid (e.g., SEP → "S" → "SER").
"""
function _ccd_canonical_resname(mol_type::String, res_name::String)::String
    if mol_type == "protein"
        # First try standard lookup
        one_letter = get(PROT_THREE_TO_ONE, res_name, nothing)
        if one_letter !== nothing && one_letter != "X"
            return get(PROT_STD_RESIDUES_ONE_TO_THREE, one_letter, "UNK")
        end
        # Fallback: CCD one_letter_code for modified residues
        ccd_ol = _ccd_one_letter_code(res_name)
        if !isempty(ccd_ol) && length(ccd_ol) == 1
            return get(PROT_STD_RESIDUES_ONE_TO_THREE, ccd_ol, "UNK")
        end
        return "UNK"
    elseif mol_type == "dna"
        haskey(DNA_STD_RESIDUES, res_name) && return res_name
        ccd_ol = _ccd_one_letter_code(res_name)
        if !isempty(ccd_ol) && length(ccd_ol) == 1
            dna_name = "D" * ccd_ol
            return haskey(DNA_STD_RESIDUES, dna_name) ? dna_name : "DN"
        end
        return "DN"
    elseif mol_type == "rna"
        haskey(RNA_STD_RESIDUES_NATURAL, res_name) && return res_name
        ccd_ol = _ccd_one_letter_code(res_name)
        if !isempty(ccd_ol) && length(ccd_ol) == 1
            return haskey(RNA_STD_RESIDUES_NATURAL, ccd_ol) ? ccd_ol : "N"
        end
        return "N"
    else
        return "UNK"
    end
end

# Maps central atom name → Vector{Vector{String}} of leaving groups for a CCD component.
# Returns nothing if any leaving group is bonded to multiple central atoms (ambiguous).
function _ccd_central_to_leaving_groups(code::AbstractString)::Union{Nothing, Dict{String, Vector{Vector{String}}}}
    uc = uppercase(strip(String(code)))
    atoms = _ccd_component_atoms(uc)
    isempty(atoms) && return Dict{String, Vector{Vector{String}}}()
    bonds_raw = get(_CCD_BOND_CACHE, uc, Tuple{String, String}[])
    isempty(bonds_raw) && return Dict{String, Vector{Vector{String}}}()

    # Build name→index and leaving flag
    name_to_idx = Dict{String, Int}()
    for (i, a) in enumerate(atoms)
        name_to_idx[a.atom_name] = i
    end
    is_leaving = [a.leaving for a in atoms]
    n = length(atoms)

    # Build adjacency (among non-H atoms already in the cache)
    adj = [Int[] for _ in 1:n]
    for (a1, a2) in bonds_raw
        i = get(name_to_idx, a1, 0)
        j = get(name_to_idx, a2, 0)
        (i == 0 || j == 0) && continue
        push!(adj[i], j)
        push!(adj[j], i)
    end

    # Mutable copy of adjacency for bond-removal during traversal
    adj_mut = [Set{Int}(a) for a in adj]
    result = Dict{String, Vector{Vector{String}}}()

    for c_idx in 1:n
        is_leaving[c_idx] && continue  # only central atoms
        for l_idx in adj[c_idx]
            is_leaving[l_idx] || continue  # only leaving neighbours
            # Remove the central↔leaving bond
            delete!(adj_mut[c_idx], l_idx)
            delete!(adj_mut[l_idx], c_idx)
            # BFS to find connected component of leaving atom
            group = Int[l_idx]
            visited = Set{Int}([l_idx])
            queue = Int[l_idx]
            while !isempty(queue)
                cur = popfirst!(queue)
                for nb in adj_mut[cur]
                    nb in visited && continue
                    push!(visited, nb)
                    push!(group, nb)
                    push!(queue, nb)
                end
            end
            # All atoms in the group must be leaving; otherwise ambiguous
            if !all(is_leaving[g] for g in group)
                return nothing
            end
            group_names = [atoms[g].atom_name for g in group]
            cname = atoms[c_idx].atom_name
            if haskey(result, cname)
                push!(result[cname], group_names)
            else
                result[cname] = [group_names]
            end
        end
    end
    return result
end

"""
    _apply_mse_to_met(atoms) → Vector{AtomRecord}

Convert MSE (selenomethionine) residues to MET (methionine), matching Python
Protenix's `mse_to_met()`. This must be called before tokenization so that
MSE residues are treated as standard MET (1 token) rather than per-atom modified.

Conversions: res_name MSE → MET, atom SE → SD, element SE → S.
"""
function _apply_mse_to_met(atoms::Vector{AtomRecord})::Vector{AtomRecord}
    result = similar(atoms)
    for (i, a) in enumerate(atoms)
        if a.res_name == "MSE"
            atom_name = a.atom_name == "SE" ? "SD" : a.atom_name
            element = uppercase(a.element) == "SE" ? "S" : a.element
            result[i] = AtomRecord(
                atom_name, "MET", a.mol_type, element, a.chain_id, a.res_id,
                a.mol_type == "protein" ? (atom_name == "CA") : a.centre_atom_mask,
                a.x, a.y, a.z, a.is_resolved,
            )
        else
            result[i] = a
        end
    end
    return result
end

"""
    _apply_ccd_mol_type_override(atoms, polymer_chain_ids; all_entities=false) → Vector{AtomRecord}

Override mol_type for atoms based on CCD `_chem_comp.type` classification.

When `all_entities=false` (default, v0.5 behavior): only polymer entities are
checked, matching Python Protenix v0.5's `add_token_mol_type()` which gates
the CCD lookup on `entity_poly_type`.

When `all_entities=true` (v1.0 behavior): ALL entities including ligand/ion
are checked. This matches Python Protenix v1.0's inference path where
`build_ligand()` adds a fake "sequence" key to ligand entities, causing them
to be included in `entity_poly_type` and thus CCD-lookup'd. This means CCD
compounds with protein-type codes (e.g. 4HT "L-PEPTIDE LINKING") declared as
ligand entities get reclassified to protein, affecting tokenization, restype,
and is_protein/is_ligand flags.

Examples:
- Modified residue SEP in a proteinChain → CCD "L-PEPTIDE LINKING" → "protein"
- DNA base 5MC in a dnaSequence → CCD "RNA LINKING" → "rna" (even in a DNA chain)
- Ligand 4HT (all_entities=false): stays "ligand" (v0.5 entity gate)
- Ligand 4HT (all_entities=true): → "protein" (v1.0 CCD reclassification)
"""
function _apply_ccd_mol_type_override(
    atoms::Vector{AtomRecord},
    polymer_chain_ids::Set{String};
    all_entities::Bool = false,
)::Vector{AtomRecord}
    result = similar(atoms)
    for (i, a) in enumerate(atoms)
        if !all_entities && a.chain_id ∉ polymer_chain_ids
            # v0.5 mode: skip non-polymer entities
            result[i] = a
            continue
        end
        ccd_mt = _ccd_mol_type(a.res_name)
        if ccd_mt != a.mol_type && ccd_mt != "ligand"
            # CCD classifies this residue differently than the entity type.
            # Override mol_type and adjust centre atom mask accordingly.
            # Guard: never override TO "ligand" (only override to polymer types).
            centre = ccd_mt == "protein" ? (a.atom_name == "CA") : (a.atom_name == "C1'")
            result[i] = AtomRecord(
                a.atom_name, a.res_name, ccd_mt, a.element, a.chain_id, a.res_id,
                centre, a.x, a.y, a.z, a.is_resolved,
            )
        else
            result[i] = a
        end
    end
    return result
end

"""
    _fix_restype_for_modified_residues!(feat, atoms, tokens)

Post-process the `restype` feature to use CCD one_letter_code for mapping modified
residues to their parent amino acid. This matches Python Protenix's `add_cano_seq_resname()`.

For standard residues, restype is already correct. For modified residues (SEP, TPO, etc.),
the CCD one_letter_code provides the parent amino acid (SEP → "S" → SER, restype index 15).
"""
function _fix_restype_for_modified_residues!(
    feat::Dict{String, Any},
    atoms::Vector{AtomRecord},
    tokens,
)
    centre_idx = Features.centre_atom_indices(tokens)
    restype = feat["restype"]
    n_classes = size(restype, 2)
    changed = false
    for (ti, ai) in enumerate(centre_idx)
        a = atoms[ai]
        canonical = _ccd_canonical_resname(a.mol_type, a.res_name)
        # Look up the standard residue index
        restype_id = get(STD_RESIDUES_WITH_GAP, canonical, nothing)
        restype_id === nothing && continue
        # Check if current restype matches
        current_max = 0
        current_id = 0
        for k in 1:n_classes
            if restype[ti, k] > current_max
                current_max = restype[ti, k]
                current_id = k - 1  # 0-indexed
            end
        end
        if current_id != restype_id
            restype[ti, :] .= 0
            restype[ti, restype_id + 1] = 1  # 0-indexed → 1-indexed
            changed = true
        end
    end
    if changed
        # Also update profile (first 32 columns of restype) and MSA query row
        feat["restype"] = restype
    end
end

@doc """
    _fix_entity_and_sym_ids!(feat, atoms, tokens, entity_chain_ids)

Fix entity_id and sym_id features to match Python Protenix's `unique_chain_and_add_ids()`.

Julia's default sets `entity_id = asym_id` (each chain gets a unique entity), but Python
groups chains by their entity (same sequence → same entity_id). sym_id tracks which copy
of an entity a chain is.

Example for homodimer + ligand:
- Python: chain A → entity_id=0, sym_id=0; chain A' → entity_id=0, sym_id=1; ligand → entity_id=1, sym_id=0
- Julia default: chain A → entity_id=0; chain A' → entity_id=1; ligand → entity_id=2 (all sym_id=0)
"""
function _fix_entity_and_sym_ids!(
    feat::Dict{String, Any},
    atoms::Vector{AtomRecord},
    tokens,
    entity_chain_ids::Vector{Vector{String}},
)
    centre_idx = Features.centre_atom_indices(tokens)
    n_token = length(tokens)

    # Build chain_id → (entity_idx, sym_idx) mapping from entity_chain_ids
    # entity_chain_ids[i] = [chain_id_1, chain_id_2, ...] for entity i
    chain_to_entity = Dict{String, Int}()  # chain_id → 0-based entity index
    chain_to_sym = Dict{String, Int}()     # chain_id → 0-based sym index within entity
    for (eidx, chain_ids) in enumerate(entity_chain_ids)
        for (sidx, cid) in enumerate(chain_ids)
            chain_to_entity[cid] = eidx - 1  # 0-based
            chain_to_sym[cid] = sidx - 1     # 0-based
        end
    end

    entity_id = feat["entity_id"]
    sym_id = feat["sym_id"]
    for (ti, ai) in enumerate(centre_idx)
        cid = atoms[ai].chain_id
        if haskey(chain_to_entity, cid)
            entity_id[ti] = chain_to_entity[cid]
            sym_id[ti] = chain_to_sym[cid]
        end
    end
    feat["entity_id"] = entity_id
    feat["sym_id"] = sym_id
end

# Remove leaving atoms from a flat atom vector based on covalent bond definitions.
# Mirrors Python's add_covalent_bonds → remove_leaving_atoms flow.
# Uses rng for random group selection to match Python's random.sample behavior.
function _remove_covalent_leaving_atoms(
    atoms::Vector{AtomRecord},
    task::AbstractDict{<:Any, <:Any},
    entity_chain_ids::Vector{Vector{String}},
    entity_atom_map::Vector{Dict{Int, String}};
    rng::AbstractRNG = Random.default_rng(),
)::Vector{AtomRecord}
    haskey(task, "covalent_bonds") || return atoms
    bonds_any = task["covalent_bonds"]
    bonds_any isa AbstractVector || return atoms
    isempty(bonds_any) && return atoms

    # Build atom lookup: (chain_id, res_id, atom_name) → [atom_indices...]
    atom_lookup = Dict{Tuple{String, Int, String}, Vector{Int}}()
    for (atom_idx, atom) in enumerate(atoms)
        key = (atom.chain_id, atom.res_id, uppercase(atom.atom_name))
        if haskey(atom_lookup, key)
            push!(atom_lookup[key], atom_idx)
        else
            atom_lookup[key] = [atom_idx]
        end
    end

    # Count bonds per atom index (how many inter-residue covalent bonds each atom participates in)
    bond_count = Dict{Int, Int}()
    for (bond_i, bond_any) in enumerate(bonds_any)
        bond_any isa AbstractDict || continue
        bond = _as_string_dict(bond_any)

        entity1_any = _bond_field(bond, "left", 1, "entity")
        entity2_any = _bond_field(bond, "right", 2, "entity")
        position1_any = _bond_field(bond, "left", 1, "position")
        position2_any = _bond_field(bond, "right", 2, "position")
        atom1_any = _bond_field(bond, "left", 1, "atom")
        atom2_any = _bond_field(bond, "right", 2, "atom")
        copy1_any = _bond_field(bond, "left", 1, "copy")
        copy2_any = _bond_field(bond, "right", 2, "copy")

        (entity1_any === nothing || entity2_any === nothing) && continue
        (position1_any === nothing || position2_any === nothing) && continue
        (atom1_any === nothing || atom2_any === nothing) && continue

        entity1 = entity1_any isa Integer ? Int(entity1_any) : parse(Int, strip(String(entity1_any)))
        entity2 = entity2_any isa Integer ? Int(entity2_any) : parse(Int, strip(String(entity2_any)))
        position1 = position1_any isa Integer ? Int(position1_any) : parse(Int, strip(String(position1_any)))
        position2 = position2_any isa Integer ? Int(position2_any) : parse(Int, strip(String(position2_any)))
        ctx = "covalent_bonds[$bond_i]"
        atom1 = _resolve_bond_atom_name(atom1_any, entity1, entity_atom_map, "$ctx side1")
        atom2 = _resolve_bond_atom_name(atom2_any, entity2, entity_atom_map, "$ctx side2")

        chains1 = _resolve_bond_chains(entity_chain_ids, entity1, copy1_any, "$ctx side1")
        chains2 = _resolve_bond_chains(entity_chain_ids, entity2, copy2_any, "$ctx side2")
        length(chains1) == length(chains2) || continue

        atoms1 = _resolve_bond_atoms(atom_lookup, chains1, position1, atom1, "$ctx side1")
        atoms2 = _resolve_bond_atoms(atom_lookup, chains2, position2, atom2, "$ctx side2")
        length(atoms1) == length(atoms2) || continue

        for (a1, a2) in zip(atoms1, atoms2)
            bond_count[a1] = get(bond_count, a1, 0) + 1
            bond_count[a2] = get(bond_count, a2, 0) + 1
        end
    end

    isempty(bond_count) && return atoms

    # Build residue start indices: for each (chain_id, res_id), the range of atom indices
    chain_res_ranges = Dict{Tuple{String, Int}, UnitRange{Int}}()
    i = 1
    while i <= length(atoms)
        cid = atoms[i].chain_id
        rid = atoms[i].res_id
        j = i
        while j < length(atoms) && atoms[j + 1].chain_id == cid && atoms[j + 1].res_id == rid
            j += 1
        end
        chain_res_ranges[(cid, rid)] = i:j
        i = j + 1
    end

    # Determine which atom indices to remove
    remove_set = Set{Int}()
    for (centre_idx, b_count) in bond_count
        a = atoms[centre_idx]
        res_name = uppercase(a.res_name)
        centre_name = uppercase(a.atom_name)

        leaving_groups = _ccd_central_to_leaving_groups(res_name)
        leaving_groups === nothing && continue
        groups = get(leaving_groups, centre_name, nothing)
        groups === nothing && continue

        n_remove = min(b_count, length(groups))
        res_range = get(chain_res_ranges, (a.chain_id, a.res_id), 0:0)
        isempty(res_range) && continue

        # Random selection of leaving groups to match Python's random.sample(leaving_groups, b_count)
        if n_remove < length(groups)
            selected_indices = sort(shuffle(rng, collect(1:length(groups)))[1:n_remove])
        else
            selected_indices = collect(1:length(groups))
        end

        for gi in selected_indices
            for leaving_name in groups[gi]
                for k in res_range
                    if uppercase(atoms[k].atom_name) == uppercase(leaving_name)
                        push!(remove_set, k)
                        break
                    end
                end
            end
        end
    end

    isempty(remove_set) && return atoms
    return [atoms[i] for i in 1:length(atoms) if !(i in remove_set)]
end

# Remove leaving atoms from disconnected polymer residues.
# Mirrors Python's _remove_non_std_ccd_leaving_atoms: for adjacent residues in the same
# chain that lack the expected inter-residue bond (C-N for protein, O3'-P for nucleic),
# remove ALL leaving atoms from both residues.
function _remove_non_std_polymer_leaving_atoms(atoms::Vector{AtomRecord})::Vector{AtomRecord}
    isempty(atoms) && return atoms

    # Group atoms into residues: (chain_id, res_id) → range of indices
    residues = Tuple{String, Int, UnitRange{Int}}[]  # (chain_id, res_id, index_range)
    i = 1
    while i <= length(atoms)
        cid = atoms[i].chain_id
        rid = atoms[i].res_id
        j = i
        while j < length(atoms) && atoms[j + 1].chain_id == cid && atoms[j + 1].res_id == rid
            j += 1
        end
        push!(residues, (cid, rid, i:j))
        i = j + 1
    end

    length(residues) < 2 && return atoms

    # Check connectivity between adjacent residues
    # connected[k] = true means residues[k] and residues[k+1] are bonded
    connected = falses(length(residues) - 1)
    for k in 1:(length(residues) - 1)
        cid1, rid1, rng1 = residues[k]
        cid2, rid2, rng2 = residues[k + 1]

        # Must be same chain and consecutive res_ids
        cid1 == cid2 || continue
        abs(rid2 - rid1) <= 1 || continue

        # Determine mol_type from atoms
        mt1 = atoms[first(rng1)].mol_type
        mt2 = atoms[first(rng2)].mol_type

        # Check for connecting atoms based on mol_type
        if mt1 == "protein" && mt2 == "protein"
            has_c = any(uppercase(atoms[idx].atom_name) == "C" for idx in rng1)
            has_n = any(uppercase(atoms[idx].atom_name) == "N" for idx in rng2)
            connected[k] = has_c && has_n
        elseif (mt1 == "dna" || mt1 == "rna") && (mt2 == "dna" || mt2 == "rna")
            has_o3p = any(uppercase(atoms[idx].atom_name) == "O3'" for idx in rng1)
            has_p = any(uppercase(atoms[idx].atom_name) == "P" for idx in rng2)
            connected[k] = has_o3p && has_p
        else
            # Different or incompatible types — not a polymer bond
            connected[k] = true  # treat as "not disconnected"
        end
    end

    # Find residues involved in disconnections
    disconnected_residue_indices = Set{Int}()
    for k in 1:(length(residues) - 1)
        connected[k] && continue
        # Only flag if they're in the same chain with consecutive res_ids
        cid1, rid1, _ = residues[k]
        cid2, rid2, _ = residues[k + 1]
        cid1 == cid2 || continue
        abs(rid2 - rid1) <= 1 || continue
        push!(disconnected_residue_indices, k)
        push!(disconnected_residue_indices, k + 1)
    end

    isempty(disconnected_residue_indices) && return atoms

    # For each disconnected residue, determine which atoms to keep (non-leaving)
    remove_set = Set{Int}()
    n_res = length(residues)
    for res_k in disconnected_residue_indices
        _, rid, rng = residues[res_k]
        res_name = uppercase(atoms[first(rng)].res_name)
        mol_type = atoms[first(rng)].mol_type

        # Get CCD staying atom names (non-leaving, non-hydrogen)
        comp_atoms = _ccd_component_atoms(res_name)
        staying_names = Set{String}()
        for ca in comp_atoms
            ca.leaving && continue
            push!(staying_names, uppercase(ca.atom_name))
        end

        # Special cases matching Python: first nucleic keeps OP3, last protein keeps OXT
        if res_k == 1 && (mol_type == "dna" || mol_type == "rna")
            push!(staying_names, "OP3")
        end
        if res_k == n_res && mol_type == "protein"
            push!(staying_names, "OXT")
        end

        for idx in rng
            if uppercase(atoms[idx].atom_name) ∉ staying_names
                push!(remove_set, idx)
            end
        end
    end

    isempty(remove_set) && return atoms
    @warn "Removed $(length(remove_set)) leaving atoms from disconnected polymer residues"
    return [atoms[i] for i in 1:length(atoms) if !(i in remove_set)]
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
                atom_name == "OP3" && (mol_type == "dna" || mol_type == "rna") && res_id > 1 && continue
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

        # For short codes (1-2 chars), treat the code as a single-atom ion/element.
        # For longer codes (e.g. ATP, FAD), CCD component data is required.
        if length(code) <= 2
            element = uppercase(code)
            atom_name = code
            push!(
                atoms,
                AtomRecord(atom_name, code, "ligand", element, chain_id, res_id, true, xoff, 0f0, 0f0, is_resolved),
            )
        else
            error(
                "CCD component data not found for ligand code '$(code)'. " *
                "Ensure the CCD components file is available (set PROTENIX_DATA_ROOT_DIR or place " *
                "components.v20240608.cif in release_data/ccd_cache/)."
            )
        end
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

# Stub: overridden by ProtenixMoleculeFlowExt when MoleculeFlow is loaded.
# Returns (atoms::Vector{AtomRecord}, bonds::Vector{Tuple{String,String}})
function _smiles_to_atoms_and_bonds end

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

function _build_ligand_atoms_from_smiles(smiles::AbstractString; chain_id::String, is_resolved::Bool = true)
    s = strip(String(smiles))
    isempty(s) && error("SMILES ligand string cannot be empty")

    # Use MoleculeFlow extension for real 3D conformers + bonds when available
    if hasmethod(_smiles_to_atoms_and_bonds, Tuple{String, String})
        mf_atoms, mf_bonds = _smiles_to_atoms_and_bonds(String(s), chain_id)
        isempty(mf_atoms) && error("MoleculeFlow returned no atoms for SMILES: $smiles")
        # Inject bonds into CCD cache so _compute_token_bonds picks them up
        if !isempty(mf_bonds)
            Features._CCD_BOND_CACHE["UNL"] = mf_bonds
        end
        atom_map = Dict{Int, String}()
        for k in 1:length(mf_atoms)
            atom_map[k] = mf_atoms[k].atom_name
        end
        return mf_atoms, atom_map
    end

    # Fallback: parse SMILES manually (no bonds, spiral placeholder coords)
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

    loaded, sdf_bonds = load_structure_atoms(path)
    atoms = AtomRecord[]
    atom_map = Dict{Int, String}()
    # Build old→new atom name mapping for SDF bonds
    old_to_new_name = Dict{String, String}()
    atom_counter = 0
    for a in loaded
        atom_counter += 1
        atom_name = uppercase(strip(a.atom_name))
        atom_map[atom_counter] = atom_name
        old_to_new_name[a.atom_name] = atom_name
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

    # Inject SDF bonds into the CCD bond cache so _compute_token_bonds picks them up
    if !isempty(sdf_bonds)
        res_name = atoms[1].res_name
        mapped_bonds = Tuple{String, String}[]
        for (a1, a2) in sdf_bonds
            n1 = get(old_to_new_name, a1, "")
            n2 = get(old_to_new_name, a2, "")
            (!isempty(n1) && !isempty(n2)) && push!(mapped_bonds, (n1, n2))
        end
        Features._CCD_BOND_CACHE[uppercase(res_name)] = mapped_bonds
    end

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
            msa_cfg = haskey(pc, "msa") ? _parse_msa_cfg(pc["msa"]) : _DEFAULT_MSA_CFG
            for _ in 1:count
                chain_id = _chain_id_from_index(chain_idx)
                chain_idx += 1
                push!(specs, ProteinChainSpec(chain_id, seq, msa_cfg))
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

function _extract_rna_chain_specs(task::AbstractDict{<:Any, <:Any}; json_dir::AbstractString = ".")
    haskey(task, "sequences") || error("Task is missing required field: sequences")
    sequences = task["sequences"]
    sequences isa AbstractVector || error("Task.sequences must be an array")

    specs = RNAChainSpec[]
    chain_idx = 1
    for (i, entity_any) in enumerate(sequences)
        entity_any isa AbstractDict || error("Task.sequences[$i] must be an object")
        entity = _as_string_dict(entity_any)
        if haskey(entity, "rnaSequence")
            rna_any = entity["rnaSequence"]
            rna_any isa AbstractDict || error("Task.sequences[$i].rnaSequence must be an object")
            rna = _as_string_dict(rna_any)
            haskey(rna, "sequence") || error("Task.sequences[$i].rnaSequence.sequence is required")
            seq = _compact_sequence(String(rna["sequence"]))
            isempty(seq) && error("Task.sequences[$i].rnaSequence.sequence must be non-empty")
            count = Int(get(rna, "count", 1))
            count > 0 || error("Task.sequences[$i].rnaSequence.count must be positive")
            unpaired_msa = if haskey(rna, "unpairedMsa")
                v = rna["unpairedMsa"]; v === nothing ? nothing : String(v)
            else
                nothing
            end
            unpaired_msa_path = if haskey(rna, "unpairedMsaPath")
                v = rna["unpairedMsaPath"]
                if v === nothing; nothing
                else
                    raw = String(v)
                    isabspath(raw) ? raw : normpath(joinpath(json_dir, raw))
                end
            else
                nothing
            end
            for _ in 1:count
                chain_id = _chain_id_from_index(chain_idx)
                chain_idx += 1
                push!(specs, RNAChainSpec(chain_id, seq, unpaired_msa, unpaired_msa_path))
            end
            continue
        end

        local count::Int
        if haskey(entity, "proteinChain")
            count = Int(get(_as_string_dict(entity["proteinChain"]), "count", 1))
        elseif haskey(entity, "dnaSequence")
            count = Int(get(_as_string_dict(entity["dnaSequence"]), "count", 1))
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
    rna_specs = RNAChainSpec[]
    dna_specs = DNAChainSpec[]
    entity_chain_ids = Vector{Vector{String}}()
    entity_atom_map = Vector{Dict{Int, String}}()
    polymer_chain_ids = Set{String}()
    ion_chain_ids = Set{String}()
    chain_idx = 1

    for (entity_idx, entity_any) in enumerate(sequences)
        entity_any isa AbstractDict || error("Task.sequences[$entity_idx] must be an object")
        entity = _as_string_dict(entity_any)
        chain_ids = String[]
        atom_map = Dict{Int, String}()

        if haskey(entity, "proteinChain")
            pc_any = entity["proteinChain"]
            pc_any isa AbstractDict || error("Task.sequences[$entity_idx].proteinChain must be an object")
            pc = _as_string_dict(pc_any)
            haskey(pc, "sequence") || error("Task.sequences[$entity_idx].proteinChain.sequence is required")
            seq = _compact_sequence(String(pc["sequence"]))
            isempty(seq) && error("Task.sequences[$entity_idx].proteinChain.sequence must be non-empty")
            count = Int(get(pc, "count", 1))
            count > 0 || error("Task.sequences[$entity_idx].proteinChain.count must be positive")
            msa_cfg = haskey(pc, "msa") ? _parse_msa_cfg(pc["msa"]) : _DEFAULT_MSA_CFG

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
                push!(polymer_chain_ids, chain_id)
                if has_mods
                    append!(atoms, _build_polymer_atoms_from_codes(ccd_codes, "protein"; chain_id = chain_id))
                else
                    append!(atoms, ProtenixMini.build_sequence_atoms(seq; chain_id = chain_id))
                end
                push!(protein_specs, ProteinChainSpec(chain_id, seq, msa_cfg))
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
                push!(polymer_chain_ids, chain_id)
                append!(atoms, _build_polymer_atoms_from_codes(ccd_codes, "dna"; chain_id = chain_id))
                push!(dna_specs, DNAChainSpec(chain_id, seq, copy(ccd_codes)))
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
            # Extract RNA MSA fields (v1 feature: unpairedMsa / unpairedMsaPath).
            rna_unpaired_msa = if haskey(rna, "unpairedMsa")
                v = rna["unpairedMsa"]
                v === nothing ? nothing : String(v)
            else
                nothing
            end
            rna_unpaired_msa_path = if haskey(rna, "unpairedMsaPath")
                v = rna["unpairedMsaPath"]
                if v === nothing
                    nothing
                else
                    raw = String(v)
                    isabspath(raw) ? raw : normpath(joinpath(json_dir, raw))
                end
            else
                nothing
            end
            for _ in 1:count
                chain_id = _chain_id_from_index(chain_idx)
                chain_idx += 1
                push!(chain_ids, chain_id)
                push!(polymer_chain_ids, chain_id)
                append!(atoms, _build_polymer_atoms_from_codes(ccd_codes, "rna"; chain_id = chain_id))
                push!(rna_specs, RNAChainSpec(chain_id, seq, rna_unpaired_msa, rna_unpaired_msa_path))
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
                    lig_atoms, built_atom_map = _build_ligand_atoms_from_file(
                        file_spec;
                        chain_id = chain_id,
                        json_dir = json_dir,
                    )
                    append!(atoms, lig_atoms)
                    isempty(inferred_map) && (inferred_map = built_atom_map)
                end
                atom_map = isempty(provided_map) ? inferred_map : provided_map
            else
                smiles_str = startswith(ligand_uc, "SMILES_") ? strip(ligand_str[8:end]) : ligand_str
                isempty(smiles_str) && error("Task.sequences[$entity_idx].$lig_key.ligand SMILES payload must be non-empty")
                provided_map = haskey(lig, "atom_map_to_atom_name") ? _normalize_ligand_atom_map(
                    lig["atom_map_to_atom_name"],
                    "Task.sequences[$entity_idx].$lig_key",
                ) : Dict{Int, String}()
                inferred_map = Dict{Int, String}()
                for _ in 1:count
                    chain_id = _chain_id_from_index(chain_idx)
                    chain_idx += 1
                    push!(chain_ids, chain_id)
                    lig_atoms, built_atom_map = _build_ligand_atoms_from_smiles(smiles_str; chain_id = chain_id)
                    append!(atoms, lig_atoms)
                    isempty(inferred_map) && (inferred_map = built_atom_map)
                end
                atom_map = isempty(provided_map) ? inferred_map : provided_map
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
                push!(ion_chain_ids, chain_id)
                append!(atoms, _build_ligand_atoms_from_codes([ion_code]; chain_id = chain_id))
            end
        else
            keys_str = join(string.(collect(keys(entity))), ", ")
            error("Unsupported sequence entry keys at sequences[$entity_idx]: [$keys_str]")
        end

        push!(entity_chain_ids, chain_ids)
        push!(entity_atom_map, atom_map)
    end

    isempty(atoms) && error("No supported sequence entities found in infer task")
    # Remove leaving atoms from disconnected polymer residues (matches Python's
    # _remove_non_std_ccd_leaving_atoms called per-chain during build_polymer_chain).
    atoms = _remove_non_std_polymer_leaving_atoms(atoms)
    return TaskEntityParseResult(atoms, protein_specs, rna_specs, dna_specs, entity_chain_ids, entity_atom_map, polymer_chain_ids, ion_chain_ids)
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

# ── MSA file format utilities ─────────────────────────────────────────────────

"""
    _infer_alignment_type(path) → Symbol

Infer MSA alignment type from file extension.
- `.a3m` → `:a3m` (lowercase = insertions, uppercase + dash = match states)
- `.fasta`, `.fa`, `.fas`, `.faa` → `:fasta` (no case semantics, all chars are match states)
- Unknown → `:a3m` (conservative default)
"""
function _infer_alignment_type(path::AbstractString)::Symbol
    ext = lowercase(splitext(path)[2])
    ext == ".a3m" && return :a3m
    ext in (".fasta", ".fa", ".fas", ".faa") && return :fasta
    return :a3m
end

"""
    _find_msa_file(dir, stem) → Union{String, Nothing}

Search for an MSA file in `dir` with the given stem and any supported extension.
Tries `.a3m` first, then `.fasta`, `.fa`, `.fas`.  Returns `nothing` if not found.
"""
function _find_msa_file(dir::AbstractString, stem::AbstractString)
    for ext in (".a3m", ".fasta", ".fa", ".fas")
        path = joinpath(dir, stem * ext)
        isfile(path) && return path
    end
    return nothing
end

"""
    _aligned_and_deletions_from_fasta(sequences) → (aligned, deletion_matrix)

Process raw FASTA-format MSA sequences.  In FASTA alignments there are no case
semantics — every character is a match state.  Lowercase is uppercased, `.` is
converted to `-` (gap).  The deletion matrix is all zeros.
"""
function _aligned_and_deletions_from_fasta(sequences::Vector{String})
    aligned = String[]
    deletion_matrix = Vector{Vector{Float32}}()
    for raw in sequences
        aln = uppercase(replace(raw, '.' => '-'))
        isempty(aln) && continue
        push!(aligned, aln)
        push!(deletion_matrix, zeros(Float32, length(aln)))
    end
    return aligned, deletion_matrix
end

"""
    _parse_and_align_msa(path; alignment_type=:infer, seq_limit=-1)
        → (aligned, deletion_matrix, descriptions)

Unified MSA parser that handles both a3m and fasta formats.

- `:a3m` — lowercase = insertions (counted in deletion matrix, stripped from aligned
  sequences).  After stripping, all rows have the same length as the query (row 0).
- `:fasta` — no case semantics; all characters are match states; deletion matrix is
  all zeros.
- `:infer` (default) — detect from file extension.
"""
function _parse_and_align_msa(path::AbstractString; alignment_type::Symbol = :infer, seq_limit::Int = -1)
    isfile(path) || error("MSA file not found: $path")
    if alignment_type == :infer
        alignment_type = _infer_alignment_type(path)
    end
    alignment_type in (:a3m, :fasta) || error(
        "Unknown alignment_type=$(repr(alignment_type)); expected :a3m, :fasta, or :infer",
    )

    seqs, descs = _parse_a3m(path; seq_limit = seq_limit)  # FASTA header format is compatible

    if alignment_type == :a3m
        aln, del = _aligned_and_deletions_from_a3m(seqs)
    else
        aln, del = _aligned_and_deletions_from_fasta(seqs)
    end

    return aln, del, descs
end

"""
    _check_msa_query_match(aligned_query, input_sequence, msa_path)

Compare the MSA query (row 0, gaps removed) against the input chain sequence.
Logs a warning with quantified mismatch details if they don't correspond.

Small mismatches (≤5 AAs) can occur legitimately from:
- Non-standard amino acids mapped to X vs their parent (e.g. MSE → M in MSA, X in input)
- Expression tags or cloning artifacts in the MSA search query
- Point mutations between MSA source and modeling target

Large mismatches almost certainly mean the wrong MSA was provided for this chain.
"""
function _check_msa_query_match(
    aligned_query::AbstractString,
    input_sequence::AbstractString,
    msa_path::AbstractString,
)
    ungapped = filter(c -> c != '-', uppercase(aligned_query))
    input_up = uppercase(strip(input_sequence))

    ungapped == input_up && return  # exact match

    ulen = length(ungapped)
    ilen = length(input_up)

    if ulen == ilen
        n_mis = count(i -> ungapped[i] != input_up[i], 1:ulen)
        if n_mis <= 5
            @warn("MSA query has $n_mis/$ulen AA mismatches vs input sequence " *
                  "(possibly non-standard residues or expression-tag differences)",
                  msa_path)
        else
            pct = round(100.0 * n_mis / ulen; digits = 1)
            @warn("MSA query has $n_mis/$ulen AA mismatches ($pct%) vs input sequence " *
                  "— likely wrong MSA for this chain",
                  msa_path)
        end
    else
        overlap = min(ulen, ilen)
        n_mis = overlap > 0 ? count(i -> ungapped[i] != input_up[i], 1:overlap) : 0
        pct = overlap > 0 ? round(100.0 * n_mis / overlap; digits = 1) : 100.0
        @warn("MSA query length ($ulen) ≠ input sequence length ($ilen); " *
              "$n_mis/$overlap mismatches in overlap ($pct%) — " *
              "MSA may not correspond to this chain",
              msa_path)
    end
end

# ── Original a3m parser ───────────────────────────────────────────────────────

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

function _pairing_key_from_description(desc::AbstractString, row_idx::Int)
    row_idx == 1 && return "__query__"
    s = lowercase(String(desc))
    for pat in (
        r"taxid[=\s:]+([0-9]+)",
        r"ncbi_taxid[=\s:]+([0-9]+)",
        r"\box=([0-9]+)",
    )
        m = match(pat, s)
        m === nothing && continue
        id = m.captures[1]
        isempty(id) || return "taxid:" * id
    end
    m_tax = match(r"\btax=([^\s;|]+)", s)
    if m_tax !== nothing
        tax = strip(m_tax.captures[1])
        isempty(tax) || return "tax:" * tax
    end
    m_os = match(r"\bos=([^=]+?)(?:\box=|\bgn=|\bpe=|\bsv=|$)", s)
    if m_os !== nothing
        os = replace(strip(m_os.captures[1]), r"\s+" => " ")
        isempty(os) || return "os:" * os
    end
    return ""
end

function _dedup_aligned_deletion_rows(
    aligned::Vector{String},
    deletion_matrix::Vector{Vector{Float32}};
    row_keys::Union{Nothing, Vector{String}} = nothing,
)
    if row_keys !== nothing && length(row_keys) != length(aligned)
        error("row_keys length mismatch for MSA deduplication.")
    end
    seen = Set{String}()
    dedup_aligned = String[]
    dedup_del = Vector{Vector{Float32}}()
    dedup_keys = row_keys === nothing ? nothing : String[]
    for i in eachindex(aligned)
        seq = aligned[i]
        seq in seen && continue
        push!(seen, seq)
        push!(dedup_aligned, seq)
        push!(dedup_del, deletion_matrix[i])
        if dedup_keys !== nothing
            push!(dedup_keys, (row_keys::Vector{String})[i])
        end
    end
    return dedup_aligned, dedup_del, dedup_keys
end

function _pairing_row_plan(chain_features::Vector{ChainMSAFeatures})
    n_chain = length(chain_features)
    n_chain > 1 || return nothing
    all(cf -> cf.pairing !== nothing && cf.pairing_keys !== nothing, chain_features) || return nothing

    row_lookup = Vector{Dict{String, Vector{Int}}}(undef, n_chain)
    ordered_keys = Vector{Vector{String}}(undef, n_chain)
    for i in 1:n_chain
        pairing = chain_features[i].pairing::ChainMSABlock
        keys = chain_features[i].pairing_keys::Vector{String}
        size(pairing.msa, 1) == length(keys) || return nothing
        lookup = Dict{String, Vector{Int}}()
        for (r, key) in enumerate(keys)
            isempty(key) && continue
            push!(get!(lookup, key, Int[]), r)
        end
        row_lookup[i] = lookup
        ordered_keys[i] = keys
    end

    plan = Vector{Vector{Int}}()
    if all(haskey(row_lookup[i], "__query__") && !isempty(row_lookup[i]["__query__"]) for i in 1:n_chain)
        push!(plan, [popfirst!(row_lookup[i]["__query__"]) for i in 1:n_chain])
    end

    seen = Set{String}()
    for key in ordered_keys[1]
        (isempty(key) || key == "__query__" || key in seen) && continue
        push!(seen, key)
        while all(haskey(row_lookup[i], key) && !isempty(row_lookup[i][key]) for i in 1:n_chain)
            push!(plan, [popfirst!(row_lookup[i][key]) for i in 1:n_chain])
        end
    end

    # Require at least query + one taxonomically paired row before enabling key-based merge.
    length(plan) >= 2 || return nothing
    return plan
end

function _build_chain_msa_features(
    sequence::AbstractString,
    aligned::Vector{String},
    deletion_matrix::Vector{Vector{Float32}},
    ;
    dedup_rows::Bool = true,
)::ChainMSABlock
    if isempty(aligned)
        query = uppercase(strip(sequence))
        push!(aligned, query)
        push!(deletion_matrix, zeros(Float32, length(query)))
    end

    dedup_aligned = aligned
    dedup_del = deletion_matrix
    if dedup_rows
        dedup_aligned, dedup_del, _ = _dedup_aligned_deletion_rows(aligned, deletion_matrix)
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

    return _chain_msa_block(msa, has_deletion, deletion_value, deletion_mean, profile)
end

function _chain_msa_features(
    sequence::AbstractString,
    msa_cfg::NamedTuple{(:precomputed_msa_dir, :pairing_db), Tuple{Union{Nothing, String}, String}},
    json_dir::AbstractString;
    require_pairing::Bool,
    msa_pair_as_unpair::Bool = false,
)::ChainMSAFeatures
    function _query_only_features()
        query_idx = _sequence_to_protenix_indices(sequence)
        profile = zeros(Float32, length(query_idx), 32)
        for (i, x) in enumerate(query_idx)
            profile[i, x + 1] = 1f0
        end
        return _chain_msa_block(
            permutedims(query_idx),
            zeros(Float32, 1, length(query_idx)),
            zeros(Float32, 1, length(query_idx)),
            zeros(Float32, length(query_idx)),
            profile,
        )
    end

    msa_dir_any = msa_cfg.precomputed_msa_dir
    if msa_dir_any === nothing || isempty(strip(String(msa_dir_any)))
        base = _query_only_features()
        return (combined = base, non_pairing = base, pairing = nothing, pairing_keys = nothing)
    end

    msa_dir_raw = String(msa_dir_any)
    msa_dir = isabspath(msa_dir_raw) ? msa_dir_raw : normpath(joinpath(json_dir, msa_dir_raw))
    isdir(msa_dir) || error("The provided precomputed_msa_dir does not exist: $msa_dir")

    # Find and read non-pairing MSA (supports .a3m, .fasta, .fa, .fas)
    non_pair_path = _find_msa_file(msa_dir, "non_pairing")
    np_seqs = String[]
    np_atype = :a3m
    if non_pair_path !== nothing
        np_seqs, _ = _parse_a3m(non_pair_path; seq_limit = -1)
        np_atype = _infer_alignment_type(non_pair_path)
    end

    # When msa_pair_as_unpair (v1.0 default), merge pairing MSA into the unpaired set.
    # Python: u_a3m = RawMsa.from_a3m(seq, ctype, p_a3m + u_a3m, dedup=True).to_a3m()
    if msa_pair_as_unpair
        pair_path = _find_msa_file(msa_dir, "pairing")
        if pair_path !== nothing
            p_seqs, _ = _parse_a3m(pair_path; seq_limit = -1)
            np_seqs = vcat(p_seqs, np_seqs)  # paired first, then unpaired
            # If pairing file has different format, use the pairing format for all
            # (pairing sequences are prepended, so row 0 alignment type matters)
            np_atype = _infer_alignment_type(pair_path)
        end
    end

    non_pairing = if !isempty(np_seqs)
        if np_atype == :fasta
            aln, del = _aligned_and_deletions_from_fasta(np_seqs)
        else
            aln, del = _aligned_and_deletions_from_a3m(np_seqs)
        end
        # Validate MSA query vs input sequence
        if !isempty(aln)
            _check_msa_query_match(aln[1], sequence,
                something(non_pair_path, joinpath(msa_dir, "non_pairing.*")))
        end
        _build_chain_msa_features(sequence, aln, del)
    else
        _query_only_features()
    end

    pairing = nothing
    pairing_keys = nothing
    if require_pairing
        pair_path = _find_msa_file(msa_dir, "pairing")
        pair_path !== nothing && isfile(pair_path) || error(
            "No pairing-MSA found in $msa_dir (tried pairing.a3m, pairing.fasta, etc.) for multi-chain assembly.",
        )
        seqs, desc = _parse_a3m(pair_path; seq_limit = -1)
        pair_atype = _infer_alignment_type(pair_path)
        if pair_atype == :fasta
            aln, del = _aligned_and_deletions_from_fasta(seqs)
        else
            aln, del = _aligned_and_deletions_from_a3m(seqs)
        end
        # Validate pairing MSA query vs input sequence
        if !isempty(aln)
            _check_msa_query_match(aln[1], sequence, pair_path)
        end
        row_keys = [_pairing_key_from_description(d, i) for (i, d) in enumerate(desc)]
        aln_d, del_d, keys_d = _dedup_aligned_deletion_rows(aln, del; row_keys = row_keys)
        pairing = _build_chain_msa_features(sequence, aln_d, del_d; dedup_rows = false)
        pairing_keys = keys_d
    end

    if pairing === nothing
        combined = non_pairing
    else
        n_pair = size(pairing.msa, 1)
        n_non = size(non_pairing.msa, 1)
        n_tot = max(n_pair + n_non, 1)
        wt_pair = Float32(n_pair) / Float32(n_tot)
        wt_non = Float32(n_non) / Float32(n_tot)
        combined = (
            msa = vcat(pairing.msa, non_pairing.msa),
            has_deletion = vcat(pairing.has_deletion, non_pairing.has_deletion),
            deletion_value = vcat(pairing.deletion_value, non_pairing.deletion_value),
            deletion_mean = wt_pair .* pairing.deletion_mean .+ wt_non .* non_pairing.deletion_mean,
            profile = wt_pair .* pairing.profile .+ wt_non .* non_pairing.profile,
        )
    end

    return (combined = combined, non_pairing = non_pairing, pairing = pairing, pairing_keys = pairing_keys)
end

# ---------- RNA MSA feature building ----------

function _rna_sequence_to_protenix_indices(sequence::AbstractString)
    seq = uppercase(strip(sequence))
    idx = Int[]
    for c in seq
        isspace(c) && continue
        push!(idx, get(_MSA_RNA_SEQ_TO_ID, c, 25))  # 25 = unknown RNA
    end
    return idx
end

function _rna_msa_idx(c::Char)
    c == '.' && return _MSA_RNA_SEQ_TO_ID['-']
    return get(_MSA_RNA_SEQ_TO_ID, uppercase(c), 25)
end

function _build_rna_chain_msa_features(
    sequence::AbstractString,
    aligned::Vector{String},
    deletion_matrix::Vector{Vector{Float32}},
    ;
    dedup_rows::Bool = true,
)::ChainMSABlock
    if isempty(aligned)
        query = uppercase(strip(sequence))
        push!(aligned, query)
        push!(deletion_matrix, zeros(Float32, length(query)))
    end

    dedup_aligned = aligned
    dedup_del = deletion_matrix
    if dedup_rows
        dedup_aligned, dedup_del, _ = _dedup_aligned_deletion_rows(aligned, deletion_matrix)
    end

    n_row = length(dedup_aligned)
    n_col = length(dedup_aligned[1])
    n_row > 0 || error("RNA MSA has zero rows after deduplication.")
    all(length(s) == n_col for s in dedup_aligned) || error("RNA MSA rows must have uniform length.")
    all(length(d) == n_col for d in dedup_del) || error("RNA deletion rows must have uniform length.")

    # Convert directly to Protenix indices using RNA char map (no HHblits intermediate).
    msa = Matrix{Int}(undef, n_row, n_col)
    deletion_mat = Matrix{Float32}(undef, n_row, n_col)
    for i in 1:n_row
        seq = dedup_aligned[i]
        del = dedup_del[i]
        for j in 1:n_col
            msa[i, j] = _rna_msa_idx(seq[j])
            deletion_mat[i, j] = del[j]
        end
    end

    # Build 32-class profile directly from Protenix indices.
    profile = zeros(Float32, n_col, 32)
    @inbounds for i in 1:n_row, j in 1:n_col
        idx = msa[i, j] + 1
        if 1 <= idx <= 32
            profile[j, idx] += 1f0
        end
    end
    profile ./= Float32(n_row)

    has_deletion = clamp.(deletion_mat, 0f0, 1f0)
    deletion_value = (2f0 / Float32(pi)) .* atan.(deletion_mat ./ 3f0)
    deletion_mean = vec(mean(deletion_mat; dims = 1))

    return _chain_msa_block(msa, has_deletion, deletion_value, deletion_mean, profile)
end

function _rna_chain_msa_features(
    spec::RNAChainSpec,
    json_dir::AbstractString,
)::ChainMSAFeatures
    # Build query-only features for RNA.
    function _rna_query_only_features()
        query_idx = _rna_sequence_to_protenix_indices(spec.sequence)
        profile = zeros(Float32, length(query_idx), 32)
        for (i, x) in enumerate(query_idx)
            profile[i, x + 1] = 1f0
        end
        return _chain_msa_block(
            permutedims(query_idx),
            zeros(Float32, 1, length(query_idx)),
            zeros(Float32, 1, length(query_idx)),
            zeros(Float32, length(query_idx)),
            profile,
        )
    end

    # Resolve MSA text: inline takes priority over file path.
    msa_text = spec.unpaired_msa
    msa_source_path = nothing  # for query-match checking
    rna_atype = :a3m           # default for inline text
    if msa_text === nothing && spec.unpaired_msa_path !== nothing
        path_raw = spec.unpaired_msa_path
        path = isabspath(path_raw) ? path_raw : normpath(joinpath(json_dir, path_raw))
        isfile(path) || error("RNA MSA file not found: $path")
        msa_text = read(path, String)
        msa_source_path = path
        rna_atype = _infer_alignment_type(path)
    end

    if msa_text === nothing || isempty(strip(msa_text))
        base = _rna_query_only_features()
        return (combined = base, non_pairing = base, pairing = nothing, pairing_keys = nothing)
    end

    seqs, _ = _parse_a3m_from_text(msa_text)
    if isempty(seqs)
        base = _rna_query_only_features()
        return (combined = base, non_pairing = base, pairing = nothing, pairing_keys = nothing)
    end

    if rna_atype == :fasta
        aln, del = _aligned_and_deletions_from_fasta(seqs)
    else
        aln, del = _aligned_and_deletions_from_a3m(seqs)
    end
    # Validate MSA query vs input sequence
    if !isempty(aln)
        _check_msa_query_match(aln[1], spec.sequence,
            something(msa_source_path, "<inline RNA MSA>"))
    end
    non_pairing = _build_rna_chain_msa_features(spec.sequence, aln, del)

    # RNA has no paired MSA.
    return (combined = non_pairing, non_pairing = non_pairing, pairing = nothing, pairing_keys = nothing)
end

# Parse A3M from an inline text string (same logic as _parse_a3m but from text).
function _parse_a3m_from_text(text::AbstractString; seq_limit::Int = -1)
    sequences = String[]
    descriptions = String[]
    idx = 0
    for line in split(text, '\n')
        ln = strip(line)
        isempty(ln) && continue
        startswith(ln, "#") && continue
        if startswith(ln, ">")
            if seq_limit > 0 && length(sequences) > seq_limit
                break
            end
            idx += 1
            push!(descriptions, String(ln[2:end]))
            push!(sequences, "")
            continue
        end
        idx > 0 || continue
        sequences[idx] *= ln
    end
    return sequences, descriptions
end

function _inject_task_msa_features!(
    feat::Dict{String, Any},
    task::AbstractDict{<:Any, <:Any},
    json_path::AbstractString;
    use_msa::Bool,
    use_rna_msa::Bool = false,
    msa_pair_as_unpair::Bool = false,
    chain_specs::Union{Nothing, Vector{ProteinChainSpec}} = nothing,
    rna_chain_specs::Union{Nothing, Vector{RNAChainSpec}} = nothing,
    dna_chain_specs::Union{Nothing, Vector{DNAChainSpec}} = nothing,
    token_chain_ids::Union{Nothing, Vector{String}} = nothing,
    ion_chain_ids::Union{Nothing, Set{String}} = nothing,
)
    use_msa || return feat
    json_dir = dirname(abspath(json_path))

    local_prot_specs = chain_specs === nothing ? _extract_protein_chain_specs(task) : chain_specs
    local_rna_specs = rna_chain_specs === nothing ? _extract_rna_chain_specs(task; json_dir = json_dir) : rna_chain_specs
    local_dna_specs = dna_chain_specs === nothing ? DNAChainSpec[] : dna_chain_specs

    has_dna_chains = !isempty(local_dna_specs) || any(
        haskey(_as_string_dict(e), "dnaSequence")
        for e in get(task, "sequences", [])
    )
    isempty(local_prot_specs) && isempty(local_rna_specs) && !has_dna_chains && return feat

    restype = Float32.(feat["restype"])
    n_tok = size(restype, 1)
    restype_idx = Vector{Int}(undef, n_tok)
    @inbounds for i in 1:n_tok
        _, idx = findmax(@view restype[i, :])
        restype_idx[i] = idx - 1
    end

    # Pre-compute residue indices for MSA broadcast (used for modified residues).
    residue_idx = Int.(feat["residue_index"])

    # --- Protein MSA features ---
    prot_token_cols = Vector{Vector{Int}}()
    prot_features = Vector{ChainMSAFeatures}()
    if !isempty(local_prot_specs)
        prot_token_cols = if token_chain_ids === nothing
            asym = Int.(feat["asym_id"])
            [findall(==(i - 1), asym) for i in 1:length(local_prot_specs)]
        else
            [findall(==(spec.chain_id), token_chain_ids) for spec in local_prot_specs]
        end
        all(!isempty(cols) for cols in prot_token_cols) || error("Failed to map protein chains to token columns.")

        chain_sequences = [s.sequence for s in local_prot_specs]
        is_homomer_or_monomer = length(Set(chain_sequences)) == 1
        prot_features = Vector{ChainMSAFeatures}(undef, length(local_prot_specs))
        for (i, spec) in enumerate(local_prot_specs)
            prot_features[i] = _chain_msa_features(
                spec.sequence,
                spec.msa_cfg,
                json_dir;
                require_pairing = !is_homomer_or_monomer,
                msa_pair_as_unpair = msa_pair_as_unpair,
            )
        end

        # Broadcast MSA from sequence-level to token-level for chains with modified
        # residues that create multiple per-atom tokens per sequence position.
        # Matches Python Protenix's expand_msa_features() / map_to_standard().
        residue_idx = Int.(feat["residue_index"])
        for (i, cols) in enumerate(prot_token_cols)
            seq_len = size(prot_features[i].combined.msa, 2)
            if seq_len != length(cols)
                token_rids = residue_idx[cols]
                prot_features[i] = _broadcast_msa_features_to_tokens(prot_features[i], token_rids)
            end
        end

        for (i, cols) in enumerate(prot_token_cols)
            size(prot_features[i].combined.msa, 2) == length(cols) || error(
                "MSA/token length mismatch on protein chain $(local_prot_specs[i].chain_id): MSA has $(size(prot_features[i].combined.msa, 2)) columns, tokens have $(length(cols)).",
            )
        end
    end

    # --- RNA MSA features ---
    rna_token_cols = Vector{Vector{Int}}()
    rna_features = Vector{ChainMSAFeatures}()
    if !isempty(local_rna_specs)
        rna_token_cols = if token_chain_ids !== nothing
            [findall(==(rspec.chain_id), token_chain_ids) for rspec in local_rna_specs]
        else
            # Map RNA chain_id → asym_id (0-indexed) → token columns.
            asym = Int.(feat["asym_id"])
            all_chain_ids = String[]
            task_seqs = task["sequences"]
            cidx = 1
            for entity_any in task_seqs
                entity = _as_string_dict(entity_any)
                cnt = 1
                for k in ("proteinChain", "dnaSequence", "rnaSequence", "ligand", "condition_ligand", "ion")
                    if haskey(entity, k)
                        cnt = Int(get(_as_string_dict(entity[k]), "count", 1))
                        break
                    end
                end
                for _ in 1:cnt
                    push!(all_chain_ids, _chain_id_from_index(cidx))
                    cidx += 1
                end
            end
            result = Vector{Vector{Int}}()
            for rspec in local_rna_specs
                aidx = findfirst(==(rspec.chain_id), all_chain_ids)
                aidx === nothing && error("Cannot find asym_id for RNA chain $(rspec.chain_id)")
                push!(result, findall(==(aidx - 1), asym))
            end
            result
        end
        all(!isempty(cols) for cols in rna_token_cols) || error("Failed to map RNA chains to token columns.")

        rna_features = Vector{ChainMSAFeatures}(undef, length(local_rna_specs))
        for (i, rspec) in enumerate(local_rna_specs)
            rna_features[i] = _rna_chain_msa_features(rspec, json_dir)
        end

        # When use_rna_msa=false (Python default), strip alignment rows and keep
        # only the query row.  Python's FeatureAssemblyLine treats RNA chains as
        # query-only when use_rna_msa is disabled.
        if !use_rna_msa
            for i in eachindex(rna_features)
                cf = rna_features[i]
                blk = cf.combined
                seq_len = size(blk.msa, 2)
                # Recompute profile as one-hot from query row (matching Python's
                # query-only behavior when use_rna_msa=false).
                n_classes = size(blk.profile, 2)
                qprofile = zeros(Float32, seq_len, n_classes)
                for j in 1:seq_len
                    idx = blk.msa[1, j] + 1  # 0-indexed → 1-indexed
                    if 1 <= idx <= n_classes
                        qprofile[j, idx] = 1f0
                    end
                end
                qonly = _chain_msa_block(
                    blk.msa[1:1, :],
                    blk.has_deletion[1:1, :],
                    blk.deletion_value[1:1, :],
                    zeros(Float32, seq_len),
                    qprofile,
                )
                rna_features[i] = (combined = qonly, non_pairing = qonly, pairing = nothing, pairing_keys = nothing)
            end
        end

        # Broadcast RNA MSA from sequence-level to token-level for modified bases.
        for (i, cols) in enumerate(rna_token_cols)
            seq_len = size(rna_features[i].combined.msa, 2)
            if seq_len != length(cols)
                token_rids = residue_idx[cols]
                rna_features[i] = _broadcast_msa_features_to_tokens(rna_features[i], token_rids)
            end
        end

        for (i, cols) in enumerate(rna_token_cols)
            size(rna_features[i].combined.msa, 2) == length(cols) || error(
                "RNA MSA/token length mismatch on chain $(local_rna_specs[i].chain_id): MSA has $(size(rna_features[i].combined.msa, 2)) columns, tokens have $(length(cols)).",
            )
        end
    end

    # --- DNA chain MSA features ---
    # Python v1.0 FeatureAssemblyLine treats DNA chains as standard polymers:
    # each DNA chain gets a "dummy" MSA with 1 query row from the sequence.
    # Modified bases (e.g. 5MC) are mapped to their canonical parent DNA base
    # (5MC → DC) via CCD one_letter_code.
    dna_token_cols = Vector{Vector{Int}}()
    dna_features = Vector{ChainMSAFeatures}()
    if !isempty(local_dna_specs)
        dna_token_cols = if token_chain_ids !== nothing
            [findall(==(dspec.chain_id), token_chain_ids) for dspec in local_dna_specs]
        else
            asym = Int.(feat["asym_id"])
            # Find the asym_id for each DNA chain by scanning entity order.
            task_seqs = task["sequences"]
            cidx = 1
            all_chain_ids = String[]
            for entity_any in task_seqs
                ent = _as_string_dict(entity_any)
                cnt = 1
                for k in ("proteinChain", "dnaSequence", "rnaSequence", "ligand", "condition_ligand", "ion")
                    if haskey(ent, k)
                        cnt = Int(get(_as_string_dict(ent[k]), "count", 1))
                        break
                    end
                end
                for _ in 1:cnt
                    push!(all_chain_ids, _chain_id_from_index(cidx))
                    cidx += 1
                end
            end
            [begin
                aidx = findfirst(==(dspec.chain_id), all_chain_ids)
                aidx === nothing && error("Cannot find asym_id for DNA chain $(dspec.chain_id)")
                findall(==(aidx - 1), asym)
            end for dspec in local_dna_specs]
        end

        for (i, dspec) in enumerate(local_dna_specs)
            cols = dna_token_cols[i]
            isempty(cols) && continue
            # Build sequence-level MSA query from the *original* DNA sequence, not
            # the modified CCD codes.  Python's FeatureAssemblyLine uses the input
            # sequence letters (e.g. "ATGC") for MSA, so modifications like 5BU at
            # position 3 still get the original base DG=27 in the MSA query.
            seq_len = length(dspec.sequence)
            seq_indices = Vector{Int}(undef, seq_len)
            for (j, c) in enumerate(dspec.sequence)
                dna3 = get(_DNA_1TO3, c, "DN")
                seq_indices[j] = get(STD_RESIDUES_WITH_GAP, dna3, STD_RESIDUES_WITH_GAP["DN"])
            end
            # Build 1-row MSA (query only) and profile.
            msa_query = reshape(seq_indices, 1, seq_len)
            n_classes = size(restype, 2)
            chain_profile = zeros(Float32, seq_len, n_classes)
            for j in 1:seq_len
                idx = seq_indices[j] + 1  # 0-indexed → 1-indexed
                if 1 <= idx <= n_classes
                    chain_profile[j, idx] = 1f0
                end
            end
            chain_has_del = zeros(Float32, 1, seq_len)
            chain_del_val = zeros(Float32, 1, seq_len)
            chain_del_mean = zeros(Float32, seq_len)
            combined = _chain_msa_block(msa_query, chain_has_del, chain_del_val, chain_del_mean, chain_profile)
            cf = convert(ChainMSAFeatures, (combined = combined, non_pairing = combined, pairing = nothing, pairing_keys = nothing))

            # Broadcast from sequence-level to token-level if modified bases created
            # per-atom tokens (e.g. 5MC → 21 tokens instead of 1).
            if seq_len != length(cols)
                token_rids = residue_idx[cols]
                cf = _broadcast_msa_features_to_tokens(cf, token_rids)
            end
            size(cf.combined.msa, 2) == length(cols) || error(
                "DNA MSA/token length mismatch on chain $(dspec.chain_id): MSA has $(size(cf.combined.msa, 2)) columns, tokens have $(length(cols)).",
            )
            push!(dna_features, cf)
        end
    end

    # --- Assemble unified MSA matrix ---
    # Combine protein, RNA, and DNA features/columns into unified lists.
    all_features = vcat(prot_features, rna_features, dna_features)
    all_token_cols = vcat(prot_token_cols, rna_token_cols, dna_token_cols)

    # Protein-only pairing logic (RNA never participates in cross-species pairing).
    heteromer_pairing_merge = false
    pairing_plan = nothing
    if !isempty(prot_features)
        chain_sequences = [s.sequence for s in local_prot_specs]
        is_prot_homomer = length(Set(chain_sequences)) == 1
        heteromer_pairing_merge = !is_prot_homomer && all(cf -> cf.pairing !== nothing, prot_features)
        pairing_plan = heteromer_pairing_merge ? _pairing_row_plan(prot_features) : nothing
    end

    is_prot_homomer_or_monomer = !isempty(prot_features) && length(Set(s.sequence for s in local_prot_specs)) == 1
    total_rows = if heteromer_pairing_merge
        paired_rows = pairing_plan === nothing ? minimum(size(cf.pairing.msa, 1) for cf in prot_features) : length(pairing_plan)
        nonpair_extra_rows = sum(max(size(cf.non_pairing.msa, 1) - 1, 0) for cf in prot_features)
        rna_rows = isempty(rna_features) ? 0 : sum(max(size(cf.combined.msa, 1) - 1, 0) for cf in rna_features)
        dna_rows = isempty(dna_features) ? 0 : sum(max(size(cf.combined.msa, 1) - 1, 0) for cf in dna_features)
        paired_rows + nonpair_extra_rows + rna_rows + dna_rows
    elseif is_prot_homomer_or_monomer || isempty(prot_features)
        # Python v1.0 FeatureAssemblyLine merges all polymer chains into a shared
        # MSA: chains contribute columns (concatenated), not extra rows.  The final
        # MSA = concat(paired_rows, unpaired_rows).  Without precomputed MSA each
        # chain produces 1 paired + 1 unpaired query row → total always 2 rows.
        max_unpaired = 0
        if !isempty(prot_features)
            max_unpaired = max(max_unpaired, maximum(size(cf.combined.msa, 1) for cf in prot_features))
        end
        if !isempty(rna_features)
            max_unpaired = max(max_unpaired, maximum(size(cf.combined.msa, 1) for cf in rna_features))
        end
        # DNA chains don't have separate MSA features in Julia but Python v1.0 treats
        # them as standard polymers that contribute at least 1 query row.
        if has_dna_chains && max_unpaired == 0
            max_unpaired = 1
        end
        # Python always prepends 1 paired query row for any polymer input.
        max_paired = max_unpaired > 0 ? 1 : 0
        max_paired + max_unpaired
    else
        # Heteromer without pairing data: Python's cleanup_unpaired_features removes
        # the unpaired query row (since it duplicates the paired query) for each chain.
        # Net result: 1 paired query row shared across all chains.  Non-query unpaired
        # rows are stacked per-chain as in the homomer branch.
        prot_nonquery_rows = sum(max(size(cf.combined.msa, 1) - 1, 0) for cf in prot_features)
        rna_rows = isempty(rna_features) ? 0 : sum(size(cf.combined.msa, 1) for cf in rna_features)
        1 + prot_nonquery_rows + rna_rows
    end
    total_rows > 0 || (total_rows = 1)

    # Row 1 (query) gets the full query sequence; rows 2+ default to gap (31).
    # Chain-specific MSA writing will fill only the columns for that chain;
    # non-participating columns in non-query rows must be gap, not query values.
    msa = fill(Int(31), total_rows, n_tok)
    msa[1, :] .= restype_idx
    has_deletion = zeros(Float32, total_rows, n_tok)
    deletion_value = zeros(Float32, total_rows, n_tok)
    profile = copy(restype)
    deletion_mean = zeros(Float32, n_tok)

    if heteromer_pairing_merge
        row = 1
        if pairing_plan === nothing
            paired_rows = minimum(size(cf.pairing.msa, 1) for cf in prot_features)
            # Fallback mode: pair by row index when taxonomic keys are unavailable.
            for r in 1:paired_rows
                for (cf, cols) in zip(prot_features, prot_token_cols)
                    msa[row, cols] .= cf.pairing.msa[r, :]
                    has_deletion[row, cols] .= cf.pairing.has_deletion[r, :]
                    deletion_value[row, cols] .= cf.pairing.deletion_value[r, :]
                end
                row += 1
            end
        else
            # Preferred mode: pair rows by inferred species/taxonomic keys.
            for row_indices in pairing_plan
                for i in eachindex(prot_features)
                    cf = prot_features[i]
                    cols = prot_token_cols[i]
                    r = row_indices[i]
                    msa[row, cols] .= cf.pairing.msa[r, :]
                    has_deletion[row, cols] .= cf.pairing.has_deletion[r, :]
                    deletion_value[row, cols] .= cf.pairing.deletion_value[r, :]
                end
                row += 1
            end
        end

        # Append protein non-pairing rows (excluding duplicate query row).
        for (cf, cols) in zip(prot_features, prot_token_cols)
            n_row = size(cf.non_pairing.msa, 1)
            if n_row >= 2
                rows = row:(row + (n_row - 2))
                msa[rows, cols] .= cf.non_pairing.msa[2:end, :]
                has_deletion[rows, cols] .= cf.non_pairing.has_deletion[2:end, :]
                deletion_value[rows, cols] .= cf.non_pairing.deletion_value[2:end, :]
                row += n_row - 1
            end
            profile[cols, :] .= cf.non_pairing.profile
            deletion_mean[cols] .= cf.non_pairing.deletion_mean
        end

        # Append RNA non-pairing rows (excluding duplicate query row).
        for (cf, cols) in zip(rna_features, rna_token_cols)
            n_row = size(cf.combined.msa, 1)
            if n_row >= 2
                rows = row:(row + (n_row - 2))
                msa[rows, cols] .= cf.combined.msa[2:end, :]
                has_deletion[rows, cols] .= cf.combined.has_deletion[2:end, :]
                deletion_value[rows, cols] .= cf.combined.deletion_value[2:end, :]
                row += n_row - 1
            end
            profile[cols, :] .= cf.combined.profile
            deletion_mean[cols] .= cf.combined.deletion_mean
        end

        # Append DNA non-pairing rows (excluding duplicate query row).
        for (cf, cols) in zip(dna_features, dna_token_cols)
            n_row = size(cf.combined.msa, 1)
            if n_row >= 2
                rows = row:(row + (n_row - 2))
                msa[rows, cols] .= cf.combined.msa[2:end, :]
                has_deletion[rows, cols] .= cf.combined.has_deletion[2:end, :]
                deletion_value[rows, cols] .= cf.combined.deletion_value[2:end, :]
                row += n_row - 1
            end
            profile[cols, :] .= cf.combined.profile
            deletion_mean[cols] .= cf.combined.deletion_mean
        end
        row - 1 == total_rows || error("internal error: heteromer pairing-row accounting mismatch (expected $total_rows, got $(row-1))")
    elseif is_prot_homomer_or_monomer || isempty(prot_features)
        # Python v1.0 FeatureAssemblyLine layout:
        #   Row 1 = paired query (set explicitly from chain MSA below)
        #   Rows 2..max_unpaired+1 = unpaired MSA (merged columns, shared rows)
        # All chains share the same row space; their columns are interleaved.
        # Chains with fewer unpaired rows than max_unpaired have their remaining
        # columns stay as gap (31) from the gap-initialized rows 2+.
        # Row 1 = paired query: set explicitly from chain MSA query rather than
        # relying on restype_idx initialization, because modified residues may
        # have MSA indices that differ from raw restype (e.g. DHA → ALA mapping).
        for (cf, cols) in zip(prot_features, prot_token_cols)
            msa[1, cols] .= cf.combined.msa[1, :]
        end
        for (cf, cols) in zip(rna_features, rna_token_cols)
            msa[1, cols] .= cf.combined.msa[1, :]
        end
        for (cf, cols) in zip(dna_features, dna_token_cols)
            msa[1, cols] .= cf.combined.msa[1, :]
        end
        # Rows 2..max_unpaired+1 = unpaired MSA (merged columns, shared rows).
        unpaired_start = 2
        for (cf, cols) in zip(prot_features, prot_token_cols)
            n_row = size(cf.combined.msa, 1)
            for r in 1:n_row
                target_row = unpaired_start + r - 1
                msa[target_row, cols] .= cf.combined.msa[r, :]
                has_deletion[target_row, cols] .= cf.combined.has_deletion[r, :]
                deletion_value[target_row, cols] .= cf.combined.deletion_value[r, :]
            end
            profile[cols, :] .= cf.combined.profile
            deletion_mean[cols] .= cf.combined.deletion_mean
        end
        for (cf, cols) in zip(rna_features, rna_token_cols)
            n_row = size(cf.combined.msa, 1)
            for r in 1:n_row
                target_row = unpaired_start + r - 1
                msa[target_row, cols] .= cf.combined.msa[r, :]
                has_deletion[target_row, cols] .= cf.combined.has_deletion[r, :]
                deletion_value[target_row, cols] .= cf.combined.deletion_value[r, :]
            end
            profile[cols, :] .= cf.combined.profile
            deletion_mean[cols] .= cf.combined.deletion_mean
        end
        for (cf, cols) in zip(dna_features, dna_token_cols)
            n_row = size(cf.combined.msa, 1)
            for r in 1:n_row
                target_row = unpaired_start + r - 1
                msa[target_row, cols] .= cf.combined.msa[r, :]
                has_deletion[target_row, cols] .= cf.combined.has_deletion[r, :]
                deletion_value[target_row, cols] .= cf.combined.deletion_value[r, :]
            end
            profile[cols, :] .= cf.combined.profile
            deletion_mean[cols] .= cf.combined.deletion_mean
        end
    else
        # Heteromer without pairing data: Row 1 is the shared query (already
        # initialized from restype_idx). Python's cleanup_unpaired_features
        # removes the duplicate unpaired query, leaving only non-query rows.
        row_start = 2  # row 1 is the query
        for (cf, cols) in zip(prot_features, prot_token_cols)
            n_row = size(cf.combined.msa, 1)
            if n_row >= 2
                row_stop = row_start + (n_row - 2)
                rows = row_start:row_stop
                msa[rows, cols] .= cf.combined.msa[2:end, :]
                has_deletion[rows, cols] .= cf.combined.has_deletion[2:end, :]
                deletion_value[rows, cols] .= cf.combined.deletion_value[2:end, :]
                row_start = row_stop + 1
            end
            profile[cols, :] .= cf.combined.profile
            deletion_mean[cols] .= cf.combined.deletion_mean
        end
        for (cf, cols) in zip(rna_features, rna_token_cols)
            n_row = size(cf.combined.msa, 1)
            row_stop = row_start + n_row - 1
            rows = row_start:row_stop
            msa[rows, cols] .= cf.combined.msa
            has_deletion[rows, cols] .= cf.combined.has_deletion
            deletion_value[rows, cols] .= cf.combined.deletion_value
            profile[cols, :] .= cf.combined.profile
            deletion_mean[cols] .= cf.combined.deletion_mean
            row_start = row_stop + 1
        end
        for (cf, cols) in zip(dna_features, dna_token_cols)
            n_row = size(cf.combined.msa, 1)
            if n_row >= 2
                row_stop = row_start + (n_row - 2)
                rows = row_start:row_stop
                msa[rows, cols] .= cf.combined.msa[2:end, :]
                has_deletion[rows, cols] .= cf.combined.has_deletion[2:end, :]
                deletion_value[rows, cols] .= cf.combined.deletion_value[2:end, :]
                row_start = row_stop + 1
            end
            profile[cols, :] .= cf.combined.profile
            deletion_mean[cols] .= cf.combined.deletion_mean
        end
    end

    # Fix 19: Match Python v1.0 numpy -1 column indexing for uncovered tokens.
    #
    # Python v1.0's FeatureAssemblyLine.assemble() builds an MSA with columns for
    # polymer chains only.  map_to_standard() maps each token to a column index;
    # tokens not found in the polymer metadata (ions, unhandled ligands) get index
    # -1.  The final reindexing `merged["msa"][:, std_idxs]` uses numpy's -1
    # wraparound, selecting the LAST column of the polymer-only MSA — i.e., the
    # last residue of the last polymer chain in entity order.
    #
    # In Julia, the MSA matrix has columns for ALL tokens.  Polymer columns are
    # filled from chain MSA features.  Uncovered columns (ions, ligands not in any
    # chain spec) must be set to match the last polymer column's values.
    #
    # Note: In v1.0's InferenceMSAFeaturizer, "ion" entities have no handler
    # (only proteinChain/rnaSequence/dnaSequence/ligand are handled), so ions
    # are never added to the MSA metadata and always get the -1 fallback.
    covered = falses(n_tok)
    for cols in prot_token_cols
        covered[cols] .= true
    end
    for cols in rna_token_cols
        covered[cols] .= true
    end
    for cols in dna_token_cols
        covered[cols] .= true
    end

    any_uncovered = any(.!covered)
    if any_uncovered
        unk_idx = 20  # STD_RESIDUES_WITH_GAP["UNK"]

        # Fix 19: Match Python v1.0 numpy -1 column indexing for ion tokens.
        #
        # In Python v1.0's InferenceMSAFeaturizer.make_msa_feature(), entities of
        # type "proteinChain", "rnaSequence", "dnaSequence", and "ligand" are each
        # handled explicitly.  "ion" entities have NO handler, so they are never
        # added to the MSA metadata.  When FeatureAssemblyLine.assemble() calls
        # map_to_standard() to build column indices (std_idxs), unmapped tokens
        # get index -1.  The final reindexing:
        #   merged[f] = merged[f][:, std_idxs].copy()      (line 350)
        # uses std_idxs as a COLUMN selector.  NumPy's -1 wraps to the LAST column
        # of the merged MSA.  The merged MSA contains columns for all HANDLED
        # entities: polymer chains + ligands.  The last column is therefore the last
        # token of the last handled entity (in entity order), which may be:
        #   - a polymer residue (if the last non-ion entity is a polymer), or
        #   - a ligand token (UNK=20, if a ligand entity comes after the polymers).
        #
        # Implementation: two phases.
        # Phase 1: Set uncovered non-ion tokens (ligands) to UNK=20.
        # Phase 2: Find the last non-ion column (now correctly filled) and copy
        #          its values to all ion columns.

        # Phase 1: Ligands and other uncovered non-ion tokens → UNK=20.
        for i in 1:n_tok
            covered[i] && continue
            26 <= restype_idx[i] <= 30 && continue
            is_ion = ion_chain_ids !== nothing &&
                     token_chain_ids !== nothing &&
                     token_chain_ids[i] in ion_chain_ids
            is_ion && continue  # Handled in phase 2.
            for r in 1:total_rows
                msa[r, i] = unk_idx
            end
            profile[i, :] .= 0f0
            if size(profile, 2) >= 21
                profile[i, 21] = 1f0
            end
        end

        # Phase 2: Ion tokens → inherit from last non-ion column.
        # This is the rightmost column that isn't an ion, matching Python's -1
        # wraparound over the merged MSA (which excludes ions).
        has_ions = ion_chain_ids !== nothing && token_chain_ids !== nothing &&
                   any(i -> !covered[i] && token_chain_ids[i] in ion_chain_ids, 1:n_tok)
        if has_ions
            last_non_ion_col = nothing
            for i in n_tok:-1:1
                if !(token_chain_ids[i] in ion_chain_ids)
                    last_non_ion_col = i
                    break
                end
            end
            if last_non_ion_col !== nothing
                for i in 1:n_tok
                    covered[i] && continue
                    26 <= restype_idx[i] <= 30 && continue
                    token_chain_ids[i] in ion_chain_ids || continue
                    for r in 1:total_rows
                        msa[r, i] = msa[r, last_non_ion_col]
                    end
                    profile[i, :] .= @view profile[last_non_ion_col, :]
                end
            end
        end
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
    entity_chain_ids::Vector{Vector{String}},
    entity_id::Int,
    copy_any,
    context::String,
)
    (1 <= entity_id <= length(entity_chain_ids)) || error("$context references unknown entity $entity_id")
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

function _resolve_bond_atom_name(atom_any, entity_id::Int, entity_atom_map::Vector{Dict{Int, String}}, context::String)
    if atom_any isa Integer
        idx = Int(atom_any)
        (1 <= entity_id <= length(entity_atom_map)) || error(
            "$context uses numeric atom index $idx for entity $entity_id, but no atom_map_to_atom_name is available.",
        )
        amap = entity_atom_map[entity_id]
        !isempty(amap) || error("$context uses numeric atom index $idx for entity $entity_id, but no atom_map_to_atom_name is available.")
        haskey(amap, idx) || error("$context atom index $idx not found in atom_map_to_atom_name for entity $entity_id.")
        return String(amap[idx])
    end

    atom_name = String(strip(String(atom_any)))
    all(isdigit, atom_name) || return atom_name
    idx = parse(Int, atom_name)
    (1 <= entity_id <= length(entity_atom_map)) || error(
        "$context uses numeric atom index $idx for entity $entity_id, but no atom_map_to_atom_name is available.",
    )
    amap = entity_atom_map[entity_id]
    !isempty(amap) || error("$context uses numeric atom index $idx for entity $entity_id, but no atom_map_to_atom_name is available.")
    haskey(amap, idx) || error("$context atom index $idx not found in atom_map_to_atom_name for entity $entity_id.")
    return String(amap[idx])
end

function _inject_task_covalent_token_bonds!(
    feat::Dict{String, Any},
    atoms::Vector{AtomRecord},
    task::AbstractDict{<:Any, <:Any},
    entity_chain_ids::Vector{Vector{String}},
    entity_atom_map::Vector{Dict{Int, String}} = Dict{Int, String}[],
)
    haskey(task, "covalent_bonds") || return feat
    bonds_any = task["covalent_bonds"]
    bonds_any isa AbstractVector || error("task.covalent_bonds must be an array when provided.")
    isempty(bonds_any) && return feat

    cif_bonds = NamedTuple{(:chain1,:res_name1,:res_id1,:atom_name1,:chain2,:res_name2,:res_id2,:atom_name2), Tuple{String,String,Int,String,String,String,Int,String}}[]
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
            # Store cross-chain bond records for CIF output
            ra1 = atoms[a1]
            ra2 = atoms[a2]
            if ra1.chain_id != ra2.chain_id
                push!(cif_bonds, (
                    chain1 = ra1.chain_id, res_name1 = ra1.res_name, res_id1 = ra1.res_id, atom_name1 = ra1.atom_name,
                    chain2 = ra2.chain_id, res_name2 = ra2.res_name, res_id2 = ra2.res_id, atom_name2 = ra2.atom_name,
                ))
            end
        end
    end

    feat["token_bonds"] = token_bonds
    feat["_cif_cross_chain_bonds"] = cif_bonds
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

function _flatten_nested!(dst::Vector{T}, x) where {T}
    if x isa AbstractArray
        for y in x
            _flatten_nested!(dst, y)
        end
    else
        push!(dst, T(x))
    end
    return dst
end

function _to_dense_array(x, ::Type{T}) where {T}
    x isa AbstractArray || error("Expected array-like template feature, got $(typeof(x)).")
    if !(x isa AbstractVector)
        return T.(x)
    end
    shape = _nested_shape(x)
    flat = T[]
    _flatten_nested!(flat, x)
    isempty(shape) && return T.(flat)
    return reshape(flat, shape...)
end

function _to_int_array(x)
    return _to_dense_array(x, Int)
end

function _to_float_array(x)
    return _to_dense_array(x, Float32)
end

# ─── Template feature extraction from CIF files ──────────────────────────────
# Extract template features from a user-provided CIF/PDB structure file.
# Matches Python Protenix v1.0.4 TemplateHitProcessor._get_atom_coords() +
# _extract_template_features() + TemplateFeatures.fix_template_features().

# ATOM37 ordering — the 37 standard protein heavy atom types.
# Identical to Python's constants.ATOM37 and ProtInterop's OF_ATOM_TYPES.
const _ATOM37_NAMES = (
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1",
    "SG", "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE",
    "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1",
    "NH2", "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
)
const _ATOM37_ORDER = Dict{String, Int}(name => i - 1 for (i, name) in enumerate(_ATOM37_NAMES))
const _ATOM37_NUM = 37

# Template residue type encoding — maps 1-letter AA codes to integer indices.
# Matches Python's encode_template_restype() in template_parser.py.
const _TEMPLATE_RESTYPE_ENCODE = Dict{Char, Int32}(
    'A' => 0, 'R' => 1, 'N' => 2, 'D' => 3, 'C' => 4, 'Q' => 5, 'E' => 6,
    'G' => 7, 'H' => 8, 'I' => 9, 'L' => 10, 'K' => 11, 'M' => 12, 'F' => 13,
    'P' => 14, 'S' => 15, 'T' => 16, 'W' => 17, 'Y' => 18, 'V' => 19,
    'X' => 20, 'U' => 4, 'B' => 3, 'Z' => 6, 'J' => 20, 'O' => 20, '-' => 31,
)

# 3-letter → 1-letter standard amino acid mapping.
# Non-standard residues (MSE, SEC, etc.) map to 'X' via the default, matching
# Python BioPython's behavior where only standard AAs get one-letter codes.
# MSE→MET normalization happens separately at the atom coordinate level.
const _AA_3TO1 = Dict{String, Char}(
    "ALA" => 'A', "ARG" => 'R', "ASN" => 'N', "ASP" => 'D', "CYS" => 'C',
    "GLN" => 'Q', "GLU" => 'E', "GLY" => 'G', "HIS" => 'H', "ILE" => 'I',
    "LEU" => 'L', "LYS" => 'K', "MET" => 'M', "PHE" => 'F', "PRO" => 'P',
    "SER" => 'S', "THR" => 'T', "TRP" => 'W', "TYR" => 'Y', "VAL" => 'V',
    "UNK" => 'X',
)

# ATOM14 per-residue canonical atom names (padded to 14).
# Matches Python's ATOM14_PADDED and ProtInterop's OF_RESTYPE_NAME_TO_ATOM14_NAMES.
const _ATOM14_NAMES = Dict{String, NTuple{14, String}}(
    "ALA" => ("N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""),
    "ARG" => ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "", "", ""),
    "ASN" => ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "", "", "", "", "", ""),
    "ASP" => ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "", "", "", "", "", ""),
    "CYS" => ("N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", ""),
    "GLN" => ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", "", "", "", "", ""),
    "GLU" => ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "", "", "", "", ""),
    "GLY" => ("N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""),
    "HIS" => ("N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", "", "", "", ""),
    "ILE" => ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "", "", "", "", "", ""),
    "LEU" => ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "", "", "", "", "", ""),
    "LYS" => ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "", "", "", "", ""),
    "MET" => ("N", "CA", "C", "O", "CB", "CG", "SD", "CE", "", "", "", "", "", ""),
    "PHE" => ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "", "", ""),
    "PRO" => ("N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", ""),
    "SER" => ("N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", ""),
    "THR" => ("N", "CA", "C", "O", "CB", "OG1", "CG2", "", "", "", "", "", "", ""),
    "TRP" => ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "TYR" => ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", "", ""),
    "VAL" => ("N", "CA", "C", "O", "CB", "CG1", "CG2", "", "", "", "", "", "", ""),
)

# Build PROTEIN_AATYPE_DENSE_ATOM_TO_ATOM37: (21, 24) matrix.
# For each protein restype (0-19 + UNK=20), maps dense atom index (1:24) to ATOM37 index (0-based).
# Matches Python's PROTEIN_AATYPE_DENSE_ATOM_TO_ATOM37.
const _PROTEIN_AATYPE_DENSE_ATOM_TO_ATOM37 = let
    restypes_1letter = ('A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                        'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V')
    n_dense = 24  # max across DENSE_ATOM (nucleic acids have 23-24 atoms)
    mat = zeros(Int32, length(restypes_1letter) + 1, n_dense)  # +1 for UNK
    for (rt_idx, rt) in enumerate(restypes_1letter)
        resname3 = string(get(Dict(
            'A' => "ALA", 'R' => "ARG", 'N' => "ASN", 'D' => "ASP", 'C' => "CYS",
            'Q' => "GLN", 'E' => "GLU", 'G' => "GLY", 'H' => "HIS", 'I' => "ILE",
            'L' => "LEU", 'K' => "LYS", 'M' => "MET", 'F' => "PHE", 'P' => "PRO",
            'S' => "SER", 'T' => "THR", 'W' => "TRP", 'Y' => "TYR", 'V' => "VAL",
        ), rt, "UNK"))
        atom14 = get(_ATOM14_NAMES, resname3, nothing)
        atom14 === nothing && continue
        for (j, name) in enumerate(atom14)
            if j > n_dense
                break
            end
            if !isempty(name) && haskey(_ATOM37_ORDER, name)
                mat[rt_idx, j] = _ATOM37_ORDER[name]
            end
        end
    end
    # UNK row (index 21) stays all zeros
    mat
end

"""
    _kalign_align(query, target) → Dict{Int,Int}

Align two sequences using the Kalign binary from kalign_jll
(matching Python Protenix's alignment).
Returns a mapping Dict{Int,Int} from query position (1-based) to target position (1-based).
Only aligned (non-gap) pairs are included.
"""
function _kalign_align(query::AbstractString, target::AbstractString)
    tmpdir = mktempdir()
    input_path = joinpath(tmpdir, "input.fasta")
    output_path = joinpath(tmpdir, "output.fasta")
    try
        open(input_path, "w") do io
            println(io, ">sequence 1")
            println(io, query)
            println(io, ">sequence 2")
            println(io, target)
        end

        # Run kalign from JLL (suppress stderr which contains version/citation info)
        cmd = `$(kalign_jll.kalign()) -i $input_path -o $output_path -format fasta`
        run(pipeline(cmd; stdout=devnull, stderr=devnull); wait=true)

        # Parse output FASTA
        lines = readlines(output_path)
        seqs = String[]
        current_seq = String[]
        for line in lines
            stripped = strip(line)
            isempty(stripped) && continue
            if startswith(stripped, '>')
                if !isempty(current_seq)
                    push!(seqs, join(current_seq))
                    current_seq = String[]
                end
            else
                push!(current_seq, stripped)
            end
        end
        !isempty(current_seq) && push!(seqs, join(current_seq))

        length(seqs) >= 2 || error("Kalign output: expected 2 aligned sequences, got $(length(seqs))")

        q_aln, t_aln = seqs[1], seqs[2]

        # Build mapping: query_pos (1-based) → target_pos (1-based)
        mapping = Dict{Int, Int}()
        q_idx, t_idx = 0, 0
        for (qa, ta) in zip(q_aln, t_aln)
            if qa != '-'
                q_idx += 1
            end
            if ta != '-'
                t_idx += 1
            end
            if qa != '-' && ta != '-'
                mapping[q_idx] = t_idx
            end
        end

        return mapping
    finally
        rm(tmpdir; recursive=true, force=true)
    end
end

"""
    _needleman_wunsch(query, target; match_score=1, mismatch_score=-1, gap_penalty=-1)

Simple Needleman-Wunsch global sequence alignment. Returns a mapping
Dict{Int,Int} from query position (1-based) to target position (1-based).
Only aligned (non-gap) pairs are included.
"""
function _needleman_wunsch(
    query::AbstractString,
    target::AbstractString;
    match_score::Int = 1,
    mismatch_score::Int = -1,
    gap_penalty::Int = -1,
)
    m = length(query)
    n = length(target)

    # Score matrix
    H = zeros(Int, m + 1, n + 1)
    for i in 1:(m + 1)
        H[i, 1] = (i - 1) * gap_penalty
    end
    for j in 1:(n + 1)
        H[1, j] = (j - 1) * gap_penalty
    end
    for i in 2:(m + 1), j in 2:(n + 1)
        s = query[i - 1] == target[j - 1] ? match_score : mismatch_score
        H[i, j] = max(H[i - 1, j - 1] + s, H[i - 1, j] + gap_penalty, H[i, j - 1] + gap_penalty)
    end

    # Traceback
    mapping = Dict{Int, Int}()
    i, j = m + 1, n + 1
    while i > 1 && j > 1
        s = query[i - 1] == target[j - 1] ? match_score : mismatch_score
        if H[i, j] == H[i - 1, j - 1] + s
            mapping[i - 1] = j - 1  # 1-based
            i -= 1
            j -= 1
        elseif H[i, j] == H[i - 1, j] + gap_penalty
            i -= 1
        else
            j -= 1
        end
    end
    return mapping
end

"""
    _parse_cif_seqres(cif_path, chain_id) → (seqres_seq, auth_seq_num_map)

Parse the `_pdbx_poly_seq_scheme` table from a CIF file to get the full SEQRES
sequence for a chain, including unresolved residues. Returns:
- `seqres_seq`: full one-letter sequence string (matching Python's chain_to_seqres)
- `auth_seq_num_map`: Dict mapping 1-based SEQRES position → auth_seq_num (Int),
  only for resolved residues (those with auth_seq_num != "?")

Returns `(nothing, nothing)` if the table is not present in the CIF.
"""
function _parse_cif_seqres(cif_path::AbstractString, chain_id::AbstractString)
    lines = readlines(cif_path)

    # Find the _pdbx_poly_seq_scheme loop
    field_names = String[]
    data_lines = String[]
    in_loop = false
    in_fields = false

    for line in lines
        stripped = rstrip(line)
        if in_loop
            if startswith(stripped, "_pdbx_poly_seq_scheme.")
                push!(field_names, stripped)
                in_fields = true
            elseif in_fields && (startswith(stripped, "_") || stripped == "#" || startswith(stripped, "loop_"))
                break  # end of this loop block
            elseif in_fields && !isempty(stripped)
                push!(data_lines, stripped)
            end
        elseif startswith(stripped, "_pdbx_poly_seq_scheme.")
            push!(field_names, stripped)
            in_loop = true
            in_fields = true
        end
    end

    isempty(field_names) && return (nothing, nothing, nothing)

    # Find column indices
    asym_col = findfirst(f -> endswith(f, ".asym_id"), field_names)
    mon_col = findfirst(f -> endswith(f, ".mon_id"), field_names)
    seq_col = findfirst(f -> endswith(f, ".seq_id"), field_names)
    strand_col = findfirst(f -> endswith(f, ".pdb_strand_id"), field_names)

    (asym_col === nothing || mon_col === nothing || seq_col === nothing) && return (nothing, nothing, nothing)

    # Match chain: try pdb_strand_id (auth chain ID, what users pass) first,
    # then fall back to asym_id (label chain ID)
    match_col = asym_col
    if strand_col !== nothing
        has_strand = any(data_lines) do dline
            tokens = split(dline)
            length(tokens) >= length(field_names) && tokens[strand_col] == chain_id
        end
        if has_strand
            match_col = strand_col
        end
    end

    # Parse data rows for the matched chain.
    # Include ALL rows (even disordered duplicates with the same seq_id),
    # matching Python's TemplateParser behavior.
    # The atom mapping uses sequential numbering: SEQRES position i → label_seq_id i.
    # This matches Python, where seqres_to_structure maps positions to sequential
    # auth_seq_num values. When disordered duplicates cause more SEQRES entries
    # than structural residues, the excess positions simply have no atom records.
    seqres_chars = Char[]
    seq_id_map = Dict{Int, Int}()  # SEQRES position (1-based) → label_seq_id (= position index)
    label_asym_id = nothing

    for dline in data_lines
        tokens = split(dline)
        length(tokens) < length(field_names) && continue
        tokens[match_col] == chain_id || continue

        if label_asym_id === nothing
            label_asym_id = String(tokens[asym_col])
        end

        mon_id = String(tokens[mon_col])
        one_letter = get(_AA_3TO1, mon_id, 'X')
        push!(seqres_chars, one_letter)

        seqres_pos = length(seqres_chars)
        # Sequential mapping: position i → label_seq_id i
        seq_id_map[seqres_pos] = seqres_pos
    end

    isempty(seqres_chars) && return (nothing, nothing, nothing)
    return (String(seqres_chars), seq_id_map, label_asym_id)
end

"""
    _parse_template_cif_atoms(cif_path, label_chain_id)

Parse `_atom_site` from a CIF file for template feature extraction.

Uses `label_asym_id` for chain matching and `label_seq_id` for residue numbering
(matching `seq_id` from `_pdbx_poly_seq_scheme`). This ensures consistent numbering
regardless of auth_seq_id quirks (insertion codes, negative numbers, offsets).

Handles:
- First model only (`pdbx_PDB_model_num`)
- Altloc selection matching BioPython's behavior:
  * Atom disorder (same residue name): first non-'.' altloc (=A, highest occupancy)
  * Residue disorder (different names): last non-'.' altloc (=B+, last conformer)
  * Atoms with altloc '.' are always kept
- HETATM records (for MSE, SEP, TPO etc.)
- MSE → MET normalization

Returns `(atoms_by_seqid, resname_by_seqid)` where:
- `atoms_by_seqid`: Dict{Int, Vector{NamedTuple{(:name,:x,:y,:z), Tuple{String,Float32,Float32,Float32}}}}`
- `resname_by_seqid`: Dict{Int, String}
"""
function _parse_template_cif_atoms(cif_path::AbstractString, label_chain_id::AbstractString)
    lines = readlines(cif_path)

    # Find _atom_site loop
    field_names = String[]
    data_start = 0
    i = 1
    while i <= length(lines)
        if strip(lines[i]) != "loop_"
            i += 1
            continue
        end
        j = i + 1
        flds = String[]
        while j <= length(lines)
            s = strip(lines[j])
            startswith(s, "_") || break
            push!(flds, s)
            j += 1
        end
        if !isempty(flds) && all(startswith(f, "_atom_site.") for f in flds)
            field_names = flds
            data_start = j
            break
        end
        i = j
    end
    isempty(field_names) && error("No _atom_site loop found in $cif_path")

    # Column index lookups
    function _col(name)
        idx = findfirst(==(name), field_names)
        idx === nothing && return 0
        return idx
    end

    col_group    = _col("_atom_site.group_PDB")
    col_asym     = _col("_atom_site.label_asym_id")
    col_seqid    = _col("_atom_site.label_seq_id")
    col_comp     = _col("_atom_site.label_comp_id")
    col_atom     = _col("_atom_site.label_atom_id")
    col_altid    = _col("_atom_site.label_alt_id")
    col_x        = _col("_atom_site.Cartn_x")
    col_y        = _col("_atom_site.Cartn_y")
    col_z        = _col("_atom_site.Cartn_z")
    col_model    = _col("_atom_site.pdbx_PDB_model_num")
    col_element  = _col("_atom_site.type_symbol")
    col_occ      = _col("_atom_site.occupancy")

    (col_asym == 0 || col_seqid == 0 || col_comp == 0 || col_atom == 0) &&
        error("Missing required _atom_site columns in $cif_path")
    (col_x == 0 || col_y == 0 || col_z == 0) &&
        error("Missing coordinate columns in $cif_path")

    n_fields = length(field_names)
    pool = String[]
    keep_model = nothing

    # Phase 1: Parse ALL atoms (including all altlocs) with their altloc labels + occupancies.
    # Phase 2 will post-filter to match BioPython's altloc selection behavior.
    RawAtom = @NamedTuple{name::String, x::Float32, y::Float32, z::Float32,
                          altloc::String, resname::String, occ::Float32}
    raw_atoms_by_seqid = Dict{Int, Vector{RawAtom}}()

    j = data_start
    while j <= length(lines)
        row = strip(lines[j])
        if isempty(row)
            j += 1
            continue
        end
        if row == "#" || row == "loop_" || startswith(row, "_") || startswith(row, "data_")
            if isempty(pool)
                break
            end
        else
            append!(pool, _split_cif_row(row))
            j += 1
        end

        while length(pool) >= n_fields
            toks = pool[1:n_fields]
            pool = length(pool) == n_fields ? String[] : pool[(n_fields + 1):end]

            # Group filter: ATOM or HETATM
            if col_group > 0
                grp = uppercase(toks[col_group])
                (grp == "ATOM" || grp == "HETATM") || continue
            end

            # Model filter: first model only
            if col_model > 0
                model_s = toks[col_model]
                model_num = tryparse(Int, model_s)
                if model_num !== nothing
                    if keep_model === nothing
                        keep_model = model_num
                    elseif model_num != keep_model
                        continue
                    end
                end
            end

            # Chain filter
            toks[col_asym] == label_chain_id || continue

            # Parse residue sequence ID
            seqid = tryparse(Int, toks[col_seqid])
            seqid === nothing && continue

            # Atom name & residue name
            atom_name = uppercase(toks[col_atom])
            isempty(atom_name) && continue
            res_name = uppercase(toks[col_comp])

            # Parse altloc and occupancy (store for post-filtering)
            alt_id = col_altid > 0 ? toks[col_altid] : "."
            occ = col_occ > 0 ? something(tryparse(Float32, toks[col_occ]), 1.0f0) : 1.0f0

            # Parse coordinates
            x = tryparse(Float32, toks[col_x])
            y = tryparse(Float32, toks[col_y])
            z = tryparse(Float32, toks[col_z])
            (x === nothing || y === nothing || z === nothing) && continue

            # Normalize MSE → MET
            element = col_element > 0 ? uppercase(toks[col_element]) : ""
            res_name, atom_name, _ = _normalize_mse(res_name, atom_name, element)

            # Store (all altlocs, will post-filter)
            v = get!(Vector{RawAtom}, raw_atoms_by_seqid, seqid)
            push!(v, RawAtom((atom_name, x, y, z, alt_id, res_name, occ)))
        end
    end

    # Phase 2: Post-filter by altloc, matching BioPython's behavior:
    #
    # BioPython has TWO different altloc selection strategies:
    #   - DisorderedAtom (same residue name, different positions): selects PER-ATOM by
    #     highest occupancy. When occupancies tie, first altloc (A) wins.
    #   - DisorderedResidue (different residue names at same position): selects the LAST
    #     conformer added (last entry in CIF order = highest altloc letter), returning
    #     all atoms from that conformer.
    #
    # We detect the two cases by checking if multiple residue names appear at a seqid.
    TemplateAtom = @NamedTuple{name::String, x::Float32, y::Float32, z::Float32}
    atoms_by_seqid = Dict{Int, Vector{TemplateAtom}}()
    resname_by_seqid = Dict{Int, String}()

    for (seqid, raw_atoms) in raw_atoms_by_seqid
        # Check if this position has residue-level disorder (multiple residue names)
        resnames = Set{String}()
        for ra in raw_atoms
            push!(resnames, ra.resname)
        end
        has_residue_disorder = length(resnames) > 1

        filtered = TemplateAtom[]
        sel_resname = "UNK"

        if has_residue_disorder
            # Residue disorder: keep LAST non-"." altloc (matches BioPython DisorderedResidue)
            last_alt = nothing
            for ra in raw_atoms
                if ra.altloc != "." && ra.altloc != ""
                    last_alt = ra.altloc
                end
            end
            for ra in raw_atoms
                if ra.altloc == "." || ra.altloc == ""
                    push!(filtered, TemplateAtom((ra.name, ra.x, ra.y, ra.z)))
                    sel_resname = ra.resname
                elseif last_alt !== nothing && ra.altloc == last_alt
                    push!(filtered, TemplateAtom((ra.name, ra.x, ra.y, ra.z)))
                    sel_resname = ra.resname
                end
            end
        else
            # Atom disorder: per-atom, pick highest occupancy (BioPython DisorderedAtom).
            # For atoms with altloc ".", keep as-is.
            # For atoms with non-"." altloc, group by atom_name and pick highest occ.
            # Build best altloc per atom_name (highest occupancy; ties → first altloc)
            best_alt = Dict{String, Tuple{String, Float32}}()  # atom_name → (altloc, occ)
            for ra in raw_atoms
                (ra.altloc == "." || ra.altloc == "") && continue
                key = ra.name
                if !haskey(best_alt, key)
                    best_alt[key] = (ra.altloc, ra.occ)
                else
                    _, prev_occ = best_alt[key]
                    if ra.occ > prev_occ
                        best_alt[key] = (ra.altloc, ra.occ)
                    end
                end
            end

            for ra in raw_atoms
                if ra.altloc == "." || ra.altloc == ""
                    push!(filtered, TemplateAtom((ra.name, ra.x, ra.y, ra.z)))
                    sel_resname = ra.resname
                elseif haskey(best_alt, ra.name) && ra.altloc == best_alt[ra.name][1]
                    push!(filtered, TemplateAtom((ra.name, ra.x, ra.y, ra.z)))
                    sel_resname = ra.resname
                end
            end
        end

        if !isempty(filtered)
            atoms_by_seqid[seqid] = filtered
            resname_by_seqid[seqid] = sel_resname
        end
    end

    return (atoms_by_seqid, resname_by_seqid)
end

"""
    _extract_template_from_cif(query_sequence, cif_path, chain_id; zero_center=true)

Extract template features from a CIF file for a given query protein sequence.

Uses the CIF's SEQRES (`_pdbx_poly_seq_scheme`) to get the full template sequence
including unresolved residues (matching Python Protenix's BioPython-based parsing).
Falls back to ATOM-record-only parsing if SEQRES is not available.

Returns a Dict with:
- `"template_restype"` — (1, N_query) Int matrix, 0-indexed residue types
- `"template_all_atom_mask"` — (1, N_query, 24) Float32 array
- `"template_all_atom_positions"` — (1, N_query, 24, 3) Float32 array

These can be passed directly to `_derive_template_features!()` to produce the
distogram, unit_vector, and mask features needed by the template embedder.
"""
function _extract_template_from_cif(
    query_sequence::AbstractString,
    cif_path::AbstractString,
    chain_id::AbstractString;
    zero_center::Bool = true,
)
    # 1. Get SEQRES sequence (full, including unresolved residues)
    #    seq_id_map maps SEQRES pos → seq_id (= label_seq_id in _atom_site)
    seqres_seq, seq_id_map, label_chain = _parse_cif_seqres(cif_path, chain_id)

    # 2. Parse ATOM records with proper altloc filtering + label_seq_id numbering
    if seqres_seq !== nothing && seq_id_map !== nothing && label_chain !== nothing
        atoms_by_seqid, resname_by_seqid = _parse_template_cif_atoms(cif_path, label_chain)
        n_template = length(seqres_seq)
        template_seq_str = seqres_seq
        seqres_to_seqid = seq_id_map  # SEQRES pos → label_seq_id
    else
        # Fallback: no SEQRES available, parse atoms and build sequence from ATOM records
        # Try label_asym_id = chain_id directly
        atoms_by_seqid, resname_by_seqid = _parse_template_cif_atoms(cif_path, chain_id)
        if isempty(atoms_by_seqid)
            error("Chain '$chain_id' not found in $cif_path (no SEQRES and no ATOM records)")
        end
        seqids_ordered = sort(collect(keys(atoms_by_seqid)))
        n_template = length(seqids_ordered)
        template_seq_str = String([get(_AA_3TO1, resname_by_seqid[sid], 'X') for sid in seqids_ordered])
        seqres_to_seqid = Dict{Int, Int}(i => seqids_ordered[i] for i in 1:n_template)
    end

    # 3. Fill atom37 arrays for the template chain
    all_pos = zeros(Float32, n_template, _ATOM37_NUM, 3)
    all_mask = zeros(Float32, n_template, _ATOM37_NUM)

    for si in 1:n_template
        haskey(seqres_to_seqid, si) || continue
        sid = seqres_to_seqid[si]
        haskey(atoms_by_seqid, sid) || continue

        res_atoms = atoms_by_seqid[sid]
        resname = get(resname_by_seqid, sid, "UNK")

        for a in res_atoms
            if haskey(_ATOM37_ORDER, a.name)
                idx = _ATOM37_ORDER[a.name] + 1  # 1-based
                all_pos[si, idx, 1] = a.x
                all_pos[si, idx, 2] = a.y
                all_pos[si, idx, 3] = a.z
                all_mask[si, idx] = 1.0f0
            end
        end

        # Correct Arginine NH1/NH2 swap (matching Python)
        if resname == "ARG"
            cd_idx = _ATOM37_ORDER["CD"] + 1
            nh1_idx = _ATOM37_ORDER["NH1"] + 1
            nh2_idx = _ATOM37_ORDER["NH2"] + 1
            if all_mask[si, cd_idx] > 0 && all_mask[si, nh1_idx] > 0 && all_mask[si, nh2_idx] > 0
                d1 = sqrt(sum((all_pos[si, nh1_idx, :] .- all_pos[si, cd_idx, :]) .^ 2))
                d2 = sqrt(sum((all_pos[si, nh2_idx, :] .- all_pos[si, cd_idx, :]) .^ 2))
                if d1 > d2
                    all_pos[si, [nh1_idx, nh2_idx], :] = all_pos[si, [nh2_idx, nh1_idx], :]
                    all_mask[si, [nh1_idx, nh2_idx]] = all_mask[si, [nh2_idx, nh1_idx]]
                end
            end
        end
    end

    # 6. Zero-center positions (matching Python default)
    if zero_center
        mask_bool = all_mask .> 0
        n_filled = sum(mask_bool)
        if n_filled > 0
            center = zeros(Float32, 3)
            cnt = 0
            for ri in 1:n_template, ai in 1:_ATOM37_NUM
                if mask_bool[ri, ai]
                    center .+= all_pos[ri, ai, :]
                    cnt += 1
                end
            end
            center ./= cnt
            for ri in 1:n_template, ai in 1:_ATOM37_NUM
                if mask_bool[ri, ai]
                    all_pos[ri, ai, :] .-= center
                end
            end
        end
    end

    # 7. Align query to template (SEQRES) sequence
    n_query = length(query_sequence)
    mapping = _kalign_align(query_sequence, template_seq_str)

    n_aligned = length(mapping)
    n_identical = count(kv -> query_sequence[kv.first] == template_seq_str[kv.second], mapping)

    # 8. Map template atoms to query positions (atom37 format)
    out_pos = zeros(Float32, n_query, _ATOM37_NUM, 3)
    out_mask = zeros(Float32, n_query, _ATOM37_NUM)
    out_seq = fill('-', n_query)

    for (q_idx, t_idx) in mapping
        out_pos[q_idx, :, :] .= all_pos[t_idx, :, :]
        out_mask[q_idx, :] .= all_mask[t_idx, :]
        out_seq[q_idx] = template_seq_str[t_idx]
    end

    # Encode residue types
    out_aatype = Int32[get(_TEMPLATE_RESTYPE_ENCODE, c, Int32(20)) for c in out_seq]

    # 8. Convert from atom37 to dense 24-atom representation
    # Matches Python's fix_template_features(): uses PROTEIN_AATYPE_DENSE_ATOM_TO_ATOM37
    dense_pos = zeros(Float32, 1, n_query, 24, 3)
    dense_mask = zeros(Float32, 1, n_query, 24)
    restype_mat = zeros(Int32, 1, n_query)

    for qi in 1:n_query
        rt = out_aatype[qi]  # 0-indexed
        restype_mat[1, qi] = rt
        rt_row = clamp(rt + 1, 1, 21)  # 1-indexed into _PROTEIN_AATYPE_DENSE_ATOM_TO_ATOM37
        for di in 1:24
            a37_idx = _PROTEIN_AATYPE_DENSE_ATOM_TO_ATOM37[rt_row, di]  # 0-indexed atom37
            a37_idx1 = a37_idx + 1  # 1-indexed
            dense_pos[1, qi, di, :] .= out_pos[qi, a37_idx1, :]
            dense_mask[1, qi, di] = out_mask[qi, a37_idx1]
        end
        # Mask positions where mask is zero
        for di in 1:24
            if dense_mask[1, qi, di] == 0
                dense_pos[1, qi, di, :] .= 0
            end
        end
    end

    return Dict{String, Any}(
        "template_restype" => restype_mat,                    # (1, N_query)
        "template_all_atom_mask" => dense_mask,               # (1, N_query, 24)
        "template_all_atom_positions" => dense_pos,           # (1, N_query, 24, 3)
        "_alignment_info" => (
            n_aligned = n_aligned,
            n_identical = n_identical,
            template_length = n_template,
            query_length = n_query,
        ),
    )
end

"""
    template_structure(cif_path, chain_id; chains=nothing)

Build template features from a CIF/PDB structure file for use with the folding API.
Returns a Dict that can be passed as `template_features` in `protenix_task()`.

# Example
```julia
h = load_protenix("protenix_base_default_v1.0.0"; gpu=true)
tmpl = template_structure("structures/1ubq.cif", "A")
task = protenix_task(
    protein_chain("MQIFVKTLTGKTITLEVEPS..."),
    template_features = tmpl,
)
r = fold(h, task)
```
"""
function template_structure(
    cif_path::AbstractString,
    chain_id::AbstractString,
)
    return Dict{String, Any}(
        "_template_cif_path" => abspath(cif_path),
        "_template_chain_id" => chain_id,
    )
end

"""
    _extract_task_protein_sequence(task) → String

Extract the concatenated protein sequence from a task Dict (as built by `protenix_task()`).
Returns the concatenation of all protein chain sequences in entity order.
"""
function _extract_task_protein_sequence(task::AbstractDict)
    seqs = String[]
    sequences = get(task, "sequences", Any[])
    for entity in sequences
        ed = _as_string_dict(entity)
        if haskey(ed, "protein")
            pd = _as_string_dict(ed["protein"])
            seq = get(pd, "sequence", nothing)
            seq !== nothing && push!(seqs, String(seq))
        elseif haskey(ed, "proteinChain")
            pd = _as_string_dict(ed["proteinChain"])
            seq = get(pd, "sequence", nothing)
            seq !== nothing && push!(seqs, String(seq))
        end
    end
    isempty(seqs) && error("No protein chains found in task — templates require at least one protein entity.")
    return join(seqs, "")
end

function _inject_task_template_features!(
    feat::Dict{String, Any},
    task::AbstractDict{<:Any, <:Any},
)
    haskey(task, "template_features") || return feat
    tf_any = task["template_features"]
    tf_any isa AbstractDict || error("task.template_features must be an object when provided.")
    tf = _as_string_dict(tf_any)

    n_tok = size(feat["restype"], 1)

    # ── Lazy CIF path: resolve template_structure() dict ──
    if haskey(tf, "_template_cif_path")
        cif_path = tf["_template_cif_path"]
        chain_id = tf["_template_chain_id"]

        # Extract protein sequence from the task entities
        protein_seq = _extract_task_protein_sequence(task)
        n_protein = length(protein_seq)

        # Extract template features aligned to the protein sequence
        tmpl = _extract_template_from_cif(protein_seq, cif_path, chain_id)
        info = tmpl["_alignment_info"]
        @info "Template: $(info.n_aligned) aligned positions, $(info.n_identical) identical " *
              "($(round(100 * info.n_identical / max(info.n_aligned, 1); digits=1))% identity)"

        if n_protein == n_tok
            # Pure protein task — template covers all tokens
            feat["template_restype"] = tmpl["template_restype"]
            feat["template_all_atom_mask"] = tmpl["template_all_atom_mask"]
            feat["template_all_atom_positions"] = tmpl["template_all_atom_positions"]
        else
            # Multi-entity task — embed protein template into full token dimension
            restype_vec = feat["restype"]  # (n_tok,)
            protein_indices = findall(i -> 0 <= restype_vec[i] <= 19, 1:n_tok)

            full_restype = fill(Int32(31), 1, n_tok)  # gap token
            full_mask = zeros(Float32, 1, n_tok, 24)
            full_pos = zeros(Float32, 1, n_tok, 24, 3)

            for (pi, ti) in enumerate(protein_indices)
                if pi <= n_protein
                    full_restype[1, ti] = tmpl["template_restype"][1, pi]
                    full_mask[1, ti, :] .= tmpl["template_all_atom_mask"][1, pi, :]
                    full_pos[1, ti, :, :] .= tmpl["template_all_atom_positions"][1, pi, :, :]
                end
            end

            feat["template_restype"] = full_restype
            feat["template_all_atom_mask"] = full_mask
            feat["template_all_atom_positions"] = full_pos
        end
        return feat
    end

    # ── Pre-computed tensor path ──
    required = ("template_restype", "template_all_atom_mask", "template_all_atom_positions")
    all(haskey(tf, k) for k in required) || error(
        "template_features must include template_restype/template_all_atom_mask/template_all_atom_positions.",
    )

    template_restype = _to_int_array(tf["template_restype"])
    template_mask = _to_float_array(tf["template_all_atom_mask"])
    template_pos = _to_float_array(tf["template_all_atom_positions"])

    ndims(template_restype) == 2 || error("template_restype must be rank-2 [N_template, N_token].")
    ndims(template_mask) == 3 || error("template_all_atom_mask must be rank-3 [N_template, N_token, N_atom].")
    ndims(template_pos) == 4 || error("template_all_atom_positions must be rank-4 [N_template, N_token, N_atom, 3].")

    size(template_restype, 2) == n_tok || error("template_restype token length mismatch.")
    size(template_mask, 2) == n_tok || error("template_all_atom_mask token length mismatch.")
    size(template_pos, 2) == n_tok || error("template_all_atom_positions token length mismatch.")
    size(template_mask, 3) == 24 || error("template_all_atom_mask must have 24 dense atom slots.")
    size(template_pos, 3) == 24 || error("template_all_atom_positions must have 24 dense atom slots.")
    size(template_pos, 4) == 3 || error("template_all_atom_positions final dimension must be xyz=3.")
    size(template_restype, 1) == size(template_mask, 1) == size(template_pos, 1) || error(
        "template feature N_template dimensions must match.",
    )

    feat["template_restype"] = template_restype
    feat["template_all_atom_mask"] = template_mask
    feat["template_all_atom_positions"] = template_pos
    return feat
end

"""
    _inject_dummy_template_features!(feat, n_templates=1)

Create dummy (zero-filled) template features matching Python Protenix v1.0.4
behaviour when `use_template=false`. This ensures the TemplateEmbedder code path
is exercised (all masks are zero → output is zero).

The derived features (distogram, unit_vector, masks) are all zeros because the
underlying atom positions and masks are zeros. template_restype is filled with 31
(gap token) matching Python's make_dummy_feature().
"""
function _inject_dummy_template_features!(
    feat::Dict{String, Any};
    n_templates::Int = 1,
)
    n_tok = size(feat["restype"], 1)

    # Raw template features: restype filled with 31 (gap), positions/masks zeros
    if !haskey(feat, "template_restype")
        feat["template_restype"] = fill(31, n_templates, n_tok)
    end
    if !haskey(feat, "template_all_atom_mask")
        feat["template_all_atom_mask"] = zeros(Float32, n_templates, n_tok, 24)
    end
    if !haskey(feat, "template_all_atom_positions")
        feat["template_all_atom_positions"] = zeros(Float32, n_templates, n_tok, 24, 3)
    end

    # Derive distogram, unit_vector, masks from raw atom data
    _derive_template_features!(feat)

    return feat
end

# ─── Template feature derivation from raw atom data ──────────────────────────
# Matches Python Protenix v1.0.4 template_utils.py

# 0-indexed pseudo-beta atom index per restype (32 entries).
# Protein: CB=4 (GLY=CA=1), RNA: A/G→C4, C/U→C2, DNA: DA/DG→C4, DC/DT→C2.
const _RESTYPE_PSEUDOBETA_INDEX = Int32[
    4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,  # ALA-VAL (0-19)
    0,                                                                 # UNK (20)
    22, 23, 14, 14, 0,                                                 # A, G, C, U, N (21-25)
    21, 22, 13, 13, 0,                                                 # DA, DG, DC, DT, DN (26-30)
    0,                                                                 # gap (31)
]

# Backbone frame atom indices (C, CA, N) per restype, group 0 of rigidgroup.
# Each row = (C_idx, CA_idx, N_idx) in dense atom format (0-indexed).
const _RESTYPE_BACKBONE_ATOM_IDX = (
    # Protein residues 0-19: C=2, CA=1, N=0
    (2, 1, 0), (2, 1, 0), (2, 1, 0), (2, 1, 0), (2, 1, 0),  # ALA-CYS
    (2, 1, 0), (2, 1, 0), (2, 1, 0), (2, 1, 0), (2, 1, 0),  # GLN-ILE
    (2, 1, 0), (2, 1, 0), (2, 1, 0), (2, 1, 0), (2, 1, 0),  # LEU-PRO
    (2, 1, 0), (2, 1, 0), (2, 1, 0), (2, 1, 0), (2, 1, 0),  # SER-VAL
    (0, 0, 0),                                                  # UNK
    (12, 8, 6), (12, 8, 6), (12, 8, 6), (12, 8, 6), (0, 0, 0), # A, G, C, U, N
    (12, 8, 6), (12, 8, 6), (12, 8, 6), (12, 8, 6), (0, 0, 0), # DA, DG, DC, DT, DN
    (0, 0, 0),                                                  # gap
)

"""
    _template_pseudo_beta(aatype, atom_positions, atom_mask)

Compute pseudo-beta positions and masks from raw template atom data.
- `aatype`: (N_res,) 0-indexed residue type indices
- `atom_positions`: (N_res, 24, 3) dense atom positions
- `atom_mask`: (N_res, 24) dense atom mask
Returns: (pb_positions, pb_mask) where pb_positions is (N_res, 3) and pb_mask is (N_res,)
"""
function _template_pseudo_beta(
    aatype::AbstractVector{<:Integer},
    atom_positions::AbstractArray{<:Real,3},
    atom_mask::AbstractMatrix{<:Real},
)
    n_res = length(aatype)
    pb_pos = zeros(Float32, n_res, 3)
    pb_mask = zeros(Float32, n_res)
    for i in 1:n_res
        # 0-indexed aatype → 1-indexed lookup into _RESTYPE_PSEUDOBETA_INDEX
        at = aatype[i]
        pb_idx = _RESTYPE_PSEUDOBETA_INDEX[clamp(at + 1, 1, 32)]
        # 0-indexed atom index → 1-indexed array access
        ai = pb_idx + 1
        if ai <= size(atom_positions, 2)
            pb_pos[i, :] .= Float32.(atom_positions[i, ai, :])
            pb_mask[i] = Float32(atom_mask[i, ai])
        end
    end
    return pb_pos, pb_mask
end

"""
    _template_distogram(positions; min_bin=3.25, max_bin=50.75, num_bins=39)

Compute distogram from positions. Input: (N_res, 3). Output: (N_res, N_res, num_bins).
Bins are in squared Ångström space (lower_breaks² < dist² < upper_breaks²).
"""
function _template_distogram(
    positions::AbstractMatrix{<:Real};
    min_bin::Float32 = 3.25f0,
    max_bin::Float32 = 50.75f0,
    num_bins::Int = 39,
)
    lower = Float32.(range(min_bin, max_bin, length=num_bins)) .^ 2
    upper = vcat(lower[2:end], Float32[1f8])
    n = size(positions, 1)
    dgram = zeros(Float32, n, n, num_bins)
    for i in 1:n, j in 1:n
        d2 = sum(k -> (positions[i, k] - positions[j, k])^2, 1:3)
        for b in 1:num_bins
            if d2 > lower[b] && d2 < upper[b]
                dgram[i, j, b] = 1f0
            end
        end
    end
    return dgram
end

"""
    _template_unit_vector(aatype, atom_positions, atom_mask; epsilon=1e-6)

Compute template unit vectors and backbone frame mask.
- Local frame: origin at CA, x-axis along C→CA, y-axis Gram-Schmidt of N→CA.
- unit_vector[i,j,:] = normalized (CA_j - CA_i) in local frame of residue i.
Returns: (unit_vector, mask_2d) where shapes are (N_res, N_res, 3) and (N_res, N_res).
"""
function _template_unit_vector(
    aatype::AbstractVector{<:Integer},
    atom_positions::AbstractArray{<:Real,3},
    atom_mask::AbstractMatrix{<:Real};
    epsilon::Float32 = 1f-6,
)
    n_res = length(aatype)
    T = Float32

    # Extract backbone atom positions
    c_pos  = zeros(T, n_res, 3)
    ca_pos = zeros(T, n_res, 3)
    n_pos  = zeros(T, n_res, 3)
    mask   = ones(T, n_res)

    for i in 1:n_res
        at = aatype[i]
        bb = _RESTYPE_BACKBONE_ATOM_IDX[clamp(at + 1, 1, 32)]
        c_idx, ca_idx, n_idx = bb[1] + 1, bb[2] + 1, bb[3] + 1  # 1-indexed

        c_pos[i, :]  .= T.(atom_positions[i, c_idx, :])
        ca_pos[i, :] .= T.(atom_positions[i, ca_idx, :])
        n_pos[i, :]  .= T.(atom_positions[i, n_idx, :])

        c_m  = T(atom_mask[i, c_idx])
        ca_m = T(atom_mask[i, ca_idx])
        n_m  = T(atom_mask[i, n_idx])
        mask[i] = c_m * ca_m * n_m
    end

    # Build local frames
    v1 = c_pos .- ca_pos   # (n_res, 3)
    v2 = n_pos .- ca_pos

    # Normalize v1 → e1
    e1_norm = sqrt.(sum(v1 .^ 2, dims=2) .+ epsilon)
    e1 = v1 ./ e1_norm

    # Gram-Schmidt: e2 = orthogonalize v2 against e1
    proj = sum(v2 .* e1, dims=2)
    e2_raw = v2 .- proj .* e1
    e2_norm = sqrt.(sum(e2_raw .^ 2, dims=2) .+ epsilon)
    e2 = e2_raw ./ e2_norm

    # e3 = e1 × e2
    e3 = hcat(
        e1[:, 2] .* e2[:, 3] .- e1[:, 3] .* e2[:, 2],
        e1[:, 3] .* e2[:, 1] .- e1[:, 1] .* e2[:, 3],
        e1[:, 1] .* e2[:, 2] .- e1[:, 2] .* e2[:, 1],
    )

    # Compute unit vectors: diff[i,j] = CA[j] - CA[i], then project into local frame
    unit_vector = zeros(T, n_res, n_res, 3)
    for i in 1:n_res, j in 1:n_res
        dx = ca_pos[j, 1] - ca_pos[i, 1]
        dy = ca_pos[j, 2] - ca_pos[i, 2]
        dz = ca_pos[j, 3] - ca_pos[i, 3]
        ux = e1[i, 1] * dx + e1[i, 2] * dy + e1[i, 3] * dz
        uy = e2[i, 1] * dx + e2[i, 2] * dy + e2[i, 3] * dz
        uz = e3[i, 1] * dx + e3[i, 2] * dy + e3[i, 3] * dz
        uv_norm = sqrt(ux^2 + uy^2 + uz^2) + epsilon
        unit_vector[i, j, 1] = ux / uv_norm
        unit_vector[i, j, 2] = uy / uv_norm
        unit_vector[i, j, 3] = uz / uv_norm
    end

    # 2D mask
    mask_2d = mask * mask'

    return unit_vector, mask_2d
end

"""
    _derive_template_features!(feat)

Derive template embedding features (distogram, unit_vector, pseudo_beta_mask,
backbone_frame_mask) from raw template features (template_restype,
template_all_atom_positions, template_all_atom_mask).

Matches Python Protenix v1.0.4 `Templates.as_protenix_dict()`.
All arrays use Python/features-last layout: (N_tmpl, N_tok, ...).
"""
function _derive_template_features!(feat::Dict{String, Any})
    has_restype = haskey(feat, "template_restype")
    has_positions = haskey(feat, "template_all_atom_positions")
    has_mask = haskey(feat, "template_all_atom_mask")
    # No template features at all — nothing to derive
    (!has_restype && !has_positions && !has_mask) && return feat
    # Partial template features — caller bug
    (has_restype && has_positions && has_mask) || error(
        "Incomplete template features: template_restype=$has_restype, " *
        "template_all_atom_positions=$has_positions, template_all_atom_mask=$has_mask " *
        "(all three must be present or all absent)"
    )

    restype = feat["template_restype"]      # (N_tmpl, N_tok)
    positions = feat["template_all_atom_positions"]  # (N_tmpl, N_tok, 24, 3)
    masks = feat["template_all_atom_mask"]  # (N_tmpl, N_tok, 24)

    n_tmpl = size(restype, 1)
    n_tok = size(restype, 2)

    dgrams = zeros(Float32, n_tmpl, n_tok, n_tok, 39)
    pb_masks_2d = zeros(Float32, n_tmpl, n_tok, n_tok)
    unit_vecs = zeros(Float32, n_tmpl, n_tok, n_tok, 3)
    bb_masks_2d = zeros(Float32, n_tmpl, n_tok, n_tok)

    for t in 1:n_tmpl
        aatype_t = Int.(restype[t, :])
        pos_t = positions[t, :, :, :]    # (N_tok, 24, 3)
        mask_t = masks[t, :, :]          # (N_tok, 24)

        # Pseudo-beta positions and mask
        pb_pos, pb_mask = _template_pseudo_beta(aatype_t, pos_t, mask_t)
        pb_mask_2d = pb_mask * pb_mask'   # (N_tok, N_tok)

        # Distogram from pseudo-beta positions
        dgram = _template_distogram(pb_pos)  # (N_tok, N_tok, 39)

        # Unit vectors and backbone frame mask
        uv, bb_mask = _template_unit_vector(aatype_t, pos_t, mask_t)

        # Apply masks (matching Python: dgram * pb_mask_2d[..., None])
        for i in 1:n_tok, j in 1:n_tok
            dgram[i, j, :] .*= pb_mask_2d[i, j]
            uv[i, j, :] .*= bb_mask[i, j]
        end

        dgrams[t, :, :, :] .= dgram
        pb_masks_2d[t, :, :] .= pb_mask_2d
        unit_vecs[t, :, :, :] .= uv
        bb_masks_2d[t, :, :] .= bb_mask
    end

    feat["template_distogram"] = dgrams
    feat["template_pseudo_beta_mask"] = pb_masks_2d
    feat["template_unit_vector"] = unit_vecs
    feat["template_backbone_frame_mask"] = bb_masks_2d
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

function _constraint_is_empty(x)
    return (x isa AbstractDict && isempty(x)) || (x isa AbstractVector && isempty(x))
end

function _token_lookup_for_constraints(
    atoms::Vector{AtomRecord},
    atom_to_token_idx::AbstractVector{<:Integer},
)
    token_lookup = Dict{Tuple{String, Int}, Vector{Int}}()
    atom_lookup = Dict{Tuple{String, Int, String}, Vector{Int}}()
    chain_tokens = Dict{String, Vector{Int}}()
    for (atom_i, a) in enumerate(atoms)
        tok = Int(atom_to_token_idx[atom_i]) + 1
        key_res = (a.chain_id, a.res_id)
        if haskey(token_lookup, key_res)
            tok in token_lookup[key_res] || push!(token_lookup[key_res], tok)
        else
            token_lookup[key_res] = [tok]
        end

        key_atom = (a.chain_id, a.res_id, uppercase(a.atom_name))
        if haskey(atom_lookup, key_atom)
            push!(atom_lookup[key_atom], atom_i)
        else
            atom_lookup[key_atom] = [atom_i]
        end

        if haskey(chain_tokens, a.chain_id)
            tok in chain_tokens[a.chain_id] || push!(chain_tokens[a.chain_id], tok)
        else
            chain_tokens[a.chain_id] = [tok]
        end
    end
    return token_lookup, atom_lookup, chain_tokens
end

function _constraint_side_tokens(
    token_lookup::Dict{Tuple{String, Int}, Vector{Int}},
    atom_lookup::Dict{Tuple{String, Int, String}, Vector{Int}},
    atom_to_token_idx::AbstractVector{<:Integer},
    chains::Vector{String},
    position::Int,
    atom_any,
    entity_id::Int,
    entity_atom_map::Vector{Dict{Int, String}},
    context::String,
)
    if atom_any === nothing
        out = Int[]
        for chain in chains
            key = (chain, position)
            haskey(token_lookup, key) || error("$context did not match residue at $chain:$position")
            append!(out, token_lookup[key])
        end
        return sort!(unique!(out))
    end

    atom_name = _resolve_bond_atom_name(atom_any, entity_id, entity_atom_map, context)
    out = Int[]
    for chain in chains
        key = (chain, position, uppercase(strip(atom_name)))
        haskey(atom_lookup, key) || error("$context did not match atom '$atom_name' at $chain:$position")
        for atom_i in atom_lookup[key]
            push!(out, Int(atom_to_token_idx[atom_i]) + 1)
        end
    end
    return sort!(unique!(out))
end

function _inject_constraint_feature_from_json!(
    feat::Dict{String, Any},
    atoms::Vector{AtomRecord},
    task::AbstractDict{<:Any, <:Any},
    entity_chain_ids::Vector{Vector{String}},
    entity_atom_map::Vector{Dict{Int, String}},
    context::AbstractString,
)
    haskey(task, "constraint") || return feat
    c_any = task["constraint"]
    c_any isa AbstractDict || error("task.constraint must be an object for $context")
    _constraint_is_empty(c_any) && return feat
    c = _as_string_dict(c_any)

    n_tok = size(feat["restype"], 1)
    atom_to_token_idx = Int.(feat["atom_to_token_idx"])
    token_lookup, atom_lookup, chain_tokens = _token_lookup_for_constraints(atoms, atom_to_token_idx)

    contact = zeros(Float32, n_tok, n_tok, 2)
    contact_atom = zeros(Float32, n_tok, n_tok, 2)
    pocket = zeros(Float32, n_tok, n_tok, 1)
    substructure = zeros(Float32, n_tok, n_tok, 4)

    if haskey(c, "contact")
        pairs_any = c["contact"]
        pairs_any isa AbstractVector || error("task.constraint.contact must be an array for $context")
        for (i, pair_any) in enumerate(pairs_any)
            pair_any isa AbstractDict || error("task.constraint.contact[$i] must be an object")
            pair = _as_string_dict(pair_any)

            left_raw = haskey(pair, "atom1") ? pair["atom1"] : get(pair, "residue1", nothing)
            right_raw = haskey(pair, "atom2") ? pair["atom2"] : get(pair, "residue2", nothing)
            left_raw === nothing && error("task.constraint.contact[$i] missing atom1/residue1")
            right_raw === nothing && error("task.constraint.contact[$i] missing atom2/residue2")
            left_raw isa AbstractVector || error("task.constraint.contact[$i].atom1/residue1 must be an array")
            right_raw isa AbstractVector || error("task.constraint.contact[$i].atom2/residue2 must be an array")

            left = collect(left_raw)
            right = collect(right_raw)
            (length(left) == 3 || length(left) == 4) || error("task.constraint.contact[$i].atom1/residue1 must have length 3 or 4")
            (length(right) == 3 || length(right) == 4) || error("task.constraint.contact[$i].atom2/residue2 must have length 3 or 4")

            left_is_atom = length(left) == 4
            right_is_atom = length(right) == 4
            left_is_atom == right_is_atom || error("task.constraint.contact[$i] must use atom1+atom2 or residue1+residue2 consistently")

            e1 = left[1] isa Integer ? Int(left[1]) : parse(Int, strip(String(left[1])))
            c1 = left[2]
            p1 = left[3] isa Integer ? Int(left[3]) : parse(Int, strip(String(left[3])))
            a1 = left_is_atom ? left[4] : nothing

            e2 = right[1] isa Integer ? Int(right[1]) : parse(Int, strip(String(right[1])))
            c2 = right[2]
            p2 = right[3] isa Integer ? Int(right[3]) : parse(Int, strip(String(right[3])))
            a2 = right_is_atom ? right[4] : nothing

            copy1 = c1 isa Integer ? Int(c1) : parse(Int, strip(String(c1)))
            copy2 = c2 isa Integer ? Int(c2) : parse(Int, strip(String(c2)))
            if e1 == e2 && copy1 == copy2
                error("A contact pair can not be specified on the same chain")
            end

            chains1 = _resolve_bond_chains(entity_chain_ids, e1, c1, "task.constraint.contact[$i] side1")
            chains2 = _resolve_bond_chains(entity_chain_ids, e2, c2, "task.constraint.contact[$i] side2")
            toks1 = _constraint_side_tokens(
                token_lookup,
                atom_lookup,
                atom_to_token_idx,
                chains1,
                p1,
                a1,
                e1,
                entity_atom_map,
                "task.constraint.contact[$i] side1",
            )
            toks2 = _constraint_side_tokens(
                token_lookup,
                atom_lookup,
                atom_to_token_idx,
                chains2,
                p2,
                a2,
                e2,
                entity_atom_map,
                "task.constraint.contact[$i] side2",
            )

            max_dist = Float32(pair["max_distance"])
            min_dist = haskey(pair, "min_distance") ? Float32(pair["min_distance"]) : 0f0
            max_dist >= min_dist || error("max_distance must be greater than or equal to min_distance")
            target = left_is_atom ? contact_atom : contact
            for t1 in toks1, t2 in toks2
                target[t1, t2, 1] = min_dist
                target[t1, t2, 2] = max_dist
                target[t2, t1, 1] = min_dist
                target[t2, t1, 2] = max_dist
            end
        end
    end

    if haskey(c, "pocket")
        p_any = c["pocket"]
        p_any isa AbstractDict || error("task.constraint.pocket must be an object for $context")
        p = _as_string_dict(p_any)
        if !_constraint_is_empty(p)
            haskey(p, "binder_chain") || error("task.constraint.pocket.binder_chain is required for $context")
            haskey(p, "contact_residues") || error("task.constraint.pocket.contact_residues is required for $context")
            haskey(p, "max_distance") || error("task.constraint.pocket.max_distance is required for $context")
            binder = collect(p["binder_chain"])
            length(binder) == 2 || error("task.constraint.pocket.binder_chain must have length 2 [entity,copy]")
            be = binder[1] isa Integer ? Int(binder[1]) : parse(Int, strip(String(binder[1])))
            bc = binder[2]
            bc_i = bc isa Integer ? Int(bc) : parse(Int, strip(String(bc)))
            bchains = _resolve_bond_chains(entity_chain_ids, be, bc, "task.constraint.pocket binder")
            btoks = Int[]
            for chain in bchains
                haskey(chain_tokens, chain) || error("task.constraint.pocket binder chain $chain not found")
                append!(btoks, chain_tokens[chain])
            end
            btoks = sort!(unique!(btoks))

            contact_res = p["contact_residues"]
            contact_res isa AbstractVector || error("task.constraint.pocket.contact_residues must be an array")
            dist = Float32(p["max_distance"])
            for (j, r_any) in enumerate(contact_res)
                r_any isa AbstractVector || error("task.constraint.pocket.contact_residues[$j] must be an array")
                r = collect(r_any)
                length(r) == 3 || error("task.constraint.pocket.contact_residues[$j] must have length 3 [entity,copy,position]")
                re = r[1] isa Integer ? Int(r[1]) : parse(Int, strip(String(r[1])))
                rc = r[2]
                rc_i = rc isa Integer ? Int(rc) : parse(Int, strip(String(rc)))
                rp = r[3] isa Integer ? Int(r[3]) : parse(Int, strip(String(r[3])))
                if be == re && bc_i == rc_i
                    error("Pockets can not be the same chain with the binder")
                end
                rchains = _resolve_bond_chains(entity_chain_ids, re, rc, "task.constraint.pocket.contact_residues[$j]")
                rtoks = _constraint_side_tokens(
                    token_lookup,
                    atom_lookup,
                    atom_to_token_idx,
                    rchains,
                    rp,
                    nothing,
                    re,
                    entity_atom_map,
                    "task.constraint.pocket.contact_residues[$j]",
                )
                for t1 in btoks, t2 in rtoks
                    pocket[t1, t2, 1] = dist
                    pocket[t2, t1, 1] = dist
                end
            end
        end
    end

    if haskey(c, "structure")
        # Python reference currently accepts `constraint.structure` but leaves this path as
        # a no-op in JSON inference (`ConstraintFeatureGenerator.generate_from_json`).
        # Keep runtime parity by accepting and ignoring non-empty payloads here.
        s_any = c["structure"]
        (s_any isa AbstractDict || s_any isa AbstractVector) ||
            error("task.constraint.structure must be an object/array for $context")
    end

    feat["constraint_feature"] = (
        contact = contact,
        pocket = pocket,
        contact_atom = contact_atom,
        substructure = substructure,
    )
    return feat
end

function _inject_task_constraint_feature!(
    feat::Dict{String, Any},
    task::AbstractDict{<:Any, <:Any},
    atoms::Vector{AtomRecord},
    entity_chain_ids::Vector{Vector{String}},
    entity_atom_map::Vector{Dict{Int, String}},
    context::AbstractString,
)
    if haskey(task, "constraint_feature")
        cf_any = task["constraint_feature"]
        cf_any isa AbstractDict || error("task.constraint_feature must be an object for $context")
        cf = _as_string_dict(cf_any)
        n_tok = size(feat["restype"], 1)
        required = ("contact", "pocket", "contact_atom", "substructure")
        all(haskey(cf, k) for k in required) || error("task.constraint_feature must include contact/pocket/contact_atom/substructure")
        contact = Float32.(cf["contact"])
        pocket = Float32.(cf["pocket"])
        contact_atom = Float32.(cf["contact_atom"])
        substructure = Float32.(cf["substructure"])
        ndims(contact) == 3 && size(contact, 1) == n_tok && size(contact, 2) == n_tok || error("task.constraint_feature.contact must be [N_token,N_token,C]")
        ndims(pocket) == 3 && size(pocket, 1) == n_tok && size(pocket, 2) == n_tok || error("task.constraint_feature.pocket must be [N_token,N_token,C]")
        ndims(contact_atom) == 3 && size(contact_atom, 1) == n_tok && size(contact_atom, 2) == n_tok || error("task.constraint_feature.contact_atom must be [N_token,N_token,C]")
        ndims(substructure) == 3 && size(substructure, 1) == n_tok && size(substructure, 2) == n_tok || error("task.constraint_feature.substructure must be [N_token,N_token,C]")
        feat["constraint_feature"] = (
            contact = contact,
            pocket = pocket,
            contact_atom = contact_atom,
            substructure = substructure,
        )
        return feat
    end

    # Needed by _inject_constraint_feature_from_json! token-lookup helper.
    _inject_constraint_feature_from_json!(feat, atoms, task, entity_chain_ids, entity_atom_map, context)
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
            "Provide task.esm_token_embedding [N_token,D], pass esm_token_embedding in sequence mode, " *
            "or configure automatic ESM generation via ESMFold.jl.",
        )
    end
    return feat
end

function _esm_variant_for_model(model_name::AbstractString)
    name = lowercase(strip(String(model_name)))
    return occursin("ism", name) ? :esm2_3b_ism : :esm2_3b
end

function _protein_chain_sequence_map(specs::AbstractVector{<:ProteinChainSpec})
    out = Dict{String, String}()
    for spec in specs
        out[spec.chain_id] = spec.sequence
    end
    return out
end

function _inject_auto_esm_token_embedding!(
    feat::Dict{String, Any},
    atoms::Vector{AtomRecord},
    tokens,
    chain_sequences::Dict{String, String},
    params::NamedTuple,
    context::AbstractString,
)
    haskey(feat, "esm_token_embedding") && return feat
    params.needs_esm_embedding || return feat
    isempty(chain_sequences) && error("No protein sequences available for automatic ESM embedding ($context).")

    n_tok = size(feat["restype"], 1)
    length(tokens) == n_tok || error(
        "Token/feature length mismatch while generating esm_token_embedding for $context.",
    )

    variant = _esm_variant_for_model(params.model_name)
    unique_sequences = sort!(collect(Set(values(chain_sequences))))
    seq_embeddings = Dict{String, Matrix{Float32}}()
    emb_dim = 0
    for seq in unique_sequences
        emb = ESMProvider.embed_sequence(seq; variant = variant)
        emb_dim = emb_dim == 0 ? size(emb, 2) : emb_dim
        size(emb, 2) == emb_dim || error("Inconsistent ESM embedding dimensions for $context.")
        seq_embeddings[seq] = emb
    end
    emb_dim > 0 || error("Failed to infer ESM embedding dimension for $context.")

    x_esm = zeros(Float32, n_tok, emb_dim)
    has_protein_token = false
    for (tok_idx, tok) in enumerate(tokens)
        centre_atom = atoms[tok.centre_atom_index]
        centre_atom.mol_type == "protein" || continue
        has_protein_token = true

        sequence = get(chain_sequences, centre_atom.chain_id, nothing)
        sequence === nothing && error(
            "Missing protein sequence for chain '$(centre_atom.chain_id)' while building esm_token_embedding ($context).",
        )
        emb = seq_embeddings[sequence]
        (1 <= centre_atom.res_id <= size(emb, 1)) || error(
            "Residue index $(centre_atom.res_id) out of range for chain '$(centre_atom.chain_id)' " *
            "with sequence length $(size(emb, 1)) while building esm_token_embedding ($context).",
        )
        @views x_esm[tok_idx, :] .= emb[centre_atom.res_id, :]
    end

    has_protein_token || error("No protein tokens found while generating esm_token_embedding for $context.")
    feat["esm_token_embedding"] = x_esm
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

    out = ChainSequenceRecord[]
    for chain_id in chain_ids
        residues = ChainResidueRecord[]
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

_is_cif_missing(x::AbstractString) = begin
    s = strip(String(x))
    isempty(s) || s == "." || s == "?"
end

function _mmcif_loop_table(path::AbstractString, field_prefix::AbstractString)
    lines = readlines(path)
    i = 1
    while i <= length(lines)
        if strip(lines[i]) != "loop_"
            i += 1
            continue
        end

        j = i + 1
        fields = String[]
        while j <= length(lines)
            s = strip(lines[j])
            startswith(s, "_") || break
            push!(fields, s)
            j += 1
        end
        if isempty(fields) || !all(startswith(f, field_prefix) for f in fields)
            i = j
            continue
        end

        n_fields = length(fields)
        pool = String[]
        rows = Vector{Vector{String}}()
        while j <= length(lines)
            row = strip(lines[j])
            if isempty(row)
                j += 1
                continue
            end
            if row == "#" || row == "loop_" || startswith(row, "_") || startswith(row, "data_")
                if isempty(pool)
                    break
                end
            else
                append!(pool, _split_cif_tokens(lines[j]))
                j += 1
            end

            while length(pool) >= n_fields
                push!(rows, pool[1:n_fields])
                if length(pool) == n_fields
                    empty!(pool)
                else
                    pool = pool[(n_fields + 1):end]
                end
            end
        end
        return (fields = fields, rows = rows)
    end
    return nothing
end

function _mmcif_field_index(fields::Vector{String}, candidates::Vector{String})
    for name in candidates
        idx = findfirst(==(name), fields)
        idx !== nothing && return idx
    end
    return nothing
end

function _mmcif_label_to_auth_map(path::AbstractString)
    tab = _mmcif_loop_table(path, "_atom_site.")
    tab === nothing && return Dict{String, String}()
    f = tab.fields
    rows = tab.rows
    idx_label = _mmcif_field_index(f, ["_atom_site.label_asym_id"])
    idx_auth = _mmcif_field_index(f, ["_atom_site.auth_asym_id"])
    idx_label === nothing && return Dict{String, String}()
    idx_label = idx_label::Int
    idx_auth = idx_auth === nothing ? nothing : (idx_auth::Int)

    out = Dict{String, String}()
    for row in rows
        idx_label <= length(row) || continue
        label = strip(String(row[idx_label]))
        _is_cif_missing(label) && continue
        auth = if idx_auth === nothing || idx_auth > length(row)
            label
        else
            v = strip(String(row[idx_auth]))
            _is_cif_missing(v) ? label : v
        end
        haskey(out, label) || (out[label] = auth)
    end
    return out
end

function _oper_group_count(group_expr::AbstractString)
    s = strip(String(group_expr))
    isempty(s) && return 0
    total = 0
    for item_raw in split(s, ',')
        item = strip(item_raw)
        isempty(item) && continue
        if occursin('-', item)
            parts = split(item, '-')
            if length(parts) == 2
                a = tryparse(Int, strip(parts[1]))
                b = tryparse(Int, strip(parts[2]))
                if a !== nothing && b !== nothing
                    total += abs(b - a) + 1
                    continue
                end
            end
        end
        total += 1
    end
    return total
end

function _operation_expression_count(expr::AbstractString)
    s = strip(String(expr))
    isempty(s) && return 1
    groups = [m.captures[1] for m in eachmatch(r"\(([^()]*)\)", s)]
    if isempty(groups)
        return max(_oper_group_count(s), 1)
    end
    count = 1
    for g in groups
        count *= max(_oper_group_count(g), 1)
    end
    return count
end

function _mmcif_assembly_chain_copy_counts(path::AbstractString, assembly_id::AbstractString)
    tab = _mmcif_loop_table(path, "_pdbx_struct_assembly_gen.")
    tab === nothing && return nothing

    f = tab.fields
    rows = tab.rows
    idx_assembly = _mmcif_field_index(f, ["_pdbx_struct_assembly_gen.assembly_id"])
    idx_oper = _mmcif_field_index(f, ["_pdbx_struct_assembly_gen.oper_expression"])
    idx_asym = _mmcif_field_index(f, ["_pdbx_struct_assembly_gen.asym_id_list"])
    idx_assembly === nothing && return nothing
    idx_oper === nothing && return nothing
    idx_asym === nothing && return nothing
    idx_assembly = idx_assembly::Int
    idx_oper = idx_oper::Int
    idx_asym = idx_asym::Int

    target = strip(String(assembly_id))
    isempty(target) && return nothing
    target_all = lowercase(target) == "all"
    label_to_auth = _mmcif_label_to_auth_map(path)
    found = false
    out = Dict{String, Int}()
    for row in rows
        idx_assembly <= length(row) || continue
        assembly = strip(String(row[idx_assembly]))
        _is_cif_missing(assembly) && continue
        if !target_all && assembly != target
            continue
        end
        found = true
        op_expr = idx_oper <= length(row) ? String(row[idx_oper]) : "1"
        asym_expr = idx_asym <= length(row) ? String(row[idx_asym]) : ""
        op_count = max(_operation_expression_count(op_expr), 1)
        for asym_raw in split(asym_expr, ',')
            asym = strip(asym_raw)
            _is_cif_missing(asym) && continue
            auth = get(label_to_auth, asym, asym)
            _is_cif_missing(auth) && continue
            out[auth] = get(out, auth, 0) + op_count
        end
    end
    if !target_all && !found
        error("File has no Assembly ID '$target'")
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

function _has_weight_prefix(weights::AbstractDict{<:AbstractString, <:Any}, prefix::String)
    for key_any in keys(weights)
        startswith(String(key_any), prefix) && return true
    end
    return false
end

function _infer_substructure_transformer_layers(weights::AbstractDict{<:AbstractString, <:Any}, prefix::String)
    max_idx = -1
    p = prefix * ".transformer.layers."
    for key_any in keys(weights)
        key = String(key_any)
        startswith(key, p) || continue
        parts = split(key, '.')
        pos = findfirst(==("layers"), parts)
        if pos !== nothing && pos < length(parts)
            idx = tryparse(Int, parts[pos + 1])
            idx !== nothing && (max_idx = max(max_idx, idx))
        end
    end
    return max_idx + 1
end

function _infer_constraint_substructure_config(weights::AbstractDict{<:AbstractString, <:Any}, prefix::String)
    sub_prefix = "$prefix.substructure_z_embedder"
    if _has_weight_prefix(weights, "$sub_prefix.transformer.layers.")
        hidden_dim = haskey(weights, "$sub_prefix.input_proj.weight") ? size(weights["$sub_prefix.input_proj.weight"], 1) : 128
        n_layers = max(_infer_substructure_transformer_layers(weights, sub_prefix), 1)
        return (architecture = :transformer, hidden_dim = hidden_dim, n_layers = n_layers, n_heads = 4)
    end

    if _has_weight_prefix(weights, "$sub_prefix.network.")
        hidden_dim = haskey(weights, "$sub_prefix.network.0.weight") ? size(weights["$sub_prefix.network.0.weight"], 1) : 256
        linear_count = 0
        for key_any in keys(weights)
            key = String(key_any)
            startswith(key, "$sub_prefix.network.") || continue
            endswith(key, ".weight") || continue
            linear_count += 1
        end
        n_layers = max(linear_count, 1)
        return (architecture = :mlp, hidden_dim = hidden_dim, n_layers = n_layers, n_heads = 4)
    end

    return (architecture = :transformer, hidden_dim = 128, n_layers = 1, n_heads = 4)
end

function _load_model(model_name::AbstractString, weights_path::AbstractString; strict::Bool = true)
    params = recommended_params(model_name)

    w = load_safetensors_weights(weights_path)
    if params.family == :mini
        m = ProtenixMini.build_protenix_mini_model(w; esm_enable = params.needs_esm_embedding)
        ProtenixMini.load_protenix_mini_model!(m, w; strict = strict)
        return (model = m, family = :mini)
    elseif params.family == :base
        constraint_enable = occursin("constraint", lowercase(String(model_name)))
        if constraint_enable
            sub_cfg = _infer_constraint_substructure_config(w, "constraint_embedder")
            m = ProtenixBase.build_protenix_base_model(
                w;
                constraint_enable = true,
                constraint_substructure_enable = true,
                constraint_substructure_architecture = sub_cfg.architecture,
                constraint_substructure_hidden_dim = sub_cfg.hidden_dim,
                constraint_substructure_n_layers = sub_cfg.n_layers,
                constraint_substructure_n_heads = sub_cfg.n_heads,
            )
        else
            m = ProtenixBase.build_protenix_base_model(w; constraint_enable = false)
        end
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

function _constraint_to_device(cf::Nothing, ref)
    return nothing
end

function _constraint_to_device(cf, ref)
    return ProtenixMini.Features.ConstraintFeatures(
        cf.contact === nothing ? nothing : copyto!(similar(ref, Float32, size(cf.contact)), cf.contact),
        cf.pocket === nothing ? nothing : copyto!(similar(ref, Float32, size(cf.pocket)), cf.pocket),
        cf.contact_atom === nothing ? nothing : copyto!(similar(ref, Float32, size(cf.contact_atom)), cf.contact_atom),
        cf.substructure === nothing ? nothing : copyto!(similar(ref, Float32, size(cf.substructure)), cf.substructure),
    )
end

function _features_to_device(feat::ProtenixMini.ProtenixFeatures, ref::AbstractArray)
    return ProtenixMini.features_to_device(feat, ref)
end

function _pred_to_cpu(pred)
    return (
        coordinate = Array(pred.coordinate),
        s_inputs = Array(pred.s_inputs),
        s_trunk = Array(pred.s_trunk),
        z_trunk = Array(pred.z_trunk),
        distogram_logits = Array(pred.distogram_logits),
        plddt = Array(pred.plddt),
        pae = Array(pred.pae),
        pde = Array(pred.pde),
        resolved = Array(pred.resolved),
    )
end

function _softmax(v::AbstractVector{<:Real})
    m = maximum(v)
    ex = exp.(Float64.(v) .- Float64(m))
    s = sum(ex)
    return ex ./ s
end

function _confidence_proxy(logits::AbstractArray{<:Real, 2})
    # Features-first: logits (bins, N_item). Softmax over dim=1 (bins), average over dim=2 (items).
    size(logits, 2) > 0 || return 0.0
    acc = 0.0
    for i in 1:size(logits, 2)
        p = _softmax(@view logits[:, i])
        acc += maximum(p)
    end
    return acc / size(logits, 2)
end

function _write_confidence_summaries(
    pred_dir::AbstractString,
    task_name::AbstractString,
    seed::Int,
    pred,
)
    # Features-first: coordinate (3, N_atom, N_sample), plddt (b, N_atom, N_sample), etc.
    n_sample = size(pred.coordinate, 3)
    for sample_idx in 1:n_sample
        plddt_i = Array{Float32, 2}(pred.plddt[:, :, sample_idx])
        pae_i = Array{Float32, 3}(pred.pae[:, :, :, sample_idx])
        pde_i = Array{Float32, 3}(pred.pde[:, :, :, sample_idx])
        resolved_i = Array{Float32, 2}(pred.resolved[:, :, sample_idx])

        summary = (
            model_output = "julia_protenix",
            sample_name = String(task_name),
            seed = Int(seed),
            sample_idx = sample_idx - 1,
            plddt_logits_shape = [size(plddt_i, 1), size(plddt_i, 2)],
            pae_logits_shape = [size(pae_i, 1), size(pae_i, 2), size(pae_i, 3)],
            pde_logits_shape = [size(pde_i, 1), size(pde_i, 2), size(pde_i, 3)],
            resolved_logits_shape = [size(resolved_i, 1), size(resolved_i, 2)],
            plddt_logits_maxprob_mean = _confidence_proxy(plddt_i),
            resolved_logits_maxprob_mean = _confidence_proxy(resolved_i),
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
    resolved_name = _resolve_model_alias(opts.model_name)
    params = recommended_params(
        resolved_name;
        use_default_params = opts.use_default_params,
        cycle = opts.cycle,
        step = opts.step,
        sample = opts.sample,
        use_msa = opts.use_msa,
    )
    mkpath(opts.out_dir)
    weights_ref = if isempty(opts.weights_path)
        default_weights_path(resolved_name)
    else
        String(opts.weights_path)
    end
    loaded = _load_model(resolved_name, weights_ref; strict = opts.strict)
    if opts.gpu
        loaded = (model = gpu(loaded.model), family = loaded.family)
    end
    return (params = params, loaded = loaded, gpu = opts.gpu)
end

function predict_json(input::AbstractString, opts::ProtenixPredictOptions)
    runtime = _resolve_predict_runtime(opts)
    params = runtime.params
    loaded = runtime.loaded
    use_gpu = runtime.gpu
    json_paths = _collect_input_paths(input; exts = (".json",))
    records = PredictJSONRecord[]

    for json_path in json_paths
        tasks = _ensure_json_tasks(json_path)
        for (task_idx, task_any) in enumerate(tasks)
            task_any isa AbstractDict || error("Task $(task_idx) in $json_path is not an object")
            task = _as_string_dict(task_any)
            task_name = haskey(task, "name") ? String(task["name"]) : "$(_default_task_name(json_path))_$(task_idx - 1)"
            parsed_task = _parse_task_entities(task; json_dir = dirname(abspath(json_path)))
            chain_sequences = _protein_chain_sequence_map(parsed_task.protein_specs)

            for seed in opts.seeds
                rng = MersenneTwister(seed)
                atoms = _remove_covalent_leaving_atoms(
                    parsed_task.atoms, task, parsed_task.entity_chain_ids, parsed_task.entity_atom_map;
                    rng = rng,
                )
                atoms = _apply_mse_to_met(atoms)
                atoms = _apply_ccd_mol_type_override(atoms, parsed_task.polymer_chain_ids;
                    all_entities = _is_v1_model(opts.model_name))
                bundle = build_feature_bundle_from_atoms(atoms; task_name = task_name, rng = rng)
                token_chain_ids = [bundle["atoms"][tok.centre_atom_index].chain_id for tok in bundle["tokens"]]
                _normalize_protenix_feature_dict!(bundle["input_feature_dict"])
                _fix_restype_for_modified_residues!(bundle["input_feature_dict"], bundle["atoms"], bundle["tokens"])
                _fix_entity_and_sym_ids!(bundle["input_feature_dict"], bundle["atoms"], bundle["tokens"], parsed_task.entity_chain_ids)
                _inject_task_msa_features!(
                    bundle["input_feature_dict"],
                    task,
                    json_path;
                    use_msa = params.use_msa,
                    msa_pair_as_unpair = params.msa_pair_as_unpair,
                    chain_specs = parsed_task.protein_specs,
                    rna_chain_specs = parsed_task.rna_specs,
                    dna_chain_specs = parsed_task.dna_specs,
                    token_chain_ids = token_chain_ids,
                    ion_chain_ids = parsed_task.ion_chain_ids,
                )
                _inject_task_covalent_token_bonds!(
                    bundle["input_feature_dict"],
                    bundle["atoms"],
                    task,
                    parsed_task.entity_chain_ids,
                    parsed_task.entity_atom_map,
                )
                _inject_task_template_features!(bundle["input_feature_dict"], task)
                # For v1.0 models, always ensure template derived features exist
                # (Python v1.0.4 always runs the template assembly pipeline, even with dummy data)
                if _is_v1_model(opts.model_name)
                    _inject_dummy_template_features!(bundle["input_feature_dict"])
                end
                _inject_task_esm_token_embedding!(bundle["input_feature_dict"], task)
                _inject_auto_esm_token_embedding!(
                    bundle["input_feature_dict"],
                    bundle["atoms"],
                    bundle["tokens"],
                    chain_sequences,
                    params,
                    "task '$task_name' in $(basename(json_path))",
                )
                _inject_task_constraint_feature!(
                    bundle["input_feature_dict"],
                    task,
                    bundle["atoms"],
                    parsed_task.entity_chain_ids,
                    parsed_task.entity_atom_map,
                    "task '$task_name' in $(basename(json_path))",
                )
                _validate_required_model_inputs!(
                    params,
                    bundle["input_feature_dict"],
                    "task '$task_name' in $(basename(json_path))",
                )
                typed_feat = ProtenixMini.as_protenix_features(bundle["input_feature_dict"])
                if use_gpu
                    ref = device_ref(loaded.model)
                    typed_feat = _features_to_device(typed_feat, ref)
                end
                pred = _run_model(
                    loaded,
                    typed_feat;
                    cycle = params.cycle,
                    step = params.step,
                    sample = params.sample,
                    rng = rng,
                )
                if use_gpu
                    pred = _pred_to_cpu(pred)
                end

                task_dump_dir = joinpath(opts.out_dir, task_name, "seed_$(seed)")
                cross_bonds = get(bundle["input_feature_dict"], "_cif_cross_chain_bonds", nothing)
                pred_dir = dump_prediction_bundle(task_dump_dir, task_name, bundle["atoms"], pred.coordinate; cross_chain_bonds=cross_bonds)
                _write_confidence_summaries(pred_dir, task_name, seed, pred)
                cif_paths = sort(filter(endswith(".cif"), readdir(pred_dir; join = true)))

                push!(
                    records,
                    (
                        input_json = String(json_path),
                        task_name = String(task_name),
                        seed = Int(seed),
                        prediction_dir = String(pred_dir),
                        cif_paths = cif_paths,
                    ),
                )
            end
        end
    end

    return records
end

"""
    predict_json(input; out_dir="./output", model_name="protenix_base_default_v0.5.0",
                 seeds=[101], gpu=false, cycle=nothing, step=nothing, sample=nothing,
                 use_msa=nothing, strict=true) → Vector{PredictJSONRecord}

Run structure prediction on one or more JSON input files. `input` can be a path to a
single JSON file or a directory containing JSON files. Each task × seed combination
produces a separate prediction.

Returns a vector of `PredictJSONRecord` named tuples, each containing:
`(input_json, task_name, seed, prediction_dir, cif_paths)`.

# Example

```julia
records = predict_json("inputs/complex.json"; model_name="protenix_base_default_v0.5.0",
                       out_dir="./output", seeds=[101, 102], gpu=true)
```
"""
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
    gpu::Bool = false,
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
        gpu = gpu,
    )
    return predict_json(input, opts)
end

function predict_sequence(sequence::AbstractString, opts::ProtenixSequenceOptions)
    seq = uppercase(strip(sequence))
    isempty(seq) && error("sequence must be non-empty")

    runtime = _resolve_predict_runtime(opts.common)
    params = runtime.params
    loaded = runtime.loaded
    use_gpu = runtime.gpu
    records = PredictSequenceRecord[]

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
        _inject_auto_esm_token_embedding!(
            bundle["input_feature_dict"],
            bundle["atoms"],
            bundle["tokens"],
            Dict(opts.chain_id => seq),
            params,
            "sequence task '$(opts.task_name)'",
        )
        _validate_required_model_inputs!(
            params,
            bundle["input_feature_dict"],
            "sequence task '$(opts.task_name)'",
        )
        typed_feat = ProtenixMini.as_protenix_features(bundle["input_feature_dict"])
        if use_gpu
            ref = device_ref(loaded.model)
            typed_feat = _features_to_device(typed_feat, ref)
        end
        pred = _run_model(
            loaded,
            typed_feat;
            cycle = params.cycle,
            step = params.step,
            sample = params.sample,
            rng = rng,
        )
        if use_gpu
            pred = _pred_to_cpu(pred)
        end

        task_dump_dir = joinpath(opts.common.out_dir, opts.task_name, "seed_$(seed)")
        cross_bonds = get(bundle["input_feature_dict"], "_cif_cross_chain_bonds", nothing)
        pred_dir = dump_prediction_bundle(task_dump_dir, opts.task_name, bundle["atoms"], pred.coordinate; cross_chain_bonds=cross_bonds)
        _write_confidence_summaries(pred_dir, opts.task_name, seed, pred)
        cif_paths = sort(filter(endswith(".cif"), readdir(pred_dir; join = true)))

        push!(
            records,
            (
                task_name = String(opts.task_name),
                seed = Int(seed),
                prediction_dir = String(pred_dir),
                cif_paths = cif_paths,
            ),
        )
    end

    return records
end

"""
    predict_sequence(sequence; out_dir="./output", model_name="protenix_base_default_v0.5.0",
                     seeds=[101], gpu=false, task_name="protenix_sequence", chain_id="A",
                     esm_token_embedding=nothing, cycle=nothing, step=nothing, sample=nothing,
                     use_msa=nothing, strict=true) → Vector{PredictSequenceRecord}

Run structure prediction on a single protein sequence string. Each seed produces a
separate prediction.

Returns a vector of `PredictSequenceRecord` named tuples, each containing:
`(task_name, seed, prediction_dir, cif_paths)`.

# Example

```julia
records = predict_sequence("ACDEFGHIKLMNPQRSTVWY";
                           model_name="protenix_mini_default_v0.5.0", gpu=true)
```
"""
function predict_sequence(
    sequence::AbstractString;
    out_dir::AbstractString = "./output",
    model_name::String = "protenix_base_default_v0.5.0",
    weights_path::AbstractString = "",
    task_name::String = "protenix_sequence",
    chain_id::String = "A",
    seeds::Vector{Int} = [101],
    use_default_params::Bool = true,
    cycle::Union{Nothing, Int} = nothing,
    step::Union{Nothing, Int} = nothing,
    sample::Union{Nothing, Int} = nothing,
    use_msa::Union{Nothing, Bool} = nothing,
    esm_token_embedding::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    strict::Bool = true,
    gpu::Bool = false,
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
        gpu = gpu,
    )
    seq_opts = ProtenixSequenceOptions(
        common = common,
        task_name = task_name,
        chain_id = chain_id,
        esm_token_embedding = esm_token_embedding,
    )
    return predict_sequence(sequence, seq_opts)
end

"""
    convert_structure_to_infer_json(input; out_dir="./output", altloc="first",
                                   assembly_id=nothing) → Vector{String}

Convert PDB or mmCIF structure files into Protenix inference JSON format. `input` can be
a single file path or a directory of structure files (`.pdb`, `.cif`, `.mmcif`).

When `assembly_id` is provided for mmCIF files, the corresponding bioassembly is expanded
with per-chain copy counts.

Returns paths to the written JSON files.

# Example

```julia
paths = convert_structure_to_infer_json("structures/5o45.cif"; out_dir="./json_inputs")
```
"""
function convert_structure_to_infer_json(
    input::AbstractString;
    out_dir::AbstractString = "./output",
    altloc::String = "first",
    assembly_id::Union{Nothing, String} = nothing,
)
    lowercase(strip(altloc)) == "first" || error("Only altloc='first' is currently supported.")

    mkpath(out_dir)
    paths = _collect_input_paths(input; exts = (".pdb", ".cif", ".mmcif"))
    out_paths = String[]

    for p in paths
        atoms, _ = load_structure_atoms(p)
        ext = lowercase(splitext(p)[2])
        chains = _protein_chain_sequences(atoms)
        isempty(chains) && error("No protein chains parsed from structure: $p")
        assembly_copy_counts = if assembly_id !== nothing && (ext == ".cif" || ext == ".mmcif")
            _mmcif_assembly_chain_copy_counts(p, String(assembly_id))
        else
            nothing
        end

        sequences = Any[]
        for chain in chains
            count = if assembly_copy_counts === nothing
                1
            else
                get(assembly_copy_counts, chain.chain_id, 0)
            end
            count > 0 || continue
            push!(
                sequences,
                Dict(
                    "proteinChain" => Dict(
                        "sequence" => chain.sequence,
                        "count" => count,
                    ),
                ),
            )
        end
        isempty(sequences) && error("No protein chains selected after assembly expansion for structure: $p")

        task = Dict{String, Any}(
            "name" => _default_task_name(p),
            "sequences" => sequences,
        )
        if assembly_id !== nothing
            task["assembly_id"] = String(assembly_id)
        end
        payload = Any[task]

        out_path = _next_available_json_path(out_dir, _default_task_name(p))
        write_json(out_path, payload)
        push!(out_paths, out_path)
    end

    return out_paths
end

"""
    add_precomputed_msa_to_json(input_json; out_dir="./output", precomputed_msa_dir,
                                pairing_db="uniref100") → Vector{String}

Attach a precomputed MSA directory to an existing inference JSON file. Adds
`msa.precomputed_msa_dir` and `msa.pairing_db` fields to each `proteinChain` entity
in every task.

The MSA directory should contain `non_pairing.a3m` (and `pairing.a3m` for multi-chain).
The original JSON task-container shape (object/array/tasks-wrapper) is preserved in output.

Returns paths to the written output JSON files.

# Example

```julia
paths = add_precomputed_msa_to_json("input.json";
            precomputed_msa_dir="msa/7r6r_chain1", out_dir="./with_msa")
```
"""
function add_precomputed_msa_to_json(
    input_json::AbstractString;
    out_dir::AbstractString = "./output",
    precomputed_msa_dir::AbstractString,
    pairing_db::String = "uniref100",
)
    json_path = abspath(input_json)
    isfile(json_path) || error("input_json not found: $json_path")

    payload = _load_json_task_payload(json_path)
    tasks = payload.tasks
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
    output_payload = if payload.shape == :tasks_wrapper
        root = copy(payload.root)
        root["tasks"] = tasks
        root
    elseif payload.shape == :object
        length(tasks) == 1 ? tasks[1] : tasks
    else
        tasks
    end
    write_json(out_path, output_payload)
    return out_path
end

# ──────────────────────────────────────────────────────────────────────
# REPL-friendly API: load_protenix / fold / confidence_metrics
# ──────────────────────────────────────────────────────────────────────

"""
    ProtenixHandle

Reusable model handle returned by [`load_protenix`](@ref).

Load once, predict many times:

```julia
h = load_protenix("protenix_mini_default_v0.5.0"; gpu=true)
r1 = fold(h, "MKQLLED..."; seed=42)
r2 = fold(h, "GGGGGGG..."; seed=7)
```
"""
struct ProtenixHandle
    model::Any          # ProtenixMiniModel or ProtenixBaseModel
    family::Symbol      # :mini or :base
    model_name::String
    on_gpu::Bool
    params::NamedTuple  # from recommended_params
end

function Base.show(io::IO, h::ProtenixHandle)
    print(io, "ProtenixHandle($(h.model_name), family=:$(h.family), gpu=$(h.on_gpu))")
end

"""
    load_protenix(model_name="protenix_base_default_v0.5.0"; gpu=false, strict=true) → ProtenixHandle

Load a Protenix model and return a reusable [`ProtenixHandle`](@ref). Weights are
downloaded from HuggingFace (`MurrellLab/PXDesign.jl`) on first use and cached locally.
Set `PROTENIX_WEIGHTS_LOCAL_FILES_ONLY=true` for offline mode after prefetching.

# Arguments
- `model_name::AbstractString`: one of the supported model names (see `list_supported_models()`)
- `gpu::Bool`: move model parameters to GPU after loading
- `strict::Bool`: enforce strict safetensors key coverage (recommended)

# Returns
A `ProtenixHandle` containing the loaded model, family, params, and GPU state.
Pass it to `fold()` for repeated predictions without reloading weights.

# Examples

```julia
h = load_protenix(gpu=true)
h = load_protenix("protenix_mini_default_v0.5.0"; gpu=true)
h = load_protenix("protenix_base_constraint_v0.5.0"; gpu=true)
```
"""
const _flux_gpu = gpu

# pLDDT: 50 bins from 0 to 1, score in 0-100 range
# pAE: 64 bins from 0 to 32 Angstroms
# Features-first: bins are in dim=1, spatial dims follow.
function _logits_to_score(logits::AbstractArray{<:Real}, min_bin::Real, max_bin::Real, no_bins::Int)
    bin_width = (max_bin - min_bin) / no_bins
    bin_centers = Float32[min_bin + bin_width * (i - 0.5f0) for i in 1:no_bins]
    # logits has shape (no_bins, ...). Apply softmax along dim=1 and dot with bin_centers.
    # Flatten all but first dim
    outer = prod(size(logits)[2:end])
    reshaped = reshape(Float64.(logits), no_bins, outer)
    scores = Vector{Float32}(undef, outer)
    bc = Float64.(bin_centers)
    for i in 1:outer
        col = @view reshaped[:, i]
        m = maximum(col)
        ex = exp.(col .- m)
        s = sum(ex)
        prob = ex ./ s
        scores[i] = Float32(sum(prob .* bc))
    end
    return reshape(scores, size(logits)[2:end]...)
end

function _logits_to_plddt(logits::AbstractArray{<:Real})
    # Features-first: bins in dim=1. pLDDT: bins from 0 to 1, scale to 0-100
    no_bins = size(logits, 1)
    scores = _logits_to_score(logits, 0f0, 1f0, no_bins)
    return scores .* 100f0
end

function _logits_to_pae(logits::AbstractArray{<:Real})
    # Features-first: bins in dim=1. PAE: bins from 0 to 32 Angstroms
    no_bins = size(logits, 1)
    return _logits_to_score(logits, 0f0, 32f0, no_bins)
end

function load_protenix(
    model_name::AbstractString = "protenix_base_default_v0.5.0";
    gpu::Bool = false,
    strict::Bool = true,
)
    resolved = _resolve_model_alias(model_name)
    params = recommended_params(resolved; use_default_params = true)
    weights_ref = default_weights_path(resolved)
    loaded = _load_model(resolved, weights_ref; strict = strict)
    if gpu
        loaded = (model = _flux_gpu(loaded.model), family = loaded.family)
    end
    return ProtenixHandle(loaded.model, loaded.family, resolved, gpu, params)
end

"""
    fold(handle, sequence; seed=101, step=nothing, sample=nothing, cycle=nothing,
         out_dir=nothing, task_name="protenix_sequence", chain_id="A",
         esm_token_embedding=nothing)

Fold a protein sequence using a loaded model handle. Returns a `NamedTuple` with
coordinates, CIF text, file paths, and confidence metrics for a single seed.

# Examples

```julia
h = load_protenix(gpu=true)
result = fold(h, "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH")
result.cif       # mmCIF text as String
result.plddt     # per-residue pLDDT vector
result.mean_plddt
```
"""
function fold(
    handle::ProtenixHandle,
    sequence::AbstractString;
    seed::Integer = 101,
    step::Union{Nothing, Integer} = nothing,
    sample::Union{Nothing, Integer} = nothing,
    cycle::Union{Nothing, Integer} = nothing,
    out_dir::Union{Nothing, AbstractString} = nothing,
    task_name::AbstractString = "protenix_sequence",
    chain_id::AbstractString = "A",
    esm_token_embedding::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
)
    seq = uppercase(strip(String(sequence)))
    isempty(seq) && error("sequence must be non-empty")

    p = handle.params
    run_cycle = cycle === nothing ? p.cycle : Int(cycle)
    run_step = step === nothing ? p.step : Int(step)
    run_sample = sample === nothing ? 1 : Int(sample)
    actual_out_dir = out_dir === nothing ? mktempdir() : mkpath(String(out_dir))

    rng = MersenneTwister(Int(seed))
    atoms = ProtenixMini.build_sequence_atoms(seq; chain_id = String(chain_id))
    bundle = build_feature_bundle_from_atoms(atoms; task_name = String(task_name), rng = rng)
    _normalize_protenix_feature_dict!(bundle["input_feature_dict"])

    if esm_token_embedding !== nothing
        emb = esm_token_embedding
        n_tok = size(bundle["input_feature_dict"]["restype"], 1)
        size(emb, 1) == n_tok || error(
            "esm_token_embedding token length mismatch: expected $n_tok, got $(size(emb, 1)).",
        )
        bundle["input_feature_dict"]["esm_token_embedding"] = emb
    end
    _inject_auto_esm_token_embedding!(
        bundle["input_feature_dict"],
        bundle["atoms"],
        bundle["tokens"],
        Dict(String(chain_id) => seq),
        p,
        "fold task '$(task_name)'",
    )
    _validate_required_model_inputs!(p, bundle["input_feature_dict"], "fold task '$(task_name)'")

    typed_feat = ProtenixMini.as_protenix_features(bundle["input_feature_dict"])
    if handle.on_gpu
        ref = device_ref(handle.model)
        typed_feat = _features_to_device(typed_feat, ref)
    end

    pred = _run_model(
        (model = handle.model, family = handle.family),
        typed_feat;
        cycle = run_cycle,
        step = run_step,
        sample = run_sample,
        rng = rng,
    )
    if handle.on_gpu
        pred = _pred_to_cpu(pred)
    end

    task_dump_dir = joinpath(actual_out_dir, String(task_name), "seed_$(seed)")
    cross_bonds = get(bundle["input_feature_dict"], "_cif_cross_chain_bonds", nothing)
    pred_dir = dump_prediction_bundle(task_dump_dir, String(task_name), bundle["atoms"], pred.coordinate; cross_chain_bonds=cross_bonds)
    _write_confidence_summaries(pred_dir, String(task_name), Int(seed), pred)

    cif_paths = sort(filter(endswith(".cif"), readdir(pred_dir; join = true)))
    cif_text = isempty(cif_paths) ? "" : read(first(cif_paths), String)

    # Convert logits to proper pLDDT scores (0-100 scale)
    plddt_scores = _logits_to_plddt(pred.plddt)
    pae_scores = _logits_to_pae(pred.pae)
    mean_plddt = Float32(Statistics.mean(plddt_scores))
    mean_pae = Float32(Statistics.mean(pae_scores))

    return (
        coordinate = pred.coordinate,
        cif = cif_text,
        cif_paths = cif_paths,
        prediction_dir = pred_dir,
        plddt = plddt_scores,
        mean_plddt = mean_plddt,
        pae = pae_scores,
        mean_pae = mean_pae,
        pde = pred.pde,
        resolved = pred.resolved,
        distogram_logits = pred.distogram_logits,
        plddt_logits = pred.plddt,
        pae_logits = pred.pae,
        seed = Int(seed),
        task_name = String(task_name),
    )
end

"""
    fold(handle::ProtenixHandle, task::AbstractDict; seed=101, step=nothing,
         sample=nothing, cycle=nothing, out_dir=nothing) → NamedTuple

Fold a multi-entity task from a Dict. The `task` Dict must follow the Protenix
JSON schema (with `"sequences"` array). Use the builder helpers [`protenix_task`](@ref),
[`protein_chain`](@ref), [`rna_chain`](@ref), [`dna_chain`](@ref), [`ligand`](@ref),
and [`ion`](@ref) to construct it conveniently.

Returns the same rich NamedTuple as the single-sequence `fold` method.

# Examples

```julia
h = load_protenix("protenix_base_default_v0.5.0"; gpu=true)

# Heterodimer
task = protenix_task(
    protein_chain("MVLSPAD..."),
    protein_chain("MVHLTPE..."),
    name = "heterodimer",
)
r = fold(h, task; seed=101)
r.mean_plddt

# Protein + ligand
task = protenix_task(protein_chain("MVLSPAD..."), ligand("CCD_ATP"))
r = fold(h, task)

# Power user — raw Dict
task = Dict(
    "name" => "manual",
    "sequences" => [
        Dict("proteinChain" => Dict("sequence" => "MVLSPAD...", "count" => 1)),
    ],
)
r = fold(h, task)
```
"""
function fold(
    handle::ProtenixHandle,
    task::AbstractDict;
    seed::Integer = 101,
    step::Union{Nothing, Integer} = nothing,
    sample::Union{Nothing, Integer} = nothing,
    cycle::Union{Nothing, Integer} = nothing,
    out_dir::Union{Nothing, AbstractString} = nothing,
)
    task = _as_string_dict(task)
    task_name = get(task, "name", "repl_task")

    p = handle.params
    run_cycle = cycle === nothing ? p.cycle : Int(cycle)
    run_step = step === nothing ? p.step : Int(step)
    run_sample = sample === nothing ? p.sample : Int(sample)
    actual_out_dir = out_dir === nothing ? mktempdir() : mkpath(String(out_dir))

    json_dir = pwd()
    parsed_task = _parse_task_entities(task; json_dir = json_dir)
    chain_sequences = _protein_chain_sequence_map(parsed_task.protein_specs)

    rng = MersenneTwister(Int(seed))
    atoms = _remove_covalent_leaving_atoms(
        parsed_task.atoms, task, parsed_task.entity_chain_ids, parsed_task.entity_atom_map;
        rng = rng,
    )
    atoms = _apply_mse_to_met(atoms)
    atoms = _apply_ccd_mol_type_override(atoms, parsed_task.polymer_chain_ids;
        all_entities = _is_v1_model(handle.model_name))
    bundle = build_feature_bundle_from_atoms(atoms; task_name = task_name, rng = rng)
    token_chain_ids = [bundle["atoms"][tok.centre_atom_index].chain_id for tok in bundle["tokens"]]
    _normalize_protenix_feature_dict!(bundle["input_feature_dict"])
    _fix_restype_for_modified_residues!(bundle["input_feature_dict"], bundle["atoms"], bundle["tokens"])
    _fix_entity_and_sym_ids!(bundle["input_feature_dict"], bundle["atoms"], bundle["tokens"], parsed_task.entity_chain_ids)

    json_path = joinpath(json_dir, "_repl_.json")
    _inject_task_msa_features!(
        bundle["input_feature_dict"],
        task,
        json_path;
        use_msa = p.use_msa,
        msa_pair_as_unpair = p.msa_pair_as_unpair,
        chain_specs = parsed_task.protein_specs,
        rna_chain_specs = parsed_task.rna_specs,
        dna_chain_specs = parsed_task.dna_specs,
        token_chain_ids = token_chain_ids,
        ion_chain_ids = parsed_task.ion_chain_ids,
    )
    _inject_task_covalent_token_bonds!(
        bundle["input_feature_dict"],
        bundle["atoms"],
        task,
        parsed_task.entity_chain_ids,
        parsed_task.entity_atom_map,
    )
    _inject_task_template_features!(bundle["input_feature_dict"], task)
    if _is_v1_model(handle.model_name)
        _inject_dummy_template_features!(bundle["input_feature_dict"])
    end
    _inject_task_esm_token_embedding!(bundle["input_feature_dict"], task)
    _inject_auto_esm_token_embedding!(
        bundle["input_feature_dict"],
        bundle["atoms"],
        bundle["tokens"],
        chain_sequences,
        p,
        "task '$task_name' (REPL)",
    )
    _inject_task_constraint_feature!(
        bundle["input_feature_dict"],
        task,
        bundle["atoms"],
        parsed_task.entity_chain_ids,
        parsed_task.entity_atom_map,
        "task '$task_name' (REPL)",
    )
    _validate_required_model_inputs!(p, bundle["input_feature_dict"], "task '$task_name' (REPL)")

    typed_feat = ProtenixMini.as_protenix_features(bundle["input_feature_dict"])
    if handle.on_gpu
        ref = device_ref(handle.model)
        typed_feat = _features_to_device(typed_feat, ref)
    end

    pred = _run_model(
        (model = handle.model, family = handle.family),
        typed_feat;
        cycle = run_cycle,
        step = run_step,
        sample = run_sample,
        rng = rng,
    )
    if handle.on_gpu
        pred = _pred_to_cpu(pred)
    end

    task_dump_dir = joinpath(actual_out_dir, task_name, "seed_$(seed)")
    cross_bonds = get(bundle["input_feature_dict"], "_cif_cross_chain_bonds", nothing)
    pred_dir = dump_prediction_bundle(task_dump_dir, task_name, bundle["atoms"], pred.coordinate; cross_chain_bonds=cross_bonds)
    _write_confidence_summaries(pred_dir, task_name, Int(seed), pred)

    cif_paths = sort(filter(endswith(".cif"), readdir(pred_dir; join = true)))
    cif_text = isempty(cif_paths) ? "" : read(first(cif_paths), String)

    plddt_scores = _logits_to_plddt(pred.plddt)
    pae_scores = _logits_to_pae(pred.pae)
    mean_plddt = Float32(Statistics.mean(plddt_scores))
    mean_pae = Float32(Statistics.mean(pae_scores))

    return (
        coordinate = pred.coordinate,
        cif = cif_text,
        cif_paths = cif_paths,
        prediction_dir = pred_dir,
        plddt = plddt_scores,
        mean_plddt = mean_plddt,
        pae = pae_scores,
        mean_pae = mean_pae,
        pde = pred.pde,
        resolved = pred.resolved,
        distogram_logits = pred.distogram_logits,
        plddt_logits = pred.plddt,
        pae_logits = pred.pae,
        seed = Int(seed),
        task_name = task_name,
    )
end

"""
    confidence_metrics(result) → NamedTuple

Extract confidence metrics from a fold result.

```julia
h = load_protenix(gpu=true)
result = fold(h, "MKQLLED...")
m = confidence_metrics(result)
m.mean_plddt  # average predicted local distance difference test
m.mean_pae    # average predicted aligned error
```
"""
function confidence_metrics(result::NamedTuple)
    return (
        plddt = result.plddt,
        mean_plddt = result.mean_plddt,
        pae = result.pae,
        mean_pae = result.mean_pae,
        pde = result.pde,
        resolved = result.resolved,
    )
end

# ── Builder helpers ──────────────────────────────────────────────────────────
# Pure functions that construct entity Dicts matching the Protenix JSON schema.
# These allow users to build task inputs from the REPL without writing JSON files.

"""
    protein_chain(sequence; count=1, msa=nothing, modifications=nothing) → Dict

Build a protein chain entity Dict for use with [`protenix_task`](@ref).

```julia
protein_chain("MVLSPAD...")
protein_chain("MVLSPAD..."; count=2)  # homodimer
```
"""
function protein_chain(
    sequence::AbstractString;
    count::Integer = 1,
    msa::Union{Nothing, AbstractDict, AbstractVector} = nothing,
    modifications::Union{Nothing, AbstractVector} = nothing,
)
    count > 0 || error("count must be positive")
    pc = Dict{String, Any}("sequence" => String(sequence), "count" => Int(count))
    msa !== nothing && (pc["msa"] = msa)
    modifications !== nothing && (pc["modifications"] = modifications)
    return Dict{String, Any}("proteinChain" => pc)
end

"""
    rna_chain(sequence; count=1, unpaired_msa=nothing, unpaired_msa_path=nothing, modifications=nothing) → Dict

Build an RNA chain entity Dict for use with [`protenix_task`](@ref).

```julia
rna_chain("AUGCAUGC")
```
"""
function rna_chain(
    sequence::AbstractString;
    count::Integer = 1,
    unpaired_msa::Union{Nothing, AbstractString} = nothing,
    unpaired_msa_path::Union{Nothing, AbstractString} = nothing,
    modifications::Union{Nothing, AbstractVector} = nothing,
)
    count > 0 || error("count must be positive")
    rc = Dict{String, Any}("sequence" => String(sequence), "count" => Int(count))
    unpaired_msa !== nothing && (rc["unpairedMsa"] = String(unpaired_msa))
    unpaired_msa_path !== nothing && (rc["unpairedMsaPath"] = String(unpaired_msa_path))
    modifications !== nothing && (rc["modifications"] = modifications)
    return Dict{String, Any}("rnaSequence" => rc)
end

"""
    dna_chain(sequence; count=1, modifications=nothing) → Dict

Build a DNA chain entity Dict for use with [`protenix_task`](@ref).

```julia
dna_chain("ATCGATCG")
```
"""
function dna_chain(
    sequence::AbstractString;
    count::Integer = 1,
    modifications::Union{Nothing, AbstractVector} = nothing,
)
    count > 0 || error("count must be positive")
    dc = Dict{String, Any}("sequence" => String(sequence), "count" => Int(count))
    modifications !== nothing && (dc["modifications"] = modifications)
    return Dict{String, Any}("dnaSequence" => dc)
end

"""
    ligand(name; count=1) → Dict

Build a ligand entity Dict for use with [`protenix_task`](@ref).
`name` follows the Protenix convention: `"CCD_ATP"`, `"SMILES_..."`, etc.

```julia
ligand("CCD_ATP")
ligand("CCD_HEM"; count=2)
```
"""
function ligand(
    name::AbstractString;
    count::Integer = 1,
)
    count > 0 || error("count must be positive")
    return Dict{String, Any}("ligand" => Dict{String, Any}("ligand" => String(name), "count" => Int(count)))
end

"""
    ion(name; count=1) → Dict

Build an ion entity Dict for use with [`protenix_task`](@ref).
`name` follows the Protenix convention, e.g. `"CCD_MG"` or `"MG"`.

```julia
ion("CCD_MG"; count=2)
```
"""
function ion(
    name::AbstractString;
    count::Integer = 1,
)
    count > 0 || error("count must be positive")
    return Dict{String, Any}("ion" => Dict{String, Any}("ion" => String(name), "count" => Int(count)))
end

"""
    protenix_task(entities...; name, constraint, covalent_bonds, template_features) → Dict

Assemble entity Dicts (from [`protein_chain`](@ref), [`rna_chain`](@ref),
[`dna_chain`](@ref), [`ligand`](@ref), [`ion`](@ref)) into a task Dict
suitable for `fold(handle, task)`.

`covalent_bonds` is an optional vector of bond specification Dicts, each with
keys `entity1`, `position1`, `atom1`, `entity2`, `position2`, `atom2`.

`template_features` is an optional Dict from [`template_structure`](@ref) for
models that support template conditioning (e.g. v1.0 models).

```julia
task = protenix_task(
    protein_chain("MVLSPAD..."),
    protein_chain("MVHLTPE..."),
    name = "heterodimer",
)
r = fold(h, task)

# With template structure
tmpl = template_structure("template.cif", "A")
task = protenix_task(protein_chain("MVLSPAD..."), template_features=tmpl)
```
"""
function protenix_task(
    entities::AbstractDict...;
    name::Union{Nothing, AbstractString} = nothing,
    constraint::Union{Nothing, AbstractDict, AbstractVector} = nothing,
    covalent_bonds::Union{Nothing, AbstractVector} = nothing,
    template_features::Union{Nothing, AbstractDict} = nothing,
)
    isempty(entities) && error("At least one entity is required")
    task = Dict{String, Any}("sequences" => collect(Any, entities))
    name !== nothing && (task["name"] = String(name))
    constraint !== nothing && (task["constraint"] = constraint)
    covalent_bonds !== nothing && (task["covalent_bonds"] = covalent_bonds)
    template_features !== nothing && (task["template_features"] = template_features)
    return task
end

# ── Design REPL API ──────────────────────────────────────────────────────────

"""
    PXDesignHandle

Holds a loaded PXDesign diffusion model and optional design condition embedder.
Created by [`load_pxdesign`](@ref), passed to [`design`](@ref).
"""
struct PXDesignHandle
    model::Any                          # DiffusionModule
    design_condition_embedder::Any      # Union{Nothing, DesignConditionEmbedder}
    model_name::String
    on_gpu::Bool
    dims::NamedTuple                    # c_token, c_s, c_z, c_s_inputs, c_atom, c_atompair, ...
    default_n_step::Int
    default_n_sample::Int
end

function Base.show(io::IO, h::PXDesignHandle)
    dce = h.design_condition_embedder !== nothing ? "yes" : "no"
    print(io, "PXDesignHandle($(h.model_name), gpu=$(h.on_gpu), dce=$dce)")
end

"""
    load_pxdesign(model_name="pxdesign_v0.1.0"; gpu=false, strict=true) → PXDesignHandle

Load a PXDesign diffusion model and return a reusable handle. Weights are
downloaded from HuggingFace on first use and cached locally.

# Examples

```julia
dh = load_pxdesign(gpu=true)
r = design(dh; binder_length=60, seed=42)
```
"""
function load_pxdesign(
    model_name::AbstractString = "pxdesign_v0.1.0";
    gpu::Bool = false,
    strict::Bool = true,
)
    model_name = _resolve_model_alias(model_name)
    weights_ref = download_model_weights(model_name)
    weights = load_safetensors_weights(weights_ref)

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

    typed_model = DiffusionModule(
        c_token, c_s, c_z, c_s_inputs;
        c_atom = c_atom,
        c_atompair = c_atompair,
        atom_encoder_blocks = atom_encoder_blocks,
        atom_encoder_heads = atom_encoder_heads,
        n_blocks = n_blocks,
        n_heads = n_heads,
        atom_decoder_blocks = atom_decoder_blocks,
        atom_decoder_heads = atom_decoder_heads,
        rng = MersenneTwister(42),
    )
    load_diffusion_module!(typed_model, weights; strict = strict)

    # Detect and build DesignConditionEmbedder if keys are present
    design_condition_embedder = nothing
    if haskey(weights, "design_condition_embedder.input_embedder.input_map.weight") &&
       haskey(weights, "design_condition_embedder.condition_template_embedder.embedder.weight")
        inferred_design = infer_design_condition_embedder_dims(weights)
        inferred_design.c_s_inputs == c_s_inputs ||
            error("DesignConditionEmbedder c_s_inputs mismatch: diffusion inferred $c_s_inputs, design inferred $(inferred_design.c_s_inputs)")
        inferred_design.c_z == c_z ||
            error("DesignConditionEmbedder c_z mismatch: diffusion inferred $c_z, design inferred $(inferred_design.c_z)")
        design_condition_embedder = DesignConditionEmbedder(
            inferred_design.c_token;
            c_s_inputs = c_s_inputs,
            c_z = c_z,
            c_atom = c_atom,
            c_atompair = c_atompair,
            n_blocks = inferred_design.n_blocks,
            n_heads = inferred_design.n_heads,
            rng = MersenneTwister(73),
        )
        load_design_condition_embedder!(design_condition_embedder, weights; strict = strict)
    end

    if strict
        report = checkpoint_coverage_report(typed_model, design_condition_embedder, weights)
        if !isempty(report.missing) || !isempty(report.unused)
            error(
                "Checkpoint key coverage mismatch: missing=$(length(report.missing)) unused=$(length(report.unused)). " *
                "Set strict=false to allow partial loads.",
            )
        end
    end

    # GPU: move diffusion model to GPU; DCE stays on CPU (matches infer.jl pattern)
    if gpu
        typed_model = _flux_gpu(typed_model)
    end

    dims_nt = (
        c_token = c_token, c_s = c_s, c_z = c_z, c_s_inputs = c_s_inputs,
        c_atom = c_atom, c_atompair = c_atompair,
        n_blocks = n_blocks, n_heads = n_heads,
        atom_encoder_blocks = atom_encoder_blocks, atom_encoder_heads = atom_encoder_heads,
        atom_decoder_blocks = atom_decoder_blocks, atom_decoder_heads = atom_decoder_heads,
    )

    return PXDesignHandle(typed_model, design_condition_embedder, String(model_name), gpu, dims_nt, 200, 5)
end

"""
    design_target(structure_file; chains=String[], crop=Dict(), msa=Dict()) → NamedTuple

Build a target specification for conditional design. Returns an intermediate
that can be passed to [`design_task`](@ref).

# Examples

```julia
target = design_target("structures/1ubq.cif"; chains=["A"])
target = design_target("structures/5o45.cif";
    chains=["A"],
    crop=Dict("A"=>"1-116"),
    msa=Dict("A"=>Dict("precomputed_msa_dir"=>"/path/to/msa")),
)
```
"""
function design_target(
    structure_file::AbstractString;
    chains::AbstractVector{<:AbstractString} = String[],
    crop::Union{Nothing, AbstractDict} = nothing,
    msa::Union{Nothing, AbstractDict} = nothing,
)
    return (
        structure_file = String(structure_file),
        chain_ids = String.(chains),
        crop = crop === nothing ? Dict{String, String}() : Dict{String, String}(String(k) => String(v) for (k, v) in crop),
        msa = msa === nothing ? Dict{String, Any}() : Dict{String, Any}(String(k) => v for (k, v) in msa),
    )
end

"""
    design_task(; binder_length, target=nothing, hotspots=nothing, name=nothing) → InputTask

Construct a `Schema.InputTask` for the design pipeline.

# Examples

```julia
# Unconditional design
task = design_task(binder_length=60)

# Conditional design with target + hotspots
target = design_target("structures/1ubq.cif"; chains=["A"])
task = design_task(binder_length=80; target=target, hotspots=Dict("A"=>[8,44,48]))
```
"""
function design_task(;
    binder_length::Integer,
    target = nothing,
    hotspots::Union{Nothing, AbstractDict} = nothing,
    name::Union{Nothing, AbstractString} = nothing,
)
    binder_length > 0 || error("binder_length must be positive")

    task_name = name === nothing ? "design_L$(binder_length)" : String(name)

    structure_file = ""
    chain_ids = String[]
    crop = Dict{String, String}()
    msa_opts = Dict{String, MSAChainOptions}()

    if target !== nothing
        structure_file = target.structure_file
        chain_ids = target.chain_ids
        crop = target.crop
        # Parse MSA options from target's Dict format into MSAChainOptions
        raw_msa = target.msa
        for (chain, cfg) in raw_msa
            cfg_dict = cfg isa AbstractDict ? cfg : Dict{String, Any}()
            precomputed = haskey(cfg_dict, "precomputed_msa_dir") ? String(cfg_dict["precomputed_msa_dir"]) : nothing
            pairing = haskey(cfg_dict, "pairing_db") ? String(cfg_dict["pairing_db"]) : nothing
            extra = Dict{String, String}()
            for (k, v) in cfg_dict
                ks = String(k)
                (ks == "precomputed_msa_dir" || ks == "pairing_db") && continue
                extra[ks] = String(v)
            end
            msa_opts[String(chain)] = (precomputed_msa_dir = precomputed, pairing_db = pairing, extra = extra)
        end
    end

    hs = Dict{String, Vector{Int}}()
    if hotspots !== nothing
        for (k, v) in hotspots
            hs[String(k)] = Int.(v)
        end
    end

    generation = [GenerationSpec("protein", Int(binder_length), 1)]

    return InputTask(task_name, structure_file, chain_ids, crop, hs, msa_opts, generation)
end

# ── Private helpers for design inference ────────────────────────────────────

function _design_to_matrix_f32(x)
    x isa AbstractMatrix || error("Expected matrix feature, got $(typeof(x))")
    return Float32.(x)
end

function _design_pad_or_truncate_columns(x::AbstractMatrix{<:Real}, width::Int)
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

function _build_design_model_inputs(
    feat::Dict{String, Any},
    c_s_inputs::Int,
    c_s::Int,
    c_z::Int;
    design_condition_embedder = nothing,
)
    relpos_input = as_relpos_input(feat)
    n_token = length(relpos_input.token_index)

    # Features-first convention: s_trunk (c_s, n_token), s_inputs (c_s_inputs, n_token), z_trunk (c_z, n_token, n_token)
    s_trunk = zeros(Float32, c_s, n_token)
    if design_condition_embedder === nothing
        restype = _design_to_matrix_f32(feat["restype"])
        profile = _design_to_matrix_f32(feat["profile"])
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
        s_inputs_fl = _design_pad_or_truncate_columns(token_features, c_s_inputs)
        s_inputs = permutedims(s_inputs_fl)  # (c_s_inputs, n_token)
        z_trunk = zeros(Float32, c_z, n_token, n_token)
        if c_z > 0
            templ_mask = _design_to_matrix_f32(feat["conditional_templ_mask"])
            z_trunk[1, :, :] .= templ_mask
        end
        if c_z > 1
            templ_bins = _design_to_matrix_f32(feat["conditional_templ"])
            z_trunk[2, :, :] .= templ_bins ./ 63f0
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

    # DesignConditionEmbedder path
    s_inputs, z_trunk = design_condition_embedder(feat)
    size(s_inputs, 2) == n_token || error("DesignConditionEmbedder returned mismatched token count.")
    size(s_inputs, 1) == c_s_inputs || error("DesignConditionEmbedder returned c_s_inputs=$(size(s_inputs, 1)) expected $c_s_inputs.")
    size(z_trunk, 2) == n_token || error("DesignConditionEmbedder returned mismatched pair token count.")
    size(z_trunk, 3) == n_token || error("DesignConditionEmbedder returned mismatched pair token count.")
    size(z_trunk, 1) == c_z || error("DesignConditionEmbedder returned c_z=$(size(z_trunk, 1)) expected $c_z.")

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

"""
    design(handle, task::InputTask; seed=42, n_step=nothing, n_sample=nothing,
           out_dir=nothing, gamma0=1.0, gamma_min=0.01, noise_scale_lambda=1.003,
           diffusion_chunk_size=0) → NamedTuple

Run design inference using a pre-loaded [`PXDesignHandle`](@ref) and an
[`InputTask`](@ref) (from [`design_task`](@ref)).

Returns a `NamedTuple` with:
- `coordinate`: raw coordinates `(3, N_atom, N_sample)`
- `cif_paths`: paths to generated CIF files
- `prediction_dir`: output directory
- `seed`, `task_name`, `n_samples`, `n_step`

# Examples

```julia
dh = load_pxdesign(gpu=true)
task = design_task(binder_length=60)
r = design(dh, task; seed=42, n_sample=5, n_step=200)
r.cif_paths
```
"""
function design(
    handle::PXDesignHandle,
    task::InputTask;
    seed::Integer = 42,
    n_step::Union{Nothing, Integer} = nothing,
    n_sample::Union{Nothing, Integer} = nothing,
    out_dir::Union{Nothing, AbstractString} = nothing,
    gamma0::Real = 1.0,
    gamma_min::Real = 0.01,
    noise_scale_lambda::Real = 1.003,
    diffusion_chunk_size::Integer = 0,
    eta_type::AbstractString = "const",
    eta_min::Real = 1.5,
    eta_max::Real = 1.5,
)
    actual_n_step = n_step === nothing ? handle.default_n_step : Int(n_step)
    actual_n_sample = n_sample === nothing ? handle.default_n_sample : Int(n_sample)
    actual_out_dir = out_dir === nothing ? mktempdir() : mkpath(String(out_dir))

    rng = MersenneTwister(Int(seed))
    feature_bundle = build_basic_feature_bundle(task; rng = rng)

    feat = feature_bundle["input_feature_dict"]
    dims = feature_bundle["dims"]
    n_atom = Int(dims["N_atom"])
    task_name = feature_bundle["task_name"]

    model_inputs = _build_design_model_inputs(
        feat,
        handle.dims.c_s_inputs,
        handle.dims.c_s,
        handle.dims.c_z;
        design_condition_embedder = handle.design_condition_embedder,
    )

    # GPU transfer of dense tensors
    dev_ref = nothing
    if handle.on_gpu
        dev_ref = device_ref(handle.model)
        _to_gpu = x::AbstractArray{Float32} -> copyto!(similar(dev_ref, Float32, size(x)...), x)
        model_inputs = (
            relpos_input = model_inputs.relpos_input,
            atom_input = model_inputs.atom_input,
            s_inputs = _to_gpu(model_inputs.s_inputs),
            s_trunk = _to_gpu(model_inputs.s_trunk),
            z_trunk = _to_gpu(model_inputs.z_trunk),
            atom_to_token_idx = model_inputs.atom_to_token_idx,
        )
    end

    # Build denoise closure
    denoise_net = function (x_noisy, t_hat; kwargs...)
        return handle.model(
            x_noisy,
            t_hat;
            relpos_input = model_inputs.relpos_input,
            s_inputs = model_inputs.s_inputs,
            s_trunk = model_inputs.s_trunk,
            z_trunk = model_inputs.z_trunk,
            atom_to_token_idx = model_inputs.atom_to_token_idx,
            input_feature_dict = model_inputs.atom_input,
        )
    end

    eta = (type = String(eta_type), min = Float64(eta_min), max = Float64(eta_max))
    scheduler = InferenceNoiseScheduler()
    noise_schedule = scheduler(actual_n_step; dtype = Float32)

    coordinates = sample_diffusion(
        denoise_net;
        noise_schedule = noise_schedule,
        N_sample = actual_n_sample,
        N_atom = n_atom,
        gamma0 = Float64(gamma0),
        gamma_min = Float64(gamma_min),
        noise_scale_lambda = Float64(noise_scale_lambda),
        step_scale_eta = eta,
        diffusion_chunk_size = Int(diffusion_chunk_size),
        rng = rng,
        device_ref = dev_ref,
    )

    # Dump CIF files
    task_dump_dir = joinpath(actual_out_dir, task_name, "seed_$(seed)")
    pred_dir = dump_prediction_bundle(task_dump_dir, task_name, feature_bundle["atoms"], coordinates)

    cif_paths = sort(filter(endswith(".cif"), readdir(pred_dir; join = true)))

    return (
        coordinate = coordinates,
        cif_paths = cif_paths,
        prediction_dir = pred_dir,
        seed = Int(seed),
        task_name = task_name,
        n_samples = actual_n_sample,
        n_step = actual_n_step,
    )
end

"""
    design(handle; binder_length, target=nothing, hotspots=nothing, name=nothing,
           seed=42, n_step=nothing, n_sample=nothing, out_dir=nothing, ...) → NamedTuple

Convenience method: builds an [`InputTask`](@ref) via [`design_task`](@ref) and
runs [`design`](@ref).

# Examples

```julia
dh = load_pxdesign(gpu=true)

# Unconditional
r = design(dh; binder_length=60, seed=42, n_sample=5, n_step=200)

# Conditional
target = design_target("structures/1ubq.cif"; chains=["A"])
r = design(dh; binder_length=80, target=target, hotspots=Dict("A"=>[8,44,48]), seed=42)
```
"""
function design(
    handle::PXDesignHandle;
    binder_length::Integer,
    target = nothing,
    hotspots::Union{Nothing, AbstractDict} = nothing,
    name::Union{Nothing, AbstractString} = nothing,
    seed::Integer = 42,
    n_step::Union{Nothing, Integer} = nothing,
    n_sample::Union{Nothing, Integer} = nothing,
    out_dir::Union{Nothing, AbstractString} = nothing,
    gamma0::Real = 1.0,
    gamma_min::Real = 0.01,
    noise_scale_lambda::Real = 1.003,
    diffusion_chunk_size::Integer = 0,
    eta_type::AbstractString = "const",
    eta_min::Real = 1.5,
    eta_max::Real = 1.5,
)
    task = design_task(;
        binder_length = binder_length,
        target = target,
        hotspots = hotspots,
        name = name,
    )
    return design(
        handle, task;
        seed = seed,
        n_step = n_step,
        n_sample = n_sample,
        out_dir = out_dir,
        gamma0 = gamma0,
        gamma_min = gamma_min,
        noise_scale_lambda = noise_scale_lambda,
        diffusion_chunk_size = diffusion_chunk_size,
        eta_type = eta_type,
        eta_min = eta_min,
        eta_max = eta_max,
    )
end

end

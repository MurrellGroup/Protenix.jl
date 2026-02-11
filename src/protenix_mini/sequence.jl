module Sequence

using Random

import ...Data: AtomRecord, build_feature_bundle_from_atoms
import ...Data.Constants: PROT_STD_RESIDUES_ONE_TO_THREE, PROTEIN_HEAVY_ATOMS
import ..Model: ProtenixMiniModel, run_inference
import ..Features: as_protenix_features

export build_sequence_atoms, build_sequence_feature_bundle, fold_sequence

const _UNK3 = "UNK"

function _aa1_to_aa3(c::Char)
    key = string(uppercase(c))
    return get(PROT_STD_RESIDUES_ONE_TO_THREE, key, _UNK3)
end

function _infer_element_from_atom_name(atom_name::AbstractString)
    for c in atom_name
        isletter(c) && return uppercase(string(c))
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

    # Deterministic pseudo-placement around CA for sidechain atoms.
    h = abs(hash(atom_name))
    dx = 0.8f0 + 0.1f0 * Float32(h % 11)
    dy = -0.9f0 + 0.3f0 * Float32((h ÷ 11) % 7)
    dz = -0.9f0 + 0.3f0 * Float32((h ÷ 77) % 7)
    return (x0 + dx, dy, dz)
end

"""
Build an all-heavy-atom protein chain from a one-letter amino-acid sequence.
"""
function build_sequence_atoms(sequence::AbstractString; chain_id::String = "A0")
    seq = uppercase(strip(sequence))
    isempty(seq) && error("sequence must be non-empty")

    atoms = AtomRecord[]
    n_res = count(c -> !isspace(c), seq)
    res_id = 0
    for c in seq
        isspace(c) && continue
        res_id += 1
        res_name = _aa1_to_aa3(c)
        atom_names = get(PROTEIN_HEAVY_ATOMS, res_name, PROTEIN_HEAVY_ATOMS[_UNK3])
        for atom_name in atom_names
            atom_name == "OXT" && res_id < n_res && continue
            element = _infer_element_from_atom_name(atom_name)
            centre = atom_name == "CA"
            x, y, z = _pseudo_atom_xyz(res_id, atom_name)
            push!(
                atoms,
                AtomRecord(
                    atom_name,
                    res_name,
                    "protein",
                    element,
                    chain_id,
                    res_id,
                    centre,
                    x,
                    y,
                    z,
                    false,
                ),
            )
        end
    end

    isempty(atoms) && error("sequence did not produce any atoms")
    return atoms
end

"""
Build a Protenix-mini compatible feature bundle directly from a sequence.
"""
function build_sequence_feature_bundle(
    sequence::AbstractString;
    chain_id::String = "A0",
    task_name::String = "protenix_mini_sequence",
    rng::AbstractRNG = Random.default_rng(),
)
    atoms = build_sequence_atoms(sequence; chain_id = chain_id)
    bundle = build_feature_bundle_from_atoms(atoms; task_name = task_name, rng = rng)
    feat = bundle["input_feature_dict"]

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

    return bundle
end

"""
Run infer-only Protenix-mini folding from a one-letter sequence.
"""
function fold_sequence(
    model::ProtenixMiniModel,
    sequence::AbstractString;
    chain_id::String = "A0",
    n_cycle::Int = model.n_cycle,
    n_step::Int = model.sample_n_step,
    n_sample::Int = model.sample_n_sample,
    rng::AbstractRNG = Random.default_rng(),
)
    bundle = build_sequence_feature_bundle(
        sequence;
        chain_id = chain_id,
        task_name = "protenix_mini_sequence",
        rng = rng,
    )
    pred = run_inference(
        model,
        as_protenix_features(bundle["input_feature_dict"]);
        n_cycle = n_cycle,
        n_step = n_step,
        n_sample = n_sample,
        rng = rng,
    )
    return (
        atoms = bundle["atoms"],
        feature_bundle = bundle,
        prediction = pred,
    )
end

end

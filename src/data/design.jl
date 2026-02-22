module Design

import ..Constants:
    DNA_STD_RESIDUES,
    RNA_STD_RESIDUES_NATURAL,
    PROT_STD_RESIDUES_ONE_TO_THREE,
    STD_RESIDUES_WITH_GAP,
    STD_RESIDUES_WITH_GAP_ID_TO_NAME
import ..Tokenizer: AtomRecord

export restype_onehot_encoded, cano_seq_resname_with_mask, canonical_resname_for_atom

const PROT_THREE_TO_ONE = let
    d = Dict{String, String}()
    for (one, three) in PROT_STD_RESIDUES_ONE_TO_THREE
        d[three] = one
    end
    d
end

function _ordered_restype_names()
    max_id = maximum(values(STD_RESIDUES_WITH_GAP))
    return [STD_RESIDUES_WITH_GAP_ID_TO_NAME[i] for i in 0:max_id]
end

function restype_onehot_encoded(restype_list::Vector{String})
    order = _ordered_restype_names()
    depth = length(order)
    id_of = Dict(name => i for (i, name) in enumerate(order))
    out = zeros(Float32, length(restype_list), depth)
    for (i, x) in enumerate(restype_list)
        haskey(id_of, x) || error("Unknown restype token '$x' for onehot encoding.")
        out[i, id_of[x]] = 1f0
    end
    return out
end

function _residue_runs(atoms::Vector{AtomRecord})
    isempty(atoms) && return Tuple{Int, Int}[]
    runs = Tuple{Int, Int}[]
    start = 1
    for i in 2:length(atoms)
        a = atoms[i - 1]
        b = atoms[i]
        if !(a.chain_id == b.chain_id && a.res_id == b.res_id)
            push!(runs, (start, i - 1))
            start = i
        end
    end
    push!(runs, (start, length(atoms)))
    return runs
end

function _canonical_resname(mol_type::String, res_name::String)
    # Design placeholder
    if res_name == "xpb"
        return mol_type == "protein" ? get(PROT_STD_RESIDUES_ONE_TO_THREE, "j", "UNK") :
               mol_type == "dna" ? "DN" :
               mol_type == "rna" ? "N" : "UNK"
    end

    if mol_type == "protein"
        one_letter = get(PROT_THREE_TO_ONE, res_name, "X")
        return get(PROT_STD_RESIDUES_ONE_TO_THREE, one_letter, "UNK")
    elseif mol_type == "dna"
        # Standard DNA names (DA, DC, DG, DT) are already canonical
        haskey(DNA_STD_RESIDUES, res_name) && return res_name
        # Modified DNA: try one-letter code lookup via protein table (some share names)
        one_letter = get(PROT_THREE_TO_ONE, res_name, "N")
        dna_name = "D" * one_letter
        return haskey(DNA_STD_RESIDUES, dna_name) ? dna_name : "DN"
    elseif mol_type == "rna"
        # Standard RNA names (A, U, C, G) are already canonical
        haskey(RNA_STD_RESIDUES_NATURAL, res_name) && return res_name
        # Modified RNA: try one-letter code lookup
        one_letter = get(PROT_THREE_TO_ONE, res_name, "N")
        return haskey(RNA_STD_RESIDUES_NATURAL, one_letter) ? one_letter : "N"
    else
        return "UNK"
    end
end

function canonical_resname_for_atom(atom::AtomRecord)
    return _canonical_resname(atom.mol_type, atom.res_name)
end

"""
Return canonical residue names per residue, matching `cano_seq_resname_with_mask` intent.
"""
function cano_seq_resname_with_mask(atoms::Vector{AtomRecord})
    out = String[]
    for (start, _) in _residue_runs(atoms)
        a = atoms[start]
        push!(out, _canonical_resname(a.mol_type, a.res_name))
    end
    return out
end

end

module Tokenizer

import ..Constants: STD_RESIDUES, ELEMS

export AtomRecord, Token, TokenArray, tokenize_atoms, centre_atom_indices

struct AtomRecord
    atom_name::String
    res_name::String
    mol_type::String
    element::String
    chain_id::String
    res_id::Int
    centre_atom_mask::Bool
    x::Float32
    y::Float32
    z::Float32
    is_resolved::Bool
end

function AtomRecord(
    atom_name::String,
    res_name::String,
    mol_type::String,
    element::String,
    chain_id::String,
    res_id::Int,
    centre_atom_mask::Bool,
)
    return AtomRecord(
        atom_name,
        res_name,
        mol_type,
        element,
        chain_id,
        res_id,
        centre_atom_mask,
        0f0,
        0f0,
        0f0,
        false,
    )
end

mutable struct Token
    value::Int
    atom_indices::Vector{Int}
    atom_names::Vector{String}
    centre_atom_index::Int
end

struct TokenArray
    tokens::Vector{Token}
end

Base.length(ta::TokenArray) = length(ta.tokens)
Base.iterate(ta::TokenArray, st...) = iterate(ta.tokens, st...)
Base.getindex(ta::TokenArray, i::Int) = ta.tokens[i]

centre_atom_indices(ta::TokenArray) = [t.centre_atom_index for t in ta.tokens]

function _residue_runs(atoms::Vector{AtomRecord})
    isempty(atoms) && return Tuple{Int, Int}[]
    runs = Tuple{Int, Int}[]
    start = 1
    for i in 2:length(atoms)
        prev = atoms[i - 1]
        cur = atoms[i]
        if !(prev.chain_id == cur.chain_id && prev.res_id == cur.res_id)
            push!(runs, (start, i - 1))
            start = i
        end
    end
    push!(runs, (start, length(atoms)))
    return runs
end

function _unknown_token_value(mol_type::String)
    if mol_type == "protein"
        return STD_RESIDUES["UNK"]
    elseif mol_type == "dna"
        return STD_RESIDUES["DN"]
    elseif mol_type == "rna"
        return STD_RESIDUES["N"]
    end
    return STD_RESIDUES["UNK"]
end

function _select_centre_atom_index(atoms::Vector{AtomRecord}, start::Int, stop::Int)
    for i in start:stop
        if atoms[i].centre_atom_mask
            return i
        end
    end
    return start
end

function tokenize_atoms(atoms::Vector{AtomRecord})::TokenArray
    tokens = Token[]
    for (start, stop) in _residue_runs(atoms)
        first_atom = atoms[start]
        res_name = first_atom.res_name
        mol_type = first_atom.mol_type
        if mol_type != "ligand"
            res_token = get(STD_RESIDUES, res_name, _unknown_token_value(mol_type))
            atom_idx = collect(start:stop)
            atom_names = [atoms[i].atom_name for i in start:stop]
            centre_idx = _select_centre_atom_index(atoms, start, stop)
            push!(tokens, Token(res_token, atom_idx, atom_names, centre_idx))
        else
            for i in start:stop
                elem = uppercase(atoms[i].element)
                elem_token = get(ELEMS, elem, nothing)
                elem_token === nothing && error("Unknown atom element: $(atoms[i].element)")
                push!(tokens, Token(elem_token, [i], [atoms[i].atom_name], i))
            end
        end
    end

    return TokenArray(tokens)
end

end

module ProtenixMoleculeFlowExt

using Protenix
using MoleculeFlow

import Protenix.ProtenixAPI: _smiles_to_atoms_and_bonds, AtomRecord

function _canonical_element_symbol(sym::AbstractString)
    s = strip(String(sym))
    isempty(s) && return "C"
    chars = collect(lowercase(s))
    chars[1] = uppercase(chars[1])
    return String(chars)
end

function _next_atom_name(sym::String, counts::Dict{String,Int})
    n = get(counts, sym, 0) + 1
    counts[sym] = n
    name = string(sym, n)
    ncodeunits(name) <= 4 && return name
    short = string(sym[1], n)
    return ncodeunits(short) <= 4 ? short : string(sym[1], mod(n, 999))
end

function _coords_from_conformer(mol, n_atoms::Int)
    coords3 = get(mol.props, :coordinates_3d, nothing)
    coords3 === nothing && error("SMILES conformer is missing :coordinates_3d")
    size(coords3, 2) >= 3 || error("SMILES conformer has invalid coordinate shape: $(size(coords3))")
    if size(coords3, 1) == n_atoms
        return Float32.(coords3[:, 1:3])
    elseif size(coords3, 1) > n_atoms
        # MoleculeFlow may include H-expanded conformer; keep heavy-atom rows only
        return Float32.(coords3[1:n_atoms, 1:3])
    end
    error("SMILES conformer atom count mismatch: coords=$(size(coords3, 1)) atoms=$n_atoms")
end

function Protenix.ProtenixAPI._smiles_to_atoms_and_bonds(smiles::String, chain_id::String)
    mol = MoleculeFlow.mol_from_smiles(smiles)
    mol.valid || error("Invalid SMILES: $smiles")

    # Match Python Protenix: bare AllChem.EmbedMolecule without MMFF optimization.
    # Coordinates will differ from Python (RDKit's internal RNG is not seeded by
    # Python's random.seed), but atom ordering and naming will match.
    confs = MoleculeFlow.generate_3d_conformers(mol, 1; optimize=false, random_seed=1)
    isempty(confs) && error("Failed to generate 3D conformer for SMILES: $smiles")
    mol_conf = confs[1].molecule

    atoms_any = MoleculeFlow.get_atoms(mol_conf)
    atoms_any === missing && error("Failed to extract atoms from SMILES: $smiles")
    atoms_vec = atoms_any::Vector
    n_atoms = length(atoms_vec)
    n_atoms > 0 || error("No atoms found in SMILES: $smiles")
    coords = _coords_from_conformer(mol_conf, n_atoms)

    # Build atom records with sequential element-based names (C1, N2, O3, ...)
    counts = Dict{String,Int}()
    atom_records = AtomRecord[]
    atom_names = String[]
    for i in 1:n_atoms
        sym = _canonical_element_symbol(MoleculeFlow.get_symbol(atoms_vec[i]))
        atom_name = _next_atom_name(sym, counts)
        push!(atom_names, atom_name)
        push!(
            atom_records,
            AtomRecord(
                atom_name, "UNL", "ligand", sym, chain_id, 1, true,
                coords[i, 1], coords[i, 2], coords[i, 3], true,
            ),
        )
    end

    # Extract bonds between heavy atoms
    seen = Dict{Tuple{Int,Int},Bool}()
    bonds = Tuple{String,String}[]
    for i in 1:n_atoms
        bonds_any = MoleculeFlow.get_bonds_from_atom(mol_conf, i)
        bonds_any === missing && continue
        for bnd in bonds_any::Vector
            a = Int(MoleculeFlow.get_begin_atom_idx(bnd))
            b = Int(MoleculeFlow.get_end_atom_idx(bnd))
            1 <= a <= n_atoms || error("SMILES bond begin index out of range: $a (n_atoms=$n_atoms)")
            1 <= b <= n_atoms || error("SMILES bond end index out of range: $b (n_atoms=$n_atoms)")
            a == b && continue
            i1, i2 = a < b ? (a, b) : (b, a)
            haskey(seen, (i1, i2)) && continue
            seen[(i1, i2)] = true
            push!(bonds, (atom_names[i1], atom_names[i2]))
        end
    end

    return atom_records, bonds
end

end

module Output

using Printf

import ..Data: AtomRecord

export dump_prediction_bundle

const ResidueRun = NamedTuple{
    (:chain_id, :res_id, :res_name, :start_idx, :stop_idx),
    Tuple{String, Int, String, Int, Int},
}

function _chain_order(atoms::Vector{AtomRecord})
    order = String[]
    seen = Set{String}()
    for a in atoms
        if !(a.chain_id in seen)
            push!(order, a.chain_id)
            push!(seen, a.chain_id)
        end
    end
    return order
end

function _quote_atom_name(name::String)
    return occursin('\'', name) ? string('"', name, '"') : name
end

function _residue_runs(atoms::Vector{AtomRecord})
    isempty(atoms) && return ResidueRun[]
    runs = ResidueRun[]
    start_idx = 1
    for i in 2:length(atoms)
        a = atoms[i - 1]
        b = atoms[i]
        if !(a.chain_id == b.chain_id && a.res_id == b.res_id)
            push!(
                runs,
                (
                    chain_id = a.chain_id,
                    res_id = a.res_id,
                    res_name = a.res_name,
                    start_idx = start_idx,
                    stop_idx = i - 1,
                ),
            )
            start_idx = i
        end
    end
    a = atoms[end]
    push!(
        runs,
        (
            chain_id = a.chain_id,
            res_id = a.res_id,
            res_name = a.res_name,
            start_idx = start_idx,
            stop_idx = length(atoms),
        ),
    )
    return runs
end

function _residue_has_atom(
    atoms::Vector{AtomRecord},
    run::ResidueRun,
    atom_name::String,
)
    for i in run.start_idx:run.stop_idx
        atoms[i].atom_name == atom_name && return true
    end
    return false
end

function _chain_mol_type(atoms::Vector{AtomRecord}, chains::Vector{String})
    chain_to_mol = Dict{String, String}()
    for a in atoms
        if !haskey(chain_to_mol, a.chain_id)
            chain_to_mol[a.chain_id] = a.mol_type
        end
    end
    return chain_to_mol
end

function _entity_type(mol_type::String)
    mol_type in ("protein", "dna", "rna") && return "polymer"
    return "non-polymer"
end

function _polymer_type(mol_type::String)
    mol_type == "protein" && return "polypeptide(L)"
    mol_type == "dna" && return "polydeoxyribonucleotide"
    mol_type == "rna" && return "polyribonucleotide"
    return nothing
end

function _chain_residue_key(chain::String, chain_to_mol::Dict{String,String}, runs::Vector{ResidueRun})
    mol = chain_to_mol[chain]
    seq = String[r.res_name for r in runs if r.chain_id == chain]
    return (mol, seq)
end

function _entity_grouping(chains::Vector{String}, chain_to_mol::Dict{String,String}, runs::Vector{ResidueRun})
    chain_to_entity = Dict{String, Int}()
    key_to_entity = Dict{Tuple{String,Vector{String}}, Int}()
    next_id = 1
    for c in chains
        key = _chain_residue_key(c, chain_to_mol, runs)
        if haskey(key_to_entity, key)
            chain_to_entity[c] = key_to_entity[key]
        else
            key_to_entity[key] = next_id
            chain_to_entity[c] = next_id
            next_id += 1
        end
    end
    return chain_to_entity
end

function _write_cif(path::AbstractString, atoms::Vector{AtomRecord}, coord::Array{Float32,2}; entry_id::String = "pxdesign", cross_chain_bonds=nothing)
    size(coord, 1) == 3 || error("Coordinate tensor must be (3, N_atom) features-first.")
    size(coord, 2) == length(atoms) || error("Coordinate column count must match atom count.")

    chains = _chain_order(atoms)
    chain_to_mol = _chain_mol_type(atoms, chains)
    runs = _residue_runs(atoms)

    # Entity grouping: merge chains with same mol_type + residue sequence
    chain_to_entity = _entity_grouping(chains, chain_to_mol, runs)

    open(path, "w") do io
        println(io, "data_$(entry_id)")
        println(io, "_entry.id ", entry_id)
        println(io, "#")

        # Build entity → chains mapping (deduplicated)
        entity_chains = Dict{Int, Vector{String}}()
        for c in chains
            eid = chain_to_entity[c]
            if !haskey(entity_chains, eid)
                entity_chains[eid] = String[]
            end
            push!(entity_chains[eid], c)
        end
        entity_ids = sort(collect(keys(entity_chains)))

        # _entity — polymer for protein/dna/rna, non-polymer for ligand/ion
        println(io, "loop_")
        println(io, "_entity.id")
        println(io, "_entity.pdbx_description")
        println(io, "_entity.type")
        for eid in entity_ids
            mol = chain_to_mol[entity_chains[eid][1]]
            println(io, eid, " . ", _entity_type(mol))
        end
        println(io, "#")

        # _entity_poly — only for polymer entities
        poly_entity_ids = filter(eid -> _entity_type(chain_to_mol[entity_chains[eid][1]]) == "polymer", entity_ids)
        if !isempty(poly_entity_ids)
            println(io, "loop_")
            println(io, "_entity_poly.entity_id")
            println(io, "_entity_poly.pdbx_strand_id")
            println(io, "_entity_poly.type")
            for eid in poly_entity_ids
                strand_ids = join(entity_chains[eid], ",")
                mol = chain_to_mol[entity_chains[eid][1]]
                println(io, eid, " ", strand_ids, " ", _polymer_type(mol))
            end
            println(io, "#")
        end

        # _entity_poly_seq — only for polymer entities (one copy per entity, from first chain)
        if !isempty(poly_entity_ids)
            println(io, "loop_")
            println(io, "_entity_poly_seq.entity_id")
            println(io, "_entity_poly_seq.hetero")
            println(io, "_entity_poly_seq.mon_id")
            println(io, "_entity_poly_seq.num")
            for eid in poly_entity_ids
                first_chain = entity_chains[eid][1]
                for run in runs
                    run.chain_id == first_chain || continue
                    println(io, eid, " n ", run.res_name, " ", run.res_id)
                end
            end
            println(io, "#")
        end

        # _struct_conn — peptide bonds (C-N) for protein, phosphodiester (O3'-P) for DNA/RNA
        println(io, "loop_")
        println(io, "_struct_conn.id")
        println(io, "_struct_conn.conn_type_id")
        println(io, "_struct_conn.pdbx_value_order")
        println(io, "_struct_conn.ptnr1_label_asym_id")
        println(io, "_struct_conn.ptnr2_label_asym_id")
        println(io, "_struct_conn.ptnr1_label_comp_id")
        println(io, "_struct_conn.ptnr2_label_comp_id")
        println(io, "_struct_conn.ptnr1_label_seq_id")
        println(io, "_struct_conn.ptnr2_label_seq_id")
        println(io, "_struct_conn.ptnr1_label_atom_id")
        println(io, "_struct_conn.ptnr2_label_atom_id")
        println(io, "_struct_conn.pdbx_ptnr1_PDB_ins_code")
        println(io, "_struct_conn.pdbx_ptnr2_PDB_ins_code")
        conn_id = 1
        for i in 2:length(runs)
            prev = runs[i - 1]
            curr = runs[i]
            prev.chain_id == curr.chain_id || continue
            mol = chain_to_mol[prev.chain_id]
            if mol == "protein"
                _residue_has_atom(atoms, prev, "C") || continue
                _residue_has_atom(atoms, curr, "N") || continue
                cid = prev.chain_id
                println(
                    io,
                    conn_id,
                    " covale sing ",
                    cid, " ", cid, " ",
                    prev.res_name, " ", curr.res_name, " ",
                    prev.res_id, " ", curr.res_id,
                    " C N . .",
                )
                conn_id += 1
            elseif mol == "dna" || mol == "rna"
                _residue_has_atom(atoms, prev, "O3'") || continue
                _residue_has_atom(atoms, curr, "P") || continue
                cid = prev.chain_id
                println(
                    io,
                    conn_id,
                    " covale sing ",
                    cid, " ", cid, " ",
                    prev.res_name, " ", curr.res_name, " ",
                    prev.res_id, " ", curr.res_id,
                    " ", _quote_atom_name("O3'"), " P . .",
                )
                conn_id += 1
            end
            # ligand/ion chains: no polymeric bonds
        end
        # Cross-chain covalent bonds (from input JSON covalent_bonds)
        if cross_chain_bonds !== nothing
            for b in cross_chain_bonds
                println(
                    io,
                    conn_id,
                    " covale sing ",
                    b.chain1, " ", b.chain2, " ",
                    b.res_name1, " ", b.res_name2, " ",
                    b.res_id1, " ", b.res_id2, " ",
                    _quote_atom_name(b.atom_name1), " ", _quote_atom_name(b.atom_name2),
                    " . .",
                )
                conn_id += 1
            end
        end
        println(io, "#")

        println(io, "loop_")
        println(io, "_atom_site.group_PDB")
        println(io, "_atom_site.type_symbol")
        println(io, "_atom_site.label_atom_id")
        println(io, "_atom_site.label_alt_id")
        println(io, "_atom_site.label_comp_id")
        println(io, "_atom_site.label_asym_id")
        println(io, "_atom_site.label_entity_id")
        println(io, "_atom_site.label_seq_id")
        println(io, "_atom_site.pdbx_PDB_ins_code")
        println(io, "_atom_site.auth_seq_id")
        println(io, "_atom_site.auth_comp_id")
        println(io, "_atom_site.auth_asym_id")
        println(io, "_atom_site.auth_atom_id")
        println(io, "_atom_site.B_iso_or_equiv")
        println(io, "_atom_site.occupancy")
        println(io, "_atom_site.Cartn_x")
        println(io, "_atom_site.Cartn_y")
        println(io, "_atom_site.Cartn_z")
        println(io, "_atom_site.pdbx_PDB_model_num")
        println(io, "_atom_site.id")

        for (i, a) in enumerate(atoms)
            group_pdb = a.mol_type in ("ligand", "ion") ? "HETATM" : "ATOM"
            entity_id = chain_to_entity[a.chain_id]
            qname = _quote_atom_name(a.atom_name)
            x = coord[1, i]
            y = coord[2, i]
            z = coord[3, i]
            println(
                io,
                string(
                    group_pdb, " ",
                    a.element, " ",
                    qname, " . ",
                    a.res_name, " ",
                    a.chain_id, " ",
                    entity_id, " ",
                    a.res_id, " . ",
                    a.res_id, " ",
                    a.res_name, " ",
                    a.chain_id, " ",
                    qname, " ",
                    "0.0 1.0 ",
                    @sprintf("%.6f", x), " ",
                    @sprintf("%.6f", y), " ",
                    @sprintf("%.6f", z), " ",
                    "1 ",
                    i,
                ),
            )
        end
        println(io, "#")
    end
    return path
end

function _mark_success(task_dump_dir::AbstractString)
    open(joinpath(task_dump_dir, "SUCCESS_FILE"), "w") do io
        write(io, "{\"prediction\": true}\n")
    end
end

function _write_sample_level_csv(pred_dir::AbstractString, task_name::AbstractString, n_sample::Int)
    path = joinpath(pred_dir, "sample_level_output.csv")
    open(path, "w") do io
        println(io, "sample_name,sample_idx,status")
        for i in 0:(n_sample - 1)
            println(io, "$(task_name),$(i),ok")
        end
    end
    return path
end

function dump_prediction_bundle(
    task_dump_dir::AbstractString,
    task_name::AbstractString,
    atoms::Vector{AtomRecord},
    coordinates::AbstractArray{Float32,3};
    cross_chain_bonds=nothing,
)
    # Features-first: (3, N_atom, N_sample)
    size(coordinates, 1) == 3 || error("Coordinates must be (3, N_atom, N_sample) features-first.")
    size(coordinates, 2) == length(atoms) || error("Coordinates dim-2 must match atom count.")

    pred_dir = joinpath(task_dump_dir, "predictions")
    mkpath(pred_dir)
    n_sample = size(coordinates, 3)
    for sample_idx in 1:n_sample
        cif_path = joinpath(pred_dir, "$(task_name)_sample_$(sample_idx - 1).cif")
        coord2 = Array{Float32,2}(coordinates[:, :, sample_idx])  # (3, N_atom)
        _write_cif(
            cif_path,
            atoms,
            coord2;
            entry_id = "$(task_name)_sample_$(sample_idx - 1)",
            cross_chain_bonds=cross_chain_bonds,
        )
    end
    _write_sample_level_csv(pred_dir, task_name, n_sample)
    _mark_success(task_dump_dir)
    return pred_dir
end

end

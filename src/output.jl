module Output

using Printf

import ..Data: AtomRecord

export dump_prediction_bundle

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

function _chain_index_map(atoms::Vector{AtomRecord})
    chain_to_idx = Dict{String, Int}()
    next_idx = 1
    for a in atoms
        if !haskey(chain_to_idx, a.chain_id)
            chain_to_idx[a.chain_id] = next_idx
            next_idx += 1
        end
    end
    return chain_to_idx
end

function _normalize_asym_id(chain_id::String)
    return occursin(r"^[A-Za-z]$", chain_id) ? string(chain_id, "0") : chain_id
end

function _strand_id(asym_id::String)
    stripped = replace(asym_id, r"[0-9]+$" => "")
    return isempty(stripped) ? asym_id : stripped
end

function _residue_runs(atoms::Vector{AtomRecord})
    isempty(atoms) && return NamedTuple[]
    runs = NamedTuple[]
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
    run::NamedTuple{(:chain_id, :res_id, :res_name, :start_idx, :stop_idx)},
    atom_name::String,
)
    for i in run.start_idx:run.stop_idx
        atoms[i].atom_name == atom_name && return true
    end
    return false
end

function _write_cif(path::AbstractString, atoms::Vector{AtomRecord}, coord::Array{Float32,2}; entry_id::String = "pxdesign")
    size(coord, 1) == length(atoms) || error("Coordinate row count must match atom count.")
    size(coord, 2) == 3 || error("Coordinate tensor must be [N_atom, 3].")

    chains = _chain_order(atoms)
    chain_to_entity = _chain_index_map(atoms)
    chain_to_asym = Dict{String, String}(c => _normalize_asym_id(c) for c in chains)
    runs = _residue_runs(atoms)

    open(path, "w") do io
        println(io, "data_$(entry_id)")
        println(io, "_entry.id ", entry_id)
        println(io, "#")

        println(io, "loop_")
        println(io, "_entity.id")
        println(io, "_entity.pdbx_description")
        println(io, "_entity.type")
        for c in chains
            println(io, chain_to_entity[c], " . polymer")
        end
        println(io, "#")

        println(io, "loop_")
        println(io, "_entity_poly.entity_id")
        println(io, "_entity_poly.pdbx_strand_id")
        println(io, "_entity_poly.type")
        for c in chains
            asym = chain_to_asym[c]
            println(io, chain_to_entity[c], " ", _strand_id(asym), " polypeptide(L)")
        end
        println(io, "#")

        println(io, "loop_")
        println(io, "_entity_poly_seq.entity_id")
        println(io, "_entity_poly_seq.hetero")
        println(io, "_entity_poly_seq.mon_id")
        println(io, "_entity_poly_seq.num")
        for run in runs
            println(io, chain_to_entity[run.chain_id], " n ", run.res_name, " ", run.res_id)
        end
        println(io, "#")

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
            _residue_has_atom(atoms, prev, "C") || continue
            _residue_has_atom(atoms, curr, "N") || continue
            asym = chain_to_asym[prev.chain_id]
            println(
                io,
                conn_id,
                " covale sing ",
                asym,
                " ",
                asym,
                " ",
                prev.res_name,
                " ",
                curr.res_name,
                " ",
                prev.res_id,
                " ",
                curr.res_id,
                " C N . .",
            )
            conn_id += 1
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
            group_pdb = a.mol_type == "ligand" ? "HETATM" : "ATOM"
            asym_id = chain_to_asym[a.chain_id]
            entity_id = chain_to_entity[a.chain_id]
            x = coord[i, 1]
            y = coord[i, 2]
            z = coord[i, 3]
            println(
                io,
                string(
                    group_pdb, " ",
                    a.element, " ",
                    a.atom_name, " . ",
                    a.res_name, " ",
                    asym_id, " ",
                    entity_id, " ",
                    a.res_id, " ? ",
                    a.res_id, " ",
                    a.res_name, " ",
                    asym_id, " ",
                    a.atom_name, " ",
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
    coordinates::Array{Float32,3},
)
    size(coordinates, 2) == length(atoms) || error("Coordinates must be [N_sample, N_atom, 3].")
    size(coordinates, 3) == 3 || error("Coordinates must be [N_sample, N_atom, 3].")

    pred_dir = joinpath(task_dump_dir, "predictions")
    mkpath(pred_dir)
    n_sample = size(coordinates, 1)
    for sample_idx in 1:n_sample
        cif_path = joinpath(pred_dir, "$(task_name)_sample_$(sample_idx - 1).cif")
        coord2 = Array{Float32,2}(coordinates[sample_idx, :, :])
        _write_cif(
            cif_path,
            atoms,
            coord2;
            entry_id = "$(task_name)_sample_$(sample_idx - 1)",
        )
    end
    _write_sample_level_csv(pred_dir, task_name, n_sample)
    _mark_success(task_dump_dir)
    return pred_dir
end

end

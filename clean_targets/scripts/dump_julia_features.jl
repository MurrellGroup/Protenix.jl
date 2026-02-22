#!/usr/bin/env julia
#
# Dump the input_feature_dict from Julia PXDesign.jl featurization pipeline
# WITHOUT running the model forward pass. Produces JLD2 files with
# all featurization tensors for parity comparison against Python.
#
# Usage: julia --project=<env> clean_targets/scripts/dump_julia_features.jl \
#       clean_targets/inputs/06_protein_ligand_ccd.json \
#       clean_targets/feature_dumps/julia

using MoleculeFlow  # triggers PXDesignMoleculeFlowExt for SMILES ligand bonds + 3D coords
using PXDesign
using Random

const ROOT = joinpath(@__DIR__, "..", "..")

"""Write a numpy .npy file (v1.0 format) for a Julia array."""
function write_npy(path::String, arr::AbstractArray)
    # Numpy .npy format: magic + version + header + data
    # Julia is column-major, numpy expects C-order (row-major)
    # We write in Fortran order and set the FORTRAN_ORDER flag
    open(path, "w") do io
        # Magic
        write(io, UInt8[0x93], codeunits("NUMPY"), UInt8[0x01, 0x00])
        # Build header
        dtype = if eltype(arr) == Float32
            "<f4"
        elseif eltype(arr) == Float64
            "<f8"
        elseif eltype(arr) == Int64
            "<i8"
        elseif eltype(arr) == Int32
            "<i4"
        elseif eltype(arr) == Bool
            "|b1"
        elseif eltype(arr) == UInt8
            "|u1"
        else
            error("Unsupported eltype: $(eltype(arr))")
        end
        shape_str = if ndims(arr) == 1
            "($(size(arr, 1)),)"
        else
            "(" * join(size(arr), ", ") * ")"
        end
        header = "{'descr': '$(dtype)', 'fortran_order': True, 'shape': $(shape_str), }"
        # Pad header to 64-byte alignment (10 bytes for magic+version+header_len)
        pad = 64 - ((10 + length(header) + 1) % 64)
        if pad == 64; pad = 0; end
        header = header * " "^pad * "\n"
        header_len = UInt16(length(header))
        write(io, reinterpret(UInt8, [header_len]))
        write(io, codeunits(header))
        # Data in column-major (Fortran) order
        write(io, arr)
    end
end

function _save_feat_dict(dir::String, feat::Dict)
    for (key, val) in feat
        if val isa BitArray
            # Convert BitArray to Array{Bool} for proper npy serialization
            write_npy(joinpath(dir, "$(key).npy"), Array{Bool}(val))
        elseif val isa AbstractArray
            write_npy(joinpath(dir, "$(key).npy"), val)
        elseif val isa Dict
            subdir = joinpath(dir, key)
            mkpath(subdir)
            for (subkey, subval) in val
                if subval isa AbstractArray
                    write_npy(joinpath(subdir, "$(subkey).npy"), subval)
                end
            end
        end
    end
end

function dump_features(json_path::String, dump_dir::String)
    mkpath(dump_dir)

    # Parse entities exactly as predict_json does
    raw = PXDesign.ProtenixAPI._ensure_json_tasks(json_path)
    for (task_idx, task_any) in enumerate(raw)
        task = PXDesign.ProtenixAPI._as_string_dict(task_any)
        task_name = get(task, "name", "sample_$(task_idx - 1)")

        parsed_task = PXDesign.ProtenixAPI._parse_task_entities(task; json_dir=dirname(abspath(json_path)))
        atoms = parsed_task.atoms

        # Build features with seed 101 (same as Python reference)
        rng = MersenneTwister(101)
        bundle = PXDesign.ProtenixAPI.build_feature_bundle_from_atoms(atoms; task_name=task_name, rng=rng)
        token_chain_ids = [bundle["atoms"][tok.centre_atom_index].chain_id for tok in bundle["tokens"]]

        PXDesign.ProtenixAPI._normalize_protenix_feature_dict!(bundle["input_feature_dict"])

        # Inject features exactly as predict_json does (minus model-specific ones)
        PXDesign.ProtenixAPI._inject_task_msa_features!(
            bundle["input_feature_dict"],
            task,
            json_path;
            use_msa=false,
            chain_specs=parsed_task.protein_specs,
            token_chain_ids=token_chain_ids,
        )
        PXDesign.ProtenixAPI._inject_task_covalent_token_bonds!(
            bundle["input_feature_dict"],
            bundle["atoms"],
            task,
            parsed_task.entity_chain_ids,
            parsed_task.entity_atom_map,
        )
        PXDesign.ProtenixAPI._inject_task_template_features!(bundle["input_feature_dict"], task)
        PXDesign.ProtenixAPI._inject_task_esm_token_embedding!(bundle["input_feature_dict"], task)
        PXDesign.ProtenixAPI._inject_task_constraint_feature!(
            bundle["input_feature_dict"],
            task,
            bundle["atoms"],
            parsed_task.entity_chain_ids,
            parsed_task.entity_atom_map,
            "task '$task_name'",
        )

        feat = bundle["input_feature_dict"]
        atoms_list = bundle["atoms"]
        tokens = bundle["tokens"]

        # Print summary
        println("\n", "="^60)
        println("Sample: $task_name")
        n_tok = length(tokens)
        n_atom = length(atoms_list)
        println("N_token: $n_tok")
        println("N_atom:  $n_atom")
        println("="^60)

        for key in sort(collect(keys(feat)))
            val = feat[key]
            if val isa AbstractArray
                println("  $(rpad(key, 35))  size=$(size(val))  eltype=$(eltype(val))")
            elseif val isa Dict
                println("  $(rpad(key, 35))  <dict with $(length(val)) keys>")
                for (subkey, subval) in sort(collect(val); by=first)
                    if subval isa AbstractArray
                        println("    .$(rpad(subkey, 31))  size=$(size(subval))  eltype=$(eltype(subval))")
                    end
                end
            else
                println("  $(rpad(key, 35))  type=$(typeof(val))  value=$val")
            end
        end

        # Save each tensor as a separate binary file with metadata
        feat_dir = joinpath(dump_dir, "$(task_name)_features")
        mkpath(feat_dir)
        _save_feat_dict(feat_dir, feat)
        println("\nSaved features to: $feat_dir/")

        # Dump per-atom info
        info_path = joinpath(dump_dir, "$(task_name)_atom_info.txt")
        open(info_path, "w") do io
            println(io, "N_token: $n_tok")
            println(io, "N_atom:  $n_atom")
            println(io)
            println(io, lpad("idx", 5), "  ", rpad("atom_name", 10), "  ", rpad("res_name", 8), "  ", rpad("chain", 6), "  ", rpad("res_id", 6), "  ", rpad("element", 8), "  ", rpad("mol_type", 10))
            println(io, "-"^70)
            for (j, a) in enumerate(atoms_list)
                println(io, lpad(j - 1, 5), "  ", rpad(a.atom_name, 10), "  ", rpad(a.res_name, 8), "  ", rpad(a.chain_id, 6), "  ", rpad(a.res_id, 6), "  ", rpad(a.element, 8), "  ", rpad(a.mol_type, 10))
            end
        end
        println("Atom info saved to: $info_path")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 2
        println("Usage: julia dump_julia_features.jl <input.json> <dump_dir>")
        exit(1)
    end
    dump_features(ARGS[1], ARGS[2])
end

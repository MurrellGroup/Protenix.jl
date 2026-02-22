module Structure

import ...Ranges: parse_ranges
import ..Constants:
    DNA_STD_RESIDUES,
    PRO_STD_RESIDUES_NATURAL,
    PROTEIN_HEAVY_ATOMS,
    RNA_STD_RESIDUES_NATURAL
import ..Tokenizer: AtomRecord

export load_structure_atoms

const _MISSING_CIF = Set([".", "?", ""])

function _split_cif_row(line::AbstractString)
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

function _substr(line::AbstractString, start_col::Int, end_col::Int)
    if start_col > lastindex(line)
        return ""
    end
    stop_col = min(end_col, lastindex(line))
    return strip(String(line[start_col:stop_col]))
end

function _parse_int_maybe(x)::Union{Nothing, Int}
    s = strip(String(x))
    s in _MISSING_CIF && return nothing
    v = tryparse(Int, s)
    return v
end

function _parse_float_maybe(x)::Union{Nothing, Float64}
    s = strip(String(x))
    s in _MISSING_CIF && return nothing
    v = tryparse(Float64, s)
    return v
end

function _infer_element_from_atom_name(atom_name::AbstractString)
    for c in atom_name
        if isletter(c)
            return uppercase(string(c))
        end
    end
    return "C"
end

function _is_hydrogen_atom(atom::AtomRecord)
    atom.element == "H" && return true
    startswith(atom.atom_name, "H") && return true
    return false
end

function _normalize_mse(res_name::String, atom_name::String, element::String)
    if uppercase(res_name) != "MSE"
        return res_name, atom_name, element
    end

    if uppercase(atom_name) == "SE"
        return "MET", "SD", "S"
    end
    return "MET", atom_name, element
end

function _infer_mol_type(group_pdb::String, res_name::String)
    res = uppercase(res_name)
    if haskey(DNA_STD_RESIDUES, res)
        return "dna"
    elseif haskey(RNA_STD_RESIDUES_NATURAL, res)
        return "rna"
    elseif haskey(PRO_STD_RESIDUES_NATURAL, res) || res == "MSE" || uppercase(group_pdb) == "ATOM"
        return "protein"
    else
        return "ligand"
    end
end

function _is_centre_atom(atom_name::String, mol_type::String)
    if mol_type == "protein"
        return atom_name == "CA"
    elseif mol_type == "dna" || mol_type == "rna"
        return atom_name == "C1'"
    end
    return true
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

function _completed_residue_atoms(res_atoms::Vector{AtomRecord})
    isempty(res_atoms) && return res_atoms
    first_atom = res_atoms[1]
    if first_atom.mol_type != "protein" || !haskey(PROTEIN_HEAVY_ATOMS, first_atom.res_name)
        return [a for a in res_atoms if !_is_hydrogen_atom(a)]
    end

    expected = PROTEIN_HEAVY_ATOMS[first_atom.res_name]
    expected_no_oxt = [n for n in expected if n != "OXT"]
    expected_set = Set(expected_no_oxt)

    kept = AtomRecord[]
    seen = Set{String}()
    for a in res_atoms
        _is_hydrogen_atom(a) && continue
        if !isempty(expected_set) && !(a.atom_name in expected_set)
            continue
        end
        if a.atom_name in seen
            continue
        end
        push!(kept, a)
        push!(seen, a.atom_name)
    end

    for atom_name in expected_no_oxt
        atom_name in seen && continue
        element = _infer_element_from_atom_name(atom_name)
        push!(
            kept,
            AtomRecord(
                atom_name,
                first_atom.res_name,
                first_atom.mol_type,
                element,
                first_atom.chain_id,
                first_atom.res_id,
                _is_centre_atom(atom_name, first_atom.mol_type),
                0f0,
                0f0,
                0f0,
                false,
            ),
        )
    end

    atom_order = Dict(name => i for (i, name) in enumerate(expected))
    sort!(kept; by = a -> get(atom_order, a.atom_name, typemax(Int)))
    return kept
end

function _normalize_loaded_atoms(atoms::Vector{AtomRecord})
    out = AtomRecord[]
    for (start, stop) in _residue_runs(atoms)
        residue = atoms[start:stop]
        append!(out, _completed_residue_atoms(residue))
    end
    return out
end

function _parse_pdb_atoms(path::AbstractString)
    atoms = AtomRecord[]
    seen = Set{Tuple{String, Int, String}}()

    for line in eachline(path)
        startswith(line, "ATOM  ") || startswith(line, "HETATM") || continue
        group_pdb = startswith(line, "ATOM  ") ? "ATOM" : "HETATM"

        atom_name = uppercase(_substr(line, 13, 16))
        isempty(atom_name) && continue
        res_name = uppercase(_substr(line, 18, 20))
        chain_id = _substr(line, 22, 22)
        chain_id = isempty(chain_id) ? "?" : chain_id
        res_id = _parse_int_maybe(_substr(line, 23, 26))
        res_id === nothing && continue

        # Altloc "first" behavior: keep the first atom for each chain/residue/atom triplet.
        dedupe_key = (chain_id, res_id::Int, atom_name)
        if dedupe_key in seen
            continue
        end
        push!(seen, dedupe_key)

        x = _parse_float_maybe(_substr(line, 31, 38))
        y = _parse_float_maybe(_substr(line, 39, 46))
        z = _parse_float_maybe(_substr(line, 47, 54))
        if x === nothing || y === nothing || z === nothing
            continue
        end

        element = uppercase(_substr(line, 77, 78))
        if isempty(element)
            element = _infer_element_from_atom_name(atom_name)
        end

        res_name, atom_name, element = _normalize_mse(res_name, atom_name, element)
        mol_type = _infer_mol_type(group_pdb, res_name)
        centre_atom_mask = _is_centre_atom(atom_name, mol_type)
        push!(
            atoms,
            AtomRecord(
                atom_name,
                res_name,
                mol_type,
                element,
                chain_id,
                res_id,
                centre_atom_mask,
                Float32(x),
                Float32(y),
                Float32(z),
                true,
            ),
        )
    end

    return atoms
end

function _get_field_token(fields::Vector{String}, tokens::Vector{String}, names::Vector{String}; default::String = "")
    for name in names
        idx = findfirst(==(name), fields)
        if idx !== nothing && idx <= length(tokens)
            return tokens[idx]
        end
    end
    return default
end

function _parse_mmcif_atoms(path::AbstractString)
    lines = readlines(path)
    atoms = AtomRecord[]

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

        if isempty(fields) || !all(startswith(f, "_atom_site.") for f in fields)
            i = j
            continue
        end

        n_fields = length(fields)
        pool = String[]
        seen = Set{Tuple{String, Int, String}}()
        keep_model = nothing

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
                if length(pool) == n_fields
                    empty!(pool)
                else
                    pool = pool[(n_fields + 1):end]
                end

                group_pdb = uppercase(
                    _get_field_token(fields, toks, ["_atom_site.group_PDB"]; default = "ATOM"),
                )
                (group_pdb == "ATOM" || group_pdb == "HETATM") || continue

                model_s = _get_field_token(fields, toks, ["_atom_site.pdbx_PDB_model_num"]; default = "1")
                model_num = _parse_int_maybe(model_s)
                if model_num !== nothing
                    if keep_model === nothing
                        keep_model = model_num
                    elseif model_num != keep_model
                        continue
                    end
                end

                atom_name = uppercase(
                    _get_field_token(
                        fields,
                        toks,
                        ["_atom_site.label_atom_id", "_atom_site.auth_atom_id"];
                        default = "",
                    ),
                )
                isempty(atom_name) && continue

                res_name = uppercase(
                    _get_field_token(
                        fields,
                        toks,
                        ["_atom_site.label_comp_id", "_atom_site.auth_comp_id"];
                        default = "",
                    ),
                )
                isempty(res_name) && continue

                chain_id = _get_field_token(
                    fields,
                    toks,
                    ["_atom_site.auth_asym_id", "_atom_site.label_asym_id"];
                    default = "?",
                )
                chain_id = (chain_id in _MISSING_CIF) ? "?" : chain_id

                res_id_tok = _get_field_token(
                    fields,
                    toks,
                    ["_atom_site.label_seq_id", "_atom_site.auth_seq_id"];
                    default = "",
                )
                res_id = _parse_int_maybe(res_id_tok)
                res_id === nothing && continue

                dedupe_key = (chain_id, res_id::Int, atom_name)
                if dedupe_key in seen
                    continue
                end
                push!(seen, dedupe_key)

                x = _parse_float_maybe(_get_field_token(fields, toks, ["_atom_site.Cartn_x"]))
                y = _parse_float_maybe(_get_field_token(fields, toks, ["_atom_site.Cartn_y"]))
                z = _parse_float_maybe(_get_field_token(fields, toks, ["_atom_site.Cartn_z"]))
                if x === nothing || y === nothing || z === nothing
                    continue
                end

                element = uppercase(_get_field_token(fields, toks, ["_atom_site.type_symbol"]; default = ""))
                if isempty(element) || element in _MISSING_CIF
                    element = _infer_element_from_atom_name(atom_name)
                end

                res_name, atom_name, element = _normalize_mse(res_name, atom_name, element)
                mol_type = _infer_mol_type(group_pdb, res_name)
                centre_atom_mask = _is_centre_atom(atom_name, mol_type)
                push!(
                    atoms,
                    AtomRecord(
                        atom_name,
                        res_name,
                        mol_type,
                        element,
                        chain_id,
                        res_id,
                        centre_atom_mask,
                        Float32(x),
                        Float32(y),
                        Float32(z),
                        true,
                    ),
                )
            end
        end

        return atoms
    end

    error("No _atom_site loop found in mmCIF file: $path")
end

function _parse_crop(crop::Dict{String, String})
    parsed = Dict{String, Vector{Tuple{Int, Int}}}()
    for (chain, range_str) in crop
        parsed[chain] = parse_ranges(range_str)
    end
    return parsed
end

function _in_ranges(x::Int, ranges::Vector{Tuple{Int, Int}})
    for (s, e) in ranges
        if s <= x <= e
            return true
        end
    end
    return false
end

function _apply_filters(
    atoms::Vector{AtomRecord},
    selected_chains::Vector{String},
    crop::Dict{String, String},
)
    if isempty(atoms)
        return atoms
    end

    all_chains = String[]
    seen_chain = Set{String}()
    for a in atoms
        if !(a.chain_id in seen_chain)
            push!(all_chains, a.chain_id)
            push!(seen_chain, a.chain_id)
        end
    end

    keep_set = Set(selected_chains)
    if !isempty(selected_chains)
        missing = [c for c in selected_chains if !(c in seen_chain)]
        isempty(missing) || error(
            "Requested chain(s) not found in structure: $(join(missing, ',')); " *
            "available chains: $(join(all_chains, ','))",
        )
    end

    crop_ranges = _parse_crop(crop)
    out = AtomRecord[]
    for a in atoms
        if !isempty(keep_set) && !(a.chain_id in keep_set)
            continue
        end
        if haskey(crop_ranges, a.chain_id) && !_in_ranges(a.res_id, crop_ranges[a.chain_id])
            continue
        end
        push!(out, a)
    end
    return out
end

function _parse_sdf_atoms(path::AbstractString)
    atoms = AtomRecord[]
    lines = readlines(path)
    length(lines) >= 4 || error("SDF file too short: $path")

    # Line 4 is the counts line: aaabbblll...
    counts_line = lines[4]
    n_atoms = parse(Int, strip(counts_line[1:3]))
    n_bonds = length(counts_line) >= 6 ? parse(Int, strip(counts_line[4:6])) : 0

    # Track which SDF atom indices are hydrogen (to skip in bonds too)
    sdf_elements = String[]  # element per SDF atom index (1-based)
    # Map from SDF atom index → non-H atom counter (0 if hydrogen)
    sdf_to_heavy = Dict{Int, Int}()
    atom_counter = 0

    for i in 5:(4 + n_atoms)
        sdf_idx = i - 4
        i > length(lines) && break
        line = lines[i]
        length(line) < 34 && continue

        x = parse(Float32, strip(line[1:10]))
        y = parse(Float32, strip(line[11:20]))
        z = parse(Float32, strip(line[21:30]))
        element = uppercase(strip(line[31:33]))
        isempty(element) && continue

        push!(sdf_elements, element)

        # Skip hydrogen atoms for ligand representation (matches CCD behavior)
        if element == "H"
            sdf_to_heavy[sdf_idx] = 0
            continue
        end

        atom_counter += 1
        sdf_to_heavy[sdf_idx] = atom_counter
        atom_name = element * string(atom_counter)

        push!(
            atoms,
            AtomRecord(
                atom_name,
                "LIG",
                "ligand",
                element,
                "A",
                1,
                true,
                x,
                y,
                z,
                true,
            ),
        )
    end

    # Parse bond block and store bonds between non-hydrogen atoms
    bonds = Tuple{String, String}[]
    bond_start = 4 + n_atoms + 1
    for i in bond_start:(bond_start + n_bonds - 1)
        i > length(lines) && break
        line = lines[i]
        startswith(strip(line), "M") && break  # M  END
        length(line) < 6 && continue
        a1_sdf = parse(Int, strip(line[1:3]))
        a2_sdf = parse(Int, strip(line[4:6]))
        h1 = get(sdf_to_heavy, a1_sdf, 0)
        h2 = get(sdf_to_heavy, a2_sdf, 0)
        (h1 == 0 || h2 == 0) && continue  # skip bonds involving hydrogen
        # Use atom names for consistency with CCD bond format
        push!(bonds, (atoms[h1].atom_name, atoms[h2].atom_name))
    end

    return atoms, bonds
end

function load_structure_atoms(
    structure_file::AbstractString;
    selected_chains::Vector{String} = String[],
    crop::Dict{String, String} = Dict{String, String}(),
)
    path = abspath(structure_file)
    isfile(path) || error("Structure file not found: $path")

    ext = lowercase(splitext(path)[2])
    sdf_bonds = Tuple{String, String}[]
    atoms = if ext == ".pdb"
        _parse_pdb_atoms(path)
    elseif ext == ".cif" || ext == ".mmcif"
        _parse_mmcif_atoms(path)
    elseif ext == ".sdf" || ext == ".mol"
        a, b = _parse_sdf_atoms(path)
        sdf_bonds = b
        a
    else
        error("Unsupported structure file extension '$ext' for $path")
    end

    isempty(atoms) && error("No atoms parsed from structure file: $path")
    filtered = _apply_filters(atoms, selected_chains, crop)
    isempty(filtered) && error("No atoms left after applying chain/crop filters for $path")
    normalized = _normalize_loaded_atoms(filtered)
    isempty(normalized) && error("No atoms left after structure normalization for $path")
    return normalized, sdf_bonds
end

end

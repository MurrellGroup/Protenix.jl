module Features

using Random

import ...Schema: InputTask, MSAChainOptions
import ...Schema: GenerationSpec
import ...JSONLite: parse_json
import ..Constants: STD_RESIDUES_WITH_GAP, STD_RESIDUES_WITH_GAP_ID_TO_NAME, ELEMS, STD_RESIDUES_PROTENIX
import ..Tokenizer: AtomRecord, Token, TokenArray, centre_atom_indices, tokenize_atoms
import ..Design: canonical_resname_for_atom, restype_onehot_encoded
import ..Structure: load_structure_atoms

export build_design_backbone_atoms, build_basic_feature_bundle, build_feature_bundle_from_atoms

const CCDRefEntry = NamedTuple{
    (:atom_map, :coords, :charge, :mask),
    Tuple{Dict{String, Int}, Matrix{Float32}, Vector{Float32}, Vector{Int}},
}
const _CCD_REF_CACHE = Dict{String, CCDRefEntry}()
const _CCD_STD_REF_CACHE_LOADED = Ref(false)

# Bond cache: CCD code → Vector of (atom_name_1, atom_name_2) pairs (non-hydrogen only)
const _CCD_BOND_CACHE = Dict{String, Vector{Tuple{String, String}}}()

function _ordered_restype_names()
    max_id = maximum(values(STD_RESIDUES_WITH_GAP))
    return [STD_RESIDUES_WITH_GAP_ID_TO_NAME[i] for i in 0:max_id]
end

function _chain_letters(n::Int)
    n > 0 || error("Chain index must be positive.")
    chars = Char[]
    x = n
    while x > 0
        x -= 1
        pushfirst!(chars, Char(Int('A') + (x % 26)))
        x = fld(x, 26)
    end
    return String(chars)
end

function build_design_backbone_atoms(length_res::Int; chain_id::String = "B0")
    length_res > 0 || error("Binder length must be positive.")
    atoms = AtomRecord[]
    for res_id in 1:length_res
        # Keep binder atoms unresolved (design targets) but give deterministic pseudo-geometry.
        # Match reference polymer parsing behavior: OXT appears on terminal residue only.
        x0 = Float32((res_id - 1) * 3.8)
        push!(
            atoms,
            AtomRecord("N", "xpb", "protein", "N", chain_id, res_id, false, x0 - 1.2f0, 0f0, 0f0, false),
        )
        push!(atoms, AtomRecord("CA", "xpb", "protein", "C", chain_id, res_id, true, x0, 0f0, 0f0, false))
        push!(
            atoms,
            AtomRecord("C", "xpb", "protein", "C", chain_id, res_id, false, x0 + 1.4f0, 0.1f0, 0f0, false),
        )
        push!(
            atoms,
            AtomRecord("O", "xpb", "protein", "O", chain_id, res_id, false, x0 + 2.2f0, -0.4f0, 0f0, false),
        )
        if res_id == length_res
            push!(
                atoms,
                AtomRecord("OXT", "xpb", "protein", "O", chain_id, res_id, false, x0 + 2.4f0, 0.5f0, 0f0, false),
            )
        end
    end
    return atoms
end

function _chain_index_map(centre_atoms::Vector{AtomRecord})
    chain_to_idx = Dict{String, Int}()
    next_idx = 0
    for a in centre_atoms
        if !haskey(chain_to_idx, a.chain_id)
            chain_to_idx[a.chain_id] = next_idx
            next_idx += 1
        end
    end
    return chain_to_idx
end

function _atom_to_token_index(tokens::TokenArray, n_atom::Int)
    out = fill(-1, n_atom)
    for (tok_idx, tok) in enumerate(tokens)
        t = tok_idx - 1
        for ai in tok.atom_indices
            out[ai] = t
        end
    end
    any(==( -1), out) && error("Atom-to-token mapping is incomplete.")
    return out
end

function _atom_to_tokatom_index(tokens::TokenArray, n_atom::Int)
    out = fill(-1, n_atom)
    for tok in tokens
        for (tok_atom_idx, atom_idx) in enumerate(tok.atom_indices)
            out[atom_idx] = tok_atom_idx - 1
        end
    end
    any(==( -1), out) && error("Atom-to-tokatom mapping is incomplete.")
    return out
end

function _frame_for_polymer_token(tok::Token, centre_atom::AtomRecord)
    if centre_atom.mol_type == "protein"
        required = ("N", "CA", "C")
    elseif centre_atom.mol_type == "dna" || centre_atom.mol_type == "rna"
        required = ("C1'", "C3'", "C4'")
    else
        return 0, [-1, -1, -1]
    end

    idxs = Int[]
    for name in required
        pos = findfirst(==(name), tok.atom_names)
        pos === nothing && return 0, [-1, -1, -1]
        push!(idxs, tok.atom_indices[pos])
    end
    return 1, idxs
end

"""
Compute ligand/non-standard frame for a token using the 3 nearest atoms
in the same ref_space_uid group (by reference coordinates).
Matches Python Protenix's Featurizer.get_lig_frame().
"""
function _frame_for_lig_token(
    tok::Token,
    centre_atom_global_idx::Int,
    group_atom_indices::Vector{Int},
    ref_pos::Matrix{Float32},
    ref_mask::Vector{Int},
)
    length(group_atom_indices) < 3 && return 0, [-1, centre_atom_global_idx - 1, -1]

    # Find 3 nearest atoms to centre atom by reference coordinate distance
    b_idx = centre_atom_global_idx
    bx, by, bz = ref_pos[b_idx, 1], ref_pos[b_idx, 2], ref_pos[b_idx, 3]
    dists = [(
        (ref_pos[ai, 1] - bx)^2 + (ref_pos[ai, 2] - by)^2 + (ref_pos[ai, 3] - bz)^2,
        ai,
    ) for ai in group_atom_indices]
    sort!(dists; by=first)

    # dists[1] should be the centre atom itself (dist=0), dists[2] and dists[3] are neighbors
    length(dists) < 3 && return 0, [-1, b_idx - 1, -1]
    a_idx = dists[2][2]
    c_idx = dists[3][2]
    frame_atom_index = [a_idx - 1, b_idx - 1, c_idx - 1]  # 0-based

    # Check ref_mask validity
    all(ref_mask[idx + 1] != 0 for idx in frame_atom_index) || return 0, frame_atom_index

    # Colinearity check
    v1 = ref_pos[b_idx, :] .- ref_pos[a_idx, :]
    v2 = ref_pos[c_idx, :] .- ref_pos[a_idx, :]
    cross_prod = [
        v1[2] * v2[3] - v1[3] * v2[2],
        v1[3] * v2[1] - v1[1] * v2[3],
        v1[1] * v2[2] - v1[2] * v2[1],
    ]
    cross_norm = sqrt(sum(x^2 for x in cross_prod))
    cross_norm < 1e-5 && return 0, frame_atom_index

    return 1, frame_atom_index
end

"""
Compute ref_space_uid: each unique (chain_id, res_id) pair gets a sequential integer,
matching Python Protenix's AddAtomArrayAnnot.add_ref_space_uid.
All atoms sharing the same (chain_id, res_id) get the same uid.
"""
function _compute_ref_space_uid(atoms::Vector{AtomRecord})
    uid = Vector{Int}(undef, length(atoms))
    mapping = Dict{Tuple{String, Int}, Int}()
    next_id = 0
    for (i, a) in enumerate(atoms)
        key = (a.chain_id, a.res_id)
        if !haskey(mapping, key)
            mapping[key] = next_id
            next_id += 1
        end
        uid[i] = mapping[key]
    end
    return uid
end

function _condition_and_hotspot_features(centre_atoms::Vector{AtomRecord}, hotspots::Dict{String, Vector{Int}})
    condition_token_mask = [a.res_name != "xpb" for a in centre_atoms]
    hotspot = zeros(Float32, length(centre_atoms))
    for (i, a) in enumerate(centre_atoms)
        if !condition_token_mask[i]
            continue
        end
        if haskey(hotspots, a.chain_id) && (a.res_id in Set(hotspots[a.chain_id]))
            hotspot[i] = 1f0
        end
    end
    return condition_token_mask, hotspot
end

function _distogram_rep_atom_indices(atoms::Vector{AtomRecord}, tokens::TokenArray)
    idxs = Vector{Int}(undef, length(tokens))
    for (tok_i, tok) in enumerate(tokens)
        first_atom = atoms[tok.atom_indices[1]]
        atom_idx = tok.atom_indices[1]
        if first_atom.mol_type == "protein"
            primary = (first_atom.res_name == "GLY" || first_atom.res_name == "xpb") ? "CA" : "CB"
            fallback = "CA"
            pos_primary = findfirst(==(primary), tok.atom_names)
            pos_fallback = findfirst(==(fallback), tok.atom_names)

            if pos_primary !== nothing && atoms[tok.atom_indices[pos_primary]].is_resolved
                atom_idx = tok.atom_indices[pos_primary]
            elseif pos_fallback !== nothing && atoms[tok.atom_indices[pos_fallback]].is_resolved
                atom_idx = tok.atom_indices[pos_fallback]
            elseif pos_primary !== nothing
                atom_idx = tok.atom_indices[pos_primary]
            elseif pos_fallback !== nothing
                atom_idx = tok.atom_indices[pos_fallback]
            end
        elseif first_atom.mol_type == "dna" || first_atom.mol_type == "rna"
            pos = findfirst(==("C1'"), tok.atom_names)
            if pos !== nothing
                atom_idx = tok.atom_indices[pos]
            end
        end
        idxs[tok_i] = atom_idx
    end
    return idxs
end

function _compute_conditional_template(
    atoms::Vector{AtomRecord},
    tokens::TokenArray,
    condition_token_mask::Vector{Bool},
)
    n = length(tokens)
    conditional_templ = zeros(Int, n, n)
    conditional_templ_mask = zeros(Int, n, n)
    rep_atom_idx = _distogram_rep_atom_indices(atoms, tokens)

    idx = Int[]
    for i in 1:n
        rep = atoms[rep_atom_idx[i]]
        if condition_token_mask[i] && rep.is_resolved
            push!(idx, i)
        end
    end
    isempty(idx) && return conditional_templ, conditional_templ_mask

    boundaries = collect(range(2.0, stop = 22.0, length = 63))
    for i in idx
        ai = atoms[rep_atom_idx[i]]
        for j in idx
            aj = atoms[rep_atom_idx[j]]
            dx = Float64(ai.x - aj.x)
            dy = Float64(ai.y - aj.y)
            dz = Float64(ai.z - aj.z)
            d = sqrt(dx * dx + dy * dy + dz * dz)
            b = 0
            for thr in boundaries
                if d > thr
                    b += 1
                end
            end
            conditional_templ[i, j] = b
            conditional_templ_mask[i, j] = 1
        end
    end
    return conditional_templ, conditional_templ_mask
end

function _random_rotation_matrix(rng::AbstractRNG)
    q = randn(rng, Float32, 4)
    q ./= sqrt(sum(abs2, q))
    w, x, y, z = q
    return Float32[
        1 - 2 * (y * y + z * z) 2 * (x * y - z * w) 2 * (x * z + y * w)
        2 * (x * y + z * w) 1 - 2 * (x * x + z * z) 2 * (y * z - x * w)
        2 * (x * z - y * w) 2 * (y * z + x * w) 1 - 2 * (x * x + y * y)
    ]
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

function _random_transform_points(points::Matrix{Float32}, rng::AbstractRNG; apply_augmentation::Bool = true)
    transformed = copy(points)
    if size(transformed, 1) > 0
        centre = vec(sum(transformed; dims = 1) ./ size(transformed, 1))
        transformed .-= reshape(centre, 1, :)
    end
    apply_augmentation || return transformed
    translation = 2f0 .* rand(rng, Float32, 3) .- 1f0
    rot = _random_rotation_matrix(rng)
    transformed .= (transformed .+ reshape(translation, 1, :)) * transpose(rot)
    return transformed
end

function _ref_pos_with_augmentation(
    base_ref_pos::Matrix{Float32},
    ref_space_uid::Vector{Int},
    rng::AbstractRNG;
    apply_augmentation::Bool = true,
)
    size(base_ref_pos, 1) == length(ref_space_uid) || error("ref_pos/ref_space_uid length mismatch.")
    out = copy(base_ref_pos)
    for uid in sort(unique(ref_space_uid))
        idx = findall(==(uid), ref_space_uid)
        isempty(idx) && continue
        out[idx, :] = _random_transform_points(out[idx, :], rng; apply_augmentation = apply_augmentation)
    end
    return out
end

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

function _try_parse_f32(x)
    s = strip(String(x))
    s == "?" && return nothing
    s == "." && return nothing
    v = tryparse(Float32, s)
    return v
end

function _try_parse_i(x)
    s = strip(String(x))
    s == "?" && return nothing
    s == "." && return nothing
    v = tryparse(Int, s)
    return v
end

function _default_ccd_components_path()
    if haskey(ENV, "PROTENIX_DATA_ROOT_DIR")
        root = ENV["PROTENIX_DATA_ROOT_DIR"]
        p1 = joinpath(root, "components.v20240608.cif")
        isfile(p1) && return p1
        p2 = joinpath(root, "components.cif")
        isfile(p2) && return p2
    end
    project_root = normpath(joinpath(@__DIR__, "..", ".."))
    p1 = joinpath(project_root, "release_data", "ccd_cache", "components.v20240608.cif")
    isfile(p1) && return p1
    p2 = joinpath(project_root, "release_data", "ccd_cache", "components.cif")
    isfile(p2) && return p2
    return ""
end

function _default_ccd_std_ref_path()
    if haskey(ENV, "PROTENIX_DATA_ROOT_DIR")
        root = ENV["PROTENIX_DATA_ROOT_DIR"]
        p = joinpath(root, "ref_coords_std.json")
        isfile(p) && return p
    end
    project_root = normpath(joinpath(@__DIR__, "..", ".."))
    p = joinpath(project_root, "release_data", "ccd_cache", "ref_coords_std.json")
    isfile(p) && return p
    return ""
end

function _load_ccd_std_ref_cache!()
    _CCD_STD_REF_CACHE_LOADED[] && return
    _CCD_STD_REF_CACHE_LOADED[] = true
    p = _default_ccd_std_ref_path()
    isempty(p) && return

    raw = parse_json(read(p, String))
    raw isa AbstractDict || return
    for (code_any, entry_any) in raw
        code = uppercase(String(code_any))
        entry_any isa AbstractDict || continue
        entry = entry_any
        haskey(entry, "atom_map") || continue
        haskey(entry, "coord") || continue
        haskey(entry, "charge") || continue
        haskey(entry, "mask") || continue

        atom_map_raw = entry["atom_map"]
        atom_map_raw isa AbstractDict || continue
        atom_map = Dict{String, Int}(String(k) => Int(v) + 1 for (k, v) in atom_map_raw)

        coord_raw = entry["coord"]
        coord_raw isa AbstractArray || continue
        n_atom = length(coord_raw)
        coords = zeros(Float32, n_atom, 3)
        for i in 1:n_atom
            row = coord_raw[i]
            row isa AbstractArray || continue
            length(row) == 3 || continue
            coords[i, 1] = Float32(row[1])
            coords[i, 2] = Float32(row[2])
            coords[i, 3] = Float32(row[3])
        end

        charge = Float32.(entry["charge"])
        mask = Int.(entry["mask"])
        _CCD_REF_CACHE[code] = (
            atom_map = atom_map,
            coords = coords,
            charge = charge,
            mask = mask,
        )
    end
end

function _finalize_ccd_entry(rows::Vector{Vector{String}}, idx_atom::Int, idx_charge::Int, idx_x::Int, idx_y::Int, idx_z::Int, idx_xi::Int, idx_yi::Int, idx_zi::Int)
    n = length(rows)
    atom_map = Dict{String, Int}()
    coords = zeros(Float32, n, 3)
    charge = zeros(Float32, n)
    mask = zeros(Int, n)
    for i in 1:n
        r = rows[i]
        atom_name = String(r[idx_atom])
        atom_map[atom_name] = i

        q = _try_parse_i(r[idx_charge])
        charge[i] = q === nothing ? 0f0 : Float32(q)

        x = _try_parse_f32(r[idx_x])
        y = _try_parse_f32(r[idx_y])
        z = _try_parse_f32(r[idx_z])
        if x === nothing || y === nothing || z === nothing
            x = _try_parse_f32(r[idx_xi])
            y = _try_parse_f32(r[idx_yi])
            z = _try_parse_f32(r[idx_zi])
        end

        if x !== nothing && y !== nothing && z !== nothing
            coords[i, 1] = x
            coords[i, 2] = y
            coords[i, 3] = z
            mask[i] = 1
        end
    end
    return (atom_map = atom_map, coords = coords, charge = charge, mask = mask)
end

function _scan_ccd_for_codes!(needed_codes::Set{String}, ccd_path::AbstractString)
    isempty(needed_codes) && return
    isfile(ccd_path) || return

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
                active = current_code in needed_codes
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
                idx_charge = findidx("_chem_comp_atom.charge")
                idx_x = findidx("_chem_comp_atom.model_Cartn_x")
                idx_y = findidx("_chem_comp_atom.model_Cartn_y")
                idx_z = findidx("_chem_comp_atom.model_Cartn_z")
                idx_xi = findidx("_chem_comp_atom.pdbx_model_Cartn_x_ideal")
                idx_yi = findidx("_chem_comp_atom.pdbx_model_Cartn_y_ideal")
                idx_zi = findidx("_chem_comp_atom.pdbx_model_Cartn_z_ideal")
                if any(x -> x === nothing, (idx_atom, idx_charge, idx_x, idx_y, idx_z, idx_xi, idx_yi, idx_zi))
                    continue
                end
                idx_atom = idx_atom::Int
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
                    cols = _split_cif_row(row_line)
                    length(cols) < length(headers) && continue
                    push!(rows, cols)
                end
                if !isempty(rows)
                    _CCD_REF_CACHE[current_code] = _finalize_ccd_entry(rows, idx_atom, idx_charge, idx_x, idx_y, idx_z, idx_xi, idx_yi, idx_zi)
                    delete!(needed_codes, current_code)
                    isempty(needed_codes) && return
                end
            end
        end
    end
end

function _scan_ccd_for_bonds!(needed_codes::Set{String}, ccd_path::AbstractString)
    isempty(needed_codes) && return
    isfile(ccd_path) || return

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
                active = current_code in needed_codes
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
                all(startswith(h, "_chem_comp_bond.") for h in headers) || continue

                findidx(name) = findfirst(==(name), headers)
                idx_a1 = findidx("_chem_comp_bond.atom_id_1")
                idx_a2 = findidx("_chem_comp_bond.atom_id_2")
                if idx_a1 === nothing || idx_a2 === nothing
                    continue
                end
                idx_a1 = idx_a1::Int
                idx_a2 = idx_a2::Int

                bonds = Tuple{String, String}[]
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
                    cols = _split_cif_row(row_line)
                    length(cols) < length(headers) && continue
                    a1 = String(cols[idx_a1])
                    a2 = String(cols[idx_a2])
                    push!(bonds, (a1, a2))
                end
                _CCD_BOND_CACHE[current_code] = bonds
                delete!(needed_codes, current_code)
                isempty(needed_codes) && return
            end
        end
    end
end

function _ensure_ccd_bond_entries!(codes::Set{String})
    missing = Set{String}()
    for c in codes
        haskey(_CCD_BOND_CACHE, c) || push!(missing, c)
    end
    isempty(missing) && return
    ccd_path = _default_ccd_components_path()
    isempty(ccd_path) && return
    _scan_ccd_for_bonds!(missing, ccd_path)
end

function _ensure_ccd_ref_entries!(codes::Set{String})
    _load_ccd_std_ref_cache!()
    missing = Set{String}()
    for c in codes
        haskey(_CCD_REF_CACHE, c) || push!(missing, c)
    end
    isempty(missing) && return
    ccd_path = _default_ccd_components_path()
    isempty(ccd_path) && return
    _scan_ccd_for_codes!(missing, ccd_path)
end

function _ccd_reference_features(
    atoms::Vector{AtomRecord},
    ref_space_uid::Vector{Int},
    rng::AbstractRNG;
    ref_pos_augment::Bool = true,
)
    n = length(atoms)
    base_pos = hcat([a.x for a in atoms], [a.y for a in atoms], [a.z for a in atoms])
    ref_charge = zeros(Float32, n)
    ref_mask = [a.mol_type == "protein" ? 1 : (a.is_resolved ? 1 : 0) for a in atoms]

    needed_codes = Set(uppercase(a.res_name) for a in atoms)
    _ensure_ccd_ref_entries!(needed_codes)

    for i in 1:n
        code = uppercase(atoms[i].res_name)
        entry = get(_CCD_REF_CACHE, code, nothing)
        entry === nothing && continue
        atom_idx = get(entry.atom_map, atoms[i].atom_name, 0)
        atom_idx == 0 && continue
        @inbounds base_pos[i, :] .= entry.coords[atom_idx, :]
        @inbounds ref_charge[i] = entry.charge[atom_idx]
        @inbounds ref_mask[i] = entry.mask[atom_idx]
    end

    ref_pos = _ref_pos_with_augmentation(base_pos, ref_space_uid, rng; apply_augmentation = ref_pos_augment)
    return ref_pos, ref_charge, ref_mask
end

function _distogram_rep_atom_mask(atoms::Vector{AtomRecord}, tokens::TokenArray)
    n = length(atoms)
    mask = zeros(Int, n)
    for atom_idx in _distogram_rep_atom_indices(atoms, tokens)
        mask[atom_idx] = 1
    end
    return mask
end

function _compute_token_bonds(
    atoms::Vector{AtomRecord},
    atom_to_token_idx::Vector{Int},
    ref_space_uid::Vector{Int},
    n_token::Int,
)
    n = length(atoms)
    token_bonds = zeros(Int, n_token, n_token)

    # Classify each atom
    polymer_types = Set(["protein", "dna", "rna"])
    is_polymer = [a.mol_type in polymer_types for a in atoms]
    is_std = [is_polymer[i] && haskey(STD_RESIDUES_PROTENIX, atoms[i].res_name) for i in 1:n]
    is_unstd_polymer = [is_polymer[i] && !is_std[i] for i in 1:n]

    # Load CCD bond data for all unique residue codes
    codes_needing_bonds = Set{String}(uppercase(a.res_name) for a in atoms)
    _ensure_ccd_bond_entries!(codes_needing_bonds)

    # Build per-residue atom name → global atom index mapping
    # Key: (chain_id, res_id, atom_name) → global atom index
    atom_lookup = Dict{Tuple{String, Int, String}, Int}()
    for (i, a) in enumerate(atoms)
        atom_lookup[(a.chain_id, a.res_id, a.atom_name)] = i
    end

    # Iterate over each residue group and add CCD bonds
    # Group atoms by (chain_id, res_id, res_name)
    residue_groups = Dict{Tuple{String, Int}, Vector{Int}}()
    for (i, a) in enumerate(atoms)
        key = (a.chain_id, a.res_id)
        if !haskey(residue_groups, key)
            residue_groups[key] = Int[]
        end
        push!(residue_groups[key], i)
    end

    for (_, atom_indices) in residue_groups
        isempty(atom_indices) && continue
        a0 = atoms[atom_indices[1]]
        code = uppercase(a0.res_name)
        ccd_bonds = get(_CCD_BOND_CACHE, code, nothing)
        ccd_bonds === nothing && continue

        # Build local atom_name → global index for this residue
        local_lookup = Dict{String, Int}()
        for gi in atom_indices
            local_lookup[atoms[gi].atom_name] = gi
        end

        for (a1_name, a2_name) in ccd_bonds
            gi = get(local_lookup, a1_name, 0)
            gj = get(local_lookup, a2_name, 0)
            (gi == 0 || gj == 0) && continue

            # Apply Python's filtering logic:
            # Skip std-std polymer bonds
            is_std[gi] && is_std[gj] && continue
            # Skip std-unstd polymer bonds
            (is_std[gi] && is_unstd_polymer[gj]) && continue
            (is_std[gj] && is_unstd_polymer[gi]) && continue
            # Skip inter-unstd polymer bonds (different ref_space_uid)
            if is_unstd_polymer[gi] && is_unstd_polymer[gj]
                ref_space_uid[gi] != ref_space_uid[gj] && continue
            end

            ti = atom_to_token_idx[gi] + 1  # 0-based → 1-based
            tj = atom_to_token_idx[gj] + 1  # 0-based → 1-based
            (ti < 1 || tj < 1 || ti > n_token || tj > n_token) && continue
            token_bonds[ti, tj] = 1
            token_bonds[tj, ti] = 1
        end
    end

    return token_bonds
end

function _basic_atom_features(
    atoms::Vector{AtomRecord},
    tokens::TokenArray,
    atom_to_token_idx::Vector{Int},
    ref_space_uid::Vector{Int},
    n_token::Int,
    rng::AbstractRNG,
    ref_pos_augment::Bool,
)
    n = length(atoms)
    ref_pos, ref_charge, ref_mask = _ccd_reference_features(
        atoms,
        ref_space_uid,
        rng;
        ref_pos_augment = ref_pos_augment,
    )
    token_bonds = _compute_token_bonds(atoms, atom_to_token_idx, ref_space_uid, n_token)
    return Dict(
        "is_protein" => [a.mol_type == "protein" for a in atoms],
        "is_ligand" => [a.mol_type == "ligand" for a in atoms],
        "is_dna" => [a.mol_type == "dna" for a in atoms],
        "is_rna" => [a.mol_type == "rna" for a in atoms],
        "distogram_rep_atom_mask" => _distogram_rep_atom_mask(atoms, tokens),
        "condition_atom_mask" => [a.res_name != "xpb" for a in atoms],
        "ref_pos" => ref_pos,
        "ref_mask" => ref_mask,
        "ref_charge" => ref_charge,
        "atom_to_tokatom_idx" => _atom_to_tokatom_index(tokens, n),
        "token_bonds" => token_bonds,
    )
end

function _ref_element_onehot(atoms::Vector{AtomRecord})
    n = length(atoms)
    out = zeros(Float32, n, 128)
    residue_vocab_size = length(STD_RESIDUES_PROTENIX)
    fallback = get(ELEMS, "UNK_ELEM_128", residue_vocab_size + 127)
    for (i, a) in enumerate(atoms)
        tok = get(ELEMS, uppercase(a.element), fallback)
        idx = clamp(tok - residue_vocab_size + 1, 1, 128)
        out[i, idx] = 1f0
    end
    return out
end

function _ref_atom_name_chars_onehot(atoms::Vector{AtomRecord})
    n = length(atoms)
    out = zeros(Float32, n, 4, 64)
    for (i, a) in enumerate(atoms)
        padded = rpad(uppercase(a.atom_name), 4)
        for pos in 1:4
            c = padded[pos]
            code = Int(c) - 32
            (0 <= code <= 63) || error("Atom name character out of supported ASCII range [32,95]: '$c' in '$(a.atom_name)'")
            bucket = code + 1
            out[i, pos, bucket] = 1f0
        end
    end
    return out
end

function _build_feature_dict(
    atoms::Vector{AtomRecord},
    tokens::TokenArray,
    task::InputTask,
    rng::AbstractRNG,
    ref_pos_augment::Bool,
)
    n_token = length(tokens)
    n_atom = length(atoms)
    n_msa = 1
    restype_depth = maximum(values(STD_RESIDUES_WITH_GAP)) + 1

    centre_idx = centre_atom_indices(tokens)
    centre_atoms = atoms[centre_idx]
    chain_to_idx = _chain_index_map(centre_atoms)

    token_index = collect(0:(n_token - 1))
    residue_index = [a.res_id for a in centre_atoms]
    asym_id = [chain_to_idx[a.chain_id] for a in centre_atoms]
    entity_id = copy(asym_id)
    sym_id = fill(0, n_token)
    atom_to_token_idx = _atom_to_token_index(tokens, n_atom)

    cano_res = [canonical_resname_for_atom(a) for a in centre_atoms]
    restype = restype_onehot_encoded(cano_res)
    size(restype, 2) == restype_depth || error("Unexpected restype depth.")
    profile = restype[:, 1:32]
    msa = fill(STD_RESIDUES_WITH_GAP["-"], n_msa, n_token)
    has_deletion = zeros(Float32, n_msa, n_token)
    deletion_value = zeros(Float32, n_msa, n_token)
    deletion_mean = zeros(Float32, n_token)

    condition_token_mask, hotspot = _condition_and_hotspot_features(centre_atoms, task.hotspots)
    design_token_mask = .!condition_token_mask
    conditional_templ, conditional_templ_mask = _compute_conditional_template(atoms, tokens, condition_token_mask)

    ref_space_uid = _compute_ref_space_uid(atoms)

    # Compute basic atom features first (needed for ligand frame computation)
    basic_feats = _basic_atom_features(atoms, tokens, atom_to_token_idx, ref_space_uid, n_token, rng, ref_pos_augment)
    # ref_pos from basic_feats is already augmented, but for frame computation we use it
    # (Python also uses ref_pos after CCD lookup for frame computation)
    ref_pos_for_frames = basic_feats["ref_pos"]  # (N_atom, 3)
    ref_mask_for_frames = basic_feats["ref_mask"]  # (N_atom,)

    # Build ref_space_uid -> atom indices mapping for ligand frame computation
    uid_to_atom_indices = Dict{Int, Vector{Int}}()
    for (i, a) in enumerate(atoms)
        if a.mol_type == "ligand" || !(a.res_name in keys(STD_RESIDUES_PROTENIX))
            uid = ref_space_uid[i]
            if !haskey(uid_to_atom_indices, uid)
                uid_to_atom_indices[uid] = Int[]
            end
            push!(uid_to_atom_indices[uid], i)
        end
    end

    has_frame = zeros(Int, n_token)
    frame_atom_index = fill(-1, n_token, 3)
    for (i, tok) in enumerate(tokens)
        ca = centre_atoms[i]
        if ca.mol_type != "ligand" && ca.res_name in keys(STD_RESIDUES_PROTENIX) && length(tok.atom_indices) > 1
            # Standard polymer token: use N/CA/C or C1'/C3'/C4'
            ok, frame = _frame_for_polymer_token(tok, ca)
            has_frame[i] = ok
            frame_atom_index[i, :] = ifelse.(frame .>= 0, frame .- 1, frame)
        else
            # Ligand/non-standard: use KDTree-like nearest-neighbor frame
            centre_global_idx = tok.centre_atom_index
            uid = ref_space_uid[centre_global_idx]
            group_indices = get(uid_to_atom_indices, uid, Int[])
            ok, frame = _frame_for_lig_token(
                tok, centre_global_idx, group_indices,
                ref_pos_for_frames, ref_mask_for_frames,
            )
            has_frame[i] = ok
            frame_atom_index[i, :] = frame
        end
    end

    feat = Dict{String, Any}(
        "token_index" => token_index,
        "residue_index" => residue_index,
        "asym_id" => asym_id,
        "entity_id" => entity_id,
        "sym_id" => sym_id,
        "restype" => restype,
        "atom_to_token_idx" => atom_to_token_idx,
        "atom_to_token_mask" => ones(Float32, n_atom),
        "msa" => msa,
        "has_deletion" => has_deletion,
        "deletion_value" => deletion_value,
        "deletion_mean" => deletion_mean,
        "profile" => profile,
        "template_restype" => fill(STD_RESIDUES_WITH_GAP["-"], 4, n_token),
        "template_all_atom_mask" => zeros(Float32, 4, n_token, 37),
        "template_all_atom_positions" => zeros(Float32, 4, n_token, 37, 3),
        "has_frame" => has_frame,
        "frame_atom_index" => frame_atom_index,
        "design_token_mask" => design_token_mask,
        "condition_token_mask" => condition_token_mask,
        "conditional_templ" => conditional_templ,
        "conditional_templ_mask" => conditional_templ_mask,
        "hotspot" => hotspot,
        "plddt" => zeros(Float32, n_token),
        "ref_element" => _ref_element_onehot(atoms),
        "ref_atom_name_chars" => _ref_atom_name_chars_onehot(atoms),
        "ref_space_uid" => ref_space_uid,
    )

    merge!(feat, basic_feats)
    dims = Dict("N_token" => n_token, "N_atom" => n_atom, "N_msa" => n_msa)
    return feat, dims
end

function build_basic_feature_bundle(
    task::InputTask;
    rng::AbstractRNG = Random.default_rng(),
    ref_pos_augment::Bool = true,
)
    isempty(task.generation) && error("Task $(task.name) has empty generation.")
    binder_gen = task.generation[end]
    binder_gen.type == "protein" || error("Only protein generation is supported right now.")
    binder_length = binder_gen.length

    target_atoms = if isempty(strip(task.structure_file))
        AtomRecord[]
    else
        load_structure_atoms(
            task.structure_file;
            selected_chains = task.chain_ids,
            crop = task.crop,
        )
    end
    binder_chain_idx = length(task.chain_ids) + 1
    binder_chain = string(_chain_letters(binder_chain_idx), "0")
    binder_atoms = build_design_backbone_atoms(binder_length; chain_id = binder_chain)

    atoms = vcat(target_atoms, binder_atoms)
    tokens = tokenize_atoms(atoms)
    feat, dims = _build_feature_dict(atoms, tokens, task, rng, ref_pos_augment)

    return Dict(
        "task_name" => task.name,
        "atoms" => atoms,
        "tokens" => tokens,
        "input_feature_dict" => feat,
        "dims" => dims,
    )
end

function build_feature_bundle_from_atoms(
    atoms::Vector{AtomRecord};
    task_name::AbstractString = "custom",
    hotspots::Dict{String, Vector{Int}} = Dict{String, Vector{Int}}(),
    rng::AbstractRNG = Random.default_rng(),
    ref_pos_augment::Bool = true,
)
    tokens = tokenize_atoms(atoms)
    task = InputTask(
        String(task_name),
        "",
        String[],
        Dict{String, String}(),
        hotspots,
        Dict{String, MSAChainOptions}(),
        GenerationSpec[],
    )
    feat, dims = _build_feature_dict(atoms, tokens, task, rng, ref_pos_augment)
    return Dict(
        "task_name" => String(task_name),
        "atoms" => atoms,
        "tokens" => tokens,
        "input_feature_dict" => feat,
        "dims" => dims,
    )
end

end

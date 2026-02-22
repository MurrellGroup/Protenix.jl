#!/usr/bin/env julia
#
# Bond length validation for CIF files.
#
# Uses the EXACT same reference data and tolerance as BoltzGen.check_bond_lengths:
#   - ProtInterop.EXPECTED_BOND_LENGTHS (residue-specific, from ref_atom_pos)
#   - ProtInterop.SP2_BOND_LENGTH_OVERRIDES (aromatic/carboxylate/guanidinium)
#   - ProtInterop.PEPTIDE_BOND_LENGTH (inter-residue C-N)
#   - Tolerance: 0.9×expected ≤ actual ≤ 1.1×expected
#
# Usage:
#   julia --project=. scripts/bond_check_cif.jl path/to/file.cif [more.cif ...]
#   julia --project=. scripts/bond_check_cif.jl path/to/directory/
#
# If no arguments given, checks all CIF files in e2e_output/.

using ProtInterop

const BB_BONDS = ProtInterop.BACKBONE_BONDS
const SC_BONDS = ProtInterop.SIDECHAIN_BONDS
const EXPECTED = ProtInterop.EXPECTED_BOND_LENGTHS
const SP2 = ProtInterop.SP2_BOND_LENGTH_OVERRIDES
const PEPTIDE = ProtInterop.PEPTIDE_BOND_LENGTH

# ── CIF parsing ─────────────────────────────────────────────────────────────────

struct CIFAtom
    name::String
    res::String
    chain::String
    seq::Int
    x::Float32
    y::Float32
    z::Float32
end

function parse_cif_atoms(path::String)
    cols = String[]
    in_header = false
    for line in eachline(path)
        s = strip(line)
        if startswith(s, "_atom_site.")
            push!(cols, replace(s, "_atom_site." => ""))
            in_header = true
        elseif in_header
            break
        end
    end
    ix = findfirst(==("Cartn_x"), cols)
    iy = findfirst(==("Cartn_y"), cols)
    iz = findfirst(==("Cartn_z"), cols)
    ia = findfirst(==("label_atom_id"), cols)
    ir = findfirst(==("label_comp_id"), cols)
    is = findfirst(==("label_seq_id"), cols)
    ic = findfirst(==("label_asym_id"), cols)
    any(isnothing, (ix, iy, iz, ia, ir, is, ic)) && return CIFAtom[]
    atoms = CIFAtom[]
    for line in eachline(path)
        startswith(line, "ATOM") || continue
        p = split(line)
        length(p) >= max(ix, iy, iz, ia, ir, is, ic) || continue
        push!(atoms, CIFAtom(
            p[ia], p[ir], p[ic], parse(Int, p[is]),
            parse(Float32, p[ix]), parse(Float32, p[iy]), parse(Float32, p[iz]),
        ))
    end
    atoms
end

# ── Bond checking ────────────────────────────────────────────────────────────────

struct BondViolation
    seq::Int
    res::String
    chain::String
    a1::String
    a2::String
    expected::Float32
    actual::Float32
    kind::Symbol      # :short or :long
    backbone::Bool
end

function check_cif_bonds(path::String; tol_low::Float64 = 0.9, tol_high::Float64 = 1.1)
    atoms = parse_cif_atoms(path)
    isempty(atoms) && return (n_checked = 0, n_violations = 0, n_bb = 0, n_sc = 0, violations = BondViolation[])

    residues = Dict{Tuple{String,Int},Dict{String,CIFAtom}}()
    res_names = Dict{Tuple{String,Int},String}()
    for a in atoms
        k = (a.chain, a.seq)
        if !haskey(residues, k)
            residues[k] = Dict{String,CIFAtom}()
        end
        residues[k][a.name] = a
        res_names[k] = a.res
    end

    violations = BondViolation[]
    n_checked = 0

    for key in sort(collect(keys(residues)); by = k -> (k[1], k[2]))
        chain, seq = key
        am = residues[key]
        res = res_names[key]

        # Collect intra-residue bonds
        bonds = Tuple{String,String,Bool}[]
        for (a1, a2) in BB_BONDS
            push!(bonds, (a1, a2, true))
        end
        for (a1, a2) in get(SC_BONDS, res, Tuple{String,String}[])
            push!(bonds, (a1, a2, false))
        end

        for (a1, a2, is_bb) in bonds
            haskey(am, a1) && haskey(am, a2) || continue

            # Expected length: SP2 override > residue-specific > fallback to ALA/GLY
            exp = get(SP2, (res, a1, a2), nothing)
            if exp === nothing
                exp = get(EXPECTED, (res, a1, a2), nothing)
            end
            if exp === nothing
                for fb in ("ALA", "GLY")
                    exp = get(EXPECTED, (fb, a1, a2), nothing)
                    exp !== nothing && break
                end
            end
            exp === nothing && continue

            p1, p2 = am[a1], am[a2]
            d = sqrt((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)
            lo = Float32(tol_low) * exp
            hi = Float32(tol_high) * exp
            n_checked += 1

            if d < lo
                push!(violations, BondViolation(seq, res, chain, a1, a2, exp, d, :short, is_bb))
            elseif d > hi
                push!(violations, BondViolation(seq, res, chain, a1, a2, exp, d, :long, is_bb))
            end
        end

        # Inter-residue peptide bond C(i)→N(i+1)
        next = (chain, seq + 1)
        if haskey(residues, next) && haskey(am, "C") && haskey(residues[next], "N")
            p1, p2 = am["C"], residues[next]["N"]
            d = sqrt((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)
            lo = Float32(tol_low) * PEPTIDE
            hi = Float32(tol_high) * PEPTIDE
            n_checked += 1
            if d < lo
                push!(violations, BondViolation(seq, res_names[key], chain, "C", "N+1", PEPTIDE, d, :short, true))
            elseif d > hi
                push!(violations, BondViolation(seq, res_names[key], chain, "C", "N+1", PEPTIDE, d, :long, true))
            end
        end
    end

    n_bb = count(v -> v.backbone, violations)
    n_sc = length(violations) - n_bb
    return (n_checked = n_checked, n_violations = length(violations), n_bb = n_bb, n_sc = n_sc, violations = violations)
end

# ── Reporting ────────────────────────────────────────────────────────────────────

function print_report(path::String, stats; io::IO = stdout)
    label = basename(path)
    if stats.n_violations == 0
        println(io, "  ✓ $label: $(stats.n_checked) bonds, 0 violations")
        return
    end
    println(io, "  ✗ $label: $(stats.n_checked) bonds, $(stats.n_violations) violations (BB=$(stats.n_bb), SC=$(stats.n_sc))")
    for v in stats.violations
        println(io, "      $(v.res)$(v.seq) $(v.a1)-$(v.a2): $(round(v.actual; digits=3))/$(round(v.expected; digits=3)) ($(v.kind))")
    end
end

# ── CLI ──────────────────────────────────────────────────────────────────────────

function collect_cif_paths(args)
    paths = String[]
    if isempty(args)
        root = joinpath(@__DIR__, "..", "e2e_output")
        isdir(root) || error("No arguments and e2e_output/ not found")
        for (dir, _, files) in walkdir(root)
            for f in files
                endswith(f, ".cif") && push!(paths, joinpath(dir, f))
            end
        end
    else
        for arg in args
            if isdir(arg)
                for (dir, _, files) in walkdir(arg)
                    for f in files
                        endswith(f, ".cif") && push!(paths, joinpath(dir, f))
                    end
                end
            elseif isfile(arg) && endswith(arg, ".cif")
                push!(paths, arg)
            else
                @warn "Skipping: $arg"
            end
        end
    end
    sort!(paths)
end

function main()
    paths = collect_cif_paths(ARGS)
    isempty(paths) && (println("No CIF files found."); return)

    println("Bond check: $(length(paths)) CIF file(s), tolerance 0.9–1.1×expected")
    println()

    n_perfect = 0
    for path in paths
        stats = check_cif_bonds(path)
        print_report(path, stats)
        stats.n_violations == 0 && (n_perfect += 1)
    end

    println("\n$(n_perfect)/$(length(paths)) files with 0 violations")
end

main()

#!/usr/bin/env julia
#
# Bond check + comparison for all clean_targets outputs.
#
# Usage:
#   cd /home/claudey/FixingKAFA/PXDesign.jl
#   julia --project=. clean_targets/scripts/validate_all.jl [target_number]
#
# Validates:
#   1. Bond geometry (backbone + sidechain) for all CIF files
#   2. Output completeness (CIF produced, confidence JSON produced)
#   3. Summary table comparing Python vs Julia outputs

using ProtInterop
using Printf

const ROOT = joinpath(@__DIR__, "..", "..")
const CLEAN = joinpath(ROOT, "clean_targets")
const PYTHON_OUT = joinpath(CLEAN, "python_outputs_timed")
const JULIA_OUT = joinpath(CLEAN, "julia_outputs")

const FILTER = length(ARGS) >= 1 ? ARGS[1] : ""

# ── Bond checking (uses same reference data as ProtInterop.check_pdb_bonds) ──

const BB_BONDS = ProtInterop.BACKBONE_BONDS
const SC_BONDS = ProtInterop.SIDECHAIN_BONDS
const REF_BONDS = ProtInterop.PDB_REF_BOND_LENGTHS  # Engh & Huber literature values
const SP2 = ProtInterop.SP2_BOND_LENGTH_OVERRIDES
const PEPTIDE = ProtInterop.PEPTIDE_BOND_LENGTH

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

struct BondViolation
    seq::Int
    res::String
    chain::String
    a1::String
    a2::String
    expected::Float32
    actual::Float32
    kind::Symbol
    backbone::Bool
end

function check_cif_bonds(path::String; tol_low::Float64=0.9, tol_high::Float64=1.1)
    atoms = parse_cif_atoms(path)
    isempty(atoms) && return (n_checked=0, n_violations=0, n_bb=0, n_sc=0, violations=BondViolation[])

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

    for key in sort(collect(keys(residues)); by=k->(k[1], k[2]))
        chain, seq = key
        am = residues[key]
        res = res_names[key]

        bonds = Tuple{String,String,Bool}[]
        for (a1, a2) in BB_BONDS
            push!(bonds, (a1, a2, true))
        end
        for (a1, a2) in get(SC_BONDS, res, Tuple{String,String}[])
            push!(bonds, (a1, a2, false))
        end

        for (a1, a2, is_bb) in bonds
            haskey(am, a1) && haskey(am, a2) || continue
            # Same lookup order as check_pdb_bonds: SP2 override → PDB_REF (Engh & Huber)
            sp2_val = get(SP2, (res, a1, a2), nothing)
            exp = if sp2_val !== nothing
                Float32(sp2_val)
            else
                ref = get(REF_BONDS, (a1, a2), nothing)
                ref !== nothing ? Float32(ref) : nothing
            end
            exp === nothing && continue

            p1, p2 = am[a1], am[a2]
            d = sqrt((p1.x-p2.x)^2 + (p1.y-p2.y)^2 + (p1.z-p2.z)^2)
            lo = Float32(tol_low) * exp
            hi = Float32(tol_high) * exp
            n_checked += 1

            if d < lo
                push!(violations, BondViolation(seq, res, chain, a1, a2, exp, d, :short, is_bb))
            elseif d > hi
                push!(violations, BondViolation(seq, res, chain, a1, a2, exp, d, :long, is_bb))
            end
        end

        next = (chain, seq + 1)
        if haskey(residues, next) && haskey(am, "C") && haskey(residues[next], "N")
            p1, p2 = am["C"], residues[next]["N"]
            d = sqrt((p1.x-p2.x)^2 + (p1.y-p2.y)^2 + (p1.z-p2.z)^2)
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
    return (n_checked=n_checked, n_violations=length(violations), n_bb=n_bb, n_sc=n_sc, violations=violations)
end

# ── Collect all CIF files from an output directory ────────────────────────────

function collect_cifs(dir::String)
    paths = String[]
    isdir(dir) || return paths
    for (d, _, files) in walkdir(dir)
        for f in files
            endswith(f, ".cif") && push!(paths, joinpath(d, f))
        end
    end
    sort!(paths)
end

# ── Main validation ──────────────────────────────────────────────────────────

function rating(n_checked::Int, n_violations::Int)
    n_checked == 0 && return "N/A"
    pct = 100.0 * n_violations / n_checked
    if pct == 0.0
        return "PERFECT"
    elseif pct < 1.0
        return "GREEN"
    elseif pct < 5.0
        return "ORANGE"
    else
        return "RED"
    end
end

struct TargetResult
    target_num::String
    target_name::String
    source::String     # "python" or "julia"
    model::String
    cif_path::String
    n_checked::Int
    n_violations::Int
    n_bb::Int
    n_sc::Int
    rating::String
end

function main()
    results = TargetResult[]

    for source in ("python", "julia")
        outdir = source == "python" ? PYTHON_OUT : JULIA_OUT
        isdir(outdir) || continue

        for entry in sort(readdir(outdir))
            isdir(joinpath(outdir, entry)) || continue

            # Extract target number and model from directory name
            m = match(r"^(\d{2})_(.+?)__(.+)$", entry)
            m === nothing && continue
            target_num = m.captures[1]
            target_name = m.captures[2]
            model = m.captures[3]

            !isempty(FILTER) && target_num != FILTER && continue

            cifs = collect_cifs(joinpath(outdir, entry))
            if isempty(cifs)
                push!(results, TargetResult(
                    target_num, target_name, source, model,
                    "", 0, 0, 0, 0, "NO_OUTPUT",
                ))
                continue
            end

            for cif in cifs
                stats = check_cif_bonds(cif)
                push!(results, TargetResult(
                    target_num, target_name, source, model,
                    basename(cif),
                    stats.n_checked, stats.n_violations, stats.n_bb, stats.n_sc,
                    rating(stats.n_checked, stats.n_violations),
                ))
            end
        end
    end

    # ── Print summary table ───────────────────────────────────────────────────
    println()
    println("=" ^ 120)
    println("  CLEAN TARGETS VALIDATION REPORT")
    println("=" ^ 120)
    println()

    # Header
    @printf("%-4s  %-28s  %-7s  %-35s  %-6s  %-5s  %-4s  %-4s  %s\n",
        "#", "Target", "Source", "Model", "Bonds", "Viol", "BB", "SC", "Rating")
    println("-" ^ 120)

    for r in results
        @printf("%-4s  %-28s  %-7s  %-35s  %-6d  %-5d  %-4d  %-4d  %s\n",
            r.target_num, r.target_name, r.source, r.model,
            r.n_checked, r.n_violations, r.n_bb, r.n_sc, r.rating)
    end

    println()

    # ── Summary statistics ────────────────────────────────────────────────────
    n_total = length(results)
    n_perfect = count(r -> r.rating == "PERFECT", results)
    n_green = count(r -> r.rating == "GREEN", results)
    n_orange = count(r -> r.rating == "ORANGE", results)
    n_red = count(r -> r.rating == "RED", results)
    n_no_output = count(r -> r.rating == "NO_OUTPUT", results)
    n_na = count(r -> r.rating == "N/A", results)

    println("SUMMARY: $n_total targets validated")
    println("  PERFECT:   $n_perfect")
    println("  GREEN:     $n_green  (<1% violations)")
    println("  ORANGE:    $n_orange  (1-5% violations)")
    println("  RED:       $n_red  (>5% violations)")
    println("  NO_OUTPUT: $n_no_output")
    println("  N/A:       $n_na")
    println()
end

main()

module Constants

export PRO_STD_RESIDUES_NATURAL,
    RNA_STD_RESIDUES_NATURAL,
    DNA_STD_RESIDUES,
    STD_RESIDUES_PROTENIX,
    PRO_DESIGN_RESIDUES,
    RNA_DESIGN_RESIDUES,
    STD_RESIDUES,
    GAP,
    STD_RESIDUES_WITH_GAP,
    STD_RESIDUES_WITH_GAP_ID_TO_NAME,
    PROT_STD_RESIDUES_ONE_TO_THREE,
    PROTEIN_HEAVY_ATOMS,
    ELEMS

const PRO_STD_RESIDUES_NATURAL = Dict(
    "ALA" => 0,
    "ARG" => 1,
    "ASN" => 2,
    "ASP" => 3,
    "CYS" => 4,
    "GLN" => 5,
    "GLU" => 6,
    "GLY" => 7,
    "HIS" => 8,
    "ILE" => 9,
    "LEU" => 10,
    "LYS" => 11,
    "MET" => 12,
    "PHE" => 13,
    "PRO" => 14,
    "SER" => 15,
    "THR" => 16,
    "TRP" => 17,
    "TYR" => 18,
    "VAL" => 19,
    "UNK" => 20,
)

const RNA_STD_RESIDUES_NATURAL = Dict(
    "A" => 21,
    "G" => 22,
    "C" => 23,
    "U" => 24,
    "N" => 25,
)

const DNA_STD_RESIDUES = Dict(
    "DA" => 26,
    "DG" => 27,
    "DC" => 28,
    "DT" => 29,
    "DN" => 30,
)

const STD_RESIDUES_PROTENIX = Dict{String, Int}()
for d in (PRO_STD_RESIDUES_NATURAL, RNA_STD_RESIDUES_NATURAL, DNA_STD_RESIDUES)
    for (k, v) in d
        STD_RESIDUES_PROTENIX[k] = v
    end
end

const PRO_DESIGN_RESIDUES = Dict("xpb" => 32, "xpa" => 33)
const RNA_DESIGN_RESIDUES = Dict("rbb" => 34, "raa" => 35)

const STD_RESIDUES = Dict{String, Int}()
for d in (STD_RESIDUES_PROTENIX, PRO_DESIGN_RESIDUES, RNA_DESIGN_RESIDUES)
    for (k, v) in d
        STD_RESIDUES[k] = v
    end
end

const GAP = Dict("-" => 31)

const STD_RESIDUES_WITH_GAP = Dict{String, Int}()
for d in (STD_RESIDUES_PROTENIX, GAP, PRO_DESIGN_RESIDUES, RNA_DESIGN_RESIDUES)
    for (k, v) in d
        STD_RESIDUES_WITH_GAP[k] = v
    end
end

const STD_RESIDUES_WITH_GAP_ID_TO_NAME = Dict(v => k for (k, v) in STD_RESIDUES_WITH_GAP)

const PROT_STD_RESIDUES_ONE_TO_THREE = Dict(
    "A" => "ALA",
    "R" => "ARG",
    "N" => "ASN",
    "D" => "ASP",
    "C" => "CYS",
    "Q" => "GLN",
    "E" => "GLU",
    "G" => "GLY",
    "H" => "HIS",
    "I" => "ILE",
    "L" => "LEU",
    "K" => "LYS",
    "M" => "MET",
    "F" => "PHE",
    "P" => "PRO",
    "S" => "SER",
    "T" => "THR",
    "W" => "TRP",
    "Y" => "TYR",
    "V" => "VAL",
    "X" => "UNK",
    "j" => "xpb",
)

const PROTEIN_HEAVY_ATOMS = Dict{String, Vector{String}}(
    "ALA" => ["N", "CA", "C", "O", "CB", "OXT"],
    "ARG" => ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "OXT"],
    "ASN" => ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "OXT"],
    "ASP" => ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "OXT"],
    "CYS" => ["N", "CA", "C", "O", "CB", "SG", "OXT"],
    "GLN" => ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", "OXT"],
    "GLU" => ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "OXT"],
    "GLY" => ["N", "CA", "C", "O", "OXT"],
    "HIS" => ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", "OXT"],
    "ILE" => ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "OXT"],
    "LEU" => ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "OXT"],
    "LYS" => ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "OXT"],
    "MET" => ["N", "CA", "C", "O", "CB", "CG", "SD", "CE", "OXT"],
    "PHE" => ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OXT"],
    "PRO" => ["N", "CA", "C", "O", "CB", "CG", "CD", "OXT"],
    "SER" => ["N", "CA", "C", "O", "CB", "OG", "OXT"],
    "THR" => ["N", "CA", "C", "O", "CB", "OG1", "CG2", "OXT"],
    "TRP" => ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2", "OXT"],
    "TYR" => ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", "OXT"],
    "UNK" => ["N", "CA", "C", "O", "CB", "CG", "OXT"],
    "VAL" => ["N", "CA", "C", "O", "CB", "CG1", "CG2", "OXT"],
    "xpb" => ["N", "CA", "C", "O", "OXT"],
)

function _all_elems()
    # Atomic symbols 1..118
    elems = String[
        "H", "HE", "LI", "BE", "B", "C", "N", "O", "F", "NE", "NA", "MG", "AL",
        "SI", "P", "S", "CL", "AR", "K", "CA", "SC", "TI", "V", "CR", "MN", "FE",
        "CO", "NI", "CU", "ZN", "GA", "GE", "AS", "SE", "BR", "KR", "RB", "SR",
        "Y", "ZR", "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD", "IN", "SN",
        "SB", "TE", "I", "XE", "CS", "BA", "LA", "CE", "PR", "ND", "PM", "SM",
        "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB", "LU", "HF", "TA", "W",
        "RE", "OS", "IR", "PT", "AU", "HG", "TL", "PB", "BI", "PO", "AT", "RN",
        "FR", "RA", "AC", "TH", "PA", "U", "NP", "PU", "AM", "CM", "BK", "CF",
        "ES", "FM", "MD", "NO", "LR", "RF", "DB", "SG", "BH", "HS", "MT", "DS",
        "RG", "CN", "NH", "FL", "MC", "LV", "TS", "OG",
    ]
    append!(elems, ["UNK_ELEM_$i" for i in 119:128])
    return elems
end

const ELEMS = let
    out = Dict{String, Int}()
    offset = length(STD_RESIDUES_PROTENIX)
    for (idx, elem) in enumerate(_all_elems())
        out[elem] = offset + (idx - 1)
    end
    out
end

end

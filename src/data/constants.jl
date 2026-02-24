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
    ELEMS,
    RES_ATOMS_DICT

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

# Pre-defined atom name → 0-indexed position mapping per residue type.
# Matches Python Protenix RES_ATOMS_DICT in constants.py.
const RES_ATOMS_DICT = Dict{String, Dict{String, Int}}(
    # Protein residues
    "ALA" => Dict("N"=>0, "CA"=>1, "C"=>2, "O"=>3, "CB"=>4, "OXT"=>5),
    "ARG" => Dict("N"=>0, "CA"=>1, "C"=>2, "O"=>3, "CB"=>4, "CG"=>5, "CD"=>6, "NE"=>7, "CZ"=>8, "NH1"=>9, "NH2"=>10, "OXT"=>11),
    "ASN" => Dict("N"=>0, "CA"=>1, "C"=>2, "O"=>3, "CB"=>4, "CG"=>5, "OD1"=>6, "ND2"=>7, "OXT"=>8),
    "ASP" => Dict("N"=>0, "CA"=>1, "C"=>2, "O"=>3, "CB"=>4, "CG"=>5, "OD1"=>6, "OD2"=>7, "OXT"=>8),
    "CYS" => Dict("N"=>0, "CA"=>1, "C"=>2, "O"=>3, "CB"=>4, "SG"=>5, "OXT"=>6),
    "GLN" => Dict("N"=>0, "CA"=>1, "C"=>2, "O"=>3, "CB"=>4, "CG"=>5, "CD"=>6, "OE1"=>7, "NE2"=>8, "OXT"=>9),
    "GLU" => Dict("N"=>0, "CA"=>1, "C"=>2, "O"=>3, "CB"=>4, "CG"=>5, "CD"=>6, "OE1"=>7, "OE2"=>8, "OXT"=>9),
    "GLY" => Dict("N"=>0, "CA"=>1, "C"=>2, "O"=>3, "OXT"=>4),
    "HIS" => Dict("N"=>0, "CA"=>1, "C"=>2, "O"=>3, "CB"=>4, "CG"=>5, "ND1"=>6, "CD2"=>7, "CE1"=>8, "NE2"=>9, "OXT"=>10),
    "ILE" => Dict("N"=>0, "CA"=>1, "C"=>2, "O"=>3, "CB"=>4, "CG1"=>5, "CG2"=>6, "CD1"=>7, "OXT"=>8),
    "LEU" => Dict("N"=>0, "CA"=>1, "C"=>2, "O"=>3, "CB"=>4, "CG"=>5, "CD1"=>6, "CD2"=>7, "OXT"=>8),
    "LYS" => Dict("N"=>0, "CA"=>1, "C"=>2, "O"=>3, "CB"=>4, "CG"=>5, "CD"=>6, "CE"=>7, "NZ"=>8, "OXT"=>9),
    "MET" => Dict("N"=>0, "CA"=>1, "C"=>2, "O"=>3, "CB"=>4, "CG"=>5, "SD"=>6, "CE"=>7, "OXT"=>8),
    "PHE" => Dict("N"=>0, "CA"=>1, "C"=>2, "O"=>3, "CB"=>4, "CG"=>5, "CD1"=>6, "CD2"=>7, "CE1"=>8, "CE2"=>9, "CZ"=>10, "OXT"=>11),
    "PRO" => Dict("N"=>0, "CA"=>1, "C"=>2, "O"=>3, "CB"=>4, "CG"=>5, "CD"=>6, "OXT"=>7),
    "SER" => Dict("N"=>0, "CA"=>1, "C"=>2, "O"=>3, "CB"=>4, "OG"=>5, "OXT"=>6),
    "THR" => Dict("N"=>0, "CA"=>1, "C"=>2, "O"=>3, "CB"=>4, "OG1"=>5, "CG2"=>6, "OXT"=>7),
    "TRP" => Dict("N"=>0, "CA"=>1, "C"=>2, "O"=>3, "CB"=>4, "CG"=>5, "CD1"=>6, "CD2"=>7, "NE1"=>8, "CE2"=>9, "CE3"=>10, "CZ2"=>11, "CZ3"=>12, "CH2"=>13, "OXT"=>14),
    "TYR" => Dict("N"=>0, "CA"=>1, "C"=>2, "O"=>3, "CB"=>4, "CG"=>5, "CD1"=>6, "CD2"=>7, "CE1"=>8, "CE2"=>9, "CZ"=>10, "OH"=>11, "OXT"=>12),
    "VAL" => Dict("N"=>0, "CA"=>1, "C"=>2, "O"=>3, "CB"=>4, "CG1"=>5, "CG2"=>6, "OXT"=>7),
    "UNK" => Dict("N"=>0, "CA"=>1, "C"=>2, "O"=>3, "CB"=>4, "CG"=>5, "OXT"=>6),
    # PXDesign protein backbone residue
    "xpb" => Dict("N"=>0, "CA"=>1, "C"=>2, "O"=>3, "OXT"=>4),
    # DNA residues
    "DA" => Dict("OP3"=>0, "P"=>1, "OP1"=>2, "OP2"=>3, "O5'"=>4, "C5'"=>5, "C4'"=>6, "O4'"=>7, "C3'"=>8, "O3'"=>9, "C2'"=>10, "C1'"=>11, "N9"=>12, "C8"=>13, "N7"=>14, "C5"=>15, "C6"=>16, "N6"=>17, "N1"=>18, "C2"=>19, "N3"=>20, "C4"=>21),
    "DC" => Dict("OP3"=>0, "P"=>1, "OP1"=>2, "OP2"=>3, "O5'"=>4, "C5'"=>5, "C4'"=>6, "O4'"=>7, "C3'"=>8, "O3'"=>9, "C2'"=>10, "C1'"=>11, "N1"=>12, "C2"=>13, "O2"=>14, "N3"=>15, "C4"=>16, "N4"=>17, "C5"=>18, "C6"=>19),
    "DG" => Dict("OP3"=>0, "P"=>1, "OP1"=>2, "OP2"=>3, "O5'"=>4, "C5'"=>5, "C4'"=>6, "O4'"=>7, "C3'"=>8, "O3'"=>9, "C2'"=>10, "C1'"=>11, "N9"=>12, "C8"=>13, "N7"=>14, "C5"=>15, "C6"=>16, "O6"=>17, "N1"=>18, "C2"=>19, "N2"=>20, "N3"=>21, "C4"=>22),
    "DT" => Dict("OP3"=>0, "P"=>1, "OP1"=>2, "OP2"=>3, "O5'"=>4, "C5'"=>5, "C4'"=>6, "O4'"=>7, "C3'"=>8, "O3'"=>9, "C2'"=>10, "C1'"=>11, "N1"=>12, "C2"=>13, "O2"=>14, "N3"=>15, "C4"=>16, "O4"=>17, "C5"=>18, "C7"=>19, "C6"=>20),
    "DN" => Dict("OP3"=>0, "P"=>1, "OP1"=>2, "OP2"=>3, "O5'"=>4, "C5'"=>5, "C4'"=>6, "O4'"=>7, "C3'"=>8, "O3'"=>9, "C2'"=>10, "C1'"=>11),
    # RNA residues
    "A" => Dict("OP3"=>0, "P"=>1, "OP1"=>2, "OP2"=>3, "O5'"=>4, "C5'"=>5, "C4'"=>6, "O4'"=>7, "C3'"=>8, "O3'"=>9, "C2'"=>10, "O2'"=>11, "C1'"=>12, "N9"=>13, "C8"=>14, "N7"=>15, "C5"=>16, "C6"=>17, "N6"=>18, "N1"=>19, "C2"=>20, "N3"=>21, "C4"=>22),
    "C" => Dict("OP3"=>0, "P"=>1, "OP1"=>2, "OP2"=>3, "O5'"=>4, "C5'"=>5, "C4'"=>6, "O4'"=>7, "C3'"=>8, "O3'"=>9, "C2'"=>10, "O2'"=>11, "C1'"=>12, "N1"=>13, "C2"=>14, "O2"=>15, "N3"=>16, "C4"=>17, "N4"=>18, "C5"=>19, "C6"=>20),
    "G" => Dict("OP3"=>0, "P"=>1, "OP1"=>2, "OP2"=>3, "O5'"=>4, "C5'"=>5, "C4'"=>6, "O4'"=>7, "C3'"=>8, "O3'"=>9, "C2'"=>10, "O2'"=>11, "C1'"=>12, "N9"=>13, "C8"=>14, "N7"=>15, "C5"=>16, "C6"=>17, "O6"=>18, "N1"=>19, "C2"=>20, "N2"=>21, "N3"=>22, "C4"=>23),
    "U" => Dict("OP3"=>0, "P"=>1, "OP1"=>2, "OP2"=>3, "O5'"=>4, "C5'"=>5, "C4'"=>6, "O4'"=>7, "C3'"=>8, "O3'"=>9, "C2'"=>10, "O2'"=>11, "C1'"=>12, "N1"=>13, "C2"=>14, "O2"=>15, "N3"=>16, "C4"=>17, "O4"=>18, "C5"=>19, "C6"=>20),
    "N" => Dict("OP3"=>0, "P"=>1, "OP1"=>2, "OP2"=>3, "O5'"=>4, "C5'"=>5, "C4'"=>6, "O4'"=>7, "C3'"=>8, "O3'"=>9, "C2'"=>10, "O2'"=>11, "C1'"=>12),
)

end

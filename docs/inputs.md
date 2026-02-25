# Input Formats

Protenix.jl accepts inputs in three ways:
1. **REPL API** — Julia functions for building tasks programmatically
2. **JSON files** — Protenix-compatible JSON for folding/prediction
3. **YAML files** — PXDesign design task specifications

## REPL API (Programmatic)

### Building Folding Tasks

Use entity builder functions to construct a task Dict, then pass it to `fold()`:

```julia
using Protenix

# Simple protein
task = protenix_task(protein_chain("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"))

# Protein + ligand
task = protenix_task(
    protein_chain("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"),
    ligand("CCD_ATP"),
)

# Protein + DNA + ion
task = protenix_task(
    protein_chain("MGSSHHHHHHSSGLVPRGSH..."),
    dna_chain("TTTCGGTACGAAC"),
    dna_chain("GTTCGTACCGAAA"),
    ion("MG"; count=2),
)

h = load_protenix("protenix_v1"; gpu=true)
result = fold(h, task; seed=101)
```

### Entity Builders

#### `protein_chain(sequence; count=1, msa=nothing, modifications=nothing)`

Build a protein entity.

- `sequence`: Amino acid sequence (one-letter codes, e.g. `"MVLSPAD..."`)
- `count`: Number of copies (for homo-oligomers)
- `msa`: MSA specification (see [MSA](#msa-multiple-sequence-alignments) below)
- `modifications`: Vector of PTM specifications (see [Modifications](#post-translational-modifications) below)

#### `dna_chain(sequence; count=1, modifications=nothing)`

Build a DNA entity. Sequence uses single-letter codes: A, G, C, T, N.

#### `rna_chain(sequence; count=1, unpaired_msa=nothing, unpaired_msa_path=nothing, modifications=nothing)`

Build an RNA entity. Sequence uses single-letter codes: A, G, C, U, N.

- `unpaired_msa`: Inline A3M/FASTA text for unpaired MSA
- `unpaired_msa_path`: Path to `.a3m` file for unpaired MSA

#### `ligand(name; count=1)`

Build a ligand entity. The `name` can be:

- **CCD code**: `"CCD_ATP"`, `"CCD_HEM"`, `"CCD_NAG"`, etc.
- **Multi-component CCD**: `"CCD_NAG_NAG_BMA_MAN_NAG_GAL"` (glycan trees — components joined with underscores)
- **SMILES string**: Any valid SMILES, e.g. `"Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP...)"`
- **File reference**: `"FILE_path/to/compound.sdf"` (SDF format)

#### `ion(name; count=1)`

Build an ion entity by CCD code: `"MG"`, `"ZN"`, `"CA"`, `"FE"`, `"MN"`, etc.

#### `protenix_task(entities...; name=nothing, constraint=nothing, covalent_bonds=nothing, template_features=nothing)`

Assemble entities into a task Dict.

- `entities`: Any number of entity Dicts from the builders above
- `name`: Task name (auto-generated if not provided)
- `constraint`: Constraint specification (see [Constraints](#constraints) below)
- `covalent_bonds`: Vector of bond Dicts (see [Covalent Bonds](#covalent-bonds) below)
- `template_features`: Template structure features (see [Templates](#template-structures) below)

### Building Design Tasks

```julia
dh = load_pxdesign("pxdesign_v0.1.0"; gpu=true)

# Unconditional design (de novo)
result = design(dh; binder_length=60, seed=42)

# Conditional design with target structure
target = design_target("structures/1ubq.cif"; chains=["A"])
result = design(dh; binder_length=80, target=target, seed=42)

# With hotspots
result = design(dh;
    binder_length=80,
    target=target,
    hotspots=Dict("A" => [8, 44, 48, 63, 70]),
    seed=42,
)

# With cropping
target = design_target("structures/5o45.cif";
    chains=["A"],
    crop=Dict("A" => "1-116"),
)

# With MSA for target
target = design_target("structures/5o45.cif";
    chains=["A"],
    crop=Dict("A" => "1-116"),
    msa=Dict("A" => Dict("precomputed_msa_dir" => "/path/to/msa")),
)
```

#### `design_target(structure_file; chains=String[], crop=nothing, msa=nothing)`

Build a target specification for conditional design.

- `structure_file`: Path to mmCIF structure file
- `chains`: Vector of chain IDs to include from the structure
- `crop`: Dict mapping chain IDs to residue ranges, e.g. `Dict("A" => "1-116")`. Supports discontinuous crops: `Dict("A" => "1-60,100-129")`
- `msa`: Dict mapping chain IDs to MSA specs, e.g. `Dict("A" => Dict("precomputed_msa_dir" => "path/to/msa"))`

#### `design_task(; binder_length, target=nothing, hotspots=nothing, name=nothing)`

Construct a `Schema.InputTask` for the design pipeline.

- `binder_length`: Required. Length of the protein to design.
- `target`: Optional target from `design_target()`
- `hotspots`: Optional Dict mapping chain IDs to residue index vectors, e.g. `Dict("A" => [8, 44, 48])`
- `name`: Optional task name

---

## JSON Input Format (Folding)

JSON is the standard format for multi-entity folding tasks, compatible with the Python Protenix CLI.

### Basic Structure

```json
[{
  "name": "task_name",
  "sequences": [
    {"proteinChain": {"sequence": "MVLSPAD...", "count": 1}},
    {"ligand": {"ligand": "CCD_ATP", "count": 1}}
  ]
}]
```

The top level can be:
- An array of task objects: `[{...}, {...}]`
- A single task object: `{...}`
- A wrapper: `{"tasks": [{...}, {...}]}`

### Entity Types in JSON

#### Protein Chain

```json
{"proteinChain": {
  "sequence": "MKQLLED...",
  "count": 1
}}
```

With MSA:
```json
{"proteinChain": {
  "sequence": "MKQLLED...",
  "count": 1,
  "msa": {
    "precomputed_msa_dir": "path/to/msa_dir",
    "pairing_db": "uniref100"
  }
}}
```

With modifications (PTMs):
```json
{"proteinChain": {
  "sequence": "MVLSPAD...",
  "count": 1,
  "modifications": [
    {"ptmType": "CCD_HY3", "ptmPosition": 1},
    {"ptmType": "CCD_P1L", "ptmPosition": 5}
  ]
}}
```

#### DNA Chain

```json
{"dnaSequence": {"sequence": "TTTCGGTACGAAC", "count": 1}}
```

With modifications:
```json
{"dnaSequence": {
  "sequence": "TTTCGGTACGAAC",
  "count": 1,
  "modifications": [
    {"ptmType": "CCD_6OG", "ptmPosition": 5}
  ]
}}
```

#### RNA Chain

```json
{"rnaSequence": {"sequence": "GCAUUGGCAUUG", "count": 1}}
```

With inline MSA:
```json
{"rnaSequence": {
  "sequence": "GCAUUGGCAUUG",
  "count": 1,
  "unpaired_msa": ">query\nGCAUUGGCAUUG\n>hit1\nGCAUUGGCAUUG\n"
}}
```

With MSA from file:
```json
{"rnaSequence": {
  "sequence": "GCAUUGGCAUUG",
  "count": 1,
  "unpaired_msa_path": "path/to/rna_msa/non_pairing.a3m"
}}
```

#### Ligand (CCD Code)

```json
{"ligand": {"ligand": "CCD_ATP", "count": 1}}
```

Multi-component (glycan tree):
```json
{"ligand": {"ligand": "CCD_NAG_NAG_BMA_MAN_NAG_GAL", "count": 1}}
```

#### Ligand (SMILES)

```json
{"ligand": {"ligand": "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)...", "count": 1}}
```

#### Ligand (SDF File)

```json
{"ligand": {"ligand": "FILE_path/to/compound.sdf", "count": 1}}
```

#### Ion

```json
{"ion": {"ligand": "MG", "count": 2}}
```

### Covalent Bonds

Add a `covalent_bonds` array to the task object. Each bond specifies two atoms by entity index, residue/component position, and atom name:

```json
{
  "name": "protein_ligand_covalent",
  "sequences": [
    {"proteinChain": {"sequence": "MVLSPAD...", "count": 1}},
    {"ligand": {"ligand": "CCD_ATP", "count": 1}}
  ],
  "covalent_bonds": [
    {
      "entity1": "1",
      "position1": "25",
      "atom1": "SG",
      "entity2": "2",
      "position2": "1",
      "atom2": "PA"
    }
  ]
}
```

Field details:
- `entity1`, `entity2`: 1-based index into the `sequences` array (as strings)
- `position1`, `position2`: 1-based residue/component position within that entity (as strings)
- `atom1`, `atom2`: PDB atom names

Common bond types:
- **Disulfide-like**: protein Cys SG to ligand atom
- **N-glycosylation**: protein Asn ND2 to sugar C1
- **Glycan tree linkages**: sugar C1 to adjacent sugar O3/O4/O6

### Constraints

Constraints are supported by `protenix_base_constraint_v0.5.0`.

#### Pocket Constraint

Force a binder chain near specific target residues:

```json
{
  "constraint": {
    "pocket": {
      "binder_chain": ["2", 1],
      "contact_residues": [["1", 1, 69]],
      "max_distance": 8
    }
  }
}
```

#### Token (Residue) Contact Constraint

Force two residues within a distance:

```json
{
  "constraint": {
    "contact": [
      {
        "residue1": ["1", 1, 72],
        "residue2": ["2", 1, 45],
        "max_distance": 15
      }
    ]
  }
}
```

#### Atom Contact Constraint

Force two specific atoms within a distance:

```json
{
  "constraint": {
    "contact": [
      {
        "atom1": ["1", 1, 50, "CA"],
        "atom2": ["2", 1, 1, "PA"],
        "max_distance": 8,
        "min_distance": 3
      }
    ]
  }
}
```

---

## YAML Input Format (Design)

YAML is the native format for PXDesign binder design tasks. YAML files are automatically converted to the internal JSON representation.

### Unconditional Design

Generate a de novo protein of specified length:

```yaml
binder_length: 60
```

### Conditional Design with Target

Design a binder against a target structure:

```yaml
target:
  file: "path/to/target.cif"
  chains:
    A:
      crop: ["1-116"]
      hotspots: [40, 99, 107]

binder_length: 80
```

### Multi-Chain Target

```yaml
target:
  file: "path/to/complex.cif"
  chains:
    A:
      crop: ["3-110"]
      hotspots: [27, 59, 73, 83, 87]
    D:
      crop: ["1-89"]
      hotspots: [35, 39, 76]

binder_length: 80
```

### Discontinuous Crops

```yaml
target:
  file: "path/to/target.cif"
  chains:
    A:
      crop: ["1-60", "100-129"]
      hotspots: [35, 52, 108, 119]

binder_length: 90
```

### Design with MSA

Provide precomputed MSA for target chains:

```yaml
target:
  file: "path/to/target.cif"
  chains:
    A:
      crop: ["1-116"]
      hotspots: [40, 99, 107]
      msa: "path/to/msa_dir"

binder_length: 80
```

The MSA directory should contain `pairing.a3m` and/or `non_pairing.a3m`.

### Design in JSON Format

Design tasks can also be specified in JSON:

```json
[{
  "name": "design_ubiquitin",
  "generation": [{"length": 50, "count": 1, "type": "protein"}],
  "condition": {
    "filter": {"crop": {}, "chain_id": ["A"]},
    "structure_file": "path/to/1ubq.cif",
    "msa": {}
  }
}]
```

With hotspots:
```json
[{
  "name": "design_with_hotspots",
  "hotspot": {"A": [8, 44, 48, 63, 70]},
  "generation": [{"length": 60, "count": 1, "type": "protein"}],
  "condition": {
    "filter": {"crop": {}, "chain_id": ["A"]},
    "structure_file": "path/to/1ubq.cif",
    "msa": {}
  }
}]
```

---

## MSA (Multiple Sequence Alignments)

### Precomputed MSA Directory

The standard way to provide MSA is via a directory containing A3M files:

```
msa_dir/
  non_pairing.a3m    # Required: unpaired MSA
  pairing.a3m        # Optional: paired MSA (for multi-chain tasks)
```

### A3M Format

A3M is a compact multiple sequence alignment format. It's an extension of FASTA where lowercase letters represent insertions (not aligned to the query):

```
>query
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH
>UniRef100_P69905/1-142
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDlSH
>UniRef100_Q9YHB3/1-143
MILSAEDKANVKAAWGKVGGHAAEYGAEALERMFLSFPTTKTYFPHFDlSH
```

### Pairing Database

When providing paired MSA, specify the database used for pairing:

```julia
protein_chain("MVLSPAD...";
    msa = Dict(
        "precomputed_msa_dir" => "/path/to/msa",
        "pairing_db" => "uniref100",
    ),
)
```

Supported values for `pairing_db`: `"uniref100"`, `"uniref90"`.

### Enabling MSA in Prediction

When using `predict_json`, MSA usage is controlled by the `use_msa` parameter:

```julia
records = predict_json("input.json"; use_msa=true, gpu=true)
```

When using `fold()` with a handle, MSA is enabled automatically if the task contains MSA data and the model supports it.

### RNA MSA

RNA MSA can be provided inline or via file path:

```json
{"rnaSequence": {
  "sequence": "GCAUUGGCAUUG",
  "unpaired_msa_path": "path/to/rna_msa/non_pairing.a3m"
}}
```

Or inline:
```json
{"rnaSequence": {
  "sequence": "GCAUUGGCAUUG",
  "unpaired_msa": ">query\nGCAUUGGCAUUG\n>hit1\nGCGUUGGCAUUG\n"
}}
```

---

## Template Structures

Template structures provide experimental structural context for folding. Only v1.0 models have an active template embedder.

### REPL API

```julia
tmpl = template_structure("path/to/template.cif", "A")
task = protenix_task(
    protein_chain("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQD..."),
    template_features = tmpl,
)
h = load_protenix("protenix_v1"; gpu=true)
result = fold(h, task; seed=101)
```

### JSON Format

Template features are injected automatically when using `protenix_mini_tmpl_v0.5.0` with appropriate input. For v1.0 models, template conditioning uses the internal template embedder.

---

## Post-Translational Modifications

Modified residues are specified via the `modifications` field on protein/DNA/RNA chains.

### Protein PTMs

```julia
protein_chain("MVLSPAD...";
    modifications = [
        Dict("ptmType" => "CCD_HY3", "ptmPosition" => 1),
        Dict("ptmType" => "CCD_P1L", "ptmPosition" => 5),
    ],
)
```

Common PTM codes:
- `CCD_SEP` — Phosphoserine
- `CCD_TPO` — Phosphothreonine
- `CCD_PTR` — Phosphotyrosine
- `CCD_MSE` — Selenomethionine
- `CCD_HY3` / `CCD_HYP` — Hydroxyproline
- `CCD_MLY` — N-dimethyl-lysine
- `CCD_CSO` — S-hydroxycysteine

### DNA Modifications

```json
{"dnaSequence": {
  "sequence": "TTTCGGTACGAAC",
  "modifications": [{"ptmType": "CCD_6OG", "ptmPosition": 5}]
}}
```

### RNA Modifications

```json
{"rnaSequence": {
  "sequence": "GCAUUGGCAUUG",
  "modifications": [{"ptmType": "CCD_PSU", "ptmPosition": 3}]
}}
```

---

## Output Format

### Folding Output

`fold()` returns a NamedTuple with:

| Field | Type | Description |
|-------|------|-------------|
| `coordinate` | `Array{Float32, 3}` | Predicted coordinates `(3, N_atom, N_sample)` |
| `cif` | `String` | mmCIF text of first sample |
| `cif_paths` | `Vector{String}` | Paths to all written CIF files |
| `prediction_dir` | `String` | Output directory |
| `plddt` | `Vector{Float32}` | Per-residue pLDDT (0-100) |
| `mean_plddt` | `Float32` | Average pLDDT |
| `pae` | `Vector{Float32}` | Per-residue PAE (Angstroms) |
| `mean_pae` | `Float32` | Average PAE |
| `pde` | confidence metric | Predicted distance error |
| `resolved` | confidence metric | Predicted experimentally-resolved mask |
| `seed` | `Int` | RNG seed used |
| `task_name` | `String` | Task identifier |

### Design Output

`design()` returns a NamedTuple with:

| Field | Type | Description |
|-------|------|-------------|
| `coordinate` | `Array{Float32, 3}` | Designed coordinates `(3, N_atom, N_sample)` |
| `cif_paths` | `Vector{String}` | Paths to CIF files |
| `prediction_dir` | `String` | Output directory |
| `seed` | `Int` | RNG seed used |
| `task_name` | `String` | Task identifier |
| `n_samples` | `Int` | Number of samples generated |
| `n_step` | `Int` | Diffusion steps used |

### CIF File Format

Output structures are written in mmCIF format, including:
- `_atom_site` records with coordinates, B-factors, and occupancy
- `_entity` and `_entity_poly` records for chain/entity classification
- `_struct_conn` records for covalent bonds (peptide, phosphodiester, cross-chain)
- `_entity_poly_seq` for polymer sequence records

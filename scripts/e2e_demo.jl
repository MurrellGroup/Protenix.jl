#!/usr/bin/env julia
#
# End-to-end demo: folding and design models on GPU with random weights.
#
# Usage: julia --project=<env> scripts/e2e_demo.jl
#
ENV["JULIA_DEBUG"] = ""  # suppress debug noise
using CUDA, cuDNN
using Flux
using Random
using PXDesign

# No allowscalar — scalar GPU indexing is forbidden

# Force line-buffered stdout so we see progress even under nohup/redirection.
flush(stdout)

const OUT_ROOT = joinpath(@__DIR__, "..", "e2e_output")
mkpath(OUT_ROOT)

println("CUDA: ", CUDA.functional(), "  GPU: ", CUDA.name(CUDA.device()))
println("Free GPU memory: ", round(CUDA.available_memory() / 1024^3, digits=1), " GB")
println(); flush(stdout)

# ══════════════════════════════════════════════════════════════════════════════
# 1. Folding model (ProtenixMini) — GPU
# ══════════════════════════════════════════════════════════════════════════════

println("=" ^ 60)
println("  FOLDING MODEL (ProtenixMini) — GPU")
println("=" ^ 60)

seq = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
println("Sequence ($(length(seq)) residues): $(seq[1:20])...")

mini_cpu = PXDesign.ProtenixMini.ProtenixMiniModel(
    32, 32, 16, 8, 97;
    c_atom = 16, c_atompair = 8,
    n_cycle = 1, pairformer_blocks = 1, msa_blocks = 1,
    diffusion_transformer_blocks = 2,
    diffusion_atom_encoder_blocks = 1, diffusion_atom_decoder_blocks = 1,
    confidence_max_atoms_per_token = 20,
    sample_gamma0 = 0.0, sample_gamma_min = 1.0,
    sample_noise_scale_lambda = 1.003, sample_step_scale_eta = 1.5,
    sample_n_step = 20, sample_n_sample = 3,
    rng = MersenneTwister(2024),
)

println("Moving model to GPU...")
mini_gpu = gpu(mini_cpu)
GC.gc(); CUDA.reclaim()
println("  GPU memory after model load: ", round(CUDA.available_memory() / 1024^3, digits=1), " GB free")

flush(stdout)
println("Running fold_sequence (20 steps, 3 samples) on GPU...")
t0 = time()
folded = PXDesign.ProtenixMini.fold_sequence(
    mini_gpu, seq;
    n_cycle = 1, n_step = 20, n_sample = 3, rng = MersenneTwister(42),
)
CUDA.synchronize()
elapsed = round(time() - t0; digits=1)
println("  Elapsed: $(elapsed)s")

coords = folded.prediction.coordinate
if coords isa CUDA.CuArray
    coords = Array(coords)
end
println("  Coordinates shape: $(size(coords))  (N_sample, N_atom, 3)")
println("  All finite: $(all(isfinite, coords))")

fold_dir = joinpath(OUT_ROOT, "folding")
pred_dir = PXDesign.Output.dump_prediction_bundle(
    fold_dir, "hemoglobin_fragment", folded.atoms, coords,
)
println("\n  Folding CIF files:")
for f in sort(readdir(pred_dir))
    fpath = joinpath(pred_dir, f)
    println("    $fpath  ($(filesize(fpath)) bytes)")
end
flush(stdout)

mini_gpu = nothing; GC.gc(); CUDA.reclaim()

# ══════════════════════════════════════════════════════════════════════════════
# 2. Folding model (ProtenixBase wrapper) — CPU
# ══════════════════════════════════════════════════════════════════════════════

println("\n" * "=" ^ 60)
println("  FOLDING MODEL (ProtenixBase wrapper) — CPU")
println("=" ^ 60)

seq_base = "ACDEFGHIKLMNPQRSTVWY"
println("Sequence ($(length(seq_base)) residues): $seq_base")

println("Running ProtenixBase.fold_sequence (10 steps, 2 samples) on CPU...")
t0 = time()
folded_base = PXDesign.ProtenixBase.fold_sequence(
    mini_cpu, seq_base;
    n_cycle = 1, n_step = 10, n_sample = 2, rng = MersenneTwister(99),
)
elapsed = round(time() - t0; digits=1)
println("  Elapsed: $(elapsed)s")

base_dir = joinpath(OUT_ROOT, "folding_base")
pred_dir_base = PXDesign.Output.dump_prediction_bundle(
    base_dir, "all_20_aa", folded_base.atoms, folded_base.prediction.coordinate,
)
println("  ProtenixBase CIF files:")
for f in sort(readdir(pred_dir_base))
    fpath = joinpath(pred_dir_base, f)
    println("    $fpath  ($(filesize(fpath)) bytes)")
end
flush(stdout)

# ══════════════════════════════════════════════════════════════════════════════
# 3. Design model (scaffold inference) — GPU
#    Uses refactored features-first Model code (DiffusionModule, etc.)
# ══════════════════════════════════════════════════════════════════════════════

println("\n" * "=" ^ 60)
println("  DESIGN MODEL (Scaffold Inference) — GPU")
println("=" ^ 60)

design_dir = joinpath(OUT_ROOT, "design")
mkpath(design_dir)

target_cif = joinpath(design_dir, "target.cif")
write(target_cif, """
data_target
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_formal_charge
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
ATOM 1  N  N  . ALA A 1 1 ? 0.000  0.000  0.000  1.00 10.00 ? 1 ALA A N  1
ATOM 2  C  CA . ALA A 1 1 ? 1.458  0.000  0.000  1.00 10.00 ? 1 ALA A CA 1
ATOM 3  C  C  . ALA A 1 1 ? 2.009  1.420  0.000  1.00 10.00 ? 1 ALA A C  1
ATOM 4  O  O  . ALA A 1 1 ? 1.246  2.390  0.000  1.00 10.00 ? 1 ALA A O  1
ATOM 5  C  CB . ALA A 1 1 ? 1.986 -0.760  1.208  1.00 10.00 ? 1 ALA A CB 1
ATOM 6  N  N  . GLY A 1 2 ? 3.321  1.502  0.000  1.00 10.00 ? 2 GLY A N  1
ATOM 7  C  CA . GLY A 1 2 ? 3.954  2.810  0.000  1.00 10.00 ? 2 GLY A CA 1
ATOM 8  C  C  . GLY A 1 2 ? 5.466  2.700  0.000  1.00 10.00 ? 2 GLY A C  1
ATOM 9  O  O  . GLY A 1 2 ? 6.030  1.610  0.000  1.00 10.00 ? 2 GLY A O  1
ATOM 10 N  N  . VAL A 1 3 ? 6.090  3.880  0.000  1.00 10.00 ? 3 VAL A N  1
ATOM 11 C  CA . VAL A 1 3 ? 7.542  3.960  0.000  1.00 10.00 ? 3 VAL A CA 1
ATOM 12 C  C  . VAL A 1 3 ? 8.050  5.380  0.000  1.00 10.00 ? 3 VAL A C  1
ATOM 13 O  O  . VAL A 1 3 ? 7.250  6.310  0.000  1.00 10.00 ? 3 VAL A O  1
ATOM 14 C  CB . VAL A 1 3 ? 8.090  3.210  1.220  1.00 10.00 ? 3 VAL A CB 1
ATOM 15 C  CG1 . VAL A 1 3 ? 9.610  3.280  1.220  1.00 10.00 ? 3 VAL A CG1 1
ATOM 16 C  CG2 . VAL A 1 3 ? 7.550  1.790  1.220  1.00 10.00 ? 3 VAL A CG2 1
#
""")

input_json = joinpath(design_dir, "input.json")
write(input_json, """
[
  {
    "name": "scaffold_demo",
    "condition": {
      "structure_file": "$target_cif",
      "filter": {"chain_id": ["A"], "crop": {}},
      "msa": {}
    },
    "hotspot": {"A": [1, 2]},
    "generation": [{"type": "protein", "length": 4, "count": 1}]
  }
]
""")

c_token, c_s, c_z, c_s_inputs = 24, 16, 8, 32
c_atom, c_atompair = 16, 8

println("Building DiffusionModule + DesignConditionEmbedder...")
dm_cpu = PXDesign.Model.DiffusionModule(
    c_token, c_s, c_z, c_s_inputs;
    c_atom = c_atom, c_atompair = c_atompair,
    atom_encoder_blocks = 1, atom_encoder_heads = 2,
    n_blocks = 2, n_heads = 2,
    atom_decoder_blocks = 1, atom_decoder_heads = 2,
    rng = MersenneTwister(42),
)
dce_cpu = PXDesign.Model.DesignConditionEmbedder(
    c_token; c_s_inputs = c_s_inputs, c_z = c_z,
    c_atom = c_atom, c_atompair = c_atompair,
    n_blocks = 1, n_heads = 2,
    rng = MersenneTwister(43),
)

println("Moving design models to GPU...")
dm_gpu = gpu(dm_cpu)
dce_gpu = gpu(dce_cpu)
GC.gc(); CUDA.reclaim()
println("  GPU memory after model load: ", round(CUDA.available_memory() / 1024^3, digits=1), " GB free")
flush(stdout)

# Parse task and build features on CPU
raw_tasks = PXDesign.Schema.parse_tasks(PXDesign.Inputs.load_input_tasks(input_json))
task = first(raw_tasks)
feat_bundle = PXDesign.Data.build_basic_feature_bundle(task; rng = MersenneTwister(44))
feat = feat_bundle["input_feature_dict"]
n_atom = Int(feat_bundle["dims"]["N_atom"])
n_tok = length(feat["token_index"])

# Run DesignConditionEmbedder on GPU (move feature dict to GPU)
feat_gpu = gpu(feat)
s_inputs_gpu, z_trunk_gpu = dce_gpu(feat_gpu)
s_trunk_gpu = CUDA.zeros(Float32, c_s, n_tok)

# relpos stays on CPU — relative_position_features does scalar loops
relpos_cpu = PXDesign.Model.as_relpos_input(feat)

# atom features: NamedTuple arrays go to GPU
_nt_to_gpu(nt::NamedTuple) = NamedTuple{keys(nt)}(map(v -> v isa AbstractArray ? gpu(v) : v, values(nt)))
atom_input_gpu = _nt_to_gpu(PXDesign.Model.as_atom_attention_input(feat))

# atom_to_token_idx stays on CPU — windowing code does scalar loops
atom_to_token_idx = Int.(feat["atom_to_token_idx"])

scheduler = PXDesign.Model.InferenceNoiseScheduler()
noise_schedule = scheduler(10; dtype = Float32)

denoise = (x_noisy, t_hat; kwargs...) -> dm_gpu(
    x_noisy, t_hat;
    relpos_input = relpos_cpu,
    s_inputs = s_inputs_gpu,
    s_trunk = s_trunk_gpu,
    z_trunk = z_trunk_gpu,
    atom_to_token_idx = atom_to_token_idx,
    input_feature_dict = atom_input_gpu,
)

flush(stdout)
println("Running diffusion sampling (N_step=10, N_sample=2) on GPU...")
t0 = time()
coords_gpu = PXDesign.Model.sample_diffusion(
    denoise;
    noise_schedule = noise_schedule,
    N_sample = 2, N_atom = n_atom,
    gamma0 = 0.0, gamma_min = 1.0,
    noise_scale_lambda = 1.003, step_scale_eta = 1.0,
    rng = MersenneTwister(42),
    device_ref = s_inputs_gpu,
)
CUDA.synchronize()
elapsed = round(time() - t0; digits=1)
println("  Elapsed: $(elapsed)s")

coords_cpu = Array(coords_gpu)  # (3, N_atom, N_sample) features-first
println("  Coordinates shape: $(size(coords_cpu))")
println("  All finite: $(all(isfinite, coords_cpu))")

design_out_dir = joinpath(design_dir, "output", task.name)
pred_dir_design = PXDesign.Output.dump_prediction_bundle(
    design_out_dir, task.name, feat_bundle["atoms"], coords_cpu,
)
println("\n  Design CIF files:")
for f in sort(readdir(pred_dir_design))
    fpath = joinpath(pred_dir_design, f)
    println("    $fpath  ($(filesize(fpath)) bytes)")
end

# ── Summary ──────────────────────────────────────────────────────────────────

println("\n" * "=" ^ 60)
println("  SUMMARY")
println("=" ^ 60)
println("All outputs in: $OUT_ROOT")
println("  folding/predictions/       — ProtenixMini 50-residue fold, 3 samples (GPU)")
println("  folding_base/predictions/  — ProtenixBase 20-residue fold, 2 samples (CPU)")
println("  design/output/             — Scaffold design, 2 samples (GPU: $(CUDA.name(CUDA.device())))")
println("=" ^ 60)

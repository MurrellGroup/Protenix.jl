#!/usr/bin/env julia
#
# Design model (scaffold inference) on GPU with random weights.
#
# Usage: julia --project=<env> scripts/design_gpu_only.jl
#
using CUDA, cuDNN  # Must load GPU backends before Flux
using Flux
using Random
using PXDesign

# No allowscalar — scalar GPU indexing is forbidden

const OUT_ROOT = joinpath(@__DIR__, "..", "e2e_output")
mkpath(OUT_ROOT)

println("CUDA: ", CUDA.functional(), "  GPU: ", CUDA.name(CUDA.device()))
println("Free GPU memory: ", round(CUDA.available_memory() / 1024^3, digits=1), " GB")
println()

println("=" ^ 60)
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
ATOM 1  N  N   . ALA A 1 1 ? 0.000  0.000  0.000  1.00 10.00 ? 1 ALA A N   1
ATOM 2  C  CA  . ALA A 1 1 ? 1.458  0.000  0.000  1.00 10.00 ? 1 ALA A CA  1
ATOM 3  C  C   . ALA A 1 1 ? 2.009  1.420  0.000  1.00 10.00 ? 1 ALA A C   1
ATOM 4  O  O   . ALA A 1 1 ? 1.246  2.390  0.000  1.00 10.00 ? 1 ALA A O   1
ATOM 5  C  CB  . ALA A 1 1 ? 1.986 -0.760  1.208  1.00 10.00 ? 1 ALA A CB  1
ATOM 6  N  N   . GLY A 1 2 ? 3.321  1.502  0.000  1.00 10.00 ? 2 GLY A N   1
ATOM 7  C  CA  . GLY A 1 2 ? 3.954  2.810  0.000  1.00 10.00 ? 2 GLY A CA  1
ATOM 8  C  C   . GLY A 1 2 ? 5.466  2.700  0.000  1.00 10.00 ? 2 GLY A C   1
ATOM 9  O  O   . GLY A 1 2 ? 6.030  1.610  0.000  1.00 10.00 ? 2 GLY A O   1
ATOM 10 N  N   . VAL A 1 3 ? 6.090  3.880  0.000  1.00 10.00 ? 3 VAL A N   1
ATOM 11 C  CA  . VAL A 1 3 ? 7.542  3.960  0.000  1.00 10.00 ? 3 VAL A CA  1
ATOM 12 C  C   . VAL A 1 3 ? 8.050  5.380  0.000  1.00 10.00 ? 3 VAL A C   1
ATOM 13 O  O   . VAL A 1 3 ? 7.250  6.310  0.000  1.00 10.00 ? 3 VAL A O   1
ATOM 14 C  CB  . VAL A 1 3 ? 8.090  3.210  1.220  1.00 10.00 ? 3 VAL A CB  1
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

println("Building models on CPU...")
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

println("Parsing task and building features on CPU...")
raw_tasks = PXDesign.Schema.parse_tasks(PXDesign.Inputs.load_input_tasks(input_json))
task = first(raw_tasks)
feat_bundle = PXDesign.Data.build_basic_feature_bundle(task; rng = MersenneTwister(44))
feat = feat_bundle["input_feature_dict"]
n_atom = Int(feat_bundle["dims"]["N_atom"])
n_tok = length(feat["token_index"])
println("  N_atom=$n_atom  N_token=$n_tok")

# Run DesignConditionEmbedder on CPU (one-shot, not in hot loop)
println("Running DesignConditionEmbedder on CPU...")
s_inputs_cpu, z_trunk_cpu = dce_cpu(feat)
s_trunk_cpu = zeros(Float32, c_s, n_tok)

# Move model and continuous-valued inputs to GPU
println("Moving model + inputs to GPU...")
dm_gpu = gpu(dm_cpu)
s_inputs_gpu = gpu(s_inputs_cpu)
s_trunk_gpu = CUDA.zeros(Float32, c_s, n_tok)
z_trunk_gpu = gpu(z_trunk_cpu)

# relpos stays on CPU — relative_position_features does scalar loops on these vectors
relpos_cpu = PXDesign.Model.as_relpos_input(feat)

# atom features: NamedTuple arrays go to GPU (used by GPU linear layers in atom attention)
_nt_to_gpu(nt::NamedTuple) = NamedTuple{keys(nt)}(map(v -> v isa AbstractArray ? gpu(v) : v, values(nt)))
atom_input_gpu = _nt_to_gpu(PXDesign.Model.as_atom_attention_input(feat))

# atom_to_token_idx stays on CPU — windowing code does scalar loops
atom_to_token_idx = Int.(feat["atom_to_token_idx"])

GC.gc(); CUDA.reclaim()
println("  GPU memory after load: ", round(CUDA.available_memory() / 1024^3, digits=1), " GB free")

# Closure: sampler creates x_noisy on CPU; we move it to GPU for the model, return CPU result.
# The coordinate tensor is tiny (3×N_atom×N_sample floats) so transfer cost is negligible.
denoise = (x_noisy, t_hat; kwargs...) -> begin
    x_gpu = cu(Float32.(x_noisy))
    out_gpu = dm_gpu(
        x_gpu, t_hat;
        relpos_input = relpos_cpu,
        s_inputs = s_inputs_gpu,
        s_trunk = s_trunk_gpu,
        z_trunk = z_trunk_gpu,
        atom_to_token_idx = atom_to_token_idx,
        input_feature_dict = atom_input_gpu,
    )
    return Array(out_gpu)
end

scheduler = PXDesign.Model.InferenceNoiseScheduler()
noise_schedule = scheduler(10; dtype = Float32)

println("Running diffusion sampling (N_step=10, N_sample=2) on GPU...")
t0 = time()
coords_cpu = PXDesign.Model.sample_diffusion(
    denoise;
    noise_schedule = noise_schedule,
    N_sample = 2, N_atom = n_atom,
    gamma0 = 0.0, gamma_min = 1.0,
    noise_scale_lambda = 1.003, step_scale_eta = 1.0,
    rng = MersenneTwister(42),
)
CUDA.synchronize()
elapsed = round(time() - t0; digits=1)
println("  Elapsed: $(elapsed)s")
# coords_cpu is features-first (3, N_atom, N_sample)
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

println("\nDone! All design outputs in: $design_out_dir")

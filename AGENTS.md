# PXDesign.jl Agent Guidance

## Core Objectives

1. Perfect parity with the Python PXDesign reference implementation for infer-only behavior.
2. Clean, modular, idiomatic Julia code with clear boundaries and testable components.

## Working Rules

1. Prefer semantic parity over clever rewrites. If behavior differs, parity wins.
2. Prefer idiomatic Julia representations over Python-style generic maps:
   - use concrete structs where schema is stable
   - use `NamedTuple` for lightweight fixed-shape records
   - keep `Dict` only at true dynamic boundaries (I/O, untyped external payloads)
3. Keep modules focused:
   - config
   - inputs
   - cache
   - model
   - infer runner
4. Add tests for every parity-sensitive ported function.
5. Record known deltas from Python explicitly in docs until closed.
6. Avoid unnecessary external runtime dependencies; isolate any temporary compatibility shims.
7. Hardware safety: never use GPU on this Mac (MPS/CUDA). Always run CPU-only to avoid host instability.

## Escalation Policy

1. Default to non-escalated commands only.
2. Treat escalation as a strict budget:
   - target: zero escalations for normal coding/test loops
   - hard cap: one escalation when possible, two only if absolutely necessary
3. Batch unavoidable escalated work into one consolidated step instead of multiple prompts.
4. Never escalate for exploratory reads that can be done from local workspace copies.
5. Do not escalate for routine runs/tests/inference once the local runtime is prepared.

## Sandbox Python Runtime (Torch/OpenMP)

Use this workflow to keep Python reference runs working inside sandbox without repeated escalations.

1. Root cause observed in sandbox:
   - `OMP: Error #179: Function Can't open SHM2 failed`
   - Triggered by `import torch` with the wheel-bundled `libomp.dylib` in `.venv_pyref`.
2. Local fix (one-time, non-escalated):
   - backup: `.venv_pyref/lib/python3.11/site-packages/torch/lib/libomp.dylib.bak`
   - replace venv `libomp.dylib` with:
     - `/opt/homebrew/lib/python3.11/site-packages/torch/lib/libomp.dylib`
3. Verification commands:
   - `source scripts/python_reference_env.sh`
   - `python -c 'import torch; print(torch.__version__)'`
   - `bash scripts/run_python_reference_smoke.sh`
4. Do not request escalation for Python reference runs after this fix. Escalation is only for installs or other true environment mutations that cannot be done in sandbox.
5. Keep this as an environment compatibility shim only; do not encode machine-specific paths into library code.
6. Force CPU execution paths for all reference runs (no `cuda`, no `mps`).

## Sandbox Julia Runtime

Use the pinned Julia binary directly in sandbox to avoid launcher network checks:

- preferred: `~/.julia/juliaup/julia-1.11.2+0.aarch64.apple.darwin14/bin/julia`
- avoid plain `julia` launcher for run/test loops (it may try to refresh juliaup metadata and fail in sandbox).

## Julia Package Installs

1. Default flow: run `Pkg.add(...)` in the project environment inside sandbox first.
2. If package install fails due to sandbox/network restrictions, escalate once and batch installs.
3. Do not escalate preemptively for package installs that may work in sandbox.

## Execution Discipline

1. Work autonomously: keep coding through implementation and validation loops without pausing for status updates.
2. Stop only when the task is fully done end-to-end or 100% blocked by a concrete external dependency.
3. If blocked, report the blocker with the minimum context needed and immediately propose the exact unblock step.

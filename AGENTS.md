## Project source of truth

`docs/project_exercises.pdf` is the authoritative specification for this mini-project. If any instruction in this file seems incomplete or conflicts with the PDF, read the PDF and follow it.

This `AGENTS.md` is a compact working guide for future Codex sessions. It should preserve the important project goals, commands, paths, and HPC workflow without copying the entire assignment text.

Supplementary local references:

- `docs/project_exercises.pdf`: canonical mini-project assignment.
- `docs/template_bash.sh`: canonical CPU batch script style for this repo.
- `docs/initializing_conda_env`: conda setup reference.
- `docs/File transfer tofrom HPC.html`: file transfer reference.

## Project goal

The project is the 02613 mini-project "Wall Heating!". The task is to optimize a Python simulation that evaluates wall heating on the Modified Swiss Dwellings dataset.

Core problem:

- Simulate steady-state 2D heat diffusion with the Jacobi method.
- Inside wall grid points are fixed at 25 C.
- Load-bearing wall grid points are fixed at 5 C.
- Interior room points are updated as the average of their four direct neighbors.
- Points on walls and outside buildings are not updated.
- The reference implementation is called as `python simulate.py <N>` and is intentionally too slow.

The optimized implementations must preserve the reference output semantics unless a task explicitly says otherwise, such as the fixed-iteration CUDA kernel task.

## Dataset and reference behavior

The dataset is available on DTU HPC at:

```bash
/dtu/projects/02613_2025/data/modified_swiss_dwellings/
```

There are 4571 building floorplans. Building IDs are listed in:

```bash
/dtu/projects/02613_2025/data/modified_swiss_dwellings/building_ids.txt
```

For each building ID:

- `{building_id}_domain.npy`: 512 x 512 NumPy array with initial temperatures. Load-bearing walls are 5, inside walls are 25, all other points are 0.
- `{building_id}_interior.npy`: 512 x 512 boolean/binary mask. Only mask points equal to 1 are updated by Jacobi.

The reference script pads each domain to a 514 x 514 grid before iteration. Summary statistics are computed over `u[1:-1, 1:-1][interior_mask]`.

Reference CSV columns:

```text
building_id, mean_temp, std_temp, pct_above_18, pct_below_15
```

Statistic definitions:

- `mean_temp`: mean interior room temperature.
- `std_temp`: standard deviation of interior room temperature.
- `pct_above_18`: percentage of interior room area above 18 C.
- `pct_below_15`: percentage of interior room area below 15 C.

## Required project tasks

Use `docs/project_exercises.pdf` for the exact wording and report questions. The main required work is:

1. Load and visualize input data for a few floorplans.
2. Run and time the reference implementation for a small subset, preferably 10 to 20 floorplans, as a batch job.
3. Visualize simulation results for a few floorplans.
4. Profile the reference `jacobi` function with `kernprof` and explain the timing.
5. Parallelize over floorplans with static scheduling, using no more than 100 floorplans for timing experiments.
6. Repeat the parallelization experiment with dynamic scheduling.
7. Implement a CPU Numba JIT version of `jacobi`, explain cache-friendly access patterns, and time it.
8. Implement a custom Numba CUDA kernel. The kernel should perform one Jacobi iteration per launch, skip early stopping, and use a helper with the same inputs as the reference except `atol`.
9. Adapt the reference solution to run on GPU using CuPy.
10. Profile the CuPy solution with `nsys`, identify the main performance issue, and try to fix it.
11. Optionally improve one or more solutions further, for example CPU JIT parallelization or job arrays.
12. Process all floorplans with a fast implementation and answer the final analysis questions using the generated CSV results, preferably with Pandas.

Final analysis questions from the PDF include histograms of mean temperatures, average mean temperature, average temperature standard deviation, number of buildings with at least 50% above 18 C, and number of buildings with at least 50% below 15 C.

Hand-in consists of a short PDF report plus a zip file with Python code and job scripts.

## Correctness and benchmarking rules

- Compare optimized outputs against the reference implementation on small `N` before trusting performance numbers.
- Keep CSV output parseable and compatible with the reference columns unless the script is only for profiling or visualization.
- Run timing experiments as DTU HPC batch jobs when reporting results, because interactive timings may be noisy.
- Use no more than 100 floorplans for the parallel timing experiments requested in the PDF.
- Record command, environment, worker count, CPU cores, GPU queue/type, walltime, and relevant job script settings for timings.
- For CPU parallel experiments, report speed-up, estimated parallel fraction from Amdahl's law, theoretical maximum speed-up, achieved speed-up, and estimated full-dataset runtime.
- For GPU experiments, use `02613_2026` and document whether timings include data transfer, compilation/warm-up, and synchronization.
- For Numba timings, avoid counting first-call compilation time unless explicitly studying compilation overhead.
- For CUDA/CuPy timings, synchronize before stopping timers when measuring GPU kernel work.

Useful local commands:

```bash
python simulate.py 10
time python -u simulate.py 10
kernprof -l -v simulate.py 10
python -c "import sys; print(sys.executable)"
python -c "import numba, cupy; print(numba.__version__, cupy.__version__)"
```

Typical submission and inspection:

```bash
bsub < job.sh
ls -lh *.out *.err batch_output/
```

## DTU HPC workflow

Normal workflow for this repo:

1. Develop or edit code locally in this mini-project repo.
2. Transfer relevant files to DTU HPC.
3. Log in to DTU HPC.
4. Initialize conda and activate the correct course environment.
5. Run quick smoke tests interactively if needed.
6. Submit reliable timing runs and larger experiments with `bsub < script.sh`.
7. Inspect `.out` and `.err` files, adjust code/resources, and resubmit if needed.
8. Copy results, CSV files, plots, and job logs back as needed for the report.

Prefer batch jobs for timings that will be used in the report.

## Conda environment rules

Before using a course environment on DTU HPC, run:

```bash
source /dtu/projects/02613_2025/conda/conda_init.sh
```

Default for this mini-project:

```bash
conda activate 02613_2026
```

Use `02613_2026` for GPU code, CuPy, Numba CUDA, and mini-project work in general.

Fallback for older non-GPU course exercises only:

```bash
conda activate 02613
```

Important:

- If you change node, initialize conda again.
- In batch scripts, place `source /dtu/projects/02613_2025/conda/conda_init.sh` and `conda activate ...` after the `#BSUB` lines and before other commands.
- Verify the environment with `python -c "import sys; print(sys.executable)"`.
- If `conda activate` fails, check `module list`; if `python3` is loaded, try `module unload python3`.
- Also check whether `~/.bashrc` auto-loads another teaching environment that interferes with conda.

## CPU batch jobs

Use `docs/template_bash.sh` as the canonical example for CPU batch scripts in this repo.

Typical CPU batch pattern:

```bash
#!/bin/bash
#BSUB -J wall_cpu
#BSUB -q hpc
#BSUB -n 4
#BSUB -W 10
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -M 2GB
#BSUB -o job_%J.out
#BSUB -e job_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

time python -u simulate.py 10
```

Common `#BSUB` fields:

- `-J`: job name shown in the queue.
- `-q`: queue, usually `hpc` for CPU jobs.
- `-n`: number of CPU cores.
- `-W`: walltime limit.
- `-R "span[hosts=1]"`: keep cores on the same node.
- `-R "rusage[mem=2GB]"`: reserve memory per core.
- `-M 2GB`: hard memory limit per process.
- `-o` and `-e`: stdout and stderr files.
- Optional hardware selection such as `select[model==XeonGold6126]` can help reproducible CPU benchmarks.

Submit with:

```bash
bsub < job.sh
```

## GPU batch jobs

Use explicit GPU-oriented shell scripts for CUDA, CuPy, and GPU profiling. Do not reuse a CPU script unchanged.

Typical GPU batch pattern:

```bash
#!/bin/sh
#BSUB -q gpua100
#BSUB -J wall_gpu
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:30
#BSUB -R "select[gpu80gb]"
#BSUB -o batch_output/gpu_%J.out
#BSUB -e batch_output/gpu_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

python -u simulate_cupy.py 10
```

GPU-specific notes:

- Use queue `gpua100` for A100 GPU jobs.
- Request at least 4 CPU cores with `-n 4`.
- Request one GPU with `#BSUB -gpu "num=1:mode=exclusive_process"`.
- `exclusive_process` reserves the GPU for the process/job.
- Use `#BSUB -R "select[gpu80gb]"` when the 80 GB A100 is required.
- Store output in a dedicated folder such as `batch_output/`.

## Profiling commands

Line profiling with `kernprof`:

```bash
kernprof -l -v simulate.py 10
```

If using `kernprof`, decorate the function under investigation as needed, typically `jacobi`.

GPU profiling with Nsight Systems should be run on a GPU node, for example:

```bash
nsys profile -o cupy_profile python -u simulate_cupy.py 10
```

For GPU profiling, keep problem sizes small enough that profiling completes within the requested walltime, but large enough to expose the bottleneck described in the PDF.

## File transfer

Prefer terminal-based transfer.

Single-file upload from local machine to HPC:

```bash
scp path/to/file.py <username>@login.hpc.dtu.dk:path/to/file.py
```

Single-file download from HPC to local machine:

```bash
scp <username>@login.hpc.dtu.dk:path/to/file.py path/to/file.py
```

Interactive transfer for many files:

```bash
sftp <username>@transfer.gbar.dtu.dk
```

Useful `sftp` commands:

- `cd`, `ls`, `mkdir` for the remote side.
- `lcd`, `lls` for the local side.
- `put file.py` to upload.
- `get file.py` to download.
- `get *` to download many files.

Important:

- Run `scp` and `sftp` from the local machine, not from inside an active HPC shell.
- If not on DTU Wi-Fi, VPN or SSH key setup may be needed.

## Guidance for future Codex sessions

- Read `docs/project_exercises.pdf` before making major project-structure or report-content decisions.
- Keep code, job scripts, generated CSVs, plots, and report artifacts organized so the final zip hand-in is easy to assemble.
- When adding new implementations, give scripts clear names such as `simulate_parallel_static.py`, `simulate_parallel_dynamic.py`, `simulate_numba_cpu.py`, `simulate_cuda.py`, and `simulate_cupy.py` unless the repo already has a naming pattern.
- Preserve a reference path for correctness checks. Do not overwrite the only known-good reference implementation without keeping a way to compare results.
- Prefer small `N` smoke tests before long HPC submissions.
- For report-facing timings, keep the batch script and output logs so the run can be documented.

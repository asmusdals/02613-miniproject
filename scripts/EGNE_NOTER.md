TODO:
1. kør q8 på dutten, for at få resultaterne, se nedenunder hvad der er lavet, hvordan, og hvad der skal gøres.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Exercise 8 — Custom Numba CUDA Kernel
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
How it works
scripts/q8_cuda.py

jacobi_kernel is a @cuda.jit 2D kernel. Each thread handles one grid point (i, j). It reads from u, writes to u_new — one Jacobi sweep per kernel launch. No early stopping, no max-delta tracking. Non-interior points just copy their value through unchanged (so boundary temps are preserved).

jacobi_cuda(u0, interior_mask, max_iter) is the helper matching the reference signature minus atol. It:

Copies both arrays to device once
Launches the kernel max_iter=20_000 times, swapping the two device buffers each iteration (pointer swap — no extra copies)
Calls cuda.synchronize() before stopping any timer
Copies the final buffer back to host
Grid/block math: (rows + 15) // 16 × (cols + 15) // 16 blocks of 16×16 threads — standard 2D tiling over the 514×514 padded grid.

What you need to do on DTU HPC
Transfer the file to the HPC (same as previous exercises, e.g. via scp or the VSCode remote extension).

Submit from the project root:


cd ~/Documents/02613/miniproject
mkdir -p batch_output results
bsub < jobs/q8_cuda.sh
Key differences from q7's job script:

Queue is gpua100 (not hpc) — this requests an A100 GPU node
-gpu "num=1:mode=exclusive_process" allocates one GPU exclusively
-n 4 CPU cores (standard for GPU jobs on this cluster)
Wall time is 1h (GPU is much faster — should complete in minutes for N=10)
After it runs, copy batch_output/q8_cuda_JOBID.{out,err} into reports/ for tracking, just like previous exercises.

One thing to verify: check that the DTU 02613_2026 conda env has numba[cuda] / cudatoolkit installed. If from numba import cuda fails, the env may need conda install cudatoolkit — but based on the exercise being part of the course, it should be there already.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------


2. Gør det samme med q9

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Exercise 9 — CuPy GPU Adaptation
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

How it works
scripts/q9_cupy.py

jacobi_cupy is structurally identical to the reference jacobi in simulate.py, with two changes:

cp.asarray(u) / cp.asarray(interior_mask) — transfers the arrays to GPU memory at the start
cp.asnumpy(u) — transfers the result back to CPU at the end
Everything in between (cp.copy, slicing, cp.abs(...).max()) uses the exact same syntax as NumPy — CuPy is a drop-in replacement. The early-stopping atol check is preserved (unlike q8).

What to expect / report
The surprising result (what q9c is asking about) is likely that CuPy is slower than plain NumPy or close to it for N=10. The reason: each building is processed one at a time, so the GPU spends most of its time idle while data is being transferred back and forth between CPU and GPU memory. The 514×514 grid is also relatively small, so GPU occupancy is low per floorplan. This is the bottleneck that exercise 10 asks you to find with nsys and fix.

To submit on DTU HPC
Same as q8 — GPU queue:


cd ~/Documents/02613/miniproject
mkdir -p batch_output results
bsub < jobs/q9_cupy.sh

Copy batch_output/q9_cupy_JOBID.{out,err} → reports/ afterward. The result CSV will be at results/q9_cupy_N10.csv and should match results/reference_10.csv (CuPy preserves the early-stopping logic, so values should be byte-identical).

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Exercise 10 — nsys Profiling + Fix
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
The problem nsys reveals
When you profile q9_cupy.py, the nsys output (q10_nsys_q9_stats.txt) will show a CUDA API summary where cudaMemcpy (Host→Device and Device→Host) dominates total time, with the actual GPU compute taking a small fraction. This is because q9 calls cp.asarray() and cp.asnumpy() inside the loop over buildings — N separate round-trip transfers, one per building.

Additionally, the float(delta) call each iteration also syncs the GPU (scalar copy back to CPU), stalling the pipeline ~20,000 times per building.

The fix — scripts/q10_cupy_batched.py
Load all N buildings on CPU first into (N, 514, 514) and (N, 512, 512) arrays
One cp.asarray() call to transfer the entire batch to GPU
Vectorized batched Jacobi — the slicing u[:, 1:-1, :-2] etc. works identically with the extra batch dimension; cp.where(mask, u_new, u_old) replaces boolean-index assignment (which doesn't generalize cleanly to 3D batches)
One cp.asnumpy() call to transfer all results back
The GPU now runs one large kernel per iteration over all N buildings simultaneously, with full occupancy, instead of N tiny sequential kernels with transfer overhead between each.

Submission order

# Profile the naive q9 first (to show the problem in your report):
bsub < jobs/q10_nsys_profile.sh

# Then run the fixed batched solution:
bsub < jobs/q10_cupy_batched.sh
Retrieve the stats file with:


scp s224473@login.hpc.dtu.dk:Documents/02613/miniproject/reports/q10_nsys_q9_stats.txt reports/
scp s224473@login.hpc.dtu.dk:Documents/02613/miniproject/results/q10_cupy_batched_N10.csv results/
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------




---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Exercise 11 — Parallelised Numba JIT
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------


How it works — scripts/q11_numba_parallel.py
It merges q6 and q7 directly:

q7's jacobi_numba (@jit(nopython=True, cache=True)) as the per-building kernel
q6's Pool.imap_unordered(..., chunksize=1) for dynamic scheduling across workers
The cache=True flag writes the compiled machine code to __pycache__ so subsequent runs don't recompile. The warm-up call in main() runs jacobi_numba once before the Pool is created — on Linux (the HPC), workers are forked from the main process and inherit the already-compiled function, so there's no recompilation penalty in any worker.

Expected gain: q7 was ~20s for N=10. With 8 workers and dynamic scheduling it should approach q6's ~7× speedup over the numpy reference, but each worker now runs at Numba speed (~4× faster than numpy). So roughly 25–30× faster than the reference total.

To submit

cd ~/Documents/02613/miniproject
mkdir -p batch_output results
bsub < jobs/q11_numba_parallel.sh
The job sweeps W=1,2,4,8 over N=100, printing N,workers,elapsed_seconds to stdout — same format as q5/q6 so you can directly compare the three on one plot. It also reruns W=8 at the end for the full-dataset extrapolation.




---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Exercise 12 — Full Dataset + Pandas Analysis
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------



scripts/q12_analysis.py
Reads the CSV with pd.read_csv, then answers each sub-question:

Sub-q	What it does
a	Histograms of all 4 stats (mean_temp, std_temp, pct_above_18, pct_below_15), saved as PNGs in reports/figures/
b	df["mean_temp"].mean()
c	df["std_temp"].mean()
d	(df["pct_above_18"] >= 50).sum()
e	(df["pct_below_15"] >= 50).sum()
All answers + figure paths are printed to stdout and saved to reports/q12_analysis_summary.txt.

jobs/q12_full_run.sh
Uses q11 (dynamic parallel Numba, W=8) on all 4571 buildings — the fastest CPU solution. Wall time is 4h (conservative; expect ~25–30 min based on the q11 extrapolation). The analysis script runs automatically at the end of the same job.

To submit

cd ~/Documents/02613/miniproject
mkdir -p batch_output results reports/figures
bsub < jobs/q12_full_run.sh
To retrieve results afterwards

# The CSV (4571 rows)
scp s224473@login.hpc.dtu.dk:Documents/02613/miniproject/results/q12_all_N4571_W8.csv results/

# The analysis summary + all 4 histogram PNGs
scp s224473@login.hpc.dtu.dk:Documents/02613/miniproject/reports/q12_analysis_summary.txt reports/
scp "s224473@login.hpc.dtu.dk:Documents/02613/miniproject
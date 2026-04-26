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




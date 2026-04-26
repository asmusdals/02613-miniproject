from os.path import dirname, join
import argparse
import sys

import numpy as np
from numba import cuda

sys.path.append(dirname(__file__))
from simulate import load_data, summary_stats


LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
MAX_ITER = 20_000
STAT_KEYS = ["mean_temp", "std_temp", "pct_above_18", "pct_below_15"]

THREADS_PER_BLOCK = (16, 16)


@cuda.jit
def jacobi_kernel(u, u_new, interior_mask):
    i, j = cuda.grid(2)
    rows, cols = u.shape

    if i < 1 or i >= rows - 1 or j < 1 or j >= cols - 1:
        return

    if interior_mask[i - 1, j - 1]:
        u_new[i, j] = 0.25 * (u[i, j - 1] + u[i, j + 1] + u[i - 1, j] + u[i + 1, j])
    else:
        u_new[i, j] = u[i, j]


def jacobi_cuda(u0, interior_mask, max_iter):
    rows, cols = u0.shape
    blocks = (
        (rows + THREADS_PER_BLOCK[0] - 1) // THREADS_PER_BLOCK[0],
        (cols + THREADS_PER_BLOCK[1] - 1) // THREADS_PER_BLOCK[1],
    )

    d_u = cuda.to_device(u0)
    d_u_new = cuda.to_device(u0.copy())
    d_mask = cuda.to_device(interior_mask)

    for _ in range(max_iter):
        jacobi_kernel[blocks, THREADS_PER_BLOCK](d_u, d_u_new, d_mask)
        d_u, d_u_new = d_u_new, d_u

    cuda.synchronize()
    return d_u.copy_to_host()


def main(n, load_dir):
    with open(join(load_dir, "building_ids.txt"), "r") as f:
        building_ids = f.read().splitlines()[:n]

    print("building_id, " + ", ".join(STAT_KEYS))
    for building_id in building_ids:
        u0, interior_mask = load_data(load_dir, building_id)
        u = jacobi_cuda(u0, interior_mask, MAX_ITER)
        stats = summary_stats(u, interior_mask)
        print(f"{building_id},", ", ".join(str(stats[key]) for key in STAT_KEYS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the wall-heating simulation with a custom Numba CUDA kernel."
    )
    parser.add_argument("N", type=int, help="Number of floorplans to process.")
    parser.add_argument(
        "--load-dir",
        default=LOAD_DIR,
        help="Directory containing building_ids.txt and the floorplan .npy files.",
    )
    args = parser.parse_args()
    main(args.N, args.load_dir)

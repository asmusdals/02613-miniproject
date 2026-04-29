from os.path import dirname, join
import argparse
import sys

import cupy as cp
import numpy as np

sys.path.append(dirname(__file__))
from simulate import load_data, summary_stats


LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
MAX_ITER = 20_000
ABS_TOL = 1e-4
STAT_KEYS = ["mean_temp", "std_temp", "pct_above_18", "pct_below_15"]


def jacobi_cupy_batched(u_batch, interior_mask_batch, max_iter, atol=1e-4):
    # u_batch: (N, 514, 514), interior_mask_batch: (N, 512, 512) — both on GPU
    u = cp.copy(u_batch)
    for _ in range(max_iter):
        u_new = 0.25 * (
            u[:, 1:-1, :-2] + u[:, 1:-1, 2:] + u[:, :-2, 1:-1] + u[:, 2:, 1:-1]
        )
        delta = float(
            cp.abs(cp.where(interior_mask_batch, u[:, 1:-1, 1:-1] - u_new, 0.0)).max()
        )
        u[:, 1:-1, 1:-1] = cp.where(interior_mask_batch, u_new, u[:, 1:-1, 1:-1])
        if delta < atol:
            break

    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(u)


def main(n, load_dir):
    with open(join(load_dir, "building_ids.txt"), "r") as f:
        building_ids = f.read().splitlines()[:n]

    # Load all buildings on CPU, then transfer to GPU in one batch
    all_u0 = np.empty((n, 514, 514))
    all_masks = np.empty((n, 512, 512), dtype=bool)
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(load_dir, bid)
        all_u0[i] = u0
        all_masks[i] = interior_mask

    d_u0 = cp.asarray(all_u0)
    d_masks = cp.asarray(all_masks)

    all_u = jacobi_cupy_batched(d_u0, d_masks, MAX_ITER, ABS_TOL)

    print("building_id, " + ", ".join(STAT_KEYS))
    for i, building_id in enumerate(building_ids):
        stats = summary_stats(all_u[i], all_masks[i])
        print(f"{building_id},", ", ".join(str(stats[key]) for key in STAT_KEYS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the wall-heating simulation with batched CuPy (single GPU transfer)."
    )
    parser.add_argument("N", type=int, help="Number of floorplans to process.")
    parser.add_argument(
        "--load-dir",
        default=LOAD_DIR,
        help="Directory containing building_ids.txt and the floorplan .npy files.",
    )
    args = parser.parse_args()
    main(args.N, args.load_dir)

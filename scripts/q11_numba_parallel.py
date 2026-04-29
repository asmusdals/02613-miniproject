from multiprocessing import Pool
from os.path import dirname, join
import argparse
import sys

import numpy as np
from numba import jit

sys.path.append(dirname(__file__))
from simulate import load_data, summary_stats


LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
MAX_ITER = 20_000
ABS_TOL = 1e-4
STAT_KEYS = ["mean_temp", "std_temp", "pct_above_18", "pct_below_15"]


@jit(nopython=True, cache=True)
def jacobi_numba(u0, interior_mask, max_iter, atol):
    u = u0.copy()
    u_new = u0.copy()

    for _ in range(max_iter):
        delta = 0.0
        for i in range(1, u.shape[0] - 1):
            for j in range(1, u.shape[1] - 1):
                if interior_mask[i - 1, j - 1]:
                    value = 0.25 * (
                        u[i, j - 1] + u[i, j + 1] + u[i - 1, j] + u[i + 1, j]
                    )
                    diff = abs(value - u[i, j])
                    if diff > delta:
                        delta = diff
                    u_new[i, j] = value

        tmp = u
        u = u_new
        u_new = tmp

        if delta < atol:
            break

    return u


def process_floorplan(task):
    index, building_id, load_dir = task
    u0, interior_mask = load_data(load_dir, building_id)
    u = jacobi_numba(u0, interior_mask, MAX_ITER, ABS_TOL)
    stats = summary_stats(u, interior_mask)
    return index, building_id, [stats[key] for key in STAT_KEYS]


def main(n, workers, load_dir):
    with open(join(load_dir, "building_ids.txt"), "r") as f:
        building_ids = f.read().splitlines()[:n]

    # Warm up Numba in the main process; forked workers inherit the compiled function
    _u, _mask = load_data(load_dir, building_ids[0])
    jacobi_numba(_u, _mask, 1, ABS_TOL)

    tasks = [
        (index, bid, load_dir) for index, bid in enumerate(building_ids)
    ]
    rows = []

    with Pool(processes=workers) as pool:
        for row in pool.imap_unordered(process_floorplan, tasks, chunksize=1):
            rows.append(row)

    rows.sort(key=lambda row: row[0])

    print("building_id, " + ", ".join(STAT_KEYS))
    for _, building_id, values in rows:
        print(f"{building_id},", ", ".join(str(v) for v in values))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dynamic parallel Numba JIT: combines q6 multiprocessing with q7 Numba kernel."
    )
    parser.add_argument("N", type=int, help="Number of floorplans to process.")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes.")
    parser.add_argument(
        "--load-dir",
        default=LOAD_DIR,
        help="Directory containing building_ids.txt and the floorplan .npy files.",
    )
    args = parser.parse_args()
    main(args.N, args.workers, args.load_dir)

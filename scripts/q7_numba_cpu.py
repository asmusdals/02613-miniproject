from os.path import dirname, join
import argparse
import sys

from numba import jit

sys.path.append(dirname(__file__))
from simulate import load_data, summary_stats


LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
MAX_ITER = 20_000
ABS_TOL = 1e-4
STAT_KEYS = ["mean_temp", "std_temp", "pct_above_18", "pct_below_15"]


@jit(nopython=True)
def jacobi_numba(u0, interior_mask, max_iter, atol):
    u = u0.copy()
    u_new = u0.copy()

    for _ in range(max_iter):
        delta = 0.0

        for i in range(1, u.shape[0] - 1):
            for j in range(1, u.shape[1] - 1):
                if interior_mask[i - 1, j - 1]:
                    value = 0.25 * (
                        u[i, j - 1]
                        + u[i, j + 1]
                        + u[i - 1, j]
                        + u[i + 1, j]
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


def main(n, load_dir):
    with open(join(load_dir, "building_ids.txt"), "r") as f:
        building_ids = f.read().splitlines()[:n]

    print("building_id, " + ", ".join(STAT_KEYS))
    for building_id in building_ids:
        u0, interior_mask = load_data(load_dir, building_id)
        u = jacobi_numba(u0, interior_mask, MAX_ITER, ABS_TOL)
        stats = summary_stats(u, interior_mask)
        print(f"{building_id},", ", ".join(str(stats[key]) for key in STAT_KEYS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the wall-heating simulation with a CPU Numba JIT Jacobi kernel."
    )
    parser.add_argument("N", type=int, help="Number of floorplans to process.")
    parser.add_argument(
        "--load-dir",
        default=LOAD_DIR,
        help="Directory containing building_ids.txt and the floorplan .npy files.",
    )
    args = parser.parse_args()
    main(args.N, args.load_dir)

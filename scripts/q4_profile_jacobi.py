from os.path import join
import sys

import numpy as np


LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
MAX_ITER = 20_000
ABS_TOL = 1e-4


def load_data(load_dir, bid):
    size = 512
    u = np.zeros((size + 2, size + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


@profile
def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)

    for i in range(max_iter):
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        if delta < atol:
            break
    return u


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    with open(join(LOAD_DIR, "building_ids.txt"), "r") as f:
        building_ids = f.read().splitlines()[:n]

    for bid in building_ids:
        print(f"Profiling building_id={bid}")
        u0, interior_mask = load_data(LOAD_DIR, bid)
        jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)


if __name__ == "__main__":
    main()


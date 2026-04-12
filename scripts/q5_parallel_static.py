from multiprocessing import Pool
from os.path import dirname, join
import argparse
import sys

import numpy as np

sys.path.append(dirname(__file__))
from simulate import jacobi, load_data, summary_stats


LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
MAX_ITER = 20_000
ABS_TOL = 1e-4
STAT_KEYS = ["mean_temp", "std_temp", "pct_above_18", "pct_below_15"]


def process_chunk(building_ids):
    rows = []
    for bid in building_ids:
        u0, interior_mask = load_data(LOAD_DIR, bid)
        u = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
        stats = summary_stats(u, interior_mask)
        rows.append((bid, [stats[key] for key in STAT_KEYS]))
    return rows


def main(n, workers):
    with open(join(LOAD_DIR, "building_ids.txt"), "r") as f:
        building_ids = f.read().splitlines()[:n]

    chunks = [chunk.tolist() for chunk in np.array_split(building_ids, workers) if len(chunk) > 0]

    with Pool(processes=workers) as pool:
        chunk_results = pool.map(process_chunk, chunks)

    print("building_id, " + ", ".join(STAT_KEYS))
    for rows in chunk_results:
        for bid, values in rows:
            print(f"{bid},", ", ".join(str(value) for value in values))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    main(args.N, args.workers)

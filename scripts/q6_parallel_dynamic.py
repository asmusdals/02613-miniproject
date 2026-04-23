from multiprocessing import Pool
from os.path import dirname, join
import argparse
import sys

sys.path.append(dirname(__file__))
from simulate import jacobi, load_data, summary_stats


LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
MAX_ITER = 20_000
ABS_TOL = 1e-4
STAT_KEYS = ["mean_temp", "std_temp", "pct_above_18", "pct_below_15"]


def process_floorplan(task):
    index, building_id, load_dir = task
    u0, interior_mask = load_data(load_dir, building_id)
    u = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
    stats = summary_stats(u, interior_mask)
    return index, building_id, [stats[key] for key in STAT_KEYS]


def main(n, workers, load_dir):
    with open(join(load_dir, "building_ids.txt"), "r") as f:
        building_ids = f.read().splitlines()[:n]

    tasks = [(index, building_id, load_dir) for index, building_id in enumerate(building_ids)]
    rows = []

    with Pool(processes=workers) as pool:
        for row in pool.imap_unordered(process_floorplan, tasks, chunksize=1):
            rows.append(row)

    rows.sort(key=lambda row: row[0])

    print("building_id, " + ", ".join(STAT_KEYS))
    for _, building_id, values in rows:
        print(f"{building_id},", ", ".join(str(value) for value in values))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the wall-heating simulation with dynamic scheduling over floorplans."
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

from os.path import dirname, join
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(dirname(__file__))
from simulate import jacobi, load_data, summary_stats


LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"


def main(n, out_dir, max_iter, atol):
    os.makedirs(out_dir, exist_ok=True)

    with open(join(LOAD_DIR, "building_ids.txt"), "r") as f:
        building_ids = f.read().splitlines()[:n]

    for bid in building_ids:
        u0, interior_mask = load_data(LOAD_DIR, bid)
        u = jacobi(u0, interior_mask, max_iter, atol)
        stats = summary_stats(u, interior_mask)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        im0 = ax[0].imshow(u0[1:-1, 1:-1], cmap="inferno", vmin=0, vmax=25)
        ax[0].set_title(f"{bid} initial")
        plt.colorbar(im0, ax=ax[0], fraction=0.046)

        im1 = ax[1].imshow(u[1:-1, 1:-1], cmap="inferno", vmin=0, vmax=25)
        ax[1].set_title(f"{bid} result, mean={stats['mean_temp']:.2f} C")
        plt.colorbar(im1, ax=ax[1], fraction=0.046)

        for axis in ax:
            axis.set_xticks([])
            axis.set_yticks([])

        fig.tight_layout()
        fig.savefig(join(out_dir, f"{bid}_result.png"), dpi=200)
        plt.close(fig)
        print(f"Saved {join(out_dir, f'{bid}_result.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--out_dir", default="reports/figures")
    parser.add_argument("--max_iter", type=int, default=20_000)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()
    main(args.n, args.out_dir, args.max_iter, args.atol)


# used this specific call to solve q3
# python -u scripts/q3_visualize_results.py --n 3 --out_dir reports/figures
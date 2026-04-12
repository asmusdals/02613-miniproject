
from os.path import join
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"

def main(n, out_dir):
    with open(join(LOAD_DIR, "building_ids.txt"), "r") as f:
        building_ids = f.read().splitlines()[:n]

    for bid in building_ids:
        domain = np.load(join(LOAD_DIR, f"{bid}_domain.npy"))      # 512x512
        interior = np.load(join(LOAD_DIR, f"{bid}_interior.npy"))  # 512x512 mask

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        im0 = ax[0].imshow(domain, cmap="inferno", vmin=0, vmax=25)
        ax[0].set_title(f"{bid} domain")
        plt.colorbar(im0, ax=ax[0], fraction=0.046)

        ax[1].imshow(interior, cmap="gray")
        ax[1].set_title(f"{bid} interior mask")

        for a in ax:
            a.set_xticks([]); a.set_yticks([])

        fig.tight_layout()
        fig.savefig(join(out_dir, f"{bid}_input.png"), dpi=200)
        plt.close(fig)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--out_dir", default="results/viz")
    args = p.parse_args()
    main(args.n, args.out_dir)

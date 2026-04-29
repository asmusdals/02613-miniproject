import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


FIGURES_DIR = Path("reports/figures")
SUMMARY_PATH = Path("reports/q12_analysis_summary.txt")


def load_results(csv_path):
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    return df


def plot_histograms(df, figures_dir):
    figures_dir.mkdir(parents=True, exist_ok=True)

    cols = {
        "mean_temp":    ("Mean temperature (°C)",          "q12_hist_mean_temp.png"),
        "std_temp":     ("Std. dev. of temperature (°C)", "q12_hist_std_temp.png"),
        "pct_above_18": ("% area above 18°C",              "q12_hist_pct_above_18.png"),
        "pct_below_15": ("% area below 15°C",              "q12_hist_pct_below_15.png"),
    }

    for col, (xlabel, fname) in cols.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df[col], bins=50, edgecolor="white", linewidth=0.3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of buildings")
        ax.set_title(f"Distribution of {xlabel}")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures_dir / fname, dpi=200)
        plt.close(fig)

    return {col: figures_dir / fname for col, (_, fname) in cols.items()}


def analyse(df):
    n_total = len(df)
    avg_mean_temp = df["mean_temp"].mean()
    avg_std_temp = df["std_temp"].mean()
    n_above_18 = int((df["pct_above_18"] >= 50).sum())
    n_below_15 = int((df["pct_below_15"] >= 50).sum())
    return {
        "n_total": n_total,
        "avg_mean_temp": avg_mean_temp,
        "avg_std_temp": avg_std_temp,
        "n_above_18": n_above_18,
        "n_below_15": n_below_15,
    }


def write_summary(stats, plot_paths, summary_path):
    lines = [
        "Q12 analysis — all floorplans",
        "",
        f"Total buildings processed: {stats['n_total']}",
        "",
        "a) Distribution of mean temperatures",
        f"   Histogram saved to: {plot_paths['mean_temp']}",
        "",
        f"b) Average mean temperature across all buildings: {stats['avg_mean_temp']:.4f} °C",
        "",
        f"c) Average temperature standard deviation:        {stats['avg_std_temp']:.4f} °C",
        "",
        f"d) Buildings with >= 50% area above 18°C: {stats['n_above_18']} / {stats['n_total']} "
        f"({stats['n_above_18'] / stats['n_total'] * 100:.1f}%)",
        "",
        f"e) Buildings with >= 50% area below 15°C: {stats['n_below_15']} / {stats['n_total']} "
        f"({stats['n_below_15'] / stats['n_total'] * 100:.1f}%)",
    ]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyse full-dataset simulation results and answer Q12 sub-questions."
    )
    parser.add_argument(
        "csv",
        type=Path,
        help="Path to the simulation results CSV (e.g. results/q12_all.csv).",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=FIGURES_DIR,
        help="Directory to save histogram plots.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=SUMMARY_PATH,
        help="Path to save the text summary.",
    )
    args = parser.parse_args()

    df = load_results(args.csv)
    plot_paths = plot_histograms(df, args.figures_dir)
    stats = analyse(df)
    summary = write_summary(stats, plot_paths, args.summary_path)
    print(summary)


if __name__ == "__main__":
    main()

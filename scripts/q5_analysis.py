import argparse
import csv
import math
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULT_PATTERN = re.compile(r"q5_static_N(?P<n>\d+)_W(?P<workers>\d+)\.csv$")
TIME_ROW_PATTERN = re.compile(
    r"^\s*(?P<n>\d+)\s*,\s*(?P<workers>\d+)\s*,\s*(?P<elapsed>[0-9]+(?:\.[0-9]+)?)\s*$"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze Q5 static-scheduling speedups and produce report-ready outputs."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing q5_static_N*_W*.csv files.",
    )
    parser.add_argument(
        "--batch-output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory containing HPC .out logs with timing rows.",
    )
    parser.add_argument(
        "--batch-output-file",
        type=Path,
        default=None,
        help="Specific HPC .out file to parse for timings. Overrides directory auto-discovery.",
    )
    parser.add_argument(
        "--timings",
        type=str,
        default=None,
        help='Manual timings, e.g. "1=120,2=68,4=37,8=24". Overrides auto-discovery.',
    )
    parser.add_argument(
        "--timings-csv",
        type=Path,
        default=None,
        help="CSV file with columns workers,elapsed_seconds or N,workers,elapsed_seconds.",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=4571,
        help="Total number of floorplans in the full dataset.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("reports/figures/q5_speedup.png"),
        help="Where to save the speedup plot.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("reports/q5_analysis_summary.txt"),
        help="Where to save the text summary.",
    )
    return parser.parse_args()


def discover_result_files(results_dir):
    runs = {}
    for path in sorted(results_dir.glob("q5_static_N*_W*.csv")):
        match = RESULT_PATTERN.match(path.name)
        if not match:
            continue
        n = int(match.group("n"))
        workers = int(match.group("workers"))
        runs[workers] = {"n": n, "path": path}
    if not runs:
        raise FileNotFoundError(f"No q5 result CSV files found in {results_dir}")
    return runs


def parse_manual_timings(raw):
    timings = {}
    for item in raw.split(","):
        worker_str, elapsed_str = item.split("=")
        timings[int(worker_str.strip())] = float(elapsed_str.strip())
    return timings


def parse_timings_csv(path):
    timings = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty")
        for row in reader:
            workers = int(row["workers"])
            elapsed = float(row["elapsed_seconds"])
            timings[workers] = elapsed
    return timings


def parse_batch_output_timings(batch_output_dir):
    timings = {}
    if not batch_output_dir.exists():
        return timings

    for path in sorted(batch_output_dir.glob("*.out")):
        with path.open() as f:
            for line in f:
                match = TIME_ROW_PATTERN.match(line)
                if not match:
                    continue
                workers = int(match.group("workers"))
                elapsed = float(match.group("elapsed"))
                timings.setdefault(workers, []).append(elapsed)

    # If there are repeated measurements, keep the fastest observed run.
    return {workers: min(samples) for workers, samples in timings.items()}


def parse_batch_output_file(path):
    timings = {}
    with path.open() as f:
        for line in f:
            match = TIME_ROW_PATTERN.match(line)
            if not match:
                continue
            workers = int(match.group("workers"))
            elapsed = float(match.group("elapsed"))
            timings[workers] = elapsed
    return timings


def read_csv_rows(path):
    with path.open(newline="") as f:
        reader = csv.reader(f, skipinitialspace=True)
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def validate_outputs(runs):
    baseline_workers = min(runs)
    baseline_rows = read_csv_rows(runs[baseline_workers]["path"])
    baseline_count = len(baseline_rows) - 1

    issues = []
    for workers, meta in sorted(runs.items()):
        rows = read_csv_rows(meta["path"])
        if rows[0] != baseline_rows[0]:
            issues.append(f"W{workers}: CSV header differs from W{baseline_workers}")
        if len(rows) != len(baseline_rows):
            issues.append(
                f"W{workers}: row count {len(rows) - 1} differs from W{baseline_count}"
            )
        if rows != baseline_rows:
            issues.append(f"W{workers}: CSV contents differ from W{baseline_workers}")

    return {
        "baseline_workers": baseline_workers,
        "building_count": baseline_count,
        "issues": issues,
    }


def amdahl_speedup(parallel_fraction, workers):
    return 1.0 / ((1.0 - parallel_fraction) + parallel_fraction / workers)


def fit_parallel_fraction(observed_speedups):
    best_p = 0.0
    best_error = float("inf")
    for i in range(10001):
        p = i / 10000.0
        error = 0.0
        for workers, speedup in observed_speedups.items():
            model = amdahl_speedup(p, workers)
            error += (model - speedup) ** 2
        if error < best_error:
            best_error = error
            best_p = p
    return best_p


def format_duration(seconds):
    minutes = seconds / 60.0
    hours = minutes / 60.0
    if seconds < 60:
        return f"{seconds:.1f} s"
    if minutes < 60:
        return f"{minutes:.2f} min"
    return f"{hours:.2f} h"


def build_analysis(runs, timings, dataset_size):
    missing = sorted(set(runs) - set(timings))
    if missing:
        missing_str = ", ".join(str(worker) for worker in missing)
        raise ValueError(
            "Missing timings for worker counts: "
            f"{missing_str}. Provide --timings, --timings-csv, or batch_output/*.out logs."
        )

    workers_sorted = sorted(runs)
    n = {meta["n"] for meta in runs.values()}
    if len(n) != 1:
        raise ValueError(f"Expected one common N across runs, found {sorted(n)}")
    n = n.pop()

    t1 = timings[1]
    speedups = {workers: t1 / timings[workers] for workers in workers_sorted}
    efficiencies = {workers: speedups[workers] / workers for workers in workers_sorted}

    fitted_f = fit_parallel_fraction(speedups)
    serial_b = 1.0 - fitted_f
    theoretical_max = math.inf if fitted_f >= 1.0 else 1.0 / (1.0 - fitted_f)

    fastest_workers = min(workers_sorted, key=lambda workers: timings[workers])
    fastest_time = timings[fastest_workers]
    achieved_max_speedup = speedups[fastest_workers]
    achieved_fraction = (
        achieved_max_speedup / theoretical_max if math.isfinite(theoretical_max) else 1.0
    )
    estimated_full_runtime = fastest_time * (dataset_size / n)

    return {
        "n": n,
        "workers_sorted": workers_sorted,
        "timings": timings,
        "speedups": speedups,
        "efficiencies": efficiencies,
        "parallel_fraction": fitted_f,
        "serial_fraction": serial_b,
        "theoretical_max_speedup": theoretical_max,
        "fastest_workers": fastest_workers,
        "fastest_time": fastest_time,
        "achieved_max_speedup": achieved_max_speedup,
        "achieved_fraction_of_theoretical": achieved_fraction,
        "estimated_full_runtime": estimated_full_runtime,
        "dataset_size": dataset_size,
    }


def make_plot(analysis, plot_path):
    workers = analysis["workers_sorted"]
    observed = [analysis["speedups"][worker] for worker in workers]
    ideal = workers
    p = analysis["parallel_fraction"]

    fitted_workers = list(range(1, max(workers) + 1))
    fitted_curve = [amdahl_speedup(p, worker) for worker in fitted_workers]

    plot_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4.5))
    plt.plot(workers, observed, marker="o", linewidth=2, label="Observed speedup")
    plt.plot(workers, ideal, linestyle="--", linewidth=1.5, label="Ideal linear speedup")
    plt.plot(
        fitted_workers,
        fitted_curve,
        linestyle=":",
        linewidth=2,
        label=f"Amdahl fit (p={p:.3f})",
    )
    plt.xlabel("Workers")
    plt.ylabel("Speedup relative to 1 worker")
    plt.title(f"Q5 Static Scheduling Speedup (N={analysis['n']})")
    plt.xticks(workers)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


def write_summary(analysis, validation, summary_path, plot_path):
    lines = []
    lines.append("Q5 analysis")
    lines.append("")
    lines.append(f"Input size used for timing: N={analysis['n']} floorplans")
    lines.append(f"Estimated total dataset size: {analysis['dataset_size']} floorplans")
    lines.append("")
    lines.append("a) Measured speed-up as more workers are added")
    for worker in analysis["workers_sorted"]:
        elapsed = analysis["timings"][worker]
        speedup = analysis["speedups"][worker]
        efficiency = analysis["efficiencies"][worker] * 100.0
        lines.append(
            f"  W={worker}: time={elapsed:.3f} s, speedup={speedup:.3f}x, efficiency={efficiency:.1f}%"
        )
    lines.append(f"  Plot saved to {plot_path}")
    lines.append("")
    lines.append("b) Estimated parallel fraction from Amdahl's law")
    lines.append("  Slide notation:")
    lines.append("    S(p) = 1 / ((1 - F) + F/p) = 1 / (B + (1 - B)/p)")
    lines.append("    B = 1 - F")
    lines.append("    S(infinity) = 1 / (1 - F) = 1 / B")
    lines.append(
        f"  Estimated F ~= {analysis['parallel_fraction']:.4f} "
        f"({analysis['parallel_fraction'] * 100.0:.1f}% parallelized)"
    )
    lines.append(
        f"  Estimated B ~= {analysis['serial_fraction']:.4f} "
        f"({analysis['serial_fraction'] * 100.0:.1f}% serial)"
    )
    lines.append("")
    lines.append("c) Theoretical maximum speed-up and achieved fraction")
    lines.append(
        f"  Theoretical maximum speedup S(infinity) = 1/B = "
        f"{analysis['theoretical_max_speedup']:.3f}x"
    )
    lines.append(
        f"  Best observed speedup = {analysis['achieved_max_speedup']:.3f}x "
        f"using {analysis['fastest_workers']} workers"
    )
    lines.append(
        f"  Achieved {analysis['achieved_fraction_of_theoretical'] * 100.0:.1f}% "
        f"of the theoretical limit"
    )
    lines.append("")
    lines.append("d) Estimated runtime for all floorplans")
    lines.append(
        f"  Fastest measured configuration: {analysis['fastest_workers']} workers "
        f"with {analysis['fastest_time']:.3f} s for {analysis['n']} floorplans"
    )
    lines.append(
        f"  Estimated runtime for all {analysis['dataset_size']} floorplans: "
        f"{format_duration(analysis['estimated_full_runtime'])}"
    )
    lines.append("")
    lines.append("Output consistency check")
    if validation["issues"]:
        for issue in validation["issues"]:
            lines.append(f"  WARNING: {issue}")
    else:
        lines.append(
            "  All discovered q5 CSV outputs have matching headers, row counts, and values."
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def main():
    args = parse_args()

    runs = discover_result_files(args.results_dir)
    validation = validate_outputs(runs)

    if args.timings:
        timings = parse_manual_timings(args.timings)
    elif args.timings_csv:
        timings = parse_timings_csv(args.timings_csv)
    elif args.batch_output_file:
        timings = parse_batch_output_file(args.batch_output_file)
    else:
        timings = parse_batch_output_timings(args.batch_output_dir)

    analysis = build_analysis(runs, timings, args.dataset_size)
    make_plot(analysis, args.plot_path)
    summary = write_summary(analysis, validation, args.summary_path, args.plot_path)
    print(summary)


if __name__ == "__main__":
    main()

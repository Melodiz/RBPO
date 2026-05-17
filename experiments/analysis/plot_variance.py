#!/usr/bin/env python3
"""Plot gradient variance results from Stage 3.

Reads from {results_dir}/stage_3_grad_variance/, produces:
- Histogram of variance ratios across utterances
- Scatter: var_flat (x) vs var_rb (y) with y=x reference line
- Scatter: variance_ratio vs dead_frame_fraction (if Stage 2 data available)
- Bar plot: per-candidate gradient norms for one example utterance

Runs on M2, no GPU.

Usage:
    python analysis/plot_variance.py --results-dir results/
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_variance_stats(results_dir: Path) -> list[dict]:
    path = results_dir / "stage_3_grad_variance" / "grad_variance_results.csv"
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in ["G_effective", "num_unique_wer", "T", "L"]:
                row[k] = int(float(row[k]))
            for k in [
                "mean_var_flat", "mean_var_rb", "variance_ratio",
                "prop41_relative_diff", "mean_advantage_magnitude",
            ]:
                row[k] = float(row[k])
            rows.append(row)
    return rows


def load_variance_summary(results_dir: Path) -> dict:
    path = results_dir / "stage_3_grad_variance" / "grad_variance_summary.json"
    with open(path) as f:
        return json.load(f)


def load_gamma_stats(results_dir: Path) -> list[dict] | None:
    """Try to load Stage 2 gamma stats for correlation analysis."""
    path = results_dir / "stage_2_gamma_analysis" / "gamma_stats.csv"
    if not path.exists():
        return None
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def plot_variance_ratio_histogram(
    stats: list[dict], summary: dict, out: Path
):
    fig, ax = plt.subplots(figsize=(8, 5))

    ratios = [s["variance_ratio"] for s in stats]

    ax.hist(
        ratios, bins=30, color="#4CAF50", alpha=0.8,
        edgecolor="white",
    )

    mean_ratio = summary["mean_variance_ratio"]
    ax.axvline(
        mean_ratio, color="red", ls="--", linewidth=2,
        label=f"Mean: {mean_ratio:.3f}",
    )
    ax.axvline(
        1.0, color="black", ls=":", linewidth=1.5,
        label="Ratio = 1.0 (no reduction)",
    )

    ax.set_xlabel("Variance ratio (flat / RB)", fontsize=12)
    ax.set_ylabel("Count (utterances)", fontsize=12)
    ax.set_title(
        "Gradient Variance Ratio Distribution\n"
        f"(N={len(stats)} utterances, G={summary.get('G', 8)} candidates)",
        fontsize=14,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def plot_variance_scatter(stats: list[dict], out: Path):
    fig, ax = plt.subplots(figsize=(7, 7))

    x = [s["mean_var_flat"] for s in stats]
    y = [s["mean_var_rb"] for s in stats]

    ax.scatter(x, y, alpha=0.6, color="#2196F3", s=50, edgecolor="white")

    # y=x reference line
    all_vals = x + y
    lo = min(all_vals) * 0.5 if all_vals else 0
    hi = max(all_vals) * 1.5 if all_vals else 1
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5, label="y = x (equal)")

    ax.set_xlabel("Mean per-parameter variance (flat)", fontsize=12)
    ax.set_ylabel("Mean per-parameter variance (RB)", fontsize=12)
    ax.set_title(
        "Per-Utterance Gradient Variance: Flat vs RB\n"
        "(points below line = RB has lower variance)",
        fontsize=12,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def plot_ratio_vs_dead_frames(
    var_stats: list[dict], gamma_stats: list[dict], out: Path
):
    """Scatter: variance_ratio vs dead_frame_fraction from Stage 2."""
    # Match utterances by utt_id
    gamma_by_id = {row["utt_id"]: float(row["dead_frame_frac"])
                   for row in gamma_stats}

    x, y = [], []
    for row in var_stats:
        utt_id = row["utt_id"]
        if utt_id in gamma_by_id:
            x.append(gamma_by_id[utt_id])
            y.append(row["variance_ratio"])

    if len(x) < 5:
        print(f"  Only {len(x)} matched utterances  --  skipping correlation plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, alpha=0.6, color="#FF9800", s=50, edgecolor="white")

    # Trendline
    if len(x) >= 3:
        coeffs = np.polyfit(x, y, 1)
        xs = np.linspace(min(x), max(x), 50)
        ax.plot(xs, np.polyval(coeffs, xs), "r--", linewidth=2,
                label=f"slope={coeffs[0]:.3f}")
        ax.legend(fontsize=10)

    ax.axhline(1.0, color="black", ls=":", linewidth=1, alpha=0.5)

    ax.set_xlabel("Dead frame fraction (gamma_t(blank) > 0.99)", fontsize=12)
    ax.set_ylabel("Variance ratio (flat / RB)", fontsize=12)
    ax.set_title(
        "Variance Reduction vs Frame Sparsity\n"
        "(do dead-frame-heavy utterances benefit more from RB?)",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def plot_prop41_verification(stats: list[dict], out: Path):
    """Bar plot showing Prop 4.1 relative diff per utterance."""
    fig, ax = plt.subplots(figsize=(10, 4))

    diffs = [s["prop41_relative_diff"] for s in stats]
    idx = range(len(diffs))

    colors = ["#4CAF50" if d < 0.01 else "#FF9800" if d < 0.1 else "#F44336"
              for d in diffs]
    ax.bar(idx, diffs, color=colors, alpha=0.8, width=0.8)
    ax.axhline(0.1, color="red", ls="--", linewidth=1.5,
               label="Threshold (0.1)")

    ax.set_xlabel("Utterance index", fontsize=12)
    ax.set_ylabel("Relative gradient difference", fontsize=12)
    ax.set_title(
        "Proposition 4.1 Verification: Weighted Gradient Equivalence",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot Stage 3 variance results")
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"),
    )
    parser.add_argument("--figures-dir", type=Path, default=None)
    args = parser.parse_args()

    figures_dir = args.figures_dir or args.results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Stage 3 results...")
    stats = load_variance_stats(args.results_dir)
    summary = load_variance_summary(args.results_dir)
    print(f"  {len(stats)} utterances, mean ratio: {summary['mean_variance_ratio']:.4f}")

    plot_variance_ratio_histogram(
        stats, summary, figures_dir / "grad_variance_ratio_histogram.png"
    )
    plot_variance_scatter(
        stats, figures_dir / "grad_variance_flat_vs_rb.png"
    )
    plot_prop41_verification(
        stats, figures_dir / "grad_variance_prop41.png"
    )

    # Correlation with Stage 2 dead frames (if available)
    gamma_stats = load_gamma_stats(args.results_dir)
    if gamma_stats:
        print("  Found Stage 2 gamma stats  --  plotting correlation...")
        plot_ratio_vs_dead_frames(
            stats, gamma_stats,
            figures_dir / "grad_variance_vs_dead_frames.png",
        )
    else:
        print("  No Stage 2 data found  --  skipping correlation plot")

    print("\nAll plots saved.")


if __name__ == "__main__":
    main()

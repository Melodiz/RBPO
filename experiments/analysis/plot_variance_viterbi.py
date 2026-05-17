#!/usr/bin/env python3
"""Plot Stage 3b Viterbi-vs-CTC gradient variance results.

Reads from {results_dir}/stage_3b_viterbi_variance/, produces:
- Bar chart: mean variance for CTC / Viterbi / Sampled (3 bars)
- Histogram of ratio_viterbi_vs_ctc
- Scatter: ratio_viterbi_vs_ctc vs T/L (compression ratio)

Runs on M2, no GPU.

Usage:
    python analysis/plot_variance_viterbi.py --results-dir results/
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_stats(results_dir: Path) -> list[dict]:
    path = results_dir / "stage_3b_viterbi_variance" / "viterbi_variance_results.csv"
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in ["G_effective", "T", "L"]:
                row[k] = int(float(row[k]))
            for k in [
                "mean_var_ctc", "mean_var_viterbi", "mean_var_sampled",
                "ratio_viterbi_vs_ctc", "ratio_sampled_vs_ctc",
            ]:
                row[k] = float(row[k])
            rows.append(row)
    return rows


def load_summary(results_dir: Path) -> dict:
    path = results_dir / "stage_3b_viterbi_variance" / "viterbi_variance_summary.json"
    with open(path) as f:
        return json.load(f)


def plot_estimator_bars(stats: list[dict], summary: dict, out: Path):
    fig, ax = plt.subplots(figsize=(7, 5))

    means = [
        np.mean([s["mean_var_ctc"] for s in stats]),
        np.mean([s["mean_var_viterbi"] for s in stats]),
        np.mean([s["mean_var_sampled"] for s in stats]),
    ]
    labels = ["CTC-marginalized\n(gamma-weighted)", "Viterbi\n(one-hot best)", "Sampled\n(one-hot ~ posterior)"]
    colors = ["#2196F3", "#FF9800", "#F44336"]

    bars = ax.bar(labels, means, color=colors, alpha=0.85, edgecolor="white")

    for bar, m in zip(bars, means):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h,
            f"{m:.2e}",
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_yscale("log")
    ax.set_ylabel("Mean per-parameter gradient variance", fontsize=12)
    ax.set_title(
        f"Gradient Variance by Estimator\n"
        f"(N={len(stats)} utterances, G={summary.get('G', 8)} candidates)",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def plot_ratio_histogram(stats: list[dict], summary: dict, out: Path):
    fig, ax = plt.subplots(figsize=(8, 5))

    ratios_v = [s["ratio_viterbi_vs_ctc"] for s in stats]
    ratios_s = [s["ratio_sampled_vs_ctc"] for s in stats]

    bins = np.linspace(0.5, max(max(ratios_v), max(ratios_s)) * 1.05, 30)

    ax.hist(
        ratios_v, bins=bins, color="#FF9800", alpha=0.7,
        edgecolor="white", label=f"Viterbi/CTC (mean={np.mean(ratios_v):.2f})",
    )
    ax.hist(
        ratios_s, bins=bins, color="#F44336", alpha=0.6,
        edgecolor="white", label=f"Sampled/CTC (mean={np.mean(ratios_s):.2f})",
    )

    ax.axvline(1.0, color="black", ls=":", linewidth=1.5,
               label="Ratio = 1.0 (no reduction)")

    ax.set_xlabel("Variance ratio (one-hot estimator / CTC-marginalized)", fontsize=12)
    ax.set_ylabel("Count (utterances)", fontsize=12)
    ax.set_title(
        "Variance Reduction from CTC Marginalization\n"
        "(values > 1 mean CTC has lower variance)",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def plot_ratio_vs_compression(stats: list[dict], out: Path):
    fig, ax = plt.subplots(figsize=(8, 5))

    x = [s["T"] / s["L"] if s["L"] > 0 else 0 for s in stats]
    y = [s["ratio_viterbi_vs_ctc"] for s in stats]

    ax.scatter(x, y, alpha=0.6, color="#FF9800", s=50, edgecolor="white",
               label="Viterbi / CTC")

    # Sampled points too
    y2 = [s["ratio_sampled_vs_ctc"] for s in stats]
    ax.scatter(x, y2, alpha=0.5, color="#F44336", s=40, edgecolor="white",
               marker="^", label="Sampled / CTC")

    if len(x) >= 3:
        coeffs = np.polyfit(x, y, 1)
        xs = np.linspace(min(x), max(x), 50)
        ax.plot(xs, np.polyval(coeffs, xs), "k--", linewidth=1.5,
                label=f"Viterbi trend: slope={coeffs[0]:.2f}")

    ax.axhline(1.0, color="black", ls=":", linewidth=1, alpha=0.5)

    ax.set_xlabel("Compression ratio T / L (frames per label)", fontsize=12)
    ax.set_ylabel("Variance ratio vs CTC-marginalized", fontsize=12)
    ax.set_title(
        "More Compression -> More Alignment Ambiguity -> Bigger RB Win",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def plot_per_utterance_variance(stats: list[dict], out: Path):
    """Scatter of var_ctc vs var_viterbi per utterance, log-log."""
    fig, ax = plt.subplots(figsize=(7, 7))

    x = [s["mean_var_ctc"] for s in stats]
    y_v = [s["mean_var_viterbi"] for s in stats]
    y_s = [s["mean_var_sampled"] for s in stats]

    ax.scatter(x, y_v, alpha=0.6, color="#FF9800", s=50, edgecolor="white",
               label="Viterbi")
    ax.scatter(x, y_s, alpha=0.5, color="#F44336", s=40, edgecolor="white",
               marker="^", label="Sampled")

    all_vals = x + y_v + y_s
    lo = min(all_vals) * 0.5 if all_vals else 1e-10
    hi = max(all_vals) * 1.5 if all_vals else 1
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5, label="y = x (equal)")

    ax.set_xlabel("Variance (CTC-marginalized)", fontsize=12)
    ax.set_ylabel("Variance (one-hot alignment)", fontsize=12)
    ax.set_title(
        "Per-Utterance Variance: CTC vs One-Hot Estimators\n"
        "(points above line = one-hot has higher variance)",
        fontsize=12,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_aspect("equal")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot Stage 3b results")
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"),
    )
    parser.add_argument("--figures-dir", type=Path, default=None)
    args = parser.parse_args()

    figures_dir = args.figures_dir or args.results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Stage 3b results...")
    stats = load_stats(args.results_dir)
    summary = load_summary(args.results_dir)
    print(
        f"  {len(stats)} utterances, "
        f"viterbi/ctc mean: {summary['mean_ratio_viterbi_vs_ctc']:.3f}"
    )

    plot_estimator_bars(
        stats, summary,
        figures_dir / "viterbi_variance_bars.png",
    )
    plot_ratio_histogram(
        stats, summary,
        figures_dir / "viterbi_variance_ratio_histogram.png",
    )
    plot_ratio_vs_compression(
        stats,
        figures_dir / "viterbi_variance_vs_compression.png",
    )
    plot_per_utterance_variance(
        stats,
        figures_dir / "viterbi_variance_scatter.png",
    )

    print("\nAll plots saved.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot oracle WER results from Stage 1.

Reads oracle_wer_summary.json and oracle_wer_results.csv, produces three plots.
Runs on M2 (no GPU needed).

Usage:
    python analysis/plot_oracle_wer.py --results-dir results/
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_summary(results_dir: Path) -> dict:
    path = results_dir / "stage_1_oracle_wer" / "oracle_wer_summary.json"
    with open(path) as f:
        return json.load(f)


def load_per_utterance(results_dir: Path) -> list[dict]:
    path = results_dir / "stage_1_oracle_wer" / "oracle_wer_results.csv"
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["oracle_wer"] = float(row["oracle_wer"])
            row["onebest_wer"] = float(row["onebest_wer"])
            row["mean_wer"] = float(row["mean_wer"])
            row["num_unique"] = int(row["num_unique"])
            row["G"] = int(row["G"])
            row["nbest_scale"] = float(row["nbest_scale"])
            rows.append(row)
    return rows


def plot_oracle_wer_vs_g(summary: dict, figures_dir: Path):
    """Line plot: Oracle WER (y) vs G (x), one curve per (method, nbest_scale)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    style_map = {
        ("beam", 1.0):    {"color": "#2196F3", "marker": "s", "ls": "-",  "label": "Beam (scale=1.0)"},
        ("lattice", 0.75): {"color": "#FF9800", "marker": "^", "ls": "--", "label": "Lattice (scale=0.75)"},
        ("lattice", 0.5):  {"color": "#4CAF50", "marker": "o", "ls": "--", "label": "Lattice (scale=0.5)"},
        ("lattice", 0.25): {"color": "#F44336", "marker": "D", "ls": "--", "label": "Lattice (scale=0.25)"},
    }

    series = defaultdict(lambda: {"G": [], "wer": []})
    for entry in summary["conditions"]:
        key = (entry["method"], entry["nbest_scale"])
        series[key]["G"].append(entry["G"])
        series[key]["wer"].append(entry["mean_oracle_wer"] * 100)

    for key, data in series.items():
        style = style_map.get(key, {"color": "gray", "marker": "x", "ls": ":", "label": str(key)})
        order = np.argsort(data["G"])
        gs = np.array(data["G"])[order]
        wers = np.array(data["wer"])[order]
        ax.plot(gs, wers, marker=style["marker"], ls=style["ls"],
                color=style["color"], label=style["label"], linewidth=2, markersize=8)

    # Add 1-best reference line
    onebest = summary["conditions"][0]["mean_onebest_wer"] * 100
    ax.axhline(y=onebest, color="gray", ls=":", linewidth=1, alpha=0.7)
    ax.text(max(data["G"]) * 0.95, onebest + 0.1, f"1-best: {onebest:.1f}%",
            ha="right", va="bottom", color="gray", fontsize=9)

    ax.set_xlabel("Number of candidates (G)", fontsize=12)
    ax.set_ylabel("Oracle WER (%)", fontsize=12)
    ax.set_title("Oracle WER vs Number of Candidates", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xticks([4, 8, 16])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = figures_dir / "oracle_wer_vs_g.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def plot_unique_candidates(summary: dict, figures_dir: Path):
    """Bar plot: Mean unique candidates for each condition at G=8."""
    fig, ax = plt.subplots(figsize=(8, 5))

    g8 = [e for e in summary["conditions"] if e["G"] == 8]
    labels = []
    values = []
    colors = []
    color_map = {
        ("beam", 1.0): "#2196F3",
        ("lattice", 0.75): "#FF9800",
        ("lattice", 0.5): "#4CAF50",
        ("lattice", 0.25): "#F44336",
    }

    for entry in g8:
        key = (entry["method"], entry["nbest_scale"])
        label = f"{entry['method'].capitalize()}\nscale={entry['nbest_scale']}"
        labels.append(label)
        values.append(entry["mean_num_unique"])
        colors.append(color_map.get(key, "gray"))

    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Mean unique candidates", fontsize=12)
    ax.set_title("Hypothesis Diversity at G=8", fontsize=14)
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out = figures_dir / "oracle_wer_unique_candidates.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def plot_oracle_wer_distribution(rows: list[dict], figures_dir: Path):
    """Box plot: Distribution of per-utterance oracle WER for beam vs lattice at G=8."""
    fig, ax = plt.subplots(figsize=(8, 5))

    beam_g8 = [r["oracle_wer"] * 100 for r in rows
               if r["method"] == "beam" and r["nbest_scale"] == 1.0 and r["G"] == 8]
    lattice_g8 = [r["oracle_wer"] * 100 for r in rows
                  if r["method"] == "lattice" and r["nbest_scale"] == 0.5 and r["G"] == 8]

    bp = ax.boxplot(
        [beam_g8, lattice_g8],
        labels=["Beam\n(scale=1.0, G=8)", "Lattice\n(scale=0.5, G=8)"],
        patch_artist=True,
        widths=0.5,
    )
    bp["boxes"][0].set_facecolor("#2196F3")
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor("#4CAF50")
    bp["boxes"][1].set_alpha(0.6)

    for i, data in enumerate([beam_g8, lattice_g8], 1):
        mean_val = np.mean(data)
        ax.plot(i, mean_val, "D", color="red", markersize=6, zorder=5)
        ax.text(i + 0.15, mean_val, f"mean={mean_val:.2f}%",
                va="center", fontsize=9, color="red")

    ax.set_ylabel("Oracle WER (%)", fontsize=12)
    ax.set_title("Oracle WER Distribution: Beam vs Lattice Sampling", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out = figures_dir / "oracle_wer_distribution.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot Stage 1 oracle WER results")
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"),
        help="Directory containing stage_1_oracle_wer/ subdirectory",
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=None,
        help="Output directory for figures (default: results/figures/)",
    )
    args = parser.parse_args()

    figures_dir = args.figures_dir or args.results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    summary = load_summary(args.results_dir)
    rows = load_per_utterance(args.results_dir)
    print(f"  {len(summary['conditions'])} conditions, {len(rows)} rows")

    plot_oracle_wer_vs_g(summary, figures_dir)
    plot_unique_candidates(summary, figures_dir)
    plot_oracle_wer_distribution(rows, figures_dir)

    print("\nAll plots saved.")


if __name__ == "__main__":
    main()

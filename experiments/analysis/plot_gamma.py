#!/usr/bin/env python3
"""Plot gamma_t(k|y) analysis results from Stage 2.

Reads from {results_dir}/stage_2_gamma_analysis/, produces:
- Histogram of dead frame fraction across utterances
- Histogram of per-frame entropy (pooled across all utterances)
- 3 heatmaps (short/medium/long example utterances)
- Scatter: dead_frame_fraction vs compression_ratio

Runs on M2, no GPU.

Usage:
    python analysis/plot_gamma.py --results-dir results/
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_stats(results_dir: Path) -> list[dict]:
    path = results_dir / "stage_2_gamma_analysis" / "gamma_stats.csv"
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in [
                "T", "L", "compression_ratio", "dead_frame_frac",
                "near_dead_frac", "ambiguous_frac", "mean_entropy",
                "max_entropy", "label_frame_frac", "wer",
            ]:
                row[k] = float(row[k]) if k not in {"T", "L"} else int(float(row[k]))
            rows.append(row)
    return rows


def load_summary(results_dir: Path) -> dict:
    path = results_dir / "stage_2_gamma_analysis" / "gamma_summary.json"
    with open(path) as f:
        return json.load(f)


def plot_dead_frame_histogram(stats: list[dict], summary: dict, out: Path):
    fig, ax = plt.subplots(figsize=(8, 5))

    fracs = [s["dead_frame_frac"] for s in stats]
    near_fracs = [s["near_dead_frac"] for s in stats]

    ax.hist(fracs, bins=30, color="#2196F3", alpha=0.7,
            edgecolor="white", label="gamma_t(blank) > 0.99")
    ax.hist(near_fracs, bins=30, color="#FF9800", alpha=0.5,
            edgecolor="white", label="gamma_t(blank) > 0.95")

    mean_dead = summary["mean_dead_frame_fraction"]
    ax.axvline(mean_dead, color="red", ls="--", linewidth=2,
               label=f"Mean: {mean_dead:.3f}")

    ax.set_xlabel("Dead frame fraction (per utterance)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("CTC Alignment Posterior Sparsity", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def plot_entropy_histogram(results_dir: Path, out: Path):
    """Pool per-frame entropies across example heatmaps (full distribution)."""
    examples_dir = results_dir / "stage_2_gamma_analysis" / "gamma_examples"
    all_entropies = []

    for tag in ["short", "medium", "long"]:
        gamma_path = examples_dir / f"example_{tag}_gamma.npy"
        meta_path = examples_dir / f"example_{tag}_meta.json"
        if not gamma_path.exists():
            continue
        gamma = np.load(gamma_path)
        with open(meta_path) as f:
            meta = json.load(f)
        T = meta["T"]
        gamma_valid = gamma[:T]
        eps = 1e-10
        ent = -(gamma_valid * np.log(gamma_valid + eps)).sum(axis=-1)
        all_entropies.extend(ent.tolist())

    if not all_entropies:
        print(f"No example data  --  skipping entropy histogram")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_entropies, bins=60, color="#4CAF50",
            edgecolor="white", alpha=0.8)
    ax.set_xlabel("Per-frame entropy H_t (nats)", fontsize=12)
    ax.set_ylabel("Frame count", fontsize=12)
    ax.set_title(
        f"Per-Frame Entropy Distribution\n"
        f"(pooled from {len(all_entropies)} frames across 3 example utterances)",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def plot_heatmap(results_dir: Path, tag: str, out: Path):
    examples_dir = results_dir / "stage_2_gamma_analysis" / "gamma_examples"
    gamma_path = examples_dir / f"example_{tag}_gamma.npy"
    meta_path = examples_dir / f"example_{tag}_meta.json"
    if not gamma_path.exists():
        print(f"No data for {tag}  --  skipping")
        return

    gamma = np.load(gamma_path)
    with open(meta_path) as f:
        meta = json.load(f)

    T = meta["T"]
    token_ids = meta["token_ids"]
    gamma_valid = gamma[:T]  # (T, V)

    # Active labels: blank + tokens in hypothesis (preserve order)
    seen = set()
    active = [0]
    seen.add(0)
    for t in token_ids:
        if t not in seen:
            active.append(t)
            seen.add(t)

    sub = gamma_valid[:, active].T  # (num_active, T)

    # Subsample frames if T is large for readability
    if T > 200:
        stride = max(1, T // 200)
        sub = sub[:, ::stride]
        x_label = f"Frame index (every {stride}-th frame, T={T})"
    else:
        x_label = f"Frame index (T={T})"

    fig, ax = plt.subplots(figsize=(12, max(4, len(active) * 0.25)))
    im = ax.imshow(sub, aspect="auto", cmap="viridis", vmin=0, vmax=1)

    y_ticks = list(range(len(active)))
    y_labels = ["blank" if k == 0 else f"tok_{k}" for k in active]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("Active labels", fontsize=11)

    title = (
        f"gamma_t(k|y) heatmap  --  {tag} utterance ({meta['duration_sec']:.1f}s, "
        f"T={T}, L={meta['L']})\n"
        f"Ref: {meta['ref_text'][:80]}{'...' if len(meta['ref_text']) > 80 else ''}\n"
        f"Hyp: {meta['hyp_text'][:80]}{'...' if len(meta['hyp_text']) > 80 else ''}"
    )
    ax.set_title(title, fontsize=10)

    fig.colorbar(im, ax=ax, label="gamma_t(k|y)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def plot_dead_vs_compression(stats: list[dict], out: Path):
    fig, ax = plt.subplots(figsize=(8, 5))

    x = [s["compression_ratio"] for s in stats]
    y = [s["dead_frame_frac"] for s in stats]

    ax.scatter(x, y, alpha=0.6, color="#2196F3", s=40, edgecolor="white")

    # Trendline
    if len(x) >= 2:
        coeffs = np.polyfit(x, y, 1)
        xs = np.linspace(min(x), max(x), 50)
        ax.plot(xs, np.polyval(coeffs, xs), "r--", linewidth=2,
                label=f"slope={coeffs[0]:.3f}")
        ax.legend(fontsize=10)

    ax.set_xlabel("Compression ratio T / L", fontsize=12)
    ax.set_ylabel("Dead frame fraction (gamma_t(blank) > 0.99)", fontsize=12)
    ax.set_title(
        "Sparsity vs Compression: Longer Frames-per-Label -> More Dead Frames",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot Stage 2 gamma_t analysis")
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"),
    )
    parser.add_argument("--figures-dir", type=Path, default=None)
    args = parser.parse_args()

    figures_dir = args.figures_dir or args.results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    stats = load_stats(args.results_dir)
    summary = load_summary(args.results_dir)
    print(f"  {len(stats)} utterances")

    plot_dead_frame_histogram(
        stats, summary, figures_dir / "gamma_dead_frame_histogram.png"
    )
    plot_entropy_histogram(
        args.results_dir, figures_dir / "gamma_entropy_histogram.png"
    )
    plot_dead_vs_compression(
        stats, figures_dir / "gamma_dead_vs_compression.png"
    )

    for tag in ["short", "medium", "long"]:
        plot_heatmap(
            args.results_dir, tag,
            figures_dir / f"gamma_heatmap_{tag}.png",
        )

    print("\nAll plots saved.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Level 1b Part B: Diversity x Temperature Sweep.

Runs temperature-scaled MBR-CER across multiple nbest_scale files and
produces a comparison table, heatmap, and report.

Expects:
    results/nbest_dev_other_G16.jsonl           (scale=1.0, from Level 1)
    results/nbest_dev_other_G16_scale0.50.jsonl (scale=0.5)
    results/nbest_dev_other_G16_scale0.75.jsonl (scale=0.75)

Usage:
    python -m experiments.analysis.partB_analysis --results-dir results
"""

import argparse
import csv
import json
import time
from itertools import combinations
from pathlib import Path

import editdistance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

TEMPERATURES = [1.0, 2.0, 5.0, 8.0, 10.0, 20.0, 50.0, float("inf")]
SCALES = [0.5, 0.75, 1.0]


def load_nbest(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def compute_wer(hyp: str, ref: str) -> float:
    ref_w = ref.split()
    hyp_w = hyp.split()
    if len(ref_w) == 0:
        return 0.0 if len(hyp_w) == 0 else 1.0
    return editdistance.eval(hyp_w, ref_w) / len(ref_w)


def char_distance(a, b):
    denom = max(len(a), len(b), 1)
    return editdistance.eval(list(a), list(b)) / denom


def tempered_softmax(log_probs: list[float], tau: float) -> np.ndarray:
    a = np.array(log_probs, dtype=np.float64)
    if tau == float("inf"):
        return np.ones(len(a)) / len(a)
    a_scaled = a / tau
    a_scaled -= np.max(a_scaled)
    w = np.exp(a_scaled)
    return w / w.sum()


def entropy(weights: np.ndarray) -> float:
    w = weights[weights > 1e-30]
    return -np.sum(w * np.log(w))


def mbr_cer_select(texts, log_probs, tau):
    n = len(texts)
    if n == 1:
        return 0
    weights = tempered_softmax(log_probs, tau)
    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i != j:
                scores[i] += weights[j] * char_distance(texts[i], texts[j])
    return int(np.argmin(scores))


def compute_diversity(records: list[dict]) -> dict:
    unique_counts = []
    pairwise_weds = []

    for rec in records:
        texts = list({c["text"].strip().lower() for c in rec["candidates"]})
        unique_counts.append(len(texts))
        if len(texts) >= 2:
            dists = []
            for a, b in combinations(texts, 2):
                wa, wb = a.split(), b.split()
                denom = max(len(wa), len(wb), 1)
                dists.append(editdistance.eval(wa, wb) / denom)
            pairwise_weds.append(float(np.mean(dists)))
        else:
            pairwise_weds.append(0.0)

    return {
        "mean_unique": float(np.mean(unique_counts)),
        "mean_pairwise_wed": float(np.mean(pairwise_weds)),
    }


def compute_baseline_wers(records: list[dict]):
    greedy_num, greedy_den = 0, 0
    oracle_num, oracle_den = 0, 0
    onebest_num, onebest_den = 0, 0

    for rec in records:
        ref = rec["ref_text"]
        ref_w = ref.split()
        n_ref = len(ref_w)
        cands = rec["candidates"]

        # candidates[0] is the 1-best from this lattice extraction
        onebest = cands[0]["text"]
        onebest_num += editdistance.eval(onebest.split(), ref_w)
        onebest_den += n_ref

        # greedy is also candidates[0] for scale=1.0; for other scales
        # it's still the frame-level argmax, which is always cands[0]
        # because generate_nbest.py forces greedy to position 0
        greedy_num += editdistance.eval(cands[0]["text"].split(), ref_w)
        greedy_den += n_ref

        wers = [compute_wer(c["text"], ref) for c in cands]
        best_idx = int(np.argmin(wers))
        oracle_num += editdistance.eval(cands[best_idx]["text"].split(), ref_w)
        oracle_den += n_ref

    return {
        "greedy_wer": greedy_num / max(greedy_den, 1),
        "oracle_wer": oracle_num / max(oracle_den, 1),
        "onebest_wer": onebest_num / max(onebest_den, 1),
    }


def sweep_one_scale(records, temperatures):
    baselines = compute_baseline_wers(records)
    greedy_wer = baselines["greedy_wer"]
    oracle_wer = baselines["oracle_wer"]
    gap = greedy_wer - oracle_wer

    rows = []
    for tau in temperatures:
        wer_num, wer_den = 0, 0
        entropies = []
        n_differ = 0

        for rec in records:
            ref = rec["ref_text"]
            cands = rec["candidates"]
            texts = [c["text"] for c in cands]
            log_probs = [c["ctc_log_prob"] for c in cands]
            ref_w = ref.split()

            weights = tempered_softmax(log_probs, tau)
            entropies.append(entropy(weights))

            idx = mbr_cer_select(texts, log_probs, tau)
            wer_num += editdistance.eval(texts[idx].split(), ref_w)
            wer_den += len(ref_w)
            if idx != 0:
                n_differ += 1

        mbr_wer = wer_num / max(wer_den, 1)
        gap_closed = (greedy_wer - mbr_wer) / gap * 100 if gap > 1e-9 else 0.0

        rows.append({
            "tau": tau,
            "mbr_cer_wer": mbr_wer,
            "gap_closed": gap_closed,
            "mean_entropy": float(np.mean(entropies)),
            "frac_differ": n_differ / len(records),
        })

    return rows, baselines


def plot_heatmap(all_data, plots_dir: Path):
    scales = sorted(all_data.keys())
    taus = TEMPERATURES
    tau_labels = [f"{t:.0f}" if t != float("inf") else "inf" for t in taus]

    wer_grid = np.full((len(scales), len(taus)), np.nan)
    for si, scale in enumerate(scales):
        for row in all_data[scale]["sweep"]:
            ti = taus.index(row["tau"])
            wer_grid[si, ti] = row["mbr_cer_wer"] * 100

    best_si, best_ti = np.unravel_index(np.nanargmin(wer_grid), wer_grid.shape)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(wer_grid, aspect="auto", cmap="RdYlGn_r",
                   vmin=np.nanmin(wer_grid) - 0.05,
                   vmax=np.nanmax(wer_grid) + 0.05)

    ax.set_xticks(range(len(tau_labels)))
    ax.set_xticklabels(tau_labels)
    ax.set_yticks(range(len(scales)))
    ax.set_yticklabels([f"{s:.2f}" for s in scales])
    ax.set_xlabel("Temperature tau")
    ax.set_ylabel("nbest_scale")
    ax.set_title("MBR-CER WER% across Scale x Temperature")

    for i in range(len(scales)):
        for j in range(len(taus)):
            if not np.isnan(wer_grid[i, j]):
                color = "white" if wer_grid[i, j] > np.nanmedian(wer_grid) else "black"
                weight = "bold" if (i == best_si and j == best_ti) else "normal"
                ax.text(j, i, f"{wer_grid[i, j]:.2f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight=weight)

    # Mark best cell
    rect = plt.Rectangle((best_ti - 0.5, best_si - 0.5), 1, 1,
                          linewidth=2.5, edgecolor="#e74c3c", facecolor="none")
    ax.add_patch(rect)

    fig.colorbar(im, ax=ax, label="WER %", shrink=0.8)
    fig.tight_layout()
    out_path = plots_dir / "temperature_diversity_heatmap.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_curves(all_data, greedy_wer_ref, plots_dir: Path):
    """WER vs tau curves, one line per scale."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {0.5: "#e74c3c", 0.75: "#e67e22", 1.0: "#3498db"}
    markers = {0.5: "D", 0.75: "s", 1.0: "o"}

    for scale in sorted(all_data.keys()):
        sweep = all_data[scale]["sweep"]
        taus = [r["tau"] if r["tau"] != float("inf") else 100.0 for r in sweep]
        wers = [r["mbr_cer_wer"] * 100 for r in sweep]
        ax.semilogx(taus, wers, f"{markers.get(scale, 'o')}-",
                    color=colors.get(scale, "#7f8c8d"),
                    linewidth=2, markersize=6,
                    label=f"scale={scale:.2f}", zorder=3)

    ax.axhline(greedy_wer_ref * 100, color="#95a5a6", linewidth=1.5,
               linestyle=":", label=f"Greedy = {greedy_wer_ref*100:.2f}%", zorder=1)

    ax.set_xlabel("Temperature tau (log scale; inf shown as 100)")
    ax.set_ylabel("MBR-CER WER %")
    ax.set_title("Temperature-Scaled MBR by Candidate Diversity")
    ax.legend(fontsize=9)
    out_path = plots_dir / "temperature_diversity_curves.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def generate_report(all_data, elapsed, results_dir: Path):
    scales = sorted(all_data.keys())
    ref_greedy = all_data[1.0]["baselines"]["greedy_wer"]
    ref_gap = ref_greedy - all_data[1.0]["baselines"]["oracle_wer"]

    lines = []
    lines.append("# Level 1b Part B: Diversity x Temperature Sweep\n")
    lines.append(f"**Dataset:** LibriSpeech dev-other (2864 utterances)")
    lines.append("**Model:** Zipformer-S CR-CTC, BPE-500")
    lines.append(f"**Reference greedy WER:** {ref_greedy*100:.2f}%")
    lines.append(f"**Temperatures:** {', '.join(str(t) if t != float('inf') else 'inf' for t in TEMPERATURES)}\n")

    # Per-scale diversity table
    lines.append("## Candidate Diversity by Scale\n")
    lines.append("| Scale | Unique (mean) | Pairwise WED | Oracle WER | Greedy WER |")
    lines.append("|------:|--------------:|-------------:|-----------:|-----------:|")
    for s in scales:
        d = all_data[s]
        b = d["baselines"]
        div = d["diversity"]
        lines.append(f"| {s:.2f} | {div['mean_unique']:.1f} | "
                     f"{div['mean_pairwise_wed']*100:.1f}% | "
                     f"{b['oracle_wer']*100:.2f}% | {b['greedy_wer']*100:.2f}% |")
    lines.append("")

    # Full sweep table
    lines.append("## Temperature Sweep Results\n")
    lines.append("| Scale | tau | MBR-CER WER% | Gap Closed | Entropy | % Differ |")
    lines.append("|------:|--:|-------------:|-----------:|--------:|---------:|")
    for s in scales:
        for r in all_data[s]["sweep"]:
            tau_s = "inf" if r["tau"] == float("inf") else f"{r['tau']:.0f}"
            lines.append(f"| {s:.2f} | {tau_s} | {r['mbr_cer_wer']*100:.2f}% | "
                         f"{r['gap_closed']:+.1f}% | {r['mean_entropy']:.3f} | "
                         f"{r['frac_differ']*100:.1f}% |")
    lines.append("")

    # Best per scale
    lines.append("## Best Configuration per Scale\n")
    lines.append("| Scale | tau* | Best WER% | Gap Closed | vs Greedy | Diversity | Oracle WER |")
    lines.append("|------:|---:|----------:|-----------:|----------:|----------:|-----------:|")

    global_best_wer = float("inf")
    global_best_scale = None
    global_best_tau = None

    for s in scales:
        sweep = all_data[s]["sweep"]
        best = min(sweep, key=lambda r: r["mbr_cer_wer"])
        tau_s = "inf" if best["tau"] == float("inf") else f"{best['tau']:.0f}"
        b = all_data[s]["baselines"]
        div = all_data[s]["diversity"]
        delta = best["mbr_cer_wer"] - ref_greedy
        lines.append(f"| {s:.2f} | {tau_s} | {best['mbr_cer_wer']*100:.2f}% | "
                     f"{best['gap_closed']:+.1f}% | {delta*100:+.2f} pp | "
                     f"{div['mean_pairwise_wed']*100:.1f}% | {b['oracle_wer']*100:.2f}% |")

        if best["mbr_cer_wer"] < global_best_wer:
            global_best_wer = best["mbr_cer_wer"]
            global_best_scale = s
            global_best_tau = best["tau"]

    lines.append("")

    gb_tau_s = "inf" if global_best_tau == float("inf") else f"{global_best_tau}"
    gb_gap_closed = (ref_greedy - global_best_wer) / ref_gap * 100 if ref_gap > 1e-9 else 0.0
    lines.append(f"**Global best:** scale={global_best_scale}, tau={gb_tau_s} -> "
                 f"WER={global_best_wer*100:.2f}%, gap closed={gb_gap_closed:+.1f}%\n")

    lines.append("![Heatmap](plots/temperature_diversity_heatmap.png)\n")
    lines.append("![Curves](plots/temperature_diversity_curves.png)\n")

    # Key findings
    lines.append("---\n")
    lines.append("## Key Findings\n")

    # Does diversity help?
    best_per_scale = {}
    for s in scales:
        best = min(all_data[s]["sweep"], key=lambda r: r["mbr_cer_wer"])
        best_per_scale[s] = best["mbr_cer_wer"]

    sorted_by_wer = sorted(best_per_scale.items(), key=lambda x: x[1])
    best_s, best_w = sorted_by_wer[0]
    worst_s, worst_w = sorted_by_wer[-1]

    lines.append(f"### Does more diversity help MBR?\n")
    if best_s < 1.0 and best_w < best_per_scale[1.0] - 1e-5:
        lines.append(f"**Yes.** Lower nbest_scale (more diverse candidates) improves MBR: "
                     f"scale={best_s} achieves {best_w*100:.2f}% vs scale=1.0 at "
                     f"{best_per_scale[1.0]*100:.2f}%.\n")
    elif best_s > min(scales) and best_w < best_per_scale.get(min(scales), 1.0) - 1e-5:
        lines.append(f"**Moderate diversity is best.** scale={best_s} outperforms both "
                     f"higher and lower diversity settings.\n")
    else:
        lines.append(f"**No clear diversity advantage.** Best scale={best_s} "
                     f"({best_w*100:.2f}%) is not meaningfully better than others "
                     f"(spread: {(worst_w - best_w)*100:.2f} pp).\n")

    # Is any config meaningfully better than greedy?
    lines.append("### Does any (scale, tau) meaningfully beat greedy?\n")
    delta_best = ref_greedy - global_best_wer
    if delta_best > 0.05 / 100:  # > 0.05 pp
        lines.append(f"**Yes.** Global best (scale={global_best_scale}, tau={gb_tau_s}) "
                     f"reduces WER by {delta_best*100:.2f} pp, closing "
                     f"{gb_gap_closed:.1f}% of the oracle gap.\n")
    else:
        lines.append(f"**No.** The best improvement is only {delta_best*100:.2f} pp  --  "
                     f"within noise. Temperature-scaled MBR with CTC probabilities alone "
                     f"cannot meaningfully close the oracle gap regardless of candidate "
                     f"diversity.\n")

    # Does optimal tau depend on scale?
    lines.append("### Does optimal tau depend on scale?\n")
    opt_taus = {s: min(all_data[s]["sweep"], key=lambda r: r["mbr_cer_wer"])["tau"]
                for s in scales}
    unique_taus = set(opt_taus.values())
    if len(unique_taus) == 1:
        t = next(iter(unique_taus))
        t_s = "inf" if t == float("inf") else f"{t}"
        lines.append(f"No  --  tau*={t_s} is optimal across all scales.\n")
    else:
        lines.append("Yes  --  optimal tau varies:\n")
        for s in scales:
            t = opt_taus[s]
            t_s = "inf" if t == float("inf") else f"{t}"
            lines.append(f"- scale={s}: tau*={t_s}")
        lines.append("")

    # Oracle WER comparison
    lines.append("### How does oracle WER change with scale?\n")
    for s in scales:
        o = all_data[s]["baselines"]["oracle_wer"]
        lines.append(f"- scale={s:.2f}: oracle WER = {o*100:.2f}%")
    lines.append("")
    oracle_10 = all_data[1.0]["baselines"]["oracle_wer"]
    oracle_05 = all_data.get(0.5, {}).get("baselines", {}).get("oracle_wer", oracle_10)
    if oracle_05 < oracle_10 - 1e-5:
        lines.append(f"Lower scale gives better oracle ({oracle_05*100:.2f}% vs "
                     f"{oracle_10*100:.2f}%)  --  more diverse candidates include better "
                     f"hypotheses.\n")
    else:
        lines.append("Oracle WER is similar across scales  --  diversity doesn't expose "
                     "better candidates, just different ones.\n")

    # Conclusion
    lines.append("---\n")
    lines.append("## Conclusion\n")
    if delta_best > 0.1 / 100:
        lines.append(f"Diversity x temperature sweep found a meaningful improvement: "
                     f"scale={global_best_scale}, tau={gb_tau_s} gives WER={global_best_wer*100:.2f}% "
                     f"(-{delta_best*100:.2f} pp vs greedy, {gb_gap_closed:.1f}% gap closure). "
                     f"This suggests that candidate diversity combined with proper probability "
                     f"flattening can partially exploit the CTC rank correlation.\n")
    else:
        lines.append(f"The diversity x temperature sweep confirms the Level 1/1b-A finding: "
                     f"**decode-time scoring with CTC probabilities cannot meaningfully close "
                     f"the oracle gap**, regardless of candidate diversity or probability "
                     f"flattening. The best configuration (scale={global_best_scale}, "
                     f"tau={gb_tau_s}) achieves only {delta_best*100:.2f} pp improvement "
                     f"over greedy. External rescoring (language model, neural rescorer) "
                     f"is required to exploit the N-best diversity.\n")

    lines.append(f"\n**Runtime:** {elapsed:.1f}s ({elapsed/60:.1f} min)\n")

    lines.append("## Generated Files\n")
    lines.append("- `temperature_diversity_sweep.csv`")
    lines.append("- `plots/temperature_diversity_heatmap.png`")
    lines.append("- `plots/temperature_diversity_curves.png`")
    lines.append("- `partB_report.md`  --  this report")

    report_path = results_dir / "partB_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {report_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Level 1b Part B: Diversity x Temperature Sweep"
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = args.results_dir
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Level 1b Part B: Diversity x Temperature Sweep")
    print("=" * 60)

    t_start = time.time()

    # Discover available scale files
    scale_paths = {}
    for s in SCALES:
        if abs(s - 1.0) < 1e-6:
            p = results_dir / "nbest_dev_other_G16.jsonl"
        else:
            p = results_dir / f"nbest_dev_other_G16_scale{s:.2f}.jsonl"
        if p.exists():
            scale_paths[s] = p
        else:
            print(f"  WARNING: {p.name} not found  --  skipping scale={s}")

    if not scale_paths:
        raise SystemExit("ERROR: no N-best files found.")

    all_data = {}

    for scale in sorted(scale_paths.keys()):
        path = scale_paths[scale]
        print(f"\n-- scale={scale:.2f} --")
        records = load_nbest(path)
        print(f"  Loaded {len(records)} utterances")

        diversity = compute_diversity(records)
        print(f"  Diversity: unique={diversity['mean_unique']:.1f}, "
              f"WED={diversity['mean_pairwise_wed']*100:.1f}%")

        sweep, baselines = sweep_one_scale(records, TEMPERATURES)
        print(f"  Greedy WER: {baselines['greedy_wer']*100:.2f}%, "
              f"Oracle WER: {baselines['oracle_wer']*100:.2f}%")

        best = min(sweep, key=lambda r: r["mbr_cer_wer"])
        tau_s = "inf" if best["tau"] == float("inf") else f"{best['tau']:.0f}"
        print(f"  Best: tau={tau_s}, WER={best['mbr_cer_wer']*100:.2f}%, "
              f"gap closed={best['gap_closed']:+.1f}%")

        all_data[scale] = {
            "sweep": sweep,
            "baselines": baselines,
            "diversity": diversity,
        }

    print("\n" + "=" * 95)
    print("BEST CONFIGURATION PER SCALE")
    print("=" * 95)
    ref_greedy = all_data[1.0]["baselines"]["greedy_wer"] if 1.0 in all_data else list(all_data.values())[0]["baselines"]["greedy_wer"]
    print(f"{'Scale':>6s} | {'tau*':>5s} | {'MBR WER%':>9s} | {'Gap Closed':>10s} | "
          f"{'vs Greedy':>9s} | {'Unique':>7s} | {'WED%':>6s} | {'Oracle%':>8s}")
    print("-" * 95)
    for s in sorted(all_data.keys()):
        d = all_data[s]
        best = min(d["sweep"], key=lambda r: r["mbr_cer_wer"])
        tau_s = "inf" if best["tau"] == float("inf") else f"{best['tau']:.0f}"
        delta = best["mbr_cer_wer"] - ref_greedy
        print(f"{s:>6.2f} | {tau_s:>5s} | {best['mbr_cer_wer']*100:>8.2f}% | "
              f"{best['gap_closed']:>+9.1f}% | {delta*100:>+8.2f} pp | "
              f"{d['diversity']['mean_unique']:>7.1f} | "
              f"{d['diversity']['mean_pairwise_wed']*100:>5.1f}% | "
              f"{d['baselines']['oracle_wer']*100:>7.2f}%")
    print("=" * 95)

    csv_path = results_dir / "temperature_diversity_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scale", "tau", "mbr_cer_wer", "gap_closed",
                         "mean_entropy", "frac_differ",
                         "greedy_wer", "oracle_wer",
                         "mean_unique", "mean_pairwise_wed"])
        for s in sorted(all_data.keys()):
            d = all_data[s]
            for r in d["sweep"]:
                writer.writerow([
                    s,
                    r["tau"] if r["tau"] != float("inf") else "inf",
                    f"{r['mbr_cer_wer']:.6f}",
                    f"{r['gap_closed']:.2f}",
                    f"{r['mean_entropy']:.4f}",
                    f"{r['frac_differ']:.4f}",
                    f"{d['baselines']['greedy_wer']:.6f}",
                    f"{d['baselines']['oracle_wer']:.6f}",
                    f"{d['diversity']['mean_unique']:.1f}",
                    f"{d['diversity']['mean_pairwise_wed']:.4f}",
                ])
    print(f"\n  Saved: {csv_path}")

    # Plots
    plot_heatmap(all_data, plots_dir)
    plot_curves(all_data, ref_greedy, plots_dir)

    # Report
    elapsed = time.time() - t_start
    generate_report(all_data, elapsed, results_dir)

    print(f"\nTotal runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 60)
    print("Part B complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

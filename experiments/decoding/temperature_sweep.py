#!/usr/bin/env python3
"""Level 1b: Temperature-Scaled MBR & Diversity Sweep.

Part A (CPU): sweep tau on existing nbest_scale=1.0 data.
Part B (GPU): generate N-best at additional scales, then sweep tau.
Part C (CPU): per-utterance analysis at best (scale, tau).

Usage:
    # Part A only (CPU, ~5 min):
    python -m experiments.decoding.temperature_sweep \
        --results-dir results --part A

    # Part B  --  generate new N-best (GPU):
    python -m experiments.decoding.temperature_sweep \
        --results-dir results --part B \
        --model-dir /content/icefall-asr-librispeech-zipformer-small-cr-ctc \
        --data-dir /content/librispeech_data

    # Part C  --  per-utterance analysis (CPU, after A and/or B):
    python -m experiments.decoding.temperature_sweep \
        --results-dir results --part C

    # All parts:
    python -m experiments.decoding.temperature_sweep \
        --results-dir results --part ABC ...
"""

import argparse
import csv
import json
import time
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

TEMPERATURES = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 50.0, float("inf")]
NBEST_SCALES_EXTRA = [0.5, 0.75]



def load_nbest(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"  Loaded {len(records)} utterances from {path}")
    return records


def compute_wer_words(hyp: str, ref: str) -> tuple[int, int]:
    ref_w = ref.split()
    hyp_w = hyp.split()
    return editdistance.eval(hyp_w, ref_w), len(ref_w)


def compute_wer(hyp: str, ref: str) -> float:
    num, den = compute_wer_words(hyp, ref)
    return num / max(den, 1)


def compute_cer(hyp: str, ref: str) -> float:
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return editdistance.eval(list(hyp), list(ref)) / len(ref)


def char_distance(a, b):
    denom = max(len(a), len(b), 1)
    return editdistance.eval(list(a), list(b)) / denom


def word_distance(a, b):
    wa, wb = a.split(), b.split()
    denom = max(len(wa), len(wb), 1)
    return editdistance.eval(wa, wb) / denom


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


def mbr_select_tempered(texts, log_probs, distance_fn, tau):
    n = len(texts)
    if n == 1:
        return 0
    weights = tempered_softmax(log_probs, tau)
    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            scores[i] += weights[j] * distance_fn(texts[i], texts[j])
    return int(np.argmin(scores))



def run_temperature_sweep(records: list[dict], temperatures: list[float],
                          label: str = "scale=1.0"):
    greedy_wer_num, greedy_wer_den = 0, 0
    oracle_wer_num, oracle_wer_den = 0, 0

    for rec in records:
        ref = rec["ref_text"]
        cands = rec["candidates"]
        ref_w = ref.split()
        n_ref = len(ref_w)

        greedy_text = cands[0]["text"]
        g_num = editdistance.eval(greedy_text.split(), ref_w)
        greedy_wer_num += g_num
        greedy_wer_den += n_ref

        wers = [compute_wer(c["text"], ref) for c in cands]
        oracle_idx = int(np.argmin(wers))
        oracle_text = cands[oracle_idx]["text"]
        o_num = editdistance.eval(oracle_text.split(), ref_w)
        oracle_wer_num += o_num
        oracle_wer_den += n_ref

    greedy_wer = greedy_wer_num / max(greedy_wer_den, 1)
    oracle_wer = oracle_wer_num / max(oracle_wer_den, 1)
    gap = greedy_wer - oracle_wer

    print(f"\n  [{label}] Greedy WER: {greedy_wer*100:.2f}%, "
          f"Oracle WER: {oracle_wer*100:.2f}%, Gap: {gap*100:.2f} pp")

    results = []

    for tau in temperatures:
        tau_label = f"tau={tau}" if tau != float("inf") else "tau=inf"
        t0 = time.time()

        cer_wer_num, cer_wer_den = 0, 0
        cer_cer_num, cer_cer_den = 0, 0
        wer_wer_num, wer_wer_den = 0, 0
        entropies = []
        n_differ_cer = 0
        n_differ_wer = 0
        per_utt_cer = []

        for rec in records:
            ref = rec["ref_text"]
            cands = rec["candidates"]
            texts = [c["text"] for c in cands]
            log_probs = [c["ctc_log_prob"] for c in cands]
            ref_w = ref.split()
            n_ref = len(ref_w)

            weights = tempered_softmax(log_probs, tau)
            entropies.append(entropy(weights))

            idx_cer = mbr_select_tempered(texts, log_probs, char_distance, tau)
            idx_wer = mbr_select_tempered(texts, log_probs, word_distance, tau)

            hyp_cer = texts[idx_cer]
            cer_wer_num += editdistance.eval(hyp_cer.split(), ref_w)
            cer_wer_den += n_ref
            cer_cer_num += editdistance.eval(list(hyp_cer), list(ref))
            cer_cer_den += len(ref)

            hyp_wer = texts[idx_wer]
            wer_wer_num += editdistance.eval(hyp_wer.split(), ref_w)
            wer_wer_den += n_ref

            if idx_cer != 0:
                n_differ_cer += 1
            if idx_wer != 0:
                n_differ_wer += 1

            wer_val = compute_wer(hyp_cer, ref)
            greedy_wer_val = compute_wer(texts[0], ref)
            per_utt_cer.append({
                "utt_id": rec["utt_id"],
                "greedy_wer": greedy_wer_val,
                "mbr_cer_wer": wer_val,
                "delta_wer": wer_val - greedy_wer_val,
                "selected_idx": idx_cer,
            })

        elapsed = time.time() - t0
        n = len(records)

        mbr_cer_wer = cer_wer_num / max(cer_wer_den, 1)
        mbr_cer_cer = cer_cer_num / max(cer_cer_den, 1)
        mbr_wer_wer = wer_wer_num / max(wer_wer_den, 1)

        gap_closed_cer = (greedy_wer - mbr_cer_wer) / gap * 100 if gap > 1e-9 else 0.0
        gap_closed_wer = (greedy_wer - mbr_wer_wer) / gap * 100 if gap > 1e-9 else 0.0

        row = {
            "tau": tau,
            "tau_label": tau_label,
            "mbr_cer_wer": mbr_cer_wer,
            "mbr_cer_cer": mbr_cer_cer,
            "mbr_wer_wer": mbr_wer_wer,
            "gap_closed_cer": gap_closed_cer,
            "gap_closed_wer": gap_closed_wer,
            "mean_entropy": float(np.mean(entropies)),
            "frac_differ_cer": n_differ_cer / n,
            "frac_differ_wer": n_differ_wer / n,
            "elapsed": elapsed,
            "per_utt": per_utt_cer,
        }
        results.append(row)

        print(f"  {tau_label:>8s}: MBR-CER WER={mbr_cer_wer*100:.2f}%, "
              f"gap={gap_closed_cer:+.1f}%, H={row['mean_entropy']:.2f}, "
              f"differ={n_differ_cer/n*100:.1f}%, {elapsed:.1f}s")

    return results, greedy_wer, oracle_wer


def print_table(results, greedy_wer, oracle_wer):
    print("\n" + "=" * 100)
    print("TEMPERATURE-SCALED MBR  --  dev-other G=16")
    print("=" * 100)
    print(f"{'tau':>8s} | {'MBR-CER WER%':>13s} | {'Gap Closed%':>11s} | "
          f"{'MBR-WER WER%':>13s} | {'Gap Closed%':>11s} | "
          f"{'Entropy':>8s} | {'% Differ':>8s}")
    print("-" * 100)
    for r in results:
        tau_s = "inf" if r["tau"] == float("inf") else f"{r['tau']:.1f}"
        print(f"{tau_s:>8s} | {r['mbr_cer_wer']*100:>12.2f}% | {r['gap_closed_cer']:>+10.1f}% | "
              f"{r['mbr_wer_wer']*100:>12.2f}% | {r['gap_closed_wer']:>+10.1f}% | "
              f"{r['mean_entropy']:>8.3f} | {r['frac_differ_cer']*100:>7.1f}%")
    print("-" * 100)
    print(f"Greedy WER: {greedy_wer*100:.2f}%  |  Oracle WER: {oracle_wer*100:.2f}%  |  "
          f"Gap: {(greedy_wer-oracle_wer)*100:.2f} pp ({(greedy_wer-oracle_wer)/greedy_wer*100:.1f}% relative)")
    print("=" * 100)


def save_temperature_csv(results, greedy_wer, oracle_wer, path: Path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tau", "mbr_cer_wer", "mbr_cer_cer", "mbr_wer_wer",
            "gap_closed_cer", "gap_closed_wer",
            "mean_entropy", "frac_differ_cer", "frac_differ_wer",
            "greedy_wer", "oracle_wer",
        ])
        for r in results:
            writer.writerow([
                r["tau"] if r["tau"] != float("inf") else "inf",
                f"{r['mbr_cer_wer']:.6f}",
                f"{r['mbr_cer_cer']:.6f}",
                f"{r['mbr_wer_wer']:.6f}",
                f"{r['gap_closed_cer']:.2f}",
                f"{r['gap_closed_wer']:.2f}",
                f"{r['mean_entropy']:.4f}",
                f"{r['frac_differ_cer']:.4f}",
                f"{r['frac_differ_wer']:.4f}",
                f"{greedy_wer:.6f}",
                f"{oracle_wer:.6f}",
            ])
    print(f"  Saved: {path}")


def plot_temperature_sweep(results, greedy_wer, oracle_wer, path: Path):
    taus = []
    wers_cer = []
    wers_wer = []
    for r in results:
        t = r["tau"]
        taus.append(t if t != float("inf") else 100.0)
        wers_cer.append(r["mbr_cer_wer"] * 100)
        wers_wer.append(r["mbr_wer_wer"] * 100)

    fig, ax = plt.subplots()
    ax.semilogx(taus, wers_cer, "o-", color="#3498db", linewidth=2,
                markersize=6, label="MBR-CER", zorder=3)
    ax.semilogx(taus, wers_wer, "s--", color="#e67e22", linewidth=1.5,
                markersize=5, alpha=0.7, label="MBR-WER", zorder=2)
    ax.axhline(greedy_wer * 100, color="#e74c3c", linewidth=1.5,
               linestyle=":", label=f"Greedy = {greedy_wer*100:.2f}%", zorder=1)
    ax.axhline(oracle_wer * 100, color="#27ae60", linewidth=1.5,
               linestyle=":", label=f"Oracle = {oracle_wer*100:.2f}%", zorder=1)

    best_idx = int(np.argmin(wers_cer))
    ax.annotate(f"Best: {wers_cer[best_idx]:.2f}%",
                xy=(taus[best_idx], wers_cer[best_idx]),
                xytext=(taus[best_idx] * 2, wers_cer[best_idx] - 0.1),
                fontsize=9, ha="left",
                arrowprops=dict(arrowstyle="->", color="#3498db"))

    ax.set_xlabel("Temperature tau (log scale; inf shown as 100)")
    ax.set_ylabel("WER %")
    ax.set_title("Temperature-Scaled MBR: WER vs Temperature")
    ax.legend(fontsize=9)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")



def compute_diversity_metrics(records: list[dict]) -> dict:
    from itertools import combinations
    unique_counts = []
    pairwise_weds = []
    oracle_wer_num, oracle_wer_den = 0, 0

    for rec in records:
        ref = rec["ref_text"]
        cands = rec["candidates"]
        texts = [c["text"].strip().lower() for c in cands]
        unique = list(set(texts))
        unique_counts.append(len(unique))

        if len(unique) >= 2:
            dists = []
            for a, b in combinations(unique, 2):
                wa, wb = a.split(), b.split()
                denom = max(len(wa), len(wb), 1)
                dists.append(editdistance.eval(wa, wb) / denom)
            pairwise_weds.append(float(np.mean(dists)))
        else:
            pairwise_weds.append(0.0)

        ref_w = ref.split()
        wers = [compute_wer(c["text"], ref) for c in cands]
        oracle_idx = int(np.argmin(wers))
        oracle_wer_num += editdistance.eval(cands[oracle_idx]["text"].split(), ref_w)
        oracle_wer_den += len(ref_w)

    return {
        "mean_unique": float(np.mean(unique_counts)),
        "mean_pairwise_wed": float(np.mean(pairwise_weds)),
        "oracle_wer": oracle_wer_num / max(oracle_wer_den, 1),
    }


def run_part_b(results_dir: Path, plots_dir: Path):
    scale_files = {1.0: results_dir / "nbest_dev_other_G16.jsonl"}
    for s in NBEST_SCALES_EXTRA:
        p = results_dir / f"nbest_dev_other_G16_scale{s:.2f}.jsonl"
        if p.exists():
            scale_files[s] = p
        else:
            print(f"  WARNING: {p} not found  --  skipping scale={s}. "
                  f"Generate with generate_nbest.py --nbest-scale {s}")

    if len(scale_files) < 2:
        print("  Only scale=1.0 available. Part B requires additional scales.")
        print("  Run generate_nbest.py with --nbest-scale 0.5 and 0.75 first.")
        return None

    all_rows = []
    best_configs = []

    for scale in sorted(scale_files.keys()):
        path = scale_files[scale]
        print(f"\n  === nbest_scale={scale} ===")
        records = load_nbest(path)
        diversity = compute_diversity_metrics(records)
        print(f"  Diversity: unique={diversity['mean_unique']:.1f}, "
              f"WED={diversity['mean_pairwise_wed']*100:.1f}%, "
              f"oracle={diversity['oracle_wer']*100:.2f}%")

        sweep_results, greedy_wer, oracle_wer = run_temperature_sweep(
            records, TEMPERATURES, label=f"scale={scale}"
        )

        best_idx = min(range(len(sweep_results)),
                       key=lambda i: sweep_results[i]["mbr_cer_wer"])
        best = sweep_results[best_idx]
        best_configs.append({
            "scale": scale,
            "best_tau": best["tau"],
            "best_wer": best["mbr_cer_wer"],
            "gap_closed": best["gap_closed_cer"],
            "diversity_unique": diversity["mean_unique"],
            "diversity_wed": diversity["mean_pairwise_wed"],
            "oracle_wer": oracle_wer,
            "greedy_wer": greedy_wer,
        })

        for r in sweep_results:
            all_rows.append({
                "scale": scale,
                "tau": r["tau"],
                "mbr_cer_wer": r["mbr_cer_wer"],
                "mbr_wer_wer": r["mbr_wer_wer"],
                "gap_closed_cer": r["gap_closed_cer"],
                "mean_entropy": r["mean_entropy"],
                "frac_differ_cer": r["frac_differ_cer"],
            })

    csv_path = results_dir / "temperature_diversity_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scale", "tau", "mbr_cer_wer", "mbr_wer_wer",
                         "gap_closed_cer", "mean_entropy", "frac_differ_cer"])
        for row in all_rows:
            writer.writerow([
                row["scale"],
                row["tau"] if row["tau"] != float("inf") else "inf",
                f"{row['mbr_cer_wer']:.6f}",
                f"{row['mbr_wer_wer']:.6f}",
                f"{row['gap_closed_cer']:.2f}",
                f"{row['mean_entropy']:.4f}",
                f"{row['frac_differ_cer']:.4f}",
            ])
    print(f"\n  Saved: {csv_path}")

    # Heatmap
    plot_diversity_heatmap(all_rows, plots_dir)

    # Summary table
    print("\n  " + "=" * 90)
    print("  DIVERSITY x TEMPERATURE: BEST CONFIGURATION PER SCALE")
    print("  " + "=" * 90)
    print(f"  {'Scale':>6s} | {'tau*':>6s} | {'Best WER%':>10s} | {'Gap Closed%':>11s} | "
          f"{'Unique':>7s} | {'WED%':>6s} | {'Oracle WER%':>11s}")
    print("  " + "-" * 90)
    for c in best_configs:
        tau_s = "inf" if c["best_tau"] == float("inf") else f"{c['best_tau']:.1f}"
        print(f"  {c['scale']:>6.2f} | {tau_s:>6s} | {c['best_wer']*100:>9.2f}% | "
              f"{c['gap_closed']:>+10.1f}% | {c['diversity_unique']:>7.1f} | "
              f"{c['diversity_wed']*100:>5.1f}% | {c['oracle_wer']*100:>10.2f}%")
    print("  " + "=" * 90)

    return best_configs


def plot_diversity_heatmap(all_rows, plots_dir: Path):
    scales = sorted(set(r["scale"] for r in all_rows))
    taus_raw = sorted(set(r["tau"] for r in all_rows if r["tau"] != float("inf")))
    taus_all = taus_raw + [float("inf")]

    wer_grid = np.full((len(scales), len(taus_all)), np.nan)
    for r in all_rows:
        si = scales.index(r["scale"])
        ti = taus_all.index(r["tau"])
        wer_grid[si, ti] = r["mbr_cer_wer"] * 100

    fig, ax = plt.subplots(figsize=(10, 4))
    tau_labels = [f"{t:.0f}" if t != float("inf") else "inf" for t in taus_all]
    scale_labels = [f"{s:.2f}" for s in scales]

    im = ax.imshow(wer_grid, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(tau_labels)))
    ax.set_xticklabels(tau_labels)
    ax.set_yticks(range(len(scale_labels)))
    ax.set_yticklabels(scale_labels)
    ax.set_xlabel("Temperature tau")
    ax.set_ylabel("nbest_scale")
    ax.set_title("MBR-CER WER% across Scale x Temperature")

    for i in range(len(scales)):
        for j in range(len(taus_all)):
            if not np.isnan(wer_grid[i, j]):
                ax.text(j, i, f"{wer_grid[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if wer_grid[i, j] > np.nanmedian(wer_grid) else "white")

    fig.colorbar(im, ax=ax, label="WER %")
    fig.tight_layout()
    fig.savefig(plots_dir / "temperature_diversity_heatmap.png")
    plt.close(fig)
    print(f"  Saved: {plots_dir / 'temperature_diversity_heatmap.png'}")



def run_part_c(records: list[dict], results: list[dict],
               greedy_wer: float, oracle_wer: float,
               results_dir: Path, plots_dir: Path):
    best_idx = min(range(len(results)), key=lambda i: results[i]["mbr_cer_wer"])
    best = results[best_idx]
    best_tau = best["tau"]
    tau_s = "inf" if best_tau == float("inf") else f"{best_tau}"
    print(f"\n  Best tau = {tau_s} (MBR-CER WER = {best['mbr_cer_wer']*100:.2f}%)")

    per_utt = best["per_utt"]

    n_improved = sum(1 for u in per_utt if u["delta_wer"] < -1e-9)
    n_degraded = sum(1 for u in per_utt if u["delta_wer"] > 1e-9)
    n_same = sum(1 for u in per_utt if abs(u["delta_wer"]) < 1e-9)

    print(f"  Improved: {n_improved}, Degraded: {n_degraded}, Same: {n_same}")

    # Load per_utterance.jsonl for recoverable info
    per_utt_l2_path = results_dir / "per_utterance.jsonl"
    recoverable_ids = set()
    if per_utt_l2_path.exists():
        with open(per_utt_l2_path) as f:
            for line in f:
                rec = json.loads(line)
                if abs(rec["greedy_wer"] - rec["oracle_wer"]) > 1e-9:
                    recoverable_ids.add(rec["utt_id"])
        print(f"  Recoverable utterances (from Level 2): {len(recoverable_ids)}")

    rec_improved = 0
    rec_degraded = 0
    rec_same = 0
    for u in per_utt:
        if u["utt_id"] in recoverable_ids:
            if u["delta_wer"] < -1e-9:
                rec_improved += 1
            elif u["delta_wer"] > 1e-9:
                rec_degraded += 1
            else:
                rec_same += 1

    if recoverable_ids:
        n_rec = len(recoverable_ids)
        print(f"  On recoverable ({n_rec}): "
              f"improved={rec_improved} ({rec_improved/n_rec*100:.1f}%), "
              f"degraded={rec_degraded} ({rec_degraded/n_rec*100:.1f}%), "
              f"same={rec_same} ({rec_same/n_rec*100:.1f}%)")

    # Scatter: delta WER vs greedy WER
    greedy_wers = [u["greedy_wer"] for u in per_utt]
    delta_wers = [u["delta_wer"] for u in per_utt]

    fig, ax = plt.subplots(figsize=(9, 6))

    # Split into recoverable vs not for coloring
    rec_gw, rec_dw = [], []
    nonrec_gw, nonrec_dw = [], []
    for u in per_utt:
        if u["utt_id"] in recoverable_ids:
            rec_gw.append(u["greedy_wer"] * 100)
            rec_dw.append(u["delta_wer"] * 100)
        else:
            nonrec_gw.append(u["greedy_wer"] * 100)
            nonrec_dw.append(u["delta_wer"] * 100)

    ax.scatter(nonrec_gw, nonrec_dw, s=8, alpha=0.2, color="#7f8c8d",
               label=f"Greedy-optimal ({len(nonrec_gw)})", zorder=1)
    if rec_gw:
        ax.scatter(rec_gw, rec_dw, s=12, alpha=0.4, color="#3498db",
                   label=f"Recoverable ({len(rec_gw)})", zorder=2)
    ax.axhline(0, color="#e74c3c", linewidth=1, linestyle="--")
    ax.set_xlabel("Greedy WER %")
    ax.set_ylabel(f"delta WER % (MBR tau={tau_s} - Greedy)")
    ax.set_title(f"Per-Utterance WER Change at tau={tau_s}")
    ax.legend(fontsize=9)
    fig.savefig(plots_dir / "mbr_per_utterance_scatter.png")
    plt.close(fig)
    print(f"  Saved: {plots_dir / 'mbr_per_utterance_scatter.png'}")

    return {
        "best_tau": best_tau,
        "best_wer": best["mbr_cer_wer"],
        "gap_closed": best["gap_closed_cer"],
        "n_improved": n_improved,
        "n_degraded": n_degraded,
        "n_same": n_same,
        "recoverable_improved": rec_improved,
        "recoverable_degraded": rec_degraded,
        "recoverable_same": rec_same,
        "recoverable_total": len(recoverable_ids),
    }



def generate_report(
    sweep_results, greedy_wer, oracle_wer,
    part_c_stats, best_configs,
    elapsed_total, results_dir: Path
):
    gap = greedy_wer - oracle_wer
    n_utts = sum(1 for _ in open(results_dir / "nbest_dev_other_G16.jsonl"))

    lines = []
    lines.append("# Level 1b: Temperature-Scaled MBR & Diversity Sweep\n")
    lines.append(f"**Dataset:** LibriSpeech dev-other ({n_utts} utterances)")
    lines.append("**Model:** Zipformer-S CR-CTC, BPE-500")
    lines.append(f"**N-best:** G=16")
    lines.append(f"**Greedy WER:** {greedy_wer*100:.2f}%")
    lines.append(f"**Oracle WER:** {oracle_wer*100:.2f}%")
    lines.append(f"**Oracle gap:** {gap*100:.2f} pp ({gap/greedy_wer*100:.1f}% relative)\n")

    # Part A
    lines.append("## Part A: Temperature Sweep (nbest_scale=1.0)\n")
    lines.append(f"| tau | MBR-CER WER% | Gap Closed | MBR-WER WER% | Gap Closed | "
                 f"Entropy | % Differ |")
    lines.append(f"|--:|-------------:|-----------:|-------------:|-----------:|"
                 f"--------:|---------:|")
    for r in sweep_results:
        tau_s = "inf" if r["tau"] == float("inf") else f"{r['tau']}"
        lines.append(
            f"| {tau_s} | {r['mbr_cer_wer']*100:.2f}% | {r['gap_closed_cer']:+.1f}% | "
            f"{r['mbr_wer_wer']*100:.2f}% | {r['gap_closed_wer']:+.1f}% | "
            f"{r['mean_entropy']:.3f} | {r['frac_differ_cer']*100:.1f}% |"
        )

    best_idx = min(range(len(sweep_results)),
                   key=lambda i: sweep_results[i]["mbr_cer_wer"])
    best = sweep_results[best_idx]
    tau_best_s = "inf" if best["tau"] == float("inf") else f"{best['tau']}"
    lines.append(f"\n**Best tau = {tau_best_s}:** MBR-CER WER = {best['mbr_cer_wer']*100:.2f}%, "
                 f"gap closed = {best['gap_closed_cer']:+.1f}%\n")

    # tau=1 verification
    tau1 = next((r for r in sweep_results if r["tau"] == 1.0), None)
    if tau1:
        lines.append(f"**Verification:** tau=1.0 MBR-CER WER = {tau1['mbr_cer_wer']*100:.2f}% "
                     f"(should match Level 1 MBR-CER result).\n")

    lines.append("![Temperature sweep](plots/temperature_sweep.png)\n")

    # MBR-CER vs MBR-WER
    best_cer_wer = best["mbr_cer_wer"]
    best_wer_idx = min(range(len(sweep_results)),
                       key=lambda i: sweep_results[i]["mbr_wer_wer"])
    best_wer = sweep_results[best_wer_idx]
    if best_wer["mbr_wer_wer"] < best_cer_wer - 1e-5:
        lines.append(f"**MBR-WER vs MBR-CER:** MBR-WER achieves lower WER "
                     f"({best_wer['mbr_wer_wer']*100:.2f}% vs {best_cer_wer*100:.2f}%)  --  "
                     f"this is expected metric gaming (optimizing WER utility when evaluating WER). "
                     f"MBR-CER is the fairer metric.\n")
    else:
        lines.append(f"**MBR-WER vs MBR-CER:** comparable performance  --  "
                     f"MBR-CER {best_cer_wer*100:.2f}% vs MBR-WER {best_wer['mbr_wer_wer']*100:.2f}%.\n")

    # Part B
    if best_configs and len(best_configs) > 1:
        lines.append("## Part B: Diversity x Temperature\n")
        lines.append(f"| Scale | tau* | Best WER% | Gap Closed | Unique | WED% | Oracle WER% |")
        lines.append(f"|------:|---:|----------:|-----------:|-------:|-----:|------------:|")
        for c in best_configs:
            tau_s = "inf" if c["best_tau"] == float("inf") else f"{c['best_tau']}"
            lines.append(
                f"| {c['scale']:.2f} | {tau_s} | {c['best_wer']*100:.2f}% | "
                f"{c['gap_closed']:+.1f}% | {c['diversity_unique']:.1f} | "
                f"{c['diversity_wed']*100:.1f}% | {c['oracle_wer']*100:.2f}% |"
            )
        lines.append("")
        lines.append("![Heatmap](plots/temperature_diversity_heatmap.png)\n")
    else:
        lines.append("## Part B: Diversity x Temperature\n")
        lines.append("*Skipped  --  additional nbest_scale data not available. "
                     "Generate with `generate_nbest.py --nbest-scale 0.5` and "
                     "`--nbest-scale 0.75` to enable.*\n")

    # Part C
    if part_c_stats:
        c = part_c_stats
        tau_s = "inf" if c["best_tau"] == float("inf") else f"{c['best_tau']}"
        lines.append("## Part C: Per-Utterance Analysis\n")
        lines.append(f"At best tau = {tau_s}:\n")
        lines.append(f"| Category | Count | % |")
        lines.append(f"|----------|------:|--:|")
        total = c["n_improved"] + c["n_degraded"] + c["n_same"]
        lines.append(f"| Improved (MBR < greedy) | {c['n_improved']} | {c['n_improved']/total*100:.1f}% |")
        lines.append(f"| Degraded (MBR > greedy) | {c['n_degraded']} | {c['n_degraded']/total*100:.1f}% |")
        lines.append(f"| Same | {c['n_same']} | {c['n_same']/total*100:.1f}% |")
        lines.append("")

        if c["recoverable_total"] > 0:
            rt = c["recoverable_total"]
            lines.append(f"On the **{rt} recoverable** utterances (where oracle < greedy):\n")
            lines.append(f"| Category | Count | % of recoverable |")
            lines.append(f"|----------|------:|-----------------:|")
            lines.append(f"| Improved | {c['recoverable_improved']} | {c['recoverable_improved']/rt*100:.1f}% |")
            lines.append(f"| Degraded | {c['recoverable_degraded']} | {c['recoverable_degraded']/rt*100:.1f}% |")
            lines.append(f"| Same | {c['recoverable_same']} | {c['recoverable_same']/rt*100:.1f}% |")
            lines.append("")

        lines.append("![Per-utterance scatter](plots/mbr_per_utterance_scatter.png)\n")

    # Key Questions
    lines.append("---\n")
    lines.append("## Answers to Key Questions\n")

    lines.append("### Q1: Does any temperature give MBR an advantage over greedy?\n")
    any_better = any(r["mbr_cer_wer"] < greedy_wer - 1e-6 for r in sweep_results)
    if any_better:
        lines.append(f"**Yes.** The best MBR-CER WER ({best['mbr_cer_wer']*100:.2f}%) "
                     f"improves over greedy ({greedy_wer*100:.2f}%) by "
                     f"{(greedy_wer - best['mbr_cer_wer'])*100:.2f} pp.\n")
    else:
        lines.append(f"**No.** No temperature setting improves MBR-CER WER below greedy. "
                     f"The best ({best['mbr_cer_wer']*100:.2f}%) matches or exceeds "
                     f"greedy ({greedy_wer*100:.2f}%).\n")

    lines.append(f"### Q2: What's the optimal tau and how much gap does it close?\n")
    lines.append(f"tau* = {tau_best_s}, closing {best['gap_closed_cer']:+.1f}% of the oracle gap.\n")

    lines.append("### Q3: Does the optimal tau depend on nbest_scale?\n")
    if best_configs and len(best_configs) > 1:
        taus_by_scale = [(c["scale"], c["best_tau"]) for c in best_configs]
        same_tau = len(set(t for _, t in taus_by_scale)) == 1
        if same_tau:
            lines.append(f"No  --  tau* = {tau_best_s} is optimal across all scales tested.\n")
        else:
            lines.append("Yes  --  optimal tau varies by scale:\n")
            for s, t in taus_by_scale:
                t_s = "inf" if t == float("inf") else f"{t}"
                lines.append(f"- scale={s}: tau*={t_s}")
            lines.append("")
    else:
        lines.append("*Cannot answer  --  only scale=1.0 tested.*\n")

    lines.append("### Q4: Is the gain concentrated on hard utterances?\n")
    if part_c_stats:
        lines.append("See per-utterance scatter plot. ")
        if part_c_stats["n_improved"] > part_c_stats["n_degraded"]:
            lines.append("MBR improves more utterances than it degrades, "
                         "suggesting a net-positive effect.\n")
        elif part_c_stats["n_improved"] < part_c_stats["n_degraded"]:
            lines.append("MBR degrades more utterances than it improves  --  "
                         "the aggregate WER may still benefit if improvements are larger "
                         "than degradations, but the method is risky.\n")
        else:
            lines.append("Roughly balanced improvements and degradations.\n")
    else:
        lines.append("*Part C not run.*\n")

    lines.append("### Q5: Does MBR-CER at optimal tau beat MBR-WER?\n")
    if best_wer["mbr_wer_wer"] < best_cer_wer - 1e-5:
        lines.append(f"MBR-WER ({best_wer['mbr_wer_wer']*100:.2f}%) beats MBR-CER "
                     f"({best_cer_wer*100:.2f}%) in WER  --  but this is metric gaming. "
                     f"MBR-CER is the more principled choice.\n")
    else:
        lines.append(f"MBR-CER ({best_cer_wer*100:.2f}%) and MBR-WER "
                     f"({best_wer['mbr_wer_wer']*100:.2f}%) are comparable.\n")

    # Conclusion
    lines.append("---\n")
    lines.append("## Conclusion\n")
    if any_better:
        lines.append(f"Temperature scaling **does** help: tau={tau_best_s} reduces WER from "
                     f"{greedy_wer*100:.2f}% to {best['mbr_cer_wer']*100:.2f}%, "
                     f"closing {best['gap_closed_cer']:.1f}% of the oracle gap. "
                     f"This revises the Level 1 finding  --  scoring CAN help, but only with "
                     f"probability flattening to let MBR exploit the moderate CTC rank correlation "
                     f"(rho = -0.347).\n")
    else:
        lines.append(f"Temperature scaling does **not** help: even with optimal flattening, "
                     f"MBR cannot beat greedy. The Level 1 conclusion stands  --  CTC probabilities "
                     f"are insufficient for decode-time hypothesis selection, and the moderate "
                     f"rank correlation (rho = -0.347) is not strong enough to overcome the noise "
                     f"in MBR distance estimates.\n")

    lines.append(f"\n**Runtime:** {elapsed_total:.1f}s\n")

    lines.append("## Generated Files\n")
    lines.append("- `temperature_sweep.csv`")
    lines.append("- `plots/temperature_sweep.png`")
    if best_configs and len(best_configs) > 1:
        lines.append("- `temperature_diversity_sweep.csv`")
        lines.append("- `plots/temperature_diversity_heatmap.png`")
    lines.append("- `plots/mbr_per_utterance_scatter.png`")
    lines.append("- `level1b_report.md`  --  this report")

    report_path = results_dir / "level1b_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Saved: {report_path}")



def parse_args():
    parser = argparse.ArgumentParser(
        description="Level 1b: Temperature-Scaled MBR & Diversity Sweep"
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--part", type=str, default="AC",
                        help="Which parts to run: A, B, C, or combinations (e.g. ABC)")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--icefall-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = args.results_dir
    parts = args.part.upper()

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Level 1b: Temperature-Scaled MBR")
    print("=" * 60)

    t_start = time.time()

    nbest_path = results_dir / "nbest_dev_other_G16.jsonl"
    records = load_nbest(nbest_path)

    sweep_results, greedy_wer, oracle_wer = None, None, None
    part_c_stats = None
    best_configs = None

    if "A" in parts:
        print("\n-- Part A: Temperature Sweep (scale=1.0) --")
        sweep_results, greedy_wer, oracle_wer = run_temperature_sweep(
            records, TEMPERATURES, label="scale=1.0"
        )
        print_table(sweep_results, greedy_wer, oracle_wer)
        save_temperature_csv(sweep_results, greedy_wer, oracle_wer,
                             results_dir / "temperature_sweep.csv")
        plot_temperature_sweep(sweep_results, greedy_wer, oracle_wer,
                               plots_dir / "temperature_sweep.png")

    if "B" in parts:
        print("\n-- Part B: Diversity x Temperature --")
        best_configs = run_part_b(results_dir, plots_dir)

    if "C" in parts:
        print("\n-- Part C: Per-Utterance Analysis --")
        if sweep_results is None:
            print("  Running Part A first (needed for Part C)...")
            sweep_results, greedy_wer, oracle_wer = run_temperature_sweep(
                records, TEMPERATURES, label="scale=1.0"
            )
        part_c_stats = run_part_c(
            records, sweep_results, greedy_wer, oracle_wer,
            results_dir, plots_dir
        )

    elapsed_total = time.time() - t_start

    if sweep_results:
        generate_report(
            sweep_results, greedy_wer, oracle_wer,
            part_c_stats, best_configs,
            elapsed_total, results_dir,
        )

    print(f"\nTotal runtime: {elapsed_total:.1f}s")
    print("=" * 60)
    print("Level 1b complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

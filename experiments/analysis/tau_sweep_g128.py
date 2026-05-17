#!/usr/bin/env python3
"""E12: tau Fine-Sweep at G=128.

We know tau=10 is best from a coarse sweep {1, 5, 10, 50, inf}. This sweeps
tauin{5, 7, 8, 9, 10, 11, 12, 15, 20, 30, 50} to characterize sensitivity
around the optimum.

The CER matrix is tau-independent  --  computed once, then re-weighted per tau.

Usage:
    python experiments/analysis/tau_sweep_g128.py \
        --data-dir rbpo/results \
        --output-dir results/tau_sweep
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import editdistance
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.cer_matrix_cache import compute_or_load_cer_matrices, mbr_select
from experiments.significance_tests import paired_bootstrap_wer, corpus_wer

TAU_VALUES = [5, 7, 8, 9, 10, 11, 12, 15, 20, 30, 50]


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def run_tau_sweep(records, cer_matrices, taus):
    """For each tau, select hypotheses via MBR-CER+PLL and compute corpus WER."""
    ref_words = [r["ref_text"].split() for r in records]
    greedy_words = [r["candidates"][0]["text"].split() for r in records]
    greedy_wer = corpus_wer(ref_words, greedy_words)

    results = {}
    for tau in taus:
        hyps = []
        for i, rec in enumerate(records):
            cands = rec["candidates"]
            log_scores = np.array([c["roberta_pll"] for c in cands])
            idx = mbr_select(cer_matrices[i], log_scores, tau)
            hyps.append(cands[idx]["text"])

        hyp_words = [h.split() for h in hyps]
        wer = corpus_wer(ref_words, hyp_words)
        results[tau] = {"wer": wer, "hyps_words": hyp_words}
        print(f"  tau={tau:>4}: WER={wer*100:.4f}%  (delta={((wer-greedy_wer)*100):+.4f}pp)")

    return results, greedy_wer, ref_words, greedy_words


def run_bootstrap(results, ref_words, greedy_words, n_bootstrap, seed):
    """Paired bootstrap for each tau vs greedy."""
    bootstrap_results = {}
    print(f"\n  Running paired bootstrap (B={n_bootstrap})...")
    for tau, res in results.items():
        boot = paired_bootstrap_wer(
            ref_words, res["hyps_words"], greedy_words,
            n_bootstrap=n_bootstrap, seed=seed,
        )
        bootstrap_results[tau] = {
            "wer": boot["wer_a"],
            "delta": boot["delta"],
            "delta_pp": boot["delta"] * 100,
            "p_value": boot["p_value"],
            "ci_lower": boot["ci_lower"] * 100,
            "ci_upper": boot["ci_upper"] * 100,
        }
    return bootstrap_results


def main():
    parser = argparse.ArgumentParser(description="E12: tau Fine-Sweep at G=128")
    parser.add_argument("--data-dir", type=Path, default=Path("rbpo/results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/tau_sweep"))
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("E12: tau Fine-Sweep at G=128")
    print("=" * 70)
    t0 = time.time()

    data_path = args.data_dir / "g128" / "neural_lm_scores.jsonl"
    print(f"\nLoading: {data_path}")
    records = load_jsonl(data_path)
    n_utts = len(records)
    avg_cands = np.mean([r["num_candidates"] for r in records])
    print(f"  {n_utts} utterances, avg {avg_cands:.1f} candidates")

    print("\nCER Matrix Computation:")
    cer_matrices = compute_or_load_cer_matrices(
        records, data_path=data_path, cache_name="cer_matrix_g128"
    )

    print(f"\ntau Sweep: {TAU_VALUES}")
    results, greedy_wer, ref_words, greedy_words = run_tau_sweep(
        records, cer_matrices, TAU_VALUES
    )

    bootstrap = run_bootstrap(results, ref_words, greedy_words, args.n_bootstrap, args.seed)

    best_tau = min(TAU_VALUES, key=lambda t: results[t]["wer"])
    best_wer = results[best_tau]["wer"]
    print(f"\n  Optimal tau = {best_tau} (WER = {best_wer*100:.4f}%)")
    print(f"  Greedy WER = {greedy_wer*100:.4f}%")
    print(f"  Improvement = {(greedy_wer - best_wer)*100:.4f}pp")

    wers = [results[t]["wer"] * 100 for t in TAU_VALUES]
    wer_range = max(wers) - min(wers)
    flat_region = [t for t in TAU_VALUES if abs(results[t]["wer"] - best_wer) * 100 < 0.05]
    print(f"\n  Sensitivity: WER range across sweep = {wer_range:.4f}pp")
    print(f"  Flat region (within 0.05pp of best): tau in {flat_region}")

    print("\n--- Verification ---")
    assert n_utts == 2864, f"Expected 2864 utterances, got {n_utts}"
    print(f"  [PASS] Utterance count: {n_utts}")
    assert abs(greedy_wer * 100 - 6.02) < 0.1, f"Greedy WER {greedy_wer*100:.4f}% not ~6.02%"
    print(f"  [PASS] Greedy WER: {greedy_wer*100:.4f}% ~ 6.02%")
    tau10_wer = results[10]["wer"]
    assert abs(tau10_wer * 100 - 5.53) < 0.1, f"tau=10 WER {tau10_wer*100:.4f}% not ~5.53%"
    print(f"  [PASS] tau=10 WER: {tau10_wer*100:.4f}% ~ 5.53%")
    assert best_wer <= tau10_wer + 0.001, "Best tau should be <= tau=10"
    print(f"  [PASS] Best tau={best_tau} WER <= tau=10 WER")

    print("\n--- Writing outputs ---")

    # 1. JSON
    out_json = {
        "experiment": "E12_tau_sweep_g128",
        "data": str(data_path),
        "n_utterances": n_utts,
        "greedy_wer": greedy_wer,
        "tau_values": TAU_VALUES,
        "optimal_tau": best_tau,
        "optimal_wer": best_wer,
        "wer_range_pp": wer_range,
        "flat_region": flat_region,
        "n_bootstrap": args.n_bootstrap,
        "results": {
            str(tau): {
                "wer": results[tau]["wer"],
                "delta_pp": bootstrap[tau]["delta_pp"],
                "p_value": bootstrap[tau]["p_value"],
                "ci_lower": bootstrap[tau]["ci_lower"],
                "ci_upper": bootstrap[tau]["ci_upper"],
            }
            for tau in TAU_VALUES
        },
    }
    p = args.output_dir / "tau_sweep_results.json"
    with open(p, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"  Wrote {p}")

    # 2. CSV
    p = args.output_dir / "tau_sweep.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "tau", "wer", "delta_pp", "p_value", "ci_lower", "ci_upper"
        ])
        w.writeheader()
        for tau in TAU_VALUES:
            w.writerow({
                "tau": tau,
                "wer": f"{results[tau]['wer']*100:.4f}",
                "delta_pp": f"{bootstrap[tau]['delta_pp']:.4f}",
                "p_value": f"{bootstrap[tau]['p_value']:.4f}",
                "ci_lower": f"{bootstrap[tau]['ci_lower']:.4f}",
                "ci_upper": f"{bootstrap[tau]['ci_upper']:.4f}",
            })
    print(f"  Wrote {p}")

    # 3. Summary markdown
    p = args.output_dir / "tau_sweep_summary.md"
    lines = ["# E12: tau Fine-Sweep at G=128", ""]
    lines.append(f"**Optimal tau = {best_tau}** (WER = {best_wer*100:.4f}%)")
    lines.append(f"- Greedy baseline: {greedy_wer*100:.4f}%")
    lines.append(f"- Improvement: {(greedy_wer - best_wer)*100:.4f}pp")
    lines.append(f"- WER range across sweep: {wer_range:.4f}pp")
    lines.append(f"- Flat region (within 0.05pp): tau in {flat_region}")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| tau | WER (%) | delta (pp) | p-value | 95% CI (pp) |")
    lines.append("|--:|--------:|-------:|--------:|-------------|")
    for tau in TAU_VALUES:
        wer_pct = results[tau]["wer"] * 100
        b = bootstrap[tau]
        marker = " **<-best**" if tau == best_tau else ""
        p_str = "<0.0001" if b["p_value"] < 0.0001 else f"{b['p_value']:.4f}"
        lines.append(
            f"| {tau} | {wer_pct:.4f} | {b['delta_pp']:+.4f} | "
            f"{p_str} | [{b['ci_lower']:+.4f}, {b['ci_upper']:+.4f}] |{marker}"
        )
    lines.append("")
    lines.append("## Sensitivity Characterization")
    lines.append("")
    if wer_range < 0.2:
        lines.append("The WER surface is **flat** around the optimum  --  "
                     "tau is not a sensitive hyperparameter in this range.")
    elif wer_range < 0.5:
        lines.append("The WER surface shows **moderate sensitivity**  --  "
                     "tau matters but the region 7-15 is generally safe.")
    else:
        lines.append("The WER surface is **sharply peaked**  --  "
                     "tau selection significantly impacts performance.")
    lines.append("")
    with open(p, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {p}")

    # 4. Stage report
    elapsed = time.time() - t0
    p = args.output_dir / "report_E12.md"
    lines = ["# Report E12: tau Fine-Sweep at G=128", ""]
    lines.append(f"**Status:** Complete. {len(TAU_VALUES)} tau values, "
                 f"B={args.n_bootstrap} bootstrap. {elapsed:.0f}s on M2.")
    lines.append("")
    lines.append("## What Ran")
    lines.append("")
    lines.append(f"- Data: `g128/neural_lm_scores.jsonl` ({n_utts} utterances, G=128)")
    lines.append(f"- Sweep: tau in {{{', '.join(map(str, TAU_VALUES))}}}")
    lines.append(f"- Method: MBR-CER with RoBERTa PLL softmax weights")
    lines.append(f"- CER matrix: computed once, re-weighted per tau")
    lines.append(f"- Bootstrap: B={args.n_bootstrap}, paired vs greedy")
    lines.append("")
    lines.append("## Key Results")
    lines.append("")
    lines.append(f"- **Optimal tau = {best_tau}** (WER = {best_wer*100:.4f}%)")
    lines.append(f"- Greedy baseline: {greedy_wer*100:.4f}%")
    lines.append(f"- tau=10 WER: {tau10_wer*100:.4f}%")
    delta_best_vs_10 = (best_wer - tau10_wer) * 100
    if abs(delta_best_vs_10) < 0.01:
        lines.append(f"- tau={best_tau} vs tau=10: negligible difference ({delta_best_vs_10:+.4f}pp)")
    else:
        lines.append(f"- tau={best_tau} vs tau=10: {delta_best_vs_10:+.4f}pp")
    lines.append(f"- Flat region: tau in {flat_region}")
    lines.append(f"- Total WER range: {wer_range:.4f}pp")
    lines.append("")
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| tau | WER (%) | delta vs greedy (pp) | p-value |")
    lines.append("|--:|--------:|-----------------:|--------:|")
    for tau in TAU_VALUES:
        wer_pct = results[tau]["wer"] * 100
        b = bootstrap[tau]
        lines.append(f"| {tau} | {wer_pct:.4f} | {b['delta_pp']:+.4f} | {b['p_value']:.4f} |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if abs(delta_best_vs_10) < 0.05:
        lines.append("tau=10 is confirmed as optimal (or within noise of optimal). "
                     "The flat region suggests robustness  --  the result is not lucky.")
    else:
        lines.append(f"tau={best_tau} slightly outperforms tau=10 by {abs(delta_best_vs_10):.4f}pp. "
                     "This may or may not be significant given bootstrap uncertainty.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("| File | Purpose |")
    lines.append("|------|---------|")
    lines.append("| `tau_sweep_results.json` | Full results with bootstrap |")
    lines.append("| `tau_sweep.csv` | Tabular: tau, WER, p-value, CI |")
    lines.append("| `tau_sweep_summary.md` | Formatted summary |")
    lines.append("| `report_E12.md` | This stage report |")
    with open(p, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {p}")

    print(f"\nDone. Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

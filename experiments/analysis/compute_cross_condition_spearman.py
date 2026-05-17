#!/usr/bin/env python3
"""A1: Spearman rho(score, WER) across TL3 and MUSAN conditions.

For each condition, computes per-utterance Spearman rho for CTC log-prob
and RoBERTa PLL vs WER, reporting median/mean/std.

Usage:
    python experiments/analysis/compute_cross_condition_spearman.py \
        --output-dir results
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import editdistance
import numpy as np
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parent.parent

CONDITIONS = {
    "tl3_g128": REPO_ROOT / "results" / "tl3_rerun" / "nbest_g128_pll.jsonl",
    "tl3_g16_700": REPO_ROOT / "results" / "tl3_rerun" / "nbest_g16_pll.jsonl",
    "musan_0dB_g16": REPO_ROOT / "results" / "musan_rerun" / "nbest_0dB_g16_pll.jsonl",
    "musan_5dB_g16": REPO_ROOT / "results" / "musan_rerun" / "nbest_5dB_g16_pll.jsonl",
    "musan_10dB_g16": REPO_ROOT / "results" / "musan_rerun" / "nbest_10dB_g16_pll.jsonl",
    "musan_20dB_g16": REPO_ROOT / "results" / "musan_rerun" / "nbest_20dB_g16_pll.jsonl",
}


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def compute_spearman_for_condition(records):
    ctc_rhos = []
    pll_rhos = []
    n_total = len(records)
    n_excluded = 0

    for rec in records:
        nbest = rec["nbest"]
        if len(nbest) < 3:
            continue

        ref = rec["ref"]
        ref_w = ref.split()
        ref_len = len(ref_w)
        if ref_len == 0:
            continue

        wers = []
        ctc_scores = []
        pll_scores = []
        for h in nbest:
            hyp_w = h["hyp"].split()
            wer = editdistance.eval(hyp_w, ref_w) / ref_len
            wers.append(wer)
            ctc_scores.append(h["score"])
            pll_scores.append(h["pll_score"])

        if len(set(wers)) < 2:
            n_excluded += 1
            continue

        if len(set(ctc_scores)) >= 2:
            rho, _ = stats.spearmanr(ctc_scores, wers)
            if not np.isnan(rho):
                ctc_rhos.append(rho)

        if len(set(pll_scores)) >= 2:
            rho, _ = stats.spearmanr(pll_scores, wers)
            if not np.isnan(rho):
                pll_rhos.append(rho)

    return {
        "n": n_total,
        "n_valid": len(ctc_rhos),
        "n_excluded_zero_var": n_excluded,
        "ctc": {
            "median": float(np.median(ctc_rhos)),
            "mean": float(np.mean(ctc_rhos)),
            "std": float(np.std(ctc_rhos)),
        },
        "pll": {
            "median": float(np.median(pll_rhos)),
            "mean": float(np.mean(pll_rhos)),
            "std": float(np.std(pll_rhos)),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="A1: Cross-condition Spearman rho")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("A1: Spearman rho(score, WER) across conditions")
    print("=" * 70)

    results = {}
    for name, path in CONDITIONS.items():
        print(f"\n--- {name} ---")
        if not path.exists():
            print(f"  SKIP: {path} not found")
            continue
        records = load_jsonl(path)
        print(f"  Loaded {len(records)} utterances")
        r = compute_spearman_for_condition(records)
        results[name] = r
        print(f"  Valid: {r['n_valid']}/{r['n']} (excluded {r['n_excluded_zero_var']} zero-var)")
        print(f"  CTC:  median={r['ctc']['median']:+.4f}  mean={r['ctc']['mean']:+.4f}  std={r['ctc']['std']:.4f}")
        print(f"  PLL:  median={r['pll']['median']:+.4f}  mean={r['pll']['mean']:+.4f}  std={r['pll']['std']:.4f}")

    out_path = args.output_dir / "spearman_cross_condition.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")

    print("\n" + "=" * 70)
    print(f"{'Condition':<20s}  {'CTC med':>8s}  {'CTC mean':>9s}  {'PLL med':>8s}  {'PLL mean':>9s}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<20s}  {r['ctc']['median']:+8.4f}  {r['ctc']['mean']:+9.4f}  "
              f"{r['pll']['median']:+8.4f}  {r['pll']['mean']:+9.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()

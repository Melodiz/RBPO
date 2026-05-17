#!/usr/bin/env python3
"""A3: TL3 G-scaling subset verification + paired bootstrap.

Step 1: Report n_utterances for each G value.
Step 2: For each G, recompute MBR-CER+PLL from scored JSONLs, then
        run paired bootstrap vs greedy (B=10000, seed=42).

Usage:
    python experiments/analysis/compute_tl3_gaps.py \
        --output-dir results/tl3_rerun
"""

import argparse
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

from experiments.significance_tests import paired_bootstrap_wer

G_VALUES = [4, 8, 16, 32, 64, 128]
TL3_DIR = REPO_ROOT / "results" / "tl3_rerun"


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def compute_cer_matrix(texts):
    n = len(texts)
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = editdistance.eval(list(texts[i]), list(texts[j]))
            denom = max(len(texts[i]), len(texts[j]), 1)
            mat[i, j] = d / denom
            mat[j, i] = mat[i, j]
    return mat


def mbr_select(cer_matrix, log_scores, tau):
    n = len(log_scores)
    if math.isinf(tau):
        weights = np.ones(n) / n
    else:
        scaled = log_scores / tau
        scaled -= np.max(scaled)
        weights = np.exp(scaled)
        weights /= weights.sum()
    risk = cer_matrix @ weights
    return int(np.argmin(risk))


def corpus_wer_from_words(ref_words_list, hyp_words_list):
    total_errors = 0
    total_ref = 0
    for ref_w, hyp_w in zip(ref_words_list, hyp_words_list):
        total_errors += editdistance.eval(hyp_w, ref_w)
        total_ref += len(ref_w)
    return total_errors / max(total_ref, 1)


def process_g_value(g, tau=10.0):
    path = TL3_DIR / f"nbest_g{g}_pll.jsonl"
    if not path.exists():
        return None

    records = load_jsonl(path)
    n = len(records)

    ref_words = []
    greedy_words = []
    mbr_words = []

    for rec in records:
        ref = rec["ref"]
        ref_w = ref.split()
        ref_words.append(ref_w)

        nbest = rec["nbest"]
        greedy_words.append(nbest[0]["hyp"].split())

        texts = [h["hyp"] for h in nbest]
        log_scores = np.array([h["pll_score"] for h in nbest])

        cer_mat = compute_cer_matrix(texts)
        idx = mbr_select(cer_mat, log_scores, tau)
        mbr_words.append(texts[idx].split())

    greedy_wer = corpus_wer_from_words(ref_words, greedy_words)
    mbr_wer = corpus_wer_from_words(ref_words, mbr_words)

    return {
        "n": n,
        "greedy_wer": greedy_wer,
        "mbr_wer": mbr_wer,
        "ref_words": ref_words,
        "greedy_words": greedy_words,
        "mbr_words": mbr_words,
    }


def main():
    parser = argparse.ArgumentParser(description="A3: TL3 G-scaling bootstrap")
    parser.add_argument("--output-dir", type=Path, default=TL3_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("A3: TL3 G-Scaling Verification + Bootstrap")
    print("=" * 70)
    t0 = time.time()

    print("\n--- Step 1: Subset check ---")
    print(f"{'G':>5s}  {'n_utts':>6s}  {'File':>30s}")
    for g in G_VALUES:
        path = TL3_DIR / f"nbest_g{g}_pll.jsonl"
        if path.exists():
            records = load_jsonl(path)
            print(f"{g:>5d}  {len(records):>6d}  {path.name}")
        else:
            print(f"{g:>5d}  {'N/A':>6s}  NOT FOUND")

    print("\n--- Step 2: MBR recompute + bootstrap ---")
    results = {}
    for g in G_VALUES:
        print(f"\nG={g}:")
        data = process_g_value(g)
        if data is None:
            print("  SKIP: file not found")
            continue

        print(f"  n={data['n']}  greedy={data['greedy_wer']*100:.4f}%  "
              f"mbr={data['mbr_wer']*100:.4f}%  "
              f"delta={(data['mbr_wer']-data['greedy_wer'])*100:+.4f}pp")

        boot = paired_bootstrap_wer(
            data["ref_words"], data["mbr_words"], data["greedy_words"],
            n_bootstrap=args.n_bootstrap, seed=args.seed,
        )

        results[str(g)] = {
            "G": g,
            "n": data["n"],
            "greedy_wer": float(data["greedy_wer"]),
            "mbr_wer": float(data["mbr_wer"]),
            "delta_pp": boot["delta"] * 100,
            "p_value": boot["p_value"],
            "ci_lower_pp": boot["ci_lower"] * 100,
            "ci_upper_pp": boot["ci_upper"] * 100,
        }
        p_str = f"{boot['p_value']:.4f}" if boot["p_value"] >= 0.0001 else "<0.0001"
        print(f"  Bootstrap: p={p_str}, CI=[{boot['ci_lower']*100:+.4f}, {boot['ci_upper']*100:+.4f}]pp")

    if "128" in results:
        r = results["128"]
        assert abs(r["greedy_wer"] * 100 - 11.30) < 0.1, f"TL3 G=128 greedy WER drift: {r['greedy_wer']*100}"
        print("\n  [PASS] TL3 G=128 greedy WER = 11.30%")

    out_path = args.output_dir / "tl3_g_scaling_bootstrap.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"{'G':>5s}  {'n':>5s}  {'Greedy':>8s}  {'MBR':>8s}  {'delta(pp)':>8s}  {'p-value':>8s}")
    print("-" * 50)
    for g in G_VALUES:
        k = str(g)
        if k in results:
            r = results[k]
            p_str = f"{r['p_value']:.4f}" if r["p_value"] >= 0.0001 else "<0.0001"
            print(f"{g:>5d}  {r['n']:>5d}  {r['greedy_wer']*100:>7.2f}%  "
                  f"{r['mbr_wer']*100:>7.2f}%  {r['delta_pp']:>+7.2f}  {p_str:>8s}")
    print(f"{'='*70}")
    print(f"Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

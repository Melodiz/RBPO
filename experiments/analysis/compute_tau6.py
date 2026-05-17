#!/usr/bin/env python3
"""A2: Compute tau=6 MBR-CER+PLL on dev-other G=128.

Fills the gap between tau=5 (near-greedy) and tau=7 in the existing sweep.
Reuses the shared CER matrix cache and significance test infrastructure.

Usage:
    python experiments/analysis/compute_tau6.py \
        --data-dir results \
        --output-dir results/tau_sweep
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.cer_matrix_cache import compute_or_load_cer_matrices, mbr_select
from experiments.significance_tests import paired_bootstrap_wer, corpus_wer


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description="A2: tau=6 MBR-CER+PLL on G=128")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "tau_sweep")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("A2: tau=6 MBR-CER+PLL on dev-other G=128")
    print("=" * 70)
    t0 = time.time()

    data_path = args.data_dir / "g128" / "neural_lm_scores.jsonl"
    print(f"\nLoading: {data_path}")
    records = load_jsonl(data_path)
    n_utts = len(records)
    print(f"  {n_utts} utterances")

    assert n_utts == 2864, f"Expected 2864, got {n_utts}"

    ref_words = [r["ref_text"].split() for r in records]
    greedy_words = [r["candidates"][0]["text"].split() for r in records]
    greedy_wer = corpus_wer(ref_words, greedy_words)
    print(f"  Greedy WER: {greedy_wer*100:.4f}%")
    assert abs(greedy_wer * 100 - 6.02) < 0.1, f"Greedy WER drift: {greedy_wer*100:.4f}%"

    print("\nComputing CER matrices...")
    cer_matrices = compute_or_load_cer_matrices(
        records, data_path=data_path, cache_name="cer_matrix_g128"
    )

    tau = 6
    print(f"\nRunning MBR-CER+PLL with tau={tau}...")
    hyps = []
    for i, rec in enumerate(records):
        cands = rec["candidates"]
        log_scores = np.array([c["roberta_pll"] for c in cands])
        idx = mbr_select(cer_matrices[i], log_scores, tau)
        hyps.append(cands[idx]["text"])

    hyp_words = [h.split() for h in hyps]
    wer = corpus_wer(ref_words, hyp_words)
    print(f"  tau={tau}: WER={wer*100:.4f}%  (delta={(wer-greedy_wer)*100:+.4f}pp)")

    print(f"\nPaired bootstrap (B={args.n_bootstrap})...")
    boot = paired_bootstrap_wer(
        ref_words, hyp_words, greedy_words,
        n_bootstrap=args.n_bootstrap, seed=args.seed,
    )

    result = {
        "tau": tau,
        "wer": float(wer),
        "wer_pct": f"{wer*100:.4f}",
        "greedy_wer": float(greedy_wer),
        "delta_pp": boot["delta"] * 100,
        "p_value": boot["p_value"],
        "ci_lower_pp": boot["ci_lower"] * 100,
        "ci_upper_pp": boot["ci_upper"] * 100,
        "n_utterances": n_utts,
        "n_bootstrap": args.n_bootstrap,
    }

    p = args.output_dir / "tau6_result.json"
    with open(p, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Wrote {p}")

    csv_path = args.output_dir / "tau_sweep.csv"
    existing_rows = []
    has_tau6 = False
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if int(row["tau"]) == 6:
                has_tau6 = True
                row = {
                    "tau": "6",
                    "wer": f"{wer*100:.4f}",
                    "delta_pp": f"{boot['delta']*100:.4f}",
                    "p_value": f"{boot['p_value']:.4f}",
                    "ci_lower": f"{boot['ci_lower']*100:.4f}",
                    "ci_upper": f"{boot['ci_upper']*100:.4f}",
                }
            existing_rows.append(row)

    if not has_tau6:
        new_row = {
            "tau": "6",
            "wer": f"{wer*100:.4f}",
            "delta_pp": f"{boot['delta']*100:.4f}",
            "p_value": f"{boot['p_value']:.4f}",
            "ci_lower": f"{boot['ci_lower']*100:.4f}",
            "ci_upper": f"{boot['ci_upper']*100:.4f}",
        }
        existing_rows.append(new_row)

    existing_rows.sort(key=lambda r: int(r["tau"]))
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(existing_rows)
    print(f"  Updated {csv_path}")

    elapsed = time.time() - t0
    print(f"\n--- Result ---")
    print(f"  tau=6: WER={wer*100:.4f}%, delta={boot['delta']*100:+.4f}pp, "
          f"p={boot['p_value']:.4f}, CI=[{boot['ci_lower']*100:+.4f}, {boot['ci_upper']*100:+.4f}]pp")
    print(f"  Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""A4: TL3 S/I/D error-type analysis for G=128 MBR.

Decomposes WER changes from greedy->MBR into substitution/insertion/deletion
deltas, matching the format of E10/E13 for dev-other/test-other.

Usage:
    python experiments/analysis/compute_tl3_error_types.py \
        --output-dir results/tl3_rerun
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import editdistance
import numpy as np

try:
    import jiwer
except ImportError:
    jiwer = None

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


def compute_sid(ref_text, hyp_text):
    """Compute word-level S/I/D using jiwer if available, else manual alignment."""
    if jiwer is not None:
        out = jiwer.process_words(ref_text, hyp_text)
        return out.substitutions, out.insertions, out.deletions
    ref_w = ref_text.split()
    hyp_w = hyp_text.split()
    total = editdistance.eval(ref_w, hyp_w)
    return total, 0, 0


def main():
    parser = argparse.ArgumentParser(description="A4: TL3 S/I/D error-type analysis")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "results" / "tl3_rerun")
    parser.add_argument("--tau", type=float, default=10.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("A4: TL3 Error-Type Analysis (S/I/D)")
    print("=" * 70)

    data_path = REPO_ROOT / "results" / "tl3_rerun" / "nbest_g128_pll.jsonl"
    print(f"\nLoading: {data_path}")
    records = load_jsonl(data_path)
    n = len(records)
    print(f"  {n} utterances")
    assert n == 1155, f"Expected 1155, got {n}"

    if jiwer is None:
        print("  WARNING: jiwer not installed, S/I/D breakdown unavailable")
        print("  Install with: pip install jiwer")
        return

    print(f"\nComputing MBR-CER+PLL tau={args.tau} selections...")
    improve_s, improve_i, improve_d = 0, 0, 0
    regress_s, regress_i, regress_d = 0, 0, 0
    tie_s, tie_i, tie_d = 0, 0, 0
    n_improve, n_regress, n_tie = 0, 0, 0

    for idx, rec in enumerate(records):
        ref = rec["ref"]
        nbest = rec["nbest"]

        greedy_text = nbest[0]["hyp"]
        texts = [h["hyp"] for h in nbest]
        log_scores = np.array([h["pll_score"] for h in nbest])
        cer_mat = compute_cer_matrix(texts)
        mbr_idx = mbr_select(cer_mat, log_scores, args.tau)
        mbr_text = texts[mbr_idx]

        ref_w = ref.split()
        ref_len = len(ref_w)
        if ref_len == 0:
            continue

        greedy_errs = editdistance.eval(greedy_text.split(), ref_w)
        mbr_errs = editdistance.eval(mbr_text.split(), ref_w)

        gs, gi, gd = compute_sid(ref, greedy_text)
        ms, mi, md = compute_sid(ref, mbr_text)

        ds = ms - gs
        di = mi - gi
        dd = md - gd

        if mbr_errs < greedy_errs:
            n_improve += 1
            improve_s += ds
            improve_i += di
            improve_d += dd
        elif mbr_errs > greedy_errs:
            n_regress += 1
            regress_s += ds
            regress_i += di
            regress_d += dd
        else:
            n_tie += 1
            tie_s += ds
            tie_i += di
            tie_d += dd

        if (idx + 1) % 200 == 0:
            print(f"  {idx+1}/{n} utterances processed")

    print(f"\n  Improve: {n_improve}  Regress: {n_regress}  Tie: {n_tie}")

    rows = []
    for outcome, ns, ni, nd, count in [
        ("improve", improve_s, improve_i, improve_d, n_improve),
        ("regress", regress_s, regress_i, regress_d, n_regress),
        ("tie", tie_s, tie_i, tie_d, n_tie),
    ]:
        total = ns + ni + nd
        if outcome == "improve" and total != 0:
            pct_s = abs(ns) / abs(total) * 100
            pct_i = abs(ni) / abs(total) * 100
            pct_d = abs(nd) / abs(total) * 100
        else:
            pct_s = pct_i = pct_d = 0.0

        rows.append({
            "outcome": outcome,
            "n_utts": count,
            "total_sub_delta": ns,
            "total_ins_delta": ni,
            "total_del_delta": nd,
            "total_delta": total,
            "pct_subs": round(pct_s, 1),
            "pct_ins": round(pct_i, 1),
            "pct_del": round(pct_d, 1),
        })

    out_path = args.output_dir / "error_type_tl3_summary.csv"
    fields = ["outcome", "n_utts", "total_sub_delta", "total_ins_delta",
              "total_del_delta", "total_delta", "pct_subs", "pct_ins", "pct_del"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out_path}")

    print(f"\n{'='*70}")
    print(f"{'outcome':<10s}  {'n':>5s}  {'S':>5s}  {'I':>5s}  {'D':>5s}  {'total':>6s}  {'S%':>5s}  {'I%':>5s}  {'D%':>5s}")
    print("-" * 70)
    for r in rows:
        print(f"{r['outcome']:<10s}  {r['n_utts']:>5d}  {r['total_sub_delta']:>5d}  "
              f"{r['total_ins_delta']:>5d}  {r['total_del_delta']:>5d}  {r['total_delta']:>6d}  "
              f"{r['pct_subs']:>5.1f}  {r['pct_ins']:>5.1f}  {r['pct_del']:>5.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()

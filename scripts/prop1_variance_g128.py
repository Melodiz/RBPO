#!/usr/bin/env python3
"""Proposition 4.1 variance ratio measurement at G=128.

Measures Var_corpus(CER_MBR) vs Var_corpus(CER_greedy) across
temperature settings to verify that MBR selection reduces per-utterance
CER variance relative to greedy.

Usage:
    python scripts/prop1_variance_g128.py \
        --nbest-jsonl results/g_scaling/nbest_dev_other_G128.jsonl \
        --output-dir results/S3_prop1_g128/ \
        --tau 1 50 inf
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import editdistance
import numpy as np


def cer(hyp, ref):
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return editdistance.eval(hyp, ref) / len(ref)


def load_nbest(path: Path):
    records = []
    with open(path) as f:
        for line in f:
            raw = json.loads(line)
            ref = raw.get("ref", raw.get("ref_text", ""))
            candidates = raw.get("nbest", raw.get("candidates", []))
            hyps = []
            scores = []
            for c in candidates:
                hyps.append(c.get("hyp", c.get("text", "")))
                scores.append(c.get("score", c.get("ctc_log_prob", 0.0)))
            records.append({
                "utt_id": raw["utt_id"],
                "ref": ref,
                "hyps": hyps,
                "scores": np.array(scores, dtype=np.float64),
            })
    return records


def softmax_with_tau(log_probs: np.ndarray, tau: float) -> np.ndarray:
    if math.isinf(tau):
        return np.ones_like(log_probs) / len(log_probs)
    scaled = log_probs / tau
    scaled -= scaled.max()
    w = np.exp(scaled)
    return w / w.sum()


def cer_matrix_for_utt(hyps: list) -> np.ndarray:
    n = len(hyps)
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = editdistance.eval(hyps[i], hyps[j])
            li = max(len(hyps[i]), 1)
            lj = max(len(hyps[j]), 1)
            mat[i, j] = d / lj
            mat[j, i] = d / li
    return mat


def mbr_select_from_matrix(cer_mat: np.ndarray, weights: np.ndarray) -> int:
    expected_cer = cer_mat @ weights
    return int(np.argmin(expected_cer))


def main():
    parser = argparse.ArgumentParser(
        description="Prop 4.1 variance ratio at G=128")
    parser.add_argument("--nbest-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tau", nargs="+", default=["1", "50", "inf"])
    args = parser.parse_args()

    assert args.nbest_jsonl.exists(), f"N-best file not found: {args.nbest_jsonl}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    taus = []
    tau_labels = []
    for t in args.tau:
        if t.lower() == "inf":
            taus.append(float("inf"))
            tau_labels.append("inf")
        else:
            taus.append(float(t))
            tau_labels.append(str(int(float(t))) if float(t) == int(float(t)) else t)

    print(f"Step 1/3: Loading N-best from {args.nbest_jsonl}...", flush=True)
    records = load_nbest(args.nbest_jsonl)
    n_utts = len(records)
    avg_cands = np.mean([len(r["hyps"]) for r in records])
    print(f"  Loaded {n_utts} utterances, avg {avg_cands:.1f} candidates", flush=True)

    assert abs(n_utts - 2864) < 10, f"Expected ~2864 utterances, got {n_utts}"

    print(f"Step 2/3: Computing CER matrices and MBR selections...", flush=True)

    cer_greedy_arr = np.zeros(n_utts, dtype=np.float64)
    cer_mbr_arrs = {label: np.zeros(n_utts, dtype=np.float64) for label in tau_labels}
    mbr_idx_arrs = {label: np.zeros(n_utts, dtype=np.int32) for label in tau_labels}

    t0 = time.time()
    for i, rec in enumerate(records):
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_utts - i - 1) / rate
            print(f"  {i+1}/{n_utts} utterances ({rate:.1f}/s, ETA {eta:.0f}s)...",
                  flush=True)

        ref = rec["ref"]
        hyps = rec["hyps"]
        scores = rec["scores"]

        cer_greedy_arr[i] = cer(hyps[0], ref)

        cer_mat = cer_matrix_for_utt(hyps)

        for tau, label in zip(taus, tau_labels):
            weights = softmax_with_tau(scores, tau)
            mbr_idx = mbr_select_from_matrix(cer_mat, weights)
            mbr_idx_arrs[label][i] = mbr_idx
            cer_mbr_arrs[label][i] = cer(hyps[mbr_idx], ref)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s", flush=True)

    print("Step 3/3: Computing statistics and saving...", flush=True)

    var_greedy = float(np.var(cer_greedy_arr, ddof=1))
    mean_greedy = float(np.mean(cer_greedy_arr))

    all_results = []
    all_per_utt = []

    for tau, label in zip(taus, tau_labels):
        cer_mbr = cer_mbr_arrs[label]
        mbr_idxs = mbr_idx_arrs[label]
        mbr_diff = mbr_idxs != 0

        var_mbr = float(np.var(cer_mbr, ddof=1))
        ratio = var_greedy / var_mbr if var_mbr > 0 else float('inf')
        mean_mbr = float(np.mean(cer_mbr))
        abs_diff = np.abs(cer_greedy_arr - cer_mbr)

        result = {
            "tau": label,
            "n_utterances": n_utts,
            "var_greedy": var_greedy,
            "var_mbr": var_mbr,
            "var_ratio_greedy_over_mbr": ratio,
            "mean_cer_greedy": mean_greedy,
            "mean_cer_mbr": mean_mbr,
            "mean_cer_delta": mean_greedy - mean_mbr,
            "n_mbr_different": int(mbr_diff.sum()),
            "frac_mbr_different": float(mbr_diff.mean()),
            "abs_diff_mean": float(abs_diff.mean()),
            "abs_diff_median": float(np.median(abs_diff)),
            "abs_diff_max": float(abs_diff.max()),
        }
        all_results.append(result)

        for j in range(n_utts):
            all_per_utt.append({
                "utt_id": records[j]["utt_id"],
                "tau": label,
                "cer_greedy": float(cer_greedy_arr[j]),
                "cer_mbr": float(cer_mbr[j]),
                "mbr_different": bool(mbr_diff[j]),
            })

    summary = {
        "experiment": "S3_prop1_g128",
        "description": "Proposition 4.1 variance ratio: Var(CER_greedy) / Var(CER_MBR)",
        "split": "dev-other",
        "G": 128,
        "n_utterances": n_utts,
        "avg_candidates": float(avg_cands),
        "results": all_results,
    }

    results_path = args.output_dir / "prop1_g128_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {results_path}", flush=True)

    csv_path = args.output_dir / "prop1_g128_per_utt.csv"
    with open(csv_path, "w") as f:
        f.write("utt_id,tau,cer_greedy,cer_mbr,mbr_different\n")
        for row in all_per_utt:
            f.write(f"{row['utt_id']},{row['tau']},"
                    f"{row['cer_greedy']:.6f},{row['cer_mbr']:.6f},"
                    f"{row['mbr_different']}\n")
    print(f"  Saved {csv_path}", flush=True)

    print("\n=== Summary ===", flush=True)
    print(f"{'tau':>6} | {'Var(greedy)':>12} | {'Var(MBR)':>12} | "
          f"{'Ratio':>8} | {'Mean CER_g':>10} | {'Mean CER_m':>10} | "
          f"{'N differ':>8} | {'Frac':>6}", flush=True)
    print("-" * 95, flush=True)
    for r in all_results:
        print(f"{r['tau']:>6} | {r['var_greedy']:>12.6f} | {r['var_mbr']:>12.6f} | "
              f"{r['var_ratio_greedy_over_mbr']:>8.4f} | {r['mean_cer_greedy']:>10.6f} | "
              f"{r['mean_cer_mbr']:>10.6f} | {r['n_mbr_different']:>8} | "
              f"{r['frac_mbr_different']:>6.3f}", flush=True)

    for r in all_results:
        assert r["var_ratio_greedy_over_mbr"] >= 1.0 - 1e-6, \
            f"Prop 4.1 violated at tau={r['tau']}: ratio={r['var_ratio_greedy_over_mbr']:.6f}"
        assert r["mean_cer_mbr"] <= r["mean_cer_greedy"] + 1e-6, \
            f"MBR mean CER > greedy at tau={r['tau']}"

    print("\nAll Prop 4.1 assertions passed.", flush=True)


if __name__ == "__main__":
    main()

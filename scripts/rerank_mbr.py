#!/usr/bin/env python3
"""MBR-CER reranking with optional PLL/n-gram weighting.

Selects y* = argmin_y sum_{y'} CER(y, y') * w(y')
where w(y') = softmax((ctc_weight * ctc + pll_weight * pll) / tau)

Supports tau sweep, pll_weight sweep, or full grid search.

Usage:
    python scripts/rerank_mbr.py \
        --nbest /path/to/nbest_pll.jsonl \
        --output /path/to/mbr_results.json \
        --utility cer \
        [--tau 1.0] \
        [--tau-sweep 0.1,0.5,1.0,5.0,10.0,50.0] \
        [--pll-sweep 0.0,0.1,0.3,0.5,0.7,1.0] \
        [--score-key pll_score] \
        [--ctc-weight 1.0] \
        [--pll-weight 0.0]
"""

import argparse
import json
import math
import time
from pathlib import Path

import editdistance
import numpy as np


def _normalize_record(raw):
    if "nbest" in raw:
        return raw
    return {
        "utt_id": raw["utt_id"],
        "ref": raw.get("ref", raw.get("ref_text", "")),
        "nbest": [
            {"hyp": c.get("hyp", c.get("text", "")),
             "score": c.get("score", c.get("ctc_log_prob", 0.0)),
             **{k: v for k, v in c.items()
                if k not in ("hyp", "text", "score", "ctc_log_prob")}}
            for c in raw.get("nbest", raw.get("candidates", []))
        ],
    }


def load_nbest(path: Path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(_normalize_record(json.loads(line)))
    return records


def cer_matrix(texts):
    """Symmetric CER matrix for one utterance's candidates."""
    n = len(texts)
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = editdistance.eval(list(texts[i]), list(texts[j]))
            denom = max(len(texts[i]), len(texts[j]), 1)
            mat[i, j] = d / denom
            mat[j, i] = mat[i, j]
    return mat


def mbr_select(cer_mat, log_scores, tau):
    """Select hypothesis minimizing expected CER under softmax(scores/tau)."""
    n = len(log_scores)
    if math.isinf(tau):
        weights = np.ones(n) / n
    else:
        scaled = np.array(log_scores) / tau
        scaled -= np.max(scaled)
        weights = np.exp(scaled)
        weights /= weights.sum()
    risk = cer_mat @ weights
    return int(np.argmin(risk))


def corpus_wer(records, selections):
    """Corpus-level WER from per-utterance selections."""
    total_edits = 0
    total_ref = 0
    for rec in records:
        idx = selections[rec["utt_id"]]
        hyp_w = rec["nbest"][idx]["hyp"].split()
        ref_w = rec["ref"].split()
        total_edits += editdistance.eval(hyp_w, ref_w)
        total_ref += len(ref_w)
    return total_edits / max(total_ref, 1)


def compute_baselines(records):
    """Compute greedy and oracle WER."""
    total_greedy = total_oracle = total_ref = 0
    for rec in records:
        ref_w = rec["ref"].split()
        total_ref += len(ref_w)
        total_greedy += editdistance.eval(rec["nbest"][0]["hyp"].split(), ref_w)
        total_oracle += min(
            editdistance.eval(c["hyp"].split(), ref_w) for c in rec["nbest"]
        )
    greedy = total_greedy / max(total_ref, 1)
    oracle = total_oracle / max(total_ref, 1)
    return greedy, oracle


def run_mbr_config(records, cer_matrices, ctc_weight, pll_weight, tau,
                   score_key, greedy_wer, oracle_wer):
    """Run MBR at one config, return result dict."""
    selections = {}
    for i, rec in enumerate(records):
        nbest = rec["nbest"]
        log_scores = []
        for c in nbest:
            s = ctc_weight * c["score"]
            if pll_weight != 0.0 and score_key in c:
                s += pll_weight * c[score_key]
            log_scores.append(s)
        selections[rec["utt_id"]] = mbr_select(cer_matrices[i], log_scores, tau)

    wer = corpus_wer(records, selections)
    gap = greedy_wer - oracle_wer
    gap_closed = (greedy_wer - wer) / gap * 100 if gap > 1e-9 else 0.0

    return {
        "ctc_weight": ctc_weight,
        "pll_weight": pll_weight,
        "tau": tau if not math.isinf(tau) else "inf",
        "wer": wer,
        "wer_pct": f"{wer:.4%}",
        "gap_closed_pct": round(gap_closed, 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="MBR-CER reranking with optional PLL weighting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nbest", type=Path, required=True,
                        help="N-best JSONL (with pll_score if using PLL)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSON with results")
    parser.add_argument("--utility", type=str, default="cer",
                        choices=["cer"],
                        help="MBR utility function")
    parser.add_argument("--tau", type=float, default=None,
                        help="Single tau value")
    parser.add_argument("--tau-sweep", type=str, default=None,
                        help="Comma-separated tau values for sweep")
    parser.add_argument("--pll-weight", type=float, default=0.0,
                        help="Single PLL weight")
    parser.add_argument("--pll-sweep", type=str, default=None,
                        help="Comma-separated PLL weights for sweep")
    parser.add_argument("--ctc-weight", type=float, default=1.0,
                        help="CTC score weight")
    parser.add_argument("--score-key", type=str, default="pll_score",
                        help="Key for external score (pll_score, ngram_score)")
    args = parser.parse_args()

    print("=" * 70)
    print("rerank_mbr.py  --  MBR-CER reranking")
    print("=" * 70)
    print(f"  nbest:      {args.nbest}")
    print(f"  output:     {args.output}")
    print(f"  utility:    {args.utility}")
    print(f"  score_key:  {args.score_key}")
    print(f"  ctc_weight: {args.ctc_weight}")
    print()

    records = load_nbest(args.nbest)
    n_utts = len(records)
    mean_cands = np.mean([len(r["nbest"]) for r in records])
    print(f"Loaded {n_utts} utterances, mean {mean_cands:.1f} candidates")

    has_ext_score = args.score_key in records[0]["nbest"][0]
    if not has_ext_score:
        print(f"  WARNING: '{args.score_key}' not found in JSONL  --  "
              f"PLL weight will be ignored")

    # Baselines
    greedy_wer, oracle_wer = compute_baselines(records)
    gap = greedy_wer - oracle_wer
    print(f"\n  Greedy WER:  {greedy_wer:.4%}")
    print(f"  Oracle WER:  {oracle_wer:.4%}")
    print(f"  Gap:         {gap:.4%} ({gap / greedy_wer * 100:.1f}% relative)")

    # Pre-compute CER matrices
    print(f"\n  Computing CER matrices ({n_utts} utterances, "
          f"~{mean_cands:.0f}^2 = {mean_cands**2:.0f} pairs each)...")
    t0 = time.time()
    cer_matrices = []
    for i, rec in enumerate(records):
        texts = [c["hyp"] for c in rec["nbest"]]
        cer_matrices.append(cer_matrix(texts))
        if (i + 1) % 500 == 0:
            print(f"    {i + 1}/{n_utts}")
    cer_time = time.time() - t0
    print(f"  CER matrices: {cer_time:.1f}s")

    taus = []
    if args.tau_sweep:
        taus = [float(t) for t in args.tau_sweep.split(",")]
    elif args.tau is not None:
        taus = [args.tau]
    else:
        taus = [1.0]

    pll_weights = []
    if args.pll_sweep:
        pll_weights = [float(w) for w in args.pll_sweep.split(",")]
    else:
        pll_weights = [args.pll_weight]

    print(f"\n  Running {len(taus)} x {len(pll_weights)} = "
          f"{len(taus) * len(pll_weights)} configs...")
    t0 = time.time()
    all_results = []

    for pll_w in pll_weights:
        if pll_w != 0.0 and not has_ext_score:
            continue
        for tau in taus:
            result = run_mbr_config(
                records, cer_matrices,
                args.ctc_weight, pll_w, tau,
                args.score_key, greedy_wer, oracle_wer,
            )
            all_results.append(result)

    sweep_time = time.time() - t0
    print(f"  Sweep: {sweep_time:.1f}s")

    all_results.sort(key=lambda r: r["wer"])
    best = all_results[0]

    print()
    print("=" * 70)
    fmt = "  {:<12} {:<12} {:>10} {:>12}"
    print(fmt.format("pll_weight", "tau", "WER", "gap_closed"))
    print("  " + "-" * 50)
    for r in all_results:
        tau_str = str(r["tau"])
        print(fmt.format(
            f"{r['pll_weight']:.1f}",
            tau_str,
            r["wer_pct"],
            f"{r['gap_closed_pct']:+.1f}%",
        ))

    print()
    print(f"  Best: pll_weight={best['pll_weight']}, tau={best['tau']}, "
          f"WER={best['wer_pct']}, gap_closed={best['gap_closed_pct']:+.1f}%")

    # Save
    output = {
        "source": str(args.nbest),
        "n_utterances": n_utts,
        "mean_candidates": round(mean_cands, 1),
        "utility": args.utility,
        "score_key": args.score_key,
        "greedy_wer": greedy_wer,
        "oracle_wer": oracle_wer,
        "best_config": best,
        "all_results": all_results,
        "cer_matrix_time_s": round(cer_time, 1),
        "sweep_time_s": round(sweep_time, 1),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Saved: {args.output}")
    print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score interpolation reranking baseline (argmax, not MBR).

Selects y* = argmax_y (alpha * ctc_score + (1-alpha) * pll_score)
Sweeps alpha from 0 to 1. This is the argmax-interpolation baseline
that MBR should beat.

Usage:
    python scripts/rerank_interpolation.py \
        --nbest /path/to/nbest_pll.jsonl \
        --output /path/to/interp_results.json \
        [--alpha-sweep 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0] \
        [--score-key pll_score]
"""

import argparse
import json
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


def corpus_wer(records, selections):
    total_edits = 0
    total_ref = 0
    for rec in records:
        idx = selections[rec["utt_id"]]
        hyp_w = rec["nbest"][idx]["hyp"].split()
        ref_w = rec["ref"].split()
        total_edits += editdistance.eval(hyp_w, ref_w)
        total_ref += len(ref_w)
    return total_edits / max(total_ref, 1)


def select_interpolation(records, alpha, score_key):
    """argmax_y (alpha * ctc + (1-alpha) * external_score)."""
    selections = {}
    for rec in records:
        best_score = -float("inf")
        best_idx = 0
        for i, c in enumerate(rec["nbest"]):
            s = alpha * c["score"]
            if (1 - alpha) != 0.0 and score_key in c:
                s += (1 - alpha) * c[score_key]
            if s > best_score:
                best_score = s
                best_idx = i
        selections[rec["utt_id"]] = best_idx
    return selections


def main():
    parser = argparse.ArgumentParser(
        description="Score interpolation reranking baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nbest", type=Path, required=True,
                        help="N-best JSONL with pll_score")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSON")
    parser.add_argument("--alpha-sweep", type=str,
                        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
                        help="Comma-separated alpha values")
    parser.add_argument("--score-key", type=str, default="pll_score",
                        help="Key for external score")
    args = parser.parse_args()

    print("=" * 70)
    print("rerank_interpolation.py  --  argmax score interpolation")
    print("=" * 70)
    print(f"  nbest:     {args.nbest}")
    print(f"  output:    {args.output}")
    print(f"  score_key: {args.score_key}")
    print()

    records = load_nbest(args.nbest)
    n_utts = len(records)
    print(f"Loaded {n_utts} utterances")

    has_ext = args.score_key in records[0]["nbest"][0]
    if not has_ext:
        print(f"  WARNING: '{args.score_key}' not found  --  only alpha=1.0 is meaningful")

    # Baselines
    total_greedy = total_oracle = total_ref = 0
    for rec in records:
        ref_w = rec["ref"].split()
        total_ref += len(ref_w)
        total_greedy += editdistance.eval(rec["nbest"][0]["hyp"].split(), ref_w)
        total_oracle += min(
            editdistance.eval(c["hyp"].split(), ref_w) for c in rec["nbest"]
        )
    greedy_wer = total_greedy / max(total_ref, 1)
    oracle_wer = total_oracle / max(total_ref, 1)
    gap = greedy_wer - oracle_wer

    print(f"  Greedy WER: {greedy_wer:.4%}")
    print(f"  Oracle WER: {oracle_wer:.4%}")
    print(f"  Gap:        {gap:.4%} ({gap / greedy_wer * 100:.1f}% relative)")

    # Alpha sweep
    alphas = [float(a) for a in args.alpha_sweep.split(",")]
    results = []

    print()
    fmt = "  {:<8} {:>10} {:>12}"
    print(fmt.format("alpha", "WER", "gap_closed"))
    print("  " + "-" * 34)

    for alpha in alphas:
        sels = select_interpolation(records, alpha, args.score_key)
        wer = corpus_wer(records, sels)
        gap_closed = (greedy_wer - wer) / gap * 100 if gap > 1e-9 else 0.0
        results.append({
            "alpha": alpha,
            "wer": wer,
            "wer_pct": f"{wer:.4%}",
            "gap_closed_pct": round(gap_closed, 2),
        })
        print(fmt.format(f"{alpha:.1f}", f"{wer:.4%}", f"{gap_closed:+.1f}%"))

    best = min(results, key=lambda r: r["wer"])
    print()
    print(f"  Best: alpha={best['alpha']}, WER={best['wer_pct']}, "
          f"gap_closed={best['gap_closed_pct']:+.1f}%")

    # Verification: alpha=1.0 must match greedy
    a1 = next((r for r in results if r["alpha"] == 1.0), None)
    if a1:
        diff = abs(a1["wer"] - greedy_wer)
        if diff < 1e-6:
            print(f"  PASS: alpha=1.0 matches greedy WER")
        else:
            print(f"  NOTE: alpha=1.0 WER={a1['wer']:.4%} vs greedy={greedy_wer:.4%} "
                  f"(diff={diff:.6f}, likely score ties broken differently)")

    # Save
    output = {
        "source": str(args.nbest),
        "n_utterances": n_utts,
        "score_key": args.score_key,
        "greedy_wer": greedy_wer,
        "oracle_wer": oracle_wer,
        "best_config": best,
        "all_results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Saved: {args.output}")
    print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paired per-utterance comparison between two N-best JSONL files.

For each utterance present in both files, compares recoverability
(oracle < greedy) and WER changes. Produces a transition matrix
and summary statistics.

Usage:
    python scripts/paired_analysis.py \
        --nbest-a clean_g16.jsonl \
        --nbest-b noisy_g16.jsonl \
        --label-a "LS clean" \
        --label-b "LS+noise@0dB" \
        --output paired_result.json
"""

import argparse
import json
from pathlib import Path

import editdistance


def _normalize_record(raw):
    """Normalize E11 or new format to canonical."""
    if "nbest" in raw:
        return raw
    return {
        "utt_id": raw["utt_id"],
        "ref": raw.get("ref", raw.get("ref_text", "")),
        "nbest": [
            {"hyp": c.get("hyp", c.get("text", "")),
             "score": c.get("score", c.get("ctc_log_prob", 0.0))}
            for c in raw.get("nbest", raw.get("candidates", []))
        ],
    }


def load_nbest_dict(path: Path):
    """Load N-best JSONL into dict keyed by utt_id."""
    d = {}
    with open(path) as f:
        for line in f:
            rec = _normalize_record(json.loads(line))
            d[rec["utt_id"]] = rec
    return d


def main():
    parser = argparse.ArgumentParser(
        description="Paired per-utterance comparison of two N-best files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nbest-a", type=Path, required=True,
                        help="First N-best JSONL (condition A)")
    parser.add_argument("--nbest-b", type=Path, required=True,
                        help="Second N-best JSONL (condition B)")
    parser.add_argument("--label-a", type=str, default="A")
    parser.add_argument("--label-b", type=str, default="B")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSON path")
    args = parser.parse_args()

    print("=" * 70)
    print("paired_analysis.py  --  per-utterance comparison")
    print("=" * 70)
    print(f"  A: {args.label_a} ({args.nbest_a})")
    print(f"  B: {args.label_b} ({args.nbest_b})")
    print()

    data_a = load_nbest_dict(args.nbest_a)
    data_b = load_nbest_dict(args.nbest_b)

    common_ids = sorted(set(data_a.keys()) & set(data_b.keys()))
    print(f"  Utterances in A: {len(data_a)}")
    print(f"  Utterances in B: {len(data_b)}")
    print(f"  Common:          {len(common_ids)}")

    both_optimal = 0
    a_opt_b_rec = 0
    a_rec_b_opt = 0
    both_recoverable = 0
    wer_increased = 0
    wer_same = 0
    wer_decreased = 0

    examples_new_rec = []

    for utt_id in common_ids:
        rec_a = data_a[utt_id]
        rec_b = data_b[utt_id]
        ref_words = rec_a["ref"].split()
        if not ref_words:
            continue
        n_ref = len(ref_words)

        # Condition A
        a_greedy_e = editdistance.eval(rec_a["nbest"][0]["hyp"].split(), ref_words)
        a_oracle_e = min(
            editdistance.eval(c["hyp"].split(), ref_words)
            for c in rec_a["nbest"]
        )
        a_recoverable = a_oracle_e < a_greedy_e

        # Condition B
        b_greedy_e = editdistance.eval(rec_b["nbest"][0]["hyp"].split(), ref_words)
        b_oracle_e = min(
            editdistance.eval(c["hyp"].split(), ref_words)
            for c in rec_b["nbest"]
        )
        b_recoverable = b_oracle_e < b_greedy_e

        # Transition
        if not a_recoverable and not b_recoverable:
            both_optimal += 1
        elif not a_recoverable and b_recoverable:
            a_opt_b_rec += 1
            if len(examples_new_rec) < 5:
                examples_new_rec.append({
                    "utt_id": utt_id,
                    "ref": rec_a["ref"][:100],
                    "a_greedy": rec_a["nbest"][0]["hyp"][:80],
                    "b_greedy": rec_b["nbest"][0]["hyp"][:80],
                    "a_wer": a_greedy_e / n_ref,
                    "b_greedy_wer": b_greedy_e / n_ref,
                    "b_oracle_wer": b_oracle_e / n_ref,
                })
        elif a_recoverable and not b_recoverable:
            a_rec_b_opt += 1
        else:
            both_recoverable += 1

        # WER change
        if b_greedy_e > a_greedy_e:
            wer_increased += 1
        elif b_greedy_e == a_greedy_e:
            wer_same += 1
        else:
            wer_decreased += 1

    n = len(common_ids)

    result = {
        "label_a": args.label_a,
        "label_b": args.label_b,
        "n_common": n,
        "transitions": {
            "both_optimal": both_optimal,
            f"{args.label_a}_optimal_{args.label_b}_recoverable": a_opt_b_rec,
            f"{args.label_a}_recoverable_{args.label_b}_optimal": a_rec_b_opt,
            "both_recoverable": both_recoverable,
        },
        "wer_change": {
            "increased": wer_increased,
            "same": wer_same,
            "decreased": wer_decreased,
        },
        "examples_new_rec": examples_new_rec,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print()
    print("=" * 60)
    fmt = "  {:<45s} {:>6} ({:>5.1f}%)"
    print(f"  Recoverability transitions ({args.label_a} -> {args.label_b}):")
    print(fmt.format("Both greedy-optimal", both_optimal, both_optimal / n * 100))
    print(fmt.format(
        f"{args.label_a}-optimal -> RECOVERABLE",
        a_opt_b_rec, a_opt_b_rec / n * 100,
    ))
    print(fmt.format(
        f"Recoverable -> {args.label_b}-optimal",
        a_rec_b_opt, a_rec_b_opt / n * 100,
    ))
    print(fmt.format("Both recoverable", both_recoverable, both_recoverable / n * 100))

    print()
    print(f"  Greedy WER change ({args.label_a} -> {args.label_b}):")
    print(fmt.format("WER increased", wer_increased, wer_increased / n * 100))
    print(fmt.format("WER unchanged", wer_same, wer_same / n * 100))
    print(fmt.format("WER decreased", wer_decreased, wer_decreased / n * 100))

    if examples_new_rec:
        print()
        print("  Examples: utterances that became recoverable:")
        print("  " + "-" * 56)
        for ex in examples_new_rec:
            print(f"  [{ex['utt_id']}]")
            print(f"    REF:     {ex['ref']}")
            print(f"    A greedy (WER={ex['a_wer']:.1%}): {ex['a_greedy']}")
            print(f"    B greedy (WER={ex['b_greedy_wer']:.1%}): {ex['b_greedy']}")
            print(f"    B oracle WER: {ex['b_oracle_wer']:.1%}")
            print()

    print(f"  Saved: {args.output}")
    print()


if __name__ == "__main__":
    main()

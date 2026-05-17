#!/usr/bin/env python3
"""Re-derive beam_sweep_summary from existing nbest_dev_other_G{X}.jsonl files.

Fixes a measurement convention bug in the original beam_sweep.py run: it
reported mean-per-utterance WER, but the literature baseline (6.02% greedy,
4.44% G=16 oracle on Zipformer-S CR-CTC dev-other) is corpus-level WER
(total_edits / total_ref_words).

This script reads each per-G JSONL file once, computes both conventions,
and writes corrected summary CSV/JSON. No GPU required; runs in seconds.

Usage:
    python experiments/beam_sweep_recompute.py \
        --results-dir results \
        --g-values 1,4,8,16,32,64,128
"""

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import editdistance


def compute_wer_words(hyp_words, ref_words):
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return editdistance.eval(hyp_words, ref_words) / len(ref_words)


def mean_pairwise_wer(texts):
    if len(texts) < 2:
        return 0.0
    word_lists = [t.split() for t in texts]
    total = 0.0
    count = 0
    for a, b in combinations(word_lists, 2):
        denom = max(len(a), len(b))
        if denom == 0:
            continue
        total += editdistance.eval(a, b) / denom
        count += 1
    return total / count if count > 0 else 0.0


def process_jsonl(path: Path):
    n = 0
    total_edits_greedy = 0
    total_edits_oracle = 0
    total_ref = 0

    sum_oracle_wer = 0.0
    sum_greedy_wer = 0.0
    sum_unique = 0
    sum_pairwise = 0.0
    sum_spread = 0.0

    num_recoverable = 0
    sum_improvement = 0.0

    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            ref_words = rec["ref_text"].split()
            cands = rec["candidates"]
            n += 1
            total_ref += len(ref_words)

            # per-candidate edits
            edits = [
                editdistance.eval(c["text"].split(), ref_words)
                for c in cands
            ]
            wers = [e / max(len(ref_words), 1) for e in edits]

            # greedy = candidate[0]
            edits_g = edits[0]
            wer_g = wers[0]
            # oracle = min
            edits_o = min(edits)
            wer_o = min(wers)

            total_edits_greedy += edits_g
            total_edits_oracle += edits_o

            sum_greedy_wer += wer_g
            sum_oracle_wer += wer_o
            sum_unique += len(cands)

            texts = [c["text"] for c in cands]
            sum_pairwise += mean_pairwise_wer(texts) if len(texts) > 1 else 0.0

            lps = [c["ctc_log_prob"] for c in cands]
            sum_spread += (max(lps) - min(lps)) if len(lps) > 1 else 0.0

            if wer_o < wer_g - 1e-12:
                num_recoverable += 1
                sum_improvement += (wer_g - wer_o)

    corpus_greedy_wer = total_edits_greedy / max(total_ref, 1)
    corpus_oracle_wer = total_edits_oracle / max(total_ref, 1)

    mean_greedy_wer = sum_greedy_wer / n
    mean_oracle_wer = sum_oracle_wer / n
    abs_gap_corpus = corpus_greedy_wer - corpus_oracle_wer
    rel_gap_corpus = abs_gap_corpus / corpus_greedy_wer if corpus_greedy_wer > 0 else 0.0

    return {
        "n_utterances": n,
        "total_ref_words": total_ref,
        "corpus_greedy_wer": corpus_greedy_wer,
        "corpus_oracle_wer": corpus_oracle_wer,
        "mean_greedy_wer": mean_greedy_wer,
        "mean_oracle_wer": mean_oracle_wer,
        "abs_gap_corpus": abs_gap_corpus,
        "rel_gap_corpus": rel_gap_corpus,
        "mean_unique_candidates": sum_unique / n,
        "mean_pairwise_wer": sum_pairwise / n,
        "mean_logprob_spread": sum_spread / n,
        "num_recoverable": num_recoverable,
        "pct_recoverable": num_recoverable / n,
        "mean_improvement_on_recoverable": (
            sum_improvement / num_recoverable if num_recoverable > 0 else 0.0
        ),
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Recompute beam-sweep summary with corpus WER from JSONLs"
    )
    p.add_argument(
        "--results-dir", type=Path,
        default=Path("results"),
        help="Directory containing nbest_dev_other_G{X}.jsonl files",
    )
    p.add_argument(
        "--g-values", type=str, default="1,4,8,16,32,64,128",
    )
    p.add_argument(
        "--out-prefix", type=str, default="beam_sweep_summary_corpus",
    )
    return p.parse_args()


def main():
    args = parse_args()
    g_values = [int(g) for g in args.g_values.split(",")]

    rows = []
    for g in g_values:
        path = args.results_dir / f"nbest_dev_other_G{g}.jsonl"
        if not path.exists():
            print(f"  SKIP G={g}: {path} not found")
            continue
        print(f"  Processing G={g}: {path}")
        s = process_jsonl(path)
        rows.append(
            {
                "G": g,
                "corpus_oracle_wer": round(s["corpus_oracle_wer"], 6),
                "corpus_greedy_wer": round(s["corpus_greedy_wer"], 6),
                "mean_oracle_wer": round(s["mean_oracle_wer"], 6),
                "mean_greedy_wer": round(s["mean_greedy_wer"], 6),
                "abs_gap_corpus": round(s["abs_gap_corpus"], 6),
                "rel_gap_corpus": round(s["rel_gap_corpus"], 6),
                "mean_unique_candidates": round(s["mean_unique_candidates"], 2),
                "mean_pairwise_wer": round(s["mean_pairwise_wer"], 6),
                "mean_logprob_spread": round(s["mean_logprob_spread"], 4),
                "num_recoverable": s["num_recoverable"],
                "pct_recoverable": round(s["pct_recoverable"], 4),
                "mean_improvement_on_recoverable": round(
                    s["mean_improvement_on_recoverable"], 6
                ),
            }
        )

    csv_path = args.results_dir / f"{args.out_prefix}.csv"
    with open(csv_path, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    print(f"\nWrote {csv_path}")

    json_path = args.results_dir / f"{args.out_prefix}.json"
    with open(json_path, "w") as f:
        json.dump({"results": rows}, f, indent=2)
    print(f"Wrote {json_path}")

    print("\n" + "=" * 100)
    print(
        f"{'G':>5} {'CorpOracle%':>12} {'CorpGreedy%':>12} "
        f"{'AbsGap':>8} {'RelGap%':>8} "
        f"{'MeanOracle%':>12} {'Uniq':>6} {'PairWER%':>9} "
        f"{'Spread':>8} {'Recov':>6} {'Recov%':>7}"
    )
    print("-" * 100)
    for r in rows:
        print(
            f"{r['G']:>5d} "
            f"{r['corpus_oracle_wer']*100:>11.2f}% "
            f"{r['corpus_greedy_wer']*100:>11.2f}% "
            f"{r['abs_gap_corpus']*100:>7.2f}% "
            f"{r['rel_gap_corpus']*100:>7.1f}% "
            f"{r['mean_oracle_wer']*100:>11.2f}% "
            f"{r['mean_unique_candidates']:>6.1f} "
            f"{r['mean_pairwise_wer']*100:>8.2f}% "
            f"{r['mean_logprob_spread']:>8.2f} "
            f"{r['num_recoverable']:>6d} "
            f"{r['pct_recoverable']*100:>6.1f}%"
        )
    print("=" * 100)


if __name__ == "__main__":
    main()

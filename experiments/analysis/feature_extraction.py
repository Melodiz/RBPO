#!/usr/bin/env python3
"""Extract per-candidate feature vectors from N-best JSONL files.

Reads nbest JSONL (same format as generate_nbest.py output), computes
14 features per candidate, and writes a CSV with one row per candidate.

Features:
  CTC-derived (4):    ctc_log_prob, ctc_log_prob_per_token,
                      ctc_log_prob_per_char, ctc_rank
  Length (4):         len_tokens, len_chars, len_words, len_deviation
  Agreement (3):     mean_cer_to_others, mean_wer_to_others,
                     agrees_with_majority
  Probability (3):   log_prob_gap, ptilde, entropy_of_group

Target:              wer (WER against reference)

Usage:
    python experiments/feature_extraction.py \
        --nbest-file results/nbest_dev_other_G16.jsonl \
        --output results/features_dev.csv

    python experiments/feature_extraction.py \
        --nbest-file results/nbest_train_clean100_G16.jsonl \
        --output results/features_train.csv
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import editdistance
import numpy as np
from tqdm import tqdm


def compute_wer(hypothesis: str, reference: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return editdistance.eval(hyp_words, ref_words) / len(ref_words)


def log_softmax(log_probs: list[float]) -> np.ndarray:
    a = np.array(log_probs, dtype=np.float64)
    max_a = np.max(a)
    log_sum = max_a + np.log(np.sum(np.exp(a - max_a)))
    return a - log_sum


def char_distance_norm(a, b):
    denom = max(len(a), len(b), 1)
    return editdistance.eval(list(a), list(b)) / denom


def word_distance_norm(a, b):
    wa, wb = a.split(), b.split()
    denom = max(len(wa), len(wb), 1)
    return editdistance.eval(wa, wb) / denom


def agrees_with_majority(candidate_words: list[str], all_candidates_words: list[list[str]]) -> float:
    """Fraction of positions where candidate agrees with majority word."""
    if not candidate_words:
        return 1.0

    max_len = max(len(cw) for cw in all_candidates_words)
    if max_len == 0:
        return 1.0

    agree_count = 0
    total_positions = len(candidate_words)

    for pos in range(total_positions):
        counter = Counter()
        for cw in all_candidates_words:
            if pos < len(cw):
                counter[cw[pos]] += 1
        if counter:
            majority_word = counter.most_common(1)[0][0]
            if candidate_words[pos] == majority_word:
                agree_count += 1

    return agree_count / total_positions


FEATURE_NAMES = [
    "ctc_log_prob",
    "ctc_log_prob_per_token",
    "ctc_log_prob_per_char",
    "ctc_rank",
    "len_tokens",
    "len_chars",
    "len_words",
    "len_deviation",
    "mean_cer_to_others",
    "mean_wer_to_others",
    "agrees_with_majority",
    "log_prob_gap",
    "ptilde",
    "entropy_of_group",
]


def extract_features(record: dict) -> list[dict]:
    """Extract features for all candidates in one utterance."""
    ref = record["ref_text"]
    cands = record["candidates"]
    n = len(cands)

    texts = [c["text"] for c in cands]
    log_probs = [c["ctc_log_prob"] for c in cands]
    len_toks = [c["len_tokens"] for c in cands]
    len_chars_list = [c["len_chars"] for c in cands]

    max_log_prob = max(log_probs)

    log_p_norm = log_softmax(log_probs)
    p_tilde = np.exp(log_p_norm)
    entropy = -np.sum(p_tilde * log_p_norm)

    sorted_indices = np.argsort(log_probs)[::-1]
    ranks = np.empty(n, dtype=int)
    for rank, idx in enumerate(sorted_indices):
        ranks[idx] = rank

    mean_len_tok = np.mean(len_toks)
    std_len_tok = max(np.std(len_toks), 1e-6)

    words_list = [t.split() for t in texts]

    rows = []
    for i in range(n):
        lt = max(len_toks[i], 1)
        lc = max(len_chars_list[i], 1)
        lw = len(words_list[i])

        cer_sum = 0.0
        wer_sum = 0.0
        for j in range(n):
            if i == j:
                continue
            cer_sum += char_distance_norm(texts[i], texts[j])
            wer_sum += word_distance_norm(texts[i], texts[j])
        denom = max(n - 1, 1)

        row = {
            "utt_id": record["utt_id"],
            "candidate_idx": i,
            "ctc_log_prob": log_probs[i],
            "ctc_log_prob_per_token": log_probs[i] / lt,
            "ctc_log_prob_per_char": log_probs[i] / lc,
            "ctc_rank": ranks[i],
            "len_tokens": len_toks[i],
            "len_chars": len_chars_list[i],
            "len_words": lw,
            "len_deviation": (len_toks[i] - mean_len_tok) / std_len_tok,
            "mean_cer_to_others": cer_sum / denom,
            "mean_wer_to_others": wer_sum / denom,
            "agrees_with_majority": agrees_with_majority(words_list[i], words_list),
            "log_prob_gap": log_probs[i] - max_log_prob,
            "ptilde": p_tilde[i],
            "entropy_of_group": entropy,
            "wer": compute_wer(texts[i], ref),
        }
        rows.append(row)

    return rows


def load_nbest(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} utterances from {path}")
    return records


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract per-candidate features from N-best JSONL"
    )
    parser.add_argument(
        "--nbest-file", type=Path, required=True,
        help="Input JSONL file (e.g. results/nbest_dev_other_G16.jsonl)",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output CSV file (e.g. results/features_dev.csv)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Feature Extraction for Discriminative Rescorer")
    print("=" * 60)

    records = load_nbest(args.nbest_file)

    all_rows = []
    for rec in tqdm(records, desc="Extracting features"):
        all_rows.extend(extract_features(rec))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["utt_id", "candidate_idx"] + FEATURE_NAMES + ["wer"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                             for k, v in row.items()})

    n_utts = len(records)
    n_cands = len(all_rows)
    print(f"\nDone: {n_utts} utterances, {n_cands} candidates")
    print(f"Avg candidates per utterance: {n_cands / n_utts:.1f}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()

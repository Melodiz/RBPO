#!/usr/bin/env python3
"""B2: Complete TL3 G=16 PLL scoring for missing utterances.

The existing nbest_g16_pll.jsonl has 700 utterances. The full TL3 test set
has 1155 utterances. This script scores the ~455 missing utterances with
RoBERTa PLL and re-runs MBR + interpolation on the full 1155.

Usage (Colab):
    python experiments/robustness/complete_tl3_g16_pll.py \
        --scored-jsonl results/tl3_rerun/nbest_g16_pll.jsonl \
        --full-nbest /path/to/full_nbest_g16.jsonl \
        --output-dir results/tl3_rerun/ \
        --roberta-model roberta-base
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def save_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def compute_pll_batch(texts, tokenizer, model, device, batch_size=32):
    import torch

    mask_id = tokenizer.mask_token_id
    special = {tokenizer.bos_token_id, tokenizer.eos_token_id,
               tokenizer.pad_token_id, tokenizer.cls_token_id,
               tokenizer.sep_token_id}
    special.discard(None)

    scores = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"][0].to(device)
        L = input_ids.size(0)
        positions = [i for i in range(L) if input_ids[i].item() not in special]
        if not positions:
            scores.append(0.0)
            continue

        total = 0.0
        for start in range(0, len(positions), batch_size):
            batch_pos = positions[start:start + batch_size]
            B = len(batch_pos)
            masked = input_ids.unsqueeze(0).expand(B, -1).clone()
            for b, pos in enumerate(batch_pos):
                masked[b, pos] = mask_id

            with torch.no_grad():
                logits = model(masked).logits

            for b, pos in enumerate(batch_pos):
                log_prob = torch.nn.functional.log_softmax(logits[b, pos], dim=-1)
                total += log_prob[input_ids[pos]].item()

        scores.append(total)
    return scores


def corpus_wer(ref_words_list, hyp_words_list):
    import editdistance
    total_errors = 0
    total_ref = 0
    for ref_w, hyp_w in zip(ref_words_list, hyp_words_list):
        total_errors += editdistance.eval(hyp_w, ref_w)
        total_ref += len(ref_w)
    return total_errors / max(total_ref, 1)


def mbr_select(texts, log_scores, tau):
    import editdistance
    n = len(texts)
    if math.isinf(tau):
        weights = np.ones(n) / n
    else:
        scaled = np.array(log_scores) / tau
        scaled -= np.max(scaled)
        weights = np.exp(scaled)
        weights /= weights.sum()

    cer_mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = editdistance.eval(list(texts[i]), list(texts[j]))
            denom = max(len(texts[i]), len(texts[j]), 1)
            cer_mat[i, j] = d / denom
            cer_mat[j, i] = cer_mat[i, j]

    risk = cer_mat @ weights
    return int(np.argmin(risk))


def main():
    parser = argparse.ArgumentParser(
        description="B2: Complete TL3 G=16 PLL scoring",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scored-jsonl", type=Path,
                        default=REPO_ROOT / "results" / "tl3_rerun" / "nbest_g16_pll.jsonl")
    parser.add_argument("--full-nbest", type=Path, required=True,
                        help="Full 1155-utt G=16 nbest JSONL (on Drive)")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "results" / "tl3_rerun")
    parser.add_argument("--roberta-model", type=str, default="roberta-base")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tau", type=float, default=10.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("B2: Complete TL3 G=16 PLL Scoring")
    print("=" * 70)
    t0 = time.time()

    print(f"\nStep 1: Load existing scored JSONL")
    scored = load_jsonl(args.scored_jsonl)
    scored_ids = {r["utt_id"] for r in scored}
    print(f"  Existing scored: {len(scored_ids)} utterances")

    print(f"\nStep 2: Load full N-best")
    full = load_jsonl(args.full_nbest)
    full_ids = {r["utt_id"] for r in full}
    print(f"  Full set: {len(full_ids)} utterances")

    missing_ids = full_ids - scored_ids
    print(f"  Missing: {len(missing_ids)} utterances to score")

    if len(missing_ids) == 0:
        print("  Nothing to do!")
        return

    print(f"\nStep 3: Score missing utterances with RoBERTa PLL")
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    tokenizer = AutoTokenizer.from_pretrained(args.roberta_model)
    model = AutoModelForMaskedLM.from_pretrained(args.roberta_model)
    model.eval().to(args.device)
    print(f"  Model loaded on {args.device}")

    missing_records = [r for r in full if r["utt_id"] in missing_ids]
    for i, rec in enumerate(missing_records):
        texts = [h["hyp"] for h in rec["nbest"]]
        pll_scores = compute_pll_batch(texts, tokenizer, model, args.device, args.batch_size)
        for h, s in zip(rec["nbest"], pll_scores):
            h["pll_score"] = s

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(missing_records)} utterances scored")

    print(f"\nStep 4: Merge and save")
    scored_map = {r["utt_id"]: r for r in scored}
    for rec in missing_records:
        scored_map[rec["utt_id"]] = rec

    all_records = sorted(scored_map.values(), key=lambda r: r["utt_id"])
    merged_path = args.output_dir / "nbest_g16_pll_full1155.jsonl"
    save_jsonl(all_records, merged_path)
    print(f"  Saved {len(all_records)} utterances to {merged_path}")

    print(f"\nStep 5: Re-run MBR-CER+PLL tau={args.tau}")
    import editdistance

    ref_words = []
    greedy_words = []
    mbr_words = []

    for rec in all_records:
        ref = rec["ref"]
        ref_w = ref.split()
        ref_words.append(ref_w)
        greedy_words.append(rec["nbest"][0]["hyp"].split())

        texts = [h["hyp"] for h in rec["nbest"]]
        log_scores = [h["pll_score"] for h in rec["nbest"]]
        idx = mbr_select(texts, log_scores, args.tau)
        mbr_words.append(texts[idx].split())

    greedy_wer = corpus_wer(ref_words, greedy_words)
    mbr_wer = corpus_wer(ref_words, mbr_words)
    print(f"  Greedy WER: {greedy_wer*100:.4f}%")
    print(f"  MBR WER:    {mbr_wer*100:.4f}%")
    print(f"  delta:          {(mbr_wer-greedy_wer)*100:+.4f}pp")

    print(f"\nStep 6: CTC+PLL interpolation")
    interp_results = {}
    for alpha in [0.5, 0.6, 0.7, 0.8, 0.9]:
        interp_words = []
        for rec in all_records:
            nbest = rec["nbest"]
            scores = [alpha * h["score"] + (1 - alpha) * h["pll_score"] for h in nbest]
            idx = int(np.argmax(scores))
            interp_words.append(nbest[idx]["hyp"].split())
        wer = corpus_wer(ref_words, interp_words)
        interp_results[str(alpha)] = float(wer)
        print(f"  alpha={alpha}: WER={wer*100:.4f}%")

    print(f"\nStep 7: Paired bootstrap")
    from experiments.significance_tests import paired_bootstrap_wer

    boot_mbr = paired_bootstrap_wer(ref_words, mbr_words, greedy_words,
                                     n_bootstrap=10000, seed=42)

    result = {
        "n_utterances": len(all_records),
        "greedy_wer": float(greedy_wer),
        "mbr_wer": float(mbr_wer),
        "mbr_delta_pp": boot_mbr["delta"] * 100,
        "mbr_p_value": boot_mbr["p_value"],
        "mbr_ci_lower_pp": boot_mbr["ci_lower"] * 100,
        "mbr_ci_upper_pp": boot_mbr["ci_upper"] * 100,
        "interpolation": interp_results,
    }

    json_path = args.output_dir / "tl3_g16_full1155_results.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Wrote {json_path}")

    boot_path = args.output_dir / "tl3_g16_full1155_bootstrap.json"
    with open(boot_path, "w") as f:
        json.dump(boot_mbr, f, indent=2)
    print(f"  Wrote {boot_path}")

    elapsed = time.time() - t0
    print(f"\nDone. Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

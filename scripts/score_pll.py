#!/usr/bin/env python3
"""Score N-best hypotheses with RoBERTa pseudo-log-likelihood (PLL).

PLL(y) = sum_i log P(y_i | y_\i)  (Salazar et al., 2020)

For each non-special token position, mask it and compute the model's
log-prob of the true token. Sum across all positions.

Input:  N-best JSONL from generate_nbest.py
Output: Same JSONL with "pll_score" added to each hypothesis

Usage:
    python scripts/score_pll.py \
        --nbest /path/to/nbest.jsonl \
        --output /path/to/nbest_pll.jsonl \
        [--model roberta-base] \
        [--device cuda] \
        [--batch-size 32] \
        [--save-every 100]
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch


def _normalize_record(raw):
    """Normalize E11 or new format to canonical (new format)."""
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


def save_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


@torch.no_grad()
def compute_pll(text, tokenizer, model, device, batch_size=32):
    """Pseudo-log-likelihood: sum_i log P(token_i | tokens_{-i})."""
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"][0].to(device)
    L = input_ids.size(0)

    special = {tokenizer.bos_token_id, tokenizer.eos_token_id,
               tokenizer.pad_token_id, tokenizer.cls_token_id,
               tokenizer.sep_token_id}
    special.discard(None)
    mask_id = tokenizer.mask_token_id

    positions = [i for i in range(L) if input_ids[i].item() not in special]
    if not positions:
        return 0.0

    total = 0.0
    for s in range(0, len(positions), batch_size):
        batch_pos = positions[s:s + batch_size]
        bsz = len(batch_pos)
        masked = input_ids.unsqueeze(0).repeat(bsz, 1).clone()
        for k, p in enumerate(batch_pos):
            masked[k, p] = mask_id
        logits = model(masked).logits
        log_probs = torch.log_softmax(logits, dim=-1)
        for k, p in enumerate(batch_pos):
            total += log_probs[k, p, input_ids[p].item()].item()

    return total


def main():
    parser = argparse.ArgumentParser(
        description="Score N-best hypotheses with RoBERTa PLL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nbest", type=Path, required=True,
                        help="Input N-best JSONL")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSONL with pll_score added")
    parser.add_argument("--model", type=str, default="roberta-base",
                        help="HuggingFace model name or path")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for masked positions (not utterances)")
    parser.add_argument("--save-every", type=int, default=100,
                        help="Save intermediate results every N utterances")
    args = parser.parse_args()

    print("=" * 70)
    print("score_pll.py  --  RoBERTa pseudo-log-likelihood scoring")
    print("=" * 70)
    print(f"  nbest:      {args.nbest}")
    print(f"  output:     {args.output}")
    print(f"  model:      {args.model}")
    print(f"  device:     {args.device}")
    print(f"  batch_size: {args.batch_size}")
    print()

    device = torch.device(args.device)

    records = load_nbest(args.nbest)
    n_utts = len(records)
    n_hyps_total = sum(len(r["nbest"]) for r in records)
    print(f"Loaded {n_utts} utterances, {n_hyps_total} hypotheses")

    start_idx = 0
    if args.output.exists() and args.output.stat().st_size > 0:
        scored = load_nbest(args.output)
        if len(scored) < n_utts and all(
            "pll_score" in c for r in scored for c in r["nbest"]
        ):
            start_idx = len(scored)
            for i in range(start_idx):
                records[i] = scored[i]
            print(f"Resuming from utterance {start_idx}/{n_utts}")

    from transformers import RobertaTokenizer, RobertaForMaskedLM
    print(f"\nLoading {args.model}...")
    tokenizer = RobertaTokenizer.from_pretrained(args.model)
    model = RobertaForMaskedLM.from_pretrained(args.model).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params / 1e6:.1f}M parameters")

    if device.type == "cuda":
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"  GPU memory: {mem:.2f} GB")

    # Sanity check
    print("\n  PLL sanity check:")
    for rec in records[:3]:
        for c in rec["nbest"][:1]:
            pll = compute_pll(c["hyp"], tokenizer, model, device, args.batch_size)
            n_words = len(c["hyp"].split())
            print(f"    PLL={pll:8.2f}  ({n_words} words)  {c['hyp'][:60]}")

    # Score all hypotheses
    print(f"\nScoring {n_utts - start_idx} remaining utterances...")
    t0 = time.time()
    n_scored = 0
    n_truncated = 0

    for i in range(start_idx, n_utts):
        rec = records[i]
        for c in rec["nbest"]:
            text = c["hyp"]
            tokens = tokenizer.encode(text)
            if len(tokens) > 510:
                text = tokenizer.decode(tokens[:510], skip_special_tokens=True)
                n_truncated += 1
            c["pll_score"] = compute_pll(
                text, tokenizer, model, device, args.batch_size
            )
            n_scored += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = n_scored / elapsed
            eta = (n_hyps_total - n_scored) / rate if rate > 0 else 0
            print(f"  {i + 1}/{n_utts} utterances  "
                  f"({n_scored} hyps, {rate:.1f} hyps/s, "
                  f"ETA {eta / 60:.1f} min)")

        if args.save_every > 0 and (i + 1) % args.save_every == 0:
            save_jsonl(records[:i + 1], args.output)

    elapsed = time.time() - t0

    save_jsonl(records, args.output)

    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n  Peak GPU memory: {peak_mem:.2f} GB")

    print(f"\n  Scored {n_scored} hypotheses in {elapsed:.1f}s "
          f"({n_scored / max(elapsed, 1):.1f} hyps/s)")
    if n_truncated:
        print(f"  Truncated {n_truncated} hypotheses to 510 tokens")
    print(f"  Output: {args.output}")
    print(f"  Size: {args.output.stat().st_size / 1e6:.1f} MB")
    print()


if __name__ == "__main__":
    main()

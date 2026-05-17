#!/usr/bin/env python3
"""Score N-best JSONL with RoBERTa PLL and (optionally) GPT-2 LL.

Reads any of these formats:
  - E21 format:   top-level "hypotheses" key
  - E11/E20 fmt:  top-level "candidates" key

In-place adds "roberta_pll" (and "gpt2_ll" if --gpt2) to each hypothesis.
Writes a NEW JSONL with the scored fields appended.

Reuses the working PLL implementation from experiments/decoding/neural_lm_rescore.py.

Usage:
    python score_neural_lm.py \\
        --input-jsonl  results/tedlium3/test_G128.jsonl \\
        --output-jsonl results/tedlium3/test_G128_scored.jsonl \\
        --device cuda:0 \\
        --pll-batch 64 \\
        --gpt2

Expected runtime on Colab T4 for 1155 utts x 128 cands = 148K hyps:
  RoBERTa PLL: ~2-4 hours (mask-each-position is the bottleneck)
  GPT-2 LL:    ~10 min (single forward pass per text)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)



def get_cands_field(rec):
    """Return the key name where candidates live in this record."""
    if "hypotheses" in rec:
        return "hypotheses"
    if "candidates" in rec:
        return "candidates"
    raise ValueError(f"Record missing 'hypotheses' or 'candidates': {list(rec.keys())}")



@torch.no_grad()
def compute_pll(text, tokenizer, model, device, batch_size=64):
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
        e = min(s + batch_size, len(positions))
        batch_pos = positions[s:e]
        bsz = len(batch_pos)
        masked = input_ids.unsqueeze(0).repeat(bsz, 1).clone()
        for k, p in enumerate(batch_pos):
            masked[k, p] = mask_id
        logits = model(masked).logits  # (bsz, L, V)
        log_probs = torch.log_softmax(logits, dim=-1)
        for k, p in enumerate(batch_pos):
            total += log_probs[k, p, input_ids[p].item()].item()
    return total


def score_with_roberta(records, model_name, device, pll_batch):
    from transformers import RobertaTokenizer, RobertaForMaskedLM

    print(f"\nLoading {model_name}...")
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params/1e6:.1f}M params")

    # Sanity: PLL on 5 short hypotheses
    print("\n  PLL sanity check:")
    n_shown = 0
    for rec in records:
        ckey = get_cands_field(rec)
        for cand in rec[ckey]:
            if n_shown >= 5:
                break
            words = cand["text"].split()
            if 3 <= len(words) <= 8:
                pll = compute_pll(cand["text"], tokenizer, model, device, pll_batch)
                print(f"    PLL={pll:8.2f}  text={cand['text']!r}")
                n_shown += 1
        if n_shown >= 5:
            break

    print(f"\n  Scoring all candidates (this is the bottleneck)...")
    t0 = time.time()
    n_hyps = 0
    n_recs = len(records)
    for i, rec in enumerate(records):
        ckey = get_cands_field(rec)
        for cand in rec[ckey]:
            cand["roberta_pll"] = compute_pll(
                cand["text"], tokenizer, model, device, pll_batch
            )
            n_hyps += 1
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = n_hyps / elapsed
            eta_h = (n_recs - i - 1) * (n_hyps / (i + 1)) / rate / 3600
            print(f"  {i+1}/{n_recs} utts, {n_hyps} hyps, "
                  f"{rate:.1f} hyps/s, ETA {eta_h:.2f}h")
    elapsed = time.time() - t0
    print(f"   {n_hyps} hyps in {elapsed/60:.1f} min "
          f"({n_hyps/elapsed:.1f} hyps/s)")
    del model
    torch.cuda.empty_cache()



@torch.no_grad()
def compute_gpt2_ll_batch(texts, tokenizer, model, device, batch_size=16):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True)
        ids = enc["input_ids"].to(device)
        am = enc["attention_mask"].to(device)
        logits = model(ids, attention_mask=am).logits  # (B, L, V)
        log_probs = torch.log_softmax(logits, dim=-1)
        shift_lp = log_probs[:, :-1, :]
        shift_lab = ids[:, 1:]
        shift_mask = am[:, 1:].float()
        gathered = shift_lp.gather(2, shift_lab.unsqueeze(-1)).squeeze(-1)
        gathered = gathered * shift_mask
        results.extend(gathered.sum(dim=1).cpu().tolist())
    return results


def score_with_gpt2(records, model_name, device, batch_size):
    from transformers import GPT2Tokenizer, GPT2LMHeadModel

    print(f"\nLoading {model_name}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params/1e6:.1f}M params")

    flat_texts = []
    flat_ref = []  # (rec_idx, cand_idx, ckey)
    for ri, rec in enumerate(records):
        ckey = get_cands_field(rec)
        for ci, cand in enumerate(rec[ckey]):
            flat_texts.append(cand["text"])
            flat_ref.append((ri, ci, ckey))

    print(f"  Scoring {len(flat_texts)} hypotheses...")
    t0 = time.time()
    lls = compute_gpt2_ll_batch(flat_texts, tokenizer, model, device, batch_size)
    elapsed = time.time() - t0
    for (ri, ci, ckey), ll in zip(flat_ref, lls):
        records[ri][ckey][ci]["gpt2_ll"] = ll
    print(f"   {len(flat_texts)} hyps in {elapsed/60:.1f} min "
          f"({len(flat_texts)/elapsed:.1f} hyps/s)")
    del model
    torch.cuda.empty_cache()



def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {path}")
    return records


def save_jsonl(records, path):
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"   Wrote {len(records)} records to {path} "
          f"({path.stat().st_size / 1e6:.1f} MB)")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--roberta-model", default="roberta-base")
    parser.add_argument("--gpt2-model", default="gpt2")
    parser.add_argument("--pll-batch", type=int, default=64)
    parser.add_argument("--gpt2", action="store_true",
                        help="Also compute GPT-2 LL")
    parser.add_argument("--gpt2-batch", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0,
                        help="Score only first N utterances (smoke testing)")
    parser.add_argument("--skip-pll", action="store_true",
                        help="Skip RoBERTa PLL (just GPT-2)")
    args = parser.parse_args()

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Neural LM scoring (PLL + GPT-2 LL)")
    print("=" * 60)
    print(f"  Input:  {args.input_jsonl}")
    print(f"  Output: {args.output_jsonl}")
    print(f"  Device: {device}")
    print(f"  RoBERTa: {args.roberta_model}  (skip={args.skip_pll})")
    print(f"  GPT-2:   {args.gpt2_model}     (enabled={args.gpt2})")

    records = load_jsonl(args.input_jsonl)
    if args.limit > 0:
        records = records[:args.limit]
        print(f"   LIMIT: scoring only first {len(records)} utts")

    n_hyps_total = sum(
        len(rec[get_cands_field(rec)]) for rec in records
    )
    print(f"  Total hypotheses to score: {n_hyps_total}")

    t_total = time.time()
    if not args.skip_pll:
        score_with_roberta(records, args.roberta_model, device, args.pll_batch)
    if args.gpt2:
        score_with_gpt2(records, args.gpt2_model, device, args.gpt2_batch)
    print(f"\nTotal scoring time: {(time.time()-t_total)/60:.1f} min")

    print(f"\nWriting output...")
    save_jsonl(records, args.output_jsonl)


if __name__ == "__main__":
    main()

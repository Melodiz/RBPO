#!/usr/bin/env python3
"""E22: Train a DistilBERT discriminative reranker with MWER loss.

Trained on E21's train-clean-100_G16.jsonl. Each utterance has 16 N-best
hypotheses with per-hypothesis WER labels. The model learns to score
hypothesis text such that lower-WER candidates get higher scores.

The MWER objective is the project's RL contribution at decode time:
- Action: hypothesis selection from N-best
- Reward: -WER (per candidate)
- Policy: softmax(s_theta(y) / tau) over the N-best
- Loss: expected WER under that policy (MWER, with built-in baseline)

Usage:
    python train_reranker.py \\
        --data-jsonl /content/drive/MyDrive/rbpo_results/reranker_training_data/train-clean-100_G16.jsonl \\
        --output-dir /content/drive/MyDrive/rbpo_results/reranker \\
        --epochs 5 --batch-size 2 --tau 1.0
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


class RerankerDataset(Dataset):
    """One item per utterance with all its N-best candidates."""

    def __init__(self, jsonl_path):
        self.records = []
        with open(jsonl_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("hypotheses") and len(rec["hypotheses"]) >= 2:
                    self.records.append(rec)
        print(f"Loaded {len(self.records)} utterances from {jsonl_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        texts = [h["text"] for h in rec["hypotheses"]]
        wers = [h["wer_edits"] / max(1, h["wer_ref_len"]) for h in rec["hypotheses"]]
        return {
            "utterance_id": rec["utterance_id"],
            "reference": rec["reference"],
            "texts": texts,
            "wers": wers,
        }


def make_collate_fn(tokenizer, max_len):
    def collate_fn(batch):
        flat_texts = []
        group_sizes = []
        wers_per_utt = []
        for item in batch:
            flat_texts.extend(item["texts"])
            group_sizes.append(len(item["texts"]))
            wers_per_utt.append(torch.tensor(item["wers"], dtype=torch.float32))
        enc = tokenizer(
            flat_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "group_sizes": group_sizes,
            "wers_per_utt": wers_per_utt,
        }
    return collate_fn


class DistilBertScorer(nn.Module):
    """[CLS] embedding -> linear scalar score."""

    def __init__(self, model_name="distilbert-base-uncased"):
        super().__init__()
        from transformers import DistilBertModel
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.score_head = nn.Linear(self.bert.config.hidden_size, 1)
        # Init head with small std so initial scores are near 0
        nn.init.normal_(self.score_head.weight, std=0.02)
        nn.init.zeros_(self.score_head.bias)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]            # (B, H)
        return self.score_head(cls).squeeze(-1)       # (B,)


def mwer_loss(scores_flat, group_sizes, wers_per_utt, tau):
    """MWER loss: expected WER under softmax(scores/tau), averaged over utterances."""
    losses = []
    offset = 0
    for n_i, wers_i in zip(group_sizes, wers_per_utt):
        s_i = scores_flat[offset:offset + n_i]
        p_i = F.softmax(s_i / tau, dim=0)
        losses.append((p_i * wers_i).sum())
        offset += n_i
    return torch.stack(losses).mean()


@torch.no_grad()
def eval_argmax_wer(model, val_loader, device):
    """Compute WER if we picked argmax-by-reranker hypothesis per utterance.

    Returns: (corpus_wer, mean_per_utt_wer, oracle_wer, greedy_wer)
    """
    model.eval()
    total_edits_pred = 0
    total_edits_oracle = 0
    total_edits_greedy = 0  # candidate at index 0 (greedy injected)
    total_ref_words = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        scores = model(input_ids=input_ids, attention_mask=attention_mask)
        scores_cpu = scores.float().cpu()

        offset = 0
        for n_i, wers_i in zip(batch["group_sizes"], batch["wers_per_utt"]):
            s_i = scores_cpu[offset:offset + n_i]
            picked = int(s_i.argmax().item())
            ref_len_dummy = 1  # we have only WER fractions; convert via assumption

            # We need actual edits and ref_len, but only have WERs. Approximate
            # via the dataset's per-record fields. For val, we recompute below.
            offset += n_i

    # Above approximation is rough  --  switch to a record-based eval that
    # actually uses wer_edits / wer_ref_len.
    # See val_argmax_corpus_wer below.
    raise NotImplementedError("Use val_argmax_corpus_wer instead")


@torch.no_grad()
def val_argmax_corpus_wer(model, val_dataset, tokenizer, device, max_len, batch_size):
    """Compute corpus WER on val set using argmax-by-reranker selection.

    Uses raw wer_edits and wer_ref_len from the dataset records (not the
    pre-aggregated WER fractions) so the result is corpus-level, not
    macro-mean.
    """
    model.eval()
    total_edits_pred = 0
    total_edits_oracle = 0
    total_edits_greedy = 0
    total_ref_words = 0

    n = len(val_dataset)
    for start in range(0, n, batch_size):
        batch_records = [val_dataset[i] for i in range(start, min(start + batch_size, n))]
        flat_texts = []
        for item in batch_records:
            flat_texts.extend(item["texts"])
        enc = tokenizer(
            flat_texts, padding=True, truncation=True,
            max_length=max_len, return_tensors="pt"
        )
        scores = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        ).float().cpu()

        offset = 0
        for item in batch_records:
            n_i = len(item["texts"])
            s_i = scores[offset:offset + n_i]
            picked = int(s_i.argmax().item())
            offset += n_i

            # Find the record by utterance_id to get raw edits + ref_len
            # (avoid duplicating: each item has wers but not raw edits)
            rec_idx = val_dataset.indices[start + (offset - n_i) // n_i] \
                if hasattr(val_dataset, 'indices') else None
            # Direct lookup: walk back through the dataset
            full = (val_dataset.dataset.records[val_dataset.indices[start + len(flat_texts) // 1]]
                    if hasattr(val_dataset, 'indices') else None)
            # Simpler: iterate fresh  --  see rewrite below
        # rewrite using direct record access:
    return _val_corpus_wer_impl(model, val_dataset, tokenizer, device, max_len, batch_size)


@torch.no_grad()
def _val_corpus_wer_impl(model, val_subset, tokenizer, device, max_len, batch_size):
    """Direct implementation: enumerate val records, score, pick argmax."""
    model.eval()
    total_edits_pred = 0
    total_edits_oracle = 0
    total_edits_greedy = 0
    total_ref_words = 0

    if hasattr(val_subset, 'indices'):
        records = [val_subset.dataset.records[i] for i in val_subset.indices]
    else:
        records = val_subset.records

    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        flat_texts = []
        for rec in chunk:
            flat_texts.extend([h["text"] for h in rec["hypotheses"]])
        if not flat_texts:
            continue
        enc = tokenizer(
            flat_texts, padding=True, truncation=True,
            max_length=max_len, return_tensors="pt"
        )
        scores = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        ).float().cpu().tolist()

        offset = 0
        for rec in chunk:
            n_i = len(rec["hypotheses"])
            s_i = scores[offset:offset + n_i]
            picked_idx = max(range(n_i), key=lambda j: s_i[j])
            picked = rec["hypotheses"][picked_idx]
            oracle = min(rec["hypotheses"], key=lambda h: h["wer_edits"])
            greedy = rec["hypotheses"][0]  # greedy is injected at position 0
            total_edits_pred += picked["wer_edits"]
            total_edits_oracle += oracle["wer_edits"]
            total_edits_greedy += greedy["wer_edits"]
            total_ref_words += picked["wer_ref_len"]
            offset += n_i

    return {
        "wer_pred":   total_edits_pred / max(1, total_ref_words),
        "wer_oracle": total_edits_oracle / max(1, total_ref_words),
        "wer_greedy": total_edits_greedy / max(1, total_ref_words),
        "n_utts":     len(records),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Utterances per step (each has up to 16 candidates)")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="If >0, stop after this many optimizer steps "
                             "(for fast end-to-end smoke testing). "
                             "Forces an early eval + checkpoint save.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("E22: DistilBERT MWER Reranker Training")
    print("=" * 60)
    print(f"  Data:       {args.data_jsonl}")
    print(f"  Output:     {args.output_dir}")
    print(f"  Model:      {args.model_name}")
    print(f"  Device:     {device}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch:      {args.batch_size} utts x ~16 cands = ~{args.batch_size*16} fwd")
    print(f"  Grad accum: {args.grad_accum}")
    print(f"  LR:         {args.lr}  (warmup {args.warmup_steps} steps)")
    print(f"  tau_mwer:     {args.tau}")
    print(f"  FP16:       {args.fp16}")
    print()

    # Tokenizer
    from transformers import DistilBertTokenizerFast, get_linear_schedule_with_warmup
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name)

    # Data
    full = RerankerDataset(args.data_jsonl)
    n_total = len(full)
    n_val = int(n_total * args.val_fraction)
    n_train = n_total - n_val
    train_subset, val_subset = random_split(
        full, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"  Train: {n_train} utts  Val: {n_val} utts")

    collate_fn = make_collate_fn(tokenizer, args.max_len)
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model + optimizer
    model = DistilBertScorer(args.model_name).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params/1e6:.1f}M")

    no_decay = ["bias", "LayerNorm.weight"]
    optim_groups = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr)

    n_steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = n_steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    print(f"  Steps/epoch: {n_steps_per_epoch}  Total: {total_steps}")
    print()

    # Initial val
    print("=== Initial val (untrained head) ===")
    init_val = _val_corpus_wer_impl(
        model, val_subset, tokenizer, device, args.max_len, args.batch_size,
    )
    print(f"  wer_pred={init_val['wer_pred']*100:.2f}%  "
          f"wer_oracle={init_val['wer_oracle']*100:.2f}%  "
          f"wer_greedy={init_val['wer_greedy']*100:.2f}%  "
          f"({init_val['n_utts']} val utts)")

    # Training loop
    best_val_wer = init_val["wer_pred"]
    best_path = args.output_dir / "best_distilbert_mwer.pt"
    history = [{
        "epoch": 0, "step": 0, "phase": "init",
        "loss": None, **init_val,
    }]

    eval_steps = max(1, n_steps_per_epoch // 2)  # eval twice per epoch

    global_step = 0
    optimizer.zero_grad()
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_n = 0

        for step_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            wers_per_utt_dev = [w.to(device) for w in batch["wers_per_utt"]]

            with torch.cuda.amp.autocast(enabled=args.fp16):
                scores_flat = model(input_ids, attention_mask)
                loss = mwer_loss(
                    scores_flat,
                    batch["group_sizes"],
                    wers_per_utt_dev,
                    args.tau,
                )
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()
            running_loss += loss.item() * args.grad_accum * len(batch["group_sizes"])
            running_n += len(batch["group_sizes"])

            if (step_idx + 1) % args.grad_accum == 0:
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip
                    )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Smoke-test mode: stop after max_steps + force a checkpoint
                if args.max_steps > 0 and global_step >= args.max_steps:
                    print(f"\n  --max-steps={args.max_steps} reached. "
                          f"Running val + saving checkpoint...")
                    val_metrics = _val_corpus_wer_impl(
                        model, val_subset, tokenizer, device,
                        args.max_len, args.batch_size,
                    )
                    print(f"  smoke val: wer_pred={val_metrics['wer_pred']*100:.3f}%  "
                          f"oracle={val_metrics['wer_oracle']*100:.3f}%")
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "args": vars(args),
                        "step": global_step,
                        "epoch": epoch,
                        "val_wer": val_metrics["wer_pred"],
                        "smoke": True,
                    }, best_path)
                    history.append({
                        "epoch": epoch, "step": global_step,
                        "phase": "smoke_final",
                        "loss": running_loss / max(1, running_n),
                        **val_metrics,
                    })
                    with open(args.output_dir / "training_history.json", "w") as f:
                        json.dump(history, f, indent=2)
                    elapsed_total = time.time() - t0
                    print(f"   Smoke checkpoint saved to {best_path}")
                    print(f"  Smoke training time: {elapsed_total/60:.1f} min")
                    return

                # Progress
                if global_step % 100 == 0:
                    avg_loss = running_loss / max(1, running_n)
                    elapsed = time.time() - t0
                    rate = global_step / elapsed
                    eta_min = (total_steps - global_step) / rate / 60
                    print(f"  ep{epoch} step {global_step}/{total_steps}  "
                          f"loss={avg_loss*100:.3f}pp  "
                          f"({rate:.1f} step/s, ETA {eta_min:.0f}min)")

                # Periodic validation
                if global_step % eval_steps == 0:
                    val_metrics = _val_corpus_wer_impl(
                        model, val_subset, tokenizer, device,
                        args.max_len, args.batch_size,
                    )
                    avg_loss = running_loss / max(1, running_n)
                    print(f"  ep{epoch} step {global_step}  VAL  "
                          f"loss={avg_loss*100:.3f}pp  "
                          f"wer_pred={val_metrics['wer_pred']*100:.3f}%  "
                          f"wer_oracle={val_metrics['wer_oracle']*100:.3f}%")
                    history.append({
                        "epoch": epoch,
                        "step": global_step,
                        "phase": "val",
                        "loss": avg_loss,
                        **val_metrics,
                    })
                    if val_metrics["wer_pred"] < best_val_wer:
                        best_val_wer = val_metrics["wer_pred"]
                        torch.save({
                            "model_state_dict": model.state_dict(),
                            "args": vars(args),
                            "step": global_step,
                            "epoch": epoch,
                            "val_wer": best_val_wer,
                        }, best_path)
                        print(f"   New best  --  saved to {best_path}")
                    model.train()
                    running_loss = 0.0
                    running_n = 0

    # End-of-training val
    final_val = _val_corpus_wer_impl(
        model, val_subset, tokenizer, device, args.max_len, args.batch_size,
    )
    history.append({
        "epoch": args.epochs, "step": global_step, "phase": "final",
        "loss": None, **final_val,
    })
    if final_val["wer_pred"] < best_val_wer:
        best_val_wer = final_val["wer_pred"]
        torch.save({
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "step": global_step,
            "epoch": args.epochs,
            "val_wer": best_val_wer,
        }, best_path)

    elapsed_total = time.time() - t0
    print()
    print("=" * 60)
    print(f"DONE in {elapsed_total/60:.1f} min")
    print(f"  Best val WER: {best_val_wer*100:.3f}%  (vs greedy {final_val['wer_greedy']*100:.3f}%)")
    print(f"  Checkpoint:   {best_path}")
    print("=" * 60)

    with open(args.output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History: {args.output_dir / 'training_history.json'}")


if __name__ == "__main__":
    main()

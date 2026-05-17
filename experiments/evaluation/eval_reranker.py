#!/usr/bin/env python3
"""E22 evaluation: score dev-other N-best with the trained reranker,
combine with CTC and PLL, and report WER under multiple selection
strategies (argmax-only, interpolation, MBR).

Reads:
  - Trained checkpoint:  best_distilbert_mwer.pt (from train_reranker.py)
  - Dev-other N-best:    results/g_scaling/neural_lm_scores_G128.jsonl
                         (has text, ctc_log_prob, roberta_pll, wer per cand)

Writes:
  - eval_results.csv: every config's WER, p-value vs E19 baseline (5.46%)
  - eval_results.json: full structured results
  - per_quartile.csv: per-length-quartile breakdown (E19 difficulty axis)
"""

import argparse
import csv
import editdistance
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the model class from training
sys.path.insert(0, str(Path(__file__).parent))
from train_reranker import DistilBertScorer  # noqa: E402

# Add experiments dir for significance tests
from significance_tests import paired_bootstrap_wer  # noqa: E402

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)



def load_dev_other(jsonl_path):
    """Load dev-other JSONL with CTC + PLL + WER per candidate."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("candidates"):
                records.append(rec)
    print(f"Loaded {len(records)} utterances from {jsonl_path}")
    return records



@torch.no_grad()
def score_with_reranker(records, model, tokenizer, device, max_len=128, batch_size=2):
    """Add 'reranker_score' to each candidate. In-place modifies records."""
    model.eval()

    n = len(records)
    t0 = time.time()
    for start in range(0, n, batch_size):
        chunk = records[start:start + batch_size]
        flat_texts = []
        for rec in chunk:
            flat_texts.extend([c["text"] for c in rec["candidates"]])
        if not flat_texts:
            continue
        enc = tokenizer(
            flat_texts, padding=True, truncation=True,
            max_length=max_len, return_tensors="pt",
        )
        scores = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        ).float().cpu().tolist()

        offset = 0
        for rec in chunk:
            n_i = len(rec["candidates"])
            for j in range(n_i):
                rec["candidates"][j]["reranker_score"] = scores[offset + j]
            offset += n_i

        if (start + batch_size) % 200 < batch_size:
            elapsed = time.time() - t0
            print(f"  scored {min(start + batch_size, n)}/{n}  ({(start+batch_size)/elapsed:.1f} utt/s)")

    print(f"   Scored {n} utterances in {(time.time()-t0)/60:.1f} min")



def select_argmax(records, score_fn):
    """Pick candidate with max score per utterance. Returns list of hyp_words."""
    hyps = []
    refs = []
    for rec in records:
        cands = rec["candidates"]
        scores = [score_fn(c) for c in cands]
        picked = cands[int(np.argmax(scores))]
        hyps.append(picked["text"].split())
        refs.append(rec["ref_text"].split())
    return refs, hyps


def select_argmax_oracle(records):
    """Oracle: pick candidate with smallest WER per utterance."""
    hyps = []
    refs = []
    for rec in records:
        cands = rec["candidates"]
        ref_words = rec["ref_text"].split()
        # Use edit distance (most reliable; "wer" field may be float per cand)
        edits = [editdistance.eval(c["text"].split(), ref_words) for c in cands]
        picked = cands[int(np.argmin(edits))]
        hyps.append(picked["text"].split())
        refs.append(ref_words)
    return refs, hyps


def select_argmax_greedy(records):
    return select_argmax(records, lambda c: c["ctc_log_prob"])



def cer_distance(a, b):
    return editdistance.eval(list(a), list(b)) / max(len(a), len(b), 1)


def mbr_select(rec, posterior_log_scores, tau=10.0):
    """MBR-CER selection with given posterior log-scores.

    risk[i] = sum_j p(j) * CER(y_i, y_j)
    Returns the index of the min-risk candidate.
    """
    cands = rec["candidates"]
    n = len(cands)
    if n == 1:
        return 0

    log_p = np.asarray(posterior_log_scores, dtype=np.float64) / tau
    log_p = log_p - log_p.max()
    p = np.exp(log_p)
    p = p / p.sum()

    # Pairwise CER matrix
    texts = [c["text"] for c in cands]
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = cer_distance(texts[i], texts[j])
            D[i, j] = d
            D[j, i] = d

    risk = D @ p
    return int(np.argmin(risk))


def select_mbr(records, score_fn, tau=10.0):
    hyps, refs = [], []
    for rec in records:
        cands = rec["candidates"]
        scores = [score_fn(c) for c in cands]
        picked_idx = mbr_select(rec, scores, tau=tau)
        hyps.append(cands[picked_idx]["text"].split())
        refs.append(rec["ref_text"].split())
    return refs, hyps



def corpus_wer(refs, hyps):
    total_edits = sum(editdistance.eval(h, r) for h, r in zip(hyps, refs))
    total_ref = sum(len(r) for r in refs)
    return total_edits / max(1, total_ref)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Trained reranker .pt from train_reranker.py")
    parser.add_argument("--nbest-jsonl", type=Path, required=True,
                        help="Dev-other N-best (g_scaling/neural_lm_scores_G128.jsonl)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--baseline-wer", type=float, default=0.0546,
                        help="E19 best for p-value comparison")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("E22 Eval: DistilBERT MWER reranker on dev-other G=128")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  N-best:     {args.nbest_jsonl}")
    print(f"  Device:     {device}")

    records = load_dev_other(args.nbest_jsonl)

    print("Loading reranker...")
    from transformers import DistilBertTokenizerFast
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name)
    model = DistilBertScorer(args.model_name).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded checkpoint from epoch {ckpt.get('epoch')}, "
          f"val_wer={ckpt.get('val_wer', 0)*100:.3f}%")

    print("\n=== Scoring all candidates with reranker ===")
    score_with_reranker(
        records, model, tokenizer, device,
        max_len=args.max_len, batch_size=args.batch_size,
    )

    # Free GPU memory (we don't need the model anymore)
    del model
    torch.cuda.empty_cache()

    print("\n=== Baselines ===")
    results = []  # list of dicts for CSV

    refs_greedy = [rec["ref_text"].split() for rec in records]
    hyps_greedy = [rec["candidates"][0]["text"].split() for rec in records]
    wer_greedy = corpus_wer(refs_greedy, hyps_greedy)
    print(f"  Greedy (top-1 CTC):  {wer_greedy*100:.3f}%")
    results.append({"method": "greedy_top1", "wer": wer_greedy, "delta": 0.0,
                    "p_value": None, "ci_lo": None, "ci_hi": None})

    refs_oracle, hyps_oracle = select_argmax_oracle(records)
    wer_oracle = corpus_wer(refs_oracle, hyps_oracle)
    print(f"  Oracle:              {wer_oracle*100:.3f}%")
    results.append({"method": "oracle", "wer": wer_oracle, "delta": None,
                    "p_value": None, "ci_lo": None, "ci_hi": None})

    print("\n=== Argmax: reranker only ===")
    refs_r, hyps_r = select_argmax(records, lambda c: c["reranker_score"])
    wer_r = corpus_wer(refs_r, hyps_r)
    print(f"  Reranker-only:       {wer_r*100:.3f}%")

    print("\n=== Argmax: alpha*log_CTC + (1-alpha)*reranker  (interp sweep) ===")
    best_ctc_r = None
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]:
        score_fn = lambda c, a=alpha: a * c["ctc_log_prob"] + (1 - a) * c["reranker_score"]
        refs_a, hyps_a = select_argmax(records, score_fn)
        wer_a = corpus_wer(refs_a, hyps_a)
        print(f"  alpha={alpha:.1f}: {wer_a*100:.3f}%")
        if best_ctc_r is None or wer_a < best_ctc_r["wer"]:
            best_ctc_r = {"alpha": alpha, "wer": wer_a, "hyps": hyps_a, "refs": refs_a}
    print(f"  -> best alpha={best_ctc_r['alpha']}: {best_ctc_r['wer']*100:.3f}%")

    print("\n=== Argmax: alpha*log_CTC + beta*reranker + gamma*log_PLL  (3-way) ===")
    best_3way = None
    for alpha in [0.0, 0.2, 0.4, 0.6]:
        for beta in [0.2, 0.4, 0.6, 0.8]:
            gamma = 1.0 - alpha - beta
            if gamma < 0 or gamma > 1.0:
                continue
            score_fn = lambda c, a=alpha, b=beta, g=gamma: (
                a * c["ctc_log_prob"] + b * c["reranker_score"] + g * c["roberta_pll"]
            )
            refs_3, hyps_3 = select_argmax(records, score_fn)
            wer_3 = corpus_wer(refs_3, hyps_3)
            if best_3way is None or wer_3 < best_3way["wer"]:
                best_3way = {
                    "alpha": alpha, "beta": beta, "gamma": gamma,
                    "wer": wer_3, "hyps": hyps_3, "refs": refs_3,
                }
    print(f"  -> best (alpha={best_3way['alpha']}, beta={best_3way['beta']}, "
          f"gamma={best_3way['gamma']:.1f}): {best_3way['wer']*100:.3f}%")

    print("\n=== MBR-CER with reranker as posterior (tau sweep) ===")
    best_mbr_r = None
    for tau in [1.0, 5.0, 10.0, 20.0, 50.0]:
        refs_m, hyps_m = select_mbr(records, lambda c: c["reranker_score"], tau=tau)
        wer_m = corpus_wer(refs_m, hyps_m)
        print(f"  tau={tau}: {wer_m*100:.3f}%")
        if best_mbr_r is None or wer_m < best_mbr_r["wer"]:
            best_mbr_r = {"tau": tau, "wer": wer_m, "hyps": hyps_m, "refs": refs_m}
    print(f"  -> best tau={best_mbr_r['tau']}: {best_mbr_r['wer']*100:.3f}%")

    print("\n=== MBR-CER with (beta*reranker + (1-beta)*PLL) posterior, tau=10 ===")
    best_mbr_pll = None
    for beta in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
        score_fn = lambda c, b=beta: b * c["reranker_score"] + (1 - b) * c["roberta_pll"]
        refs_p, hyps_p = select_mbr(records, score_fn, tau=10.0)
        wer_p = corpus_wer(refs_p, hyps_p)
        print(f"  beta={beta:.1f}: {wer_p*100:.3f}%")
        if best_mbr_pll is None or wer_p < best_mbr_pll["wer"]:
            best_mbr_pll = {"beta": beta, "wer": wer_p, "hyps": hyps_p, "refs": refs_p}
    print(f"  -> best beta={best_mbr_pll['beta']}: {best_mbr_pll['wer']*100:.3f}%")

    print(f"\n=== Bootstrap significance vs E19 baseline ({args.baseline_wer*100:.2f}%) ===")
    # We don't have the E19 baseline hyps directly. Use the best PLL-only baseline
    # we can reconstruct: argmax interp(alpha=0.7, CTC vs PLL). That's E19's argmax-PLL
    # baseline at 5.89%. For the *MBR* baseline (5.46%), we'd need the actual hyps.
    # Use this as approximation: build E19-like hyps as MBR-CER + PLL tau=10.
    print("  Reconstructing baseline: MBR-CER + PLL tau=10 ...")
    refs_b, hyps_b = select_mbr(records, lambda c: c["roberta_pll"], tau=10.0)
    wer_b = corpus_wer(refs_b, hyps_b)
    print(f"  Reconstructed baseline WER: {wer_b*100:.3f}% "
          f"(target ~{args.baseline_wer*100:.2f}%)")

    def boot(refs, hyps_new, label):
        result = paired_bootstrap_wer(refs, hyps_new, hyps_b, n_bootstrap=10000)
        return {
            "delta_pp": result["delta"] * 100,
            "p_value": result["p_value"],
            "ci_lo_pp": result["ci_lower"] * 100,
            "ci_hi_pp": result["ci_upper"] * 100,
        }

    candidates_to_test = [
        ("reranker_only", wer_r, hyps_r),
        ("argmax_ctc+rerank", best_ctc_r["wer"], best_ctc_r["hyps"]),
        ("argmax_ctc+rerank+pll", best_3way["wer"], best_3way["hyps"]),
        ("mbr_reranker_post", best_mbr_r["wer"], best_mbr_r["hyps"]),
        ("mbr_rerank+pll_post", best_mbr_pll["wer"], best_mbr_pll["hyps"]),
    ]
    for name, wer_v, hyps_v in candidates_to_test:
        bres = boot(refs_b, hyps_v, name)
        print(f"  {name:30s}  delta={bres['delta_pp']:+.3f}pp  "
              f"p={bres['p_value']:.4f}  "
              f"CI=[{bres['ci_lo_pp']:+.2f}, {bres['ci_hi_pp']:+.2f}]pp")
        results.append({
            "method": name,
            "wer": wer_v,
            "delta": bres["delta_pp"] / 100,
            "p_value": bres["p_value"],
            "ci_lo": bres["ci_lo_pp"] / 100,
            "ci_hi": bres["ci_hi_pp"] / 100,
        })

    results.append({
        "method": "baseline_mbr_cer_pll_tau10",
        "wer": wer_b, "delta": 0.0, "p_value": None,
        "ci_lo": None, "ci_hi": None,
    })

    print("\n=== Per-quartile (by reference length) ===")
    refs_words_all = [rec["ref_text"].split() for rec in records]
    ref_lens = [len(r) for r in refs_words_all]
    quartiles = np.percentile(ref_lens, [25, 50, 75])
    print(f"  Quartiles (words): Q1<={quartiles[0]:.0f}, Q2<={quartiles[1]:.0f}, Q3<={quartiles[2]:.0f}")

    quartile_rows = []
    best_method_hyps = best_ctc_r["hyps"]  # use best interp config
    for q_label, lo, hi in [
        ("Q1", 0, quartiles[0]),
        ("Q2", quartiles[0], quartiles[1]),
        ("Q3", quartiles[1], quartiles[2]),
        ("Q4", quartiles[2], float("inf")),
    ]:
        idx = [i for i, L in enumerate(ref_lens) if lo < L <= hi or (q_label == "Q1" and L <= hi)]
        if not idx:
            continue
        refs_q = [refs_words_all[i] for i in idx]
        hyps_b_q = [hyps_b[i] for i in idx]
        hyps_r_q = [best_method_hyps[i] for i in idx]

        wer_b_q = corpus_wer(refs_q, hyps_b_q)
        wer_r_q = corpus_wer(refs_q, hyps_r_q)
        wer_oracle_q = corpus_wer(refs_q, [hyps_oracle[i] for i in idx])

        print(f"  {q_label} (n={len(idx)}):  baseline={wer_b_q*100:.2f}%  "
              f"reranker_interp={wer_r_q*100:.2f}%  oracle={wer_oracle_q*100:.2f}%")
        quartile_rows.append({
            "quartile": q_label,
            "n": len(idx),
            "len_lo": lo,
            "len_hi": hi if hi != float("inf") else max(ref_lens),
            "wer_baseline": wer_b_q,
            "wer_reranker_best": wer_r_q,
            "wer_oracle": wer_oracle_q,
        })

    csv_path = args.output_dir / "eval_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "wer", "delta", "p_value", "ci_lo", "ci_hi"])
        w.writeheader()
        for row in results:
            w.writerow(row)
    print(f"\n  Saved: {csv_path}")

    quartile_csv = args.output_dir / "per_quartile.csv"
    with open(quartile_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "quartile", "n", "len_lo", "len_hi",
            "wer_baseline", "wer_reranker_best", "wer_oracle",
        ])
        w.writeheader()
        for row in quartile_rows:
            w.writerow(row)
    print(f"  Saved: {quartile_csv}")

    json_path = args.output_dir / "eval_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "baseline_wer_target": args.baseline_wer,
            "baseline_wer_reconstructed": wer_b,
            "wer_greedy": wer_greedy,
            "wer_oracle": wer_oracle,
            "wer_reranker_only": wer_r,
            "best_ctc_reranker": {k: best_ctc_r[k] for k in ("alpha", "wer")},
            "best_3way": {k: best_3way[k] for k in ("alpha", "beta", "gamma", "wer")},
            "best_mbr_reranker": {k: best_mbr_r[k] for k in ("tau", "wer")},
            "best_mbr_rerank_pll": {k: best_mbr_pll[k] for k in ("beta", "wer")},
            "results": results,
            "quartiles": quartile_rows,
        }, f, indent=2)
    print(f"  Saved: {json_path}")

    print("\n" + "=" * 60)
    print("HEADLINE")
    print("=" * 60)
    print(f"  Greedy (top-1 CTC):       {wer_greedy*100:.3f}%")
    print(f"  Baseline (MBR+PLL tau=10):  {wer_b*100:.3f}%")
    print(f"  Reranker-only argmax:     {wer_r*100:.3f}%")
    print(f"  Best CTC+reranker:        {best_ctc_r['wer']*100:.3f}%  "
          f"(alpha={best_ctc_r['alpha']})")
    print(f"  Best CTC+rerank+PLL:      {best_3way['wer']*100:.3f}%  "
          f"(alpha={best_3way['alpha']}, beta={best_3way['beta']})")
    print(f"  Best MBR with reranker:   {best_mbr_r['wer']*100:.3f}%  "
          f"(tau={best_mbr_r['tau']})")
    print(f"  Best MBR with rerank+PLL: {best_mbr_pll['wer']*100:.3f}%  "
          f"(beta={best_mbr_pll['beta']})")
    print(f"  Oracle:                   {wer_oracle*100:.3f}%")


if __name__ == "__main__":
    main()

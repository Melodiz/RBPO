#!/usr/bin/env python3
"""Level 5: Neural LM rescoring of CTC N-best (RoBERTa PLL + GPT-2 LL).

Tests whether external linguistic information closes the oracle gap that
CTC-internal methods cannot. R7 baseline: 15-25% relative WER reduction
expected on CTC ASR with neural LM rescoring.

Computes:
- RoBERTa-base pseudo-log-likelihood (PLL): sum_i log P(y_i | y_{-i})
  where y_{-i} is y with position i replaced by <mask>.
- GPT-2 (124M) autoregressive log-likelihood: sum_i log P(y_i | y_{<i}).
- Interpolated rescoring: s = a*log_ctc + (1-a)*log_lm.
- Per-utterance Spearman rho between each scorer and WER.
- MBR-CER with PLL-derived posterior weights (vs CTC-derived).

Usage (Colab):
    pip install transformers torch tqdm scipy editdistance

    python experiments/neural_lm_rescore.py \
        --nbest-file results/nbest_dev_other_G16.jsonl \
        --results-dir results \
        --device cuda:0
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import editdistance
import numpy as np
import torch
from scipy import stats
from tqdm import tqdm

def compute_wer(hyp: str, ref: str) -> float:
    ref_w = ref.split()
    hyp_w = hyp.split()
    if len(ref_w) == 0:
        return 0.0 if len(hyp_w) == 0 else 1.0
    return editdistance.eval(hyp_w, ref_w) / len(ref_w)

def corpus_wer_from_selections(records, selections):
    total_edits = 0
    total_ref_words = 0
    for rec in records:
        sel = selections.get(rec["utt_id"], 0)
        sel = min(sel, len(rec["candidates"]) - 1)
        hyp = rec["candidates"][sel]["text"]
        ref = rec["ref_text"]
        ref_w = ref.split()
        hyp_w = hyp.split()
        total_edits += editdistance.eval(hyp_w, ref_w)
        total_ref_words += len(ref_w)
    return total_edits / max(total_ref_words, 1)

def load_nbest(path: Path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} utterances from {path}")
    return records

def save_jsonl(records, path: Path):
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {path}")

@torch.no_grad()
def compute_pll(text: str, tokenizer, model, device, batch_size: int = 64):
    """Pseudo-log-likelihood: sum_i log P(token_i | tokens_{-i}).

    For each non-special token position, mask it and compute the model's
    log-prob of the true token. Sum across positions.
    """
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"][0].to(device)
    L = input_ids.size(0)

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    cls = tokenizer.cls_token_id
    sep = tokenizer.sep_token_id
    mask_id = tokenizer.mask_token_id

    special = {bos, eos, pad, cls, sep}
    special.discard(None)

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

def score_with_roberta(records, model_name: str, device, batch_size: int):
    """Add `roberta_pll` field to each candidate in-place."""
    from transformers import RobertaTokenizer, RobertaForMaskedLM

    print(f"\nLoading {model_name}...")
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params/1e6:.1f}M parameters")

    # Sanity print: PLL on 5 short hypotheses to verify sensible values
    print("\n  PLL sanity check (first 5 short hypotheses):")
    n_shown = 0
    for rec in records:
        for cand in rec["candidates"]:
            if n_shown >= 5:
                break
            if 3 <= len(cand["text"].split()) <= 8:
                pll = compute_pll(cand["text"], tokenizer, model, device, batch_size)
                print(f"    PLL={pll:8.2f}  text={cand['text']!r}")
                n_shown += 1
        if n_shown >= 5:
            break

    print("\n  Scoring all candidates with RoBERTa PLL...")
    t0 = time.time()
    n_hyps = 0
    for rec in tqdm(records, desc="RoBERTa PLL"):
        for cand in rec["candidates"]:
            cand["roberta_pll"] = compute_pll(
                cand["text"], tokenizer, model, device, batch_size
            )
            n_hyps += 1
    elapsed = time.time() - t0
    print(f"  Scored {n_hyps} hypotheses in {elapsed:.1f}s "
          f"({n_hyps/elapsed:.1f} hyps/s)")

    del model
    torch.cuda.empty_cache()
    return elapsed

@torch.no_grad()
def compute_gpt2_lls_batch(texts, tokenizer, model, device, batch_size: int = 16):
    """Autoregressive LL for a list of texts; batched with padding."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="GPT-2 LL"):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True)
        ids = enc["input_ids"].to(device)
        am = enc["attention_mask"].to(device)
        logits = model(ids, attention_mask=am).logits  # (B, L, V)
        log_probs = torch.log_softmax(logits, dim=-1)
        # Shift so that position t predicts token t+1
        shift_lp = log_probs[:, :-1, :]
        shift_lab = ids[:, 1:]
        shift_mask = am[:, 1:].float()
        gathered = shift_lp.gather(2, shift_lab.unsqueeze(-1)).squeeze(-1)
        gathered = gathered * shift_mask
        seq_lls = gathered.sum(dim=1).cpu().tolist()
        results.extend(seq_lls)
    return results

def score_with_gpt2(records, model_name: str, device, batch_size: int):
    """Add `gpt2_ll` field to each candidate in-place."""
    from transformers import GPT2Tokenizer, GPT2LMHeadModel

    print(f"\nLoading {model_name}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params/1e6:.1f}M parameters")

    # Flatten all hypothesis texts; remember mapping back
    flat_texts = []
    flat_index = []  # (rec_idx, cand_idx)
    for ri, rec in enumerate(records):
        for ci, cand in enumerate(rec["candidates"]):
            flat_texts.append(cand["text"])
            flat_index.append((ri, ci))

    # Sanity: verify GPT-2 LL is negative & scales with length
    print("\n  GPT-2 LL sanity check (first 5 hypotheses):")
    short_lls = compute_gpt2_lls_batch(
        flat_texts[:5], tokenizer, model, device, batch_size=5
    )
    for txt, ll in zip(flat_texts[:5], short_lls):
        n_words = len(txt.split())
        print(f"    LL={ll:9.2f}  ({n_words:3d} words, LL/word={ll/max(n_words,1):.3f})  "
              f"text={txt[:60]!r}{'...' if len(txt)>60 else ''}")

    print("\n  Scoring all candidates with GPT-2 LL...")
    t0 = time.time()
    lls = compute_gpt2_lls_batch(flat_texts, tokenizer, model, device, batch_size)
    elapsed = time.time() - t0

    for (ri, ci), ll in zip(flat_index, lls):
        records[ri]["candidates"][ci]["gpt2_ll"] = ll

    print(f"  Scored {len(flat_texts)} hypotheses in {elapsed:.1f}s "
          f"({len(flat_texts)/elapsed:.1f} hyps/s)")

    del model
    torch.cuda.empty_cache()
    return elapsed

def annotate_wers(records):
    """Add `wer` to each candidate; add `greedy_wer`, `oracle_wer` per record.

    Greedy = candidate index 0 (per generate_nbest.py convention).
    """
    for rec in records:
        ref = rec["ref_text"]
        for cand in rec["candidates"]:
            cand["wer"] = compute_wer(cand["text"], ref)
        wers = [c["wer"] for c in rec["candidates"]]
        rec["greedy_wer"] = wers[0]
        rec["oracle_wer"] = min(wers)
        rec["is_recoverable"] = rec["oracle_wer"] < rec["greedy_wer"] - 1e-12

def per_utterance_spearman(records, score_field: str):
    """Mean per-utterance Spearman rho(score, WER) across utterances.

    Higher score should correspond to lower WER, so rho is expected to be
    negative. NaN rho values (constant scores or constant WERs) are skipped.
    """
    rhos = []
    for rec in records:
        cands = rec["candidates"]
        if len(cands) < 2:
            continue
        scores = [c[score_field] for c in cands]
        wers = [c["wer"] for c in cands]
        if len(set(scores)) < 2 or len(set(wers)) < 2:
            continue
        rho, _ = stats.spearmanr(scores, wers)
        if np.isnan(rho):
            continue
        rhos.append(rho)
    return float(np.mean(rhos)) if rhos else float("nan"), rhos

def per_utterance_spearman_combined(records, alpha: float, lm_field: str):
    rhos = []
    for rec in records:
        cands = rec["candidates"]
        if len(cands) < 2:
            continue
        scores = [
            alpha * c["ctc_log_prob"] + (1 - alpha) * c[lm_field] for c in cands
        ]
        wers = [c["wer"] for c in cands]
        if len(set(scores)) < 2 or len(set(wers)) < 2:
            continue
        rho, _ = stats.spearmanr(scores, wers)
        if np.isnan(rho):
            continue
        rhos.append(rho)
    return float(np.mean(rhos)) if rhos else float("nan")

def select_with_interpolation(records, alpha: float, lm_field: str):
    sels = {}
    for rec in records:
        best_score = -float("inf")
        best_idx = 0
        for i, c in enumerate(rec["candidates"]):
            score = alpha * c["ctc_log_prob"] + (1 - alpha) * c[lm_field]
            if score > best_score:
                best_score = score
                best_idx = i
        sels[rec["utt_id"]] = best_idx
    return sels

def alpha_sweep(records, lm_field: str, alphas, method_name: str,
                greedy_wer: float, oracle_wer: float):
    rows = []
    for a in alphas:
        sels = select_with_interpolation(records, a, lm_field)
        wer = corpus_wer_from_selections(records, sels)
        gap_total = greedy_wer - oracle_wer
        gap_closed = (greedy_wer - wer) / gap_total * 100 if gap_total > 0 else 0.0
        mean_rho = per_utterance_spearman_combined(records, a, lm_field)
        rows.append({
            "method": method_name,
            "alpha": round(a, 2),
            "wer": round(wer, 6),
            "gap_closed_pct": round(gap_closed, 2),
            "mean_spearman_rho": round(mean_rho, 4),
        })
        print(f"  {method_name:>8}  a={a:.1f}  WER={wer*100:.2f}%  "
              f"gap_closed={gap_closed:+6.1f}%  rho={mean_rho:+.3f}")
    return rows

def mbr_cer_with_weights(records, score_field: str, tau: float):
    """Select hypothesis minimizing expected CER with weights derived from
    softmax of `score_field` at temperature tau.
    """
    sels = {}
    for rec in records:
        cands = rec["candidates"]
        n = len(cands)
        texts = [c["text"] for c in cands]
        log_scores = np.array([c[score_field] for c in cands])

        if math.isinf(tau):
            weights = np.ones(n) / n
        else:
            scaled = log_scores / tau
            scaled -= np.max(scaled)
            weights = np.exp(scaled)
            weights /= weights.sum()

        cer = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = editdistance.eval(list(texts[i]), list(texts[j]))
                denom = max(len(texts[i]), len(texts[j]), 1)
                cer[i, j] = d / denom
                cer[j, i] = cer[i, j]

        risk = cer @ weights
        sels[rec["utt_id"]] = int(np.argmin(risk))
    return corpus_wer_from_selections(records, sels), sels

def mbr_pll_sweep(records, taus, greedy_wer, oracle_wer):
    rows = []
    for tau in taus:
        wer, _ = mbr_cer_with_weights(records, "roberta_pll", tau)
        gap_total = greedy_wer - oracle_wer
        gap_closed = (greedy_wer - wer) / gap_total * 100 if gap_total > 0 else 0.0
        tau_str = "inf" if math.isinf(tau) else f"{tau:.1f}"
        rows.append({
            "method": "mbr_pll",
            "alpha": "",
            "tau": tau_str,
            "wer": round(wer, 6),
            "gap_closed_pct": round(gap_closed, 2),
            "mean_spearman_rho": "",
        })
        print(f"  mbr_pll  tau={tau_str:>4}  WER={wer*100:.2f}%  "
              f"gap_closed={gap_closed:+6.1f}%")
    return rows

def best_alpha(rows, method_name):
    method_rows = [r for r in rows if r["method"] == method_name]
    return min(method_rows, key=lambda r: r["wer"])

def per_utterance_breakdown(records, best_alpha_per_method, output_path: Path):
    """Write per-utterance CSV with rho and selected WER for each method.

    Also prints recoverable-utterance and confusion analysis to stdout.
    """
    methods = list(best_alpha_per_method.keys())  # e.g. ['roberta_pll','gpt2_ll']

    # Pre-select per method using best alpha
    selections = {}
    for m, alpha in best_alpha_per_method.items():
        selections[m] = select_with_interpolation(records, alpha, m)

    rows = []
    summary = {m: {"recovered": 0, "regressed": 0, "tied": 0,
                   "diff_from_greedy": 0,
                   "diff_better": 0, "diff_worse": 0, "diff_equal": 0}
               for m in methods}
    n_recoverable = 0

    for rec in records:
        uid = rec["utt_id"]
        cands = rec["candidates"]
        greedy_wer = rec["greedy_wer"]
        oracle_wer = rec["oracle_wer"]
        is_recov = rec["is_recoverable"]
        if is_recov:
            n_recoverable += 1

        # Per-method WER from picked candidate
        method_wers = {}
        method_idxs = {}
        for m in methods:
            idx = selections[m][uid]
            method_idxs[m] = idx
            method_wers[m] = cands[idx]["wer"]

        # Per-utt Spearman rho per scorer (for diagnostic)
        def safe_rho(scores, wers):
            if len(scores) < 2 or len(set(scores)) < 2 or len(set(wers)) < 2:
                return float("nan")
            r, _ = stats.spearmanr(scores, wers)
            return r if not np.isnan(r) else float("nan")

        wers = [c["wer"] for c in cands]
        rho_ctc = safe_rho([c["ctc_log_prob"] for c in cands], wers)
        rho_pll = safe_rho([c["roberta_pll"] for c in cands], wers) \
            if "roberta_pll" in cands[0] else float("nan")
        rho_gpt = safe_rho([c["gpt2_ll"] for c in cands], wers) \
            if "gpt2_ll" in cands[0] else float("nan")

        best_interp = min(method_wers.values())

        rows.append({
            "utt_id": uid,
            "ctc_rho": f"{rho_ctc:.4f}" if not np.isnan(rho_ctc) else "",
            "pll_rho": f"{rho_pll:.4f}" if not np.isnan(rho_pll) else "",
            "gpt2_rho": f"{rho_gpt:.4f}" if not np.isnan(rho_gpt) else "",
            "ctc_wer": f"{greedy_wer:.4f}",
            "pll_wer": f"{method_wers.get('roberta_pll', greedy_wer):.4f}",
            "gpt2_wer": f"{method_wers.get('gpt2_ll', greedy_wer):.4f}",
            "best_interp_wer": f"{best_interp:.4f}",
            "oracle_wer": f"{oracle_wer:.4f}",
            "is_recoverable": int(is_recov),
        })

        # Recoverable analysis: did each method pick a candidate with
        # strictly lower WER than greedy?
        for m in methods:
            mw = method_wers[m]
            if mw < greedy_wer - 1e-12:
                summary[m]["recovered"] += int(is_recov)
            elif mw > greedy_wer + 1e-12:
                summary[m]["regressed"] += 1
            else:
                summary[m]["tied"] += 1

            # Confusion: when method != greedy
            if method_idxs[m] != 0:
                summary[m]["diff_from_greedy"] += 1
                if mw < greedy_wer - 1e-12:
                    summary[m]["diff_better"] += 1
                elif mw > greedy_wer + 1e-12:
                    summary[m]["diff_worse"] += 1
                else:
                    summary[m]["diff_equal"] += 1

    fields = ["utt_id", "ctc_rho", "pll_rho", "gpt2_rho",
              "ctc_wer", "pll_wer", "gpt2_wer", "best_interp_wer",
              "oracle_wer", "is_recoverable"]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Per-utterance CSV: {output_path}")

    print(f"\n  Recoverable utterances (oracle < greedy): {n_recoverable}")
    for m in methods:
        s = summary[m]
        recov_rate = (s["recovered"] / n_recoverable * 100
                      if n_recoverable > 0 else 0.0)
        print(f"\n  Method: {m}  (best alpha={best_alpha_per_method[m]:.1f})")
        print(f"    Recovered (recoverable utts): {s['recovered']}/{n_recoverable} "
              f"({recov_rate:.1f}%)")
        print(f"    Total picked != greedy: {s['diff_from_greedy']}")
        print(f"      -> better than greedy: {s['diff_better']}")
        print(f"      -> worse than greedy:  {s['diff_worse']}")
        print(f"      -> same WER as greedy: {s['diff_equal']}")

    return summary, n_recoverable

def generate_report(out_path: Path, *, greedy_wer, oracle_wer, n_utts,
                    interp_rows, mbr_rows, summary, n_recoverable,
                    best_alpha_per_method, mean_rhos, runtimes):
    lines = []
    lines.append("# Level 5: Neural LM Rescoring (RoBERTa PLL + GPT-2 LL)")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- Dataset: LibriSpeech dev-other ({n_utts} utterances)")
    lines.append("- N-best: G=16 (CTC lattice, nbest_scale=1.0) "
                 "from Zipformer-S CR-CTC")
    lines.append(f"- Greedy WER: {greedy_wer*100:.2f}%")
    lines.append(f"- Oracle WER (G=16): {oracle_wer*100:.2f}%")
    gap_pp = (greedy_wer - oracle_wer) * 100
    gap_rel = (greedy_wer - oracle_wer) / greedy_wer * 100
    lines.append(f"- Oracle gap: {gap_pp:.2f} pp ({gap_rel:.1f}% relative)")
    lines.append(f"- Recoverable utterances: {n_recoverable}")
    lines.append("")

    # Key numbers
    best_pll = min((r for r in interp_rows if r["method"] == "roberta_pll"),
                   key=lambda r: r["wer"])
    best_gpt = min((r for r in interp_rows if r["method"] == "gpt2_ll"),
                   key=lambda r: r["wer"])
    lines.append("## Key Numbers")
    lines.append("")
    lines.append("| Method | Best alpha | WER | Gap closed | Mean rho |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(f"| Greedy (CTC, alpha=1.0) | 1.0 | "
                 f"{greedy_wer*100:.2f}% | 0.0% | {mean_rhos['ctc']:+.3f} |")
    lines.append(f"| RoBERTa PLL interp | {best_pll['alpha']} | "
                 f"{best_pll['wer']*100:.2f}% | "
                 f"{best_pll['gap_closed_pct']:+.1f}% | "
                 f"{best_pll['mean_spearman_rho']:+.3f} |")
    lines.append(f"| GPT-2 LL interp | {best_gpt['alpha']} | "
                 f"{best_gpt['wer']*100:.2f}% | "
                 f"{best_gpt['gap_closed_pct']:+.1f}% | "
                 f"{best_gpt['mean_spearman_rho']:+.3f} |")
    lines.append(f"| Oracle (lower bound) | - | "
                 f"{oracle_wer*100:.2f}% | 100.0% | - |")
    lines.append("")

    # Spearman comparison
    lines.append("## Per-Utterance Spearman rho(score, WER)")
    lines.append("Lower (more negative) is better  --  score should be "
                 "anti-correlated with WER.")
    lines.append("")
    lines.append("| Scorer | Mean rho |")
    lines.append("|---|---:|")
    lines.append(f"| CTC log-prob | {mean_rhos['ctc']:+.3f} |")
    if "pll" in mean_rhos:
        lines.append(f"| RoBERTa PLL alone | {mean_rhos['pll']:+.3f} |")
    if "gpt" in mean_rhos:
        lines.append(f"| GPT-2 LL alone | {mean_rhos['gpt']:+.3f} |")
    lines.append("")

    # Alpha sweep table
    lines.append("## Alpha Sweep (RoBERTa PLL)")
    lines.append("Combined score: s = alpha * log_ctc + (1-alpha) * roberta_pll")
    lines.append("")
    lines.append("| alpha | WER | Gap closed | Mean rho |")
    lines.append("|---:|---:|---:|---:|")
    for r in interp_rows:
        if r["method"] == "roberta_pll":
            lines.append(f"| {r['alpha']} | {r['wer']*100:.2f}% | "
                         f"{r['gap_closed_pct']:+.1f}% | "
                         f"{r['mean_spearman_rho']:+.3f} |")
    lines.append("")

    lines.append("## Alpha Sweep (GPT-2 LL)")
    lines.append("Combined score: s = alpha * log_ctc + (1-alpha) * gpt2_ll")
    lines.append("")
    lines.append("| alpha | WER | Gap closed | Mean rho |")
    lines.append("|---:|---:|---:|---:|")
    for r in interp_rows:
        if r["method"] == "gpt2_ll":
            lines.append(f"| {r['alpha']} | {r['wer']*100:.2f}% | "
                         f"{r['gap_closed_pct']:+.1f}% | "
                         f"{r['mean_spearman_rho']:+.3f} |")
    lines.append("")

    # MBR
    if mbr_rows:
        lines.append("## MBR-CER with RoBERTa PLL Posterior Weights")
        lines.append("Tests whether MBR collapsed to greedy because of CTC's "
                     "peaked posteriors specifically. PLL is a flatter "
                     "non-CTC distribution.")
        lines.append("")
        lines.append("| tau | WER | Gap closed |")
        lines.append("|---:|---:|---:|")
        for r in mbr_rows:
            lines.append(f"| {r['tau']} | {r['wer']*100:.2f}% | "
                         f"{r['gap_closed_pct']:+.1f}% |")
        lines.append("")

    # Per-utterance recoverable
    lines.append("## Per-Utterance Recoverable Analysis")
    lines.append(f"Recoverable utterances: {n_recoverable} "
                 f"(oracle WER < greedy WER)")
    lines.append("")
    lines.append("| Method | Best alpha | Recovered | Recovery % | "
                 "Differ-from-greedy | Better | Worse | Same |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for m, s in summary.items():
        recov_rate = (s["recovered"] / n_recoverable * 100
                      if n_recoverable > 0 else 0.0)
        lines.append(f"| {m} | {best_alpha_per_method[m]:.1f} | "
                     f"{s['recovered']}/{n_recoverable} | {recov_rate:.1f}% | "
                     f"{s['diff_from_greedy']} | {s['diff_better']} | "
                     f"{s['diff_worse']} | {s['diff_equal']} |")
    lines.append("")

    # Information bottleneck thesis
    lines.append("## Information Bottleneck Thesis")
    best_external_wer = min(best_pll["wer"], best_gpt["wer"])
    best_external_gap = max(best_pll["gap_closed_pct"], best_gpt["gap_closed_pct"])
    if best_external_gap > 5.0:
        verdict = ("**CONFIRMED**: external linguistic signal closes a "
                   f"meaningful fraction ({best_external_gap:+.1f}%) of the "
                   "oracle gap that CTC-internal methods (MBR, length norm, "
                   "self-consistency) could not. The N-best list does contain "
                   "the right answer; the CTC posterior alone cannot identify "
                   "it because all its scores are projections of the same "
                   "acoustic encoding.")
    else:
        verdict = ("**INCONCLUSIVE**: even external LM signal only closes "
                   f"{best_external_gap:+.1f}% of the gap. The bottleneck "
                   "may be deeper than 'CTC scores are uninformative'  --  "
                   "candidate diversity itself may already be exhausted.")
    lines.append(verdict)
    lines.append("")

    # Runtime
    lines.append("## Runtime")
    lines.append("")
    for k, v in runtimes.items():
        lines.append(f"- {k}: {v:.1f}s ({v/60:.1f} min)")
    lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"Report: {out_path}")

def parse_args():
    p = argparse.ArgumentParser(
        description="Neural LM rescoring (RoBERTa PLL + GPT-2 LL) of CTC N-best"
    )
    p.add_argument(
        "--nbest-file", type=Path,
        default=Path("results/nbest_dev_other_G16.jsonl"),
    )
    p.add_argument(
        "--results-dir", type=Path,
        default=Path("/content/drive/MyDrive/rbpo_results"),
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--roberta-name", type=str, default="roberta-base")
    p.add_argument("--gpt2-name", type=str, default="gpt2")
    p.add_argument("--pll-batch-size", type=int, default=64)
    p.add_argument("--gpt2-batch-size", type=int, default=16)
    p.add_argument(
        "--num-utterances", type=int, default=-1,
        help="Limit utterances (-1 = all). Useful for smoke testing.",
    )
    p.add_argument(
        "--cached-scores", type=Path, default=None,
        help="Path to neural_lm_scores.jsonl with PLL/GPT-2 already "
             "computed. If provided, skip neural scoring and reuse.",
    )
    p.add_argument(
        "--skip-roberta", action="store_true",
        help="Skip RoBERTa PLL scoring (e.g., if reusing cached scores).",
    )
    p.add_argument(
        "--skip-gpt2", action="store_true",
        help="Skip GPT-2 LL scoring.",
    )
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 70)
    print("Neural LM Rescoring: RoBERTa PLL + GPT-2 LL on CTC N-best")
    print("=" * 70)

    # Load data
    if args.cached_scores and args.cached_scores.exists():
        print(f"\nLoading cached scores from {args.cached_scores}")
        records = load_nbest(args.cached_scores)
    else:
        records = load_nbest(args.nbest_file)

    if args.num_utterances > 0:
        records = records[: args.num_utterances]
        print(f"  Limited to {len(records)} utterances")

    # Score with neural LMs
    runtimes = {}
    have_pll = all(
        "roberta_pll" in c
        for r in records for c in r["candidates"]
    )
    have_gpt2 = all(
        "gpt2_ll" in c
        for r in records for c in r["candidates"]
    )

    if not have_pll and not args.skip_roberta:
        runtimes["roberta_pll"] = score_with_roberta(
            records, args.roberta_name, device, args.pll_batch_size
        )
    elif have_pll:
        print("\nRoBERTa PLL already cached; skipping scoring.")

    if not have_gpt2 and not args.skip_gpt2:
        runtimes["gpt2_ll"] = score_with_gpt2(
            records, args.gpt2_name, device, args.gpt2_batch_size
        )
    elif have_gpt2:
        print("\nGPT-2 LL already cached; skipping scoring.")

    # Compute per-candidate WER and per-record greedy/oracle
    print("\nAnnotating WERs...")
    annotate_wers(records)

    n_utts = len(records)
    greedy_wer = sum(r["greedy_wer"] * len(r["ref_text"].split())
                     for r in records) \
        / sum(len(r["ref_text"].split()) for r in records)
    # corpus-level oracle WER (best per utt, weighted by ref length)
    total_edits_oracle = 0
    total_words = 0
    for r in records:
        ref_w = r["ref_text"].split()
        total_words += len(ref_w)
        best = min(r["candidates"], key=lambda c: c["wer"])
        total_edits_oracle += editdistance.eval(best["text"].split(), ref_w)
    oracle_wer = total_edits_oracle / max(total_words, 1)

    # corpus greedy WER (cand[0] per utt)
    greedy_corpus_wer = corpus_wer_from_selections(
        records, {r["utt_id"]: 0 for r in records}
    )
    print(f"  Greedy corpus WER: {greedy_corpus_wer*100:.2f}%")
    print(f"  Oracle corpus WER: {oracle_wer*100:.2f}%")

    # Save scored JSONL
    args.results_dir.mkdir(parents=True, exist_ok=True)
    scores_path = args.results_dir / "neural_lm_scores.jsonl"
    save_jsonl(records, scores_path)

    # Verification: alpha=1.0 must reproduce greedy WER (~6.02%)
    print("\n" + "=" * 70)
    print("VERIFICATION CHECKS")
    print("=" * 70)
    # Pick the LM field that exists for the alpha=1 check
    lm_field = ("roberta_pll" if "roberta_pll" in records[0]["candidates"][0]
                else "gpt2_ll" if "gpt2_ll" in records[0]["candidates"][0]
                else None)
    if lm_field:
        sels_a1 = select_with_interpolation(records, 1.0, lm_field)
        wer_a1 = corpus_wer_from_selections(records, sels_a1)
        print(f"  alpha=1.0 (CTC only) WER: {wer_a1*100:.4f}%  "
              f"(greedy: {greedy_corpus_wer*100:.4f}%)")
        if abs(wer_a1 - greedy_corpus_wer) > 1e-6:
            print("  WARNING: alpha=1.0 != greedy "
                  "(some candidates may have higher CTC than candidate[0])")
        else:
            print("  PASS: alpha=1.0 matches greedy")
    else:
        print("  No neural LM scores available; cannot run alpha sweep.")
        return

    # Per-utterance Spearman (alone, not interpolated)
    print("\n" + "=" * 70)
    print("PER-UTTERANCE SPEARMAN rho(score, WER)")
    print("=" * 70)
    mean_rhos = {}
    rho_ctc, _ = per_utterance_spearman(records, "ctc_log_prob")
    print(f"  CTC log-prob:    rho = {rho_ctc:+.3f}")
    mean_rhos["ctc"] = rho_ctc

    if "roberta_pll" in records[0]["candidates"][0]:
        rho_pll, _ = per_utterance_spearman(records, "roberta_pll")
        print(f"  RoBERTa PLL:     rho = {rho_pll:+.3f}")
        mean_rhos["pll"] = rho_pll

    if "gpt2_ll" in records[0]["candidates"][0]:
        rho_gpt, _ = per_utterance_spearman(records, "gpt2_ll")
        print(f"  GPT-2 LL:        rho = {rho_gpt:+.3f}")
        mean_rhos["gpt"] = rho_gpt

    # Alpha sweep
    alphas = [round(0.1 * i, 2) for i in range(11)]  # 0.0 ... 1.0
    interp_rows = []

    if "roberta_pll" in records[0]["candidates"][0]:
        print("\n" + "=" * 70)
        print("ALPHA SWEEP  --  RoBERTa PLL interpolation")
        print("=" * 70)
        interp_rows.extend(alpha_sweep(
            records, "roberta_pll", alphas, "roberta_pll",
            greedy_corpus_wer, oracle_wer,
        ))

    if "gpt2_ll" in records[0]["candidates"][0]:
        print("\n" + "=" * 70)
        print("ALPHA SWEEP  --  GPT-2 LL interpolation")
        print("=" * 70)
        interp_rows.extend(alpha_sweep(
            records, "gpt2_ll", alphas, "gpt2_ll",
            greedy_corpus_wer, oracle_wer,
        ))

    # MBR-CER with PLL weights
    mbr_rows = []
    if "roberta_pll" in records[0]["candidates"][0]:
        print("\n" + "=" * 70)
        print("MBR-CER with RoBERTa PLL posterior weights")
        print("=" * 70)
        mbr_rows = mbr_pll_sweep(
            records, [1.0, 5.0, 10.0, 50.0, float("inf")],
            greedy_corpus_wer, oracle_wer,
        )

    # Save results CSV
    csv_path = args.results_dir / "neural_lm_rescore_results.csv"
    fieldnames = ["method", "alpha", "wer", "gap_closed_pct", "mean_spearman_rho"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(interp_rows)
        for r in mbr_rows:
            r2 = {k: r.get(k, "") for k in fieldnames}
            r2["method"] = f"mbr_pll_tau{r['tau']}"
            r2["alpha"] = ""
            w.writerow(r2)
    print(f"\nResults CSV: {csv_path}")

    # Per-utterance breakdown
    print("\n" + "=" * 70)
    print("PER-UTTERANCE RECOVERABLE BREAKDOWN")
    print("=" * 70)
    best_alpha_per_method = {}
    if "roberta_pll" in records[0]["candidates"][0]:
        best = best_alpha(interp_rows, "roberta_pll")
        best_alpha_per_method["roberta_pll"] = best["alpha"]
    if "gpt2_ll" in records[0]["candidates"][0]:
        best = best_alpha(interp_rows, "gpt2_ll")
        best_alpha_per_method["gpt2_ll"] = best["alpha"]

    summary, n_recoverable = per_utterance_breakdown(
        records, best_alpha_per_method,
        args.results_dir / "neural_lm_per_utterance.csv",
    )

    # Generate report
    report_path = args.results_dir / "report_neural_lm.md"
    generate_report(
        report_path,
        greedy_wer=greedy_corpus_wer,
        oracle_wer=oracle_wer,
        n_utts=n_utts,
        interp_rows=interp_rows,
        mbr_rows=mbr_rows,
        summary=summary,
        n_recoverable=n_recoverable,
        best_alpha_per_method=best_alpha_per_method,
        mean_rhos=mean_rhos,
        runtimes=runtimes,
    )

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Greedy WER:  {greedy_corpus_wer*100:.2f}%")
    print(f"  Oracle WER:  {oracle_wer*100:.2f}%")
    if interp_rows:
        best = min(interp_rows, key=lambda r: r["wer"])
        print(f"  Best rescored WER: {best['wer']*100:.2f}%  "
              f"({best['method']}, alpha={best['alpha']})")
        print(f"  Gap closed: {best['gap_closed_pct']:+.1f}%")
    print(f"  Output dir: {args.results_dir}")

if __name__ == "__main__":
    main()

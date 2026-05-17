#!/usr/bin/env python3
"""
E19: Systematic MBR selection sweep on existing G=128 N-best.
No new inference  --  re-scores existing candidates on CPU.

Sweeps: temperature, utility function, posterior model, two-stage selection.
"""

import json
import csv
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import editdistance

DATA_PATH = Path("/Users/melodiz/Desktop/RBPO/results/g_scaling/neural_lm_scores_G128.jsonl")
OUT_DIR = Path("/Users/melodiz/Desktop/RBPO/results/diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Add project root so we can import the bootstrap test
sys.path.insert(0, str(Path("/Users/melodiz/Desktop/RBPO")))
from experiments.significance_tests import paired_bootstrap_wer


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def corpus_wer(ref_words_list, hyp_words_list):
    """Corpus-level WER = total_edits / total_ref_words."""
    total_edits = sum(editdistance.eval(r, h) for r, h in zip(ref_words_list, hyp_words_list))
    total_ref = sum(len(r) for r in ref_words_list)
    return total_edits / total_ref if total_ref > 0 else 0.0


def compute_cer_matrix(texts):
    """Pairwise CER matrix (matches g_scaling_curve.py)."""
    n = len(texts)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = editdistance.eval(list(texts[i]), list(texts[j]))
            denom = max(len(texts[i]), len(texts[j]), 1)
            mat[i, j] = d / denom
            mat[j, i] = mat[i, j]
    return mat


def compute_wer_matrix(texts):
    """Pairwise WER matrix (word-level edit distance)."""
    n = len(texts)
    word_lists = [t.split() for t in texts]
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = editdistance.eval(word_lists[i], word_lists[j])
            denom = max(len(word_lists[i]), len(word_lists[j]), 1)
            mat[i, j] = d / denom
            mat[j, i] = mat[i, j]
    return mat


def compute_token_matrix(cands):
    """Pairwise BPE token edit distance matrix."""
    n = len(cands)
    token_lists = [c["tokens"] for c in cands]
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = editdistance.eval(token_lists[i], token_lists[j])
            denom = max(len(token_lists[i]), len(token_lists[j]), 1)
            mat[i, j] = d / denom
            mat[j, i] = mat[i, j]
    return mat


def compute_neg_bleu_matrix(texts, max_n=4):
    """Pairwise negative sentence-level BLEU (smoothed)."""
    n = len(texts)
    word_lists = [t.split() for t in texts]
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            # Symmetric: average BLEU(i|j) and BLEU(j|i)
            b1 = _smoothed_sent_bleu(word_lists[i], word_lists[j], max_n)
            b2 = _smoothed_sent_bleu(word_lists[j], word_lists[i], max_n)
            neg_bleu = 1.0 - (b1 + b2) / 2.0
            mat[i, j] = neg_bleu
            mat[j, i] = neg_bleu
    return mat


def _smoothed_sent_bleu(hyp, ref, max_n=4):
    """Smoothed sentence-level BLEU (Chen & Cherry, 2014 method 1)."""
    if not hyp or not ref:
        return 0.0
    brevity = min(1.0, len(hyp) / len(ref)) if len(ref) > 0 else 0.0
    log_bleu = 0.0
    for n in range(1, max_n + 1):
        hyp_ngrams = defaultdict(int)
        ref_ngrams = defaultdict(int)
        for k in range(len(hyp) - n + 1):
            hyp_ngrams[tuple(hyp[k:k+n])] += 1
        for k in range(len(ref) - n + 1):
            ref_ngrams[tuple(ref[k:k+n])] += 1
        clipped = sum(min(hyp_ngrams[ng], ref_ngrams[ng]) for ng in hyp_ngrams)
        total = max(len(hyp) - n + 1, 0)
        # Smoothing: add 1 to both numerator and denominator
        precision = (clipped + 1) / (total + 1) if total > 0 else 1.0 / (1.0 + 1.0)
        log_bleu += math.log(precision) / max_n
    return brevity * math.exp(log_bleu)


def make_weights(log_scores, tau):
    """Softmax with temperature. tau=inf => uniform."""
    n = len(log_scores)
    if math.isinf(tau):
        return np.ones(n) / n
    scaled = log_scores / tau
    scaled -= np.max(scaled)
    w = np.exp(scaled)
    w /= w.sum()
    return w


def mbr_select(dist_matrix, log_scores, tau):
    """MBR: argmin expected risk under softmax posterior."""
    weights = make_weights(log_scores, tau)
    risk = dist_matrix @ weights
    return int(np.argmin(risk)), risk


print("Loading G=128 data...")
t0 = time.time()
records = load_jsonl(DATA_PATH)
n_utts = len(records)
print(f"  {n_utts} utterances loaded in {time.time()-t0:.1f}s")

# Pre-compute references
ref_words_list = [rec["ref_text"].split() for rec in records]

greedy_hyps = [rec["candidates"][0]["text"].split() for rec in records]
greedy_wer = corpus_wer(ref_words_list, greedy_hyps)
print(f"  Greedy WER: {greedy_wer*100:.2f}% (expect ~6.02%)")


print("\nPre-computing distance matrices (this takes ~5-8 minutes)...")

cer_matrices = []
wer_matrices = []
tok_matrices = []
bleu_matrices = []
cand_texts_list = []
cand_lists = []

for i, rec in enumerate(records):
    cands = [c for c in rec["candidates"] if c["text"].strip()]
    if not cands:
        cands = rec["candidates"]
    cand_lists.append(cands)
    texts = [c["text"] for c in cands]
    cand_texts_list.append(texts)

    cer_matrices.append(compute_cer_matrix(texts))
    wer_matrices.append(compute_wer_matrix(texts))
    tok_matrices.append(compute_token_matrix(cands))
    bleu_matrices.append(compute_neg_bleu_matrix(texts))

    if (i + 1) % 500 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (n_utts - i - 1)
        print(f"  {i+1}/{n_utts} ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

print(f"  All matrices computed in {time.time()-t0:.0f}s")

UTILITY_MATRICES = {
    "cer": cer_matrices,
    "wer": wer_matrices,
    "token": tok_matrices,
    "neg_bleu": bleu_matrices,
}


def run_mbr(utility_name, score_arrays, tau):
    """Run MBR decoding across all utterances, return hypothesis word-lists."""
    matrices = UTILITY_MATRICES[utility_name]
    hyp_words = []
    for i in range(n_utts):
        idx, _ = mbr_select(matrices[i], score_arrays[i], tau)
        hyp_words.append(cand_texts_list[i][idx].split())
    return hyp_words


def run_mbr_with_risks(utility_name, score_arrays, tau):
    """Like run_mbr but also returns risks and selected indices."""
    matrices = UTILITY_MATRICES[utility_name]
    hyp_words = []
    all_risks = []
    all_indices = []
    for i in range(n_utts):
        idx, risk = mbr_select(matrices[i], score_arrays[i], tau)
        hyp_words.append(cand_texts_list[i][idx].split())
        all_risks.append(risk)
        all_indices.append(idx)
    return hyp_words, all_risks, all_indices


# Pre-extract score arrays for each posterior
print("\nExtracting score arrays...")
pll_scores = [np.array([c["roberta_pll"] for c in cl]) for cl in cand_lists]
ctc_scores = [np.array([c["ctc_log_prob"] for c in cl]) for cl in cand_lists]
gpt2_scores = [np.array([c["gpt2_ll"] for c in cl]) for cl in cand_lists]

all_results = []  # list of dicts for CSV


print("\n" + "="*70)
print("SWEEP 1: TEMPERATURE (CER utility, PLL posterior)")
print("="*70)

TAUS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, float("inf")]
tau_results = {}

for tau in TAUS:
    tau_str = "inf" if math.isinf(tau) else f"{tau}"
    hyps = run_mbr("cer", pll_scores, tau)
    wer = corpus_wer(ref_words_list, hyps)
    tau_results[tau] = {"wer": wer, "hyps": hyps}
    row = {"sweep": "tau", "utility": "cer", "posterior": "pll",
           "tau": tau_str, "wer_pct": f"{wer*100:.4f}"}
    all_results.append(row)
    print(f"  tau={tau_str:>6s}: WER = {wer*100:.4f}%")

best_tau = min(tau_results, key=lambda t: tau_results[t]["wer"])
best_tau_wer = tau_results[best_tau]["wer"]
best_tau_str = "inf" if math.isinf(best_tau) else f"{best_tau}"
print(f"\n  BEST tau = {best_tau_str} -> WER = {best_tau_wer*100:.4f}%")
print(f"  Current (tau=10): {tau_results[10.0]['wer']*100:.4f}%")
print(f"  Improvement: {(tau_results[10.0]['wer'] - best_tau_wer)*100:.4f}pp")


print("\n" + "="*70)
print("SWEEP 2: UTILITY FUNCTION")
print("="*70)

UTILITY_NAMES = ["cer", "wer", "token", "neg_bleu"]
tau_sweep_for_utility = sorted(set([
    best_tau,
    best_tau * 0.5 if not math.isinf(best_tau) else 50.0,
    best_tau * 2.0 if not math.isinf(best_tau) else float("inf"),
    10.0,  # current baseline
]))

utility_results = {}
for util_name in UTILITY_NAMES:
    for tau in tau_sweep_for_utility:
        tau_str = "inf" if math.isinf(tau) else f"{tau}"
        hyps = run_mbr(util_name, pll_scores, tau)
        wer = corpus_wer(ref_words_list, hyps)
        key = (util_name, tau)
        utility_results[key] = {"wer": wer, "hyps": hyps}
        row = {"sweep": "utility", "utility": util_name, "posterior": "pll",
               "tau": tau_str, "wer_pct": f"{wer*100:.4f}"}
        all_results.append(row)
        print(f"  {util_name:10s} tau={tau_str:>6s}: WER = {wer*100:.4f}%")

best_util_key = min(utility_results, key=lambda k: utility_results[k]["wer"])
print(f"\n  BEST utility={best_util_key[0]} tau={'inf' if math.isinf(best_util_key[1]) else best_util_key[1]} "
      f"-> WER = {utility_results[best_util_key]['wer']*100:.4f}%")


print("\n" + "="*70)
print("SWEEP 3: POSTERIOR MODEL")
print("="*70)

best_util = best_util_key[0]
best_tau_for_post = best_util_key[1]

# Also test the original best_tau with CER in case best_util differs
taus_for_post = sorted(set([best_tau, best_tau_for_post]))

posterior_results = {}

# 3a. Single-model posteriors
print("\n  --- Single posteriors ---")
for tau in taus_for_post:
    tau_str = "inf" if math.isinf(tau) else f"{tau}"
    for post_name, scores in [("pll", pll_scores), ("ctc", ctc_scores), ("gpt2", gpt2_scores)]:
        hyps = run_mbr(best_util, scores, tau)
        wer = corpus_wer(ref_words_list, hyps)
        key = (post_name, tau)
        posterior_results[key] = {"wer": wer, "hyps": hyps}
        row = {"sweep": "posterior", "utility": best_util, "posterior": post_name,
               "tau": tau_str, "wer_pct": f"{wer*100:.4f}"}
        all_results.append(row)
        print(f"    {post_name:>12s} tau={tau_str:>6s}: WER = {wer*100:.4f}%")

# 3b. CTC+PLL interpolation
print("\n  --- CTC+PLL interpolation ---")
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
interp_results = {}
for alpha in ALPHAS:
    for tau in taus_for_post:
        tau_str = "inf" if math.isinf(tau) else f"{tau}"
        interp_scores = [alpha * ctc + (1 - alpha) * pll
                         for ctc, pll in zip(ctc_scores, pll_scores)]
        hyps = run_mbr(best_util, interp_scores, tau)
        wer = corpus_wer(ref_words_list, hyps)
        key = (f"ctc_pll_a{alpha:.1f}", tau)
        posterior_results[key] = {"wer": wer, "hyps": hyps}
        interp_results[(alpha, tau)] = wer
        row = {"sweep": "posterior_interp", "utility": best_util,
               "posterior": f"ctc_pll_a{alpha:.1f}", "tau": tau_str,
               "wer_pct": f"{wer*100:.4f}"}
        all_results.append(row)
        print(f"    CTC+PLL alpha={alpha:.1f} tau={tau_str:>6s}: WER = {wer*100:.4f}%")

# 3c. CTC+PLL+GPT2 product of experts
print("\n  --- Three-way product of experts ---")
# Coarse grid: (alpha_ctc, alpha_pll, alpha_gpt2) summing to 1
TRIPLES = [(0.2, 0.6, 0.2), (0.3, 0.5, 0.2), (0.1, 0.7, 0.2),
           (0.2, 0.7, 0.1), (0.3, 0.6, 0.1), (0.1, 0.8, 0.1),
           (0.0, 0.8, 0.2), (0.0, 0.7, 0.3), (0.4, 0.5, 0.1)]
for a_ctc, a_pll, a_gpt2 in TRIPLES:
    for tau in taus_for_post:
        tau_str = "inf" if math.isinf(tau) else f"{tau}"
        combo_scores = [a_ctc * ctc + a_pll * pll + a_gpt2 * gpt2
                        for ctc, pll, gpt2 in zip(ctc_scores, pll_scores, gpt2_scores)]
        hyps = run_mbr(best_util, combo_scores, tau)
        wer = corpus_wer(ref_words_list, hyps)
        label = f"poe_{a_ctc:.1f}_{a_pll:.1f}_{a_gpt2:.1f}"
        key = (label, tau)
        posterior_results[key] = {"wer": wer, "hyps": hyps}
        row = {"sweep": "posterior_poe", "utility": best_util,
               "posterior": label, "tau": tau_str,
               "wer_pct": f"{wer*100:.4f}"}
        all_results.append(row)
        print(f"    PoE ({a_ctc:.1f},{a_pll:.1f},{a_gpt2:.1f}) tau={tau_str:>6s}: WER = {wer*100:.4f}%")

best_post_key = min(posterior_results, key=lambda k: posterior_results[k]["wer"])
print(f"\n  BEST posterior={best_post_key[0]} tau={'inf' if math.isinf(best_post_key[1]) else best_post_key[1]} "
      f"-> WER = {posterior_results[best_post_key]['wer']*100:.4f}%")


print("\n" + "="*70)
print("SWEEP 4: TWO-STAGE (MBR top-K -> argmax rescore)")
print("="*70)

KS = [3, 5, 10, 20]

# Use CER utility and PLL posterior at best tau for stage 1
# Then rescore with different scoring functions
two_stage_results = {}

# First, compute MBR risks at best_tau (for CER+PLL)
_, all_risks_best, all_indices_best = run_mbr_with_risks("cer", pll_scores, best_tau)

for K in KS:
    # Stage 1: MBR-CER top-K candidates
    # Stage 2: argmax various scores within top-K
    stage2_methods = {
        "pll": pll_scores,
        "ctc": ctc_scores,
        "gpt2": gpt2_scores,
        "interp_0.8": [0.8 * ctc + 0.2 * pll for ctc, pll in zip(ctc_scores, pll_scores)],
        "interp_0.7": [0.7 * ctc + 0.3 * pll for ctc, pll in zip(ctc_scores, pll_scores)],
    }

    for s2_name, s2_scores in stage2_methods.items():
        hyps = []
        for i in range(n_utts):
            risk = all_risks_best[i] if all_risks_best[i] is not None else np.zeros(len(cand_lists[i]))
            top_k_indices = np.argsort(risk)[:K]
            # Argmax of stage-2 score within top-K
            best_in_topk = max(top_k_indices, key=lambda idx: s2_scores[i][idx])
            hyps.append(cand_texts_list[i][best_in_topk].split())

        wer = corpus_wer(ref_words_list, hyps)
        key = (K, s2_name)
        two_stage_results[key] = {"wer": wer, "hyps": hyps}
        row = {"sweep": "two_stage", "utility": "cer",
               "posterior": f"mbr_top{K}_then_{s2_name}",
               "tau": best_tau_str, "wer_pct": f"{wer*100:.4f}"}
        all_results.append(row)
        print(f"  K={K:2d} -> argmax {s2_name:>12s}: WER = {wer*100:.4f}%")

# Also: pure argmax baselines (no MBR filter)
print("\n  --- Pure argmax (no MBR) ---")
argmax_baselines = {}
for s_name, s_scores in [("pll", pll_scores), ("ctc", ctc_scores),
                          ("gpt2", gpt2_scores),
                          ("interp_0.8", [0.8*c+0.2*p for c,p in zip(ctc_scores, pll_scores)]),
                          ("interp_0.7", [0.7*c+0.3*p for c,p in zip(ctc_scores, pll_scores)])]:
    hyps = []
    for i in range(n_utts):
        idx = int(np.argmax(s_scores[i]))
        hyps.append(cand_texts_list[i][idx].split())
    wer = corpus_wer(ref_words_list, hyps)
    argmax_baselines[s_name] = {"wer": wer, "hyps": hyps}
    row = {"sweep": "argmax_baseline", "utility": "none",
           "posterior": f"argmax_{s_name}", "tau": "0",
           "wer_pct": f"{wer*100:.4f}"}
    all_results.append(row)
    print(f"  argmax {s_name:>12s}: WER = {wer*100:.4f}%")

best_2stage_key = min(two_stage_results, key=lambda k: two_stage_results[k]["wer"])
print(f"\n  BEST two-stage: K={best_2stage_key[0]} rescore={best_2stage_key[1]} "
      f"-> WER = {two_stage_results[best_2stage_key]['wer']*100:.4f}%")


print("\n" + "="*70)
print("SWEEP 5: ORACLE ANALYSIS BY DIFFICULTY")
print("="*70)

# Use best config from sweep 1 (CER + PLL + best tau)
best_hyps = tau_results[best_tau]["hyps"]

# Per-utterance stats
utt_stats = []
for i in range(n_utts):
    ref_w = ref_words_list[i]
    ref_len = len(ref_w)
    n_unique = len(set(cand_texts_list[i]))
    ctc_arr = ctc_scores[i]
    ctc_entropy = -np.sum(np.exp(ctc_arr - np.max(ctc_arr)) / np.sum(np.exp(ctc_arr - np.max(ctc_arr)))
                          * (ctc_arr - np.max(ctc_arr) - np.log(np.sum(np.exp(ctc_arr - np.max(ctc_arr))))))

    # Oracle WER
    oracle_edits = min(editdistance.eval(ref_w, c.split()) for c in cand_texts_list[i])
    # Greedy WER
    greedy_edits = editdistance.eval(ref_w, cand_texts_list[i][0].split())
    # MBR WER (best tau)
    mbr_edits = editdistance.eval(ref_w, best_hyps[i])

    utt_stats.append({
        "ref_len": ref_len,
        "n_unique": n_unique,
        "ctc_entropy": ctc_entropy,
        "oracle_edits": oracle_edits,
        "greedy_edits": greedy_edits,
        "mbr_edits": mbr_edits,
        "ref_words_n": ref_len,
    })

# Quartile analysis by ref length
ref_lens = np.array([s["ref_len"] for s in utt_stats])
quartile_edges = np.percentile(ref_lens, [0, 25, 50, 75, 100])

print("\n  --- By reference length quartile ---")
quartile_rows = []
for q in range(4):
    lo = np.percentile(ref_lens, q * 25)
    hi = np.percentile(ref_lens, (q + 1) * 25)
    if q == 3:
        mask = (ref_lens >= lo) & (ref_lens <= hi)
    else:
        mask = (ref_lens >= lo) & (ref_lens < hi)
    if q == 0:
        mask = ref_lens <= hi

    indices = np.where(mask)[0]
    n_q = len(indices)
    total_ref = sum(utt_stats[j]["ref_len"] for j in indices)
    oracle_e = sum(utt_stats[j]["oracle_edits"] for j in indices)
    greedy_e = sum(utt_stats[j]["greedy_edits"] for j in indices)
    mbr_e = sum(utt_stats[j]["mbr_edits"] for j in indices)

    o_wer = oracle_e / total_ref * 100 if total_ref > 0 else 0
    g_wer = greedy_e / total_ref * 100 if total_ref > 0 else 0
    m_wer = mbr_e / total_ref * 100 if total_ref > 0 else 0

    qr = {"quartile": f"Q{q+1}", "ref_len_range": f"{lo:.0f}-{hi:.0f}",
           "n_utts": n_q, "greedy_wer": f"{g_wer:.2f}",
           "mbr_wer": f"{m_wer:.2f}", "oracle_wer": f"{o_wer:.2f}",
           "selection_gap_pp": f"{m_wer - o_wer:.2f}"}
    quartile_rows.append(qr)
    print(f"  Q{q+1} (len {lo:.0f}-{hi:.0f}, n={n_q}): "
          f"greedy={g_wer:.2f}% MBR={m_wer:.2f}% oracle={o_wer:.2f}% gap={m_wer-o_wer:.2f}pp")

# By unique hypotheses quartile
n_uniques = np.array([s["n_unique"] for s in utt_stats])
print("\n  --- By unique hypotheses quartile ---")
for q in range(4):
    lo = np.percentile(n_uniques, q * 25)
    hi = np.percentile(n_uniques, (q + 1) * 25)
    mask = (n_uniques >= lo) & (n_uniques <= hi) if q == 3 else (n_uniques >= lo) & (n_uniques < hi)
    if q == 0:
        mask = n_uniques <= hi
    indices = np.where(mask)[0]
    if len(indices) == 0:
        continue
    total_ref = sum(utt_stats[j]["ref_len"] for j in indices)
    oracle_e = sum(utt_stats[j]["oracle_edits"] for j in indices)
    mbr_e = sum(utt_stats[j]["mbr_edits"] for j in indices)
    o_wer = oracle_e / total_ref * 100 if total_ref > 0 else 0
    m_wer = mbr_e / total_ref * 100 if total_ref > 0 else 0
    print(f"  Q{q+1} (unique {lo:.0f}-{hi:.0f}, n={len(indices)}): "
          f"MBR={m_wer:.2f}% oracle={o_wer:.2f}% gap={m_wer-o_wer:.2f}pp")

# By CTC entropy quartile
entropies = np.array([s["ctc_entropy"] for s in utt_stats])
print("\n  --- By CTC entropy quartile ---")
for q in range(4):
    lo = np.percentile(entropies, q * 25)
    hi = np.percentile(entropies, (q + 1) * 25)
    mask = (entropies >= lo) & (entropies <= hi) if q == 3 else (entropies >= lo) & (entropies < hi)
    if q == 0:
        mask = entropies <= hi
    indices = np.where(mask)[0]
    if len(indices) == 0:
        continue
    total_ref = sum(utt_stats[j]["ref_len"] for j in indices)
    oracle_e = sum(utt_stats[j]["oracle_edits"] for j in indices)
    mbr_e = sum(utt_stats[j]["mbr_edits"] for j in indices)
    o_wer = oracle_e / total_ref * 100 if total_ref > 0 else 0
    m_wer = mbr_e / total_ref * 100 if total_ref > 0 else 0
    print(f"  Q{q+1} (entropy {lo:.2f}-{hi:.2f}, n={len(indices)}): "
          f"MBR={m_wer:.2f}% oracle={o_wer:.2f}% gap={m_wer-o_wer:.2f}pp")


print("\n" + "="*70)
print("OVERALL BEST & SIGNIFICANCE")
print("="*70)

all_configs = {}

# From tau sweep
for tau, res in tau_results.items():
    tau_s = "inf" if math.isinf(tau) else str(tau)
    all_configs[f"cer_pll_tau{tau_s}"] = res

# From utility sweep
for (util, tau), res in utility_results.items():
    tau_s = "inf" if math.isinf(tau) else str(tau)
    all_configs[f"{util}_pll_tau{tau_s}"] = res

# From posterior sweep
for (post, tau), res in posterior_results.items():
    tau_s = "inf" if math.isinf(tau) else str(tau)
    all_configs[f"{best_util}_{post}_tau{tau_s}"] = res

# From two-stage
for (K, s2), res in two_stage_results.items():
    all_configs[f"2stage_K{K}_{s2}"] = res

# From argmax baselines
for name, res in argmax_baselines.items():
    all_configs[f"argmax_{name}"] = res

sorted_configs = sorted(all_configs.items(), key=lambda kv: kv[1]["wer"])

print("\n  TOP 10 configurations:")
for rank, (name, res) in enumerate(sorted_configs[:10]):
    print(f"  {rank+1:2d}. {name:45s} WER = {res['wer']*100:.4f}%")

best_name = sorted_configs[0][0]
best_res = sorted_configs[0][1]
baseline_name = "cer_pll_tau10.0"
baseline_res = all_configs.get(baseline_name, tau_results.get(10.0))

print(f"\n  Best:     {best_name} -> {best_res['wer']*100:.4f}%")
print(f"  Baseline: cer_pll_tau10 -> {baseline_res['wer']*100:.4f}%")
print(f"  Delta:    {(baseline_res['wer'] - best_res['wer'])*100:.4f}pp")

# Bootstrap: best vs baseline
print("\n  Running bootstrap (B=10000)...")
boot = paired_bootstrap_wer(
    ref_words_list,
    best_res["hyps"],     # system A (candidate for improvement)
    baseline_res["hyps"], # system B (current baseline)
    n_bootstrap=10000,
    seed=42,
)
print(f"  Bootstrap result:")
print(f"    Best WER:     {boot['wer_a']*100:.4f}%")
print(f"    Baseline WER: {boot['wer_b']*100:.4f}%")
print(f"    Delta:        {boot['delta']*100:.4f}pp")
print(f"    p-value:      {boot['p_value']:.6f}")
print(f"    95% CI:       [{boot['ci_lower']*100:.4f}, {boot['ci_upper']*100:.4f}]pp")

# Also bootstrap: best vs greedy
boot_greedy = paired_bootstrap_wer(
    ref_words_list,
    best_res["hyps"],
    greedy_hyps,
    n_bootstrap=10000,
    seed=42,
)
print(f"\n  Best vs greedy:")
print(f"    Delta:   {boot_greedy['delta']*100:.4f}pp")
print(f"    p-value: {boot_greedy['p_value']:.6f}")


csv_path = OUT_DIR / "mbr_sweep_results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["sweep", "utility", "posterior", "tau", "wer_pct"])
    writer.writeheader()
    writer.writerows(all_results)
print(f"\nCSV saved to {csv_path}")


print("\n=== GENERATING PLOTS ===")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Plot 1: Temperature sweep
    fig, ax = plt.subplots(figsize=(8, 5))
    tau_vals = [t for t in TAUS if not math.isinf(t)]
    tau_wers = [tau_results[t]["wer"] * 100 for t in tau_vals]
    uniform_wer = tau_results[float("inf")]["wer"] * 100

    ax.plot(tau_vals, tau_wers, "o-", color="tab:blue", linewidth=2, label="MBR-CER + PLL")
    ax.axhline(y=uniform_wer, color="gray", linestyle="--", alpha=0.7,
               label=f"Uniform (tau=inf): {uniform_wer:.2f}%")
    ax.axhline(y=greedy_wer * 100, color="tab:red", linestyle=":", alpha=0.7,
               label=f"Greedy: {greedy_wer*100:.2f}%")
    ax.set_xscale("log")
    ax.set_xlabel("Temperature (tau)")
    ax.set_ylabel("Corpus WER (%)")
    ax.set_title("MBR Temperature Sweep (CER utility, PLL posterior)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "mbr_tau_sweep.png", dpi=150, bbox_inches="tight")
    print("  Saved mbr_tau_sweep.png")
    plt.close()

    # Plot 2: Utility x tau heatmap
    fig, ax = plt.subplots(figsize=(10, 5))
    tau_labels = [("inf" if math.isinf(t) else str(t)) for t in tau_sweep_for_utility]
    util_labels = UTILITY_NAMES
    heatmap_data = np.zeros((len(util_labels), len(tau_sweep_for_utility)))
    for ui, u in enumerate(util_labels):
        for ti, t in enumerate(tau_sweep_for_utility):
            key = (u, t)
            if key in utility_results:
                heatmap_data[ui, ti] = utility_results[key]["wer"] * 100
            else:
                heatmap_data[ui, ti] = float("nan")

    im = ax.imshow(heatmap_data, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(tau_labels)))
    ax.set_xticklabels(tau_labels)
    ax.set_yticks(range(len(util_labels)))
    ax.set_yticklabels(util_labels)
    ax.set_xlabel("Temperature (tau)")
    ax.set_ylabel("Utility function")
    ax.set_title("WER (%) by Utility x Temperature")
    for ui in range(len(util_labels)):
        for ti in range(len(tau_labels)):
            v = heatmap_data[ui, ti]
            if not np.isnan(v):
                ax.text(ti, ui, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if v > np.nanmedian(heatmap_data) else "black")
    plt.colorbar(im, ax=ax, label="WER (%)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "mbr_utility_heatmap.png", dpi=150, bbox_inches="tight")
    print("  Saved mbr_utility_heatmap.png")
    plt.close()

    # Plot 3: Two-stage bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    groups = sorted(set(k[1] for k in two_stage_results.keys()))
    x = np.arange(len(KS))
    width = 0.15
    for gi, grp in enumerate(groups):
        vals = [two_stage_results[(K, grp)]["wer"] * 100 for K in KS]
        ax.bar(x + gi * width, vals, width, label=f"rescore: {grp}")
    ax.axhline(y=best_tau_wer * 100, color="red", linestyle="--", alpha=0.7,
               label=f"MBR best: {best_tau_wer*100:.2f}%")
    ax.axhline(y=greedy_wer * 100, color="gray", linestyle=":", alpha=0.5,
               label=f"Greedy: {greedy_wer*100:.2f}%")
    ax.set_xticks(x + width * (len(groups) - 1) / 2)
    ax.set_xticklabels([f"K={K}" for K in KS])
    ax.set_ylabel("Corpus WER (%)")
    ax.set_title("Two-Stage: MBR top-K then argmax rescore")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "mbr_two_stage.png", dpi=150, bbox_inches="tight")
    print("  Saved mbr_two_stage.png")
    plt.close()

    # Plot 4: Difficulty analysis
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    qs = [f"Q{i+1}" for i in range(4)]

    ax = axes[0]
    g_wers = [float(r["greedy_wer"]) for r in quartile_rows]
    m_wers = [float(r["mbr_wer"]) for r in quartile_rows]
    o_wers = [float(r["oracle_wer"]) for r in quartile_rows]
    x = np.arange(4)
    ax.bar(x - 0.2, g_wers, 0.2, label="Greedy", color="gray")
    ax.bar(x, m_wers, 0.2, label="MBR", color="tab:blue")
    ax.bar(x + 0.2, o_wers, 0.2, label="Oracle", color="tab:green")
    labels = [f"Q{i+1}\n{quartile_rows[i]['ref_len_range']}w" for i in range(4)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("WER (%)")
    ax.set_title("By Reference Length")
    ax.legend()

    ax = axes[1]
    gaps = [float(r["selection_gap_pp"]) for r in quartile_rows]
    ax.bar(x, gaps, color="tab:red", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Selection gap (pp)")
    ax.set_title("Selection Gap by Length")

    ax = axes[2]
    # Scatter: ref_len vs selection gap per utt
    sel_gaps_per_utt = [(s["mbr_edits"] - s["oracle_edits"]) / max(s["ref_len"], 1) * 100
                        for s in utt_stats]
    ax.scatter(ref_lens, sel_gaps_per_utt, alpha=0.1, s=5, color="tab:purple")
    ax.set_xlabel("Reference length (words)")
    ax.set_ylabel("Per-utt selection gap (WER pp)")
    ax.set_title("Selection Gap vs Length (per utterance)")
    ax.set_ylim(-5, 50)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "mbr_difficulty_analysis.png", dpi=150, bbox_inches="tight")
    print("  Saved mbr_difficulty_analysis.png")
    plt.close()

    print("All plots saved.")
except ImportError:
    print("matplotlib not available, skipping plots")


print("\n=== WRITING REPORT ===")

report = []
report.append("# E19: MBR Selection Sweep  --  Systematic Optimization")
report.append("")
report.append("**Dataset:** LibriSpeech dev-other, 2864 utterances, G=128")
report.append("**Date:** 2026-05-06")
report.append(f"**Baseline:** MBR-CER + RoBERTa-PLL tau=10 -> {tau_results[10.0]['wer']*100:.2f}% WER")
report.append(f"**Greedy:** {greedy_wer*100:.2f}%")
report.append("")

# Sweep 1
report.append("## 1. Temperature Sweep (CER utility, PLL posterior)")
report.append("")
report.append("| tau | WER (%) | Delta vs tau=10 (pp) |")
report.append("|---:|---:|---:|")
baseline_w = tau_results[10.0]["wer"] * 100
for tau in TAUS:
    ts = "inf" if math.isinf(tau) else f"{tau}"
    w = tau_results[tau]["wer"] * 100
    d = w - baseline_w
    report.append(f"| {ts} | {w:.4f} | {d:+.4f} |")
report.append("")
report.append(f"**Best tau = {best_tau_str}** -> {best_tau_wer*100:.4f}% "
              f"({(baseline_w/100 - best_tau_wer)*100:+.4f}pp vs tau=10)")
report.append("")
report.append("![Temperature Sweep](mbr_tau_sweep.png)")
report.append("")

# Sweep 2
report.append("## 2. Utility Function Sweep")
report.append("")
report.append("| Utility | " + " | ".join([f"tau={'inf' if math.isinf(t) else t}" for t in tau_sweep_for_utility]) + " |")
report.append("|---:" + "|---:" * len(tau_sweep_for_utility) + "|")
for util in UTILITY_NAMES:
    cells = []
    for tau in tau_sweep_for_utility:
        key = (util, tau)
        if key in utility_results:
            cells.append(f"{utility_results[key]['wer']*100:.4f}")
        else:
            cells.append(" -- ")
    flag = " **(circular!)**" if util == "wer" else ""
    report.append(f"| {util}{flag} | " + " | ".join(cells) + " |")
report.append("")
bu = best_util_key
report.append(f"**Best:** {bu[0]} at tau={'inf' if math.isinf(bu[1]) else bu[1]} "
              f"-> {utility_results[bu]['wer']*100:.4f}%")
report.append("")
report.append("![Utility Heatmap](mbr_utility_heatmap.png)")
report.append("")

# Sweep 3
report.append("## 3. Posterior Model Sweep")
report.append("")
report.append("### Single posteriors")
report.append("")
report.append("| Posterior | " + " | ".join([f"tau={'inf' if math.isinf(t) else t}" for t in taus_for_post]) + " |")
report.append("|---:" + "|---:" * len(taus_for_post) + "|")
for pname in ["pll", "ctc", "gpt2"]:
    cells = []
    for tau in taus_for_post:
        key = (pname, tau)
        if key in posterior_results:
            cells.append(f"{posterior_results[key]['wer']*100:.4f}")
        else:
            cells.append(" -- ")
    report.append(f"| {pname} | " + " | ".join(cells) + " |")
report.append("")

report.append("### CTC+PLL interpolation (best tau)")
report.append("")
report.append("| alpha (CTC weight) | " + " | ".join([f"tau={'inf' if math.isinf(t) else t}" for t in taus_for_post]) + " |")
report.append("|---:" + "|---:" * len(taus_for_post) + "|")
for alpha in ALPHAS:
    cells = []
    for tau in taus_for_post:
        key = (f"ctc_pll_a{alpha:.1f}", tau)
        if key in posterior_results:
            cells.append(f"{posterior_results[key]['wer']*100:.4f}")
        else:
            cells.append(" -- ")
    report.append(f"| {alpha:.1f} | " + " | ".join(cells) + " |")
report.append("")

report.append("### Three-way product of experts")
report.append("")
report.append("| CTC | PLL | GPT-2 | WER (%) |")
report.append("|---:|---:|---:|---:|")
for a_ctc, a_pll, a_gpt2 in TRIPLES:
    label = f"poe_{a_ctc:.1f}_{a_pll:.1f}_{a_gpt2:.1f}"
    for tau in taus_for_post[:1]:
        key = (label, tau)
        if key in posterior_results:
            report.append(f"| {a_ctc:.1f} | {a_pll:.1f} | {a_gpt2:.1f} | "
                          f"{posterior_results[key]['wer']*100:.4f} |")
report.append("")

bp = best_post_key
report.append(f"**Best posterior:** {bp[0]} at tau={'inf' if math.isinf(bp[1]) else bp[1]} "
              f"-> {posterior_results[bp]['wer']*100:.4f}%")
report.append("")

# Sweep 4
report.append("## 4. Two-Stage: MBR Top-K then Argmax Rescore")
report.append("")
report.append("| K | " + " | ".join(groups) + " |")
report.append("|---:" + "|---:" * len(groups) + "|")
for K in KS:
    cells = [f"{two_stage_results[(K, g)]['wer']*100:.4f}" for g in groups]
    report.append(f"| {K} | " + " | ".join(cells) + " |")
report.append("")
report.append("### Argmax baselines (no MBR)")
report.append("")
for name in sorted(argmax_baselines):
    report.append(f"- argmax {name}: {argmax_baselines[name]['wer']*100:.4f}%")
report.append("")
report.append("![Two-Stage](mbr_two_stage.png)")
report.append("")

# Sweep 5
report.append("## 5. Difficulty Analysis")
report.append("")
report.append("### By reference length quartile")
report.append("")
report.append("| Quartile | Length | N | Greedy | MBR | Oracle | Gap (pp) |")
report.append("|---|---|---:|---:|---:|---:|---:|")
for r in quartile_rows:
    report.append(f"| {r['quartile']} | {r['ref_len_range']} | {r['n_utts']} | "
                  f"{r['greedy_wer']}% | {r['mbr_wer']}% | {r['oracle_wer']}% | "
                  f"{r['selection_gap_pp']} |")
report.append("")
report.append("![Difficulty Analysis](mbr_difficulty_analysis.png)")
report.append("")

# Overall best
report.append("## 6. Overall Best Configuration")
report.append("")
report.append("### Top 10")
report.append("")
report.append("| Rank | Configuration | WER (%) |")
report.append("|---:|---|---:|")
for rank, (name, res) in enumerate(sorted_configs[:10]):
    report.append(f"| {rank+1} | {name} | {res['wer']*100:.4f} |")
report.append("")
report.append(f"### Best vs baseline (bootstrap B=10000)")
report.append("")
report.append(f"| | Config | WER (%) |")
report.append(f"|---|---|---:|")
report.append(f"| Best | {best_name} | {boot['wer_a']*100:.4f} |")
report.append(f"| Baseline | MBR-CER+PLL tau=10 | {boot['wer_b']*100:.4f} |")
report.append(f"| Delta | | {boot['delta']*100:.4f}pp |")
report.append(f"| p-value | | {boot['p_value']:.6f} |")
report.append(f"| 95% CI | | [{boot['ci_lower']*100:.4f}, {boot['ci_upper']*100:.4f}]pp |")
report.append("")

improvement = (baseline_res["wer"] - best_res["wer"]) * 100
remaining_gap = (best_res["wer"] - 3.53/100) * 100  # oracle = 3.53%
report.append(f"### Gap analysis")
report.append("")
report.append(f"| | WER (%) | Gap to oracle (pp) |")
report.append(f"|---|---:|---:|")
report.append(f"| Greedy | {greedy_wer*100:.2f} | {(greedy_wer - 0.0353)*100:.2f} |")
report.append(f"| MBR baseline (tau=10) | {baseline_res['wer']*100:.2f} | {(baseline_res['wer'] - 0.0353)*100:.2f} |")
report.append(f"| **Best config** | **{best_res['wer']*100:.2f}** | **{(best_res['wer'] - 0.0353)*100:.2f}** |")
report.append(f"| Oracle | 3.53 | 0.00 |")
report.append("")

# Verdict
report.append("## 7. Verdict")
report.append("")

if improvement > 0.3:
    report.append(f"Tuning yields a **meaningful improvement** of {improvement:.2f}pp over the baseline. ")
elif improvement > 0.05:
    report.append(f"Tuning yields a **modest improvement** of {improvement:.2f}pp over the baseline. ")
else:
    report.append(f"Tuning yields **negligible improvement** ({improvement:.2f}pp) over the baseline. ")

report.append(f"However, {remaining_gap:.2f}pp of selection error remains (out of the original 2.01pp). ")

if remaining_gap > 1.5:
    report.append("**A trained reranker is still necessary**  --  hyperparameter tuning cannot close "
                  "the majority of the selection gap. The MBR scoring function is fundamentally "
                  "limited: CER-based consensus with log-linear posteriors lacks the capacity "
                  "to model what makes a hypothesis correct.")
elif remaining_gap > 0.5:
    report.append("Tuning helps but a trained reranker would likely close more of the gap. "
                  "The scoring function captures some signal but lacks capacity for fine-grained "
                  "discrimination.")
else:
    report.append("Tuning largely closes the gap  --  a trained reranker may not be necessary.")

report.append("")

with open(OUT_DIR / "mbr_selection_sweep.md", "w") as f:
    f.write("\n".join(report))

print(f"\nReport saved to {OUT_DIR / 'mbr_selection_sweep.md'}")
print(f"Total runtime: {time.time()-t0:.0f}s")
print("DONE.")

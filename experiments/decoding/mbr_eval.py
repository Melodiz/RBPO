#!/usr/bin/env python3
"""MBR-CER + PLL evaluation with tau sweep and E18-style gap diagnostics.

Reads PLL-scored N-best (E21 format with 'hypotheses' or E11/E20 format
with 'candidates'). Computes:
  - Greedy WER (top-1 by CTC log-prob)
  - Oracle WER (best per utt by edit distance)
  - MBR-CER + PLL tau sweep
  - MBR-WER + PLL (E19's circular-but-better utility)
  - Selection-bottleneck diagnostics (matches E18):
    * MBR selects oracle: count
    * Oracle rank in MBR ordering: median, mean
    * Coverage (oracle from candidates) vs selection (MBR vs oracle) gap split

Output:
  - reports-style markdown summary
  - results.json with all numbers
"""

import argparse
import editdistance
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)



def get_cands(rec):
    if "hypotheses" in rec:
        return rec["hypotheses"]
    return rec["candidates"]


def get_ref(rec):
    return rec.get("reference", rec.get("ref_text", ""))



def cer_distance(a, b):
    return editdistance.eval(list(a), list(b)) / max(len(a), len(b), 1)


def wer_distance(a, b):
    aw, bw = a.split(), b.split()
    return editdistance.eval(aw, bw) / max(len(aw), len(bw), 1)


def precompute_dist_matrix(texts, dist_fn):
    n = len(texts)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = dist_fn(texts[i], texts[j])
            D[i, j] = d
            D[j, i] = d
    return D


def mbr_select_with_matrix(D, log_scores, tau):
    log_p = np.asarray(log_scores, dtype=np.float64) / tau
    log_p = log_p - log_p.max()
    p = np.exp(log_p); p = p / p.sum()
    risk = D @ p
    return int(np.argmin(risk)), risk


def corpus_wer(refs, hyps):
    edits = sum(editdistance.eval(h.split(), r.split()) for h, r in zip(hyps, refs))
    ref_words = sum(len(r.split()) for r in refs)
    return edits / max(1, ref_words)



def select_greedy(records):
    """Top-1 by CTC log-prob (greedy injected at position 0)."""
    refs, hyps = [], []
    for rec in records:
        cands = get_cands(rec)
        # Position 0 is the injected greedy in E21 format; otherwise
        # find argmax CTC explicitly.
        idx = int(np.argmax([c["ctc_log_prob"] for c in cands]))
        hyps.append(cands[idx]["text"])
        refs.append(get_ref(rec))
    return refs, hyps


def select_oracle(records):
    refs, hyps = [], []
    for rec in records:
        cands = get_cands(rec)
        ref = get_ref(rec)
        rw = ref.split()
        edits = [editdistance.eval(c["text"].split(), rw) for c in cands]
        idx = int(np.argmin(edits))
        hyps.append(cands[idx]["text"])
        refs.append(ref)
    return refs, hyps


def select_mbr_pll(records, tau, utility="cer"):
    """MBR with PLL posterior, CER (or WER) utility matrix."""
    refs, hyps = [], []
    dist_fn = cer_distance if utility == "cer" else wer_distance
    for rec in records:
        cands = get_cands(rec)
        if "roberta_pll" not in cands[0]:
            raise RuntimeError("PLL field missing  --  run score_neural_lm.py first")
        texts = [c["text"] for c in cands]
        scores = [c["roberta_pll"] for c in cands]
        D = precompute_dist_matrix(texts, dist_fn)
        idx, _ = mbr_select_with_matrix(D, scores, tau)
        hyps.append(cands[idx]["text"])
        refs.append(get_ref(rec))
    return refs, hyps



def diagnostics(records, mbr_tau=10.0):
    """Selection-bottleneck analysis matching E18's framework."""
    n_total = len(records)
    n_unique = []
    oracle_in_set = 0  # oracle WER < greedy WER (at least one cand beats greedy)
    mbr_picks_oracle = 0
    oracle_ranks = []
    greedy_edits = 0
    oracle_edits = 0
    mbr_edits = 0
    total_ref_words = 0

    for rec in records:
        cands = get_cands(rec)
        ref = get_ref(rec)
        rw = ref.split()
        ref_len = len(rw)
        total_ref_words += ref_len

        # Per-cand edits
        edits = [editdistance.eval(c["text"].split(), rw) for c in cands]

        # Greedy: top-1 CTC
        g_idx = int(np.argmax([c["ctc_log_prob"] for c in cands]))
        # Oracle: min edits
        o_idx = int(np.argmin(edits))
        # MBR with PLL tau
        scores = [c["roberta_pll"] for c in cands]
        texts = [c["text"] for c in cands]
        D = precompute_dist_matrix(texts, cer_distance)
        m_idx, risk = mbr_select_with_matrix(D, scores, mbr_tau)

        greedy_edits += edits[g_idx]
        oracle_edits += edits[o_idx]
        mbr_edits += edits[m_idx]

        # Diversity
        unique_texts = set(c["text"] for c in cands)
        n_unique.append(len(unique_texts))

        # Oracle in set: did the candidate set actually contain a hyp better than greedy?
        if edits[o_idx] < edits[g_idx]:
            oracle_in_set += 1

        # MBR picks oracle
        if m_idx == o_idx:
            mbr_picks_oracle += 1

        # Oracle rank in MBR ordering (rank by ascending risk, lower = MBR prefers more)
        ordering = np.argsort(risk)
        oracle_rank = int(np.where(ordering == o_idx)[0][0]) + 1  # 1-indexed
        oracle_ranks.append(oracle_rank)

    wer_greedy = greedy_edits / max(1, total_ref_words)
    wer_oracle = oracle_edits / max(1, total_ref_words)
    wer_mbr = mbr_edits / max(1, total_ref_words)

    total_gap = wer_greedy - wer_oracle
    selection_gap = wer_mbr - wer_oracle  # what MBR fails to recover
    coverage_recovered = wer_greedy - wer_mbr  # what MBR did recover

    return {
        "n_utterances": n_total,
        "wer_greedy": wer_greedy,
        "wer_oracle": wer_oracle,
        "wer_mbr_pll": wer_mbr,
        "total_gap_pp": total_gap * 100,
        "covered_by_mbr_pp": coverage_recovered * 100,
        "selection_residual_pp": selection_gap * 100,
        "selection_pct_of_gap": selection_gap / max(total_gap, 1e-9) * 100,
        "mean_unique": float(np.mean(n_unique)),
        "median_unique": float(np.median(n_unique)),
        "utts_with_better_than_greedy_in_nbest": oracle_in_set,
        "mbr_picks_oracle_count": mbr_picks_oracle,
        "mbr_picks_oracle_pct": mbr_picks_oracle / n_total * 100,
        "oracle_rank_in_mbr_median": float(np.median(oracle_ranks)),
        "oracle_rank_in_mbr_mean": float(np.mean(oracle_ranks)),
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tau-sweep", type=float, nargs="+",
                        default=[5.0, 10.0, 20.0])
    parser.add_argument("--include-wer-utility", action="store_true",
                        help="Also run MBR-WER (E19's circular-but-better)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MBR-CER + PLL evaluation with diagnostics")
    print("=" * 60)
    print(f"  Input:    {args.input_jsonl}")
    print(f"  Output:   {args.output_dir}")
    print(f"  tau sweep:  {args.tau_sweep}")

    print("Loading...")
    records = []
    with open(args.input_jsonl) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"  {len(records)} utterances")

    cands0 = get_cands(records[0])
    if "roberta_pll" not in cands0[0]:
        print(" FATAL: 'roberta_pll' missing from candidates. "
              "Run score_neural_lm.py first.")
        sys.exit(1)

    refs_g, hyps_g = select_greedy(records)
    wer_g = corpus_wer(refs_g, hyps_g)
    refs_o, hyps_o = select_oracle(records)
    wer_o = corpus_wer(refs_o, hyps_o)

    print(f"\n=== Baselines ===")
    print(f"  Greedy WER:  {wer_g*100:.3f}%")
    print(f"  Oracle WER:  {wer_o*100:.3f}%")
    print(f"  Total gap:   {(wer_g-wer_o)*100:.3f} pp")

    print(f"\n=== MBR-CER + PLL (tau sweep) ===")
    sweep_results = []
    best_cer = None
    for tau in args.tau_sweep:
        t0 = time.time()
        refs_m, hyps_m = select_mbr_pll(records, tau, utility="cer")
        wer_m = corpus_wer(refs_m, hyps_m)
        elapsed = time.time() - t0
        sweep_results.append({"utility": "cer", "tau": tau, "wer": wer_m,
                              "elapsed_s": elapsed})
        print(f"  tau={tau:>5.1f}  WER={wer_m*100:.3f}%  ({elapsed:.0f}s)")
        if best_cer is None or wer_m < best_cer["wer"]:
            best_cer = {"tau": tau, "wer": wer_m}

    best_wer_util = None
    if args.include_wer_utility:
        print(f"\n=== MBR-WER + PLL (E19 circular-but-better) ===")
        for tau in args.tau_sweep:
            t0 = time.time()
            refs_w, hyps_w = select_mbr_pll(records, tau, utility="wer")
            wer_w = corpus_wer(refs_w, hyps_w)
            elapsed = time.time() - t0
            sweep_results.append({"utility": "wer", "tau": tau, "wer": wer_w,
                                  "elapsed_s": elapsed})
            print(f"  tau={tau:>5.1f}  WER={wer_w*100:.3f}%  ({elapsed:.0f}s)")
            if best_wer_util is None or wer_w < best_wer_util["wer"]:
                best_wer_util = {"tau": tau, "wer": wer_w}

    print(f"\n=== E18-style diagnostics @ tau={best_cer['tau']} ===")
    t0 = time.time()
    diag = diagnostics(records, mbr_tau=best_cer["tau"])
    print(f"  Done in {(time.time()-t0)/60:.1f} min")

    print(f"\n  WER greedy:                    {diag['wer_greedy']*100:.3f}%")
    print(f"  WER oracle (min edits/cand):    {diag['wer_oracle']*100:.3f}%")
    print(f"  WER MBR-CER+PLL tau={best_cer['tau']}:   {diag['wer_mbr_pll']*100:.3f}%")
    print(f"  Total gap (greedy -> oracle):    {diag['total_gap_pp']:.3f} pp")
    print(f"  Covered by MBR (greedy -> MBR):  {diag['covered_by_mbr_pp']:.3f} pp "
          f"({diag['covered_by_mbr_pp']/diag['total_gap_pp']*100:.1f}% of gap)")
    print(f"  Selection residual (MBR->oracle):{diag['selection_residual_pp']:.3f} pp "
          f"({diag['selection_pct_of_gap']:.1f}% of gap)")
    print()
    print(f"  Mean unique per utt:            {diag['mean_unique']:.1f}")
    print(f"  Median unique per utt:          {diag['median_unique']:.1f}")
    print(f"  Utts where N-best beats greedy: {diag['utts_with_better_than_greedy_in_nbest']}/{diag['n_utterances']} "
          f"({diag['utts_with_better_than_greedy_in_nbest']/diag['n_utterances']*100:.1f}%)")
    print(f"  MBR picks oracle:               {diag['mbr_picks_oracle_count']}/{diag['n_utterances']} "
          f"({diag['mbr_picks_oracle_pct']:.1f}%)")
    print(f"  Oracle rank in MBR (median):    {diag['oracle_rank_in_mbr_median']:.0f}")
    print(f"  Oracle rank in MBR (mean):      {diag['oracle_rank_in_mbr_mean']:.1f}")

    summary = {
        "input_jsonl": str(args.input_jsonl),
        "n_utterances": len(records),
        "wer_greedy": wer_g,
        "wer_oracle": wer_o,
        "best_mbr_cer_pll": best_cer,
        "best_mbr_wer_pll": best_wer_util,
        "tau_sweep_results": sweep_results,
        "diagnostics": diag,
    }
    out_json = args.output_dir / "mbr_eval_results.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {out_json}")

    print(f"\n{'='*60}")
    print("HEADLINE")
    print(f"{'='*60}")
    print(f"  Greedy:                       {wer_g*100:.3f}%")
    print(f"  Best MBR-CER+PLL (tau={best_cer['tau']}): {best_cer['wer']*100:.3f}%")
    if best_wer_util:
        print(f"  Best MBR-WER+PLL (tau={best_wer_util['tau']}): {best_wer_util['wer']*100:.3f}%")
    print(f"  Oracle:                       {wer_o*100:.3f}%")
    print(f"  Selection error (gap to oracle): "
          f"{diag['selection_residual_pp']:.2f}pp "
          f"({diag['selection_pct_of_gap']:.1f}% of total gap)")


if __name__ == "__main__":
    main()

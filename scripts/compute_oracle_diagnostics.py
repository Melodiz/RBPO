#!/usr/bin/env python3
"""Compute oracle WER, calibration, and error diagnostics from N-best JSONL.

Input: JSONL from generate_nbest.py, one line per utterance:
  {"utt_id": "...", "ref": "...", "nbest": [{"hyp": "...", "score": float}, ...]}

Output: JSON with all metrics + printed summary table.

Usage:
    python scripts/compute_oracle_diagnostics.py \
        --nbest /path/to/nbest.jsonl \
        --output /path/to/diagnostics.json \
        [--bootstrap-B 1000] \
        [--reference-json /path/to/reference.json]
"""

import argparse
import json
import time
from pathlib import Path

import editdistance
import numpy as np
from scipy.stats import spearmanr


def load_nbest(path: Path):
    """Load N-best JSONL, normalizing to canonical format.

    Handles both:
      - New format: {"utt_id", "ref", "nbest": [{"hyp", "score"}, ...]}
      - E11 format: {"utt_id", "ref_text", "candidates": [{"text", "ctc_log_prob"}, ...]}
    """
    records = []
    with open(path) as f:
        for line in f:
            raw = json.loads(line)
            if "nbest" in raw:
                records.append(raw)
            else:
                rec = {
                    "utt_id": raw["utt_id"],
                    "ref": raw.get("ref", raw.get("ref_text", "")),
                    "nbest": [
                        {"hyp": c.get("hyp", c.get("text", "")),
                         "score": c.get("score", c.get("ctc_log_prob", 0.0))}
                        for c in raw.get("nbest", raw.get("candidates", []))
                    ],
                }
                records.append(rec)
    return records


def compute_error_decomposition(ref_words, hyp_words):
    """Alignment-based sub/ins/del counts via DP backtrace."""
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])

    sub, ins, del_ = 0, 0, 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            sub += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            del_ += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ins += 1
            j -= 1
        else:
            break
    return sub, ins, del_


def main():
    parser = argparse.ArgumentParser(
        description="Compute oracle WER and diagnostics from N-best JSONL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nbest", type=Path, required=True,
                        help="N-best JSONL from generate_nbest.py")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSON path for all metrics")
    parser.add_argument("--bootstrap-B", type=int, default=1000,
                        help="Bootstrap iterations for Spearman CI")
    parser.add_argument("--reference-json", type=Path, default=None,
                        help="Reference diagnostics JSON for side-by-side comparison")
    args = parser.parse_args()

    print("=" * 70)
    print("compute_oracle_diagnostics.py")
    print("=" * 70)
    print(f"  nbest:        {args.nbest}")
    print(f"  output:       {args.output}")
    print(f"  bootstrap_B:  {args.bootstrap_B}")
    if args.reference_json:
        print(f"  reference:    {args.reference_json}")
    print()

    records = load_nbest(args.nbest)
    n_utts = len(records)
    print(f"Loaded {n_utts} utterances")

    # Oracle + greedy WER
    total_greedy_edits = 0
    total_oracle_edits = 0
    total_ref_words = 0
    recoverable = 0
    greedy_optimal = 0

    # Calibration
    per_utt_rho = []
    recoverable_rho = []
    greedy_opt_rho = []
    log_prob_spreads = []
    cand_counts = []

    # Error decomposition (greedy)
    total_sub, total_ins, total_del = 0, 0, 0

    for rec in records:
        ref_words = rec["ref"].split()
        if not ref_words:
            continue

        n_ref = len(ref_words)
        total_ref_words += n_ref

        nbest = rec["nbest"]
        cand_counts.append(len(nbest))

        # Greedy = rank-0
        greedy_hyp = nbest[0]["hyp"]
        greedy_words = greedy_hyp.split()
        greedy_edits = editdistance.eval(greedy_words, ref_words)
        total_greedy_edits += greedy_edits

        # Error decomposition for greedy
        s, i, d = compute_error_decomposition(ref_words, greedy_words)
        total_sub += s
        total_ins += i
        total_del += d

        # Oracle = min WER across all candidates
        best_edits = greedy_edits
        for c in nbest:
            e = editdistance.eval(c["hyp"].split(), ref_words)
            if e < best_edits:
                best_edits = e
        total_oracle_edits += best_edits

        if best_edits < greedy_edits:
            recoverable += 1
        else:
            greedy_optimal += 1

        # Spearman rho (score vs WER)
        if len(nbest) >= 3:
            scores = [c["score"] for c in nbest]
            wers = [
                editdistance.eval(c["hyp"].split(), ref_words) / n_ref
                for c in nbest
            ]
            if len(set(wers)) >= 2 and len(set(scores)) >= 2:
                rho, _ = spearmanr(scores, wers)
                if not np.isnan(rho):
                    per_utt_rho.append(rho)
                    if best_edits < greedy_edits:
                        recoverable_rho.append(rho)
                    else:
                        greedy_opt_rho.append(rho)

            log_prob_spreads.append(max(scores) - min(scores))

    # Corpus-level WER
    greedy_wer = total_greedy_edits / max(1, total_ref_words)
    oracle_wer = total_oracle_edits / max(1, total_ref_words)
    abs_gap = greedy_wer - oracle_wer
    rel_gap = abs_gap / max(1e-9, greedy_wer) * 100

    # Error decomposition percentages
    total_errors = total_sub + total_ins + total_del
    sub_pct = total_sub / max(1, total_errors) * 100
    ins_pct = total_ins / max(1, total_errors) * 100
    del_pct = total_del / max(1, total_errors) * 100

    # Spearman bootstrap CI
    rho_arr = np.array(per_utt_rho)
    rng = np.random.default_rng(42)
    boot_means = []
    for _ in range(args.bootstrap_B):
        idx = rng.integers(0, len(rho_arr), size=len(rho_arr))
        boot_means.append(rho_arr[idx].mean())
    boot_means = np.sort(boot_means)
    ci_lo = boot_means[int(0.025 * args.bootstrap_B)]
    ci_hi = boot_means[int(0.975 * args.bootstrap_B)]

    mean_cands = np.mean(cand_counts)

    result = {
        "source": str(args.nbest),
        "num_utterances": n_utts,
        "total_ref_words": total_ref_words,
        "greedy_wer": greedy_wer,
        "oracle_wer": oracle_wer,
        "abs_gap_pp": abs_gap,
        "rel_gap_pct": rel_gap,
        "recoverable_count": recoverable,
        "recoverable_pct": recoverable / max(1, n_utts) * 100,
        "greedy_optimal_count": greedy_optimal,
        "greedy_optimal_pct": greedy_optimal / max(1, n_utts) * 100,
        "mean_unique_hyps": round(mean_cands, 1),
        "calibration": {
            "mean_rho": float(rho_arr.mean()),
            "median_rho": float(np.median(rho_arr)),
            "ci_95_lo": float(ci_lo),
            "ci_95_hi": float(ci_hi),
            "n_utts_with_rho": len(per_utt_rho),
            "recoverable_mean_rho": float(np.mean(recoverable_rho)) if recoverable_rho else None,
            "recoverable_n": len(recoverable_rho),
            "greedy_opt_mean_rho": float(np.mean(greedy_opt_rho)) if greedy_opt_rho else None,
            "greedy_opt_n": len(greedy_opt_rho),
            "mean_logprob_spread": float(np.mean(log_prob_spreads)) if log_prob_spreads else None,
        },
        "error_decomp": {
            "sub_pct": sub_pct,
            "ins_pct": ins_pct,
            "del_pct": del_pct,
            "total_errors": total_errors,
        },
    }

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print()
    print("=" * 50)
    print("  DIAGNOSTICS SUMMARY")
    print("=" * 50)
    print(f"  Utterances:       {n_utts}")
    print(f"  Ref words:        {total_ref_words}")
    print(f"  Mean candidates:  {mean_cands:.1f}")
    print(f"  Greedy WER:       {greedy_wer:.4%}")
    print(f"  Oracle WER:       {oracle_wer:.4%}")
    print(f"  Abs gap:          {abs_gap:.4%} pp")
    print(f"  Rel gap:          {rel_gap:.1f}%")
    print(f"  Recoverable:      {recoverable}/{n_utts} ({recoverable/max(1,n_utts):.1%})")
    print(f"  Greedy-optimal:   {greedy_optimal}/{n_utts} ({greedy_optimal/max(1,n_utts):.1%})")
    print(f"  Spearman rho:     {rho_arr.mean():.3f} "
          f"[95% CI: {ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"    Recoverable:    {np.mean(recoverable_rho):.3f} (n={len(recoverable_rho)})"
          if recoverable_rho else "    Recoverable:    n/a")
    print(f"    Greedy-opt:     {np.mean(greedy_opt_rho):.3f} (n={len(greedy_opt_rho)})"
          if greedy_opt_rho else "    Greedy-opt:     n/a")
    print(f"  Sub/Ins/Del:      {sub_pct:.0f}/{ins_pct:.0f}/{del_pct:.0f}")
    print(f"  Log-prob spread:  {np.mean(log_prob_spreads):.1f}" if log_prob_spreads else "")
    print()
    print(f"  Saved: {args.output}")

    # Side-by-side comparison
    if args.reference_json and args.reference_json.exists():
        ref = json.load(open(args.reference_json))
        print()
        print("=" * 60)
        print("  COMPARISON (this vs reference)")
        print("=" * 60)
        fmt = "  {:<20s} {:>12s} {:>12s}"
        print(fmt.format("Metric", "This", "Reference"))
        print("  " + "-" * 46)

        def _f(v, pct=False):
            if v is None:
                return "n/a"
            return f"{v:.2%}" if pct else f"{v:.3f}"

        print(fmt.format("Greedy WER",
                          _f(greedy_wer, True), _f(ref.get("greedy_wer"), True)))
        print(fmt.format("Oracle WER",
                          _f(oracle_wer, True), _f(ref.get("oracle_wer"), True)))
        print(fmt.format("Rel gap (%)",
                          f"{rel_gap:.1f}%", f"{ref.get('rel_gap_pct', 0):.1f}%"))
        print(fmt.format("Recoverable",
                          f"{recoverable}", f"{ref.get('recoverable_count', '?')}"))
        ref_cal = ref.get("calibration", {})
        print(fmt.format("Spearman rho",
                          f"{rho_arr.mean():.3f}", _f(ref_cal.get("mean_rho"))))
    print()


if __name__ == "__main__":
    main()

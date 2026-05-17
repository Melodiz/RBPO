#!/usr/bin/env python3
"""E13: Error Type Analysis on Test-Other G=128.

Mirrors E10 (dev-other) on test-other to confirm that substitution
dominance generalizes across splits.

Usage:
    python experiments/analysis/error_type_test_other.py \
        --data-dir rbpo/results \
        --output-dir results/error_analysis_test_other
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import editdistance
import jiwer
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.significance_tests import corpus_wer


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def select_mbr_pll(rec, tau):
    """MBR-CER selection with RoBERTa PLL weights at given tau."""
    cands = [c for c in rec["candidates"] if c["text"].strip()]
    if not cands:
        cands = rec["candidates"]
    n = len(cands)
    texts = [c["text"] for c in cands]
    log_scores = np.array([c["roberta_pll"] for c in cands])

    if math.isinf(tau):
        weights = np.ones(n) / n
    else:
        scaled = log_scores / tau
        scaled -= np.max(scaled)
        weights = np.exp(scaled)
        weights /= weights.sum()

    cer_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = editdistance.eval(list(texts[i]), list(texts[j]))
            denom = max(len(texts[i]), len(texts[j]), 1)
            cer_matrix[i, j] = d / denom
            cer_matrix[j, i] = cer_matrix[i, j]
    risk = cer_matrix @ weights
    return texts[int(np.argmin(risk))]


def compute_error_types(ref_text, hyp_text):
    """Compute word-level S/I/D counts using jiwer."""
    out = jiwer.process_words(ref_text, hyp_text)
    return {
        "subs": out.substitutions,
        "ins": out.insertions,
        "dels": out.deletions,
        "hits": out.hits,
        "total_errors": out.substitutions + out.insertions + out.deletions,
    }


def analyze_switches(records, tau=10.0):
    """Analyze switched utterances for MBR-CER+PLL at given tau."""
    results = []
    for i, rec in enumerate(records):
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(records)}")
        greedy_text = rec["candidates"][0]["text"]
        method_text = select_mbr_pll(rec, tau)

        if greedy_text == method_text:
            continue

        ref_text = rec["ref_text"]
        ref_words = ref_text.split()
        greedy_errors = compute_error_types(ref_text, greedy_text)
        method_errors = compute_error_types(ref_text, method_text)

        greedy_wer = greedy_errors["total_errors"] / max(len(ref_words), 1)
        method_wer = method_errors["total_errors"] / max(len(ref_words), 1)

        if method_wer < greedy_wer:
            outcome = "improve"
        elif method_wer > greedy_wer:
            outcome = "regress"
        else:
            outcome = "tie"

        results.append({
            "utt_id": rec["utt_id"],
            "ref_text": ref_text,
            "greedy_text": greedy_text,
            "method_text": method_text,
            "outcome": outcome,
            "ref_words": len(ref_words),
            "greedy_subs": greedy_errors["subs"],
            "greedy_ins": greedy_errors["ins"],
            "greedy_del": greedy_errors["dels"],
            "greedy_total": greedy_errors["total_errors"],
            "method_subs": method_errors["subs"],
            "method_ins": method_errors["ins"],
            "method_del": method_errors["dels"],
            "method_total": method_errors["total_errors"],
            "delta_subs": method_errors["subs"] - greedy_errors["subs"],
            "delta_ins": method_errors["ins"] - greedy_errors["ins"],
            "delta_del": method_errors["dels"] - greedy_errors["dels"],
            "delta_total": method_errors["total_errors"] - greedy_errors["total_errors"],
        })

    return results


def aggregate_by_outcome(results):
    agg = {}
    for outcome in ["improve", "regress", "tie"]:
        subset = [r for r in results if r["outcome"] == outcome]
        if not subset:
            agg[outcome] = {"n_utts": 0, "total_sub_delta": 0,
                            "total_ins_delta": 0, "total_del_delta": 0, "total_delta": 0}
            continue
        agg[outcome] = {
            "n_utts": len(subset),
            "total_sub_delta": sum(r["delta_subs"] for r in subset),
            "total_ins_delta": sum(r["delta_ins"] for r in subset),
            "total_del_delta": sum(r["delta_del"] for r in subset),
            "total_delta": sum(r["delta_total"] for r in subset),
        }
    return agg


def compute_error_budget(results):
    return {
        "greedy_subs": sum(r["greedy_subs"] for r in results),
        "greedy_ins": sum(r["greedy_ins"] for r in results),
        "greedy_del": sum(r["greedy_del"] for r in results),
        "method_subs": sum(r["method_subs"] for r in results),
        "method_ins": sum(r["method_ins"] for r in results),
        "method_del": sum(r["method_del"] for r in results),
        "delta_subs": sum(r["delta_subs"] for r in results),
        "delta_ins": sum(r["delta_ins"] for r in results),
        "delta_del": sum(r["delta_del"] for r in results),
        "delta_total": sum(r["delta_total"] for r in results),
    }


def pick_examples(results, outcome, n=5):
    subset = [r for r in results if r["outcome"] == outcome]
    if outcome == "improve":
        subset.sort(key=lambda r: r["delta_total"])
    elif outcome == "regress":
        subset.sort(key=lambda r: -r["delta_total"])
    return subset[:n]


def main():
    parser = argparse.ArgumentParser(description="E13: Error Type Analysis (test-other G=128)")
    parser.add_argument("--data-dir", type=Path, default=Path("rbpo/results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/error_analysis_test_other"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("E13: Error Type Analysis  --  Test-Other G=128")
    print("=" * 70)
    t0 = time.time()

    data_path = args.data_dir / "test_other_g128" / "neural_lm_scores_test_other_G128.jsonl"
    print(f"\nLoading: {data_path}")
    records = load_jsonl(data_path)
    n_utts = len(records)
    print(f"  {n_utts} utterances")

    ref_words = [r["ref_text"].split() for r in records]
    greedy_words = [r["candidates"][0]["text"].split() for r in records]
    greedy_wer = corpus_wer(ref_words, greedy_words)
    print(f"  Greedy corpus WER: {greedy_wer*100:.4f}%")

    # Analyze switches
    print(f"\nAnalyzing MBR-CER+PLL tau=10 switches...")
    results = analyze_switches(records, tau=10.0)
    agg = aggregate_by_outcome(results)
    budget = compute_error_budget(results)

    n_switched = len(results)
    n_improve = agg["improve"]["n_utts"]
    n_regress = agg["regress"]["n_utts"]
    n_tie = agg["tie"]["n_utts"]
    print(f"\n  Switched: {n_switched}")
    print(f"  Improve: {n_improve} | Regress: {n_regress} | Tie: {n_tie}")
    print(f"  Error budget: deltaS={budget['delta_subs']:+d} deltaI={budget['delta_ins']:+d} "
          f"deltaD={budget['delta_del']:+d} = deltaTotal={budget['delta_total']:+d}")

    imp = agg["improve"]
    total_fixed = abs(imp["total_sub_delta"]) + abs(imp["total_ins_delta"]) + abs(imp["total_del_delta"])
    if total_fixed > 0:
        pct_subs = abs(imp["total_sub_delta"]) / total_fixed * 100
        pct_ins = abs(imp["total_ins_delta"]) / total_fixed * 100
        pct_del = abs(imp["total_del_delta"]) / total_fixed * 100
    else:
        pct_subs = pct_ins = pct_del = 0

    print(f"\n  Improving utterances  --  error reduction breakdown:")
    print(f"    Substitutions: {abs(imp['total_sub_delta'])} ({pct_subs:.1f}%)")
    print(f"    Insertions:    {abs(imp['total_ins_delta'])} ({pct_ins:.1f}%)")
    print(f"    Deletions:     {abs(imp['total_del_delta'])} ({pct_del:.1f}%)")

    # Verification
    print("\n--- Verification ---")
    assert n_utts == 2939, f"Expected 2939 utterances, got {n_utts}"
    print(f"  [PASS] Utterance count: {n_utts}")
    for r in results:
        assert r["delta_subs"] + r["delta_ins"] + r["delta_del"] == r["delta_total"]
    print(f"  [PASS] All per-utterance deltas consistent")
    assert n_improve > n_regress, f"Expected improve > regress, got {n_improve} vs {n_regress}"
    print(f"  [PASS] Improvements ({n_improve}) > Regressions ({n_regress})")
    assert pct_subs > 50, f"Expected substitution dominance >50%, got {pct_subs:.1f}%"
    print(f"  [PASS] Substitution dominance: {pct_subs:.1f}% > 50%")

    print("\n--- Writing outputs ---")

    # 1. JSON
    p = args.output_dir / "error_type_test_other.json"
    with open(p, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Wrote {p}")

    # 2. Summary CSV
    p = args.output_dir / "error_type_test_other_summary.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "outcome", "n_utts", "total_sub_delta", "total_ins_delta",
            "total_del_delta", "total_delta", "pct_subs", "pct_ins", "pct_del"
        ])
        w.writeheader()
        for outcome in ["improve", "regress", "tie"]:
            a = agg[outcome]
            tot = abs(a["total_sub_delta"]) + abs(a["total_ins_delta"]) + abs(a["total_del_delta"])
            ps = abs(a["total_sub_delta"]) / tot * 100 if tot > 0 else 0
            pi = abs(a["total_ins_delta"]) / tot * 100 if tot > 0 else 0
            pd = abs(a["total_del_delta"]) / tot * 100 if tot > 0 else 0
            w.writerow({
                "outcome": outcome,
                "n_utts": a["n_utts"],
                "total_sub_delta": a["total_sub_delta"],
                "total_ins_delta": a["total_ins_delta"],
                "total_del_delta": a["total_del_delta"],
                "total_delta": a["total_delta"],
                "pct_subs": f"{ps:.1f}",
                "pct_ins": f"{pi:.1f}",
                "pct_del": f"{pd:.1f}",
            })
    print(f"  Wrote {p}")

    # 3. Dev vs test comparison markdown
    p = args.output_dir / "dev_vs_test_error_comparison.md"
    lines = ["# Dev-Other vs Test-Other Error Type Comparison", ""]
    lines.append("## MBR-CER + PLL tau=10, G=128")
    lines.append("")
    lines.append("| Metric | Dev-Other (E10) | Test-Other (E13) |")
    lines.append("|--------|----------------:|-----------------:|")
    lines.append(f"| Utterances | 2864 | {n_utts} |")
    lines.append(f"| Switched |  --  | {n_switched} |")
    lines.append(f"| Improve |  --  | {n_improve} |")
    lines.append(f"| Regress |  --  | {n_regress} |")
    lines.append(f"| %Sub (improving) | ~60-68% (E10) | {pct_subs:.1f}% |")
    lines.append(f"| %Ins (improving) | ~15-20% (E10) | {pct_ins:.1f}% |")
    lines.append(f"| %Del (improving) | ~15-20% (E10) | {pct_del:.1f}% |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if pct_subs > 55:
        lines.append("**Confirmed:** Substitution dominance generalizes to test-other. "
                     "The information bottleneck is specifically a linguistic disambiguation bottleneck.")
    else:
        lines.append("**Partial:** Substitution fraction is lower on test-other. "
                     "The error profile may differ between splits.")
    lines.append("")
    lines.append("## Examples (Improvements)")
    lines.append("")
    for i, ex in enumerate(pick_examples(results, "improve", 3), 1):
        lines.append(f"### Example {i}")
        lines.append(f"- **Ref:** \"{ex['ref_text']}\"")
        lines.append(f"- **Greedy:** \"{ex['greedy_text']}\" "
                     f"({ex['greedy_subs']}S {ex['greedy_ins']}I {ex['greedy_del']}D)")
        lines.append(f"- **Method:** \"{ex['method_text']}\" "
                     f"({ex['method_subs']}S {ex['method_ins']}I {ex['method_del']}D)")
        lines.append(f"- delta: {ex['delta_subs']:+d}S {ex['delta_ins']:+d}I {ex['delta_del']:+d}D")
        lines.append("")
    lines.append("## Examples (Regressions)")
    lines.append("")
    for i, ex in enumerate(pick_examples(results, "regress", 2), 1):
        lines.append(f"### Regression {i}")
        lines.append(f"- **Ref:** \"{ex['ref_text']}\"")
        lines.append(f"- **Greedy:** \"{ex['greedy_text']}\" "
                     f"({ex['greedy_subs']}S {ex['greedy_ins']}I {ex['greedy_del']}D)")
        lines.append(f"- **Method:** \"{ex['method_text']}\" "
                     f"({ex['method_subs']}S {ex['method_ins']}I {ex['method_del']}D)")
        lines.append(f"- delta: {ex['delta_subs']:+d}S {ex['delta_ins']:+d}I {ex['delta_del']:+d}D")
        lines.append("")
    with open(p, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {p}")

    # 4. Stage report
    elapsed = time.time() - t0
    p = args.output_dir / "report_E13.md"
    lines = ["# Report E13: Error Type Analysis (Test-Other G=128)", ""]
    lines.append(f"**Status:** Complete. {n_utts} utterances, {elapsed:.0f}s on M2.")
    lines.append("")
    lines.append("## What Ran")
    lines.append("")
    lines.append(f"- Data: `test_other_g128/neural_lm_scores_test_other_G128.jsonl`")
    lines.append(f"- Method: MBR-CER + RoBERTa PLL tau=10, G=128")
    lines.append(f"- Analysis: word-level S/I/D decomposition of switched utterances")
    lines.append("")
    lines.append("## Key Results")
    lines.append("")
    lines.append(f"- Switched: {n_switched} utterances")
    lines.append(f"- Improve: {n_improve} | Regress: {n_regress} | Tie: {n_tie}")
    lines.append(f"- **Substitution dominance: {pct_subs:.1f}%** of improvement from sub fixes")
    lines.append(f"- Insertions: {pct_ins:.1f}% | Deletions: {pct_del:.1f}%")
    lines.append(f"- Error budget: deltaS={budget['delta_subs']:+d} deltaI={budget['delta_ins']:+d} "
                 f"deltaD={budget['delta_del']:+d}")
    lines.append("")
    lines.append("## Cross-Split Consistency")
    lines.append("")
    lines.append(f"E10 (dev-other) showed 60-68% substitution dominance. "
                 f"Test-other shows {pct_subs:.1f}%. ")
    if abs(pct_subs - 64) < 15:
        lines.append("**Consistent.** The linguistic disambiguation hypothesis holds across splits.")
    else:
        lines.append("The profiles differ somewhat but the overall pattern is similar.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("| File | Purpose |")
    lines.append("|------|---------|")
    lines.append("| `error_type_test_other.json` | Per-utterance breakdown |")
    lines.append("| `error_type_test_other_summary.csv` | Aggregate by outcome |")
    lines.append("| `dev_vs_test_error_comparison.md` | Side-by-side with E10 |")
    lines.append("| `report_E13.md` | This stage report |")
    with open(p, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {p}")

    print(f"\nDone. Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

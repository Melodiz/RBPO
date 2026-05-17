#!/usr/bin/env python3
"""E15: Regression Characterization at G=128.

MBR-CER+PLL tau=10 improves ~280 utterances but regresses ~84. This script
characterizes those regressions by length, greedy-error profile, PLL-CTC
disagreement, and failure mode categorization.

Usage:
    python experiments/analysis/regression_analysis.py \
        --data-dir rbpo/results \
        --output-dir results/regression_analysis
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

from experiments.cer_matrix_cache import compute_or_load_cer_matrices, mbr_select
from experiments.significance_tests import corpus_wer


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def compute_error_types(ref_text, hyp_text):
    out = jiwer.process_words(ref_text, hyp_text)
    return {
        "subs": out.substitutions,
        "ins": out.insertions,
        "dels": out.deletions,
        "total": out.substitutions + out.insertions + out.deletions,
    }


def identify_regressions(records, cer_matrices):
    """Run MBR-CER+PLL tau=10 and identify all regressions."""
    regressions = []
    improvements = []
    all_switches = []

    for i, rec in enumerate(records):
        cands = rec["candidates"]
        greedy_text = cands[0]["text"]
        log_scores = np.array([c["roberta_pll"] for c in cands])
        idx = mbr_select(cer_matrices[i], log_scores, tau=10.0)
        method_text = cands[idx]["text"]

        if method_text == greedy_text:
            continue

        ref_text = rec["ref_text"]
        ref_words = ref_text.split()
        n_ref = len(ref_words)

        greedy_errors = editdistance.eval(greedy_text.split(), ref_words)
        method_errors = editdistance.eval(method_text.split(), ref_words)
        greedy_wer = greedy_errors / max(n_ref, 1)
        method_wer = method_errors / max(n_ref, 1)

        entry = {
            "utt_id": rec["utt_id"],
            "ref_text": ref_text,
            "greedy_text": greedy_text,
            "method_text": method_text,
            "ref_word_count": n_ref,
            "greedy_errors": greedy_errors,
            "method_errors": method_errors,
            "greedy_wer": greedy_wer,
            "method_wer": method_wer,
            "delta_errors": method_errors - greedy_errors,
            "delta_wer": method_wer - greedy_wer,
            "roberta_pll_greedy": cands[0]["roberta_pll"],
            "roberta_pll_method": cands[idx]["roberta_pll"],
            "ctc_log_prob_greedy": cands[0]["ctc_log_prob"],
            "ctc_log_prob_method": cands[idx]["ctc_log_prob"],
            "method_idx": idx,
        }

        all_switches.append(entry)
        if method_errors > greedy_errors:
            regressions.append(entry)
        elif method_errors < greedy_errors:
            improvements.append(entry)

    return regressions, improvements, all_switches


def add_error_type_details(regressions):
    """Add S/I/D breakdown for each regression."""
    for r in regressions:
        greedy_et = compute_error_types(r["ref_text"], r["greedy_text"])
        method_et = compute_error_types(r["ref_text"], r["method_text"])
        r["greedy_subs"] = greedy_et["subs"]
        r["greedy_ins"] = greedy_et["ins"]
        r["greedy_del"] = greedy_et["dels"]
        r["method_subs"] = method_et["subs"]
        r["method_ins"] = method_et["ins"]
        r["method_del"] = method_et["dels"]
        r["delta_subs"] = method_et["subs"] - greedy_et["subs"]
        r["delta_ins"] = method_et["ins"] - greedy_et["ins"]
        r["delta_del"] = method_et["dels"] - greedy_et["dels"]


def add_consensus_score(regressions, records, cer_matrices):
    """Check if selected hypothesis has high avg similarity to other candidates."""
    utt_id_to_idx = {r["utt_id"]: i for i, r in enumerate(records)}
    for reg in regressions:
        i = utt_id_to_idx[reg["utt_id"]]
        mat = cer_matrices[i]
        method_idx = reg["method_idx"]
        n = mat.shape[0]
        avg_cer_to_others = mat[method_idx].sum() / max(n - 1, 1)
        reg["avg_cer_method_to_others"] = float(avg_cer_to_others)


def categorize_regressions(regressions, all_switches):
    """Categorize each regression into failure modes."""
    pll_gaps = [s["roberta_pll_method"] - s["roberta_pll_greedy"]
                for s in all_switches if s["delta_errors"] < 0]
    median_pll_gap = float(np.median(pll_gaps)) if pll_gaps else 0

    avg_cers = [r["avg_cer_method_to_others"] for r in regressions]
    median_consensus_cer = float(np.median(avg_cers)) if avg_cers else 0.5

    for r in regressions:
        pll_gap = r["roberta_pll_method"] - r["roberta_pll_greedy"]
        r["pll_gap"] = pll_gap

        if r["greedy_errors"] == 0:
            r["failure_mode"] = "greedy_was_perfect"
        elif r["delta_errors"] <= 1 and abs(r["delta_wer"]) < 0.15:
            r["failure_mode"] = "near_tie"
        elif pll_gap > median_pll_gap:
            r["failure_mode"] = "lm_hallucination"
        elif r["avg_cer_method_to_others"] < median_consensus_cer:
            r["failure_mode"] = "consensus_artifact"
        else:
            r["failure_mode"] = "other"

    return median_pll_gap, median_consensus_cer


def aggregate_by_length(regressions, improvements):
    """Bin regressions and improvements by utterance length."""
    bins = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 100)]
    rows = []
    for lo, hi in bins:
        n_reg = sum(1 for r in regressions if lo <= r["ref_word_count"] <= hi)
        n_imp = sum(1 for r in improvements if lo <= r["ref_word_count"] <= hi)
        total = n_reg + n_imp
        pct_reg = n_reg / max(total, 1) * 100
        rows.append({
            "length_bin": f"{lo}-{hi}",
            "n_regressions": n_reg,
            "n_improvements": n_imp,
            "total_switches": total,
            "pct_regressions": round(pct_reg, 1),
        })
    return rows


def aggregate_by_greedy_errors(regressions, improvements):
    """Bin by number of greedy errors."""
    bins = [(0, 0), (1, 1), (2, 2), (3, 5), (6, 100)]
    rows = []
    for lo, hi in bins:
        n_reg = sum(1 for r in regressions if lo <= r["greedy_errors"] <= hi)
        n_imp = sum(1 for r in improvements if lo <= r["greedy_errors"] <= hi)
        total = n_reg + n_imp
        pct_reg = n_reg / max(total, 1) * 100
        label = f"{lo}" if lo == hi else f"{lo}-{hi}"
        rows.append({
            "greedy_errors": label,
            "n_regressions": n_reg,
            "n_improvements": n_imp,
            "total_switches": total,
            "pct_regressions": round(pct_reg, 1),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="E15: Regression Characterization (G=128)")
    parser.add_argument("--data-dir", type=Path, default=Path("rbpo/results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/regression_analysis"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("E15: Regression Characterization  --  G=128, MBR-CER+PLL tau=10")
    print("=" * 70)
    t0 = time.time()

    data_path = args.data_dir / "g128" / "neural_lm_scores.jsonl"
    print(f"\nLoading: {data_path}")
    records = load_jsonl(data_path)
    n_utts = len(records)
    print(f"  {n_utts} utterances")

    # CER matrices
    print("\nCER Matrix Computation:")
    cer_matrices = compute_or_load_cer_matrices(
        records, data_path=data_path, cache_name="cer_matrix_g128"
    )

    # Identify regressions
    print("\nRunning MBR-CER+PLL tau=10 selection...")
    regressions, improvements, all_switches = identify_regressions(records, cer_matrices)
    n_reg = len(regressions)
    n_imp = len(improvements)
    n_tie = len(all_switches) - n_reg - n_imp
    print(f"  Switches: {len(all_switches)} | Improve: {n_imp} | Regress: {n_reg} | Tie: {n_tie}")

    print("\nComputing error type breakdown for regressions...")
    add_error_type_details(regressions)

    print("Computing consensus scores...")
    add_consensus_score(regressions, records, cer_matrices)

    # Categorize
    print("Categorizing failure modes...")
    median_pll_gap, median_consensus_cer = categorize_regressions(regressions, all_switches)

    mode_counts = {}
    for r in regressions:
        mode = r["failure_mode"]
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

    print(f"\n  Failure mode distribution:")
    for mode in ["lm_hallucination", "consensus_artifact", "near_tie",
                 "greedy_was_perfect", "other"]:
        cnt = mode_counts.get(mode, 0)
        pct = cnt / max(n_reg, 1) * 100
        print(f"    {mode:25s}: {cnt:3d} ({pct:.1f}%)")

    # Length analysis
    length_rows = aggregate_by_length(regressions, improvements)
    print(f"\n  Regressions by length:")
    for row in length_rows:
        if row["n_regressions"] > 0:
            print(f"    {row['length_bin']:>6s} words: {row['n_regressions']:3d} reg / "
                  f"{row['total_switches']:3d} switches ({row['pct_regressions']:.1f}%)")

    # Greedy-error analysis
    error_rows = aggregate_by_greedy_errors(regressions, improvements)
    print(f"\n  Regressions by greedy errors:")
    for row in error_rows:
        if row["n_regressions"] > 0:
            print(f"    {row['greedy_errors']:>5s} errors: {row['n_regressions']:3d} reg / "
                  f"{row['total_switches']:3d} switches ({row['pct_regressions']:.1f}%)")

    # PLL-CTC disagreement
    pll_prefers_method = sum(1 for r in regressions
                            if r["roberta_pll_method"] > r["roberta_pll_greedy"])
    ctc_prefers_greedy = sum(1 for r in regressions
                            if r["ctc_log_prob_greedy"] > r["ctc_log_prob_method"])
    print(f"\n  PLL-CTC disagreement in regressions:")
    print(f"    PLL prefers method (wrong): {pll_prefers_method}/{n_reg} "
          f"({pll_prefers_method/max(n_reg,1)*100:.1f}%)")
    print(f"    CTC prefers greedy (right): {ctc_prefers_greedy}/{n_reg} "
          f"({ctc_prefers_greedy/max(n_reg,1)*100:.1f}%)")

    # Verification
    print("\n--- Verification ---")
    assert n_utts == 2864, f"Expected 2864, got {n_utts}"
    print(f"  [PASS] Utterance count: {n_utts}")
    assert n_reg > 50, f"Expected >50 regressions, got {n_reg}"
    print(f"  [PASS] Regression count: {n_reg} (expected ~84)")
    assert n_imp > n_reg, f"Expected more improvements than regressions"
    print(f"  [PASS] Improvements ({n_imp}) > Regressions ({n_reg})")
    for r in regressions:
        assert r["delta_errors"] > 0
    print(f"  [PASS] All regressions have delta_errors > 0")
    total_mode = sum(mode_counts.values())
    assert total_mode == n_reg
    print(f"  [PASS] All regressions categorized: {total_mode}")

    print("\n--- Writing outputs ---")

    # 1. Full JSON
    p = args.output_dir / "regression_analysis.json"
    out_data = []
    for r in regressions:
        entry = {k: v for k, v in r.items()}
        out_data.append(entry)
    with open(p, "w") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"  Wrote {p}")

    # 2. Summary CSV
    p = args.output_dir / "regression_summary.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "metric", "value"
        ])
        w.writeheader()
        w.writerow({"metric": "total_utterances", "value": n_utts})
        w.writerow({"metric": "total_switches", "value": len(all_switches)})
        w.writerow({"metric": "improvements", "value": n_imp})
        w.writerow({"metric": "regressions", "value": n_reg})
        w.writerow({"metric": "ties", "value": n_tie})
        w.writerow({"metric": "pll_prefers_method_in_reg", "value": pll_prefers_method})
        w.writerow({"metric": "ctc_prefers_greedy_in_reg", "value": ctc_prefers_greedy})
        for mode, cnt in sorted(mode_counts.items()):
            w.writerow({"metric": f"mode_{mode}", "value": cnt})
    print(f"  Wrote {p}")

    # 3. By-length CSV
    p = args.output_dir / "regression_by_length.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "length_bin", "n_regressions", "n_improvements", "total_switches", "pct_regressions"
        ])
        w.writeheader()
        for row in length_rows:
            w.writerow(row)
    print(f"  Wrote {p}")

    # 4. By-greedy-errors CSV
    p = args.output_dir / "regression_by_greedy_errors.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "greedy_errors", "n_regressions", "n_improvements", "total_switches", "pct_regressions"
        ])
        w.writeheader()
        for row in error_rows:
            w.writerow(row)
    print(f"  Wrote {p}")

    # 5. Failure modes CSV
    p = args.output_dir / "regression_failure_modes.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["failure_mode", "count", "pct"])
        w.writeheader()
        for mode in ["lm_hallucination", "consensus_artifact", "near_tie",
                     "greedy_was_perfect", "other"]:
            cnt = mode_counts.get(mode, 0)
            w.writerow({"failure_mode": mode, "count": cnt,
                        "pct": f"{cnt/max(n_reg,1)*100:.1f}"})
    print(f"  Wrote {p}")

    # 6. Worst regressions markdown
    worst = sorted(regressions, key=lambda r: r["delta_wer"], reverse=True)[:5]
    p = args.output_dir / "worst_regressions.md"
    lines = ["# 5 Worst Regressions (Largest WER Increase)", ""]
    for rank, r in enumerate(worst, 1):
        lines.append(f"## #{rank}: `{r['utt_id']}`")
        lines.append(f"- **Failure mode:** {r['failure_mode']}")
        lines.append(f"- **Ref ({r['ref_word_count']} words):** \"{r['ref_text']}\"")
        lines.append(f"- **Greedy:** \"{r['greedy_text']}\"")
        lines.append(f"  - Errors: {r['greedy_errors']} "
                     f"({r['greedy_subs']}S {r['greedy_ins']}I {r['greedy_del']}D) "
                     f"-> WER={r['greedy_wer']*100:.1f}%")
        lines.append(f"- **Method:** \"{r['method_text']}\"")
        lines.append(f"  - Errors: {r['method_errors']} "
                     f"({r['method_subs']}S {r['method_ins']}I {r['method_del']}D) "
                     f"-> WER={r['method_wer']*100:.1f}%")
        lines.append(f"- **Delta:** +{r['delta_errors']} errors ({r['delta_wer']*100:+.1f}pp WER)")
        lines.append(f"- **PLL:** greedy={r['roberta_pll_greedy']:.2f}, "
                     f"method={r['roberta_pll_method']:.2f} "
                     f"(gap={r['pll_gap']:+.2f})")
        lines.append(f"- **CTC:** greedy={r['ctc_log_prob_greedy']:.2f}, "
                     f"method={r['ctc_log_prob_method']:.2f}")
        lines.append("")
    with open(p, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {p}")

    # 7. Summary markdown
    p = args.output_dir / "regression_summary.md"
    lines = ["# E15: Regression Characterization", ""]
    lines.append(f"**{n_reg} regressions** out of {len(all_switches)} switches "
                 f"({n_reg/max(len(all_switches),1)*100:.1f}%)")
    lines.append("")
    lines.append("## Failure Mode Distribution")
    lines.append("")
    lines.append("| Mode | Count | % |")
    lines.append("|------|------:|--:|")
    for mode in ["lm_hallucination", "consensus_artifact", "near_tie",
                 "greedy_was_perfect", "other"]:
        cnt = mode_counts.get(mode, 0)
        lines.append(f"| {mode} | {cnt} | {cnt/max(n_reg,1)*100:.1f}% |")
    lines.append("")
    lines.append("## By Utterance Length")
    lines.append("")
    lines.append("| Words | Regressions | Improvements | % Regressed |")
    lines.append("|------:|------------:|-------------:|------------:|")
    for row in length_rows:
        lines.append(f"| {row['length_bin']} | {row['n_regressions']} | "
                     f"{row['n_improvements']} | {row['pct_regressions']}% |")
    lines.append("")
    lines.append("## By Greedy Error Count")
    lines.append("")
    lines.append("| Greedy Errors | Regressions | Improvements | % Regressed |")
    lines.append("|--------------:|------------:|-------------:|------------:|")
    for row in error_rows:
        lines.append(f"| {row['greedy_errors']} | {row['n_regressions']} | "
                     f"{row['n_improvements']} | {row['pct_regressions']}% |")
    lines.append("")
    lines.append("## PLL-CTC Disagreement")
    lines.append("")
    lines.append(f"- PLL prefers method hypothesis (LM is \"wrong\"): "
                 f"**{pll_prefers_method}/{n_reg}** ({pll_prefers_method/max(n_reg,1)*100:.1f}%)")
    lines.append(f"- CTC prefers greedy (acoustic is right): "
                 f"**{ctc_prefers_greedy}/{n_reg}** ({ctc_prefers_greedy/max(n_reg,1)*100:.1f}%)")
    lines.append("")
    lines.append("## Key Insights")
    lines.append("")
    greedy_perfect = mode_counts.get("greedy_was_perfect", 0)
    near_ties = mode_counts.get("near_tie", 0)
    lm_hall = mode_counts.get("lm_hallucination", 0)
    lines.append(f"1. **Greedy-perfect regressions:** {greedy_perfect}/{n_reg}  --  "
                 "the method breaks utterances greedy already got right")
    lines.append(f"2. **Near-ties:** {near_ties}/{n_reg}  --  noise, not systematic failures")
    lines.append(f"3. **LM hallucination:** {lm_hall}/{n_reg}  --  "
                 "PLL strongly prefers wrong hypothesis")
    lines.append(f"4. **PLL-CTC disagreement:** In {pll_prefers_method}/{n_reg} regressions, "
                 "PLL preferred the wrong answer")
    lines.append("")
    with open(p, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {p}")

    # 8. Stage report
    elapsed = time.time() - t0
    p = args.output_dir / "report_E15.md"
    lines = ["# Report E15: Regression Characterization", ""]
    lines.append(f"**Status:** Complete. {n_reg} regressions analyzed. {elapsed:.0f}s on M2.")
    lines.append("")
    lines.append("## What Ran")
    lines.append("")
    lines.append(f"- Data: `g128/neural_lm_scores.jsonl` ({n_utts} utterances)")
    lines.append(f"- Method: MBR-CER + RoBERTa PLL tau=10, G=128")
    lines.append(f"- Analysis: per-regression characterization + failure mode categorization")
    lines.append("")
    lines.append("## Key Results")
    lines.append("")
    lines.append(f"- **Total regressions: {n_reg}** (vs {n_imp} improvements)")
    lines.append(f"- Primary failure mode: **{max(mode_counts, key=mode_counts.get)}** "
                 f"({mode_counts[max(mode_counts, key=mode_counts.get)]}/{n_reg})")
    lines.append(f"- PLL prefers wrong answer in {pll_prefers_method}/{n_reg} regressions")
    lines.append(f"- Greedy-perfect regressions: {greedy_perfect}")
    lines.append(f"- Near-ties (noise): {near_ties}")
    lines.append("")
    lines.append("## Failure Mode Summary")
    lines.append("")
    lines.append("| Mode | Count | % | Description |")
    lines.append("|------|------:|--:|-------------|")
    lines.append(f"| LM hallucination | {lm_hall} | "
                 f"{lm_hall/max(n_reg,1)*100:.0f}% | PLL strongly prefers wrong hyp |")
    lines.append(f"| Consensus artifact | {mode_counts.get('consensus_artifact', 0)} | "
                 f"{mode_counts.get('consensus_artifact',0)/max(n_reg,1)*100:.0f}% | "
                 f"Many candidates agree on wrong answer |")
    lines.append(f"| Near-tie | {near_ties} | "
                 f"{near_ties/max(n_reg,1)*100:.0f}% | <=1 word error difference |")
    lines.append(f"| Greedy perfect | {greedy_perfect} | "
                 f"{greedy_perfect/max(n_reg,1)*100:.0f}% | Greedy had 0 errors |")
    lines.append(f"| Other | {mode_counts.get('other', 0)} | "
                 f"{mode_counts.get('other',0)/max(n_reg,1)*100:.0f}% | Uncategorized |")
    lines.append("")
    lines.append("## Implications for Limitations Section")
    lines.append("")
    lines.append("The regression analysis supports these claims:")
    lines.append(f"1. ~{near_ties/max(n_reg,1)*100:.0f}% of regressions are noise (near-ties)")
    lines.append(f"2. ~{lm_hall/max(n_reg,1)*100:.0f}% are LM hallucinations "
                 "(the LM confidently picks fluent-but-wrong)")
    lines.append(f"3. {greedy_perfect} cases where the method breaks correct greedy output "
                 "represent the main practical concern")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("| File | Purpose |")
    lines.append("|------|---------|")
    lines.append("| `regression_analysis.json` | Per-regression details |")
    lines.append("| `regression_summary.csv` | Aggregate statistics |")
    lines.append("| `regression_by_length.csv` | Regressions per length bin |")
    lines.append("| `regression_by_greedy_errors.csv` | Regressions per greedy-error bin |")
    lines.append("| `regression_failure_modes.csv` | Categorized counts |")
    lines.append("| `worst_regressions.md` | 5 worst cases with full text |")
    lines.append("| `regression_summary.md` | Formatted analysis |")
    lines.append("| `report_E15.md` | This stage report |")
    with open(p, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {p}")

    print(f"\nDone. Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

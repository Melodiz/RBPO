#!/usr/bin/env python3
"""E10: Error Type Analysis  --  Where Does RoBERTa Win?

Decomposes WER improvements into substitution/insertion/deletion deltas
to verify the hypothesis that RoBERTa primarily fixes substitution errors
(linguistic disambiguation).

Analyzes three methods:
  (a) RoBERTa PLL interp alpha=0.7, G=16
  (b) MBR-CER + PLL tau=10, G=16
  (c) MBR-CER + PLL tau=10, G=128

Usage:
    python experiments/analysis/error_type_analysis.py \
        --data-dir rbpo/results \
        --output-dir results/error_analysis
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import editdistance
import jiwer
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def select_interp(rec, alpha, score_field):
    cands = [c for c in rec["candidates"] if c["text"].strip() != ""]
    if not cands:
        cands = rec["candidates"]
    scores = [alpha * c["ctc_log_prob"] + (1 - alpha) * c[score_field] for c in cands]
    return cands[int(np.argmax(scores))]["text"]


def select_mbr_pll(rec, tau):
    cands = [c for c in rec["candidates"] if c["text"].strip() != ""]
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
    subs = out.substitutions
    ins = out.insertions
    dels = out.deletions
    hits = out.hits
    return {"subs": subs, "ins": ins, "dels": dels, "hits": hits,
            "total_errors": subs + ins + dels}


def analyze_method(records, method_name, select_fn):
    """Analyze a single method: identify switches, compute error deltas."""
    results = []
    for rec in records:
        greedy_text = rec["candidates"][0]["text"]
        method_text = select_fn(rec)
        switched = greedy_text != method_text
        if not switched:
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
    """Aggregate error deltas by outcome category."""
    agg = {}
    for outcome in ["improve", "regress", "tie"]:
        subset = [r for r in results if r["outcome"] == outcome]
        if not subset:
            agg[outcome] = {
                "n_utts": 0, "total_sub_delta": 0, "total_ins_delta": 0,
                "total_del_delta": 0, "total_delta": 0,
            }
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
    """Corpus-level error budget across all switched utterances."""
    total_greedy_subs = sum(r["greedy_subs"] for r in results)
    total_greedy_ins = sum(r["greedy_ins"] for r in results)
    total_greedy_del = sum(r["greedy_del"] for r in results)
    total_method_subs = sum(r["method_subs"] for r in results)
    total_method_ins = sum(r["method_ins"] for r in results)
    total_method_del = sum(r["method_del"] for r in results)
    return {
        "greedy_subs": total_greedy_subs,
        "greedy_ins": total_greedy_ins,
        "greedy_del": total_greedy_del,
        "greedy_total": total_greedy_subs + total_greedy_ins + total_greedy_del,
        "method_subs": total_method_subs,
        "method_ins": total_method_ins,
        "method_del": total_method_del,
        "method_total": total_method_subs + total_method_ins + total_method_del,
        "delta_subs": total_method_subs - total_greedy_subs,
        "delta_ins": total_method_ins - total_greedy_ins,
        "delta_del": total_method_del - total_greedy_del,
        "delta_total": (total_method_subs + total_method_ins + total_method_del)
                       - (total_greedy_subs + total_greedy_ins + total_greedy_del),
    }


def pick_examples(results, outcome, n=5):
    """Pick the most illustrative examples for a given outcome."""
    subset = [r for r in results if r["outcome"] == outcome]
    if outcome == "improve":
        subset.sort(key=lambda r: r["delta_total"])
    elif outcome == "regress":
        subset.sort(key=lambda r: -r["delta_total"])
    else:
        subset.sort(key=lambda r: r["ref_words"], reverse=True)
    return subset[:n]


def format_example(r):
    lines = []
    lines.append(f"**Utterance:** `{r['utt_id']}`")
    lines.append(f"- Reference: \"{r['ref_text']}\"")
    lines.append(f"- Greedy:    \"{r['greedy_text']}\"  "
                 f"({r['greedy_subs']}S {r['greedy_ins']}I {r['greedy_del']}D)")
    lines.append(f"- Method:    \"{r['method_text']}\"  "
                 f"({r['method_subs']}S {r['method_ins']}I {r['method_del']}D)")
    lines.append(f"- Delta: {r['delta_subs']:+d}S {r['delta_ins']:+d}I {r['delta_del']:+d}D "
                 f"= {r['delta_total']:+d} total")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="E10: Error Type Analysis")
    parser.add_argument("--data-dir", type=Path, default=Path("rbpo/results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/error_analysis"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    g16_path = args.data_dir / "neural_lm_scores.jsonl"
    g128_path = args.data_dir / "g128" / "neural_lm_scores.jsonl"

    print("Loading G=16 scored data...")
    g16_records = load_jsonl(g16_path)
    print(f"  {len(g16_records)} utterances, {sum(r['num_candidates'] for r in g16_records)} candidates")

    print("Loading G=128 scored data...")
    g128_records = load_jsonl(g128_path)
    print(f"  {len(g128_records)} utterances, {sum(r['num_candidates'] for r in g128_records)} candidates")

    # Define methods
    methods = {
        "RoBERTa PLL interp alpha=0.7 (G=16)": (
            g16_records, lambda rec: select_interp(rec, 0.7, "roberta_pll")
        ),
        "MBR-CER + PLL tau=10 (G=16)": (
            g16_records, lambda rec: select_mbr_pll(rec, 10.0)
        ),
        "MBR-CER + PLL tau=10 (G=128)": (
            g128_records, lambda rec: select_mbr_pll(rec, 10.0)
        ),
    }

    all_results = {}
    all_aggregates = {}
    all_budgets = {}

    for method_name, (records, select_fn) in methods.items():
        print(f"\nAnalyzing: {method_name}")
        results = analyze_method(records, method_name, select_fn)
        agg = aggregate_by_outcome(results)
        budget = compute_error_budget(results)

        all_results[method_name] = results
        all_aggregates[method_name] = agg
        all_budgets[method_name] = budget

        n_switched = len(results)
        n_improve = agg["improve"]["n_utts"]
        n_regress = agg["regress"]["n_utts"]
        n_tie = agg["tie"]["n_utts"]
        print(f"  Switched: {n_switched} | Improve: {n_improve} | "
              f"Regress: {n_regress} | Tie: {n_tie}")
        print(f"  Error budget (switched utts): "
              f"deltaS={budget['delta_subs']:+d} deltaI={budget['delta_ins']:+d} "
              f"deltaD={budget['delta_del']:+d} = deltaTotal={budget['delta_total']:+d}")

    print("\n--- Writing outputs ---")
    write_json(args.output_dir, all_results)
    write_csv(args.output_dir, all_aggregates)
    write_summary_md(args.output_dir, all_results, all_aggregates, all_budgets)
    write_budget_md(args.output_dir, all_budgets)
    write_examples_md(args.output_dir, all_results)

    # Verification
    print("\n--- Verification ---")
    verify(g16_records, all_results)

    print("\nDone.")


def verify(g16_records, all_results):
    """Verify switched counts and error totals."""
    # Check RoBERTa alpha=0.7 G=16 switched count
    rob_results = all_results["RoBERTa PLL interp alpha=0.7 (G=16)"]
    n_switched = len(rob_results)
    n_improve = sum(1 for r in rob_results if r["outcome"] == "improve")
    # The spec says 268 switched with 100 improve  --  but let's just report what we get
    print(f"  RoBERTa alpha=0.7 G=16: {n_switched} switched, {n_improve} improve")

    # Check greedy total errors ~ 3064 (6.02% x 50,948)
    total_ref = sum(len(r["ref_text"].split()) for r in g16_records)
    greedy_errors = 0
    for rec in g16_records:
        ref = rec["ref_text"]
        hyp = rec["candidates"][0]["text"]
        greedy_errors += editdistance.eval(ref.split(), hyp.split())
    expected = round(0.0602 * total_ref)
    print(f"  Total ref words: {total_ref} (expected ~50,948)")
    print(f"  Total greedy errors: {greedy_errors} (expected ~{expected})")

    # Check delta consistency: sum of per-type deltas = total delta
    for method_name, results in all_results.items():
        for r in results:
            assert r["delta_subs"] + r["delta_ins"] + r["delta_del"] == r["delta_total"], \
                f"Delta mismatch in {r['utt_id']}"
    print("  [PASS] All per-utterance delta_subs + delta_ins + delta_del == delta_total")


def write_json(output_dir, all_results):
    p = output_dir / "error_type_analysis.json"
    out = {}
    for method, results in all_results.items():
        out[method] = results
    with open(p, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  Wrote {p}")


def write_csv(output_dir, all_aggregates):
    p = output_dir / "error_type_summary.csv"
    fields = ["method", "outcome", "n_utts", "total_sub_delta", "total_ins_delta",
              "total_del_delta", "pct_improvement_from_subs", "pct_improvement_from_ins",
              "pct_improvement_from_del"]
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for method, agg in all_aggregates.items():
            for outcome in ["improve", "regress", "tie"]:
                a = agg[outcome]
                total_improvement = abs(a["total_sub_delta"]) + abs(a["total_ins_delta"]) + abs(a["total_del_delta"])
                if total_improvement > 0 and outcome == "improve":
                    pct_s = abs(a["total_sub_delta"]) / total_improvement * 100
                    pct_i = abs(a["total_ins_delta"]) / total_improvement * 100
                    pct_d = abs(a["total_del_delta"]) / total_improvement * 100
                else:
                    pct_s = pct_i = pct_d = 0
                w.writerow({
                    "method": method,
                    "outcome": outcome,
                    "n_utts": a["n_utts"],
                    "total_sub_delta": a["total_sub_delta"],
                    "total_ins_delta": a["total_ins_delta"],
                    "total_del_delta": a["total_del_delta"],
                    "pct_improvement_from_subs": f"{pct_s:.1f}",
                    "pct_improvement_from_ins": f"{pct_i:.1f}",
                    "pct_improvement_from_del": f"{pct_d:.1f}",
                })
    print(f"  Wrote {p}")


def write_summary_md(output_dir, all_results, all_aggregates, all_budgets):
    p = output_dir / "error_type_summary.md"
    lines = ["# E10: Error Type Analysis  --  Where Does RoBERTa Win?", ""]
    lines.append("## Summary")
    lines.append("")

    for method in all_results:
        results = all_results[method]
        agg = all_aggregates[method]
        budget = all_budgets[method]
        imp = agg["improve"]

        lines.append(f"### {method}")
        lines.append("")
        lines.append(f"Switched utterances: **{len(results)}** "
                     f"(improve: {imp['n_utts']}, regress: {agg['regress']['n_utts']}, "
                     f"tie: {agg['tie']['n_utts']})")
        lines.append("")

        # Error budget for improving utterances
        if imp["n_utts"] > 0:
            total_fixed = abs(imp["total_sub_delta"]) + abs(imp["total_ins_delta"]) + abs(imp["total_del_delta"])
            if total_fixed > 0:
                pct_s = abs(imp["total_sub_delta"]) / total_fixed * 100
                pct_i = abs(imp["total_ins_delta"]) / total_fixed * 100
                pct_d = abs(imp["total_del_delta"]) / total_fixed * 100
            else:
                pct_s = pct_i = pct_d = 0

            lines.append("**Improving utterances  --  error reduction breakdown:**")
            lines.append("")
            lines.append("| Error type | Errors fixed | % of improvement |")
            lines.append("|------------|------------:|-----------------:|")
            lines.append(f"| Substitutions | {abs(imp['total_sub_delta'])} | {pct_s:.1f}% |")
            lines.append(f"| Insertions | {abs(imp['total_ins_delta'])} | {pct_i:.1f}% |")
            lines.append(f"| Deletions | {abs(imp['total_del_delta'])} | {pct_d:.1f}% |")
            lines.append("")

        # Corpus-level budget (all switched)
        lines.append("**Corpus-level error budget (all switched utterances):**")
        lines.append("")
        lines.append("| Error type | Greedy | Method | delta |")
        lines.append("|------------|-------:|-------:|---:|")
        lines.append(f"| Substitutions | {budget['greedy_subs']} | {budget['method_subs']} | {budget['delta_subs']:+d} |")
        lines.append(f"| Insertions | {budget['greedy_ins']} | {budget['method_ins']} | {budget['delta_ins']:+d} |")
        lines.append(f"| Deletions | {budget['greedy_del']} | {budget['method_del']} | {budget['delta_del']:+d} |")
        lines.append(f"| **Total** | **{budget['greedy_total']}** | **{budget['method_total']}** | **{budget['delta_total']:+d}** |")
        lines.append("")

    # Thesis validation
    lines.append("## Hypothesis Validation")
    lines.append("")
    lines.append("**Hypothesis:** RoBERTa primarily fixes substitution errors (linguistic disambiguation).")
    lines.append("")

    for method in all_results:
        budget = all_budgets[method]
        total_delta = budget["delta_total"]
        if total_delta < 0:
            pct_from_subs = budget["delta_subs"] / total_delta * 100
            lines.append(f"- **{method}:** substitutions account for "
                         f"**{pct_from_subs:.0f}%** of net error reduction")
        else:
            lines.append(f"- **{method}:** net error increase (delta={total_delta:+d})")
    lines.append("")

    with open(p, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {p}")


def write_budget_md(output_dir, all_budgets):
    p = output_dir / "error_budget_comparison.md"
    lines = ["# Error Budget Comparison Across Methods", ""]
    lines.append("| Method | deltaSub | deltaIns | deltaDel | deltaTotal | %Sub | %Ins | %Del |")
    lines.append("|--------|-----:|-----:|-----:|-------:|-----:|-----:|-----:|")

    for method, budget in all_budgets.items():
        ds = budget["delta_subs"]
        di = budget["delta_ins"]
        dd = budget["delta_del"]
        dt = budget["delta_total"]
        # Contribution = how much of the net improvement each type provides
        # Negative delta = improvement. Show as % of |deltaTotal|.
        if dt != 0:
            ps = ds / dt * 100
            pi = di / dt * 100
            pd = dd / dt * 100
        else:
            ps = pi = pd = 0
        lines.append(f"| {method} | {ds:+d} | {di:+d} | {dd:+d} | {dt:+d} | "
                     f"{ps:.0f}% | {pi:.0f}% | {pd:.0f}% |")
    lines.append("")
    lines.append("**%Sub/Ins/Del** = fraction of net error change from each type. "
                 "Values >60% for substitutions confirm the linguistic disambiguation hypothesis.")
    lines.append("")
    with open(p, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {p}")


def write_examples_md(output_dir, all_results):
    p = output_dir / "examples_improve.md"
    lines = ["# Best Improvement Examples", ""]

    for method, results in all_results.items():
        lines.append(f"## {method}")
        lines.append("")
        examples = pick_examples(results, "improve", n=5)
        for i, ex in enumerate(examples, 1):
            lines.append(f"### Example {i}")
            lines.append("")
            lines.append(format_example(ex))
            lines.append("")

    # Also show regression examples
    lines.append("---")
    lines.append("")
    lines.append("# Regression Examples")
    lines.append("")
    for method, results in all_results.items():
        lines.append(f"## {method}")
        lines.append("")
        examples = pick_examples(results, "regress", n=3)
        for i, ex in enumerate(examples, 1):
            lines.append(f"### Regression {i}")
            lines.append("")
            lines.append(format_example(ex))
            lines.append("")

    with open(p, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {p}")


if __name__ == "__main__":
    main()

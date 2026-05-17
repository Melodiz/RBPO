#!/usr/bin/env python3
"""E1: Bootstrap confidence intervals for Spearman rho (scorer vs WER rank).

Computes 95% bootstrap CIs for correlations between scoring functions and
WER, both corpus-level and stratified by utterance length and error regime.

Usage (Colab):
    pip install editdistance numpy scipy

    python experiments/analysis/spearman_bootstrap.py \
        --data-dir /content/drive/MyDrive/rbpo_results \
        --output-dir results/significance \
        --n-bootstrap 10000
"""

import argparse
import csv
import json
import time
from pathlib import Path

import editdistance
import numpy as np
from scipy import stats

def load_jsonl(path: Path) -> list:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"  Loaded {len(records)} records from {path.name}")
    return records

def compute_wer(hyp: str, ref: str) -> float:
    ref_w = ref.split()
    hyp_w = hyp.split()
    if len(ref_w) == 0:
        return 0.0 if len(hyp_w) == 0 else 1.0
    return editdistance.eval(hyp_w, ref_w) / len(ref_w)

def per_utterance_spearman(records, score_fn):
    """Compute mean per-utterance Spearman rho(score, WER).

    score_fn: callable(candidate_dict) -> float
    Returns list of per-utterance rho values (NaN-filtered).
    """
    rhos = []
    for rec in records:
        cands = rec["candidates"]
        if len(cands) < 3:
            continue
        scores = [score_fn(c) for c in cands]
        wers = [c["wer"] for c in cands]
        if len(set(scores)) < 2 or len(set(wers)) < 2:
            continue
        rho, _ = stats.spearmanr(scores, wers)
        if not np.isnan(rho):
            rhos.append(rho)
    return rhos

def bootstrap_mean_ci(values, n_bootstrap=10000, seed=42, ci=0.95):
    """Bootstrap CI for the mean of a list of values."""
    values = np.array(values)
    n = len(values)
    rng = np.random.default_rng(seed)

    means = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        means[b] = values[idx].mean()

    alpha = (1 - ci) / 2
    ci_lower = float(np.percentile(means, alpha * 100))
    ci_upper = float(np.percentile(means, (1 - alpha) * 100))
    return {
        "mean": float(values.mean()),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_utterances": n,
        "n_bootstrap": n_bootstrap,
    }

def annotate_wers(records):
    """Add 'wer' field to each candidate if not present."""
    for rec in records:
        ref = rec["ref_text"]
        for cand in rec["candidates"]:
            if "wer" not in cand:
                cand["wer"] = compute_wer(cand["text"], ref)
        wers = [c["wer"] for c in rec["candidates"]]
        rec["greedy_wer"] = wers[0]
        rec["oracle_wer"] = min(wers)
        rec["is_recoverable"] = rec["oracle_wer"] < rec["greedy_wer"] - 1e-12
        rec["ref_word_count"] = len(ref.split())

def stratify_by_length_terciles(records):
    """Split records into short/medium/long by ref word count."""
    lengths = [rec["ref_word_count"] for rec in records]
    p33 = np.percentile(lengths, 33.33)
    p66 = np.percentile(lengths, 66.67)

    strata = {"short": [], "medium": [], "long": []}
    for rec in records:
        wc = rec["ref_word_count"]
        if wc <= p33:
            strata["short"].append(rec)
        elif wc <= p66:
            strata["medium"].append(rec)
        else:
            strata["long"].append(rec)

    print(f"  Length terciles: short<={p33:.0f}, medium<={p66:.0f}, long>{p66:.0f} words")
    for k, v in strata.items():
        print(f"    {k}: {len(v)} utterances")
    return strata

def stratify_by_error_regime(records):
    """Split into greedy-optimal (oracle==greedy) vs recoverable."""
    strata = {"greedy_optimal": [], "recoverable": []}
    for rec in records:
        if rec["is_recoverable"]:
            strata["recoverable"].append(rec)
        else:
            strata["greedy_optimal"].append(rec)

    for k, v in strata.items():
        print(f"    {k}: {len(v)} utterances ({len(v)/len(records)*100:.1f}%)")
    return strata

def run_spearman_analysis(records, n_bootstrap, seed):
    """Run all Spearman analyses. Returns results dict."""
    has_pll = "roberta_pll" in records[0]["candidates"][0]
    has_gpt2 = "gpt2_ll" in records[0]["candidates"][0]

    scorers = {
        "CTC log-prob": lambda c: c["ctc_log_prob"],
    }
    if has_pll:
        scorers["RoBERTa PLL"] = lambda c: c["roberta_pll"]
        scorers["Interpolated (alpha=0.6 CTC + 0.4 PLL)"] = (
            lambda c: 0.6 * c["ctc_log_prob"] + 0.4 * c["roberta_pll"]
        )
    if has_gpt2:
        scorers["GPT-2 LL"] = lambda c: c["gpt2_ll"]

    print("\n--- Corpus-level Spearman rho ---")
    corpus_results = {}
    for name, fn in scorers.items():
        rhos = per_utterance_spearman(records, fn)
        ci = bootstrap_mean_ci(rhos, n_bootstrap=n_bootstrap, seed=seed)
        corpus_results[name] = ci
        print(f"  {name:45s}  rho={ci['mean']:+.4f}  "
              f"95% CI=[{ci['ci_lower']:+.4f}, {ci['ci_upper']:+.4f}]  "
              f"N={ci['n_utterances']}")

    print("\n--- Stratified by utterance length ---")
    length_strata = stratify_by_length_terciles(records)
    stratified_length = {}
    for stratum_name, stratum_records in length_strata.items():
        stratified_length[stratum_name] = {}
        for scorer_name, fn in scorers.items():
            rhos = per_utterance_spearman(stratum_records, fn)
            if len(rhos) < 10:
                print(f"    {scorer_name} / {stratum_name}: too few utterances ({len(rhos)})")
                continue
            ci = bootstrap_mean_ci(rhos, n_bootstrap=n_bootstrap, seed=seed)
            stratified_length[stratum_name][scorer_name] = ci
            print(f"    {stratum_name:8s} | {scorer_name:45s}  rho={ci['mean']:+.4f}  "
                  f"CI=[{ci['ci_lower']:+.4f}, {ci['ci_upper']:+.4f}]  N={ci['n_utterances']}")

    print("\n--- Stratified by error regime ---")
    error_strata = stratify_by_error_regime(records)
    stratified_error = {}
    for stratum_name, stratum_records in error_strata.items():
        stratified_error[stratum_name] = {}
        for scorer_name, fn in scorers.items():
            rhos = per_utterance_spearman(stratum_records, fn)
            if len(rhos) < 10:
                print(f"    {scorer_name} / {stratum_name}: too few utterances ({len(rhos)})")
                continue
            ci = bootstrap_mean_ci(rhos, n_bootstrap=n_bootstrap, seed=seed)
            stratified_error[stratum_name][scorer_name] = ci
            print(f"    {stratum_name:16s} | {scorer_name:45s}  rho={ci['mean']:+.4f}  "
                  f"CI=[{ci['ci_lower']:+.4f}, {ci['ci_upper']:+.4f}]  N={ci['n_utterances']}")

    return {
        "corpus": corpus_results,
        "by_length": stratified_length,
        "by_error_regime": stratified_error,
    }

def write_json(all_results, n_bootstrap, seed, output_dir):
    output = {
        "metadata": {
            "n_bootstrap": n_bootstrap,
            "seed": seed,
            "date": time.strftime("%Y-%m-%d"),
            "metric": "per-utterance Spearman rho (score vs WER)",
        },
        "corpus": all_results["corpus"],
        "by_length": all_results["by_length"],
        "by_error_regime": all_results["by_error_regime"],
    }
    path = output_dir / "spearman_bootstrap.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Wrote {path}")

def write_csv(all_results, output_dir):
    path = output_dir / "spearman_stratified.csv"
    rows = []

    # Corpus-level
    for scorer, ci in all_results["corpus"].items():
        rows.append({
            "stratum_type": "corpus",
            "stratum": "all",
            "scorer": scorer,
            "rho": f"{ci['mean']:+.4f}",
            "ci_lower": f"{ci['ci_lower']:+.4f}",
            "ci_upper": f"{ci['ci_upper']:+.4f}",
            "n_utterances": ci["n_utterances"],
        })

    # By length
    for stratum, scorers in all_results["by_length"].items():
        for scorer, ci in scorers.items():
            rows.append({
                "stratum_type": "length",
                "stratum": stratum,
                "scorer": scorer,
                "rho": f"{ci['mean']:+.4f}",
                "ci_lower": f"{ci['ci_lower']:+.4f}",
                "ci_upper": f"{ci['ci_upper']:+.4f}",
                "n_utterances": ci["n_utterances"],
            })

    # By error regime
    for stratum, scorers in all_results["by_error_regime"].items():
        for scorer, ci in scorers.items():
            rows.append({
                "stratum_type": "error_regime",
                "stratum": stratum,
                "scorer": scorer,
                "rho": f"{ci['mean']:+.4f}",
                "ci_lower": f"{ci['ci_lower']:+.4f}",
                "ci_upper": f"{ci['ci_upper']:+.4f}",
                "n_utterances": ci["n_utterances"],
            })

    fields = ["stratum_type", "stratum", "scorer", "rho", "ci_lower", "ci_upper", "n_utterances"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {path}")

def write_markdown(all_results, output_dir):
    path = output_dir / "spearman_summary.md"
    lines = []
    lines.append("# Spearman rho Bootstrap Analysis")
    lines.append("")
    lines.append("Per-utterance Spearman rho(score, WER) with 95% bootstrap CIs.")
    lines.append("")

    # Corpus table
    lines.append("## Corpus-Level")
    lines.append("")
    lines.append("| Scorer | rho | 95% CI | N |")
    lines.append("|--------|---:|--------|---:|")
    for scorer, ci in all_results["corpus"].items():
        lines.append(f"| {scorer} | {ci['mean']:+.4f} | "
                     f"[{ci['ci_lower']:+.4f}, {ci['ci_upper']:+.4f}] | {ci['n_utterances']} |")

    # Length table
    lines.append("")
    lines.append("## By Utterance Length (Terciles)")
    lines.append("")
    lines.append("| Stratum | Scorer | rho | 95% CI | N |")
    lines.append("|---------|--------|---:|--------|---:|")
    for stratum, scorers in all_results["by_length"].items():
        for scorer, ci in scorers.items():
            lines.append(f"| {stratum} | {scorer} | {ci['mean']:+.4f} | "
                         f"[{ci['ci_lower']:+.4f}, {ci['ci_upper']:+.4f}] | {ci['n_utterances']} |")

    # Error regime table
    lines.append("")
    lines.append("## By Error Regime")
    lines.append("")
    lines.append("| Regime | Scorer | rho | 95% CI | N |")
    lines.append("|--------|--------|---:|--------|---:|")
    for stratum, scorers in all_results["by_error_regime"].items():
        for scorer, ci in scorers.items():
            lines.append(f"| {stratum} | {scorer} | {ci['mean']:+.4f} | "
                         f"[{ci['ci_lower']:+.4f}, {ci['ci_upper']:+.4f}] | {ci['n_utterances']} |")

    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {path}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Bootstrap CIs for Spearman rho (E1)"
    )
    parser.add_argument("--data-dir", type=Path,
                        default=Path("/content/drive/MyDrive/rbpo_results"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/significance"))
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("E1: Spearman rho Bootstrap Confidence Intervals")
    print("=" * 70)
    print(f"  Data dir:    {args.data_dir}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"  Bootstrap:   {args.n_bootstrap} samples, seed={args.seed}")

    print("\n--- Loading data ---")

    # Prefer neural_lm_scores.jsonl (has PLL + GPT-2 fields)
    neural_path = args.data_dir / "neural_lm_scores.jsonl"
    nbest_path = args.data_dir / "nbest_dev_other_G16.jsonl"

    if neural_path.exists():
        records = load_jsonl(neural_path)
        print("  Using neural_lm_scores.jsonl (includes RoBERTa PLL + GPT-2 LL)")
    else:
        records = load_jsonl(nbest_path)
        print("  WARNING: neural_lm_scores.jsonl not found. Only CTC rho available.")

    cand0 = records[0]["candidates"][0]
    print(f"  Candidate fields: {list(cand0.keys())}")
    print(f"  Has roberta_pll: {'roberta_pll' in cand0}")
    print(f"  Has gpt2_ll: {'gpt2_ll' in cand0}")

    print("\n  Annotating per-candidate WERs...")
    annotate_wers(records)
    print(f"  Done. {len(records)} utterances, "
          f"{sum(rec['is_recoverable'] for rec in records)} recoverable.")

    all_results = run_spearman_analysis(records, args.n_bootstrap, args.seed)

    print("\n--- Writing outputs ---")
    write_json(all_results, args.n_bootstrap, args.seed, args.output_dir)
    write_csv(all_results, args.output_dir)
    write_markdown(all_results, args.output_dir)

    print("\n" + "=" * 70)
    print("DONE. Outputs in:", args.output_dir)
    print("=" * 70)

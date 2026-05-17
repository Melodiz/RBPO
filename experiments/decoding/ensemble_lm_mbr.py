#!/usr/bin/env python3
"""E14: Ensemble RoBERTa + GPT-2 MBR Weights.

Combines RoBERTa PLL and GPT-2 LL as weighted MBR scoring to test whether
the ensemble pushes WER below the RoBERTa-only 5.53%.

Sweep: beta in {0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0} x tau in {7, 10, 15}
where combined_score = beta*roberta_pll + (1-beta)*gpt2_ll

Usage:
    python experiments/decoding/ensemble_lm_mbr.py \
        --data-dir rbpo/results \
        --output-dir results/ensemble_lm
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import editdistance
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.cer_matrix_cache import compute_or_load_cer_matrices, mbr_select
from experiments.significance_tests import paired_bootstrap_wer, corpus_wer

BETA_VALUES = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
TAU_VALUES = [7, 10, 15]


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def run_ensemble_grid(records, cer_matrices, betas, taus):
    """Run beta x tau grid and return WER for each combination."""
    ref_words = [r["ref_text"].split() for r in records]
    grid = {}

    for beta in betas:
        for tau in taus:
            hyps = []
            for i, rec in enumerate(records):
                cands = rec["candidates"]
                roberta = np.array([c["roberta_pll"] for c in cands])
                gpt2 = np.array([c["gpt2_ll"] for c in cands])
                combined = beta * roberta + (1 - beta) * gpt2
                idx = mbr_select(cer_matrices[i], combined, tau)
                hyps.append(cands[idx]["text"])

            hyp_words = [h.split() for h in hyps]
            wer = corpus_wer(ref_words, hyp_words)
            grid[(beta, tau)] = {"wer": wer, "hyps_words": hyp_words}
            print(f"  beta={beta:.1f} tau={tau:>2}: WER={wer*100:.4f}%")

    return grid, ref_words


def run_tiebreaker(records, cer_matrices, tau=10, top_k=5):
    """Two-stage: MBR-CER with RoBERTa selects top-K, GPT-2 re-ranks."""
    ref_words = [r["ref_text"].split() for r in records]
    hyps = []

    for i, rec in enumerate(records):
        cands = rec["candidates"]
        n = len(cands)
        roberta = np.array([c["roberta_pll"] for c in cands])
        gpt2 = np.array([c["gpt2_ll"] for c in cands])

        # Stage 1: MBR-CER risk with RoBERTa weights
        if math.isinf(tau):
            weights = np.ones(n) / n
        else:
            scaled = roberta / tau
            scaled -= np.max(scaled)
            weights = np.exp(scaled)
            weights /= weights.sum()
        risk = cer_matrices[i] @ weights

        # Stage 2: among top-K lowest risk, pick highest GPT-2
        k = min(top_k, n)
        top_indices = np.argsort(risk)[:k]
        best_idx = top_indices[np.argmax(gpt2[top_indices])]
        hyps.append(cands[best_idx]["text"])

    hyp_words = [h.split() for h in hyps]
    wer = corpus_wer(ref_words, hyp_words)
    return wer, hyp_words, ref_words


def main():
    parser = argparse.ArgumentParser(description="E14: Ensemble RoBERTa + GPT-2 MBR")
    parser.add_argument("--data-dir", type=Path, default=Path("rbpo/results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/ensemble_lm"))
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("E14: Ensemble RoBERTa + GPT-2 MBR Weights")
    print("=" * 70)
    t0 = time.time()

    data_path = args.data_dir / "g128" / "neural_lm_scores.jsonl"
    print(f"\nLoading: {data_path}")
    records = load_jsonl(data_path)
    n_utts = len(records)
    print(f"  {n_utts} utterances")

    c0 = records[0]["candidates"][0]
    assert "roberta_pll" in c0, "Missing roberta_pll"
    assert "gpt2_ll" in c0, "Missing gpt2_ll"
    print(f"  Both roberta_pll and gpt2_ll present")

    # CER matrices (shared cache)
    print("\nCER Matrix Computation:")
    cer_matrices = compute_or_load_cer_matrices(
        records, data_path=data_path, cache_name="cer_matrix_g128"
    )

    # Greedy baseline
    ref_words = [r["ref_text"].split() for r in records]
    greedy_words = [r["candidates"][0]["text"].split() for r in records]
    greedy_wer = corpus_wer(ref_words, greedy_words)
    print(f"\n  Greedy WER: {greedy_wer*100:.4f}%")

    # Pure RoBERTa baseline (beta=1.0, tau=10)
    print(f"\n--- Ensemble Grid: beta x tau ---")
    grid, ref_words = run_ensemble_grid(records, cer_matrices, BETA_VALUES, TAU_VALUES)

    pure_roberta_wer = grid[(1.0, 10)]["wer"]
    print(f"\n  Pure RoBERTa (beta=1.0, tau=10): {pure_roberta_wer*100:.4f}%")

    best_key = min(grid.keys(), key=lambda k: grid[k]["wer"])
    best_beta, best_tau = best_key
    best_wer = grid[best_key]["wer"]
    print(f"  Best ensemble: beta={best_beta:.1f}, tau={best_tau} -> WER={best_wer*100:.4f}%")
    print(f"  vs Pure RoBERTa: {(best_wer - pure_roberta_wer)*100:+.4f}pp")

    # Tiebreaker approach
    print(f"\n--- Tiebreaker: MBR-CER(RoBERTa) -> GPT-2 re-rank top-5 ---")
    tb_wer, tb_hyps, _ = run_tiebreaker(records, cer_matrices, tau=10, top_k=5)
    print(f"  Tiebreaker WER: {tb_wer*100:.4f}%")
    print(f"  vs Pure RoBERTa: {(tb_wer - pure_roberta_wer)*100:+.4f}pp")

    # Bootstrap: best ensemble vs greedy AND vs pure RoBERTa
    print(f"\n--- Bootstrap (B={args.n_bootstrap}) ---")
    boot_vs_greedy = paired_bootstrap_wer(
        ref_words, grid[best_key]["hyps_words"], greedy_words,
        n_bootstrap=args.n_bootstrap, seed=args.seed,
    )
    print(f"  Best ensemble vs greedy: delta={boot_vs_greedy['delta']*100:+.4f}pp, "
          f"p={boot_vs_greedy['p_value']:.4f}")

    boot_vs_roberta = paired_bootstrap_wer(
        ref_words, grid[best_key]["hyps_words"], grid[(1.0, 10)]["hyps_words"],
        n_bootstrap=args.n_bootstrap, seed=args.seed,
    )
    print(f"  Best ensemble vs pure RoBERTa: delta={boot_vs_roberta['delta']*100:+.4f}pp, "
          f"p={boot_vs_roberta['p_value']:.4f}")

    boot_tb_vs_roberta = paired_bootstrap_wer(
        ref_words, tb_hyps, grid[(1.0, 10)]["hyps_words"],
        n_bootstrap=args.n_bootstrap, seed=args.seed,
    )
    print(f"  Tiebreaker vs pure RoBERTa: delta={boot_tb_vs_roberta['delta']*100:+.4f}pp, "
          f"p={boot_tb_vs_roberta['p_value']:.4f}")

    # Verification
    print("\n--- Verification ---")
    assert n_utts == 2864, f"Expected 2864, got {n_utts}"
    print(f"  [PASS] Utterance count: {n_utts}")
    assert abs(pure_roberta_wer * 100 - 5.53) < 0.1
    print(f"  [PASS] Pure RoBERTa tau=10: {pure_roberta_wer*100:.4f}% ~ 5.53%")
    pure_gpt2_wer = grid[(0.0, 10)]["wer"]
    assert pure_gpt2_wer > pure_roberta_wer, "GPT-2 should be weaker than RoBERTa"
    print(f"  [PASS] Pure GPT-2 ({pure_gpt2_wer*100:.4f}%) > Pure RoBERTa ({pure_roberta_wer*100:.4f}%)")

    print("\n--- Writing outputs ---")

    # 1. JSON
    out_json = {
        "experiment": "E14_ensemble_lm_mbr",
        "n_utterances": n_utts,
        "greedy_wer": greedy_wer,
        "pure_roberta_wer": pure_roberta_wer,
        "pure_gpt2_wer": pure_gpt2_wer,
        "best_ensemble": {"beta": best_beta, "tau": best_tau, "wer": best_wer},
        "tiebreaker_wer": tb_wer,
        "grid": {
            f"beta={b:.1f}_tau={t}": {"wer": grid[(b, t)]["wer"]}
            for b, t in grid
        },
        "bootstrap": {
            "best_vs_greedy": {
                "delta_pp": boot_vs_greedy["delta"] * 100,
                "p_value": boot_vs_greedy["p_value"],
                "ci_lower": boot_vs_greedy["ci_lower"] * 100,
                "ci_upper": boot_vs_greedy["ci_upper"] * 100,
            },
            "best_vs_roberta": {
                "delta_pp": boot_vs_roberta["delta"] * 100,
                "p_value": boot_vs_roberta["p_value"],
                "ci_lower": boot_vs_roberta["ci_lower"] * 100,
                "ci_upper": boot_vs_roberta["ci_upper"] * 100,
            },
            "tiebreaker_vs_roberta": {
                "delta_pp": boot_tb_vs_roberta["delta"] * 100,
                "p_value": boot_tb_vs_roberta["p_value"],
                "ci_lower": boot_tb_vs_roberta["ci_lower"] * 100,
                "ci_upper": boot_tb_vs_roberta["ci_upper"] * 100,
            },
        },
    }
    p = args.output_dir / "ensemble_results.json"
    with open(p, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"  Wrote {p}")

    # 2. Grid CSV
    p = args.output_dir / "ensemble_grid.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["beta", "tau", "wer", "delta_vs_roberta_pp"])
        w.writeheader()
        for beta in BETA_VALUES:
            for tau in TAU_VALUES:
                wer = grid[(beta, tau)]["wer"]
                w.writerow({
                    "beta": f"{beta:.1f}",
                    "tau": tau,
                    "wer": f"{wer*100:.4f}",
                    "delta_vs_roberta_pp": f"{(wer - pure_roberta_wer)*100:+.4f}",
                })
    print(f"  Wrote {p}")

    # 3. Summary markdown
    p = args.output_dir / "ensemble_summary.md"
    lines = ["# E14: Ensemble RoBERTa + GPT-2 MBR Weights", ""]
    lines.append(f"**Best ensemble:** beta={best_beta:.1f}, tau={best_tau} -> WER={best_wer*100:.4f}%")
    lines.append(f"- Pure RoBERTa (beta=1.0, tau=10): {pure_roberta_wer*100:.4f}%")
    lines.append(f"- Pure GPT-2 (beta=0.0, tau=10): {pure_gpt2_wer*100:.4f}%")
    lines.append(f"- Ensemble gain vs RoBERTa: {(best_wer - pure_roberta_wer)*100:+.4f}pp "
                 f"(p={boot_vs_roberta['p_value']:.4f})")
    lines.append(f"- Tiebreaker (top-5 re-rank): {tb_wer*100:.4f}% "
                 f"({(tb_wer - pure_roberta_wer)*100:+.4f}pp vs RoBERTa)")
    lines.append("")
    lines.append("## Grid: WER (%) by beta x tau")
    lines.append("")
    header = "| beta \\ tau | " + " | ".join(str(t) for t in TAU_VALUES) + " |"
    sep = "|------:|" + "|".join("------:" for _ in TAU_VALUES) + "|"
    lines.append(header)
    lines.append(sep)
    for beta in BETA_VALUES:
        cells = [f"{grid[(beta, t)]['wer']*100:.4f}" for t in TAU_VALUES]
        marker = ""
        if beta == best_beta:
            marker = " <-"
        lines.append(f"| {beta:.1f}{marker} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"beta=1.0 = pure RoBERTa, beta=0.0 = pure GPT-2")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if best_wer < pure_roberta_wer - 0.0005:
        lines.append(f"The ensemble (beta={best_beta:.1f}) provides a small improvement over "
                     f"pure RoBERTa ({(pure_roberta_wer - best_wer)*100:.4f}pp). "
                     f"GPT-2 contributes complementary signal.")
        if boot_vs_roberta["p_value"] < 0.05:
            lines.append("This improvement is **statistically significant** (p<0.05).")
        else:
            lines.append("However, this improvement is **not statistically significant** "
                         f"(p={boot_vs_roberta['p_value']:.3f}).")
    else:
        lines.append("The ensemble does **not** improve over pure RoBERTa. "
                     "RoBERTa PLL already captures most accessible linguistic signal at G=128. "
                     "GPT-2 (rho=-0.361) is too weakly correlated to add value.")
    lines.append("")
    with open(p, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {p}")

    # 4. Stage report
    elapsed = time.time() - t0
    p = args.output_dir / "report_E14.md"
    lines = ["# Report E14: Ensemble RoBERTa + GPT-2 MBR Weights", ""]
    lines.append(f"**Status:** Complete. {len(BETA_VALUES)}x{len(TAU_VALUES)} grid + tiebreaker. "
                 f"{elapsed:.0f}s on M2.")
    lines.append("")
    lines.append("## What Ran")
    lines.append("")
    lines.append(f"- Data: `g128/neural_lm_scores.jsonl` ({n_utts} utterances)")
    lines.append(f"- Combined score: s = beta*roberta_pll + (1-beta)*gpt2_ll")
    lines.append(f"- Grid: beta in {{{', '.join(f'{b:.1f}' for b in BETA_VALUES)}}}, "
                 f"tau in {{{', '.join(map(str, TAU_VALUES))}}}")
    lines.append(f"- Tiebreaker: MBR-CER(RoBERTa tau=10) -> GPT-2 re-rank top-5")
    lines.append(f"- Bootstrap: B={args.n_bootstrap}")
    lines.append("")
    lines.append("## Key Results")
    lines.append("")
    lines.append(f"| Method | WER (%) | delta vs RoBERTa (pp) | p-value |")
    lines.append(f"|--------|--------:|-------------------:|--------:|")
    lines.append(f"| Greedy | {greedy_wer*100:.4f} |  --  |  --  |")
    lines.append(f"| Pure RoBERTa (beta=1.0, tau=10) | {pure_roberta_wer*100:.4f} | 0 |  --  |")
    lines.append(f"| Best ensemble (beta={best_beta:.1f}, tau={best_tau}) | {best_wer*100:.4f} | "
                 f"{(best_wer-pure_roberta_wer)*100:+.4f} | {boot_vs_roberta['p_value']:.4f} |")
    lines.append(f"| Pure GPT-2 (beta=0.0, tau=10) | {pure_gpt2_wer*100:.4f} | "
                 f"{(pure_gpt2_wer-pure_roberta_wer)*100:+.4f} |  --  |")
    lines.append(f"| Tiebreaker (top-5) | {tb_wer*100:.4f} | "
                 f"{(tb_wer-pure_roberta_wer)*100:+.4f} | {boot_tb_vs_roberta['p_value']:.4f} |")
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    if best_wer >= pure_roberta_wer - 0.0005:
        lines.append("Ensemble does not improve over pure RoBERTa. RoBERTa already captures "
                     "the accessible linguistic signal. This is informative: it means the "
                     "remaining errors are NOT addressable by simply combining a second LM.")
    else:
        improvement = (pure_roberta_wer - best_wer) * 100
        lines.append(f"Ensemble provides {improvement:.4f}pp improvement. GPT-2 adds "
                     "complementary signal that RoBERTa misses.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("| File | Purpose |")
    lines.append("|------|---------|")
    lines.append("| `ensemble_results.json` | Full grid + bootstrap |")
    lines.append("| `ensemble_grid.csv` | beta x tau -> WER tabular |")
    lines.append("| `ensemble_summary.md` | Formatted analysis |")
    lines.append("| `report_E14.md` | This stage report |")
    with open(p, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {p}")

    print(f"\nDone. Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

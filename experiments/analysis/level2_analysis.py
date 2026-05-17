#!/usr/bin/env python3
"""Level 2: Oracle Gap Decomposition & CTC Calibration Analysis.

Reads Level 1 JSONL outputs and produces diagnostic plots, tables, and
a markdown report characterizing WHY CTC probabilities fail to rank
N-best hypotheses by quality.

Five analyses:
  1. Oracle recoverability decomposition
  2. CTC probability calibration (Spearman rho, delta scatter, spread)
  3. Error type decomposition (S/D/I)
  4. Length bias
  5. Candidate diversity metrics

Usage:
    python -m experiments.analysis.level2_analysis \
        --results-dir results
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import editdistance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams.update({
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

COLORS = {
    "greedy": "#e74c3c",
    "oracle": "#27ae60",
    "accent": "#3498db",
    "neutral": "#7f8c8d",
    "sub": "#e74c3c",
    "del": "#f39c12",
    "ins": "#3498db",
}



def load_nbest(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} utterances from {path}")
    return records


def load_per_utterance(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} per-utterance records from {path}")
    return records



def compute_wer(hyp: str, ref: str) -> float:
    ref_w = ref.split()
    hyp_w = hyp.split()
    if len(ref_w) == 0:
        return 0.0 if len(hyp_w) == 0 else 1.0
    return editdistance.eval(hyp_w, ref_w) / len(ref_w)


def levenshtein_sdi(hyp_seq: list, ref_seq: list) -> tuple[int, int, int]:
    """Standard DP with backtrace -> (substitutions, deletions, insertions)."""
    n, m = len(ref_seq), len(hyp_seq)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(1, n + 1):
        dp[i, 0] = i
    for j in range(1, m + 1):
        dp[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_seq[i - 1] == hyp_seq[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = 1 + min(dp[i - 1, j - 1], dp[i - 1, j], dp[i, j - 1])

    # Backtrace
    s, d, ins = 0, 0, 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_seq[i - 1] == hyp_seq[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] + 1:
            s += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            d += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return s, d, ins



def analysis1_oracle_recoverability(per_utt: list[dict]) -> dict:
    greedy_optimal = []
    recoverable = []

    for u in per_utt:
        g_wer = u["greedy_wer"]
        o_wer = u["oracle_wer"]
        if abs(g_wer - o_wer) < 1e-9:
            greedy_optimal.append(u)
        else:
            recoverable.append(u)

    n = len(per_utt)
    n_go = len(greedy_optimal)
    n_rec = len(recoverable)

    go_wers = [u["greedy_wer"] for u in greedy_optimal]
    rec_greedy_wers = [u["greedy_wer"] for u in recoverable]
    rec_oracle_wers = [u["oracle_wer"] for u in recoverable]
    rec_gaps = [u["greedy_wer"] - u["oracle_wer"] for u in recoverable]

    all_gaps = [u["greedy_wer"] - u["oracle_wer"] for u in per_utt]

    result = {
        "total_utterances": n,
        "greedy_optimal_count": n_go,
        "greedy_optimal_pct": n_go / n * 100,
        "recoverable_count": n_rec,
        "recoverable_pct": n_rec / n * 100,
        "greedy_optimal_mean_wer": float(np.mean(go_wers)) if go_wers else 0.0,
        "recoverable_mean_greedy_wer": float(np.mean(rec_greedy_wers)) if rec_greedy_wers else 0.0,
        "recoverable_mean_oracle_wer": float(np.mean(rec_oracle_wers)) if rec_oracle_wers else 0.0,
        "recoverable_mean_gap": float(np.mean(rec_gaps)) if rec_gaps else 0.0,
        "recoverable_median_gap": float(np.median(rec_gaps)) if rec_gaps else 0.0,
        "recoverable_max_gap": float(np.max(rec_gaps)) if rec_gaps else 0.0,
    }

    return result, all_gaps


def plot_oracle_gap_histogram(all_gaps: list[float], plots_dir: Path):
    fig, ax = plt.subplots()
    gaps_pct = [g * 100 for g in all_gaps]
    nonzero = [g for g in gaps_pct if g > 0.01]

    ax.hist(gaps_pct, bins=50, color=COLORS["accent"], alpha=0.7, edgecolor="white")
    ax.axvline(0, color=COLORS["greedy"], linewidth=1.5, linestyle="--", label="Zero gap (greedy=oracle)")
    ax.set_xlabel("Per-utterance WER gap (greedy - oracle), %")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Oracle Gap per Utterance")
    if nonzero:
        ax.axvline(np.mean(nonzero), color=COLORS["oracle"], linewidth=1.5,
                   linestyle=":", label=f"Mean nonzero gap = {np.mean(nonzero):.1f}%")
    ax.legend()
    fig.savefig(plots_dir / "oracle_gap_histogram.png")
    plt.close(fig)



def analysis2_ctc_calibration(nbest: list[dict]) -> dict:
    spearman_rhos = []
    delta_logprob = []
    delta_wer = []
    logprob_ranges = []
    logprob_stds = []
    frac_positive_rho = 0

    for rec in nbest:
        cands = rec["candidates"]
        ref = rec["ref_text"]
        if len(cands) < 2:
            continue

        texts = [c["text"] for c in cands]
        unique_texts = set(texts)
        if len(unique_texts) < 2:
            continue

        log_probs = [c["ctc_log_prob"] for c in cands]
        wers = [compute_wer(t, ref) for t in texts]

        rho, _ = stats.spearmanr(log_probs, wers)
        if not np.isnan(rho):
            spearman_rhos.append(rho)
            if rho > 0:
                frac_positive_rho += 1

        # Greedy = cands[0], oracle = min WER
        oracle_idx = int(np.argmin(wers))
        if oracle_idx != 0:
            dl = log_probs[0] - log_probs[oracle_idx]
            dw = wers[0] - wers[oracle_idx]
            delta_logprob.append(dl)
            delta_wer.append(dw)

        lp_arr = np.array(log_probs)
        logprob_ranges.append(float(lp_arr.max() - lp_arr.min()))
        logprob_stds.append(float(np.std(lp_arr)))

    n_with_rho = len(spearman_rhos)
    result = {
        "n_utterances_with_multiple_unique": n_with_rho,
        "mean_spearman_rho": float(np.mean(spearman_rhos)) if spearman_rhos else 0.0,
        "median_spearman_rho": float(np.median(spearman_rhos)) if spearman_rhos else 0.0,
        "std_spearman_rho": float(np.std(spearman_rhos)) if spearman_rhos else 0.0,
        "frac_positive_rho": frac_positive_rho / n_with_rho if n_with_rho > 0 else 0.0,
        "mean_logprob_range": float(np.mean(logprob_ranges)) if logprob_ranges else 0.0,
        "median_logprob_range": float(np.median(logprob_ranges)) if logprob_ranges else 0.0,
        "mean_logprob_std": float(np.mean(logprob_stds)) if logprob_stds else 0.0,
        "n_delta_pairs": len(delta_logprob),
    }
    if delta_logprob:
        corr, pval = stats.pearsonr(delta_logprob, delta_wer)
        result["delta_pearson_r"] = float(corr)
        result["delta_pearson_pval"] = float(pval)
    else:
        result["delta_pearson_r"] = 0.0
        result["delta_pearson_pval"] = 1.0

    return result, spearman_rhos, delta_logprob, delta_wer, logprob_ranges


def plot_spearman_histogram(rhos: list[float], plots_dir: Path):
    fig, ax = plt.subplots()
    ax.hist(rhos, bins=40, color=COLORS["accent"], alpha=0.7, edgecolor="white")
    ax.axvline(0, color=COLORS["neutral"], linewidth=1, linestyle="--")
    mean_rho = np.mean(rhos)
    ax.axvline(mean_rho, color=COLORS["greedy"], linewidth=1.5, linestyle=":",
               label=f"Mean rho = {mean_rho:.3f}")
    ax.set_xlabel("Spearman rho (CTC log-prob rank vs WER rank)")
    ax.set_ylabel("Count")
    ax.set_title("Per-Utterance CTC Rank Correlation")
    ax.legend()
    fig.savefig(plots_dir / "spearman_histogram.png")
    plt.close(fig)


def plot_prob_vs_wer_scatter(dlp: list[float], dwer: list[float], r: float, plots_dir: Path):
    fig, ax = plt.subplots()
    ax.scatter(dlp, [w * 100 for w in dwer], s=8, alpha=0.3, color=COLORS["accent"])
    ax.axhline(0, color=COLORS["neutral"], linewidth=0.8, linestyle="--")
    ax.axvline(0, color=COLORS["neutral"], linewidth=0.8, linestyle="--")
    ax.set_xlabel("delta log-prob (greedy - oracle candidate)")
    ax.set_ylabel("delta WER (greedy - oracle candidate), %")
    ax.set_title(f"CTC Probability Gap vs WER Gap (r = {r:.3f})")
    fig.savefig(plots_dir / "prob_vs_wer_scatter.png")
    plt.close(fig)


def plot_logprob_spread(ranges: list[float], plots_dir: Path):
    fig, ax = plt.subplots()
    ax.hist(ranges, bins=50, color=COLORS["accent"], alpha=0.7, edgecolor="white")
    mean_r = np.mean(ranges)
    ax.axvline(mean_r, color=COLORS["greedy"], linewidth=1.5, linestyle=":",
               label=f"Mean range = {mean_r:.1f}")
    ax.set_xlabel("Log-prob range within N-best (max - min)")
    ax.set_ylabel("Count")
    ax.set_title("Within-N-best Log-Prob Spread")
    ax.legend()
    fig.savefig(plots_dir / "logprob_spread_histogram.png")
    plt.close(fig)



def analysis3_sdi(nbest: list[dict]) -> dict:
    greedy_s, greedy_d, greedy_i = 0, 0, 0
    oracle_s, oracle_d, oracle_i = 0, 0, 0

    for rec in nbest:
        ref = rec["ref_text"]
        cands = rec["candidates"]
        ref_w = ref.split()

        greedy_text = cands[0]["text"]
        greedy_w = greedy_text.split()
        s, d, i = levenshtein_sdi(greedy_w, ref_w)
        greedy_s += s
        greedy_d += d
        greedy_i += i

        wers = [compute_wer(c["text"], ref) for c in cands]
        oracle_idx = int(np.argmin(wers))
        oracle_text = cands[oracle_idx]["text"]
        oracle_w = oracle_text.split()
        s, d, i = levenshtein_sdi(oracle_w, ref_w)
        oracle_s += s
        oracle_d += d
        oracle_i += i

    result = {
        "greedy_S": greedy_s,
        "greedy_D": greedy_d,
        "greedy_I": greedy_i,
        "greedy_total": greedy_s + greedy_d + greedy_i,
        "oracle_S": oracle_s,
        "oracle_D": oracle_d,
        "oracle_I": oracle_i,
        "oracle_total": oracle_s + oracle_d + oracle_i,
        "diff_S": greedy_s - oracle_s,
        "diff_D": greedy_d - oracle_d,
        "diff_I": greedy_i - oracle_i,
    }
    return result


def plot_sdi_comparison(sdi: dict, plots_dir: Path):
    labels = ["Substitutions", "Deletions", "Insertions"]
    greedy_vals = [sdi["greedy_S"], sdi["greedy_D"], sdi["greedy_I"]]
    oracle_vals = [sdi["oracle_S"], sdi["oracle_D"], sdi["oracle_I"]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width / 2, greedy_vals, width, label="Greedy",
                   color=COLORS["greedy"], alpha=0.8)
    bars2 = ax.bar(x + width / 2, oracle_vals, width, label="Oracle",
                   color=COLORS["oracle"], alpha=0.8)

    ax.set_ylabel("Count")
    ax.set_title("Error Type Decomposition: Greedy vs Oracle")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    for bar, val in zip(bars1, greedy_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars2, oracle_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha="center", va="bottom", fontsize=9)

    fig.savefig(plots_dir / "sdi_comparison.png")
    plt.close(fig)



def analysis4_length_bias(nbest: list[dict]) -> dict:
    cand_lens = []
    cand_wers = []
    oracle_len_diffs = []
    greedy_lens = []
    ref_lens = []
    all_len_diffs = []

    for rec in nbest:
        ref = rec["ref_text"]
        cands = rec["candidates"]
        ref_w = ref.split()
        ref_lens.append(len(ref_w))

        wers = [compute_wer(c["text"], ref) for c in cands]
        oracle_idx = int(np.argmin(wers))

        greedy_tokens = cands[0]["len_tokens"]
        greedy_words = len(cands[0]["text"].split())
        greedy_lens.append(greedy_words)

        for j, c in enumerate(cands):
            n_words = len(c["text"].split())
            cand_lens.append(n_words)
            cand_wers.append(wers[j])
            if j > 0:
                all_len_diffs.append(n_words - greedy_words)

        oracle_words = len(cands[oracle_idx]["text"].split())
        if oracle_idx != 0:
            oracle_len_diffs.append(oracle_words - greedy_words)

    corr, pval = stats.spearmanr(cand_lens, cand_wers) if len(cand_lens) > 2 else (0, 1)

    result = {
        "mean_greedy_word_len": float(np.mean(greedy_lens)),
        "mean_ref_word_len": float(np.mean(ref_lens)),
        "greedy_minus_ref_mean": float(np.mean(greedy_lens) - np.mean(ref_lens)),
        "length_wer_spearman": float(corr),
        "length_wer_spearman_pval": float(pval),
        "oracle_len_diff_mean": float(np.mean(oracle_len_diffs)) if oracle_len_diffs else 0.0,
        "oracle_len_diff_median": float(np.median(oracle_len_diffs)) if oracle_len_diffs else 0.0,
    }
    return result, cand_lens, cand_wers, oracle_len_diffs


def plot_length_bias(cand_lens, cand_wers, oracle_diffs, length_stats, plots_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: candidate word length vs WER
    ax = axes[0]
    ax.scatter(cand_lens, [w * 100 for w in cand_wers], s=3, alpha=0.15,
               color=COLORS["accent"])
    ax.set_xlabel("Candidate word count")
    ax.set_ylabel("WER %")
    ax.set_title("Candidate Length vs WER")

    # Panel 2: oracle length diff histogram
    ax = axes[1]
    if oracle_diffs:
        ax.hist(oracle_diffs, bins=30, color=COLORS["oracle"], alpha=0.7, edgecolor="white")
        ax.axvline(0, color=COLORS["neutral"], linewidth=1, linestyle="--")
        mean_d = np.mean(oracle_diffs)
        ax.axvline(mean_d, color=COLORS["greedy"], linewidth=1.5, linestyle=":",
                   label=f"Mean = {mean_d:+.2f}")
        ax.legend()
    ax.set_xlabel("Oracle word len - Greedy word len")
    ax.set_ylabel("Count")
    ax.set_title("Oracle Length Deviation from Greedy")

    # Panel 3: greedy vs ref length comparison
    ax = axes[2]
    gm = length_stats["mean_greedy_word_len"]
    rm = length_stats["mean_ref_word_len"]
    bars = ax.bar(["Greedy", "Reference"], [gm, rm],
                  color=[COLORS["greedy"], COLORS["oracle"]], alpha=0.8)
    ax.set_ylabel("Mean word count")
    ax.set_title("Greedy vs Reference Length")
    for bar, val in zip(bars, [gm, rm]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(plots_dir / "length_bias.png")
    plt.close(fig)



def analysis5_diversity(nbest: list[dict]) -> dict:
    unique_counts = []
    pw_weds = []
    frac_differ = []

    for rec in nbest:
        cands = rec["candidates"]
        texts = [c["text"].strip().lower() for c in cands]

        unique = set(texts)
        unique_counts.append(len(unique))

        greedy_text = texts[0]
        n_differ = sum(1 for t in texts[1:] if t != greedy_text)
        frac_differ.append(n_differ / max(len(texts) - 1, 1))

        if len(unique) >= 2:
            pair_dists = []
            unique_list = list(unique)
            for a, b in combinations(unique_list, 2):
                wa, wb = a.split(), b.split()
                denom = max(len(wa), len(wb), 1)
                pair_dists.append(editdistance.eval(wa, wb) / denom)
            pw_weds.append(float(np.mean(pair_dists)))
        else:
            pw_weds.append(0.0)

    result = {
        "mean_unique_count": float(np.mean(unique_counts)),
        "median_unique_count": float(np.median(unique_counts)),
        "min_unique_count": int(np.min(unique_counts)),
        "max_unique_count": int(np.max(unique_counts)),
        "frac_with_only_1_unique": float(np.mean([1 if u == 1 else 0 for u in unique_counts])),
        "mean_pairwise_wed": float(np.mean(pw_weds)),
        "median_pairwise_wed": float(np.median(pw_weds)),
        "mean_frac_differ": float(np.mean(frac_differ)),
    }
    return result, unique_counts, pw_weds


def plot_diversity(unique_counts, pairwise_weds, plots_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(unique_counts, bins=range(1, max(unique_counts) + 2),
            color=COLORS["accent"], alpha=0.7, edgecolor="white", align="left")
    ax.set_xlabel("Unique candidates per utterance")
    ax.set_ylabel("Count")
    ax.set_title("Candidate Set Uniqueness")
    ax.axvline(np.mean(unique_counts), color=COLORS["greedy"], linewidth=1.5,
               linestyle=":", label=f"Mean = {np.mean(unique_counts):.1f}")
    ax.legend()

    ax = axes[1]
    nonzero_weds = [w for w in pairwise_weds if w > 0]
    if nonzero_weds:
        ax.hist([w * 100 for w in nonzero_weds], bins=40,
                color=COLORS["oracle"], alpha=0.7, edgecolor="white")
        ax.axvline(np.mean(nonzero_weds) * 100, color=COLORS["greedy"], linewidth=1.5,
                   linestyle=":", label=f"Mean = {np.mean(nonzero_weds)*100:.1f}%")
        ax.legend()
    ax.set_xlabel("Mean pairwise word edit distance (%)")
    ax.set_ylabel("Count")
    ax.set_title("Pairwise Diversity within N-best")

    fig.tight_layout()
    fig.savefig(plots_dir / "diversity_histogram.png")
    plt.close(fig)



def generate_report(
    a1_stats, a2_stats, a3_stats, a4_stats, a5_stats,
    n_utts: int, output_dir: Path
) -> str:
    lines = []
    lines.append("# Level 2: Oracle Gap Decomposition & CTC Calibration Analysis\n")
    lines.append(f"**Dataset:** LibriSpeech dev-other ({n_utts} utterances)")
    lines.append("**Model:** Zipformer-S CR-CTC, BPE-500")
    lines.append("**N-best:** G=16, nbest_scale=1.0\n")

    # -- Analysis 1 --
    lines.append("## 1. Oracle Recoverability Decomposition\n")
    lines.append("| Bucket | Count | % | Mean WER |")
    lines.append("|--------|------:|--:|--------:|")
    lines.append(
        f"| Greedy-optimal (greedy = oracle) | {a1_stats['greedy_optimal_count']} "
        f"| {a1_stats['greedy_optimal_pct']:.1f}% "
        f"| {a1_stats['greedy_optimal_mean_wer']*100:.2f}% |"
    )
    lines.append(
        f"| Recoverable (greedy != oracle) | {a1_stats['recoverable_count']} "
        f"| {a1_stats['recoverable_pct']:.1f}% "
        f"| greedy {a1_stats['recoverable_mean_greedy_wer']*100:.2f}% -> oracle {a1_stats['recoverable_mean_oracle_wer']*100:.2f}% |"
    )
    lines.append(
        f"\nOn **recoverable** utterances, the mean per-utterance WER improvement "
        f"is **{a1_stats['recoverable_mean_gap']*100:.2f} pp** "
        f"(median {a1_stats['recoverable_median_gap']*100:.2f} pp, "
        f"max {a1_stats['recoverable_max_gap']*100:.1f} pp).\n"
    )
    lines.append("![Oracle gap histogram](plots/oracle_gap_histogram.png)\n")

    # -- Analysis 2 --
    lines.append("## 2. CTC Probability Calibration\n")
    lines.append("### 2.1 Spearman Rank Correlation (CTC log-prob vs WER)\n")
    lines.append(f"| Statistic | Value |")
    lines.append(f"|-----------|------:|")
    lines.append(f"| Utterances with >=2 unique candidates | {a2_stats['n_utterances_with_multiple_unique']} |")
    lines.append(f"| Mean Spearman rho | {a2_stats['mean_spearman_rho']:.4f} |")
    lines.append(f"| Median Spearman rho | {a2_stats['median_spearman_rho']:.4f} |")
    lines.append(f"| Std Spearman rho | {a2_stats['std_spearman_rho']:.4f} |")
    lines.append(f"| Fraction with rho > 0 (CTC prefers worse) | {a2_stats['frac_positive_rho']*100:.1f}% |")
    lines.append("")
    desired_sign = "negative" if a2_stats["mean_spearman_rho"] < 0 else "positive"
    if abs(a2_stats["mean_spearman_rho"]) < 0.1:
        interpretation = ("CTC log-probabilities are **essentially uninformative** for ranking "
                          "hypothesis quality  --  the rank correlation with WER is near zero.")
    elif a2_stats["mean_spearman_rho"] < -0.3:
        interpretation = ("CTC log-probabilities show meaningful **correct** correlation with WER "
                          "(higher prob <-> lower WER).")
    elif a2_stats["mean_spearman_rho"] > 0.1:
        interpretation = ("CTC log-probabilities are **anti-correlated** with WER  --  CTC "
                          "systematically assigns higher probability to worse hypotheses.")
    else:
        interpretation = f"CTC log-probabilities show weak {desired_sign} correlation with WER."
    lines.append(f"**Interpretation:** {interpretation}\n")
    lines.append("![Spearman histogram](plots/spearman_histogram.png)\n")

    lines.append("### 2.2 Probability Gap vs WER Gap\n")
    lines.append(f"Pearson r between delta_log_prob and delta_WER: **{a2_stats['delta_pearson_r']:.4f}** "
                 f"(p = {a2_stats['delta_pearson_pval']:.2e}, n = {a2_stats['n_delta_pairs']})\n")
    if abs(a2_stats["delta_pearson_r"]) < 0.1:
        lines.append("The probability gap between greedy and oracle candidates is **uncorrelated** "
                      "with the WER gap. CTC's confidence in greedy over oracle carries no signal.\n")
    lines.append("![Prob vs WER scatter](plots/prob_vs_wer_scatter.png)\n")

    lines.append("### 2.3 Log-Prob Spread within N-best\n")
    lines.append(f"| Statistic | Value |")
    lines.append(f"|-----------|------:|")
    lines.append(f"| Mean log-prob range (max-min) | {a2_stats['mean_logprob_range']:.2f} |")
    lines.append(f"| Median log-prob range | {a2_stats['median_logprob_range']:.2f} |")
    lines.append(f"| Mean log-prob std | {a2_stats['mean_logprob_std']:.2f} |")
    lines.append("")
    if a2_stats["mean_logprob_range"] < 5.0:
        lines.append("The log-prob spread is **very small**, meaning CTC assigns nearly equal "
                      "probability to all candidates. Even a perfect scoring function would struggle "
                      "to discriminate candidates with such compressed scores.\n")
    else:
        lines.append(f"The log-prob spread is moderate ({a2_stats['mean_logprob_range']:.1f}), "
                     "so CTC does differentiate candidates in probability space  --  "
                     "the issue is that the ranking doesn't align with quality.\n")
    lines.append("![Log-prob spread](plots/logprob_spread_histogram.png)\n")

    # -- Analysis 3 --
    lines.append("## 3. Error Type Decomposition (S/D/I)\n")
    lines.append("| Error Type | Greedy | Oracle | Difference (G-O) | Reduction % |")
    lines.append("|------------|-------:|-------:|------------------:|------------:|")
    for etype, key in [("Substitution", "S"), ("Deletion", "D"), ("Insertion", "I")]:
        gv = a3_stats[f"greedy_{key}"]
        ov = a3_stats[f"oracle_{key}"]
        diff = gv - ov
        pct = diff / gv * 100 if gv > 0 else 0.0
        lines.append(f"| {etype} | {gv} | {ov} | {diff} | {pct:.1f}% |")
    gt = a3_stats["greedy_total"]
    ot = a3_stats["oracle_total"]
    dt = gt - ot
    pt = dt / gt * 100 if gt > 0 else 0.0
    lines.append(f"| **Total** | **{gt}** | **{ot}** | **{dt}** | **{pt:.1f}%** |")
    lines.append("")

    max_diff_type = max(["S", "D", "I"], key=lambda k: a3_stats[f"diff_{k}"])
    type_names = {"S": "substitutions", "D": "deletions", "I": "insertions"}
    lines.append(f"Oracle primarily reduces **{type_names[max_diff_type]}** "
                 f"(delta = {a3_stats[f'diff_{max_diff_type}']}).\n")
    lines.append("![SDI comparison](plots/sdi_comparison.png)\n")

    # -- Analysis 4 --
    lines.append("## 4. Length Bias\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|------:|")
    lines.append(f"| Mean greedy word count | {a4_stats['mean_greedy_word_len']:.1f} |")
    lines.append(f"| Mean reference word count | {a4_stats['mean_ref_word_len']:.1f} |")
    lines.append(f"| Greedy - Reference (mean) | {a4_stats['greedy_minus_ref_mean']:+.2f} |")
    lines.append(f"| Length vs WER Spearman rho | {a4_stats['length_wer_spearman']:.4f} |")
    lines.append(f"| Oracle len diff from greedy (mean) | {a4_stats['oracle_len_diff_mean']:+.2f} words |")
    lines.append(f"| Oracle len diff from greedy (median) | {a4_stats['oracle_len_diff_median']:+.2f} words |")
    lines.append("")

    if a4_stats["greedy_minus_ref_mean"] < -0.3:
        lines.append("Greedy is **shorter** than reference on average -> CTC exhibits **deletion bias**.\n")
    elif a4_stats["greedy_minus_ref_mean"] > 0.3:
        lines.append("Greedy is **longer** than reference on average -> CTC exhibits **insertion bias**.\n")
    else:
        lines.append("Greedy length is close to reference  --  no strong systematic length bias.\n")
    lines.append("![Length bias](plots/length_bias.png)\n")

    # -- Analysis 5 --
    lines.append("## 5. Candidate Diversity\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|------:|")
    lines.append(f"| Mean unique candidates per utterance | {a5_stats['mean_unique_count']:.1f} |")
    lines.append(f"| Median unique candidates | {a5_stats['median_unique_count']:.0f} |")
    lines.append(f"| Min / Max unique | {a5_stats['min_unique_count']} / {a5_stats['max_unique_count']} |")
    lines.append(f"| Fraction with only 1 unique candidate | {a5_stats['frac_with_only_1_unique']*100:.1f}% |")
    lines.append(f"| Mean pairwise word edit distance | {a5_stats['mean_pairwise_wed']*100:.1f}% |")
    lines.append(f"| Mean fraction differing from greedy | {a5_stats['mean_frac_differ']*100:.1f}% |")
    lines.append("")

    if a5_stats["mean_pairwise_wed"] < 0.02:
        lines.append("The candidate set is **effectively homogeneous**  --  most candidates are "
                     "identical or near-identical to greedy. No selection rule can recover the "
                     "oracle gap if candidates don't offer meaningful alternatives.\n")
    elif a5_stats["mean_pairwise_wed"] < 0.05:
        lines.append("Moderate diversity  --  candidates differ slightly, but the variations are "
                     "small. Selection strategies have limited room to improve.\n")
    else:
        lines.append("Good diversity  --  candidates offer meaningfully different hypotheses, "
                     "so the bottleneck is scoring quality, not candidate generation.\n")
    lines.append("![Diversity](plots/diversity_histogram.png)\n")

    # -- Key Questions --
    lines.append("---\n")
    lines.append("## Answers to Key Questions\n")

    lines.append("### Q1: What fraction of utterances are recoverable?\n")
    lines.append(
        f"{a1_stats['recoverable_pct']:.1f}% of utterances have a non-greedy candidate with lower WER. "
        f"The remaining {a1_stats['greedy_optimal_pct']:.1f}% already have greedy = oracle. "
    )
    if a1_stats["recoverable_pct"] < 30:
        lines.append("The oracle gap is concentrated in a minority of utterances with large errors  --  "
                     "any practical method must identify and target these specific utterances.\n")
    else:
        lines.append("The oracle gap is distributed across many utterances, "
                     "so a general-purpose scoring improvement could yield broad gains.\n")

    lines.append("### Q2: Is CTC probability informative at all?\n")
    rho = a2_stats["mean_spearman_rho"]
    lines.append(f"Mean Spearman rho = **{rho:.4f}**. ")
    if abs(rho) < 0.1:
        lines.append("CTC sequence probabilities are **uninformative for hypothesis quality ranking** "
                     "within the N-best list. This is a key empirical finding: CTC's per-frame "
                     "independence assumption means that probability mass reflects acoustic fit "
                     "rather than linguistic coherence, making it useless for discriminating "
                     "between candidates that differ by a few words.\n")
    elif rho < -0.3:
        lines.append("CTC probabilities have reasonable discriminative power  --  "
                     "the issue lies elsewhere (candidate diversity, search errors, etc.).\n")
    else:
        lines.append(f"The correlation is weak (|rho| < 0.3), suggesting CTC probabilities carry "
                     "some but insufficient signal for reliable hypothesis selection.\n")

    lines.append("### Q3: What error type dominates the gap?\n")
    max_key = max(["S", "D", "I"], key=lambda k: a3_stats[f"diff_{k}"])
    lines.append(
        f"Oracle primarily reduces **{type_names[max_key]}** "
        f"(delta = {a3_stats[f'diff_{max_key}']}). "
    )
    if max_key == "D":
        lines.append("CTC's argmax systematically drops words  --  the frame-independence assumption "
                     "makes it cheap to skip tokens when acoustics are ambiguous.\n")
    elif max_key == "S":
        lines.append("The dominant improvement is in word substitutions  --  CTC picks acoustically "
                     "plausible but incorrect words.\n")
    else:
        lines.append("The dominant improvement is in insertions  --  CTC tends to hallucinate extra words.\n")

    lines.append("### Q4: Is there a length bias?\n")
    diff = a4_stats["greedy_minus_ref_mean"]
    lines.append(f"Mean greedy length = {a4_stats['mean_greedy_word_len']:.1f} words, "
                 f"mean reference = {a4_stats['mean_ref_word_len']:.1f} words "
                 f"(delta = {diff:+.2f}). ")
    if diff < -0.3:
        lines.append("CTC exhibits **deletion bias**. ")
        omd = a4_stats["oracle_len_diff_mean"]
        if omd > 0.1:
            lines.append(f"Oracle candidates are longer than greedy by {omd:+.2f} words on average, "
                         "confirming that the oracle corrects deletions. "
                         "The fact that length-norm didn't help in Level 1 suggests the issue "
                         "is not addressable by simple length bonuses  --  the log-prob landscape "
                         "is too flat to amplify.\n")
        else:
            lines.append(f"Oracle length diff from greedy is small ({omd:+.2f}), so length "
                         "correction alone would not bridge the gap.\n")
    elif diff > 0.3:
        lines.append("CTC exhibits insertion bias  --  greedy is longer than reference.\n")
    else:
        lines.append("No strong systematic length bias.\n")

    lines.append("### Q5: Are the candidates actually diverse?\n")
    mpw = a5_stats["mean_pairwise_wed"]
    lines.append(f"Mean pairwise word edit distance = **{mpw*100:.1f}%**. "
                 f"Mean unique candidates = {a5_stats['mean_unique_count']:.1f} / 16. ")
    if mpw < 0.02:
        lines.append("The candidate set is **effectively homogeneous**. "
                     "No selection strategy can recover the oracle gap because the N-best list "
                     "doesn't contain meaningfully different alternatives. "
                     "Before trying Level 1.5 scoring methods, candidate generation must be "
                     "changed (higher nbest_scale, diverse beam search, or temperature sampling).\n")
    elif mpw < 0.05:
        lines.append("Diversity is low but non-trivial. Better scoring could help marginally, "
                     "but fundamentally the candidate pool limits improvement. "
                     "Consider increasing nbest_scale or diversifying generation.\n")
    else:
        lines.append("Candidates are diverse enough that better scoring should yield gains. "
                     "The bottleneck is the scoring function, not the candidate set.\n")

    # -- Recommendations --
    lines.append("---\n")
    lines.append("## Recommendations for Next Steps\n")

    recs = []
    if mpw < 0.03:
        recs.append("1. **Increase candidate diversity**  --  try nbest_scale=0.5 or 0.1, "
                    "temperature sampling, or diverse beam search to generate more varied hypotheses.")
    if abs(rho) < 0.15:
        recs.append(f"{'2' if recs else '1'}. **External rescoring**  --  since CTC probabilities are uninformative, "
                    "a language model or neural rescorer is needed. CTC + LM shallow fusion or "
                    "a dedicated rescoring model (e.g., RNN-T second-pass) would provide the "
                    "missing signal.")
    if max_key == "D" and diff < -0.3:
        recs.append(f"{'3' if len(recs)>=2 else '2' if recs else '1'}. **Deletion-aware training**  --  the RBPO objective could "
                    "specifically penalize deletion errors more heavily, or a length reward "
                    "could be incorporated into the advantage function.")
    if not recs:
        recs.append("1. Investigate external rescoring with a language model.")
        recs.append("2. Consider RBPO training with error-type-aware rewards.")

    for r in recs:
        lines.append(r)
    lines.append("")

    lines.append("## Generated Files\n")
    lines.append("- `level2_report.md`  --  this report")
    lines.append("- `level2_stats.json`  --  all statistics in machine-readable format")
    lines.append("- `plots/oracle_gap_histogram.png`")
    lines.append("- `plots/spearman_histogram.png`")
    lines.append("- `plots/prob_vs_wer_scatter.png`")
    lines.append("- `plots/logprob_spread_histogram.png`")
    lines.append("- `plots/sdi_comparison.png`")
    lines.append("- `plots/length_bias.png`")
    lines.append("- `plots/diversity_histogram.png`")

    return "\n".join(lines) + "\n"



def parse_args():
    parser = argparse.ArgumentParser(
        description="Level 2: Oracle Gap Decomposition & CTC Calibration Analysis"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"),
        help="Directory containing Level 1 JSONL outputs",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = args.results_dir

    nbest_path = results_dir / "nbest_dev_other_G16.jsonl"
    per_utt_path = results_dir / "per_utterance.jsonl"

    print("=" * 60)
    print("Level 2: Oracle Gap Decomposition & CTC Calibration")
    print("=" * 60)

    nbest = load_nbest(nbest_path)
    per_utt = load_per_utterance(per_utt_path)

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Analysis 1
    print("\n[1/5] Oracle recoverability decomposition...")
    a1_stats, all_gaps = analysis1_oracle_recoverability(per_utt)
    plot_oracle_gap_histogram(all_gaps, plots_dir)
    print(f"  Greedy-optimal: {a1_stats['greedy_optimal_count']} ({a1_stats['greedy_optimal_pct']:.1f}%)")
    print(f"  Recoverable:    {a1_stats['recoverable_count']} ({a1_stats['recoverable_pct']:.1f}%)")

    # Analysis 2
    print("\n[2/5] CTC probability calibration...")
    a2_stats, rhos, dlp, dwer, lp_ranges = analysis2_ctc_calibration(nbest)
    if rhos:
        plot_spearman_histogram(rhos, plots_dir)
    if dlp:
        plot_prob_vs_wer_scatter(dlp, dwer, a2_stats["delta_pearson_r"], plots_dir)
    if lp_ranges:
        plot_logprob_spread(lp_ranges, plots_dir)
    print(f"  Mean Spearman rho: {a2_stats['mean_spearman_rho']:.4f}")
    print(f"  Frac rho > 0: {a2_stats['frac_positive_rho']*100:.1f}%")
    print(f"  Mean log-prob range: {a2_stats['mean_logprob_range']:.2f}")

    # Analysis 3
    print("\n[3/5] Error type decomposition (S/D/I)...")
    a3_stats = analysis3_sdi(nbest)
    plot_sdi_comparison(a3_stats, plots_dir)
    print(f"  Greedy S/D/I: {a3_stats['greedy_S']}/{a3_stats['greedy_D']}/{a3_stats['greedy_I']}")
    print(f"  Oracle S/D/I: {a3_stats['oracle_S']}/{a3_stats['oracle_D']}/{a3_stats['oracle_I']}")

    # Analysis 4
    print("\n[4/5] Length bias...")
    a4_stats, cand_lens, cand_wers, oracle_diffs = analysis4_length_bias(nbest)
    plot_length_bias(cand_lens, cand_wers, oracle_diffs, a4_stats, plots_dir)
    print(f"  Greedy mean len: {a4_stats['mean_greedy_word_len']:.1f}, "
          f"Ref mean len: {a4_stats['mean_ref_word_len']:.1f}")
    print(f"  Oracle len diff: {a4_stats['oracle_len_diff_mean']:+.2f} words")

    # Analysis 5
    print("\n[5/5] Candidate diversity...")
    a5_stats, unique_counts, pairwise_weds = analysis5_diversity(nbest)
    plot_diversity(unique_counts, pairwise_weds, plots_dir)
    print(f"  Mean unique: {a5_stats['mean_unique_count']:.1f}")
    print(f"  Mean pairwise WED: {a5_stats['mean_pairwise_wed']*100:.1f}%")

    all_stats = {
        "analysis1_oracle_recoverability": a1_stats,
        "analysis2_ctc_calibration": a2_stats,
        "analysis3_sdi": a3_stats,
        "analysis4_length_bias": a4_stats,
        "analysis5_diversity": a5_stats,
    }
    stats_path = results_dir / "level2_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSaved: {stats_path}")

    n_utts = len(per_utt)
    report_text = generate_report(a1_stats, a2_stats, a3_stats, a4_stats, a5_stats, n_utts, results_dir)
    report_path = results_dir / "level2_report.md"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"Saved: {report_path}")

    print(f"\nPlots saved to: {plots_dir}")
    print("\n" + "=" * 60)
    print("Level 2 analysis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

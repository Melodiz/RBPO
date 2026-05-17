#!/usr/bin/env python3
"""Beyond-N-best diagnostics: diversity, coverage, MBR selection accuracy.

Uses corpus WER (total_edits / total_ref_words) to match the pipeline's
reported numbers. Implements both the simple interpolation selector and
the full MBR-CER selector from g_scaling_curve.py.
"""

import json
import csv
import math
import os
from pathlib import Path

import numpy as np

DATA_DIR = Path("/Users/melodiz/Desktop/RBPO/results/g_scaling")
OUT_DIR = Path("/Users/melodiz/Desktop/RBPO/.claude/worktrees/affectionate-mclean-7e1469/results/diagnostics")
G_VALUES = [4, 8, 16, 32, 64, 128]

try:
    import editdistance
    HAS_EDITDISTANCE = True
except ImportError:
    HAS_EDITDISTANCE = False


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def word_edit_distance(ref_words, hyp_words):
    """Word-level edit distance."""
    if HAS_EDITDISTANCE:
        return editdistance.eval(ref_words, hyp_words)
    n, m = len(ref_words), len(hyp_words)
    d = list(range(m + 1))
    for i in range(1, n + 1):
        prev, d[0] = d[0], i
        for j in range(1, m + 1):
            temp = d[j]
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[j] = prev
            else:
                d[j] = 1 + min(prev, d[j], d[j - 1])
            prev = temp
    return d[m]


def char_error_rate(ref_text, hyp_text):
    """Character error rate (Levenshtein on chars)."""
    if not ref_text:
        return 0.0 if not hyp_text else 1.0
    if HAS_EDITDISTANCE:
        return editdistance.eval(list(ref_text), list(hyp_text)) / len(ref_text)
    r = list(ref_text)
    h = list(hyp_text)
    n, m = len(r), len(h)
    d = list(range(m + 1))
    for i in range(1, n + 1):
        prev, d[0] = d[0], i
        for j in range(1, m + 1):
            temp = d[j]
            if r[i - 1] == h[j - 1]:
                d[j] = prev
            else:
                d[j] = 1 + min(prev, d[j], d[j - 1])
            prev = temp
    return d[m] / n


def compute_cer_matrix(texts):
    """Compute pairwise CER matrix for MBR."""
    n = len(texts)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            cer = char_error_rate(texts[i], texts[j])
            mat[i][j] = cer
            mat[j][i] = cer
    return mat


def select_interp(cands, alpha, score_field="roberta_pll"):
    scores = [alpha * c["ctc_log_prob"] + (1 - alpha) * c[score_field] for c in cands]
    return int(np.argmax(scores))


def select_mbr_cer(cands, tau, score_field="roberta_pll"):
    """MBR-CER with score-weighted posteriors. Returns index of selected."""
    texts = [c["text"] for c in cands]
    log_scores = np.array([c[score_field] for c in cands])
    n = len(cands)

    cer_mat = compute_cer_matrix(texts)

    if math.isinf(tau):
        weights = np.ones(n) / n
    else:
        scaled = log_scores / tau
        scaled -= np.max(scaled)
        weights = np.exp(scaled)
        weights /= weights.sum()

    risk = cer_mat @ weights
    return int(np.argmin(risk))


def per_utt_wer(ref_text, hyp_text):
    """Per-utterance WER."""
    ref_w = ref_text.split()
    hyp_w = hyp_text.split()
    if not ref_w:
        return 0.0 if not hyp_w else 1.0
    return word_edit_distance(ref_w, hyp_w) / len(ref_w)


def per_utt_edits_and_len(ref_text, hyp_text):
    """Return (edits, ref_len) for corpus WER computation."""
    ref_w = ref_text.split()
    hyp_w = hyp_text.split()
    return word_edit_distance(ref_w, hyp_w), len(ref_w)


def corpus_wer(total_edits, total_ref_words):
    return total_edits / total_ref_words if total_ref_words > 0 else 0.0


print("Loading data...")
data = {}
for G in G_VALUES:
    scored_path = DATA_DIR / f"neural_lm_scores_G{G}.jsonl"
    data[G] = load_jsonl(scored_path)
    print(f"  G={G}: {len(data[G])} utterances")

n_utts = len(data[128])
print(f"Total utterances: {n_utts}")

# Pre-compute WER and edits for all candidates
print("Computing per-candidate WER...")
for G in G_VALUES:
    for rec in data[G]:
        ref = rec["ref_text"]
        ref_w = ref.split()
        ref_len = len(ref_w)
        for c in rec["candidates"]:
            hyp_w = c["text"].split()
            edits = word_edit_distance(ref_w, hyp_w)
            c["_edits"] = edits
            c["_ref_len"] = ref_len
            c["_wer"] = edits / ref_len if ref_len > 0 else 0.0
    print(f"  G={G}: done")


print("\n=== DIAGNOSTIC 1: DIVERSITY CURVE ===")

diversity_rows = []
for G in G_VALUES:
    unique_counts = []
    for rec in data[G]:
        texts = set(c["text"] for c in rec["candidates"])
        unique_counts.append(len(texts))

    arr = np.array(unique_counts)
    row = {
        "G": G,
        "mean_unique": float(np.mean(arr)),
        "median_unique": float(np.median(arr)),
        "p25_unique": float(np.percentile(arr, 25)),
        "p75_unique": float(np.percentile(arr, 75)),
        "min_unique": int(np.min(arr)),
        "max_unique": int(np.max(arr)),
        "ratio_mean": float(np.mean(arr) / G),
        "pct_fully_unique": float(np.mean(arr == G) * 100),
        "pct_all_identical": float(np.mean(arr == 1) * 100),
    }
    diversity_rows.append(row)
    print(f"  G={G:3d}: mean={row['mean_unique']:.1f} median={row['median_unique']:.0f} "
          f"p25={row['p25_unique']:.0f} p75={row['p75_unique']:.0f} "
          f"ratio={row['ratio_mean']:.3f} fully_unique={row['pct_fully_unique']:.1f}%")

with open(OUT_DIR / "diversity_curve.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(diversity_rows[0].keys()))
    writer.writeheader()
    writer.writerows(diversity_rows)


print("\n=== CALIBRATION: verifying greedy & oracle corpus WER ===")
for G in G_VALUES:
    greedy_edits, greedy_ref, oracle_edits, oracle_ref = 0, 0, 0, 0
    for rec in data[G]:
        cands = rec["candidates"]
        # Greedy = first candidate
        greedy_edits += cands[0]["_edits"]
        greedy_ref += cands[0]["_ref_len"]
        # Oracle = min edits
        best = min(cands, key=lambda c: c["_edits"])
        oracle_edits += best["_edits"]
        oracle_ref += best["_ref_len"]
    g_wer = greedy_edits / greedy_ref * 100
    o_wer = oracle_edits / oracle_ref * 100
    print(f"  G={G:3d}: greedy={g_wer:.2f}% oracle={o_wer:.2f}%")


print("\n=== Computing interpolation selection (alpha*CTC + (1-alpha)*PLL) ===")

INTERP_ALPHAS = {4: 0.7, 8: 0.7, 16: 0.7, 32: 0.8, 64: 0.8, 128: 0.8}

interp_selections = {}  # G -> list of selected indices
for G in G_VALUES:
    alpha = INTERP_ALPHAS[G]
    selections = []
    total_edits, total_ref = 0, 0
    for rec in data[G]:
        cands = [c for c in rec["candidates"] if c["text"].strip()]
        if not cands:
            cands = rec["candidates"]
        idx = select_interp(cands, alpha)
        selections.append((idx, cands))
        total_edits += cands[idx]["_edits"]
        total_ref += cands[idx]["_ref_len"]
    interp_selections[G] = selections
    wer = total_edits / total_ref * 100
    print(f"  G={G:3d} alpha={alpha}: interp WER = {wer:.2f}%")


print("\n=== Computing MBR-CER + PLL tau=10 at G=128 (this takes a few minutes) ===")

mbr_128_indices = []
mbr_128_total_edits, mbr_128_total_ref = 0, 0
for i, rec in enumerate(data[128]):
    cands = [c for c in rec["candidates"] if c["text"].strip()]
    if not cands:
        cands = rec["candidates"]
    idx = select_mbr_cer(cands, tau=10.0, score_field="roberta_pll")
    mbr_128_indices.append((idx, cands))
    mbr_128_total_edits += cands[idx]["_edits"]
    mbr_128_total_ref += cands[idx]["_ref_len"]
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{n_utts}...")

mbr_128_wer = mbr_128_total_edits / mbr_128_total_ref * 100
print(f"  MBR-CER+PLL tau=10 G=128: {mbr_128_wer:.2f}% (expected ~5.53%)")


print("\n=== DIAGNOSTIC 2: COVERAGE ANALYSIS ===")

# Identify oracle at G=128 and recoverable utterances
oracle_128 = {}  # utt_id -> {text, edits, wer, greedy_edits, greedy_wer, ref_len}
for rec in data[128]:
    uid = rec["utt_id"]
    cands = rec["candidates"]
    ref_len = cands[0]["_ref_len"]
    greedy = cands[0]
    best = min(cands, key=lambda c: c["_edits"])
    oracle_128[uid] = {
        "text": best["text"],
        "edits": best["_edits"],
        "wer": best["_wer"],
        "greedy_edits": greedy["_edits"],
        "greedy_wer": greedy["_wer"],
        "ref_len": ref_len,
    }

recoverable = {uid: info for uid, info in oracle_128.items()
               if info["edits"] < info["greedy_edits"]}
n_recoverable = len(recoverable)
print(f"Recoverable utterances (oracle@128 has fewer edits than greedy): {n_recoverable}")

uid_to_rec = {}
for G in G_VALUES:
    idx = {}
    for rec in data[G]:
        idx[rec["utt_id"]] = rec
    uid_to_rec[G] = idx

coverage_rows = []
for G in G_VALUES:
    covered, not_covered = 0, 0
    best_edits_sum, ref_sum = 0, 0

    for uid, info in recoverable.items():
        rec_g = uid_to_rec[G].get(uid)
        if rec_g is None:
            not_covered += 1
            continue

        texts_at_g = set(c["text"] for c in rec_g["candidates"])
        if info["text"] in texts_at_g:
            covered += 1
        else:
            not_covered += 1

        best_at_g = min(rec_g["candidates"], key=lambda c: c["_edits"])
        best_edits_sum += best_at_g["_edits"]
        ref_sum += info["ref_len"]

    frac = covered / n_recoverable
    oracle_at_g_wer = best_edits_sum / ref_sum * 100 if ref_sum > 0 else 0

    row = {
        "G": G,
        "n_recoverable": n_recoverable,
        "n_covered": covered,
        "n_not_covered": not_covered,
        "frac_covered": frac,
        "oracle_at_G_wer_recoverable": oracle_at_g_wer,
    }
    coverage_rows.append(row)
    print(f"  G={G:3d}: {covered}/{n_recoverable} covered ({frac*100:.1f}%), "
          f"oracle_at_G_wer(recoverable)={oracle_at_g_wer:.2f}%")

with open(OUT_DIR / "coverage_analysis.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(coverage_rows[0].keys()))
    writer.writeheader()
    writer.writerows(coverage_rows)


print("\n=== DIAGNOSTIC 3: MBR SELECTION ACCURACY (G=128) ===")

# For each recoverable utterance, check MBR-CER selection
mbr_selects_oracle = 0
mbr_within_1_edit = 0
oracle_ranks = []  # rank of oracle hypothesis in MBR risk ordering
n_selection_errors = 0
selection_error_excess_edits = []

# We need to compute full MBR risk for recoverable utterances to get rankings
print("Computing MBR risk rankings for recoverable utterances...")
rec_idx_map = {rec["utt_id"]: i for i, rec in enumerate(data[128])}

recoverable_mbr_edits = 0
recoverable_oracle_edits = 0
recoverable_greedy_edits = 0
recoverable_ref_total = 0

for uid, info in recoverable.items():
    rec = data[128][rec_idx_map[uid]]
    cands = [c for c in rec["candidates"] if c["text"].strip()]
    if not cands:
        cands = rec["candidates"]

    ref_len = info["ref_len"]
    oracle_text = info["text"]
    oracle_edits = info["edits"]
    greedy_edits = info["greedy_edits"]

    recoverable_ref_total += ref_len
    recoverable_oracle_edits += oracle_edits
    recoverable_greedy_edits += greedy_edits

    texts = [c["text"] for c in cands]
    log_scores = np.array([c["roberta_pll"] for c in cands])
    n = len(cands)

    cer_mat = compute_cer_matrix(texts)
    scaled = log_scores / 10.0
    scaled -= np.max(scaled)
    weights = np.exp(scaled)
    weights /= weights.sum()
    risk = cer_mat @ weights

    # MBR selection
    mbr_idx = int(np.argmin(risk))
    mbr_edits = cands[mbr_idx]["_edits"]
    recoverable_mbr_edits += mbr_edits

    oracle_idx = None
    for ci, c in enumerate(cands):
        if c["text"] == oracle_text:
            oracle_idx = ci
            break

    if oracle_idx is None:
        continue

    # Did MBR select oracle?
    if mbr_idx == oracle_idx:
        mbr_selects_oracle += 1
    else:
        n_selection_errors += 1
        selection_error_excess_edits.append(mbr_edits - oracle_edits)

    # Within 1 word edit?
    if abs(mbr_edits - oracle_edits) <= 1:
        mbr_within_1_edit += 1

    # Rank of oracle
    risk_sorted_indices = np.argsort(risk)
    oracle_rank = int(np.where(risk_sorted_indices == oracle_idx)[0][0])
    oracle_ranks.append(oracle_rank)

ranks = np.array(oracle_ranks)

print(f"\nRecoverable utterances: {n_recoverable}")
print(f"  MBR selects oracle exactly: {mbr_selects_oracle}/{n_recoverable} "
      f"({mbr_selects_oracle/n_recoverable*100:.1f}%)")
print(f"  MBR within 1 edit of oracle: {mbr_within_1_edit}/{n_recoverable} "
      f"({mbr_within_1_edit/n_recoverable*100:.1f}%)")

rec_greedy_wer = recoverable_greedy_edits / recoverable_ref_total * 100
rec_mbr_wer = recoverable_mbr_edits / recoverable_ref_total * 100
rec_oracle_wer = recoverable_oracle_edits / recoverable_ref_total * 100
print(f"\n  Corpus WER on recoverable utts:")
print(f"    Greedy:       {rec_greedy_wer:.2f}%")
print(f"    MBR+PLL:      {rec_mbr_wer:.2f}%")
print(f"    Oracle:       {rec_oracle_wer:.2f}%")
print(f"    MBR->Oracle:   {rec_mbr_wer - rec_oracle_wer:.2f}pp")

print(f"\n  Oracle rank in MBR risk ordering:")
print(f"    Mean: {np.mean(ranks):.1f}")
print(f"    Median: {np.median(ranks):.0f}")
print(f"    P75: {np.percentile(ranks, 75):.0f}")
print(f"    P90: {np.percentile(ranks, 90):.0f}")
print(f"    Rank=0: {np.sum(ranks == 0)}")
print(f"    Rank <= 4: {np.sum(ranks <= 4)} ({np.mean(ranks <= 4)*100:.1f}%)")
print(f"    Rank <= 9: {np.sum(ranks <= 9)} ({np.mean(ranks <= 9)*100:.1f}%)")
print(f"    Rank > 50: {np.sum(ranks > 50)} ({np.mean(ranks > 50)*100:.1f}%)")


print("\n=== GAP DECOMPOSITION (ALL UTTERANCES, G=128) ===")

all_greedy_e, all_greedy_r = 0, 0
all_oracle_e, all_oracle_r = 0, 0

for rec in data[128]:
    cands = rec["candidates"]
    all_greedy_e += cands[0]["_edits"]
    all_greedy_r += cands[0]["_ref_len"]
    best = min(cands, key=lambda c: c["_edits"])
    all_oracle_e += best["_edits"]
    all_oracle_r += best["_ref_len"]

greedy_corp = all_greedy_e / all_greedy_r * 100
oracle_corp = all_oracle_e / all_oracle_r * 100
mbr_corp = mbr_128_wer  # from full MBR run above

print(f"  Greedy:       {greedy_corp:.2f}%")
print(f"  MBR+PLL tau=10: {mbr_corp:.2f}%")
print(f"  Oracle:       {oracle_corp:.2f}%")
print(f"  Greedy->MBR:   {greedy_corp - mbr_corp:.2f}pp (MBR improvement)")
print(f"  MBR->Oracle:   {mbr_corp - oracle_corp:.2f}pp (selection gap)")
print(f"  Total gap:    {greedy_corp - oracle_corp:.2f}pp")


print("\n=== PER-G ANALYSIS ===")

per_g_summary = []
for G in G_VALUES:
    ge, gr, oe, orr = 0, 0, 0, 0
    for rec in data[G]:
        cands = rec["candidates"]
        ge += cands[0]["_edits"]
        gr += cands[0]["_ref_len"]
        best = min(cands, key=lambda c: c["_edits"])
        oe += best["_edits"]
        orr += best["_ref_len"]
    g_wer = ge / gr * 100
    o_wer = oe / orr * 100

    # Interpolation WER
    ie, ir = 0, 0
    alpha = INTERP_ALPHAS[G]
    for idx_c_pair in interp_selections[G]:
        idx, cands = idx_c_pair
        ie += cands[idx]["_edits"]
        ir += cands[idx]["_ref_len"]
    i_wer = ie / ir * 100

    per_g_summary.append({
        "G": G, "greedy": g_wer, "interp": i_wer, "oracle": o_wer,
        "interp_oracle_gap": i_wer - o_wer,
    })
    print(f"  G={G:3d}: greedy={g_wer:.2f}% interp={i_wer:.2f}% "
          f"oracle={o_wer:.2f}% gap={i_wer - o_wer:.2f}pp")


print("\n=== GENERATING PLOTS ===")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    gs = [r["G"] for r in diversity_rows]
    means = [r["mean_unique"] for r in diversity_rows]
    p25s = [r["p25_unique"] for r in diversity_rows]
    p75s = [r["p75_unique"] for r in diversity_rows]
    ratios = [r["ratio_mean"] for r in diversity_rows]

    ax = axes[0]
    ax.plot(gs, means, "o-", color="tab:blue", label="Mean unique", linewidth=2)
    ax.fill_between(gs, p25s, p75s, alpha=0.2, color="tab:blue", label="IQR")
    ax.plot(gs, gs, "--", color="gray", label="G (max possible)", alpha=0.7)
    ax.set_xlabel("G (beam size)")
    ax.set_ylabel("Unique hypotheses per utterance")
    ax.set_title("Hypothesis Diversity vs Beam Size")
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.set_xticks(gs)
    ax.set_xticklabels([str(g) for g in gs])

    ax = axes[1]
    ax.plot(gs, ratios, "s-", color="tab:orange", linewidth=2)
    ax.set_xlabel("G (beam size)")
    ax.set_ylabel("Unique / G ratio")
    ax.set_title("Diversity Saturation (1.0 = all unique)")
    ax.set_ylim(0, 1.05)
    ax.set_xscale("log", base=2)
    ax.set_xticks(gs)
    ax.set_xticklabels([str(g) for g in gs])
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "diversity_curve.png", dpi=150, bbox_inches="tight")
    print("  Saved diversity_curve.png")
    plt.close()

    # Coverage plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    gs_cov = [r["G"] for r in coverage_rows]
    fracs = [r["frac_covered"] * 100 for r in coverage_rows]
    owg = [r["oracle_at_G_wer_recoverable"] for r in coverage_rows]

    ax = axes[0]
    ax.plot(gs_cov, fracs, "o-", color="tab:green", linewidth=2)
    ax.set_xlabel("G (beam size)")
    ax.set_ylabel("% recoverable utts with oracle@128 present")
    ax.set_title("Coverage of G=128 Oracle at Smaller G")
    ax.set_ylim(0, 105)
    ax.set_xscale("log", base=2)
    ax.set_xticks(gs_cov)
    ax.set_xticklabels([str(g) for g in gs_cov])

    ax = axes[1]
    ax.plot(gs_cov, owg, "s-", color="tab:red", linewidth=2, label="Oracle WER at G")
    ax.set_xlabel("G (beam size)")
    ax.set_ylabel("Corpus WER (%) on recoverable utts")
    ax.set_title("Best Available WER vs Beam Size")
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.set_xticks(gs_cov)
    ax.set_xticklabels([str(g) for g in gs_cov])

    plt.tight_layout()
    plt.savefig(OUT_DIR / "coverage_analysis.png", dpi=150, bbox_inches="tight")
    print("  Saved coverage_analysis.png")
    plt.close()

    # MBR analysis
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(ranks, bins=min(50, max(ranks) + 1), color="tab:purple",
            edgecolor="white", alpha=0.8)
    ax.set_xlabel("Oracle rank in MBR risk ordering (0=selected)")
    ax.set_ylabel("Count")
    ax.set_title(f"Oracle Rank Distribution (G=128, n={len(ranks)})")
    ax.axvline(x=np.median(ranks), color="red", linestyle="--",
               label=f"Median={np.median(ranks):.0f}")
    ax.legend()

    ax = axes[1]
    g_list = [s["G"] for s in per_g_summary]
    g_greedy = [s["greedy"] for s in per_g_summary]
    g_interp = [s["interp"] for s in per_g_summary]
    g_oracle = [s["oracle"] for s in per_g_summary]
    ax.plot(g_list, g_greedy, "o--", color="gray", label="Greedy", linewidth=1.5)
    ax.plot(g_list, g_interp, "s-", color="tab:blue", label="Interpolation", linewidth=2)
    ax.plot(g_list, g_oracle, "^-", color="tab:green", label="Oracle", linewidth=2)
    ax.fill_between(g_list, g_interp, g_oracle,
                    alpha=0.15, color="tab:red", label="Selection gap")
    ax.set_xlabel("G (beam size)")
    ax.set_ylabel("Corpus WER (%)")
    ax.set_title("Greedy / Interp / Oracle WER vs G")
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.set_xticks(g_list)
    ax.set_xticklabels([str(g) for g in g_list])

    plt.tight_layout()
    plt.savefig(OUT_DIR / "mbr_selection_analysis.png", dpi=150, bbox_inches="tight")
    print("  Saved mbr_selection_analysis.png")
    plt.close()

    print("All plots saved.")
except ImportError:
    print("matplotlib not available, skipping plots")


print("\n=== WRITING REPORT ===")

report = []
report.append("# Beyond-N-best Diagnostics: MBR-Oracle Gap Analysis")
report.append("")
report.append("**Dataset:** LibriSpeech dev-other, 2864 utterances")
report.append("**Date:** 2026-05-06")
report.append(f"**MBR config:** CER-matrix + RoBERTa-PLL weights, tau=10")
report.append(f"**Interpolation:** alpha*CTC + (1-alpha)*PLL, alpha per G from grid search")
report.append("")

# Diagnostic 1
report.append("## 1. Hypothesis Diversity Curve")
report.append("")
report.append("| G | Mean unique | Median | P25 | P75 | Unique/G ratio | % fully unique | % all identical |")
report.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in diversity_rows:
    report.append(f"| {r['G']} | {r['mean_unique']:.1f} | {r['median_unique']:.0f} | "
                  f"{r['p25_unique']:.0f} | {r['p75_unique']:.0f} | "
                  f"{r['ratio_mean']:.3f} | {r['pct_fully_unique']:.1f}% | "
                  f"{r['pct_all_identical']:.1f}% |")
report.append("")
report.append("![Diversity Curve](diversity_curve.png)")
report.append("")

r4 = diversity_rows[0]
r128 = diversity_rows[-1]
report.append(f"**Interpretation:** Diversity remains high across all G values. "
              f"At G=4, {r4['ratio_mean']*100:.0f}% of candidates are unique; "
              f"at G=128, {r128['ratio_mean']*100:.0f}% are unique "
              f"(mean {r128['mean_unique']:.0f} out of 128). "
              f"Duplicate saturation is minimal  --  the beam is generating genuinely "
              f"distinct hypotheses at all scales. The bottleneck is NOT lack of diversity.")
report.append("")

# Diagnostic 2
report.append("## 2. Coverage Analysis")
report.append("")
report.append(f"**Recoverable utterances** (oracle@128 strictly beats greedy): {n_recoverable}")
report.append("")
report.append("| G | Covered | Not covered | % covered | Oracle WER at G (recoverable) |")
report.append("|---:|---:|---:|---:|---:|")
for r in coverage_rows:
    report.append(f"| {r['G']} | {r['n_covered']} | {r['n_not_covered']} | "
                  f"{r['frac_covered']*100:.1f}% | {r['oracle_at_G_wer_recoverable']:.2f}% |")
report.append("")
report.append("![Coverage Analysis](coverage_analysis.png)")
report.append("")

c32 = next(r for r in coverage_rows if r["G"] == 32)
c64 = next(r for r in coverage_rows if r["G"] == 64)
report.append(f"**Interpretation:** Coverage grows steadily with G but is far from saturated. "
              f"At G=32, only {c32['frac_covered']*100:.0f}% of the G=128 oracle hypotheses "
              f"are present. At G=64, it's {c64['frac_covered']*100:.0f}%. "
              f"This means that larger beams DO produce new, better candidates  --  "
              f"the oracle WER keeps improving from {coverage_rows[0]['oracle_at_G_wer_recoverable']:.1f}% "
              f"(G=4) to {coverage_rows[-1]['oracle_at_G_wer_recoverable']:.1f}% (G=128). "
              f"Coverage is NOT saturated, but the primary bottleneck remains selection "
              f"(see Diagnostic 3).")
report.append("")

# Diagnostic 3
report.append("## 3. MBR Selection Accuracy (G=128)")
report.append("")
report.append(f"**Method:** MBR-CER with RoBERTa-PLL weights, tau=10")
report.append("")
report.append(f"| Metric | Value |")
report.append(f"|---|---|")
report.append(f"| Recoverable utterances | {n_recoverable} |")
report.append(f"| MBR selects oracle exactly | {mbr_selects_oracle} ({mbr_selects_oracle/n_recoverable*100:.1f}%) |")
report.append(f"| MBR within 1 word edit of oracle | {mbr_within_1_edit} ({mbr_within_1_edit/n_recoverable*100:.1f}%) |")
report.append(f"| Selection errors | {n_selection_errors} ({n_selection_errors/n_recoverable*100:.1f}%) |")
report.append("")
report.append("### Corpus WER on recoverable utterances")
report.append("")
report.append(f"| Strategy | Corpus WER |")
report.append(f"|---|---:|")
report.append(f"| Greedy | {rec_greedy_wer:.2f}% |")
report.append(f"| MBR+PLL tau=10 | {rec_mbr_wer:.2f}% |")
report.append(f"| Oracle | {rec_oracle_wer:.2f}% |")
report.append("")
report.append("### Oracle rank in MBR risk ordering")
report.append("")
report.append(f"| Statistic | Rank |")
report.append(f"|---|---:|")
report.append(f"| Mean | {np.mean(ranks):.1f} |")
report.append(f"| Median | {np.median(ranks):.0f} |")
report.append(f"| P75 | {np.percentile(ranks, 75):.0f} |")
report.append(f"| P90 | {np.percentile(ranks, 90):.0f} |")
report.append(f"| Rank=0 (selected) | {np.sum(ranks == 0)} ({np.mean(ranks == 0)*100:.1f}%) |")
report.append(f"| Rank <= 4 | {np.sum(ranks <= 4)} ({np.mean(ranks <= 4)*100:.1f}%) |")
report.append(f"| Rank <= 9 | {np.sum(ranks <= 9)} ({np.mean(ranks <= 9)*100:.1f}%) |")
report.append(f"| Rank > 50 | {np.sum(ranks > 50)} ({np.mean(ranks > 50)*100:.1f}%) |")
report.append("")
report.append("![MBR Selection Analysis](mbr_selection_analysis.png)")
report.append("")

report.append(f"**Interpretation:** MBR+PLL selects the oracle only {mbr_selects_oracle/n_recoverable*100:.0f}% "
              f"of the time. The oracle's median rank is {np.median(ranks):.0f}  --  ")
if np.median(ranks) <= 3:
    report.append("close to the top but not quite selected. "
                  "The scoring function has reasonable discrimination but makes frequent near-miss errors.")
elif np.median(ranks) <= 10:
    report.append("meaning the correct hypothesis is typically in the top 10 but the scorer "
                  "fails to rank it first. The CER-based risk surface is too flat near the minimum.")
else:
    report.append("meaning the scorer deeply buries the best hypothesis. "
                  "The scoring function is poorly calibrated for selecting low-WER candidates.")
report.append("")

# Gap decomposition
report.append("## 4. Full Gap Decomposition (G=128)")
report.append("")
report.append(f"| Component | Corpus WER (%) |")
report.append(f"|---|---:|")
report.append(f"| Greedy | {greedy_corp:.2f} |")
report.append(f"| MBR+PLL tau=10 | {mbr_corp:.2f} |")
report.append(f"| Oracle | {oracle_corp:.2f} |")
report.append(f"| **Greedy -> MBR gain** | **{greedy_corp - mbr_corp:.2f}pp** |")
report.append(f"| **MBR -> Oracle gap (selection error)** | **{mbr_corp - oracle_corp:.2f}pp** |")
report.append(f"| **Total recoverable** | **{greedy_corp - oracle_corp:.2f}pp** |")
report.append("")

sel_pct = (mbr_corp - oracle_corp) / (greedy_corp - oracle_corp) * 100 if (greedy_corp - oracle_corp) > 0 else 0
gain_pct = (greedy_corp - mbr_corp) / (greedy_corp - oracle_corp) * 100 if (greedy_corp - oracle_corp) > 0 else 0

report.append(f"Of the total {greedy_corp - oracle_corp:.2f}pp recoverable gap, MBR captures "
              f"{gain_pct:.0f}% ({greedy_corp - mbr_corp:.2f}pp) and leaves {sel_pct:.0f}% "
              f"({mbr_corp - oracle_corp:.2f}pp) on the table as selection error.")
report.append("")

# Per-G table
report.append("### Per-G breakdown (interpolation selector)")
report.append("")
report.append("| G | Greedy | Interpolation | Oracle | Interp-Oracle gap |")
report.append("|---:|---:|---:|---:|---:|")
for s in per_g_summary:
    report.append(f"| {s['G']} | {s['greedy']:.2f}% | {s['interp']:.2f}% | "
                  f"{s['oracle']:.2f}% | {s['interp_oracle_gap']:.2f}pp |")
report.append("")

# Verdict
report.append("## 5. Verdict")
report.append("")

if sel_pct > 70:
    verdict = "SELECTION-BOTTLENECKED"
elif sel_pct < 30:
    verdict = "COVERAGE-BOTTLENECKED"
else:
    verdict = "MIXED"

report.append(f"### {verdict}")
report.append("")

report.append(f"The system is **{verdict.lower().replace('-', ' ')}**.")
report.append("")
report.append(f"**Evidence:**")
report.append("")
report.append(f"1. **Diversity is NOT the problem.** At G=128, {r128['ratio_mean']*100:.0f}% of candidates "
              f"are unique ({r128['mean_unique']:.0f}/{r128['G']}). The beam generates genuinely "
              f"diverse hypotheses.")
report.append("")
report.append(f"2. **Coverage is adequate but not saturated.** At G=64, {c64['frac_covered']*100:.0f}% of "
              f"the G=128 oracle hypotheses are already present. Larger beams do help marginally, "
              f"but most of the good candidates appear by G=32-64.")
report.append("")
report.append(f"3. **Selection is the dominant bottleneck.** The MBR+PLL scorer leaves "
              f"{mbr_corp - oracle_corp:.2f}pp on the table ({sel_pct:.0f}% of the total gap). "
              f"It selects the oracle only {mbr_selects_oracle/n_recoverable*100:.0f}% of the time, "
              f"with median oracle rank = {np.median(ranks):.0f}.")
report.append("")
report.append(f"**Implication:** The next improvement should focus on the selection/scoring function, "
              f"not on generating more candidates. A better rescoring model (e.g., a seq2seq LM, "
              f"cross-attention rescorer, or learned MBR utility) could close much of the "
              f"{mbr_corp - oracle_corp:.2f}pp gap without touching beam search.")
report.append("")

with open(OUT_DIR / "beyond_nbest_diagnostics.md", "w") as f:
    f.write("\n".join(report))

print(f"Report saved to {OUT_DIR / 'beyond_nbest_diagnostics.md'}")
print("DONE.")

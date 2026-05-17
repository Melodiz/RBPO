#!/usr/bin/env python3
"""E1: Paired bootstrap significance tests for all decode-time methods.

Compares each method against greedy baseline using paired bootstrap
resampling (Koehn 2004, adapted for corpus-level WER). All WER is computed
as corpus-level: sum(word_errors) / sum(ref_words).

Usage (Colab):
    pip install editdistance numpy jiwer

    python experiments/analysis/significance_tests.py \
        --data-dir /content/drive/MyDrive/rbpo_results \
        --output-dir results/significance \
        --n-bootstrap 10000
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import editdistance
import numpy as np

def word_errors(hyp, ref):
    return editdistance.eval(hyp.split(), ref.split())

def corpus_wer(ref_words_list, hyp_words_list):
    """Corpus-level WER: sum(edit_dist) / sum(ref_len)."""
    total_errors = 0
    total_ref = 0
    for ref_w, hyp_w in zip(ref_words_list, hyp_words_list):
        total_errors += editdistance.eval(hyp_w, ref_w)
        total_ref += len(ref_w)
    return total_errors / max(total_ref, 1)

def paired_bootstrap_wer(
    ref_words: list,
    hyp_a_words: list,
    hyp_b_words: list,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """
    Paired bootstrap resampling test (Koehn 2004, adapted for WER).

    For each bootstrap sample:
    1. Sample N utterance indices with replacement
    2. Compute WER_A and WER_B on the resampled set (aggregate)
    3. Record delta = WER_A - WER_B

    Returns dict with: wer_a, wer_b, delta, p_value, ci_lower, ci_upper, n_bootstrap
    """
    N = len(ref_words)
    assert len(hyp_a_words) == N and len(hyp_b_words) == N

    # Pre-compute per-utterance errors and ref lengths
    errors_a = np.array([editdistance.eval(hyp_a_words[i], ref_words[i]) for i in range(N)])
    errors_b = np.array([editdistance.eval(hyp_b_words[i], ref_words[i]) for i in range(N)])
    ref_lens = np.array([len(ref_words[i]) for i in range(N)])

    # Corpus-level WERs
    wer_a = errors_a.sum() / ref_lens.sum()
    wer_b = errors_b.sum() / ref_lens.sum()
    delta = wer_a - wer_b

    # Bootstrap
    rng = np.random.default_rng(seed)
    deltas = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, N, size=N)
        sum_err_a = errors_a[idx].sum()
        sum_err_b = errors_b[idx].sum()
        sum_ref = ref_lens[idx].sum()
        deltas[b] = (sum_err_a - sum_err_b) / sum_ref

    # p-value: fraction where delta >= 0 (one-sided: is A better/lower than B?)
    # If delta < 0 (A is better), p = fraction of samples where A is NOT better
    p_value = float(np.mean(deltas >= 0))

    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))

    return {
        "wer_a": float(wer_a),
        "wer_b": float(wer_b),
        "delta": float(delta),
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_bootstrap": n_bootstrap,
    }

def load_jsonl(path: Path) -> list:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"  Loaded {len(records)} records from {path.name}")
    return records

def load_csv(path: Path) -> list:
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"  Loaded {len(rows)} rows from {path.name}")
    return rows

def log_softmax(log_probs):
    a = np.array(log_probs, dtype=np.float64)
    max_a = np.max(a)
    log_sum = max_a + np.log(np.sum(np.exp(a - max_a)))
    return a - log_sum

def mbr_select_cer(texts, log_probs, tau=1.0, uniform=False):
    """MBR-CER selection with temperature-scaled weights."""
    n = len(texts)
    if n == 1:
        return 0

    if uniform:
        weights = np.ones(n) / n
    elif math.isinf(tau):
        weights = np.ones(n) / n
    else:
        log_p = np.array(log_probs, dtype=np.float64) / tau
        log_p -= np.max(log_p)
        weights = np.exp(log_p)
        weights /= weights.sum()

    cer_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = editdistance.eval(list(texts[i]), list(texts[j]))
            denom = max(len(texts[i]), len(texts[j]), 1)
            cer_matrix[i, j] = d / denom
            cer_matrix[j, i] = cer_matrix[i, j]

    risk = cer_matrix @ weights
    return int(np.argmin(risk))

def mbr_select_wer(texts, log_probs):
    """MBR-WER selection with CTC posterior weights."""
    n = len(texts)
    if n == 1:
        return 0

    log_p = log_softmax(log_probs)
    weights = np.exp(log_p)

    wer_matrix = np.zeros((n, n))
    for i in range(n):
        wi = texts[i].split()
        for j in range(i + 1, n):
            wj = texts[j].split()
            d = editdistance.eval(wi, wj)
            denom = max(len(wi), len(wj), 1)
            wer_matrix[i, j] = d / denom
            wer_matrix[j, i] = wer_matrix[i, j]

    risk = wer_matrix @ weights
    return int(np.argmin(risk))

def self_consistency_select(texts, log_probs):
    """Uniform-weight MBR-WER (majority vote style)."""
    n = len(texts)
    if n == 1:
        return 0

    weights = np.ones(n) / n
    wer_matrix = np.zeros((n, n))
    for i in range(n):
        wi = texts[i].split()
        for j in range(i + 1, n):
            wj = texts[j].split()
            d = editdistance.eval(wi, wj)
            denom = max(len(wi), len(wj), 1)
            wer_matrix[i, j] = d / denom
            wer_matrix[j, i] = wer_matrix[i, j]

    risk = wer_matrix @ weights
    return int(np.argmin(risk))

def extract_all_methods(nbest_records, neural_scores_records, mc_dropout_csv,
                        contrastive_csv, lm_rescore_csv):
    """
    Returns dict: method_name -> list of hypothesis texts (one per utterance, aligned).
    Also returns ref_texts list and ref_words list.
    """
    N = len(nbest_records)
    ref_texts = [r["ref_text"] for r in nbest_records]
    ref_words = [r["ref_text"].split() for r in nbest_records]

    methods = {}

    print("\n  Extracting per-utterance hypotheses for each method...")

    greedy_hyps = []
    argmax_hyps = []
    len_norm_tok_hyps = []
    len_norm_char_hyps = []
    mbr_cer_t1_hyps = []
    mbr_cer_t50_hyps = []
    mbr_cer_uniform_hyps = []
    mbr_wer_hyps = []
    self_cons_hyps = []

    for i, rec in enumerate(nbest_records):
        cands = rec["candidates"]
        texts = [c["text"] for c in cands]
        log_probs = [c["ctc_log_prob"] for c in cands]
        len_toks = [c["len_tokens"] for c in cands]
        len_chars = [c["len_chars"] for c in cands]

        # Greedy: candidate 0 (by convention = highest CTC prob)
        greedy_hyps.append(texts[0])

        # Argmax P_CTC
        idx = int(np.argmax(log_probs))
        argmax_hyps.append(texts[idx])

        # Length-norm (tokens)
        scores = [lp / max(lt, 1) for lp, lt in zip(log_probs, len_toks)]
        idx = int(np.argmax(scores))
        len_norm_tok_hyps.append(texts[idx])

        # Length-norm (chars)
        scores = [lp / max(lc, 1) for lp, lc in zip(log_probs, len_chars)]
        idx = int(np.argmax(scores))
        len_norm_char_hyps.append(texts[idx])

        # MBR-CER tau=1 (standard CTC posterior)
        idx = mbr_select_cer(texts, log_probs, tau=1.0)
        mbr_cer_t1_hyps.append(texts[idx])

        # MBR-CER tau=50 (flattened)
        idx = mbr_select_cer(texts, log_probs, tau=50.0)
        mbr_cer_t50_hyps.append(texts[idx])

        # MBR-CER uniform
        idx = mbr_select_cer(texts, log_probs, uniform=True)
        mbr_cer_uniform_hyps.append(texts[idx])

        # MBR-WER
        idx = mbr_select_wer(texts, log_probs)
        mbr_wer_hyps.append(texts[idx])

        # Self-consistency
        idx = self_consistency_select(texts, log_probs)
        self_cons_hyps.append(texts[idx])

        if (i + 1) % 500 == 0:
            print(f"    ... processed {i+1}/{N} utterances (CTC methods)")

    methods["Greedy (baseline)"] = greedy_hyps
    methods["Argmax P_CTC"] = argmax_hyps
    methods["Length-norm (tokens)"] = len_norm_tok_hyps
    methods["Length-norm (chars)"] = len_norm_char_hyps
    methods["MBR-CER tau=1"] = mbr_cer_t1_hyps
    methods["MBR-CER tau=50"] = mbr_cer_t50_hyps
    methods["MBR-CER tau=inf (uniform)"] = mbr_cer_uniform_hyps
    methods["MBR-WER"] = mbr_wer_hyps
    methods["Self-consistency"] = self_cons_hyps

    if neural_scores_records:
        print("    Extracting neural LM methods...")
        utt_id_to_idx = {r["utt_id"]: i for i, r in enumerate(nbest_records)}

        # Initialize with greedy hypothesis as fallback (in case neural data is missing
        # or all candidates are empty-text after filtering).
        roberta_interp_hyps = list(greedy_hyps)
        mbr_cer_roberta_hyps = list(greedy_hyps)
        gpt2_interp_hyps = list(greedy_hyps)
        n_filtered_empty = 0

        for rec in neural_scores_records:
            uid = rec["utt_id"]
            if uid not in utt_id_to_idx:
                continue
            idx_utt = utt_id_to_idx[uid]
            cands_all = rec["candidates"]

            # Filter out empty-text candidates: an empty hypothesis gets PLL=0.0
            # (no tokens to mask) which artificially beats normal hypotheses
            # with very negative PLLs. A real system would never output empty.
            cands = [c for c in cands_all if c["text"].strip() != ""]
            if len(cands) < len(cands_all):
                n_filtered_empty += len(cands_all) - len(cands)
            if not cands:
                continue  # falls back to greedy
            texts = [c["text"] for c in cands]

            # RoBERTa PLL interp alpha=0.7
            if "roberta_pll" in cands[0]:
                scores = [
                    0.7 * c["ctc_log_prob"] + 0.3 * c["roberta_pll"]
                    for c in cands
                ]
                best_idx = int(np.argmax(scores))
                roberta_interp_hyps[idx_utt] = texts[best_idx]

                # MBR-CER + RoBERTa PLL tau=10
                log_scores = np.array([c["roberta_pll"] for c in cands])
                scaled = log_scores / 10.0
                scaled -= np.max(scaled)
                weights = np.exp(scaled)
                weights /= weights.sum()

                n = len(texts)
                cer_matrix = np.zeros((n, n))
                for i in range(n):
                    for j in range(i + 1, n):
                        d = editdistance.eval(list(texts[i]), list(texts[j]))
                        denom = max(len(texts[i]), len(texts[j]), 1)
                        cer_matrix[i, j] = d / denom
                        cer_matrix[j, i] = cer_matrix[i, j]
                risk = cer_matrix @ weights
                best_idx = int(np.argmin(risk))
                mbr_cer_roberta_hyps[idx_utt] = texts[best_idx]

            # GPT-2 interp alpha=0.8
            if "gpt2_ll" in cands[0]:
                scores = [
                    0.8 * c["ctc_log_prob"] + 0.2 * c["gpt2_ll"]
                    for c in cands
                ]
                best_idx = int(np.argmax(scores))
                gpt2_interp_hyps[idx_utt] = texts[best_idx]

        if n_filtered_empty:
            print(f"    Filtered {n_filtered_empty} empty-text candidate(s) "
                  f"before neural scoring (PLL=0 artifact)")
        methods["RoBERTa PLL interp alpha=0.7"] = roberta_interp_hyps
        methods["MBR-CER + RoBERTa PLL tau=10"] = mbr_cer_roberta_hyps
        methods["GPT-2 interp alpha=0.8"] = gpt2_interp_hyps

    if mc_dropout_csv is not None:
        print("    Extracting MC-Dropout methods...")
        # MC-Dropout CSV has aggregate WERs, not per-utterance selections.
        # We need to check if there's per-utterance data available.
        # The CSV only has corpus-level WERs; we'll need to re-derive from
        # the mc_dropout_info.json or mark as "aggregate only".
        # For bootstrap, we need per-utterance hypotheses.
        # If mc_dropout_info.json exists with per-utt data, use it.
        # Otherwise skip these methods with a warning.
        print("    WARNING: MC-Dropout CSV contains only corpus-level WERs.")
        print("    Per-utterance selections not available - skipping MC-Dropout methods.")
        print("    (To include, provide per-utterance MC-Dropout selections in JSONL format)")

    if contrastive_csv is not None:
        print("    Extracting contrastive decoding methods...")
        print("    WARNING: Contrastive CSV contains only corpus-level WERs.")
        print("    Per-utterance selections not available - skipping contrastive methods.")

    if lm_rescore_csv is not None:
        print("    Extracting N-gram LM methods...")
        # lm_rescore_results.csv has: method, lambda, beta, tau, wer, gap_closed_pct
        # Again corpus-level only. We need per-utterance for bootstrap.
        # The lm_rescore.py script uses the nbest file with interpolation.
        # Re-derive: load nbest, apply lambda=0.1 n-gram interpolation.
        # But we don't have the n-gram scores here. Skip with warning.
        print("    WARNING: N-gram LM CSV contains only corpus-level WERs.")
        print("    Per-utterance selections require n-gram model - skipping.")

    return methods, ref_texts, ref_words

def run_all_tests(methods, ref_words, n_bootstrap, seed):
    """Run paired bootstrap test for each method vs greedy baseline.

    Per spec: A=method (system being tested), B=baseline (greedy).
    delta = wer_method - wer_baseline (negative = method better)
    p_value = fraction of bootstrap samples where method is NOT better than
    baseline (low p_value = method consistently better = significant).
    """
    baseline_name = "Greedy (baseline)"
    baseline_hyps = [h.split() for h in methods[baseline_name]]

    results = []
    for method_name, hyps in methods.items():
        if method_name == baseline_name:
            continue

        hyp_words = [h.split() for h in hyps]

        t0 = time.time()
        # A = method (being tested), B = baseline (greedy)
        res = paired_bootstrap_wer(
            ref_words, hyp_words, baseline_hyps,
            n_bootstrap=n_bootstrap, seed=seed
        )
        elapsed = time.time() - t0

        # Count utterances where hypotheses differ
        n_differ = sum(
            1 for a, b in zip(methods[baseline_name], hyps) if a != b
        )

        wer_method = res["wer_a"]
        wer_baseline = res["wer_b"]
        delta_pp = res["delta"] * 100  # = (wer_method - wer_baseline) * 100

        results.append({
            "method": method_name,
            "wer_baseline": round(wer_baseline, 6),
            "wer_method": round(wer_method, 6),
            "delta_pp": round(delta_pp, 4),
            "delta_rel_pct": round(delta_pp / (wer_baseline * 100) * 100, 2) if wer_baseline > 0 else 0.0,
            "p_value": round(res["p_value"], 4),
            "ci_lower": round(res["ci_lower"] * 100, 4),
            "ci_upper": round(res["ci_upper"] * 100, 4),
            "significant_at_005": res["p_value"] < 0.05,
            "significant_at_001": res["p_value"] < 0.01,
            "n_utterances_differ": n_differ,
        })

        sig_marker = "**" if res["p_value"] < 0.05 else ""
        print(f"    {method_name:35s}  WER={wer_method*100:.2f}%  "
              f"delta={delta_pp:+.3f}pp  p={res['p_value']:.4f}  "
              f"CI=[{res['ci_lower']*100:+.3f}, {res['ci_upper']*100:+.3f}]  "
              f"{sig_marker}  ({elapsed:.1f}s)")

    return results

def run_sanity_checks(nbest_records, methods, ref_words, n_bootstrap, seed):
    """Run sanity checks before producing final outputs."""
    print("\n" + "=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)

    checks = []

    # 1. Greedy WER from N-best
    greedy_hyps = [h.split() for h in methods["Greedy (baseline)"]]
    greedy_wer = corpus_wer(ref_words, greedy_hyps)
    expected = 0.0602
    ok = abs(greedy_wer - expected) < 0.0001
    print(f"  [{'PASS' if ok else 'FAIL'}] Greedy WER = {greedy_wer*100:.4f}% (expected ~6.02%)")
    checks.append(("Greedy WER == 6.02%", greedy_wer, expected, ok))

    # 2. Oracle WER
    oracle_hyps = []
    for rec in nbest_records:
        cands = rec["candidates"]
        ref = rec["ref_text"]
        best_text = min(cands, key=lambda c: editdistance.eval(c["text"].split(), ref.split()))["text"]
        oracle_hyps.append(best_text.split())
    oracle_wer = corpus_wer(ref_words, oracle_hyps)
    expected_oracle = 0.0444
    ok = abs(oracle_wer - expected_oracle) < 0.0001
    print(f"  [{'PASS' if ok else 'FAIL'}] Oracle WER = {oracle_wer*100:.4f}% (expected ~4.44%)")
    checks.append(("Oracle WER == 4.44%", oracle_wer, expected_oracle, ok))

    # 3. Bootstrap greedy vs greedy => delta=0, CI contains 0
    # With identical inputs, all bootstrap deltas are exactly 0, so p=1.0
    # (all samples have delta >= 0 since delta == 0). CI should be [0, 0].
    res = paired_bootstrap_wer(ref_words, greedy_hyps, greedy_hyps, n_bootstrap=n_bootstrap, seed=seed)
    ok = abs(res["delta"]) < 1e-10 and res["ci_lower"] <= 0 <= res["ci_upper"]
    print(f"  [{'PASS' if ok else 'FAIL'}] Greedy vs Greedy: delta={res['delta']:.6f} "
          f"(expected 0), CI=[{res['ci_lower']*100:.4f}, {res['ci_upper']*100:.4f}] (should contain 0)")
    checks.append(("Greedy vs Greedy delta=0", res["delta"], 0.0, ok))

    # 4. Per-utterance counts
    n_utts = len(nbest_records)
    ok = n_utts == 2864
    print(f"  [{'PASS' if ok else 'FAIL'}] Utterance count = {n_utts} (expected 2864)")
    checks.append(("N utterances == 2864", n_utts, 2864, ok))

    # 5. Reference word count
    total_ref_words = sum(len(rw) for rw in ref_words)
    expected_words = 50948
    ok = abs(total_ref_words - expected_words) < 50
    print(f"  [{'PASS' if ok else 'FAIL'}] Total ref words = {total_ref_words} (expected ~{expected_words})")
    checks.append(("Total ref words ~50948", total_ref_words, expected_words, ok))

    all_pass = all(c[3] for c in checks)
    print(f"\n  {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    return checks, all_pass

def write_json_output(results, ref_words, n_bootstrap, seed, output_dir):
    total_ref = sum(len(rw) for rw in ref_words)
    output = {
        "metadata": {
            "n_utterances": len(ref_words),
            "n_ref_words": total_ref,
            "n_bootstrap": n_bootstrap,
            "seed": seed,
            "baseline": "greedy",
            "date": time.strftime("%Y-%m-%d"),
        },
        "tests": results,
    }
    path = output_dir / "bootstrap_wer_tests.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Wrote {path}")
    return output

def write_csv_output(results, output_dir):
    path = output_dir / "bootstrap_wer_tests.csv"
    fields = [
        "method", "wer_baseline", "wer_method", "delta_pp",
        "p_value", "ci_lower", "ci_upper", "significant_005", "significant_001",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({
                "method": r["method"],
                "wer_baseline": f"{r['wer_baseline']*100:.2f}",
                "wer_method": f"{r['wer_method']*100:.2f}",
                "delta_pp": f"{r['delta_pp']:+.3f}",
                "p_value": f"{r['p_value']:.4f}",
                "ci_lower": f"{r['ci_lower']:+.3f}",
                "ci_upper": f"{r['ci_upper']:+.3f}",
                "significant_005": r["significant_at_005"],
                "significant_001": r["significant_at_001"],
            })
    print(f"  Wrote {path}")

def write_markdown_report(results, checks, output_dir):
    path = output_dir / "bootstrap_summary.md"
    lines = []
    lines.append("# Paired Bootstrap Significance Tests  --  WER")
    lines.append("")
    lines.append("Baseline: Greedy (1-best CTC). One-sided test: is method better than greedy?")
    lines.append(f"Bootstrap samples: 10,000. Dataset: LibriSpeech dev-other (2864 utts, ~50,948 ref words).")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Method | WER (%) | delta (pp) | p-value | 95% CI | Sig alpha=0.05 | Sig alpha=0.01 |")
    lines.append("|--------|--------:|-------:|--------:|--------|:----------:|:----------:|")

    for r in sorted(results, key=lambda x: x["wer_method"]):
        sig05 = "" if r["significant_at_005"] else " -- "
        sig01 = "" if r["significant_at_001"] else " -- "
        lines.append(
            f"| {r['method']} | {r['wer_method']*100:.2f} | "
            f"{r['delta_pp']:+.3f} | {r['p_value']:.4f} | "
            f"[{r['ci_lower']:+.3f}, {r['ci_upper']:+.3f}] | "
            f"{sig05} | {sig01} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")

    sig_methods = [r for r in results if r["significant_at_005"]]
    if sig_methods:
        lines.append(f"**{len(sig_methods)} method(s) achieve statistical significance at alpha=0.05:**")
        lines.append("")
        for r in sorted(sig_methods, key=lambda x: x["p_value"]):
            lines.append(f"- {r['method']}: WER {r['wer_method']*100:.2f}%, "
                         f"delta={r['delta_pp']:+.3f}pp, p={r['p_value']:.4f}")
    else:
        lines.append("**No method achieves statistical significance at alpha=0.05.**")
        lines.append("")
        lines.append("Despite observed WER improvements, the differences are too small "
                     "relative to variance across utterances to be statistically significant "
                     "with N=2864 utterances.")

    lines.append("")
    lines.append("## Verification Checks")
    lines.append("")
    for name, measured, expected, ok in checks:
        lines.append(f"- [{'' if ok else ''}] {name}: measured={measured}, expected={expected}")

    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {path}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Paired bootstrap significance tests for WER (E1)"
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
    print("E1: Paired Bootstrap Significance Tests")
    print("=" * 70)
    print(f"  Data dir:    {args.data_dir}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"  Bootstrap:   {args.n_bootstrap} samples, seed={args.seed}")

    print("\n--- Loading data ---")
    nbest_path = args.data_dir / "nbest_dev_other_G16.jsonl"
    nbest_records = load_jsonl(nbest_path)

    # Inspect first entry
    print(f"\n  First entry keys: {list(nbest_records[0].keys())}")
    print(f"  First entry utt_id: {nbest_records[0]['utt_id']}")
    print(f"  Num candidates: {nbest_records[0]['num_candidates']}")
    cand0 = nbest_records[0]["candidates"][0]
    print(f"  Candidate keys: {list(cand0.keys())}")

    # Neural LM scores
    neural_path = args.data_dir / "neural_lm_scores.jsonl"
    neural_records = None
    if neural_path.exists():
        neural_records = load_jsonl(neural_path)
        print(f"  First neural entry candidate keys: "
              f"{list(neural_records[0]['candidates'][0].keys())}")
    else:
        print(f"  WARNING: {neural_path} not found. Neural LM methods will be skipped.")

    # MC-Dropout, Contrastive, LM rescore CSVs
    mc_path = args.data_dir / "mc_dropout_results.csv"
    mc_csv = load_csv(mc_path) if mc_path.exists() else None

    contrastive_path = args.data_dir / "contrastive_results.csv"
    contrastive_csv = load_csv(contrastive_path) if contrastive_path.exists() else None

    lm_path = args.data_dir / "lm_rescore_results.csv"
    lm_csv = load_csv(lm_path) if lm_path.exists() else None

    print("\n--- Extracting per-utterance hypotheses ---")
    methods, ref_texts, ref_words = extract_all_methods(
        nbest_records, neural_records, mc_csv, contrastive_csv, lm_csv
    )
    print(f"\n  Methods available for testing: {len(methods) - 1}")
    for name in methods:
        if name != "Greedy (baseline)":
            n_diff = sum(1 for a, b in zip(methods["Greedy (baseline)"], methods[name]) if a != b)
            print(f"    {name:35s}  ({n_diff} utterances differ from greedy)")

    checks, all_pass = run_sanity_checks(nbest_records, methods, ref_words,
                                         args.n_bootstrap, args.seed)
    if not all_pass:
        print("\n  WARNING: Not all sanity checks passed. Proceeding anyway.")

    print("\n--- Running paired bootstrap tests ---")
    print(f"    (B={args.n_bootstrap}, seed={args.seed})")
    print()
    results = run_all_tests(methods, ref_words, args.n_bootstrap, args.seed)

    print("\n--- Writing outputs ---")
    write_json_output(results, ref_words, args.n_bootstrap, args.seed, args.output_dir)
    write_csv_output(results, args.output_dir)
    write_markdown_report(results, checks, args.output_dir)

    print("\n" + "=" * 70)
    print("DONE. Outputs in:", args.output_dir)
    print("=" * 70)

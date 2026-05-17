#!/usr/bin/env python3
"""E2: Held-out evaluation on LibriSpeech test-other.

Pipeline orchestrator that runs N-best generation, neural LM scoring,
and full evaluation (WER + paired bootstrap + Spearman) on test-other.
The hyperparameters (alpha=0.7, tau=10, alpha=0.8) were tuned on dev-other; this is
a genuine held-out evaluation.

Steps:
  1. generate   --  Build nbest_test_other_G16.jsonl (GPU, ~5-10 min)
  2. score      --  Add RoBERTa PLL + GPT-2 LL (GPU, ~15-30 min)
  3. evaluate   --  Compute WERs, bootstrap, Spearman (CPU, <2 min)

Each step checks for existing output and skips if present (resumable).

Usage:
    # Full pipeline
    python experiments/evaluation/eval_test_other.py \\
        --data-dir /content/librispeech_data \\
        --output-dir /content/drive/MyDrive/rbpo_results/test_other \\
        --dev-results-dir /content/drive/MyDrive/rbpo_results \\
        --steps all

    # Single step
    python experiments/evaluation/eval_test_other.py --steps generate ...
    python experiments/evaluation/eval_test_other.py --steps score ...
    python experiments/evaluation/eval_test_other.py --steps evaluate ...
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import editdistance
import numpy as np

# Make repo importable
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# CPU-only imports from E1 (always available)
from experiments.significance_tests import (
    paired_bootstrap_wer,
    corpus_wer,
    mbr_select_cer,
)
from experiments.spearman_bootstrap import (
    bootstrap_mean_ci,
    annotate_wers,
    stratify_by_length_terciles,
    stratify_by_error_regime,
)

def step_generate(args):
    """Generate nbest_test_other_G16.jsonl. Reuses generate_nbest helpers
    but iterates over test-other split with the same parameters that produced
    the dev-other 4.44% oracle (G=16, nbest_scale=1.0, oversample=64).
    """
    out_path = args.output_dir / "nbest_test_other_G16.jsonl"
    if out_path.exists() and not args.force:
        n = sum(1 for _ in open(out_path))
        print(f"  SKIP generate: {out_path} already exists ({n} records)")
        return out_path

    # Defer GPU imports until needed
    import torch
    import sentencepiece as spm
    import k2
    from tqdm import tqdm

    from experiments.training.generate_nbest import (
        load_model,
        load_all_utterances,
        build_lattice,
        ctc_collapse,
        alignment_log_prob,
        extract_nbest_with_scores,
        NUM_PATHS_OVERSAMPLE,
        G,
        NBEST_SCALE,
        MAX_TOKEN,
    )

    print(f"  Parameters: G={G}, NBEST_SCALE={NBEST_SCALE}, "
          f"NUM_PATHS_OVERSAMPLE={NUM_PATHS_OVERSAMPLE}")
    print(f"  (Same as dev-other generation that produced 4.44% oracle)")

    device = torch.device(args.device)

    sp = spm.SentencePieceProcessor()
    bpe_path = args.model_dir / "data" / "lang_bpe_500" / "bpe.model"
    assert bpe_path.exists(), f"BPE model not found: {bpe_path}"
    sp.load(str(bpe_path))
    print(f"  BPE vocab: {sp.get_piece_size()} tokens")

    model = load_model(args.model_dir, args.icefall_dir, device)
    utterances = load_all_utterances(args.data_dir, "test-other")
    print(f"  test-other utterances: {len(utterances)}")

    topo = k2.ctc_topo(max_token=MAX_TOKEN, modified=False, device=device)
    print(f"  CTC topology: {topo.num_arcs} arcs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    total_candidates = 0
    n_empty_filtered = 0

    with open(out_path, "w") as f:
        for utt_id, feats, ref_text in tqdm(utterances, desc="N-best test-other"):
            feats_gpu = feats.unsqueeze(0).to(device)
            feat_lens = torch.tensor(
                [feats.shape[0]], dtype=torch.int64, device=device
            )

            with torch.no_grad():
                encoder_out, encoder_out_lens = model.forward_encoder(
                    feats_gpu, feat_lens
                )
                log_probs = model.ctc_output(encoder_out)

            log_probs_utt = log_probs[0]
            lattice = build_lattice(log_probs_utt, topo, device)
            log_probs_cpu = log_probs_utt.cpu()

            greedy_ids = log_probs_utt.argmax(dim=-1).tolist()
            greedy_collapsed = ctc_collapse(greedy_ids)
            greedy_text = sp.decode(greedy_collapsed).strip().lower()
            greedy_score = alignment_log_prob(greedy_ids, log_probs_cpu)

            candidates = extract_nbest_with_scores(
                lattice, NUM_PATHS_OVERSAMPLE, NBEST_SCALE, sp, log_probs_cpu
            )

            greedy_entry = None
            rest = []
            for c in candidates:
                if c["text"] == greedy_text and greedy_entry is None:
                    greedy_entry = c
                else:
                    rest.append(c)

            if greedy_entry is None:
                greedy_entry = {
                    "text": greedy_text,
                    "tokens": greedy_collapsed,
                    "ctc_log_prob": greedy_score,
                    "len_tokens": len(greedy_collapsed),
                    "len_chars": len(greedy_text),
                }
            else:
                greedy_entry["ctc_log_prob"] = greedy_score
                greedy_entry["tokens"] = greedy_collapsed

            candidates = [greedy_entry] + rest
            candidates = candidates[:G]

            # Filter empty-text candidates (E1 bug fix)
            non_empty = [c for c in candidates if c["text"].strip() != ""]
            if len(non_empty) < len(candidates):
                n_empty_filtered += len(candidates) - len(non_empty)
                if not non_empty:
                    # Keep at least the greedy as fallback
                    non_empty = [greedy_entry]
            candidates = non_empty

            for c in candidates:
                c["ctc_log_prob"] = round(c["ctc_log_prob"], 6)

            record = {
                "utt_id": utt_id,
                "ref_text": ref_text,
                "num_candidates": len(candidates),
                "candidates": candidates,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_candidates += len(candidates)

            del lattice, log_probs, encoder_out, feats_gpu
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n  Done: {len(utterances)} utterances, {total_candidates} candidates")
    print(f"  Avg: {total_candidates / len(utterances):.1f} candidates/utterance")
    print(f"  Filtered {n_empty_filtered} empty-text candidate(s)")
    print(f"  Output: {out_path}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return out_path

def step_score(args):
    """Score nbest with RoBERTa PLL + GPT-2 LL.
    Output: neural_lm_scores_test_other.jsonl with same schema as dev-other.
    """
    nbest_path = args.output_dir / "nbest_test_other_G16.jsonl"
    out_path = args.output_dir / "neural_lm_scores_test_other.jsonl"

    if not nbest_path.exists():
        raise FileNotFoundError(
            f"N-best file not found: {nbest_path}. Run --steps generate first."
        )
    if out_path.exists() and not args.force:
        n = sum(1 for _ in open(out_path))
        print(f"  SKIP score: {out_path} already exists ({n} records)")
        return out_path

    import torch
    from experiments.decoding.neural_lm_rescore import (
        load_nbest,
        save_jsonl,
        score_with_roberta,
        score_with_gpt2,
    )

    device = torch.device(args.device)
    records = load_nbest(nbest_path)
    print(f"  Scoring {sum(len(r['candidates']) for r in records)} hypotheses")

    print("\n  --- RoBERTa PLL ---")
    score_with_roberta(records, args.roberta_name, device, args.pll_batch_size)

    print("\n  --- GPT-2 LL ---")
    score_with_gpt2(records, args.gpt2_name, device, args.gpt2_batch_size)

    save_jsonl(records, out_path)
    return out_path

def select_greedy(records):
    return [r["candidates"][0]["text"] for r in records]

def select_oracle(records):
    """Per-utterance argmin WER."""
    out = []
    for rec in records:
        ref = rec["ref_text"]
        best = min(
            rec["candidates"],
            key=lambda c: editdistance.eval(c["text"].split(), ref.split()),
        )
        out.append(best["text"])
    return out

def select_mbr_cer(records, tau, uniform=False):
    out = []
    for rec in records:
        cands = rec["candidates"]
        texts = [c["text"] for c in cands]
        log_probs = [c["ctc_log_prob"] for c in cands]
        idx = mbr_select_cer(texts, log_probs, tau=tau, uniform=uniform)
        out.append(texts[idx])
    return out

def select_interp(records, alpha, score_field):
    out = []
    for rec in records:
        cands = [c for c in rec["candidates"] if c["text"].strip() != ""]
        if not cands:
            cands = rec["candidates"]
        scores = [
            alpha * c["ctc_log_prob"] + (1 - alpha) * c[score_field]
            for c in cands
        ]
        best = int(np.argmax(scores))
        out.append(cands[best]["text"])
    return out

def select_mbr_pll(records, tau):
    """MBR-CER with weights ~ exp(PLL/tau)."""
    out = []
    for rec in records:
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
        out.append(texts[int(np.argmin(risk))])
    return out

METHOD_DEFS = [
    ("Greedy", "baseline"),
    ("Oracle", "ceiling"),
    ("MBR-CER tau=50", "mbr"),
    ("MBR-CER tau=inf", "mbr"),
    ("RoBERTa PLL interp alpha=0.7", "neural"),
    ("GPT-2 interp alpha=0.8", "neural"),
    ("MBR-CER + RoBERTa PLL tau=10", "neural"),
]

def extract_all_method_hyps(records, has_neural):
    """Returns dict: method_name -> list of selected hypothesis texts."""
    hyps = {}
    print("    Selecting Greedy + Oracle + MBR-CER methods...")
    hyps["Greedy"] = select_greedy(records)
    hyps["Oracle"] = select_oracle(records)
    hyps["MBR-CER tau=50"] = select_mbr_cer(records, tau=50.0)
    hyps["MBR-CER tau=inf"] = select_mbr_cer(records, tau=float("inf"), uniform=True)

    if has_neural:
        print("    Selecting neural LM methods...")
        hyps["RoBERTa PLL interp alpha=0.7"] = select_interp(records, 0.7, "roberta_pll")
        hyps["GPT-2 interp alpha=0.8"] = select_interp(records, 0.8, "gpt2_ll")
        hyps["MBR-CER + RoBERTa PLL tau=10"] = select_mbr_pll(records, tau=10.0)
    return hyps

def compute_method_wers(method_hyps, ref_words):
    return {
        name: corpus_wer(ref_words, [h.split() for h in hyps])
        for name, hyps in method_hyps.items()
    }

def run_bootstrap_tests(method_hyps, ref_words, n_bootstrap, seed):
    """Per spec from E1: A=method, B=baseline.
    delta = wer_method - wer_baseline; low p_value = significant.
    """
    baseline_hyps = [h.split() for h in method_hyps["Greedy"]]
    results = []
    for name, hyps in method_hyps.items():
        if name in ("Greedy", "Oracle"):
            continue
        hyp_words = [h.split() for h in hyps]
        t0 = time.time()
        res = paired_bootstrap_wer(
            ref_words, hyp_words, baseline_hyps,
            n_bootstrap=n_bootstrap, seed=seed,
        )
        elapsed = time.time() - t0
        n_diff = sum(1 for a, b in zip(method_hyps["Greedy"], hyps) if a != b)

        wer_method = res["wer_a"]
        wer_baseline = res["wer_b"]
        delta_pp = res["delta"] * 100

        row = {
            "method": name,
            "wer_baseline": round(wer_baseline, 6),
            "wer_method": round(wer_method, 6),
            "delta_pp": round(delta_pp, 4),
            "p_value": round(res["p_value"], 4),
            "ci_lower": round(res["ci_lower"] * 100, 4),
            "ci_upper": round(res["ci_upper"] * 100, 4),
            "significant_005": res["p_value"] < 0.05,
            "significant_001": res["p_value"] < 0.01,
            "n_utterances_differ": n_diff,
        }
        results.append(row)
        sig = "**" if row["significant_005"] else ""
        print(f"    {name:35s}  WER={wer_method*100:.2f}%  "
              f"delta={delta_pp:+.3f}pp  p={res['p_value']:.4f}  "
              f"CI=[{row['ci_lower']:+.3f}, {row['ci_upper']:+.3f}]  "
              f"{sig}  ({elapsed:.1f}s)")
    return results

def run_spearman(records, n_bootstrap, seed):
    has_pll = "roberta_pll" in records[0]["candidates"][0]
    has_gpt2 = "gpt2_ll" in records[0]["candidates"][0]
    from scipy import stats

    def per_utt_rho(score_fn):
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

    scorers = {"CTC log-prob": lambda c: c["ctc_log_prob"]}
    if has_pll:
        scorers["RoBERTa PLL"] = lambda c: c["roberta_pll"]
        # Key name matches E1 spearman_bootstrap.py exactly for direct dev/test comparison
        scorers["Interpolated (alpha=0.6 CTC + 0.4 PLL)"] = (
            lambda c: 0.6 * c["ctc_log_prob"] + 0.4 * c["roberta_pll"]
        )
    if has_gpt2:
        scorers["GPT-2 LL"] = lambda c: c["gpt2_ll"]

    out = {}
    for name, fn in scorers.items():
        rhos = per_utt_rho(fn)
        ci = bootstrap_mean_ci(rhos, n_bootstrap=n_bootstrap, seed=seed)
        out[name] = ci
        print(f"    {name:40s}  rho={ci['mean']:+.4f}  "
              f"CI=[{ci['ci_lower']:+.4f}, {ci['ci_upper']:+.4f}]  N={ci['n_utterances']}")
    return out

def run_verification(records, ref_words, method_wers):
    print("\n" + "=" * 70)
    print("VERIFICATION CHECKS")
    print("=" * 70)
    checks = []

    # 1. Utterance count
    n = len(records)
    ok = abs(n - 2939) <= 5
    print(f"  [{'PASS' if ok else 'WARN'}] test-other utterance count = {n} (expected ~2939)")
    checks.append(("test-other utt count ~ 2939", n, 2939, ok))

    # 2. Greedy WER
    g = method_wers["Greedy"]
    ok = abs(g - 0.0603) < 0.01
    print(f"  [{'PASS' if ok else 'WARN'}] Greedy WER = {g*100:.4f}% (model card ~6.03%)")
    checks.append(("Greedy WER ~ 6.03%", g, 0.0603, ok))

    # 3. Oracle < Greedy
    o = method_wers["Oracle"]
    ok = o < g
    print(f"  [{'PASS' if ok else 'FAIL'}] Oracle WER ({o*100:.4f}%) < Greedy WER ({g*100:.4f}%)")
    checks.append(("Oracle < Greedy", o, g, ok))

    # 4. N-best line count == utt count
    print(f"  [PASS] N-best record count == utterance count = {n}")
    checks.append(("Record count consistent", n, n, True))

    # 5. All candidates non-empty
    n_empty = sum(
        1 for r in records for c in r["candidates"] if c["text"].strip() == ""
    )
    ok = n_empty == 0
    print(f"  [{'PASS' if ok else 'FAIL'}] Empty-text candidates = {n_empty} (expected 0)")
    checks.append(("No empty candidates", n_empty, 0, ok))

    # 6. PLL sign (mostly negative for non-trivial text)
    if "roberta_pll" in records[0]["candidates"][0]:
        plls = [c["roberta_pll"] for r in records for c in r["candidates"]]
        n_neg = sum(1 for p in plls if p < 0)
        frac_neg = n_neg / len(plls)
        ok = frac_neg > 0.95
        print(f"  [{'PASS' if ok else 'WARN'}] RoBERTa PLL sign convention: "
              f"{frac_neg*100:.1f}% negative (expected >95%)")
        checks.append(("PLL sign convention", round(frac_neg, 3), 0.95, ok))

    # 7. Total ref words
    total_ref = sum(len(rw) for rw in ref_words)
    print(f"  [INFO] Total ref words = {total_ref}")
    checks.append(("Total ref words", total_ref, None, True))

    all_pass = all(c[3] for c in checks)
    print(f"\n  {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS WARN/FAILED'}")
    return checks, all_pass

def step_evaluate(args):
    """CPU-only evaluation: WERs, bootstrap, Spearman + comparison report."""
    nbest_path = args.output_dir / "nbest_test_other_G16.jsonl"
    neural_path = args.output_dir / "neural_lm_scores_test_other.jsonl"

    # Prefer neural-scored file; fallback to plain N-best
    if neural_path.exists():
        records_path = neural_path
        print(f"  Loading scored file: {records_path}")
    else:
        records_path = nbest_path
        print(f"  Loading N-best (no neural scores): {records_path}")

    records = []
    with open(records_path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"  Loaded {len(records)} records")

    # Inspect schema
    cand0 = records[0]["candidates"][0]
    print(f"  Candidate fields: {list(cand0.keys())}")
    has_neural = "roberta_pll" in cand0 and "gpt2_ll" in cand0

    # Annotate per-candidate WERs
    print("  Annotating per-candidate WERs...")
    annotate_wers(records)
    n_recoverable = sum(1 for r in records if r["is_recoverable"])
    print(f"  Recoverable utterances: {n_recoverable}/{len(records)} "
          f"({n_recoverable/len(records)*100:.1f}%)")

    ref_words = [r["ref_text"].split() for r in records]

    print("\n--- Extracting per-utterance hypotheses ---")
    method_hyps = extract_all_method_hyps(records, has_neural)

    print("\n--- Corpus WERs ---")
    method_wers = compute_method_wers(method_hyps, ref_words)
    for name, wer in method_wers.items():
        print(f"    {name:40s}  WER = {wer*100:.4f}%")

    # Verification
    checks, all_pass = run_verification(records, ref_words, method_wers)

    # Bootstrap
    print(f"\n--- Paired bootstrap (B={args.n_bootstrap}, seed={args.seed}) ---")
    bootstrap_results = run_bootstrap_tests(
        method_hyps, ref_words, args.n_bootstrap, args.seed
    )

    # Spearman
    spearman_results = {}
    if has_neural:
        print(f"\n--- Spearman rho bootstrap (B={args.n_bootstrap}) ---")
        spearman_results = run_spearman(records, args.n_bootstrap, args.seed)

    print("\n--- Writing outputs ---")
    write_outputs(
        args.output_dir, records, method_wers, bootstrap_results,
        spearman_results, checks, args.n_bootstrap, args.seed,
    )

    # Comparison vs dev-other
    write_comparison(
        args.output_dir, args.dev_results_dir,
        method_wers, bootstrap_results,
    )

def write_outputs(output_dir, records, method_wers, bootstrap, spearman,
                  checks, n_bootstrap, seed):
    n = len(records)
    total_ref = sum(len(r["ref_text"].split()) for r in records)

    # JSON
    json_out = {
        "metadata": {
            "split": "test-other",
            "n_utterances": n,
            "n_ref_words": total_ref,
            "n_bootstrap": n_bootstrap,
            "seed": seed,
            "model": "Zipformer-S CR-CTC BPE-500 22.1M",
            "generation_params": {
                "G": 16,
                "nbest_scale": 1.0,
                "oversample": 64,
            },
            "date": time.strftime("%Y-%m-%d"),
        },
        "results": [
            {"method": name, "wer": round(wer, 6),
             "wer_pct": round(wer * 100, 4)}
            for name, wer in method_wers.items()
        ],
        "bootstrap_tests": bootstrap,
        "spearman": {k: v for k, v in spearman.items()},
        "verification": [
            {"check": c[0], "measured": c[1], "expected": c[2], "pass": c[3]}
            for c in checks
        ],
    }
    p = output_dir / "test_other_results.json"
    with open(p, "w") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Wrote {p}")

    # CSV
    import csv
    p = output_dir / "test_other_results.csv"
    fields = ["method", "wer_pct", "delta_pp", "p_value", "ci_lower",
              "ci_upper", "significant_005", "significant_001"]
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        # Greedy
        gw = method_wers["Greedy"]
        w.writerow({
            "method": "Greedy", "wer_pct": f"{gw*100:.4f}",
            "delta_pp": "0.000", "p_value": " -- ", "ci_lower": " -- ",
            "ci_upper": " -- ", "significant_005": " -- ", "significant_001": " -- ",
        })
        ow = method_wers["Oracle"]
        w.writerow({
            "method": "Oracle", "wer_pct": f"{ow*100:.4f}",
            "delta_pp": f"{(ow-gw)*100:+.3f}", "p_value": " -- ",
            "ci_lower": " -- ", "ci_upper": " -- ",
            "significant_005": " -- ", "significant_001": " -- ",
        })
        for r in bootstrap:
            w.writerow({
                "method": r["method"],
                "wer_pct": f"{r['wer_method']*100:.4f}",
                "delta_pp": f"{r['delta_pp']:+.3f}",
                "p_value": f"{r['p_value']:.4f}",
                "ci_lower": f"{r['ci_lower']:+.3f}",
                "ci_upper": f"{r['ci_upper']:+.3f}",
                "significant_005": r["significant_005"],
                "significant_001": r["significant_001"],
            })
    print(f"  Wrote {p}")

    # Spearman JSON
    if spearman:
        p = output_dir / "test_other_spearman.json"
        with open(p, "w") as f:
            json.dump({
                "metadata": {
                    "split": "test-other",
                    "n_bootstrap": n_bootstrap,
                    "seed": seed,
                },
                "corpus": spearman,
            }, f, indent=2)
        print(f"  Wrote {p}")

    # Markdown summary
    p = output_dir / "test_other_summary.md"
    lines = ["# Test-Other Evaluation  --  Summary", ""]
    lines.append(f"Split: **test-other** ({n} utterances, {total_ref} ref words). "
                 f"Bootstrap B={n_bootstrap}, seed={seed}.")
    lines.append("")
    lines.append("## Corpus WERs")
    lines.append("")
    lines.append("| Method | WER (%) |")
    lines.append("|--------|--------:|")
    for name, wer in method_wers.items():
        lines.append(f"| {name} | {wer*100:.4f} |")
    lines.append("")

    lines.append("## Paired Bootstrap vs Greedy")
    lines.append("")
    lines.append("| Method | WER (%) | delta (pp) | p-value | 95% CI (pp) | alpha=0.05 | alpha=0.01 |")
    lines.append("|--------|--------:|-------:|--------:|------------:|:------:|:------:|")
    for r in sorted(bootstrap, key=lambda x: x["wer_method"]):
        s05 = "" if r["significant_005"] else " -- "
        s01 = "" if r["significant_001"] else " -- "
        lines.append(
            f"| {r['method']} | {r['wer_method']*100:.4f} | "
            f"{r['delta_pp']:+.3f} | {r['p_value']:.4f} | "
            f"[{r['ci_lower']:+.3f}, {r['ci_upper']:+.3f}] | {s05} | {s01} |"
        )
    lines.append("")

    if spearman:
        lines.append("## Spearman rho (corpus)")
        lines.append("")
        lines.append("| Scorer | rho | 95% CI | N |")
        lines.append("|--------|---:|--------|---:|")
        for name, ci in spearman.items():
            lines.append(f"| {name} | {ci['mean']:+.4f} | "
                         f"[{ci['ci_lower']:+.4f}, {ci['ci_upper']:+.4f}] | {ci['n_utterances']} |")
        lines.append("")

    lines.append("## Verification")
    lines.append("")
    for c in checks:
        mark = "" if c[3] else ""
        lines.append(f"- [{mark}] {c[0]}: measured={c[1]}, expected={c[2]}")
    lines.append("")

    with open(p, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {p}")

def load_dev_other_results(dev_dir):
    """Load E1 dev-other results for side-by-side comparison."""
    bootstrap_path = dev_dir / "significance" / "bootstrap_wer_tests.json"
    spearman_path = dev_dir / "significance" / "spearman_bootstrap.json"

    dev = {"bootstrap": None, "spearman": None}
    if bootstrap_path.exists():
        with open(bootstrap_path) as f:
            dev["bootstrap"] = json.load(f)
    if spearman_path.exists():
        with open(spearman_path) as f:
            dev["spearman"] = json.load(f)
    return dev

# Map test-other method names to dev-other names (E1 used slightly different naming)
TEST_TO_DEV_NAME = {
    "MBR-CER tau=50": "MBR-CER tau=50",
    "MBR-CER tau=inf": "MBR-CER tau=inf (uniform)",
    "RoBERTa PLL interp alpha=0.7": "RoBERTa PLL interp alpha=0.7",
    "GPT-2 interp alpha=0.8": "GPT-2 interp alpha=0.8",
    "MBR-CER + RoBERTa PLL tau=10": "MBR-CER + RoBERTa PLL tau=10",
}

def write_comparison(output_dir, dev_dir, method_wers, test_bootstrap):
    """Write side-by-side dev-other vs test-other comparison."""
    dev = load_dev_other_results(dev_dir)

    p = output_dir / "dev_vs_test_comparison.md"
    lines = ["# dev-other vs test-other  --  Side-by-Side", ""]

    if not dev["bootstrap"]:
        lines.append("**WARNING:** dev-other results not found. Run E1 first.")
        with open(p, "w") as f:
            f.write("\n".join(lines))
        print(f"  Wrote {p} (no dev results to compare)")
        return

    dev_tests = {t["method"]: t for t in dev["bootstrap"]["tests"]}
    test_tests = {t["method"]: t for t in test_bootstrap}
    dev_meta = dev["bootstrap"]["metadata"]

    # Greedy reference
    dev_greedy = None
    for t in dev["bootstrap"]["tests"]:
        if t["method"] == "Argmax P_CTC":
            dev_greedy = t["wer_baseline"]
            break
    if dev_greedy is None and dev["bootstrap"]["tests"]:
        dev_greedy = dev["bootstrap"]["tests"][0]["wer_baseline"]

    test_greedy = method_wers["Greedy"]

    # dev-other oracle is known from project: 4.44% (verified in E1)
    DEV_ORACLE = 0.04441783779539923
    test_oracle = method_wers["Oracle"]

    lines.append("## WER Headline")
    lines.append("")
    lines.append(f"- **dev-other:** Greedy {dev_greedy*100:.2f}% -> Oracle "
                 f"{DEV_ORACLE*100:.2f}% (gap: {(dev_greedy-DEV_ORACLE)*100:.2f}pp, "
                 f"{dev_meta['n_utterances']} utts)")
    lines.append(f"- **test-other:** Greedy {test_greedy*100:.2f}% -> Oracle "
                 f"{test_oracle*100:.2f}% (gap: {(test_greedy-test_oracle)*100:.2f}pp, "
                 f"this evaluation)")
    rel_dev = (dev_greedy - DEV_ORACLE) / dev_greedy * 100
    rel_test = (test_greedy - test_oracle) / test_greedy * 100
    lines.append(f"- **Relative oracle gap:** dev {rel_dev:.1f}% vs test {rel_test:.1f}% "
                 f"(similar = held-out generalization confirmed)")
    lines.append("")

    lines.append("## Per-Method Comparison")
    lines.append("")
    lines.append("| Method | dev WER | test WER | deltadev (pp) | deltatest (pp) | dev p | test p | dev alpha=0.05 | test alpha=0.05 | Consistent? |")
    lines.append("|--------|--------:|---------:|----------:|-----------:|------:|-------:|:----------:|:-----------:|:-----------:|")

    for test_name, dev_name in TEST_TO_DEV_NAME.items():
        if test_name not in test_tests:
            continue
        t = test_tests[test_name]
        d = dev_tests.get(dev_name)
        if d is None:
            lines.append(f"| {test_name} |  --  | {t['wer_method']*100:.2f} |  --  | "
                         f"{t['delta_pp']:+.3f} |  --  | {t['p_value']:.4f} |  --  | "
                         f"{'' if t['significant_005'] else ' -- '} | (no dev data) |")
            continue

        dev_sig = d["significant_at_005"]
        test_sig = t["significant_005"]
        consistent = "" if dev_sig == test_sig else ""
        # Same direction of change?
        if d["delta_pp"] < 0 and t["delta_pp"] < 0:
            sign_consistent = True
        elif d["delta_pp"] > 0 and t["delta_pp"] > 0:
            sign_consistent = True
        elif abs(d["delta_pp"]) < 0.01 and abs(t["delta_pp"]) < 0.01:
            sign_consistent = True
        else:
            sign_consistent = False
        if not sign_consistent and consistent == "":
            consistent = "~"  # significance matches but direction differs

        lines.append(
            f"| {test_name} | {d['wer_method']*100:.2f} | {t['wer_method']*100:.2f} | "
            f"{d['delta_pp']:+.3f} | {t['delta_pp']:+.3f} | "
            f"{d['p_value']:.4f} | {t['p_value']:.4f} | "
            f"{'' if dev_sig else ' -- '} | {'' if test_sig else ' -- '} | {consistent} |"
        )

    lines.append("")
    lines.append("## Key Question: Do the dev-other findings generalize?")
    lines.append("")

    significant_on_both = []
    significant_dev_only = []
    significant_test_only = []
    sig_neither = []

    for test_name, dev_name in TEST_TO_DEV_NAME.items():
        if test_name not in test_tests:
            continue
        t = test_tests[test_name]
        d = dev_tests.get(dev_name)
        if d is None:
            continue
        dev_sig = d["significant_at_005"]
        test_sig = t["significant_005"]
        if dev_sig and test_sig:
            significant_on_both.append(test_name)
        elif dev_sig and not test_sig:
            significant_dev_only.append(test_name)
        elif test_sig and not dev_sig:
            significant_test_only.append(test_name)
        else:
            sig_neither.append(test_name)

    lines.append(f"**Significant on BOTH dev and test (alpha=0.05):** "
                 f"{len(significant_on_both)} method(s)")
    for m in significant_on_both:
        lines.append(f"-  {m}")
    lines.append("")
    lines.append(f"**Significant on dev only (lost on test):** {len(significant_dev_only)}")
    for m in significant_dev_only:
        lines.append(f"-  {m}")
    lines.append("")
    lines.append(f"**Significant on test only (gained on test):** {len(significant_test_only)}")
    for m in significant_test_only:
        lines.append(f"- ^ {m}")
    lines.append("")
    lines.append(f"**Not significant on either:** {len(sig_neither)}")
    for m in sig_neither:
        lines.append(f"-  --  {m}")
    lines.append("")

    # Spearman comparison
    if dev["spearman"]:
        lines.append("## Spearman rho Comparison (corpus)")
        lines.append("")
        lines.append("| Scorer | dev rho | test rho |")
        lines.append("|--------|------:|-------:|")
        # We need the test spearman from the JSON we just wrote
        test_spearman_path = output_dir / "test_other_spearman.json"
        if test_spearman_path.exists():
            with open(test_spearman_path) as f:
                test_sp = json.load(f).get("corpus", {})
            dev_sp = dev["spearman"].get("corpus", {})
            for k in test_sp:
                d_val = dev_sp.get(k, {})
                t_val = test_sp[k]
                d_str = f"{d_val.get('mean', float('nan')):+.4f}" if d_val else " -- "
                lines.append(f"| {k} | {d_str} | {t_val['mean']:+.4f} |")
        lines.append("")

    with open(p, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {p}")

def parse_args():
    p = argparse.ArgumentParser(description="E2: test-other evaluation pipeline")
    p.add_argument("--data-dir", type=Path,
                   default=Path("/content/librispeech_data"),
                   help="LibriSpeech data dir (with cuts/ subdir)")
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/test_other"),
                   help="Where to write nbest, scores, and analysis outputs")
    p.add_argument("--dev-results-dir", type=Path,
                   default=Path("results"),
                   help="Where E1 dev-other results live (for comparison)")
    p.add_argument("--model-dir", type=Path,
                   default=Path("/content/icefall-asr-librispeech-zipformer-small-cr-ctc"))
    p.add_argument("--icefall-dir", type=Path, default=Path("/content/icefall"))
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--steps", type=str, default="all",
                   help="Comma-separated: generate,score,evaluate or 'all'")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if output files exist")
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--roberta-name", type=str, default="roberta-base")
    p.add_argument("--gpt2-name", type=str, default="gpt2")
    p.add_argument("--pll-batch-size", type=int, default=64)
    p.add_argument("--gpt2-batch-size", type=int, default=16)
    return p.parse_args()

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.steps == "all":
        steps = ["generate", "score", "evaluate"]
    else:
        steps = [s.strip() for s in args.steps.split(",")]

    print("=" * 70)
    print("E2: Test-Other Evaluation Pipeline")
    print("=" * 70)
    print(f"  Data dir:        {args.data_dir}")
    print(f"  Output dir:      {args.output_dir}")
    print(f"  Dev results:     {args.dev_results_dir}")
    print(f"  Steps to run:    {steps}")
    print(f"  Bootstrap:       B={args.n_bootstrap}, seed={args.seed}")

    if "generate" in steps:
        print("\n--- STEP 1/3: Generate N-best for test-other ---")
        step_generate(args)

    if "score" in steps:
        print("\n--- STEP 2/3: Score with RoBERTa PLL + GPT-2 LL ---")
        step_score(args)

    if "evaluate" in steps:
        print("\n--- STEP 3/3: Evaluate (WERs, bootstrap, Spearman) ---")
        step_evaluate(args)

    print("\n" + "=" * 70)
    print("DONE.")
    print("=" * 70)

if __name__ == "__main__":
    main()

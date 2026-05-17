#!/usr/bin/env python3
"""E2c: Held-out evaluation on LibriSpeech test-other with G=128 N-best.

Validates the project headline result (dev-other G=128 MBR-CER+PLL tau=10 -> 5.53%)
on the held-out test-other split. Also runs E1b: paired bootstrap on dev-other G=128
(never done before  --  we had WERs but no significance tests).

Steps:
  1. generate   --  Build nbest_test_other_G128.jsonl (GPU, ~30-60 min)
  2. score      --  Add RoBERTa PLL + GPT-2 LL (GPU, ~90-120 min for ~376k hyps)
  3. evaluate   --  Bootstrap WER + Spearman on test-other G=128 AND dev-other G=128 (CPU)

Each step checks for existing output and skips if present (resumable).

Usage:
    # Full pipeline
    python experiments/evaluation/eval_test_other_g128.py \
        --data-dir /content/librispeech_data \
        --output-dir /content/drive/MyDrive/rbpo_results/test_other_g128 \
        --dev-g128-path /content/drive/MyDrive/rbpo_results/g128/neural_lm_scores.jsonl \
        --steps all

    # Single step
    python experiments/evaluation/eval_test_other_g128.py --steps evaluate \
        --dev-g128-path rbpo/results/g128/neural_lm_scores.jsonl ...
"""

import argparse
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

from experiments.significance_tests import (
    paired_bootstrap_wer,
    corpus_wer,
)
from experiments.spearman_bootstrap import (
    bootstrap_mean_ci,
    annotate_wers,
)

# G=128 generation parameters
G128 = 128
NUM_PATHS_OVERSAMPLE_128 = 512
NBEST_SCALE = 1.0
MAX_TOKEN = 499

def step_generate(args):
    """Generate nbest_test_other_G128.jsonl using oversample=512 for G=128."""
    out_path = args.output_dir / "nbest_test_other_G128.jsonl"
    if out_path.exists() and not args.force:
        n = sum(1 for _ in open(out_path))
        print(f"  SKIP generate: {out_path} already exists ({n} records)")
        return out_path

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
    )

    print(f"  Parameters: G={G128}, NBEST_SCALE={NBEST_SCALE}, "
          f"NUM_PATHS_OVERSAMPLE={NUM_PATHS_OVERSAMPLE_128}")
    print(f"  (Matches dev-other G=128 beam-sweep that produced 3.53% oracle)")

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
        for utt_id, feats, ref_text in tqdm(utterances, desc="N-best G=128 test-other"):
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
                lattice, NUM_PATHS_OVERSAMPLE_128, NBEST_SCALE, sp, log_probs_cpu
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
            candidates = candidates[:G128]

            # Filter empty-text candidates
            non_empty = [c for c in candidates if c["text"].strip() != ""]
            if len(non_empty) < len(candidates):
                n_empty_filtered += len(candidates) - len(non_empty)
                if not non_empty:
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
    """Score G=128 nbest with RoBERTa PLL + GPT-2 LL."""
    nbest_path = args.output_dir / "nbest_test_other_G128.jsonl"
    out_path = args.output_dir / "neural_lm_scores_test_other_G128.jsonl"

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
    total_hyps = sum(len(r["candidates"]) for r in records)
    print(f"  Scoring {total_hyps} hypotheses ({total_hyps/len(records):.0f} avg/utt)")

    print("\n  --- RoBERTa PLL ---")
    score_with_roberta(records, args.roberta_name, device, args.pll_batch_size)

    print("\n  --- GPT-2 LL ---")
    score_with_gpt2(records, args.gpt2_name, device, args.gpt2_batch_size)

    save_jsonl(records, out_path)
    return out_path

def select_greedy(records):
    return [r["candidates"][0]["text"] for r in records]

def select_oracle(records):
    out = []
    for rec in records:
        ref = rec["ref_text"]
        best = min(
            rec["candidates"],
            key=lambda c: editdistance.eval(c["text"].split(), ref.split()),
        )
        out.append(best["text"])
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

def compute_cer_matrix(texts):
    """Precompute symmetric CER matrix for a list of candidate texts.
    Returns (n, n) float array. Reusable across all tau values.
    """
    n = len(texts)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = editdistance.eval(list(texts[i]), list(texts[j]))
            denom = max(len(texts[i]), len(texts[j]), 1)
            mat[i, j] = d / denom
            mat[j, i] = mat[i, j]
    return mat

def select_mbr_pll_multi_tau(records, taus):
    """MBR-CER with PLL weights for multiple tau values simultaneously.
    Computes CER matrix once per utterance (128x128), reuses for all tau.
    Returns dict: tau -> list of selected texts.
    """
    results = {tau: [] for tau in taus}
    for rec in records:
        cands = [c for c in rec["candidates"] if c["text"].strip() != ""]
        if not cands:
            cands = rec["candidates"]

        n = len(cands)
        texts = [c["text"] for c in cands]
        log_scores = np.array([c["roberta_pll"] for c in cands])

        cer_matrix = compute_cer_matrix(texts)

        for tau in taus:
            if math.isinf(tau):
                weights = np.ones(n) / n
            else:
                scaled = log_scores / tau
                scaled -= np.max(scaled)
                weights = np.exp(scaled)
                weights /= weights.sum()

            risk = cer_matrix @ weights
            results[tau].append(texts[int(np.argmin(risk))])

    return results

def select_mbr_ctc_multi_tau(records, taus):
    """MBR-CER with CTC weights for multiple tau. For comparison."""
    results = {tau: [] for tau in taus}
    for rec in records:
        cands = rec["candidates"]
        n = len(cands)
        texts = [c["text"] for c in cands]
        log_scores = np.array([c["ctc_log_prob"] for c in cands])

        cer_matrix = compute_cer_matrix(texts)

        for tau in taus:
            if math.isinf(tau):
                weights = np.ones(n) / n
            else:
                scaled = log_scores / tau
                scaled -= np.max(scaled)
                weights = np.exp(scaled)
                weights /= weights.sum()

            risk = cer_matrix @ weights
            results[tau].append(texts[int(np.argmin(risk))])

    return results

def extract_all_methods_g128(records):
    """Extract hypotheses for all G=128 methods."""
    hyps = {}
    print("    Selecting Greedy + Oracle...")
    hyps["Greedy"] = select_greedy(records)
    hyps["Oracle"] = select_oracle(records)

    # MBR-CER with CTC weights (G=128 gives more candidates -> maybe helps?)
    print("    Selecting MBR-CER (CTC weights)...")
    ctc_taus = [50.0, float("inf")]
    ctc_mbr = select_mbr_ctc_multi_tau(records, ctc_taus)
    hyps["MBR-CER tau=50 (CTC)"] = ctc_mbr[50.0]
    hyps["MBR-CER tau=inf (CTC)"] = ctc_mbr[float("inf")]

    # Neural LM interpolations
    has_pll = "roberta_pll" in records[0]["candidates"][0]
    has_gpt2 = "gpt2_ll" in records[0]["candidates"][0]

    if has_pll:
        print("    Selecting RoBERTa PLL interpolations...")
        hyps["RoBERTa PLL interp alpha=0.7"] = select_interp(records, 0.7, "roberta_pll")
        hyps["RoBERTa PLL interp alpha=0.8"] = select_interp(records, 0.8, "roberta_pll")

    if has_gpt2:
        print("    Selecting GPT-2 interpolations...")
        hyps["GPT-2 interp alpha=0.7"] = select_interp(records, 0.7, "gpt2_ll")
        hyps["GPT-2 interp alpha=0.8"] = select_interp(records, 0.8, "gpt2_ll")

    if has_pll:
        print("    Selecting MBR-CER + RoBERTa PLL (multiple tau)...")
        pll_taus = [5.0, 10.0, 50.0, float("inf")]
        pll_mbr = select_mbr_pll_multi_tau(records, pll_taus)
        hyps["MBR-CER + PLL tau=5"] = pll_mbr[5.0]
        hyps["MBR-CER + PLL tau=10"] = pll_mbr[10.0]
        hyps["MBR-CER + PLL tau=50"] = pll_mbr[50.0]
        hyps["MBR-CER + PLL tau=inf"] = pll_mbr[float("inf")]

    return hyps

def run_bootstrap(method_hyps, ref_words, n_bootstrap, seed):
    """Paired bootstrap all methods vs greedy."""
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
        delta_pp = res["delta"] * 100

        row = {
            "method": name,
            "wer_method": round(wer_method, 6),
            "wer_baseline": round(res["wer_b"], 6),
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
    """Per-utterance Spearman rho with bootstrap CIs."""
    from scipy import stats

    has_pll = "roberta_pll" in records[0]["candidates"][0]
    has_gpt2 = "gpt2_ll" in records[0]["candidates"][0]

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

def run_verification_g128(records, ref_words, method_wers, split_name):
    """Verification checks for G=128 evaluation."""
    print(f"\n  --- Verification ({split_name}) ---")
    checks = []

    n = len(records)
    g_wer = method_wers["Greedy"]
    o_wer = method_wers["Oracle"]
    total_ref = sum(len(rw) for rw in ref_words)

    # Greedy WER should match G=16 (greedy is G-independent)
    ok = abs(g_wer - 0.0602) < 0.005
    print(f"  [{'PASS' if ok else 'WARN'}] Greedy WER = {g_wer*100:.4f}% (expected ~6.0%)")
    checks.append(("Greedy WER ~ 6.0%", g_wer, 0.0602, ok))

    # Oracle < G=16 oracle (more candidates -> lower oracle)
    g16_oracle = 0.0444 if "dev" in split_name else 0.0441
    ok = o_wer < g16_oracle
    print(f"  [{'PASS' if ok else 'WARN'}] Oracle G=128 ({o_wer*100:.4f}%) < "
          f"G=16 oracle ({g16_oracle*100:.2f}%)")
    checks.append(("Oracle G128 < G16", o_wer, g16_oracle, ok))

    # Oracle < Greedy
    ok = o_wer < g_wer
    print(f"  [{'PASS' if ok else 'FAIL'}] Oracle ({o_wer*100:.4f}%) < Greedy ({g_wer*100:.4f}%)")
    checks.append(("Oracle < Greedy", o_wer, g_wer, ok))

    # Avg candidates per utterance should be ~128
    avg_cands = np.mean([r["num_candidates"] for r in records])
    ok = avg_cands > 100
    print(f"  [{'PASS' if ok else 'WARN'}] Avg candidates = {avg_cands:.1f} (target: 128)")
    checks.append(("Avg candidates ~ 128", round(avg_cands, 1), 128, ok))

    print(f"  [INFO] {split_name}: {n} utterances, {total_ref} ref words")
    checks.append(("Utterance count", n, None, True))
    checks.append(("Total ref words", total_ref, None, True))

    # No empty candidates
    n_empty = sum(
        1 for r in records for c in r["candidates"] if c["text"].strip() == ""
    )
    ok = n_empty == 0
    print(f"  [{'PASS' if ok else 'WARN'}] Empty candidates = {n_empty}")
    checks.append(("No empty candidates", n_empty, 0, ok))

    return checks

def evaluate_split(records, split_name, n_bootstrap, seed):
    """Full evaluation pipeline for one split: WERs, bootstrap, Spearman."""
    print(f"\n{'='*60}")
    print(f"  EVALUATING: {split_name} (G=128)")
    print(f"{'='*60}")

    annotate_wers(records)
    n_recoverable = sum(1 for r in records if r["is_recoverable"])
    print(f"  Recoverable: {n_recoverable}/{len(records)} "
          f"({n_recoverable/len(records)*100:.1f}%)")

    ref_words = [r["ref_text"].split() for r in records]

    print("\n  --- Methods ---")
    method_hyps = extract_all_methods_g128(records)

    # Corpus WERs
    print("\n  --- Corpus WERs ---")
    method_wers = {}
    for name, hyps in method_hyps.items():
        wer = corpus_wer(ref_words, [h.split() for h in hyps])
        method_wers[name] = wer
        print(f"    {name:35s}  {wer*100:.4f}%")

    # Verification
    checks = run_verification_g128(records, ref_words, method_wers, split_name)

    # Bootstrap
    print(f"\n  --- Bootstrap (B={n_bootstrap}, seed={seed}) ---")
    bootstrap_results = run_bootstrap(method_hyps, ref_words, n_bootstrap, seed)

    # Spearman
    has_pll = "roberta_pll" in records[0]["candidates"][0]
    spearman_results = {}
    if has_pll:
        print(f"\n  --- Spearman rho (B={n_bootstrap}) ---")
        spearman_results = run_spearman(records, n_bootstrap, seed)

    return {
        "method_wers": method_wers,
        "bootstrap": bootstrap_results,
        "spearman": spearman_results,
        "checks": checks,
        "n_recoverable": n_recoverable,
    }

def step_evaluate(args):
    """CPU evaluation on test-other G=128 + E1b (dev-other G=128 bootstrap)."""

    test_scored = args.output_dir / "neural_lm_scores_test_other_G128.jsonl"
    test_nbest = args.output_dir / "nbest_test_other_G128.jsonl"

    if test_scored.exists():
        test_path = test_scored
    elif test_nbest.exists():
        test_path = test_nbest
    else:
        raise FileNotFoundError(
            f"No test-other G=128 file found. Run --steps generate,score first.\n"
            f"  Looked for: {test_scored}\n"
            f"  Or fallback: {test_nbest}"
        )

    print(f"  Loading test-other G=128: {test_path}")
    test_records = []
    with open(test_path) as f:
        for line in f:
            test_records.append(json.loads(line))
    print(f"  Loaded {len(test_records)} test-other records "
          f"(avg {np.mean([r['num_candidates'] for r in test_records]):.0f} cands)")

    # Evaluate test-other G=128
    test_results = evaluate_split(
        test_records, "test-other", args.n_bootstrap, args.seed
    )

    dev_results = None
    dev_g128_path = args.dev_g128_path
    if dev_g128_path and dev_g128_path.exists():
        print(f"\n\n  Loading dev-other G=128: {dev_g128_path}")
        dev_records = []
        with open(dev_g128_path) as f:
            for line in f:
                dev_records.append(json.loads(line))
        print(f"  Loaded {len(dev_records)} dev-other records")

        dev_results = evaluate_split(
            dev_records, "dev-other", args.n_bootstrap, args.seed
        )
    else:
        print(f"\n  SKIP E1b: dev-other G=128 file not found at {dev_g128_path}")

    print("\n\n--- Writing outputs ---")
    write_all_outputs(args, test_results, dev_results)

def format_bootstrap_table(results, title):
    """Format bootstrap results as markdown table."""
    lines = [f"### {title}", ""]
    lines.append("| Method | WER (%) | delta (pp) | p-value | 95% CI (pp) | alpha=0.05 | alpha=0.01 | N differ |")
    lines.append("|--------|--------:|-------:|--------:|------------:|:------:|:------:|---------:|")
    for r in sorted(results, key=lambda x: x["wer_method"]):
        s05 = "" if r["significant_005"] else " -- "
        s01 = "" if r["significant_001"] else " -- "
        p_str = "<0.0001" if r["p_value"] < 0.0001 else f"{r['p_value']:.4f}"
        lines.append(
            f"| {r['method']} | {r['wer_method']*100:.2f} | "
            f"{r['delta_pp']:+.3f} | {p_str} | "
            f"[{r['ci_lower']:+.3f}, {r['ci_upper']:+.3f}] | {s05} | {s01} | "
            f"{r['n_utterances_differ']} |"
        )
    lines.append("")
    return lines

def write_master_comparison(args, test_res, dev_res):
    """The crown jewel: master table across all splits x G values x methods."""
    lines = ["# Master Comparison: G=16 vs G=128, dev-other vs test-other", ""]
    lines.append("All methods evaluated against greedy CTC baseline (B=10,000 bootstrap).")
    lines.append("")

    # Known G=16 results from E1/E2
    g16_dev = {
        "MBR-CER + PLL tau=10": {"wer": 5.79, "p": "<0.0001", "sig": True},
        "RoBERTa PLL interp alpha=0.7": {"wer": 5.92, "p": "0.0019", "sig": True},
        "GPT-2 interp alpha=0.8": {"wer": 5.99, "p": "0.0238", "sig": True},
        "MBR-CER tau=50 (CTC)": {"wer": 5.99, "p": "0.1630", "sig": False},
        "MBR-CER tau=inf (CTC)": {"wer": 5.99, "p": "0.1812", "sig": False},
    }
    g16_test = {
        "MBR-CER + PLL tau=10": {"wer": 5.77, "p": "0.0003", "sig": True},
        "RoBERTa PLL interp alpha=0.7": {"wer": 5.85, "p": "0.0007", "sig": True},
        "GPT-2 interp alpha=0.8": {"wer": 5.91, "p": "0.0015", "sig": True},
        "MBR-CER tau=50 (CTC)": {"wer": 5.92, "p": "0.1535", "sig": False},
        "MBR-CER tau=inf (CTC)": {"wer": 5.92, "p": "0.1706", "sig": False},
    }

    # G=128 results
    g128_test_b = {r["method"]: r for r in test_res["bootstrap"]}
    g128_dev_b = {r["method"]: r for r in dev_res["bootstrap"]} if dev_res else {}

    lines.append("## Headline WERs")
    lines.append("")
    lines.append("| Split | G | Greedy | Oracle | Gap (pp) | Rel gap |")
    lines.append("|-------|--:|-------:|-------:|---------:|--------:|")
    lines.append("| dev-other  | 16  | 6.02% | 4.44% | 1.58 | 26.2% |")
    lines.append("| test-other | 16  | 5.96% | 4.41% | 1.55 | 26.0% |")

    # G=128
    if dev_res:
        dg = dev_res["method_wers"]["Greedy"] * 100
        do = dev_res["method_wers"]["Oracle"] * 100
        dgap = dg - do
        drel = dgap / dg * 100
        lines.append(f"| dev-other  | 128 | {dg:.2f}% | {do:.2f}% | {dgap:.2f} | {drel:.1f}% |")

    tg = test_res["method_wers"]["Greedy"] * 100
    to = test_res["method_wers"]["Oracle"] * 100
    tgap = tg - to
    trel = tgap / tg * 100
    lines.append(f"| test-other | 128 | {tg:.2f}% | {to:.2f}% | {tgap:.2f} | {trel:.1f}% |")
    lines.append("")

    # Main comparison table
    lines.append("## Per-Method Results (sorted by best WER)")
    lines.append("")
    lines.append("| Method | G16 dev | G16 test | G128 dev | G128 test | Best config |")
    lines.append("|--------|--------:|---------:|---------:|----------:|:-----------:|")

    method_map = [
        ("MBR-CER + PLL tau=10", "MBR-CER + PLL tau=10", "MBR-CER + PLL tau=10"),
        ("RoBERTa PLL interp alpha=0.7", "RoBERTa PLL interp alpha=0.7", "RoBERTa PLL interp alpha=0.7"),
        ("RoBERTa PLL interp alpha=0.8", None, "RoBERTa PLL interp alpha=0.8"),
        ("GPT-2 interp alpha=0.8", "GPT-2 interp alpha=0.8", "GPT-2 interp alpha=0.8"),
        ("GPT-2 interp alpha=0.7", None, "GPT-2 interp alpha=0.7"),
        ("MBR-CER tau=50 (CTC)", "MBR-CER tau=50 (CTC)", "MBR-CER tau=50 (CTC)"),
        ("MBR-CER tau=inf (CTC)", "MBR-CER tau=inf (CTC)", "MBR-CER tau=inf (CTC)"),
        ("MBR-CER + PLL tau=5", None, "MBR-CER + PLL tau=5"),
        ("MBR-CER + PLL tau=50", None, "MBR-CER + PLL tau=50"),
        ("MBR-CER + PLL tau=inf", None, "MBR-CER + PLL tau=inf"),
    ]

    for g128_name, g16_name, _ in method_map:
        g16d = g16_dev.get(g16_name, {}).get("wer", None) if g16_name else None
        g16t = g16_test.get(g16_name, {}).get("wer", None) if g16_name else None
        g128d_row = g128_dev_b.get(g128_name)
        g128t_row = g128_test_b.get(g128_name)

        g16d_s = f"{g16d:.2f}%" if g16d else " -- "
        g16t_s = f"{g16t:.2f}%" if g16t else " -- "
        g128d_s = f"{g128d_row['wer_method']*100:.2f}%" if g128d_row else " -- "
        g128t_s = f"{g128t_row['wer_method']*100:.2f}%" if g128t_row else " -- "

        vals = []
        if g16d:
            vals.append(("G16 dev", g16d))
        if g16t:
            vals.append(("G16 test", g16t))
        if g128d_row:
            vals.append(("G128 dev", g128d_row["wer_method"] * 100))
        if g128t_row:
            vals.append(("G128 test", g128t_row["wer_method"] * 100))
        best = min(vals, key=lambda x: x[1])[0] if vals else " -- "

        lines.append(f"| {g128_name} | {g16d_s} | {g16t_s} | {g128d_s} | {g128t_s} | {best} |")

    lines.append("")

    # Significance summary
    lines.append("## Significance Summary (alpha=0.05)")
    lines.append("")
    lines.append("| Method | G16 dev | G16 test | G128 dev | G128 test |")
    lines.append("|--------|:-------:|:--------:|:--------:|:---------:|")

    for g128_name, g16_name, _ in method_map:
        g16d_sig = "" if g16_dev.get(g16_name, {}).get("sig") else " -- "
        g16t_sig = "" if g16_test.get(g16_name, {}).get("sig") else " -- "
        g128d_sig = "" if g128_dev_b.get(g128_name, {}).get("significant_005") else " -- "
        g128t_sig = "" if g128_test_b.get(g128_name, {}).get("significant_005") else " -- "
        if not g16_name:
            g16d_sig = " -- "
            g16t_sig = " -- "
        lines.append(f"| {g128_name} | {g16d_sig} | {g16t_sig} | {g128d_sig} | {g128t_sig} |")

    lines.append("")

    # Spearman comparison
    lines.append("## Spearman rho Comparison (G=16 vs G=128)")
    lines.append("")
    lines.append("| Scorer | G16 dev | G16 test | G128 dev | G128 test |")
    lines.append("|--------|--------:|---------:|---------:|----------:|")

    g16_dev_sp = {"CTC log-prob": -0.3474, "RoBERTa PLL": -0.4844,
                  "GPT-2 LL": -0.4005, "Interpolated (alpha=0.6 CTC + 0.4 PLL)": -0.5270}
    g16_test_sp = {"CTC log-prob": -0.3385, "RoBERTa PLL": -0.4747,
                   "GPT-2 LL": -0.3934, "Interpolated (alpha=0.6 CTC + 0.4 PLL)": -0.5165}

    test_sp = test_res.get("spearman", {})
    dev_sp = dev_res.get("spearman", {}) if dev_res else {}

    for scorer in ["CTC log-prob", "RoBERTa PLL", "GPT-2 LL",
                   "Interpolated (alpha=0.6 CTC + 0.4 PLL)"]:
        g16d = g16_dev_sp.get(scorer)
        g16t = g16_test_sp.get(scorer)
        g128d = dev_sp.get(scorer, {}).get("mean")
        g128t = test_sp.get(scorer, {}).get("mean")

        g16d_s = f"{g16d:+.4f}" if g16d else " -- "
        g16t_s = f"{g16t:+.4f}" if g16t else " -- "
        g128d_s = f"{g128d:+.4f}" if g128d else " -- "
        g128t_s = f"{g128t:+.4f}" if g128t else " -- "
        lines.append(f"| {scorer} | {g16d_s} | {g16t_s} | {g128d_s} | {g128t_s} |")

    lines.append("")

    # Interpretation
    lines.append("## Key Finding")
    lines.append("")

    # Check if headline generalizes
    headline_test = g128_test_b.get("MBR-CER + PLL tau=10")
    if headline_test:
        h_wer = headline_test["wer_method"] * 100
        h_p = headline_test["p_value"]
        h_sig = headline_test["significant_005"]
        if h_sig:
            lines.append(f"> **The headline result generalizes.** MBR-CER + RoBERTa PLL tau=10 "
                         f"achieves {h_wer:.2f}% WER on test-other G=128 "
                         f"(p={'<0.0001' if h_p < 0.0001 else f'{h_p:.4f}'}), "
                         f"confirming the dev-other finding (5.53%) is not overfit.")
        else:
            lines.append(f"> **Caution:** MBR-CER + RoBERTa PLL tau=10 achieves {h_wer:.2f}% "
                         f"on test-other G=128 but p={h_p:.4f}  --  "
                         f"does {'not ' if not h_sig else ''}reach significance.")
    lines.append("")

    p = args.output_dir / "master_comparison.md"
    with open(p, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {p}")

def write_all_outputs(args, test_results, dev_results):
    """Write all output files."""
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Test-other G=128 JSON
    n_test = len(test_results["bootstrap"])
    json_out = {
        "metadata": {
            "split": "test-other",
            "G": 128,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "date": time.strftime("%Y-%m-%d"),
        },
        "method_wers": {k: round(v, 6) for k, v in test_results["method_wers"].items()},
        "bootstrap": test_results["bootstrap"],
        "spearman": test_results["spearman"],
        "verification": [
            {"check": c[0], "measured": c[1], "expected": c[2], "pass": c[3]}
            for c in test_results["checks"]
        ],
    }
    p = args.output_dir / "test_other_g128_results.json"
    with open(p, "w") as f:
        json.dump(json_out, f, indent=2, default=str)
    print(f"  Wrote {p}")

    # 2. Dev-other G=128 JSON (E1b)
    if dev_results:
        json_out_dev = {
            "metadata": {
                "split": "dev-other",
                "G": 128,
                "n_bootstrap": args.n_bootstrap,
                "seed": args.seed,
                "date": time.strftime("%Y-%m-%d"),
            },
            "method_wers": {k: round(v, 6) for k, v in dev_results["method_wers"].items()},
            "bootstrap": dev_results["bootstrap"],
            "spearman": dev_results["spearman"],
            "verification": [
                {"check": c[0], "measured": c[1], "expected": c[2], "pass": c[3]}
                for c in dev_results["checks"]
            ],
        }
        p = args.output_dir / "dev_other_g128_results.json"
        with open(p, "w") as f:
            json.dump(json_out_dev, f, indent=2, default=str)
        print(f"  Wrote {p}")

    # 3. Bootstrap markdown tables
    p = args.output_dir / "bootstrap_g128_summary.md"
    lines = ["# G=128 Paired Bootstrap Results", ""]
    lines.append(f"Bootstrap B={args.n_bootstrap}, seed={args.seed}.")
    lines.append("")
    lines += format_bootstrap_table(
        test_results["bootstrap"], "test-other G=128 vs Greedy"
    )
    if dev_results:
        lines += format_bootstrap_table(
            dev_results["bootstrap"], "dev-other G=128 vs Greedy (E1b)"
        )
    with open(p, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {p}")

    # 4. Master comparison
    write_master_comparison(args, test_results, dev_results)

    # 5. CSV for quick reference
    import csv
    p = args.output_dir / "test_other_g128_bootstrap.csv"
    fields = ["method", "wer_pct", "delta_pp", "p_value", "ci_lower",
              "ci_upper", "significant_005", "significant_001", "n_differ"]
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(test_results["bootstrap"], key=lambda x: x["wer_method"]):
            w.writerow({
                "method": r["method"],
                "wer_pct": f"{r['wer_method']*100:.4f}",
                "delta_pp": f"{r['delta_pp']:+.3f}",
                "p_value": f"{r['p_value']:.4f}",
                "ci_lower": f"{r['ci_lower']:+.3f}",
                "ci_upper": f"{r['ci_upper']:+.3f}",
                "significant_005": r["significant_005"],
                "significant_001": r["significant_001"],
                "n_differ": r["n_utterances_differ"],
            })
    print(f"  Wrote {p}")

def parse_args():
    p = argparse.ArgumentParser(
        description="E2c: test-other G=128 evaluation + E1b dev-other G=128 bootstrap"
    )
    p.add_argument("--data-dir", type=Path,
                   default=Path("/content/librispeech_data"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/test_other_g128"))
    p.add_argument("--dev-g128-path", type=Path,
                   default=Path("rbpo/results/g128/neural_lm_scores.jsonl"),
                   help="Path to dev-other G=128 scored JSONL (for E1b)")
    p.add_argument("--model-dir", type=Path,
                   default=Path("/content/icefall-asr-librispeech-zipformer-small-cr-ctc"))
    p.add_argument("--icefall-dir", type=Path, default=Path("/content/icefall"))
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--steps", type=str, default="all",
                   help="Comma-separated: generate,score,evaluate or 'all'")
    p.add_argument("--force", action="store_true")
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
    print("E2c: Test-Other G=128 Evaluation + E1b Dev-Other G=128 Bootstrap")
    print("=" * 70)
    print(f"  Data dir:        {args.data_dir}")
    print(f"  Output dir:      {args.output_dir}")
    print(f"  Dev G=128 file:  {args.dev_g128_path}")
    print(f"  Steps:           {steps}")
    print(f"  Generation:      G={G128}, oversample={NUM_PATHS_OVERSAMPLE_128}")
    print(f"  Bootstrap:       B={args.n_bootstrap}, seed={args.seed}")

    if "generate" in steps:
        print(f"\n--- STEP 1/3: Generate G=128 N-best for test-other ---")
        step_generate(args)

    if "score" in steps:
        print(f"\n--- STEP 2/3: Score with RoBERTa PLL + GPT-2 LL ---")
        step_score(args)

    if "evaluate" in steps:
        print(f"\n--- STEP 3/3: Evaluate + E1b bootstrap ---")
        step_evaluate(args)

    print("\n" + "=" * 70)
    print("DONE.")
    print("=" * 70)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""B3: dev-clean G=128 full pipeline (N-best -> PLL -> MBR + interp + bootstrap).

Three stages:
  1. Generate G=128 N-best via CTC lattice sampling
  2. Score with RoBERTa PLL
  3. Evaluate: greedy, oracle, MBR-CER+PLL, CTC+PLL interp, Spearman, bootstrap

Usage (Colab):
    python experiments/evaluation/run_dev_clean_g128.py \
        --checkpoint /path/to/pretrained.pt \
        --manifest /path/to/dev-clean-cuts.jsonl.gz \
        --roberta-model roberta-base \
        --output-dir results/dev_clean_g128/ \
        --nbest-scale 1.0 --oversample 512 --max-G 128 --tau 10
"""

import argparse
import csv
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BLANK_ID = 0
MAX_TOKEN = 499
VOCAB_SIZE = 500

_TAG_RE = re.compile(r"\{[^}]+\}|<[^>]+>")
_MULTI_SPACE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

def normalize_text(text: str) -> str:
    text = _TAG_RE.sub(" ", text)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text

def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records

def save_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="B3: dev-clean G=128 full pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--icefall-dir", type=Path,
                        default=Path("/content/icefall"))
    parser.add_argument("--bpe", type=Path,
                        default=Path("/content/icefall/egs/librispeech/ASR/data/lang_bpe_500/bpe.model"))
    parser.add_argument("--roberta-model", type=str, default="roberta-base")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "results" / "dev_clean_g128")
    parser.add_argument("--nbest-scale", type=float, default=1.0)
    parser.add_argument("--oversample", type=int, default=512)
    parser.add_argument("--max-G", type=int, default=128)
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    assert args.nbest_scale == 1.0, "nbest_scale must be 1.0 (canonical)"

    print("=" * 70)
    print("B3: dev-clean G=128 Full Pipeline")
    print("=" * 70)
    t0 = time.time()

    nbest_path = args.output_dir / "nbest_dev_clean_G128.jsonl"
    scored_path = args.output_dir / "neural_lm_scores_dev_clean_G128.jsonl"

    if nbest_path.exists():
        print(f"\nStage 1: SKIP (N-best exists: {nbest_path})")
        records = load_jsonl(nbest_path)
    else:
        print(f"\nStage 1: Generate N-best (G={args.max_G}, scale={args.nbest_scale})")
        import torch
        import sentencepiece as spm
        import k2
        from lhotse import CutSet, Fbank, FbankConfig

        from scripts.generate_nbest import (
            load_model, ctc_collapse, alignment_log_prob,
            generate_nbest_for_utt, normalize_text,
        )

        model = load_model(args.checkpoint, args.icefall_dir, args.device)
        sp = spm.SentencePieceProcessor()
        sp.load(str(args.bpe))
        fbank = Fbank(FbankConfig(num_mel_bins=80))
        topo = k2.ctc_topo(max_token=MAX_TOKEN, modified=False).to(args.device)

        cuts = CutSet.from_file(args.manifest)
        records = []

        for i, cut in enumerate(cuts):
            audio = cut.load_audio()
            feat = fbank.extract(audio, sampling_rate=16000)
            feat_tensor = torch.from_numpy(feat).unsqueeze(0).to(args.device)
            feat_lens = torch.tensor([feat.shape[0]], dtype=torch.int64, device=args.device)

            with torch.no_grad():
                encoder_out, encoder_out_lens = model.forward_encoder(feat_tensor, feat_lens)
                log_probs = model.ctc_output(encoder_out)

            lp = log_probs[0, :encoder_out_lens[0].item()]

            candidates = generate_nbest_for_utt(
                lp, topo, num_paths=args.oversample,
                nbest_scale=args.nbest_scale, sp=sp, device=args.device,
            )
            candidates = candidates[:args.max_G]

            ref_text = normalize_text(cut.supervisions[0].text)
            rec = {
                "utt_id": cut.id,
                "ref": ref_text,
                "nbest": candidates,
            }
            records.append(rec)

            if (i + 1) % 100 == 0:
                print(f"  Stage 1: {i+1} utterances done")

        save_jsonl(records, nbest_path)
        print(f"  Saved {len(records)} utterances to {nbest_path}")

    if scored_path.exists():
        print(f"\nStage 2: SKIP (scored exists: {scored_path})")
        records = load_jsonl(scored_path)
    else:
        print(f"\nStage 2: Score with RoBERTa PLL")
        import torch
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        from scripts.score_pll import compute_pll

        tokenizer = AutoTokenizer.from_pretrained(args.roberta_model)
        lm_model = AutoModelForMaskedLM.from_pretrained(args.roberta_model)
        lm_model.eval().to(args.device)

        for i, rec in enumerate(records):
            for h in rec["nbest"]:
                h["pll_score"] = compute_pll(
                    h["hyp"], tokenizer, lm_model, args.device,
                    batch_size=args.batch_size,
                )
            if (i + 1) % 100 == 0:
                print(f"  Stage 2: {i+1}/{len(records)} utterances scored")

        save_jsonl(records, scored_path)
        print(f"  Saved scored data to {scored_path}")

    print(f"\nStage 3: Evaluate")
    import editdistance
    from scipy import stats
    from experiments.significance_tests import paired_bootstrap_wer

    ref_words = [rec["ref"].split() for rec in records]
    greedy_words = [rec["nbest"][0]["hyp"].split() for rec in records]

    def corpus_wer_fn(hw):
        total_e = sum(editdistance.eval(hw[i], ref_words[i]) for i in range(len(hw)))
        total_r = sum(len(r) for r in ref_words)
        return total_e / max(total_r, 1)

    greedy_wer = corpus_wer_fn(greedy_words)
    oracle_words = []
    for rec in records:
        ref = rec["ref"]
        ref_w = ref.split()
        best = min(rec["nbest"], key=lambda h: editdistance.eval(h["hyp"].split(), ref_w))
        oracle_words.append(best["hyp"].split())
    oracle_wer = corpus_wer_fn(oracle_words)

    print(f"  Greedy WER:  {greedy_wer*100:.4f}%")
    print(f"  Oracle WER:  {oracle_wer*100:.4f}%")

    mbr_words = []
    for rec in records:
        nbest = rec["nbest"]
        texts = [h["hyp"] for h in nbest]
        log_scores = np.array([h["pll_score"] for h in nbest])
        n = len(texts)

        cer_mat = np.zeros((n, n), dtype=np.float32)
        for ii in range(n):
            for jj in range(ii + 1, n):
                d = editdistance.eval(list(texts[ii]), list(texts[jj]))
                denom = max(len(texts[ii]), len(texts[jj]), 1)
                cer_mat[ii, jj] = d / denom
                cer_mat[jj, ii] = cer_mat[ii, jj]

        scaled = log_scores / args.tau
        scaled -= np.max(scaled)
        weights = np.exp(scaled)
        weights /= weights.sum()
        risk = cer_mat @ weights
        idx = int(np.argmin(risk))
        mbr_words.append(texts[idx].split())

    mbr_wer = corpus_wer_fn(mbr_words)
    print(f"  MBR WER:     {mbr_wer*100:.4f}%  (delta={(mbr_wer-greedy_wer)*100:+.4f}pp)")

    interp_results = {}
    best_alpha = None
    best_interp_wer = float("inf")
    for alpha in [0.5, 0.6, 0.7, 0.8, 0.9]:
        iw = []
        for rec in records:
            nbest = rec["nbest"]
            scores = [alpha * h["score"] + (1 - alpha) * h["pll_score"] for h in nbest]
            idx = int(np.argmax(scores))
            iw.append(nbest[idx]["hyp"].split())
        w = corpus_wer_fn(iw)
        interp_results[str(alpha)] = float(w)
        if w < best_interp_wer:
            best_interp_wer = w
            best_alpha = alpha
        print(f"  Interp alpha={alpha}: WER={w*100:.4f}%")

    ctc_rhos = []
    pll_rhos = []
    for rec in records:
        nbest = rec["nbest"]
        ref = rec["ref"]
        ref_w = ref.split()
        ref_len = len(ref_w)
        if ref_len == 0 or len(nbest) < 3:
            continue
        wers = [editdistance.eval(h["hyp"].split(), ref_w) / ref_len for h in nbest]
        if len(set(wers)) < 2:
            continue
        ctc_s = [h["score"] for h in nbest]
        pll_s = [h["pll_score"] for h in nbest]
        if len(set(ctc_s)) >= 2:
            r, _ = stats.spearmanr(ctc_s, wers)
            if not np.isnan(r):
                ctc_rhos.append(r)
        if len(set(pll_s)) >= 2:
            r, _ = stats.spearmanr(pll_s, wers)
            if not np.isnan(r):
                pll_rhos.append(r)

    print(f"  Spearman rho(CTC): median={np.median(ctc_rhos):+.4f}, mean={np.mean(ctc_rhos):+.4f}")
    print(f"  Spearman rho(PLL): median={np.median(pll_rhos):+.4f}, mean={np.mean(pll_rhos):+.4f}")

    boot_mbr = paired_bootstrap_wer(ref_words, mbr_words, greedy_words,
                                     n_bootstrap=10000, seed=42)
    best_iw = []
    for rec in records:
        nbest = rec["nbest"]
        scores = [best_alpha * h["score"] + (1 - best_alpha) * h["pll_score"] for h in nbest]
        idx = int(np.argmax(scores))
        best_iw.append(nbest[idx]["hyp"].split())
    boot_interp = paired_bootstrap_wer(ref_words, best_iw, greedy_words,
                                        n_bootstrap=10000, seed=42)

    result = {
        "n_utterances": len(records),
        "greedy_wer": float(greedy_wer),
        "oracle_wer": float(oracle_wer),
        "mbr_wer": float(mbr_wer),
        "mbr_delta_pp": boot_mbr["delta"] * 100,
        "mbr_p_value": boot_mbr["p_value"],
        "mbr_ci_pp": [boot_mbr["ci_lower"] * 100, boot_mbr["ci_upper"] * 100],
        "best_interp_alpha": best_alpha,
        "best_interp_wer": float(best_interp_wer),
        "interp_delta_pp": boot_interp["delta"] * 100,
        "interp_p_value": boot_interp["p_value"],
        "interp_ci_pp": [boot_interp["ci_lower"] * 100, boot_interp["ci_upper"] * 100],
        "interpolation": interp_results,
        "spearman_ctc": {"median": float(np.median(ctc_rhos)), "mean": float(np.mean(ctc_rhos))},
        "spearman_pll": {"median": float(np.median(pll_rhos)), "mean": float(np.mean(pll_rhos))},
    }

    json_path = args.output_dir / "dev_clean_g128_results.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Wrote {json_path}")

    boot_csv = args.output_dir / "dev_clean_g128_bootstrap.csv"
    with open(boot_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "wer", "delta_pp", "p_value", "ci_lower", "ci_upper"])
        w.writeheader()
        w.writerow({"method": "MBR-CER+PLL", "wer": f"{mbr_wer*100:.4f}",
                     "delta_pp": f"{boot_mbr['delta']*100:.4f}",
                     "p_value": f"{boot_mbr['p_value']:.4f}",
                     "ci_lower": f"{boot_mbr['ci_lower']*100:.4f}",
                     "ci_upper": f"{boot_mbr['ci_upper']*100:.4f}"})
        w.writerow({"method": f"Interp alpha={best_alpha}", "wer": f"{best_interp_wer*100:.4f}",
                     "delta_pp": f"{boot_interp['delta']*100:.4f}",
                     "p_value": f"{boot_interp['p_value']:.4f}",
                     "ci_lower": f"{boot_interp['ci_lower']*100:.4f}",
                     "ci_upper": f"{boot_interp['ci_upper']*100:.4f}"})
    print(f"  Wrote {boot_csv}")

    elapsed = time.time() - t0
    print(f"\nDone. Total time: {elapsed:.1f}s")

if __name__ == "__main__":
    main()

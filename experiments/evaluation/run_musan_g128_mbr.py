#!/usr/bin/env python3
"""B4: MUSAN G=128 N-best generation + PLL scoring + MBR reranking.

End-to-end pipeline for MUSAN-augmented dev-other at a given SNR:
  1. (if needed) Generate G=128 N-best with noise augmentation
  2. (if needed) Score with RoBERTa PLL
  3. Run MBR-CER+PLL and CTC+PLL interpolation
  4. Paired bootstrap

Usage (Colab):
    python experiments/evaluation/run_musan_g128_mbr.py \
        --checkpoint /path/to/pretrained.pt \
        --manifest /path/to/dev-other-cuts.jsonl.gz \
        --musan-dir /path/to/musan/ \
        --snr 10 \
        --roberta-model roberta-base \
        --output-dir results/musan_rerun/ \
        --nbest-scale 1.0 --max-G 128 --tau 10
"""

import argparse
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
        description="B4: MUSAN G=128 MBR pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="dev-other CutSet (clean) or pre-augmented cuts")
    parser.add_argument("--musan-dir", type=Path, default=None,
                        help="MUSAN noise dir (only needed if --augmented-cuts not given)")
    parser.add_argument("--augmented-cuts", type=Path, default=None,
                        help="Pre-made augmented CutSet (skips noise mixing)")
    parser.add_argument("--icefall-dir", type=Path,
                        default=Path("/content/icefall"))
    parser.add_argument("--bpe", type=Path,
                        default=Path("/content/icefall/egs/librispeech/ASR/data/lang_bpe_500/bpe.model"))
    parser.add_argument("--snr", type=int, default=10, help="Target SNR in dB")
    parser.add_argument("--roberta-model", type=str, default="roberta-base")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "results" / "musan_rerun")
    parser.add_argument("--nbest-scale", type=float, default=1.0)
    parser.add_argument("--oversample", type=int, default=512)
    parser.add_argument("--max-G", type=int, default=128)
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    assert args.nbest_scale == 1.0, "nbest_scale must be 1.0 (canonical)"

    snr = args.snr
    nbest_path = args.output_dir / f"nbest_{snr}dB_g128.jsonl"
    scored_path = args.output_dir / f"nbest_{snr}dB_g128_pll.jsonl"

    print("=" * 70)
    print(f"B4: MUSAN {snr}dB G=128 MBR Pipeline")
    print("=" * 70)
    t0 = time.time()

    if nbest_path.exists():
        print(f"\nStage 1: SKIP (N-best exists: {nbest_path})")
        records = load_jsonl(nbest_path)
    else:
        print(f"\nStage 1: Generate MUSAN-augmented N-best (SNR={snr}dB, G={args.max_G})")
        import torch
        import sentencepiece as spm
        import k2
        from lhotse import CutSet, Fbank, FbankConfig

        from scripts.generate_nbest import load_model, generate_nbest_for_utt

        model = load_model(args.checkpoint, args.icefall_dir, args.device)
        sp = spm.SentencePieceProcessor()
        sp.load(str(args.bpe))
        fbank = Fbank(FbankConfig(num_mel_bins=80))
        topo = k2.ctc_topo(max_token=MAX_TOKEN, modified=False).to(args.device)

        if args.augmented_cuts and args.augmented_cuts.exists():
            print(f"  Using pre-augmented cuts: {args.augmented_cuts}")
            cuts = CutSet.from_file(str(args.augmented_cuts))
        else:
            print(f"  Creating augmented cuts from {args.manifest} + {args.musan_dir}")
            assert args.musan_dir is not None, "--musan-dir required when no --augmented-cuts"
            cuts_path = args.output_dir / f"cuts_{snr}dB.jsonl.gz"
            import subprocess
            subprocess.run([
                sys.executable, str(REPO_ROOT / "scripts" / "prepare_musan_cuts.py"),
                "--ls-cuts", str(args.manifest),
                "--musan-dir", str(args.musan_dir),
                "--snr", str(snr),
                "--output", str(cuts_path),
            ], check=True)
            cuts = CutSet.from_file(str(cuts_path))
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
            records.append({
                "utt_id": cut.id.split("_musan")[0] if "_musan" in cut.id else cut.id,
                "ref": ref_text,
                "nbest": candidates,
            })

            if (i + 1) % 200 == 0:
                print(f"  {i+1} utterances done")

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
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(records)} utterances scored")

        save_jsonl(records, scored_path)
        print(f"  Saved scored data to {scored_path}")

    print(f"\nStage 3: MBR-CER+PLL tau={args.tau} and CTC+PLL interp")
    import editdistance
    from experiments.significance_tests import paired_bootstrap_wer

    ref_words = [r["ref"].split() for r in records]
    greedy_words = [r["nbest"][0]["hyp"].split() for r in records]

    def corpus_wer_fn(hw):
        total_e = sum(editdistance.eval(hw[i], ref_words[i]) for i in range(len(hw)))
        total_r = sum(len(r) for r in ref_words)
        return total_e / max(total_r, 1)

    greedy_wer = corpus_wer_fn(greedy_words)
    print(f"  Greedy WER: {greedy_wer*100:.4f}%")

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
    print(f"  MBR WER:    {mbr_wer*100:.4f}%  (delta={(mbr_wer-greedy_wer)*100:+.4f}pp)")

    interp_results = {}
    for alpha in [0.5, 0.6, 0.7, 0.8, 0.9]:
        iw = []
        for rec in records:
            nbest = rec["nbest"]
            scores = [alpha * h["score"] + (1 - alpha) * h["pll_score"] for h in nbest]
            idx = int(np.argmax(scores))
            iw.append(nbest[idx]["hyp"].split())
        w = corpus_wer_fn(iw)
        interp_results[str(alpha)] = float(w)
        print(f"  Interp alpha={alpha}: WER={w*100:.4f}%")

    print(f"\nStage 4: Paired bootstrap (B=10000)")
    boot = paired_bootstrap_wer(ref_words, mbr_words, greedy_words,
                                 n_bootstrap=10000, seed=42)

    result = {
        "snr_dB": snr,
        "n_utterances": len(records),
        "greedy_wer": float(greedy_wer),
        "mbr_wer": float(mbr_wer),
        "mbr_delta_pp": boot["delta"] * 100,
        "mbr_p_value": boot["p_value"],
        "mbr_ci_pp": [boot["ci_lower"] * 100, boot["ci_upper"] * 100],
        "interpolation": interp_results,
    }

    result_path = args.output_dir / f"musan_{snr}dB_g128_results.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Wrote {result_path}")

    boot_path = args.output_dir / f"musan_{snr}dB_g128_bootstrap.json"
    with open(boot_path, "w") as f:
        json.dump(boot, f, indent=2)
    print(f"  Wrote {boot_path}")

    elapsed = time.time() - t0
    print(f"\nDone. Total time: {elapsed:.1f}s")

if __name__ == "__main__":
    main()

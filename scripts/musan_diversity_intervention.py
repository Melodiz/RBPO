#!/usr/bin/env python3
"""R-MUSAN0: Test whether diversity explains MUSAN 0dB MBR failure.

For each condition (baseline, wide_beam, flat_sampling):
  1. Generate CTC lattice from MUSAN 0dB audio
  2. Sample N-best with condition-specific parameters
  3. Score with RoBERTa PLL
  4. Run MBR-CER+PLL tau=10
  5. Report: oracle, MBR, greedy, diversity metrics

The encoder forward pass is shared across all conditions  --  only the
lattice construction (output_beam) and path sampling (nbest_scale,
oversample) differ.

Conditions:
  baseline:      output_beam=8.0,  nbest_scale=1.0, oversample=64
  wide_beam:     output_beam=20.0, nbest_scale=1.0, oversample=128
  flat_sampling: output_beam=8.0,  nbest_scale=0.5, oversample=256

Usage:
    python scripts/musan_diversity_intervention.py \
        --checkpoint /path/to/pretrained.pt \
        --augmented-cuts /path/to/cuts_0dB.jsonl.gz \
        --output-dir /path/to/results/R_musan_diversity/ \
        [--manifest /path/to/dev-other.jsonl.gz --musan-dir /path/to/musan/noise] \
        [--G 16] [--tau 10] [--device cuda]
"""

import argparse
import csv
import gc
import json
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

CONDITIONS = {
    "baseline": {
        "output_beam": 8.0,
        "nbest_scale": 1.0,
        "oversample": 64,
    },
    "wide_beam": {
        "output_beam": 20.0,
        "nbest_scale": 1.0,
        "oversample": 128,
    },
    "flat_sampling": {
        "output_beam": 8.0,
        "nbest_scale": 0.5,
        "oversample": 256,
    },
}

def normalize_text(text: str) -> str:
    text = _TAG_RE.sub(" ", text)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text

def ctc_collapse(token_ids):
    result = []
    prev = None
    for t in token_ids:
        if t != BLANK_ID and t != prev:
            result.append(t)
        prev = t
    return result

def alignment_log_prob(label_seq, log_probs_cpu):
    import torch
    T = log_probs_cpu.shape[0]
    if len(label_seq) != T:
        return float("-inf")
    idx = torch.tensor(label_seq, dtype=torch.long)
    return log_probs_cpu[torch.arange(T), idx].sum().item()

def generate_nbest_parameterized(
    log_probs_utt, topo, num_paths, nbest_scale, output_beam, sp, device,
):
    """Build lattice and extract N-best with variable output_beam and nbest_scale.

    Identical to generate_nbest.generate_nbest_for_utt except output_beam
    is a parameter instead of hardcoded 8.0.
    """
    import k2
    import torch

    T = log_probs_utt.shape[0]
    lp_cpu = log_probs_utt.cpu()

    greedy_ids = log_probs_utt.argmax(dim=-1).cpu().tolist()
    greedy_collapsed = ctc_collapse(greedy_ids)
    greedy_text = normalize_text(sp.decode(greedy_collapsed))
    greedy_score = alignment_log_prob(greedy_ids, lp_cpu)

    supervision_segments = torch.tensor([[0, 0, T]], dtype=torch.int32)
    dense_fsa = k2.DenseFsaVec(log_probs_utt.unsqueeze(0), supervision_segments)
    lattice = k2.intersect_dense(topo, dense_fsa, output_beam=output_beam)
    lattice = k2.connect(lattice)

    nbest = k2.Nbest.from_lattice(
        lattice,
        num_paths=num_paths,
        use_double_scores=True,
        nbest_scale=nbest_scale,
    )

    all_labels = nbest.fsa.labels.cpu().tolist()
    paths = []
    current = []
    for label in all_labels:
        if label == -1:
            paths.append(current)
            current = []
        else:
            current.append(label)

    seen = {}
    for raw_ids in paths:
        score = alignment_log_prob(raw_ids, lp_cpu)
        if score == float("-inf"):
            continue
        token_ids = ctc_collapse(raw_ids)
        text = normalize_text(sp.decode(token_ids))
        if not text:
            continue
        entry = {"hyp": text, "score": round(score, 6)}
        if text not in seen or score > seen[text]["score"]:
            seen[text] = entry

    greedy_entry = {"hyp": greedy_text, "score": round(greedy_score, 6)}
    seen[greedy_text] = greedy_entry

    candidates = sorted(seen.values(), key=lambda c: c["score"], reverse=True)
    rest = [c for c in candidates if c["hyp"] != greedy_text]
    candidates = [greedy_entry] + rest

    del lattice, nbest
    return candidates

def compute_mean_pairwise_wer(texts):
    """Mean pairwise normalized edit distance across all candidate pairs."""
    import editdistance
    if len(texts) < 2:
        return 0.0
    word_lists = [t.split() for t in texts]
    total = 0.0
    count = 0
    for i in range(len(word_lists)):
        for j in range(i + 1, len(word_lists)):
            a, b = word_lists[i], word_lists[j]
            denom = max(len(a), len(b))
            if denom == 0:
                continue
            total += editdistance.eval(a, b) / denom
            count += 1
    return total / count if count > 0 else 0.0

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
        description="R-MUSAN0: Diversity intervention for MUSAN 0dB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bpe", type=Path,
                        default=Path("/content/icefall/egs/librispeech/ASR/"
                                     "data/lang_bpe_500/bpe.model"))
    parser.add_argument("--icefall-dir", type=Path,
                        default=Path("/content/icefall"))
    parser.add_argument("--augmented-cuts", type=Path, default=None,
                        help="Pre-made MUSAN 0dB CutSet (jsonl.gz)")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Clean dev-other CutSet (if no augmented-cuts)")
    parser.add_argument("--musan-dir", type=Path, default=None,
                        help="MUSAN noise dir (if no augmented-cuts)")
    parser.add_argument("--snr", type=int, default=0)
    parser.add_argument("--G", type=int, default=16)
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--roberta-model", type=str, default="roberta-base")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pll-batch-size", type=int, default=32)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("R-MUSAN0: MUSAN 0dB Diversity Intervention")
    print("=" * 70)
    print(f"  checkpoint:      {args.checkpoint}")
    print(f"  augmented_cuts:  {args.augmented_cuts}")
    print(f"  G:               {args.G}")
    print(f"  tau:             {args.tau}")
    print(f"  output_dir:      {args.output_dir}")
    print()
    for name, cfg in CONDITIONS.items():
        print(f"  {name:20s}  beam={cfg['output_beam']:<5}  "
              f"scale={cfg['nbest_scale']:<4}  oversample={cfg['oversample']}")
    print()

    t0_total = time.time()

    conds_need_nbest = [
        c for c in CONDITIONS
        if not (args.output_dir / f"nbest_{c}.jsonl").exists()
        and not (args.output_dir / f"nbest_{c}_pll.jsonl").exists()
    ]

    if conds_need_nbest:
        print(f"Step 1: Generate N-best ({', '.join(conds_need_nbest)})")

        import torch
        import sentencepiece as spm
        import k2
        from lhotse import CutSet, Fbank, FbankConfig
        from scripts.generate_nbest import load_model

        device = torch.device(args.device)
        model = load_model(args.checkpoint, args.icefall_dir, device)
        sp = spm.SentencePieceProcessor()
        sp.load(str(args.bpe))
        fbank = Fbank(FbankConfig(num_mel_bins=80, dither=0.0))
        topo = k2.ctc_topo(max_token=MAX_TOKEN, modified=False, device=device)

        if args.augmented_cuts and args.augmented_cuts.exists():
            print(f"  Using pre-augmented cuts: {args.augmented_cuts}")
            cuts = list(CutSet.from_file(str(args.augmented_cuts)))
        else:
            assert args.manifest and args.musan_dir, \
                "Need --augmented-cuts OR --manifest + --musan-dir"
            cuts_path = args.output_dir / f"cuts_{args.snr}dB.jsonl.gz"
            if not cuts_path.exists():
                print(f"  Creating augmented cuts (SNR={args.snr}dB)...")
                import subprocess
                subprocess.run([
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "prepare_musan_cuts.py"),
                    "--ls-cuts", str(args.manifest),
                    "--musan-dir", str(args.musan_dir),
                    "--snr", str(args.snr),
                    "--output", str(cuts_path),
                ], check=True)
            cuts = list(CutSet.from_file(str(cuts_path)))

        print(f"  {len(cuts)} utterances")

        # Pre-compute encoder outputs (shared across all conditions)
        print("  Pre-computing encoder outputs (one pass)...")
        encoder_cache = []
        t0_enc = time.time()
        for i, cut in enumerate(cuts):
            audio = cut.load_audio()
            feat = fbank.extract(audio, sampling_rate=16000)
            feat_t = torch.from_numpy(feat).unsqueeze(0).to(device)
            feat_lens = torch.tensor(
                [feat.shape[0]], dtype=torch.int64, device=device
            )
            with torch.no_grad():
                enc_out, enc_lens = model.forward_encoder(feat_t, feat_lens)
                log_probs = model.ctc_output(enc_out)

            lp = log_probs[0, : enc_lens[0].item()].cpu()
            ref_text = normalize_text(cut.supervisions[0].text)
            utt_id = (cut.id.split("_musan")[0]
                      if "_musan" in cut.id else cut.id)

            encoder_cache.append({
                "log_probs": lp,
                "ref": ref_text,
                "utt_id": utt_id,
            })

            del enc_out, feat_t, log_probs
            torch.cuda.empty_cache()

            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0_enc
                speed = (i + 1) / elapsed
                print(f"    {i+1}/{len(cuts)} encoded "
                      f"({speed:.1f} utt/s, "
                      f"ETA {(len(cuts) - i - 1) / speed:.0f}s)")

        print(f"  Encoder done: {time.time() - t0_enc:.1f}s")

        for cond_name in conds_need_nbest:
            cfg = CONDITIONS[cond_name]
            nbest_path = args.output_dir / f"nbest_{cond_name}.jsonl"
            print(f"\n  Generating {cond_name}: beam={cfg['output_beam']}, "
                  f"scale={cfg['nbest_scale']}, oversample={cfg['oversample']}")

            records = []
            t0_cond = time.time()
            for i, cached in enumerate(encoder_cache):
                lp_gpu = cached["log_probs"].to(device)
                candidates = generate_nbest_parameterized(
                    lp_gpu, topo,
                    num_paths=cfg["oversample"],
                    nbest_scale=cfg["nbest_scale"],
                    output_beam=cfg["output_beam"],
                    sp=sp, device=device,
                )
                candidates = candidates[: args.G]
                records.append({
                    "utt_id": cached["utt_id"],
                    "ref": cached["ref"],
                    "nbest": candidates,
                })
                del lp_gpu

                if (i + 1) % 200 == 0:
                    elapsed = time.time() - t0_cond
                    speed = (i + 1) / elapsed
                    print(f"    {i+1}/{len(encoder_cache)} "
                          f"({speed:.1f} utt/s, "
                          f"ETA {(len(encoder_cache) - i - 1) / speed:.0f}s)")

            save_jsonl(records, nbest_path)
            avg_c = np.mean([len(r["nbest"]) for r in records])
            print(f"  Saved {len(records)} utts to {nbest_path} "
                  f"(avg {avg_c:.1f} cands, {time.time() - t0_cond:.1f}s)")

        # Free ASR model before loading RoBERTa
        del model, topo, encoder_cache, fbank
        torch.cuda.empty_cache()
        gc.collect()
    else:
        print("Step 1: SKIP (all N-best files exist)")

    conds_need_pll = [
        c for c in CONDITIONS
        if not (args.output_dir / f"nbest_{c}_pll.jsonl").exists()
    ]

    all_nbest = {}

    if conds_need_pll:
        print(f"\nStep 2: Score with RoBERTa PLL ({', '.join(conds_need_pll)})")

        import torch
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        from scripts.score_pll import compute_pll

        pll_device = torch.device(args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.roberta_model)
        lm_model = AutoModelForMaskedLM.from_pretrained(args.roberta_model)
        lm_model.eval().to(pll_device)

        for cond_name in conds_need_pll:
            nbest_path = args.output_dir / f"nbest_{cond_name}.jsonl"
            scored_path = args.output_dir / f"nbest_{cond_name}_pll.jsonl"
            print(f"\n  Scoring {cond_name}...")

            records = load_jsonl(nbest_path)
            t0_pll = time.time()
            n_scored = 0
            for i, rec in enumerate(records):
                for h in rec["nbest"]:
                    h["pll_score"] = compute_pll(
                        h["hyp"], tokenizer, lm_model, pll_device,
                        batch_size=args.pll_batch_size,
                    )
                    n_scored += 1

                if (i + 1) % 200 == 0:
                    elapsed = time.time() - t0_pll
                    rate = n_scored / elapsed if elapsed > 0 else 0
                    print(f"    {i+1}/{len(records)} utterances "
                          f"({rate:.1f} hyps/s)")

            save_jsonl(records, scored_path)
            all_nbest[cond_name] = records
            print(f"  {cond_name}: {n_scored} hyps in "
                  f"{time.time() - t0_pll:.1f}s -> {scored_path}")

        del lm_model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
    else:
        print("\nStep 2: SKIP (all PLL-scored files exist)")

    for cond_name in CONDITIONS:
        if cond_name not in all_nbest:
            scored_path = args.output_dir / f"nbest_{cond_name}_pll.jsonl"
            all_nbest[cond_name] = load_jsonl(scored_path)

    print(f"\nStep 3: MBR-CER+PLL tau={args.tau} + diversity metrics")
    import editdistance
    from experiments.significance_tests import paired_bootstrap_wer

    # Greedy WER is shared (same model, same audio)
    ref_words_all = [r["ref"].split() for r in all_nbest["baseline"]]
    greedy_words = [r["nbest"][0]["hyp"].split()
                    for r in all_nbest["baseline"]]
    total_ref = sum(len(rw) for rw in ref_words_all)
    greedy_wer = (
        sum(editdistance.eval(greedy_words[i], ref_words_all[i])
            for i in range(len(ref_words_all)))
        / total_ref
    )
    print(f"  Greedy WER (shared): {greedy_wer * 100:.4f}%")

    for cond_name in CONDITIONS:
        cond_greedy = [r["nbest"][0]["hyp"] for r in all_nbest[cond_name]]
        base_greedy = [r["nbest"][0]["hyp"] for r in all_nbest["baseline"]]
        n_diff = sum(1 for a, b in zip(cond_greedy, base_greedy) if a != b)
        if n_diff > 0:
            print(f"  WARNING: {cond_name} greedy differs in {n_diff} utts")

    condition_results = []
    per_utt_rows = []

    for cond_name, cfg in CONDITIONS.items():
        records = all_nbest[cond_name]
        ref_words = [r["ref"].split() for r in records]

        # Oracle WER
        oracle_words = []
        for rec in records:
            ref_w = rec["ref"].split()
            best = min(
                rec["nbest"],
                key=lambda c: editdistance.eval(c["hyp"].split(), ref_w),
            )
            oracle_words.append(best["hyp"].split())
        oracle_wer = (
            sum(editdistance.eval(oracle_words[i], ref_words[i])
                for i in range(len(ref_words)))
            / total_ref
        )

        # MBR-CER+PLL selection
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

        mbr_wer = (
            sum(editdistance.eval(mbr_words[i], ref_words[i])
                for i in range(len(ref_words)))
            / total_ref
        )
        delta_vs_greedy = mbr_wer - greedy_wer

        # Diversity metrics
        all_unique = []
        all_pairwise = []
        for i, rec in enumerate(records):
            texts = [h["hyp"] for h in rec["nbest"]]
            n_unique = len(set(texts))
            pw_wer = compute_mean_pairwise_wer(texts)
            all_unique.append(n_unique)
            all_pairwise.append(pw_wer)

            # Per-utterance diagnostics row
            ref_w = rec["ref"].split()
            ref_len = max(len(ref_w), 1)
            g_wer_utt = editdistance.eval(
                rec["nbest"][0]["hyp"].split(), ref_w) / ref_len
            m_wer_utt = editdistance.eval(mbr_words[i], ref_w) / ref_len
            oracle_in = any(
                editdistance.eval(h["hyp"].split(), ref_w) == 0
                for h in rec["nbest"]
            )
            per_utt_rows.append({
                "utt_id": rec["utt_id"],
                "condition": cond_name,
                "mbr_wer": round(m_wer_utt, 6),
                "greedy_wer": round(g_wer_utt, 6),
                "n_unique": n_unique,
                "oracle_in_set": oracle_in,
            })

        mean_unique = float(np.mean(all_unique))
        mean_pairwise = float(np.mean(all_pairwise))

        # Paired bootstrap: MBR vs greedy
        print(f"  {cond_name}: running bootstrap...")
        boot = paired_bootstrap_wer(
            ref_words, mbr_words, greedy_words,
            n_bootstrap=10000, seed=42,
        )

        gap_oracle = greedy_wer - oracle_wer
        gap_closed = ((greedy_wer - mbr_wer) / gap_oracle * 100
                      if gap_oracle > 0 else 0.0)

        result = {
            "condition": cond_name,
            "output_beam": cfg["output_beam"],
            "nbest_scale": cfg["nbest_scale"],
            "oversample": cfg["oversample"],
            "greedy_wer": round(greedy_wer, 6),
            "oracle_wer": round(oracle_wer, 6),
            "mbr_wer": round(mbr_wer, 6),
            "delta_vs_greedy_pp": round(delta_vs_greedy * 100, 4),
            "gap_closed_pct": round(gap_closed, 2),
            "p_value": round(boot["p_value"], 4),
            "ci_pp": [round(boot["ci_lower"] * 100, 4),
                      round(boot["ci_upper"] * 100, 4)],
            "mean_unique": round(mean_unique, 2),
            "mean_pairwise_wer": round(mean_pairwise, 4),
            "n_utterances": len(records),
        }
        condition_results.append(result)

        print(f"  {cond_name}:")
        print(f"    Oracle WER:        {oracle_wer * 100:.4f}%")
        print(f"    MBR WER:           {mbr_wer * 100:.4f}%")
        print(f"    Delta vs greedy:   {delta_vs_greedy * 100:+.4f}pp  "
              f"(p={boot['p_value']:.4f})")
        print(f"    Gap closed:        {gap_closed:.2f}%")
        print(f"    Mean unique:       {mean_unique:.2f}")
        print(f"    Mean pairwise WER: {mean_pairwise:.4f}")

    print(f"\nStep 4: Save outputs")

    out_json = args.output_dir / "diversity_intervention.json"
    with open(out_json, "w") as f:
        json.dump(condition_results, f, indent=2)
    print(f"  Wrote {out_json}")

    out_csv = args.output_dir / "per_condition_diagnostics.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "utt_id", "condition", "mbr_wer", "greedy_wer",
            "n_unique", "oracle_in_set",
        ])
        writer.writeheader()
        writer.writerows(per_utt_rows)
    print(f"  Wrote {out_csv}")

    print(f"\nVerification:")
    baseline = next(r for r in condition_results
                    if r["condition"] == "baseline")
    wide = next(r for r in condition_results
                if r["condition"] == "wide_beam")
    flat = next(r for r in condition_results
                if r["condition"] == "flat_sampling")

    checks = []

    # V1: Baseline reproduces existing 0dB result
    ok = abs(baseline["greedy_wer"] - 0.17877) < 0.005
    checks.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] Baseline greedy ~17.88%: "
          f"{baseline['greedy_wer'] * 100:.2f}%")

    ok = abs(baseline["mbr_wer"] - 0.17643) < 0.005
    checks.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] Baseline MBR ~17.64%: "
          f"{baseline['mbr_wer'] * 100:.2f}%")

    # V2: Conditions have different candidate sets
    ok = abs(baseline["mean_unique"] - flat["mean_unique"]) > 0.1
    checks.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] Diversity differs: "
          f"baseline={baseline['mean_unique']:.1f}, "
          f"flat={flat['mean_unique']:.1f}")

    # V3: Flat sampling has worse oracle (expected from nbest_debug)
    ok = flat["oracle_wer"] > baseline["oracle_wer"]
    checks.append(ok)
    print(f"  [{'PASS' if ok else 'WARN'}] Flat oracle > baseline oracle: "
          f"{flat['oracle_wer'] * 100:.2f}% > "
          f"{baseline['oracle_wer'] * 100:.2f}%")

    # V4: Greedy WER identical across conditions
    greedy_same = all(
        abs(r["greedy_wer"] - baseline["greedy_wer"]) < 1e-6
        for r in condition_results
    )
    checks.append(greedy_same)
    print(f"  [{'PASS' if greedy_same else 'FAIL'}] "
          f"Greedy WER identical across conditions")

    # V5: Wide beam has oracle <= baseline oracle
    ok = wide["oracle_wer"] <= baseline["oracle_wer"] + 0.001
    checks.append(ok)
    print(f"  [{'PASS' if ok else 'WARN'}] Wide beam oracle <= baseline: "
          f"{wide['oracle_wer'] * 100:.2f}% <= "
          f"{baseline['oracle_wer'] * 100:.2f}%")

    all_pass = all(checks)
    print(f"\n  {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")

    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    print(f"{'Condition':<20s} {'Oracle':>8s} {'MBR':>8s} {'delta pp':>8s} "
          f"{'p':>7s} {'Unique':>7s} {'PW-WER':>7s}")
    print("-" * 70)
    for r in condition_results:
        print(f"{r['condition']:<20s} "
              f"{r['oracle_wer'] * 100:>7.2f}% "
              f"{r['mbr_wer'] * 100:>7.2f}% "
              f"{r['delta_vs_greedy_pp']:>+7.3f} "
              f"{r['p_value']:>7.4f} "
              f"{r['mean_unique']:>7.1f} "
              f"{r['mean_pairwise_wer']:>7.4f}")
    print(f"\n  Greedy WER (all conditions): {greedy_wer * 100:.4f}%")

    elapsed = time.time() - t0_total
    print(f"\nDone. Total time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")

if __name__ == "__main__":
    main()

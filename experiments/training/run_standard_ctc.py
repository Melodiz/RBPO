#!/usr/bin/env python3
"""E26: Standard CTC vs CR-CTC comparison  --  Phases 0-4 (inference/analysis).

Tests whether RBPO findings generalize beyond CR-CTC to standard CTC.
Phases:
  0  Checkpoint info
  1  Greedy baseline (dev-other, dev-clean)
  2  Oracle gap characterization at G=16
  3  Spearman rho calibration analysis
  4  CTC-internal MBR at G=16 + paired bootstrap

Usage (Colab):
    python experiments/training/run_standard_ctc.py \
        --checkpoint /content/standard_ctc_model/exp/pretrained.pt \
        --bpe /content/standard_ctc_model/data/lang_bpe_500/bpe.model \
        --icefall-dir /content/icefall \
        --data-dir /content/librispeech_data \
        --output-dir /content/drive/MyDrive/rbpo_results/standard_ctc \
        --recipe zipformer_ctc \
        --model-size medium \
        --phase all
"""

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

BLANK_ID = 0
MAX_TOKEN = 499
_TAG_RE = re.compile(r"\{[^}]+\}|<[^>]+>")
_MULTI_SPACE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

# Architecture presets  --  encoder dims for known Zipformer sizes
MODEL_PRESETS = {
    "small": {
        "num_encoder_layers": "2,2,2,2,2,2",
        "encoder_dim": "192,256,256,256,256,256",
        "encoder_unmasked_dim": "192,192,192,192,192,192",
        "feedforward_dim": "512,768,768,768,768,768",
    },
    "medium": {
        "num_encoder_layers": "2,2,3,4,3,2",
        "encoder_dim": "384,384,384,384,384,384",
        "encoder_unmasked_dim": "256,256,256,256,256,256",
        "feedforward_dim": "1536,1536,1536,1536,1536,1536",
    },
    "large": {
        "num_encoder_layers": "2,2,4,5,4,2",
        "encoder_dim": "512,512,512,512,512,512",
        "encoder_unmasked_dim": "384,384,384,384,384,384",
        "feedforward_dim": "2048,2048,2048,2048,2048,2048",
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

def add_icefall_to_path(icefall_dir: Path, recipe: str):
    dirs = [
        icefall_dir,
        icefall_dir / "egs" / "librispeech" / "ASR",
        icefall_dir / "egs" / "librispeech" / "ASR" / recipe,
    ]
    for d in dirs:
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))

def load_model(args, device):
    """Load a Zipformer model with configurable architecture and CTC type."""
    import torch

    add_icefall_to_path(Path(args.icefall_dir), args.recipe)
    import train as train_module
    add_model_arguments = train_module.add_model_arguments
    get_params = train_module.get_params
    # zipformer_ctc recipe uses get_ctc_model; zipformer uses get_model
    _get_model = getattr(train_module, "get_ctc_model",
                         getattr(train_module, "get_model", None))
    assert _get_model is not None, (
        f"Neither get_ctc_model nor get_model found in {args.recipe}/train.py"
    )

    params = get_params()
    parser = argparse.ArgumentParser(add_help=False)
    add_model_arguments(parser)
    model_args = parser.parse_args([])
    for k, v in vars(model_args).items():
        params[k] = v

    preset = MODEL_PRESETS[args.model_size]
    params.num_encoder_layers = preset["num_encoder_layers"]
    params.encoder_dim = preset["encoder_dim"]
    params.encoder_unmasked_dim = preset["encoder_unmasked_dim"]
    params.feedforward_dim = preset["feedforward_dim"]
    params.vocab_size = args.vocab_size
    params.feature_dim = 80
    # Only set flags that exist in the recipe's params
    for flag, val in [
        ("use_transducer", False),
        ("use_ctc", True),
        ("use_cr_ctc", args.use_cr_ctc),
        ("use_attention_decoder", False),
    ]:
        if hasattr(params, flag):
            setattr(params, flag, val)

    model = _get_model(params)
    checkpoint = torch.load(
        str(args.checkpoint), map_location="cpu", weights_only=False
    )
    state_dict = checkpoint.get("model", checkpoint)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"  INFO: {len(unexpected)} unexpected keys ignored "
              f"(first 5): {unexpected[:5]}")

    model.eval().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params / 1e6:.1f}M parameters on {device}")
    print(f"  CTC type: {'CR-CTC' if args.use_cr_ctc else 'standard CTC'}")
    print(f"  Arch: {args.model_size} ({preset['num_encoder_layers']})")
    return model, n_params

def phase0_checkpoint_info(args, n_params):
    out_path = Path(args.output_dir) / "checkpoint_info.json"
    if out_path.exists() and not args.force:
        print(f"Phase 0: SKIP (exists: {out_path})")
        return

    print("\n" + "=" * 70)
    print("Phase 0: Checkpoint info")
    print("=" * 70)

    info = {
        "checkpoint_path": str(args.checkpoint),
        "bpe_path": str(args.bpe),
        "recipe": args.recipe,
        "model_size": args.model_size,
        "architecture": MODEL_PRESETS[args.model_size],
        "parameter_count": n_params,
        "parameter_count_M": round(n_params / 1e6, 1),
        "vocab_size": args.vocab_size,
        "use_cr_ctc": args.use_cr_ctc,
        "ctc_type": "CR-CTC" if args.use_cr_ctc else "standard CTC",
        "model_url": args.model_url or "not specified",
        "training_data": args.training_data or "not specified",
        "notes": args.notes or "",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  Saved: {out_path}")

def phase1_baseline(args, model, sp, device):
    out_path = Path(args.output_dir) / "baseline.json"
    if out_path.exists() and not args.force:
        print(f"Phase 1: SKIP (exists: {out_path})")
        return json.load(open(out_path))

    print("\n" + "=" * 70)
    print("Phase 1: Greedy baseline")
    print("=" * 70)

    import torch
    from lhotse import Fbank, FbankConfig, load_manifest_lazy
    import editdistance

    fbank = Fbank(FbankConfig(num_mel_bins=80, dither=0.0))
    results = {}

    for split in ["dev-other", "dev-clean"]:
        cuts_path = Path(args.data_dir) / "cuts" / f"librispeech_cuts_{split}.jsonl.gz"
        if not cuts_path.exists():
            print(f"  {split}: SKIP (cuts not found: {cuts_path})")
            continue

        print(f"\n  Decoding {split}...")
        cuts = list(load_manifest_lazy(str(cuts_path)))
        total_edits = 0
        total_char_edits = 0
        total_ref_words = 0
        total_ref_chars = 0
        t0 = time.time()

        for i, cut in enumerate(cuts):
            audio = cut.load_audio()
            feat = fbank.extract(audio, sampling_rate=16000)
            feat_t = torch.from_numpy(feat).unsqueeze(0).to(device)
            feat_lens = torch.tensor([feat.shape[0]], dtype=torch.int64, device=device)

            with torch.no_grad():
                enc_out, enc_lens = model.forward_encoder(feat_t, feat_lens)
                log_probs = model.ctc_output(enc_out)

            greedy_ids = log_probs[0, :enc_lens[0].item()].argmax(dim=-1).cpu().tolist()
            greedy_toks = ctc_collapse(greedy_ids)
            hyp = normalize_text(sp.decode(greedy_toks))

            ref_raw = " ".join(s.text for s in cut.supervisions if s.text)
            ref = normalize_text(ref_raw)

            ref_words = ref.split()
            hyp_words = hyp.split()
            total_edits += editdistance.eval(hyp_words, ref_words)
            total_ref_words += len(ref_words)
            total_char_edits += editdistance.eval(list(hyp), list(ref))
            total_ref_chars += len(ref)

            del log_probs, enc_out, feat_t
            torch.cuda.empty_cache()

            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                print(f"    {i+1}/{len(cuts)} ({(i+1)/elapsed:.1f} utt/s)")

        elapsed = time.time() - t0
        wer = total_edits / max(1, total_ref_words)
        cer = total_char_edits / max(1, total_ref_chars)

        results[split] = {
            "wer": wer,
            "cer": cer,
            "num_utterances": len(cuts),
            "num_ref_words": total_ref_words,
            "total_edits": total_edits,
            "wall_time_s": round(elapsed, 1),
        }
        print(f"  {split}: WER={wer:.4%}, CER={cer:.4%} "
              f"({len(cuts)} utts, {elapsed:.0f}s)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")

    dev_other_wer = results.get("dev-other", {}).get("wer")
    if dev_other_wer is not None:
        print(f"\n  dev-other WER = {dev_other_wer:.4%}")
        if dev_other_wer < 0.055:
            print("  WARNING: WER is very close to CR-CTC (6.02%). "
                  "Verify this is genuinely standard CTC.")
    return results

def generate_nbest_for_utt(log_probs_utt, topo, num_paths, nbest_scale, sp, device):
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
    lattice = k2.intersect_dense(topo, dense_fsa, output_beam=8.0)
    lattice = k2.connect(lattice)

    nbest = k2.Nbest.from_lattice(
        lattice, num_paths=num_paths,
        use_double_scores=True, nbest_scale=nbest_scale,
    )

    all_labels = nbest.fsa.labels.cpu().tolist()
    paths, current = [], []
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

def phase2_oracle(args, model, sp, device):
    nbest_path = Path(args.output_dir) / "nbest_dev_other_g16.jsonl"
    diag_path = Path(args.output_dir) / "diagnostics_g16.json"

    if diag_path.exists() and not args.force:
        print(f"Phase 2: SKIP (exists: {diag_path})")
        return

    print("\n" + "=" * 70)
    print("Phase 2: Oracle gap characterization at G=16")
    print("=" * 70)

    import torch
    import k2
    import editdistance
    import numpy as np
    from lhotse import Fbank, FbankConfig, load_manifest_lazy
    from scipy.stats import spearmanr

    G = 16
    oversample = 64
    nbest_scale = 1.0
    num_paths = G * oversample

    cuts_path = Path(args.data_dir) / "cuts" / "librispeech_cuts_dev-other.jsonl.gz"
    assert cuts_path.exists(), f"dev-other cuts not found: {cuts_path}"
    cuts = list(load_manifest_lazy(str(cuts_path)))
    print(f"  {len(cuts)} utterances, G={G}, oversample={oversample}, "
          f"nbest_scale={nbest_scale}")

    fbank = Fbank(FbankConfig(num_mel_bins=80, dither=0.0))
    topo = k2.ctc_topo(max_token=MAX_TOKEN, modified=False, device=device)

    # Step 2a: Generate N-best (or load existing)
    if nbest_path.exists() and not args.force:
        print(f"  N-best exists, loading: {nbest_path}")
        records = []
        with open(nbest_path) as f:
            for line in f:
                records.append(json.loads(line))
    else:
        print(f"  Generating N-best lists...")
        records = []
        nbest_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        with open(nbest_path, "w") as f_out:
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

                lp_utt = log_probs[0, :enc_lens[0].item()]
                candidates = generate_nbest_for_utt(
                    lp_utt, topo, num_paths, nbest_scale, sp, device
                )
                candidates = candidates[:G]

                ref_raw = " ".join(s.text for s in cut.supervisions if s.text)
                ref = normalize_text(ref_raw)
                record = {"utt_id": cut.id, "ref": ref, "nbest": candidates}
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append(record)

                del log_probs, enc_out, feat_t
                torch.cuda.empty_cache()

                if (i + 1) % 50 == 0 or i == len(cuts) - 1:
                    elapsed = time.time() - t0
                    speed = (i + 1) / elapsed
                    avg_c = sum(len(r["nbest"]) for r in records) / len(records)
                    print(f"    {i+1}/{len(cuts)}  avg_cands={avg_c:.1f}  "
                          f"({speed:.1f} utt/s, ETA {(len(cuts)-i-1)/speed:.0f}s)")

        elapsed = time.time() - t0
        print(f"  N-best generated: {len(records)} utts, {elapsed:.0f}s")

    # Step 2b: Compute diagnostics
    print("\n  Computing oracle diagnostics...")
    total_greedy_edits = 0
    total_oracle_edits = 0
    total_ref_words = 0
    recoverable = 0
    cand_counts = []
    per_utt_rho = []
    log_prob_spreads = []
    pairwise_wers = []

    for rec in records:
        ref_words = rec["ref"].split()
        if not ref_words:
            continue
        n_ref = len(ref_words)
        total_ref_words += n_ref
        nbest = rec["nbest"]
        cand_counts.append(len(nbest))

        greedy_edits = editdistance.eval(nbest[0]["hyp"].split(), ref_words)
        total_greedy_edits += greedy_edits

        best_edits = greedy_edits
        for c in nbest:
            e = editdistance.eval(c["hyp"].split(), ref_words)
            if e < best_edits:
                best_edits = e
        total_oracle_edits += best_edits
        if best_edits < greedy_edits:
            recoverable += 1

        if len(nbest) >= 3:
            scores = [c["score"] for c in nbest]
            wers = [editdistance.eval(c["hyp"].split(), ref_words) / n_ref
                    for c in nbest]
            if len(set(wers)) >= 2 and len(set(scores)) >= 2:
                rho, _ = spearmanr(scores, wers)
                if not np.isnan(rho):
                    per_utt_rho.append(rho)
            log_prob_spreads.append(max(scores) - min(scores))

        # Pairwise WER diversity (sample pairs to keep O(N))
        if len(nbest) >= 2:
            pair_wers = []
            texts = [c["hyp"] for c in nbest]
            for j in range(min(len(texts), 8)):
                for k in range(j + 1, min(len(texts), 8)):
                    w_j = texts[j].split()
                    w_k = texts[k].split()
                    denom = max(len(w_j), len(w_k), 1)
                    pair_wers.append(editdistance.eval(w_j, w_k) / denom)
            if pair_wers:
                pairwise_wers.append(np.mean(pair_wers))

    greedy_wer = total_greedy_edits / max(1, total_ref_words)
    oracle_wer = total_oracle_edits / max(1, total_ref_words)
    abs_gap = greedy_wer - oracle_wer
    rel_gap = abs_gap / max(1e-9, greedy_wer) * 100

    result = {
        "G": G,
        "oversample": oversample,
        "nbest_scale": nbest_scale,
        "num_utterances": len(records),
        "total_ref_words": total_ref_words,
        "greedy_wer": greedy_wer,
        "oracle_wer": oracle_wer,
        "abs_gap_pp": abs_gap,
        "rel_gap_pct": rel_gap,
        "recoverable_count": recoverable,
        "recoverable_pct": recoverable / max(1, len(records)) * 100,
        "greedy_optimal_count": len(records) - recoverable,
        "mean_unique_hyps": round(float(np.mean(cand_counts)), 1),
        "mean_pairwise_wer_diversity": round(float(np.mean(pairwise_wers)), 4)
            if pairwise_wers else None,
        "mean_logprob_spread": round(float(np.mean(log_prob_spreads)), 2)
            if log_prob_spreads else None,
        "spearman_rho_mean": round(float(np.mean(per_utt_rho)), 4)
            if per_utt_rho else None,
        "nbest_path": str(nbest_path),
    }

    diag_path.parent.mkdir(parents=True, exist_ok=True)
    with open(diag_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Greedy WER:        {greedy_wer:.4%}")
    print(f"  Oracle WER:        {oracle_wer:.4%}")
    print(f"  Abs gap:           {abs_gap:.4%}")
    print(f"  Rel gap:           {rel_gap:.1f}%")
    print(f"  Recoverable:       {recoverable}/{len(records)} "
          f"({recoverable/max(1,len(records)):.1%})")
    print(f"  Mean candidates:   {np.mean(cand_counts):.1f}")
    if pairwise_wers:
        print(f"  Pairwise WER div:  {np.mean(pairwise_wers):.4f}")
    print(f"\n  Saved: {diag_path}")

    # Comparison table
    print("\n  " + "=" * 60)
    print("  CR-CTC vs Standard CTC comparison (G=16):")
    fmt = "  {:<25s} {:>10s} {:>14s}"
    print(fmt.format("Metric", "CR-CTC", "Standard CTC"))
    print("  " + "-" * 52)
    print(fmt.format("Greedy WER", "6.02%", f"{greedy_wer:.2%}"))
    print(fmt.format("Oracle G=16", "4.44%", f"{oracle_wer:.2%}"))
    print(fmt.format("Absolute gap (pp)", "1.58", f"{abs_gap*100:.2f}"))
    print(fmt.format("Relative gap (%)", "26.2%", f"{rel_gap:.1f}%"))
    print(fmt.format("Recoverable utts", "665", f"{recoverable}"))
    cr_rec_frac = 665 / 2864
    rec_frac = recoverable / max(1, len(records))
    print(fmt.format("Recoverable frac", f"{cr_rec_frac:.1%}", f"{rec_frac:.1%}"))
    print(fmt.format("Mean candidates", "15.5", f"{np.mean(cand_counts):.1f}"))
    if pairwise_wers:
        print(fmt.format("Pairwise WER div", "19.1%",
                          f"{np.mean(pairwise_wers):.1%}"))
    print()

def phase3_spearman(args):
    out_path = Path(args.output_dir) / "spearman_analysis.json"
    nbest_path = Path(args.output_dir) / "nbest_dev_other_g16.jsonl"

    if out_path.exists() and not args.force:
        print(f"Phase 3: SKIP (exists: {out_path})")
        return

    print("\n" + "=" * 70)
    print("Phase 3: Spearman rho calibration analysis")
    print("=" * 70)

    assert nbest_path.exists(), (
        f"N-best JSONL not found: {nbest_path}. Run Phase 2 first."
    )

    import editdistance
    import numpy as np
    from scipy.stats import spearmanr

    records = []
    with open(nbest_path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"  Loaded {len(records)} utterances from {nbest_path.name}")

    all_rho = []
    recoverable_rho = []
    greedy_opt_rho = []
    anti_correlated = 0

    for rec in records:
        ref_words = rec["ref"].split()
        if not ref_words:
            continue
        n_ref = len(ref_words)
        nbest = rec["nbest"]
        if len(nbest) < 3:
            continue

        scores = [c["score"] for c in nbest]
        wers = [editdistance.eval(c["hyp"].split(), ref_words) / n_ref
                for c in nbest]
        if len(set(wers)) < 2 or len(set(scores)) < 2:
            continue

        rho, _ = spearmanr(scores, wers)
        if np.isnan(rho):
            continue

        all_rho.append(rho)
        if rho > 0:
            anti_correlated += 1

        greedy_edits = editdistance.eval(nbest[0]["hyp"].split(), ref_words)
        best_edits = min(
            editdistance.eval(c["hyp"].split(), ref_words) for c in nbest
        )
        if best_edits < greedy_edits:
            recoverable_rho.append(rho)
        else:
            greedy_opt_rho.append(rho)

    rho_arr = np.array(all_rho)
    n = len(rho_arr)

    # Bootstrap CI (B=10000, seed=42)
    rng = np.random.default_rng(42)
    B = 10000
    boot_means = np.zeros(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = rho_arr[idx].mean()
    boot_means.sort()
    ci_lo = float(boot_means[int(0.025 * B)])
    ci_hi = float(boot_means[int(0.975 * B)])

    result = {
        "n_utterances_with_rho": n,
        "corpus_mean_rho": round(float(rho_arr.mean()), 4),
        "corpus_median_rho": round(float(np.median(rho_arr)), 4),
        "corpus_std_rho": round(float(rho_arr.std()), 4),
        "ci_95_lo": round(ci_lo, 4),
        "ci_95_hi": round(ci_hi, 4),
        "bootstrap_B": B,
        "bootstrap_seed": 42,
        "recoverable_subset": {
            "mean_rho": round(float(np.mean(recoverable_rho)), 4)
                if recoverable_rho else None,
            "n": len(recoverable_rho),
        },
        "greedy_optimal_subset": {
            "mean_rho": round(float(np.mean(greedy_opt_rho)), 4)
                if greedy_opt_rho else None,
            "n": len(greedy_opt_rho),
        },
        "anti_correlated_count": anti_correlated,
        "anti_correlated_pct": round(anti_correlated / max(1, n) * 100, 1),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  n utterances:     {n}")
    print(f"  Mean Spearman rho:  {rho_arr.mean():.4f} "
          f"[95% CI: {ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Median rho:         {np.median(rho_arr):.4f}")
    print(f"  Std rho:            {rho_arr.std():.4f}")
    if recoverable_rho:
        print(f"  Recoverable rho:    {np.mean(recoverable_rho):.4f} "
              f"(n={len(recoverable_rho)})")
    if greedy_opt_rho:
        print(f"  Greedy-opt rho:     {np.mean(greedy_opt_rho):.4f} "
              f"(n={len(greedy_opt_rho)})")
    print(f"  Anti-correlated:  {anti_correlated}/{n} "
          f"({anti_correlated/max(1,n):.1%})")

    # Comparison
    print(f"\n  CR-CTC reference: rho=-0.347 (median -0.365, std 0.227)")
    print(f"  This model:       rho={rho_arr.mean():.3f} "
          f"(median {np.median(rho_arr):.3f}, std {rho_arr.std():.3f})")
    diff = rho_arr.mean() - (-0.347)
    if diff > 0.05:
        print("  -> Standard CTC has WORSE calibration (rho closer to 0)")
    elif diff < -0.05:
        print("  -> Standard CTC has BETTER calibration (rho more negative)")
    else:
        print("  -> Similar calibration (architecture-driven, not loss-driven)")

    print(f"\n  Saved: {out_path}")

def phase4_mbr(args):
    out_path = Path(args.output_dir) / "mbr_ctc_only_g16.json"
    nbest_path = Path(args.output_dir) / "nbest_dev_other_g16.jsonl"

    if out_path.exists() and not args.force:
        print(f"Phase 4: SKIP (exists: {out_path})")
        return

    print("\n" + "=" * 70)
    print("Phase 4: CTC-internal MBR at G=16")
    print("=" * 70)

    assert nbest_path.exists(), (
        f"N-best JSONL not found: {nbest_path}. Run Phase 2 first."
    )

    import editdistance
    import numpy as np

    records = []
    with open(nbest_path) as f:
        for line in f:
            records.append(json.loads(line))
    n_utts = len(records)
    print(f"  Loaded {n_utts} utterances")

    # CER matrix computation
    def cer_matrix(texts):
        n = len(texts)
        mat = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                d = editdistance.eval(list(texts[i]), list(texts[j]))
                denom = max(len(texts[i]), len(texts[j]), 1)
                mat[i, j] = d / denom
                mat[j, i] = mat[i, j]
        return mat

    def mbr_select(cer_mat, log_scores, tau):
        n = len(log_scores)
        if math.isinf(tau):
            weights = np.ones(n) / n
        else:
            scaled = np.array(log_scores) / tau
            scaled -= np.max(scaled)
            weights = np.exp(scaled)
            weights /= weights.sum()
        risk = cer_mat @ weights
        return int(np.argmin(risk))

    # Pre-compute CER matrices
    print("  Computing CER matrices...")
    t0 = time.time()
    cer_matrices = []
    for rec in records:
        texts = [c["hyp"] for c in rec["nbest"]]
        cer_matrices.append(cer_matrix(texts))
    print(f"  CER matrices: {time.time() - t0:.1f}s")

    # Baselines
    total_greedy_edits = total_oracle_edits = total_ref = 0
    utt_greedy_edits = []
    utt_ref_words = []
    for rec in records:
        ref_w = rec["ref"].split()
        total_ref += len(ref_w)
        utt_ref_words.append(len(ref_w))
        ge = editdistance.eval(rec["nbest"][0]["hyp"].split(), ref_w)
        total_greedy_edits += ge
        utt_greedy_edits.append(ge)
        total_oracle_edits += min(
            editdistance.eval(c["hyp"].split(), ref_w) for c in rec["nbest"]
        )
    greedy_wer = total_greedy_edits / max(1, total_ref)
    oracle_wer = total_oracle_edits / max(1, total_ref)
    gap = greedy_wer - oracle_wer
    print(f"  Greedy WER: {greedy_wer:.4%}, Oracle: {oracle_wer:.4%}, "
          f"Gap: {gap:.4%}")

    taus = [1.0, 10.0, 50.0, float("inf")]
    mbr_results = []

    for tau in taus:
        tau_str = "inf" if math.isinf(tau) else str(tau)
        selections = {}
        utt_mbr_edits = []

        for i, rec in enumerate(records):
            log_scores = [c["score"] for c in rec["nbest"]]
            idx = mbr_select(cer_matrices[i], log_scores, tau)
            selections[rec["utt_id"]] = idx
            ref_w = rec["ref"].split()
            mbr_edits = editdistance.eval(
                rec["nbest"][idx]["hyp"].split(), ref_w
            )
            utt_mbr_edits.append(mbr_edits)

        total_mbr_edits = sum(utt_mbr_edits)
        mbr_wer = total_mbr_edits / max(1, total_ref)
        gap_closed = (greedy_wer - mbr_wer) / gap * 100 if gap > 1e-9 else 0.0

        # Paired bootstrap (B=10000, seed=42)
        rng = np.random.default_rng(42)
        B = 10000
        boot_deltas = np.zeros(B)
        greedy_arr = np.array(utt_greedy_edits)
        mbr_arr = np.array(utt_mbr_edits)
        ref_arr = np.array(utt_ref_words, dtype=np.float64)

        for b in range(B):
            idx = rng.integers(0, n_utts, size=n_utts)
            g_wer = greedy_arr[idx].sum() / ref_arr[idx].sum()
            m_wer = mbr_arr[idx].sum() / ref_arr[idx].sum()
            boot_deltas[b] = g_wer - m_wer

        p_value = float((boot_deltas <= 0).mean())

        result = {
            "tau": tau_str,
            "wer": mbr_wer,
            "wer_pct": f"{mbr_wer:.4%}",
            "gap_closed_pct": round(gap_closed, 2),
            "delta_wer_pp": round((greedy_wer - mbr_wer) * 100, 4),
            "p_value": round(p_value, 4),
            "significant_005": p_value < 0.05,
            "bootstrap_B": B,
        }
        mbr_results.append(result)
        sig = "*" if p_value < 0.05 else " "
        print(f"  tau={tau_str:>4s}: WER={mbr_wer:.4%}, "
              f"gap_closed={gap_closed:+.1f}%, p={p_value:.4f} {sig}")

    output = {
        "source": str(nbest_path),
        "n_utterances": n_utts,
        "utility": "cer",
        "greedy_wer": greedy_wer,
        "oracle_wer": oracle_wer,
        "gap_pp": gap,
        "results": mbr_results,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Comparison
    print(f"\n  CR-CTC reference: all CTC-internal MBR fail significance at G=16")
    print(f"  CR-CTC best: tau=50, WER=5.99%, gap_closed=2.2%")
    best = min(mbr_results, key=lambda r: r["wer"])
    print(f"  This model best: tau={best['tau']}, WER={best['wer_pct']}, "
          f"gap_closed={best['gap_closed_pct']:+.1f}%")

    print(f"\n  Saved: {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="E26: Standard CTC vs CR-CTC comparison (Phases 0-4)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to pretrained.pt checkpoint")
    parser.add_argument("--bpe", type=Path, required=True,
                        help="Path to bpe.model")
    parser.add_argument("--icefall-dir", type=Path, default=Path("/content/icefall"),
                        help="Path to icefall repo")
    parser.add_argument("--recipe", type=str, default="zipformer",
                        choices=["zipformer", "zipformer_ctc"],
                        help="icefall recipe dir for model loading")
    parser.add_argument("--model-size", type=str, default="medium",
                        choices=["small", "medium", "large"],
                        help="Zipformer architecture size preset")
    parser.add_argument("--use-cr-ctc", action="store_true", default=False,
                        help="Use CR-CTC (default: standard CTC)")
    parser.add_argument("--vocab-size", type=int, default=500,
                        help="BPE vocabulary size")

    # Data
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Path to LibriSpeech data (with cuts/ subdir)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for all results")

    # Execution
    parser.add_argument("--phase", type=str, default="all",
                        help="Phase(s) to run: all, 0, 1, 2, 3, 4, or "
                             "comma-separated (e.g. '2,3,4')")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing outputs")
    parser.add_argument("--device", type=str, default="cuda:0")

    # Metadata
    parser.add_argument("--model-url", type=str, default=None,
                        help="URL of the model (for checkpoint_info.json)")
    parser.add_argument("--training-data", type=str, default=None,
                        help="Training data description")
    parser.add_argument("--notes", type=str, default=None,
                        help="Additional notes for checkpoint_info.json")

    args = parser.parse_args()

    print("=" * 70)
    print("E26: Standard CTC vs CR-CTC comparison")
    print("=" * 70)
    print(f"  checkpoint:   {args.checkpoint}")
    print(f"  recipe:       {args.recipe}")
    print(f"  model_size:   {args.model_size}")
    print(f"  use_cr_ctc:   {args.use_cr_ctc}")
    print(f"  data_dir:     {args.data_dir}")
    print(f"  output_dir:   {args.output_dir}")
    print(f"  phase:        {args.phase}")
    print(f"  device:       {args.device}")
    print()

    # Determine which phases to run
    if args.phase == "all":
        phases = {0, 1, 2, 3, 4}
    else:
        phases = set()
        for p in args.phase.split(","):
            phases.add(int(p.strip()))

    import torch
    import sentencepiece as spm

    device = torch.device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load model (needed for phases 0-2)
    model, n_params = None, 0
    if phases & {0, 1, 2}:
        print("Loading model...")
        model, n_params = load_model(args, device)

    sp = spm.SentencePieceProcessor()
    sp.load(str(args.bpe))
    print(f"  BPE vocab: {sp.get_piece_size()}")
    assert sp.get_piece_size() == args.vocab_size, (
        f"BPE vocab {sp.get_piece_size()} != expected {args.vocab_size}"
    )

    if 0 in phases:
        phase0_checkpoint_info(args, n_params)
    if 1 in phases:
        phase1_baseline(args, model, sp, device)
    if 2 in phases:
        phase2_oracle(args, model, sp, device)

    # Free GPU memory before CPU-only phases
    if model is not None and not (phases & {2}):
        del model
        torch.cuda.empty_cache()

    if 3 in phases:
        phase3_spearman(args)
    if 4 in phases:
        phase4_mbr(args)

    print("\n" + "=" * 70)
    print("E26 analysis phases complete.")
    completed = sorted(phases)
    print(f"  Completed phases: {completed}")
    print(f"  Results in: {args.output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()

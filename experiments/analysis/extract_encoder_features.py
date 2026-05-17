#!/usr/bin/env python3
"""Stage 1: Extract Zipformer encoder features for hypothesis ranking.

For each utterance in an N-best file, run the Zipformer-S CR-CTC encoder
and extract:
  - utterance-level features (mean encoder, CTC entropy / blank stats)
  - per-hypothesis features (encoder mean at aligned frames,
    alignment confidence, length)

Alignment is done via monotonic argmax: for each non-blank token in the
hypothesis (in order), find the frame after the previous token's frame
that maximizes log P(token | t). Approximate but robust; we are
extracting features, not training a CTC head.

Outputs an NPZ:
    utt_encoder_mean    (N_utts, D) float32
    utt_scalar_features (N_utts, 5) float32
                        [T_frames, ctc_entropy_mean, ctc_entropy_std,
                         ctc_blank_mean, ctc_max_nonblank_mean]
    hyp_encoder_mean    (N_hyps, D) float32
    hyp_scalar_features (N_hyps, 4) float32
                        [ctc_log_prob, hyp_align_confidence,
                         hyp_length_tokens, hyp_length_chars]
    wer                 (N_hyps,) float32
    utt_index           (N_hyps,) int32
    is_greedy           (N_hyps,) bool
    utt_ids             (N_utts,) string
    feature_names_utt   list of 5 strings
    feature_names_hyp   list of 4 strings

Usage:
    python experiments/extract_encoder_features.py \
        --model-dir /content/icefall-asr-librispeech-zipformer-small-cr-ctc \
        --data-dir /content/librispeech_data \
        --nbest-file results/nbest_dev_other_G16.jsonl \
        --split dev-other \
        --output results/encoder_features_dev.npz \
        --device cuda:0
"""

import argparse
import json
import sys
import time
from pathlib import Path

import editdistance
import numpy as np
import sentencepiece as spm
import torch
from tqdm import tqdm

BLANK_ID = 0

def add_icefall_to_path(icefall_dir: Path):
    for d in [
        icefall_dir,
        icefall_dir / "egs" / "librispeech" / "ASR",
        icefall_dir / "egs" / "librispeech" / "ASR" / "zipformer",
    ]:
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))

def load_model(model_dir: Path, icefall_dir: Path, device: torch.device):
    add_icefall_to_path(icefall_dir)
    from train import add_model_arguments, get_model, get_params

    params = get_params()
    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    model_args = parser.parse_args([])
    for k, v in vars(model_args).items():
        params[k] = v

    params.num_encoder_layers = "2,2,2,2,2,2"
    params.encoder_dim = "192,256,256,256,256,256"
    params.encoder_unmasked_dim = "192,192,192,192,192,192"
    params.feedforward_dim = "512,768,768,768,768,768"
    params.use_transducer = False
    params.use_ctc = True
    params.use_cr_ctc = True
    params.use_attention_decoder = False
    params.vocab_size = 500
    params.feature_dim = 80

    model = get_model(params)
    checkpoint = torch.load(
        model_dir / "exp" / "pretrained.pt", map_location="cpu"
    )
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params/1e6:.1f}M parameters")
    return model

def load_cuts_indexed(data_dir: Path, split: str):
    """Return dict: cut_id -> (features_tensor, ref_text)."""
    from lhotse import load_manifest_lazy

    cuts_path = data_dir / "cuts" / f"librispeech_cuts_{split}.jsonl.gz"
    assert cuts_path.exists(), f"CutSet not found: {cuts_path}"

    cuts = load_manifest_lazy(str(cuts_path))
    out = {}
    for cut in cuts:
        feats = torch.from_numpy(cut.load_features())
        ref_text = " ".join(
            s.text for s in cut.supervisions if s.text
        ).strip().lower()
        out[cut.id] = (feats, ref_text)
    print(f"Loaded {len(out)} cuts from {split}")
    return out

def load_nbest(path: Path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} N-best records from {path}")
    return records

def compute_wer(hyp: str, ref: str) -> float:
    ref_w = ref.split()
    hyp_w = hyp.split()
    if len(ref_w) == 0:
        return 0.0 if len(hyp_w) == 0 else 1.0
    return editdistance.eval(hyp_w, ref_w) / len(ref_w)

def align_monotonic_argmax(tokens, log_probs):
    """For each token in `tokens`, find a frame in [start, T) that maximizes
    log_probs[t, token], then advance `start` past that frame.

    tokens: list[int] (non-blank BPE token IDs)
    log_probs: numpy (T, V) of log-softmax probs
    Returns: list[int] of frame indices, same length as tokens.
    """
    T, V = log_probs.shape
    alignment = []
    start = 0
    for k in tokens:
        if start >= T:
            alignment.append(T - 1)
            continue
        t_star = start + int(np.argmax(log_probs[start:, k]))
        alignment.append(t_star)
        start = t_star + 1
        if start >= T:
            start = T - 1  # subsequent tokens reuse the last frame
    return alignment

def utt_features(encoder_out_np, log_probs_np, blank_id=BLANK_ID):
    """Compute utterance-level features.

    encoder_out_np: (T, D) float32
    log_probs_np:   (T, V) float32, log-softmax
    Returns: dict with `encoder_mean` (D,) and 5 scalars.
    """
    T = encoder_out_np.shape[0]

    enc_mean = encoder_out_np.mean(axis=0).astype(np.float32)

    probs = np.exp(log_probs_np)
    # Per-frame entropy: -sum p log p (NaN-safe via masked log-probs)
    entropy = -(probs * np.where(probs > 0, log_probs_np, 0.0)).sum(axis=1)
    entropy_mean = float(entropy.mean())
    entropy_std = float(entropy.std())

    blank_prob = probs[:, blank_id]
    nonblank_probs = np.delete(probs, blank_id, axis=1)
    max_nonblank = nonblank_probs.max(axis=1)

    return {
        "encoder_mean": enc_mean,
        "T_frames": float(T),
        "ctc_entropy_mean": entropy_mean,
        "ctc_entropy_std": entropy_std,
        "ctc_blank_mean": float(blank_prob.mean()),
        "ctc_max_nonblank_mean": float(max_nonblank.mean()),
    }

def hyp_features(tokens, encoder_out_np, log_probs_np, ctc_log_prob, text):
    """Compute hypothesis-level features via monotonic-argmax alignment."""
    if len(tokens) == 0:
        return {
            "encoder_mean": encoder_out_np.mean(axis=0).astype(np.float32),
            "ctc_log_prob": float(ctc_log_prob),
            "align_confidence": 0.0,
            "length_tokens": 0.0,
            "length_chars": float(len(text)),
        }

    align = align_monotonic_argmax(tokens, log_probs_np)
    hyp_enc_mean = encoder_out_np[align].mean(axis=0).astype(np.float32)
    align_lps = [log_probs_np[t, k] for t, k in zip(align, tokens)]
    align_conf = float(np.mean(align_lps))

    return {
        "encoder_mean": hyp_enc_mean,
        "ctc_log_prob": float(ctc_log_prob),
        "align_confidence": align_conf,
        "length_tokens": float(len(tokens)),
        "length_chars": float(len(text)),
    }

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract Zipformer encoder features for value-head training"
    )
    p.add_argument(
        "--model-dir", type=Path,
        default=Path("/content/icefall-asr-librispeech-zipformer-small-cr-ctc"),
    )
    p.add_argument(
        "--icefall-dir", type=Path,
        default=Path("/content/icefall"),
    )
    p.add_argument(
        "--data-dir", type=Path,
        default=Path("/content/librispeech_data"),
    )
    p.add_argument("--nbest-file", type=Path, required=True)
    p.add_argument(
        "--split", type=str, required=True,
        help="lhotse cuts split name, e.g. dev-other or train-clean-100",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--num-utterances", type=int, default=-1,
        help="Limit utterances (-1 = all). For smoke testing.",
    )
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 70)
    print("Encoder Feature Extraction (Stage 1)")
    print("=" * 70)
    print(f"  Split:      {args.split}")
    print(f"  N-best:     {args.nbest_file}")
    print(f"  Output:     {args.output}")
    print(f"  Device:     {device}")

    bpe_path = args.model_dir / "data" / "lang_bpe_500" / "bpe.model"
    sp = spm.SentencePieceProcessor()
    sp.load(str(bpe_path))
    print(f"  BPE vocab:  {sp.get_piece_size()}")

    t_load = time.time()
    model = load_model(args.model_dir, args.icefall_dir, device)
    cuts = load_cuts_indexed(args.data_dir, args.split)
    nbest = load_nbest(args.nbest_file)

    if args.num_utterances > 0:
        nbest = nbest[: args.num_utterances]
        print(f"  Limited to {len(nbest)} utterances")
    print(f"  Load time:  {time.time()-t_load:.1f}s")

    # First pass: figure out encoder dim D from a single utterance
    first_uid = nbest[0]["utt_id"]
    feats0, _ = cuts[first_uid]
    feats_gpu = feats0.unsqueeze(0).to(device)
    feat_lens = torch.tensor([feats0.shape[0]], dtype=torch.int64, device=device)
    with torch.no_grad():
        enc0, _ = model.forward_encoder(feats_gpu, feat_lens)
    D = enc0.shape[-1]
    V_dim = model.ctc_output(enc0).shape[-1]
    print(f"  Encoder D:  {D}")
    print(f"  CTC V:      {V_dim}")
    del enc0, feats_gpu

    # Allocate output buffers
    n_utts = len(nbest)
    n_hyps_total = sum(rec["num_candidates"] for rec in nbest)
    print(f"  Utts:       {n_utts}")
    print(f"  Total hyps: {n_hyps_total}")

    utt_encoder_mean = np.zeros((n_utts, D), dtype=np.float32)
    utt_scalar = np.zeros((n_utts, 5), dtype=np.float32)
    hyp_encoder_mean = np.zeros((n_hyps_total, D), dtype=np.float32)
    hyp_scalar = np.zeros((n_hyps_total, 4), dtype=np.float32)
    wer_arr = np.zeros(n_hyps_total, dtype=np.float32)
    utt_index = np.zeros(n_hyps_total, dtype=np.int32)
    is_greedy = np.zeros(n_hyps_total, dtype=bool)
    utt_ids_arr = np.empty(n_utts, dtype=object)

    feature_names_utt = [
        "T_frames", "ctc_entropy_mean", "ctc_entropy_std",
        "ctc_blank_mean", "ctc_max_nonblank_mean",
    ]
    feature_names_hyp = [
        "ctc_log_prob", "align_confidence",
        "length_tokens", "length_chars",
    ]

    # Spot-check buffer
    spotcheck_data = []
    SPOTCHECK_LIMIT = 3

    # Verify dev WER if neural_lm_scores.jsonl exists alongside
    wer_check = None
    nlm_path = args.nbest_file.parent / "neural_lm_scores.jsonl"
    if "dev" in args.split and nlm_path.exists():
        try:
            wer_check = {}
            with open(nlm_path) as f:
                for line in f:
                    rec = json.loads(line)
                    wer_check[rec["utt_id"]] = [
                        c.get("wer") for c in rec["candidates"]
                    ]
            print(f"  Will verify WER against {nlm_path}")
        except Exception as e:
            print(f"  (could not load neural_lm_scores for verification: {e})")
            wer_check = None

    # Main loop
    t_extract = time.time()
    hyp_offset = 0
    skipped = 0
    wer_check_diffs = []

    for ui, rec in enumerate(tqdm(nbest, desc="Extracting features")):
        uid = rec["utt_id"]
        utt_ids_arr[ui] = uid

        if uid not in cuts:
            skipped += 1
            continue
        feats, ref = cuts[uid]

        feats_gpu = feats.unsqueeze(0).to(device)
        feat_lens = torch.tensor(
            [feats.shape[0]], dtype=torch.int64, device=device
        )

        with torch.no_grad():
            enc, enc_lens = model.forward_encoder(feats_gpu, feat_lens)
            log_probs = model.ctc_output(enc)  # (1, T, V)

        T_eff = int(enc_lens[0].item())
        encoder_out_np = enc[0, :T_eff].cpu().numpy()
        log_probs_np = log_probs[0, :T_eff].cpu().numpy()

        u_feat = utt_features(encoder_out_np, log_probs_np)
        utt_encoder_mean[ui] = u_feat["encoder_mean"]
        utt_scalar[ui] = [
            u_feat["T_frames"],
            u_feat["ctc_entropy_mean"],
            u_feat["ctc_entropy_std"],
            u_feat["ctc_blank_mean"],
            u_feat["ctc_max_nonblank_mean"],
        ]

        ref_use = rec.get("ref_text", ref)
        cand_align_confs = []

        for ci, cand in enumerate(rec["candidates"]):
            tokens = cand.get("tokens", [])
            text = cand.get("text", "")
            ctc_lp = cand.get("ctc_log_prob", 0.0)

            h_feat = hyp_features(
                tokens, encoder_out_np, log_probs_np,
                ctc_lp, text,
            )

            row = hyp_offset + ci
            hyp_encoder_mean[row] = h_feat["encoder_mean"]
            hyp_scalar[row] = [
                h_feat["ctc_log_prob"],
                h_feat["align_confidence"],
                h_feat["length_tokens"],
                h_feat["length_chars"],
            ]
            wer_arr[row] = compute_wer(text, ref_use)
            utt_index[row] = ui
            is_greedy[row] = (ci == 0)
            cand_align_confs.append(h_feat["align_confidence"])

            if wer_check and uid in wer_check:
                expected = wer_check[uid][ci] if ci < len(wer_check[uid]) else None
                if expected is not None:
                    diff = abs(expected - wer_arr[row])
                    wer_check_diffs.append(diff)

        # Spot-check first 3 utterances: greedy text vs max-confidence hyp
        if len(spotcheck_data) < SPOTCHECK_LIMIT:
            best_conf_idx = int(np.argmax(cand_align_confs))
            spotcheck_data.append({
                "utt_id": uid,
                "greedy_text": rec["candidates"][0]["text"],
                "best_conf_idx": best_conf_idx,
                "best_conf_text": rec["candidates"][best_conf_idx]["text"],
                "best_conf_value": cand_align_confs[best_conf_idx],
                "greedy_conf_value": cand_align_confs[0],
                "match": best_conf_idx == 0,
            })

        hyp_offset += rec["num_candidates"]

        del enc, log_probs, feats_gpu

    extract_time = time.time() - t_extract
    print(f"\nExtraction time: {extract_time:.1f}s ({extract_time/60:.1f} min)")
    print(f"Throughput:      {n_utts/extract_time:.1f} utt/s")
    if skipped:
        print(f"Skipped (no cut found): {skipped}")

    # Trim arrays in case of skips (rare)
    actual_hyps = hyp_offset
    if actual_hyps != n_hyps_total:
        hyp_encoder_mean = hyp_encoder_mean[:actual_hyps]
        hyp_scalar = hyp_scalar[:actual_hyps]
        wer_arr = wer_arr[:actual_hyps]
        utt_index = utt_index[:actual_hyps]
        is_greedy = is_greedy[:actual_hyps]

    # -- Verification & stats --------------------------------------------
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    print(f"  Encoder D = {D}")
    print(f"  Total hypotheses extracted: {actual_hyps}")
    print(f"  Mean hypotheses per utterance: {actual_hyps/n_utts:.2f}")

    print("\n  Hyp-level scalar stats:")
    print(f"  {'feature':<20} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
    for i, name in enumerate(feature_names_hyp):
        col = hyp_scalar[:, i]
        print(f"  {name:<20} {col.mean():>10.3f} {col.std():>10.3f} "
              f"{col.min():>10.3f} {col.max():>10.3f}")

    print("\n  Utt-level scalar stats:")
    print(f"  {'feature':<25} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
    for i, name in enumerate(feature_names_utt):
        col = utt_scalar[:, i]
        print(f"  {name:<25} {col.mean():>10.3f} {col.std():>10.3f} "
              f"{col.min():>10.3f} {col.max():>10.3f}")

    print("\n  WER stats:")
    print(f"    overall mean: {wer_arr.mean():.4f}")
    print(f"    greedy mean:  {wer_arr[is_greedy].mean():.4f}")
    print(f"    fraction zero: {(wer_arr == 0).mean():.4f}")

    if wer_check_diffs:
        diffs = np.array(wer_check_diffs)
        print(f"\n  WER cross-check vs neural_lm_scores.jsonl: "
              f"max diff {diffs.max():.6f}, mean diff {diffs.mean():.6f}")
        if diffs.max() > 1e-4:
            print("  WARNING: WER values differ from neural_lm_scores.jsonl")

    # Spot check
    print("\n  SPOT CHECK: greedy hyp vs max-align-confidence hyp")
    for s in spotcheck_data:
        marker = "MATCH " if s["match"] else "DIFFER"
        print(f"  [{s['utt_id']}] {marker}")
        print(f"    greedy (idx 0, conf={s['greedy_conf_value']:+.3f}): "
              f"{s['greedy_text'][:70]}")
        if not s["match"]:
            print(f"    max-conf (idx {s['best_conf_idx']}, "
                  f"conf={s['best_conf_value']:+.3f}): "
                  f"{s['best_conf_text'][:70]}")

    # -- Save ------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        utt_encoder_mean=utt_encoder_mean,
        utt_scalar_features=utt_scalar,
        hyp_encoder_mean=hyp_encoder_mean,
        hyp_scalar_features=hyp_scalar,
        wer=wer_arr,
        utt_index=utt_index,
        is_greedy=is_greedy,
        utt_ids=utt_ids_arr.astype(str),
        feature_names_utt=np.array(feature_names_utt, dtype=object),
        feature_names_hyp=np.array(feature_names_hyp, dtype=object),
        encoder_dim=np.array([D]),
    )
    size_mb = args.output.stat().st_size / 1e6
    print(f"\n  Wrote {args.output} ({size_mb:.1f} MB)")
    print(f"  Total elapsed: {time.time()-t_load:.1f}s")

if __name__ == "__main__":
    main()

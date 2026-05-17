#!/usr/bin/env python3
"""Stage 2: CTC alignment posterior gamma_t(k|y) extraction and analysis.

Tests the core RBPO theoretical claim: CTC alignment posteriors are
extremely sparse  --  most frames are blank-dominated (gamma_t(blank|y) ~ 1)
and contribute nothing to the reward signal. Only frames at label
emission boundaries carry meaningful gradient.

Extracts gamma_t(k|y) via the autograd trick:
  d log P_CTC(y|x) / d log_prob_t^k = gamma_t(k|y)

Usage:
    python experiments/gamma_analysis.py \
        --model-dir /content/icefall-asr-librispeech-zipformer-small-cr-ctc \
        --data-dir /content/librispeech_data \
        --results-dir /content/drive/MyDrive/rbpo_results \
        --num-utterances 100
"""

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import editdistance
import numpy as np
import sentencepiece as spm
import torch
from tqdm import tqdm

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

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {num_params / 1e6:.1f}M parameters")
    return model

BLANK_ID = 0
MAX_TOKEN = 499

def ctc_collapse(token_ids: list[int]) -> list[int]:
    result = []
    prev = None
    for t in token_ids:
        if t != BLANK_ID and t != prev:
            result.append(t)
        prev = t
    return result

def compute_wer(hypothesis: str, reference: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return editdistance.eval(hyp_words, ref_words) / len(ref_words)

def extract_gamma(
    log_probs: torch.Tensor,
    token_ids: list[int],
    T: int,
    device: torch.device,
):
    """Extract alignment posterior gamma_t(k|y) via autograd.

    Uses the identity:
      d log P_CTC(y|x) / d log_prob_t^k = gamma_t(k|y)

    Args:
        log_probs: (1, T, V) log-softmax outputs from CTC head.
        token_ids: hypothesis token IDs (after CTC collapse, no blanks).
        T: actual sequence length (frames).
        device: torch device.

    Returns:
        gamma: (T, V) tensor of alignment posteriors. Each row sums to 1.
    """
    import k2

    lp_gamma = log_probs.detach().requires_grad_(True)

    supervision_segments = torch.tensor([[0, 0, T]], dtype=torch.int32)
    dense_fsa = k2.DenseFsaVec(lp_gamma, supervision_segments)
    ctc_graph = k2.ctc_graph([token_ids], modified=False, device=device)
    lattice = k2.intersect_dense(ctc_graph, dense_fsa, output_beam=10.0)
    tot_score = lattice.get_tot_scores(
        log_semiring=True, use_double_scores=True
    )

    tot_score.sum().backward()
    gamma = lp_gamma.grad.squeeze(0)  # (T, V)
    return gamma

def verify_gamma(
    gamma: torch.Tensor,
    log_probs: torch.Tensor,
    token_ids: list[int],
    T: int,
):
    """Run correctness checks. HALT if any fails.

    Verifies Proposition 4.1 from the RBPO paper.
    """
    V = gamma.shape[-1]
    gamma_valid = gamma[:T]  # (T, V)

    # CHECK 1: gamma_t sums to 1 per frame
    frame_sums = gamma_valid.sum(dim=-1)
    max_dev = (frame_sums - 1.0).abs().max().item()
    assert max_dev < 1e-3, (
        f"gamma_t doesn't sum to 1: max deviation = {max_dev}"
    )

    # CHECK 2: gamma_t is non-negative
    min_val = gamma_valid.min().item()
    assert min_val >= -1e-4, f"gamma_t has negative values: min = {min_val}"

    # CHECK 3: Sparsity  --  no mass on labels not in hypothesis
    valid_labels = {0} | set(token_ids)
    for k in range(V):
        if k not in valid_labels:
            mass = gamma_valid[:, k].abs().max().item()
            assert mass < 1e-4, (
                f"gamma_t has mass {mass} on label {k} not in hypothesis"
            )

    # CHECK 4: CTC gradient identity (Eq. 2)
    # d(-log P_CTC)/dz_t^k = P(k|x_t) - gamma_t(k|y), should sum to 0 per frame
    log_probs_valid = log_probs[:T] if log_probs.dim() == 2 else log_probs[0, :T]
    P_frame = log_probs_valid.exp()
    ctc_grad = P_frame - gamma_valid
    grad_sums = ctc_grad.sum(dim=-1)
    max_grad_sum_dev = grad_sums.abs().max().item()
    assert max_grad_sum_dev < 1e-3, (
        f"CTC gradient doesn't sum to 0 per frame: max = {max_grad_sum_dev}"
    )

    # CHECK 5: Label count lower bound
    # sum_t gamma_t(k|y) >= count(k in token_ids), since each label emission
    # spans at least one frame in any valid CTC alignment. The upper
    # bound depends on emission span (typically 1-3 frames per label).
    label_counts = Counter(token_ids)
    for k, expected in label_counts.items():
        gamma_count = gamma_valid[:, k].sum().item()
        assert gamma_count >= expected - 1e-3, (
            f"Label {k}: gamma sum {gamma_count:.4f} < expected count {expected} "
            f"(violates CTC alignment lower bound)"
        )

def compute_gamma_stats(gamma: torch.Tensor, T: int, L: int) -> dict:
    """Compute per-utterance statistics from gamma_t(k|y)."""
    gamma_valid = gamma[:T]  # (T, V)

    blank_post = gamma_valid[:, BLANK_ID]  # (T,)

    dead_frac = (blank_post > 0.99).float().mean().item()
    near_dead_frac = (blank_post > 0.95).float().mean().item()
    max_per_frame = gamma_valid.max(dim=-1).values
    ambiguous_frac = (max_per_frame < 0.5).float().mean().item()

    eps = 1e-10
    entropy = -(gamma_valid * (gamma_valid + eps).log()).sum(dim=-1)  # (T,)
    mean_entropy = entropy.mean().item()
    max_entropy = entropy.max().item()

    label_frames = (blank_post < 0.5).sum().item()
    label_frame_frac = label_frames / T if T > 0 else 0.0

    return {
        "T": T,
        "L": L,
        "compression_ratio": T / L if L > 0 else 0.0,
        "dead_frame_frac": dead_frac,
        "near_dead_frac": near_dead_frac,
        "ambiguous_frac": ambiguous_frac,
        "mean_entropy": mean_entropy,
        "max_entropy": max_entropy,
        "num_label_frames": label_frames,
        "label_frame_frac": label_frame_frac,
    }

def load_utterances(data_dir: Path, split: str, num_utterances: int):
    from lhotse import load_manifest_lazy

    cuts_path = data_dir / "cuts" / f"librispeech_cuts_{split}.jsonl.gz"
    assert cuts_path.exists(), f"CutSet not found: {cuts_path}"

    cuts = load_manifest_lazy(str(cuts_path))
    utterances = []
    for cut in cuts:
        if len(utterances) >= num_utterances:
            break
        feats = torch.from_numpy(cut.load_features())  # (T, 80)
        ref_text = " ".join(
            s.text for s in cut.supervisions if s.text
        ).strip().lower()
        if not ref_text:
            continue
        utterances.append((cut.id, feats, ref_text, cut.duration))

    print(f"Loaded {len(utterances)} utterances from {split}")
    return utterances

def run_experiment(
    model,
    utterances: list,
    sp: spm.SentencePieceProcessor,
    device: torch.device,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = output_dir / "gamma_examples"
    examples_dir.mkdir(exist_ok=True)

    csv_path = output_dir / "gamma_stats.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "utt_id", "T", "L", "compression_ratio",
        "dead_frame_frac", "near_dead_frac", "ambiguous_frac",
        "mean_entropy", "max_entropy", "label_frame_frac",
        "ref_text", "hyp_text", "wer",
    ])

    # Pick 3 example utterances by duration: short, medium, long.
    durations = [(i, u[3]) for i, u in enumerate(utterances)]
    durations.sort(key=lambda x: x[1])
    n = len(durations)
    example_idxs = {
        durations[n // 6][0]: "short",
        durations[n // 2][0]: "medium",
        durations[(5 * n) // 6][0]: "long",
    }

    all_stats = []
    verification_passed = 0
    verification_failed = 0

    for utt_idx, (utt_id, feats, ref_text, duration) in enumerate(
        tqdm(utterances, desc="Extracting gamma_t")
    ):
        feats_gpu = feats.unsqueeze(0).to(device)
        feat_lens = torch.tensor(
            [feats.shape[0]], dtype=torch.int64, device=device
        )

        with torch.no_grad():
            encoder_out, encoder_out_lens = model.forward_encoder(
                feats_gpu, feat_lens
            )
            log_probs = model.ctc_output(encoder_out)  # (1, T, V)

        T = encoder_out_lens[0].item()
        log_probs_utt = log_probs[0, :T]  # (T, V)

        # Greedy 1-best
        greedy_ids = log_probs_utt.argmax(dim=-1).tolist()
        token_ids = ctc_collapse(greedy_ids)
        L = len(token_ids)
        hyp_text = sp.decode(token_ids).strip().lower()
        wer = compute_wer(hyp_text, ref_text)

        if L == 0:
            # Empty hypothesis  --  skip
            continue

        gamma = extract_gamma(log_probs[:, :T, :], token_ids, T, device)

        # Verify (HALT on failure)
        verify_gamma(gamma, log_probs_utt, token_ids, T)
        verification_passed += 1

        # Stats
        stats = compute_gamma_stats(gamma, T, L)
        stats["utt_id"] = utt_id
        stats["ref_text"] = ref_text
        stats["hyp_text"] = hyp_text
        stats["wer"] = wer
        all_stats.append(stats)

        writer.writerow([
            utt_id, T, L, f"{stats['compression_ratio']:.4f}",
            f"{stats['dead_frame_frac']:.4f}",
            f"{stats['near_dead_frac']:.4f}",
            f"{stats['ambiguous_frac']:.4f}",
            f"{stats['mean_entropy']:.4f}",
            f"{stats['max_entropy']:.4f}",
            f"{stats['label_frame_frac']:.4f}",
            ref_text, hyp_text, f"{wer:.4f}",
        ])

        if utt_idx in example_idxs:
            tag = example_idxs[utt_idx]
            np.save(
                examples_dir / f"example_{tag}_gamma.npy",
                gamma.detach().cpu().numpy(),
            )
            with open(examples_dir / f"example_{tag}_meta.json", "w") as f:
                json.dump({
                    "utt_id": utt_id,
                    "tag": tag,
                    "T": T,
                    "L": L,
                    "duration_sec": duration,
                    "ref_text": ref_text,
                    "hyp_text": hyp_text,
                    "token_ids": token_ids,
                    "wer": wer,
                }, f, indent=2)

        # Cleanup
        del gamma, log_probs, encoder_out, feats_gpu
        torch.cuda.empty_cache()

        if (utt_idx + 1) % 25 == 0:
            mean_dead = np.mean([s["dead_frame_frac"] for s in all_stats])
            print(
                f"  [{utt_idx+1}/{len(utterances)}] "
                f"mean dead_frame_frac so far: {mean_dead:.3f}"
            )

    csv_file.close()
    print(f"CSV saved: {csv_path}")

    # Aggregate
    def mean(key):
        return float(np.mean([s[key] for s in all_stats]))

    def std(key):
        return float(np.std([s[key] for s in all_stats]))

    summary = {
        "n_utterances": len(all_stats),
        "verification_passed": verification_passed,
        "verification_failed": verification_failed,
        "mean_dead_frame_fraction": mean("dead_frame_frac"),
        "std_dead_frame_fraction": std("dead_frame_frac"),
        "mean_near_dead_fraction": mean("near_dead_frac"),
        "mean_ambiguous_fraction": mean("ambiguous_frac"),
        "mean_entropy": mean("mean_entropy"),
        "mean_label_frame_fraction": mean("label_frame_frac"),
        "mean_compression_ratio": mean("compression_ratio"),
        "mean_T": mean("T"),
        "mean_L": mean("L"),
    }

    summary_path = output_dir / "gamma_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")

    print("\n" + "=" * 70)
    print("STAGE 2 SUMMARY: gamma_t(k|y) Sparsity")
    print("=" * 70)
    print(f"  Utterances analyzed:          {summary['n_utterances']}")
    print(f"  Verification passed:          {summary['verification_passed']}")
    print(f"  Mean dead frame fraction:     {summary['mean_dead_frame_fraction']:.3f}")
    print(f"  (gamma_t(blank) > 0.99)")
    print(f"  Mean near-dead fraction:      {summary['mean_near_dead_fraction']:.3f}")
    print(f"  (gamma_t(blank) > 0.95)")
    print(f"  Mean label frame fraction:    {summary['mean_label_frame_fraction']:.3f}")
    print(f"  (gamma_t(blank) < 0.5)")
    print(f"  Mean per-frame entropy:       {summary['mean_entropy']:.3f} nats")
    print(f"  Mean compression ratio T/L:   {summary['mean_compression_ratio']:.2f}")
    print("=" * 70)

    return summary

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 2: CTC alignment posterior gamma_t(k|y) analysis"
    )
    parser.add_argument(
        "--model-dir", type=Path,
        default=Path("/content/icefall-asr-librispeech-zipformer-small-cr-ctc"),
    )
    parser.add_argument(
        "--icefall-dir", type=Path, default=Path("/content/icefall"),
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("/content/librispeech_data"),
    )
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path("/content/drive/MyDrive/rbpo_results"),
    )
    parser.add_argument("--num-utterances", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 60)
    print("RBPO Stage 2  --  gamma_t(k|y) Alignment Posterior Analysis")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Utterances: {args.num_utterances}")

    sp = spm.SentencePieceProcessor()
    bpe_path = args.model_dir / "data" / "lang_bpe_500" / "bpe.model"
    assert bpe_path.exists(), f"BPE model not found: {bpe_path}"
    sp.load(str(bpe_path))
    print(f"BPE vocab: {sp.get_piece_size()} tokens")

    model = load_model(args.model_dir, args.icefall_dir, device)

    utterances = load_utterances(
        args.data_dir, "dev-other", args.num_utterances
    )

    t0 = time.time()
    output_dir = args.results_dir / "stage_2_gamma_analysis"
    summary = run_experiment(model, utterances, sp, device, output_dir)
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

if __name__ == "__main__":
    main()

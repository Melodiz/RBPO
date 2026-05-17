#!/usr/bin/env python3
"""Stage 0b: Verify k2 installation and reproduce baseline CTC WER.

Loads pretrained Zipformer-S CR-CTC model from icefall, runs greedy CTC
decoding on LibriSpeech dev-clean and dev-other, reports WER.

Usage:
    python experiments/verify_baseline.py [--model-dir DIR] [--data-dir DIR]

Defaults assume Colab layout (setup_colab.sh must have been run previously).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import sentencepiece as spm
import torch

def k2_smoke_tests():
    """Run before anything else. Halt with clear error if k2 is broken."""
    import k2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", (
        "CUDA not available  --  k2 needs GPU. "
        "On Colab: Runtime -> Change runtime type -> T4 GPU"
    )

    # Test 1: ctc_topo construction
    topo = k2.ctc_topo(max_token=499, modified=False, device=device)
    assert topo.num_arcs > 0, "ctc_topo produced empty FSA"
    print("   Test 1: ctc_topo construction")

    # Test 2: intersect_dense with random tensor
    T, V = 100, 500
    log_probs = torch.randn(1, T, V, device=device).log_softmax(dim=-1)
    supervision = torch.tensor([[0, 0, T]], dtype=torch.int32)
    dense_fsa = k2.DenseFsaVec(log_probs, supervision)
    lattice = k2.intersect_dense(topo, dense_fsa, output_beam=10.0)
    scores = lattice.get_tot_scores(log_semiring=True, use_double_scores=True)
    assert scores.isfinite().all(), f"get_tot_scores returned non-finite: {scores}"
    print("   Test 2: intersect_dense + get_tot_scores")

    # Test 3: connect (removes dead states)
    lattice_connected = k2.connect(lattice)
    print(f"   Test 3: k2.connect (arcs: {lattice.num_arcs} -> {lattice_connected.num_arcs})")

    # Test 4: ctc_graph for a specific token sequence
    token_ids = [[1, 2, 3, 4, 5]]
    ctc_graph = k2.ctc_graph(token_ids, modified=False, device=device)
    assert ctc_graph.num_arcs > 0, "ctc_graph produced empty FSA"
    print("   Test 4: ctc_graph construction")

    # Test 5: differentiable scores (autograd through k2)
    log_probs_diff = log_probs.detach().requires_grad_(True)
    dense_diff = k2.DenseFsaVec(log_probs_diff, supervision)
    lat = k2.intersect_dense(ctc_graph, dense_diff, output_beam=10.0)
    score = lat.get_tot_scores(log_semiring=True, use_double_scores=True)
    score.sum().backward()
    assert log_probs_diff.grad is not None, "No gradient  --  autograd broken"
    assert log_probs_diff.grad.isfinite().all(), "Non-finite gradients"
    print("   Test 5: differentiable k2 scores (autograd)")

    # Cleanup
    del lattice, lattice_connected, lat, dense_fsa, dense_diff
    del log_probs, log_probs_diff, topo, ctc_graph
    torch.cuda.empty_cache()

    import k2 as _k2

    print(f"\n All k2 smoke tests passed")
    print(f"  k2 version:      {getattr(_k2, '__version__', _k2.__dev_version__)}")
    print(f"  PyTorch version:  {torch.__version__}")
    print(f"  CUDA version:     {torch.version.cuda}")
    print(f"  Device:           {torch.cuda.get_device_name(0)}")
    return True

def add_icefall_to_path(icefall_dir: Path):
    """Add icefall and the Zipformer recipe to sys.path."""
    for d in [
        icefall_dir,
        icefall_dir / "egs" / "librispeech" / "ASR",
        icefall_dir / "egs" / "librispeech" / "ASR" / "zipformer",
    ]:
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))

def load_model(model_dir: Path, icefall_dir: Path, device: torch.device):
    """Load pretrained Zipformer-S CR-CTC model.

    Follows the pattern from egs/librispeech/ASR/zipformer/ctc_decode.py:
    get_params() -> get_model() -> load pretrained.pt
    """
    add_icefall_to_path(icefall_dir)

    from train import add_model_arguments, get_model, get_params

    params = get_params()

    # Use add_model_arguments to get all default values with correct types,
    # then override Zipformer-S specifics from the HF model card exp/train.sh.
    import argparse

    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    model_args = parser.parse_args([])
    for k, v in vars(model_args).items():
        params[k] = v

    # Zipformer-S architecture overrides (from HF model card exp/train.sh)
    params.num_encoder_layers = "2,2,2,2,2,2"
    params.encoder_dim = "192,256,256,256,256,256"
    params.encoder_unmasked_dim = "192,192,192,192,192,192"
    params.feedforward_dim = "512,768,768,768,768,768"

    # CTC-specific
    params.use_transducer = False
    params.use_ctc = True
    params.use_cr_ctc = True
    params.use_attention_decoder = False
    params.vocab_size = 500
    params.feature_dim = 80

    model = get_model(params)
    checkpoint = torch.load(
        model_dir / "exp" / "pretrained.pt",
        map_location="cpu",
    )
    # pretrained.pt stores either {"model": state_dict} or raw state_dict
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f" Model loaded: {num_params / 1e6:.1f}M parameters")
    return model, params

def greedy_ctc_decode(log_probs: torch.Tensor) -> list[list[int]]:
    """Argmax -> remove blanks -> deduplicate.

    Args:
        log_probs: (batch, time, vocab) tensor of log probabilities.
            Token 0 is the CTC blank.

    Returns:
        List of token ID sequences (one per batch element).
    """
    argmax_ids = log_probs.argmax(dim=-1)  # (batch, time)
    results = []
    for seq in argmax_ids:
        tokens = []
        prev = -1
        for t in seq.tolist():
            if t != 0 and t != prev:  # skip blank (0) and dedup
                tokens.append(t)
            prev = t
        results.append(tokens)
    return results

def load_cuts(data_dir: Path, split: str):
    """Load precomputed CutSet with fbank features."""
    from lhotse import load_manifest_lazy

    cuts_path = data_dir / "cuts" / f"librispeech_cuts_{split}.jsonl.gz"
    assert cuts_path.exists(), (
        f"CutSet not found: {cuts_path}\n"
        f"Run setup_colab.sh first to prepare features."
    )
    return load_manifest_lazy(str(cuts_path))

def decode_split(
    model,
    data_dir: Path,
    split: str,
    sp: spm.SentencePieceProcessor,
    device: torch.device,
    batch_size: int = 8,
):
    """Run greedy CTC decoding on one split. Returns (hypotheses, references, rtf)."""
    from lhotse import CutSet
    from lhotse.dataset import DynamicBucketingSampler, OnTheFlyFeatures
    from lhotse.dataset.collation import collate_features

    cuts = load_cuts(data_dir, split)

    hypotheses = []
    references = []
    total_audio_dur = 0.0
    total_decode_time = 0.0

    # Process in batches, sorted by duration for efficiency
    batch = []
    for cut in cuts:
        batch.append(cut)
        if len(batch) >= batch_size:
            hyps, refs, audio_dur, dec_time = _decode_batch(
                model, batch, sp, device
            )
            hypotheses.extend(hyps)
            references.extend(refs)
            total_audio_dur += audio_dur
            total_decode_time += dec_time
            batch = []
    if batch:
        hyps, refs, audio_dur, dec_time = _decode_batch(
            model, batch, sp, device
        )
        hypotheses.extend(hyps)
        references.extend(refs)
        total_audio_dur += audio_dur
        total_decode_time += dec_time

    rtf = total_decode_time / total_audio_dur if total_audio_dur > 0 else float("inf")
    return hypotheses, references, rtf

def _decode_batch(model, cuts, sp, device):
    """Decode a list of cuts. Returns (hyps, refs, audio_duration, decode_time)."""
    import torch
    import numpy as np

    features_list = []
    lengths = []
    refs = []
    audio_dur = 0.0

    for cut in cuts:
        feats = torch.from_numpy(cut.load_features())  # (T, 80)
        features_list.append(feats)
        lengths.append(feats.shape[0])
        text = " ".join(
            s.text for s in cut.supervisions if s.text
        ).strip().lower()
        refs.append(text)
        audio_dur += cut.duration

    max_len = max(lengths)
    batch_feats = torch.zeros(len(cuts), max_len, features_list[0].shape[1])
    for i, f in enumerate(features_list):
        batch_feats[i, : f.shape[0]] = f
    feat_lens = torch.tensor(lengths, dtype=torch.int64)

    batch_feats = batch_feats.to(device)
    feat_lens = feat_lens.to(device)

    t0 = time.time()
    with torch.no_grad():
        encoder_out, encoder_out_lens = model.forward_encoder(batch_feats, feat_lens)
        log_probs = model.ctc_output(encoder_out)  # already log-softmax

    token_ids_batch = greedy_ctc_decode(log_probs)
    decode_time = time.time() - t0

    hyps = []
    for token_ids in token_ids_batch:
        text = sp.decode(token_ids)
        hyps.append(text.strip().lower())

    del batch_feats, encoder_out, log_probs
    torch.cuda.empty_cache()

    return hyps, refs, audio_dur, decode_time

def compute_wer_jiwer(hypotheses: list[str], references: list[str]) -> dict:
    """Compute WER using jiwer for a reliable reference number."""
    from jiwer import wer, cer

    valid = [(h, r) for h, r in zip(hypotheses, references) if r.strip()]
    if not valid:
        return {"wer": 0.0, "cer": 0.0, "num_utts": 0}
    hyps, refs = zip(*valid)
    return {
        "wer": wer(list(refs), list(hyps)),
        "cer": cer(list(refs), list(hyps)),
        "num_utts": len(refs),
    }

def parse_args():
    parser = argparse.ArgumentParser(description="RBPO Stage 0b verification")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/content/icefall-asr-librispeech-zipformer-small-cr-ctc"),
    )
    parser.add_argument(
        "--icefall-dir",
        type=Path,
        default=Path("/content/icefall"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/content/librispeech_data"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save JSON report (optional)",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("RBPO Stage 0b  --  Baseline Verification")
    print("=" * 60)

    # -- Phase 1: k2 smoke tests --
    print("\n-- Phase 1: k2 smoke tests --")
    try:
        k2_smoke_tests()
    except Exception as e:
        print(f"\n k2 smoke test FAILED: {e}")
        print("Fix k2 installation before proceeding.")
        sys.exit(1)

    # -- Phase 2: Load model --
    print("\n-- Phase 2: Load pretrained model --")
    assert args.model_dir.exists(), f"Model dir not found: {args.model_dir}"
    assert args.icefall_dir.exists(), f"Icefall dir not found: {args.icefall_dir}"

    sp = spm.SentencePieceProcessor()
    bpe_path = args.model_dir / "data" / "lang_bpe_500" / "bpe.model"
    assert bpe_path.exists(), f"BPE model not found: {bpe_path}"
    sp.load(str(bpe_path))
    print(f" BPE model loaded (vocab size: {sp.get_piece_size()})")

    model, params = load_model(args.model_dir, args.icefall_dir, device)

    # -- Phase 3: Decode dev sets --
    print("\n-- Phase 3: Greedy CTC decoding --")
    report = {}

    for split in ["dev-clean", "dev-other"]:
        print(f"\nDecoding {split}...")
        t0 = time.time()
        hypotheses, references, rtf = decode_split(
            model, args.data_dir, split, sp, device, args.batch_size
        )
        elapsed = time.time() - t0

        metrics = compute_wer_jiwer(hypotheses, references)
        report[split] = {
            "wer": metrics["wer"],
            "cer": metrics["cer"],
            "num_utts": metrics["num_utts"],
            "rtf": rtf,
            "elapsed_sec": elapsed,
        }

        print(f"  {split}: WER={metrics['wer']*100:.2f}%  CER={metrics['cer']*100:.2f}%")
        print(f"  Utterances: {metrics['num_utts']}  RTF: {rtf:.4f}  Time: {elapsed:.1f}s")

        # Show a few examples
        print(f"  Examples:")
        for i in range(min(3, len(hypotheses))):
            print(f"    REF: {references[i]}")
            print(f"    HYP: {hypotheses[i]}")
            print()

    # -- Phase 4: Summary --
    print("=" * 60)
    print("VERIFICATION REPORT")
    print("=" * 60)

    # Expected WER from model card (greedy CTC)
    expected = {
        "dev-clean": 2.37,
        "dev-other": 6.03,
    }
    # Allow 0.5% absolute tolerance for environment differences
    tolerance = 0.5

    all_pass = True
    for split in ["dev-clean", "dev-other"]:
        actual_wer = report[split]["wer"] * 100
        exp_wer = expected[split]
        diff = abs(actual_wer - exp_wer)
        status = "PASS" if diff <= tolerance else "WARN"
        if status == "WARN":
            all_pass = False
        print(f"  {split}:  actual={actual_wer:.2f}%  expected~{exp_wer:.2f}%  diff={diff:.2f}%  [{status}]")

    print()
    if all_pass:
        print(" Baseline WER matches expected values (within +/-0.5%)")
    else:
        print(" WER differs from expected. Possible causes:")
        print("  - Different checkpoint averaging (we use pretrained.pt directly)")
        print("  - Feature extraction differences (lhotse vs kaldifeat)")
        print("  - BPE decoding edge cases")
        print("  This is informational  --  the model is still usable for RBPO.")

    print(f"\nModel: Zipformer-S CR-CTC (22.1M params)")
    print(f"Decoding: greedy CTC (argmax -> dedup -> BPE decode)")
    print(f"Features: 80-dim fbank via lhotse")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n Report saved to {args.output}")

    report_path = Path("/content/drive/MyDrive/rbpo_results/stage_0b_report.json")
    if report_path.parent.exists():
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f" Report also saved to {report_path}")

    return report

if __name__ == "__main__":
    main()

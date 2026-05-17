#!/usr/bin/env python3
"""Generate N-best hypotheses from a CutSet using CTC lattice sampling.

Lattice construction matches E11 (experiments/analysis/g_scaling_curve.py) exactly:
  - k2.ctc_topo(max_token=499, modified=False)
  - k2.intersect_dense(topo, dense_fsa, output_beam=8.0)
  - k2.Nbest.from_lattice(lattice, num_paths=G*4, nbest_scale=1.0, use_double_scores=True)
  - CTC scores via frame-level alignment log-prob
  - Greedy injected as rank-0 candidate

Usage:
    python scripts/generate_nbest.py \
        --cuts /path/to/cuts.jsonl.gz \
        --checkpoint /path/to/pretrained.pt \
        --bpe /path/to/bpe.model \
        --G 16 \
        --output /path/to/nbest.jsonl \
        [--nbest-scale 1.0] \
        [--oversample-factor 4] \
        [--output-beam 8.0] \
        [--device cuda:0] \
        [--batch-size 1]
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

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


def ctc_collapse(token_ids):
    result = []
    prev = None
    for t in token_ids:
        if t != BLANK_ID and t != prev:
            result.append(t)
        prev = t
    return result


def alignment_log_prob(label_seq, log_probs_cpu):
    """CTC alignment score: sum log P(label_t | t) over all frames.

    Matches E11's scoring exactly.
    """
    import torch
    T = log_probs_cpu.shape[0]
    if len(label_seq) != T:
        return float("-inf")
    idx = torch.tensor(label_seq, dtype=torch.long)
    return log_probs_cpu[torch.arange(T), idx].sum().item()


def add_icefall_to_path(icefall_dir: Path):
    for d in [
        icefall_dir,
        icefall_dir / "egs" / "librispeech" / "ASR",
        icefall_dir / "egs" / "librispeech" / "ASR" / "zipformer",
    ]:
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))


def load_model(checkpoint_path: Path, icefall_dir: Path, device):
    import torch
    add_icefall_to_path(icefall_dir)
    from train import add_model_arguments, get_model, get_params

    params = get_params()
    parser = argparse.ArgumentParser(add_help=False)
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
    params.vocab_size = VOCAB_SIZE
    params.feature_dim = 80

    model = get_model(params)
    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False
    )
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params / 1e6:.1f}M parameters on {device}")
    return model


def extract_features(cut, fbank_extractor):
    """On-the-fly fbank extraction from a lhotse Cut."""
    audio = cut.load_audio()
    feat = fbank_extractor.extract(audio, sampling_rate=16000)
    return feat


def generate_nbest_for_utt(
    log_probs_utt, topo, num_paths, nbest_scale, sp, device
):
    """Build lattice and extract N-best. Matches E11 exactly."""
    import k2
    import torch

    T = log_probs_utt.shape[0]
    lp_cpu = log_probs_utt.cpu()

    # Greedy decode
    greedy_ids = log_probs_utt.argmax(dim=-1).cpu().tolist()
    greedy_collapsed = ctc_collapse(greedy_ids)
    greedy_text = normalize_text(sp.decode(greedy_collapsed))
    greedy_score = alignment_log_prob(greedy_ids, lp_cpu)

    # Build lattice (E11: output_beam=8.0)
    supervision_segments = torch.tensor([[0, 0, T]], dtype=torch.int32)
    dense_fsa = k2.DenseFsaVec(log_probs_utt.unsqueeze(0), supervision_segments)
    lattice = k2.intersect_dense(topo, dense_fsa, output_beam=8.0)
    lattice = k2.connect(lattice)

    # Sample paths (E11: use_double_scores=True)
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

    # Inject greedy as rank-0
    greedy_entry = {"hyp": greedy_text, "score": round(greedy_score, 6)}
    if greedy_text in seen:
        seen[greedy_text] = greedy_entry
    else:
        seen[greedy_text] = greedy_entry

    candidates = sorted(seen.values(), key=lambda c: c["score"], reverse=True)

    # Ensure greedy is first regardless of score
    rest = [c for c in candidates if c["hyp"] != greedy_text]
    candidates = [greedy_entry] + rest

    del lattice, nbest
    return candidates


def main():
    parser = argparse.ArgumentParser(
        description="Generate N-best hypotheses via CTC lattice sampling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cuts", type=Path, required=True,
                        help="Path to lhotse CutSet (jsonl.gz)")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to pretrained.pt")
    parser.add_argument("--bpe", type=Path, required=True,
                        help="Path to bpe.model")
    parser.add_argument("--icefall-dir", type=Path, default=Path("/content/icefall"),
                        help="Path to icefall repo")
    parser.add_argument("--G", type=int, required=True,
                        help="Number of hypotheses to keep per utterance")
    parser.add_argument("--nbest-scale", type=float, default=1.0,
                        help="Lattice score scale for path sampling (E11=1.0)")
    parser.add_argument("--oversample-factor", type=int, default=4,
                        help="Sample G*factor paths before dedup")
    parser.add_argument("--output-beam", type=float, default=8.0,
                        help="Output beam for intersect_dense (E11=8.0)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSONL path")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Utterances per forward pass")
    args = parser.parse_args()

    assert args.nbest_scale != 0.5, (
        "nbest_scale=0.5 is the known bug from E23/E24. "
        "Use 1.0 to match E11. Pass --nbest-scale 1.0 explicitly."
    )

    num_paths = args.G * args.oversample_factor

    print("=" * 70)
    print("generate_nbest.py  --  CTC lattice N-best generation")
    print("=" * 70)
    print(f"  cuts:             {args.cuts}")
    print(f"  checkpoint:       {args.checkpoint}")
    print(f"  bpe:              {args.bpe}")
    print(f"  icefall_dir:      {args.icefall_dir}")
    print(f"  G:                {args.G}")
    print(f"  nbest_scale:      {args.nbest_scale}")
    print(f"  oversample:       {num_paths} paths -> top-{args.G}")
    print(f"  output_beam:      {args.output_beam}")
    print(f"  output:           {args.output}")
    print(f"  device:           {args.device}")
    print(f"  batch_size:       {args.batch_size}")
    print()

    import torch
    import sentencepiece as spm
    import k2
    from lhotse import Fbank, FbankConfig, load_manifest_lazy

    device = torch.device(args.device)

    print("Loading model...")
    model = load_model(args.checkpoint, args.icefall_dir, device)

    sp = spm.SentencePieceProcessor()
    sp.load(str(args.bpe))
    assert sp.get_piece_size() == VOCAB_SIZE, (
        f"BPE vocab size {sp.get_piece_size()} != expected {VOCAB_SIZE}"
    )

    print("Loading CutSet...")
    cuts = list(load_manifest_lazy(str(args.cuts)))
    print(f"  {len(cuts)} utterances")

    # Feature extractor
    fbank = Fbank(FbankConfig(num_mel_bins=80, dither=0.0))

    # CTC topology
    topo = k2.ctc_topo(max_token=MAX_TOKEN, modified=False, device=device)

    # Generate
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_cands = 0
    t0 = time.time()

    with open(args.output, "w") as f_out:
        for i, cut in enumerate(cuts):
            feat = extract_features(cut, fbank)
            feat_t = torch.from_numpy(feat).unsqueeze(0).to(device)
            feat_lens = torch.tensor(
                [feat.shape[0]], dtype=torch.int64, device=device
            )

            with torch.no_grad():
                enc_out, enc_lens = model.forward_encoder(feat_t, feat_lens)
                log_probs = model.ctc_output(enc_out)

            lp_utt = log_probs[0, :enc_lens[0].item()]

            candidates = generate_nbest_for_utt(
                lp_utt, topo, num_paths, args.nbest_scale, sp, device
            )
            candidates = candidates[:args.G]

            ref_raw = " ".join(s.text for s in cut.supervisions if s.text)
            ref = normalize_text(ref_raw)

            record = {
                "utt_id": cut.id,
                "ref": ref,
                "nbest": candidates,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_cands += len(candidates)

            del log_probs, enc_out, feat_t
            torch.cuda.empty_cache()

            if (i + 1) % 50 == 0 or i == len(cuts) - 1:
                elapsed = time.time() - t0
                speed = (i + 1) / elapsed
                eta = (len(cuts) - i - 1) / speed if speed > 0 else 0
                avg_c = total_cands / (i + 1)
                print(
                    f"  {i+1}/{len(cuts)}  avg_cands={avg_c:.1f}  "
                    f"({speed:.1f} utt/s, ETA {eta:.0f}s)"
                )

    elapsed = time.time() - t0
    avg_cands = total_cands / len(cuts)
    size_mb = args.output.stat().st_size / 1e6

    print()
    print(f"Done: {len(cuts)} utterances, {total_cands} total candidates")
    print(f"  Mean candidates/utt: {avg_cands:.1f}")
    print(f"  Wall time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  Output: {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

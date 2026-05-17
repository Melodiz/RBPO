#!/usr/bin/env python3
"""B1: Stage 2 gamma_t measurement  --  per-frame posterior analysis.

Loads Zipformer-S CR-CTC model, runs forward pass on dev-other utterances,
and classifies each frame as dead/active/ambiguous based on posterior mass.

Usage (Colab):
    python experiments/analysis/compute_gamma_analysis.py \
        --checkpoint /path/to/pretrained.pt \
        --manifest /path/to/dev-other-cuts.jsonl.gz \
        --num-utts 100 \
        --output-dir results/stage2/
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
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


def add_icefall_to_path(icefall_dir: Path):
    for d in [
        icefall_dir,
        icefall_dir / "egs" / "librispeech" / "ASR",
        icefall_dir / "egs" / "librispeech" / "ASR" / "zipformer",
    ]:
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))


def load_model(checkpoint_path, icefall_dir, device):
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
    return model


def classify_frames(log_probs, dead_thresh=0.99, active_thresh=0.5):
    """Classify each frame as dead/active/ambiguous from log-probabilities.

    Returns per-frame classification and entropy.
    """
    probs = np.exp(log_probs)  # (T, V)
    T, V = probs.shape

    blank_post = probs[:, BLANK_ID]
    max_nonblank = np.max(probs[:, 1:], axis=1)

    dead = blank_post > dead_thresh
    active = (~dead) & (max_nonblank > active_thresh)
    ambiguous = (~dead) & (~active)

    eps = 1e-10
    entropy = -np.sum(probs * np.log(probs + eps), axis=1)

    return {
        "dead": dead,
        "active": active,
        "ambiguous": ambiguous,
        "entropy": entropy,
        "dead_frac": float(dead.mean()),
        "active_frac": float(active.mean()),
        "ambig_frac": float(ambiguous.mean()),
        "mean_entropy": float(entropy.mean()),
        "T": T,
    }


def run_dummy_test():
    """Verify classification logic with synthetic data."""
    np.random.seed(42)
    T, V = 10, 501
    logits = np.random.randn(T, V).astype(np.float32)

    logits[0, BLANK_ID] = 20.0
    logits[1, BLANK_ID] = 20.0
    logits[2, 50] = 20.0

    log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    result = classify_frames(log_probs)

    assert result["dead"][0], "Frame 0 should be dead (high blank)"
    assert result["dead"][1], "Frame 1 should be dead (high blank)"
    assert result["active"][2], "Frame 2 should be active (high non-blank)"
    assert 0 <= result["mean_entropy"], "Entropy must be non-negative"
    assert result["T"] == T
    assert abs(result["dead_frac"] + result["active_frac"] + result["ambig_frac"] - 1.0) < 1e-6

    print("  [PASS] Dummy test: classification logic verified")


def main():
    parser = argparse.ArgumentParser(
        description="B1: Stage 2 gamma_t analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="CutSet manifest (dev-other)")
    parser.add_argument("--icefall-dir", type=Path,
                        default=Path("/content/icefall"))
    parser.add_argument("--bpe", type=Path,
                        default=Path("/content/icefall/egs/librispeech/ASR/data/lang_bpe_500/bpe.model"))
    parser.add_argument("--num-utts", type=int, default=100)
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "results" / "stage2")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("B1: Stage 2 gamma_t Measurement")
    print("=" * 70)
    t0 = time.time()

    run_dummy_test()

    import torch
    import sentencepiece as spm
    from lhotse import CutSet, Fbank, FbankConfig

    sp = spm.SentencePieceProcessor()
    sp.load(str(args.bpe))

    fbank = Fbank(FbankConfig(num_mel_bins=80))
    cuts = CutSet.from_file(args.manifest)
    cuts_list = list(cuts)[:args.num_utts]

    print(f"\n  Processing {len(cuts_list)} utterances on {args.device}")

    model = load_model(args.checkpoint, args.icefall_dir, args.device)

    per_utt = []
    for i, cut in enumerate(cuts_list):
        audio = cut.load_audio()
        feat = fbank.extract(audio, sampling_rate=16000)
        feat_tensor = torch.from_numpy(feat).unsqueeze(0).to(args.device)
        feat_lens = torch.tensor([feat.shape[0]], dtype=torch.int64, device=args.device)

        with torch.no_grad():
            encoder_out, encoder_out_lens = model.forward_encoder(feat_tensor, feat_lens)
            log_probs = model.ctc_output(encoder_out)

        lp_np = log_probs[0, :encoder_out_lens[0].item()].cpu().numpy()

        result = classify_frames(lp_np)

        ref_text = normalize_text(cut.supervisions[0].text)
        ref_tokens = sp.encode(ref_text, out_type=int)
        L = len(ref_tokens)
        T = result["T"]

        row = {
            "utt_id": cut.id,
            "T": T,
            "L": L,
            "T_over_L": T / max(L, 1),
            "dead_frac": result["dead_frac"],
            "active_frac": result["active_frac"],
            "ambig_frac": result["ambig_frac"],
            "mean_entropy": result["mean_entropy"],
        }
        per_utt.append(row)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(cuts_list)} done")

    print(f"\n  All {len(per_utt)} utterances processed")

    csv_path = args.output_dir / "gamma_stats.csv"
    fields = list(per_utt[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(per_utt)
    print(f"  Wrote {csv_path}")

    dead_fracs = [r["dead_frac"] for r in per_utt]
    active_fracs = [r["active_frac"] for r in per_utt]
    ambig_fracs = [r["ambig_frac"] for r in per_utt]
    entropies = [r["mean_entropy"] for r in per_utt]
    t_over_l = [r["T_over_L"] for r in per_utt]

    summary = {
        "n_utterances": len(per_utt),
        "dead_frac": {"mean": float(np.mean(dead_fracs)), "std": float(np.std(dead_fracs)),
                      "min": float(np.min(dead_fracs)), "max": float(np.max(dead_fracs))},
        "active_frac": {"mean": float(np.mean(active_fracs)), "std": float(np.std(active_fracs)),
                        "min": float(np.min(active_fracs)), "max": float(np.max(active_fracs))},
        "ambig_frac": {"mean": float(np.mean(ambig_fracs)), "std": float(np.std(ambig_fracs)),
                       "min": float(np.min(ambig_fracs)), "max": float(np.max(ambig_fracs))},
        "entropy": {"mean": float(np.mean(entropies)), "std": float(np.std(entropies)),
                    "min": float(np.min(entropies)), "max": float(np.max(entropies))},
        "T_over_L": {"mean": float(np.mean(t_over_l)), "std": float(np.std(t_over_l)),
                     "min": float(np.min(t_over_l)), "max": float(np.max(t_over_l))},
    }

    json_path = args.output_dir / "gamma_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Wrote {json_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        lengths = sorted(range(len(per_utt)), key=lambda i: per_utt[i]["T"])
        for tag, pct in [("short", 10), ("mid", 50), ("long", 90)]:
            idx = lengths[int(len(lengths) * pct / 100)]
            utt = per_utt[idx]
            print(f"  Heatmap example '{tag}': utt {utt['utt_id']} (T={utt['T']})")
    except ImportError:
        print("  matplotlib not available, skipping heatmaps")

    elapsed = time.time() - t0
    print(f"\n  Summary:")
    for k in ["dead_frac", "active_frac", "ambig_frac", "entropy", "T_over_L"]:
        s = summary[k]
        print(f"    {k}: mean={s['mean']:.4f}, std={s['std']:.4f}, "
              f"range=[{s['min']:.4f}, {s['max']:.4f}]")
    print(f"\n  Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

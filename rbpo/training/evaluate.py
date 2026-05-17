#!/usr/bin/env python3
"""Stage 4: Standalone WER evaluation for fine-tuned RBPO checkpoints.

Loads a checkpoint, runs greedy CTC decoding on dev splits, reports WER.

Usage:
    python training/evaluate.py \\
        --model-dir /content/icefall-asr-librispeech-zipformer-small-cr-ctc \\
        --icefall-dir /content/icefall \\
        --data-dir /content/librispeech_data \\
        --checkpoint /path/to/checkpoint_epoch_10.pt \\
        --splits dev-clean dev-other \\
        --output /path/to/eval_results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import sentencepiece as spm
import torch

THIS_FILE = Path(__file__).resolve()
RBPO_ROOT = THIS_FILE.parent.parent
if str(RBPO_ROOT) not in sys.path:
    sys.path.insert(0, str(RBPO_ROOT))


def add_icefall_to_path(icefall_dir: Path):
    for d in [
        icefall_dir,
        icefall_dir / "egs" / "librispeech" / "ASR",
        icefall_dir / "egs" / "librispeech" / "ASR" / "zipformer",
    ]:
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))


def load_model(
    model_dir: Path,
    icefall_dir: Path,
    device: torch.device,
    checkpoint: Path = None,
):
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

    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location="cpu")
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        ckpt = torch.load(model_dir / "exp" / "pretrained.pt", map_location="cpu")
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"Loaded pretrained model: {model_dir / 'exp' / 'pretrained.pt'}")

    model.eval()
    model.to(device)
    return model


def greedy_ctc_decode(log_probs):
    argmax_ids = log_probs.argmax(dim=-1)
    out = []
    for seq in argmax_ids:
        toks = []
        prev = -1
        for t in seq.tolist():
            if t != 0 and t != prev:
                toks.append(t)
            prev = t
        out.append(toks)
    return out


@torch.no_grad()
def eval_split(model, data_dir, split, sp, device, batch_size=8, max_utts=0):
    from lhotse import load_manifest_lazy
    import editdistance

    path = Path(data_dir) / "cuts" / f"librispeech_cuts_{split}.jsonl.gz"
    assert path.exists(), f"Cuts not found: {path}"
    cuts = load_manifest_lazy(str(path))

    hyps = []
    refs = []
    batch = []
    n = 0
    audio_dur = 0.0
    t0 = time.time()

    def flush():
        if not batch:
            return
        feats_list = [torch.from_numpy(c.load_features()) for c in batch]
        lengths = [f.shape[0] for f in feats_list]
        max_len = max(lengths)
        bf = torch.zeros(len(batch), max_len, feats_list[0].shape[1])
        for i, f in enumerate(feats_list):
            bf[i, : f.shape[0]] = f
        bf = bf.to(device)
        feat_lens = torch.tensor(lengths, dtype=torch.int64, device=device)

        encoder_out, _ = model.forward_encoder(bf, feat_lens)
        log_probs = model.ctc_output(encoder_out)
        token_ids_batch = greedy_ctc_decode(log_probs)
        for c, toks in zip(batch, token_ids_batch):
            text = sp.decode(toks).strip().lower()
            ref = " ".join(s.text for s in c.supervisions if s.text).strip().lower()
            hyps.append(text)
            refs.append(ref)
        del bf, encoder_out, log_probs
        torch.cuda.empty_cache()

    for cut in cuts:
        batch.append(cut)
        audio_dur += cut.duration
        n += 1
        if len(batch) >= batch_size:
            flush()
            batch = []
        if max_utts and n >= max_utts:
            break
    flush()

    decode_time = time.time() - t0

    valid = [(h, r) for h, r in zip(hyps, refs) if r.strip()]
    if not valid:
        return {"wer": None, "num_utts": 0}

    wer_sum = 0.0
    for h, r in valid:
        ref_words = r.split()
        hyp_words = h.split()
        if not ref_words:
            continue
        wer_sum += editdistance.eval(hyp_words, ref_words) / len(ref_words)
    mean_wer = wer_sum / len(valid)

    return {
        "wer": mean_wer,
        "num_utts": len(valid),
        "rtf": decode_time / audio_dur if audio_dur > 0 else None,
        "decode_time_sec": round(decode_time, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="RBPO checkpoint evaluation")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--icefall-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Checkpoint to evaluate. If omitted, evaluates the pretrained model.",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["dev-clean", "dev-other"],
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-utts", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)

    sp = spm.SentencePieceProcessor()
    sp.load(str(args.model_dir / "data" / "lang_bpe_500" / "bpe.model"))

    model = load_model(
        args.model_dir, args.icefall_dir, device, args.checkpoint
    )

    results = {}
    print("\n" + "=" * 60)
    print("Evaluation")
    print("=" * 60)
    for split in args.splits:
        m = eval_split(
            model, args.data_dir, split, sp, device,
            args.batch_size, args.max_utts,
        )
        results[split] = m
        wer_str = f"{m['wer']*100:.2f}%" if m["wer"] is not None else "n/a"
        rtf_str = f"{m['rtf']:.4f}" if m.get("rtf") else "n/a"
        print(
            f"  {split}: WER={wer_str} "
            f"({m['num_utts']} utts, RTF={rtf_str}, "
            f"{m.get('decode_time_sec', 0)}s)"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({
                "checkpoint": str(args.checkpoint) if args.checkpoint else "pretrained",
                "results": results,
            }, f, indent=2)
        print(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()

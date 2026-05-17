#!/usr/bin/env python3
"""Generate and cache N-best hypotheses for train-clean-100.

Identical pipeline to generate_nbest.py but targets train-clean-100
(~28,539 utterances) for training a discriminative rescorer.

Usage:
    python experiments/generate_nbest_train.py \
        --model-dir /content/icefall-asr-librispeech-zipformer-small-cr-ctc \
        --data-dir /content/librispeech_data \
        --results-dir results \
        --device cuda:0 \
        --num-utterances -1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import sentencepiece as spm
import torch
from tqdm import tqdm

BLANK_ID = 0
MAX_TOKEN = 499
NUM_PATHS_OVERSAMPLE = 64
G = 16
NBEST_SCALE = 1.0


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


def ctc_collapse(token_ids: list[int]) -> list[int]:
    result = []
    prev = None
    for t in token_ids:
        if t != BLANK_ID and t != prev:
            result.append(t)
        prev = t
    return result


def build_lattice(log_probs: torch.Tensor, topo, device: torch.device):
    import k2

    T = log_probs.shape[0]
    supervision_segments = torch.tensor([[0, 0, T]], dtype=torch.int32)
    dense_fsa = k2.DenseFsaVec(log_probs.unsqueeze(0), supervision_segments)
    lattice = k2.intersect_dense(topo, dense_fsa, output_beam=8.0)
    lattice = k2.connect(lattice)
    return lattice


def alignment_log_prob(label_seq: list[int], log_probs_cpu: torch.Tensor) -> float:
    T = log_probs_cpu.shape[0]
    if len(label_seq) != T:
        return float("-inf")
    idx = torch.tensor(label_seq, dtype=torch.long)
    return log_probs_cpu[torch.arange(T), idx].sum().item()


def extract_nbest_with_scores(lattice, num_paths, nbest_scale, sp, log_probs_cpu):
    import k2

    nbest = k2.Nbest.from_lattice(
        lattice,
        num_paths=num_paths,
        use_double_scores=True,
        nbest_scale=nbest_scale,
    )

    all_labels = nbest.fsa.labels.cpu().tolist()
    paths_labels = []
    current = []
    for label in all_labels:
        if label == -1:
            paths_labels.append(current)
            current = []
        else:
            current.append(label)

    seen = {}
    for raw_ids in paths_labels:
        score = alignment_log_prob(raw_ids, log_probs_cpu)
        if score == float("-inf"):
            continue

        token_ids = ctc_collapse(raw_ids)
        text = sp.decode(token_ids).strip().lower()

        entry = {
            "text": text,
            "tokens": token_ids,
            "ctc_log_prob": score,
            "len_tokens": len(token_ids),
            "len_chars": len(text),
        }

        if text in seen:
            if score > seen[text]["ctc_log_prob"]:
                seen[text] = entry
        else:
            seen[text] = entry

    candidates = sorted(seen.values(), key=lambda c: c["ctc_log_prob"], reverse=True)
    return candidates


def load_all_utterances(data_dir: Path, split: str):
    from lhotse import load_manifest_lazy

    cuts_path = data_dir / "cuts" / f"librispeech_cuts_{split}.jsonl.gz"
    assert cuts_path.exists(), f"CutSet not found: {cuts_path}"

    cuts = load_manifest_lazy(str(cuts_path))
    utterances = []
    for cut in cuts:
        feats = torch.from_numpy(cut.load_features())
        ref_text = " ".join(
            s.text for s in cut.supervisions if s.text
        ).strip().lower()
        if not ref_text:
            continue
        utterances.append((cut.id, feats, ref_text))

    print(f"Loaded {len(utterances)} utterances from {split}")
    return utterances


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and cache N-best data for train-clean-100"
    )
    parser.add_argument(
        "--model-dir", type=Path,
        default=Path("/content/icefall-asr-librispeech-zipformer-small-cr-ctc"),
    )
    parser.add_argument(
        "--icefall-dir", type=Path,
        default=Path("/content/icefall"),
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("/content/librispeech_data"),
    )
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path("/content/drive/MyDrive/rbpo_results"),
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-utterances", type=int, default=-1,
                        help="Limit utterances (-1 = all, 10000 = subsample)")
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=2000,
                        help="Flush file every N utterances")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    split = "train-clean-100"

    print("=" * 60)
    print(f"N-best Generation  --  {split} G={G}, nbest_scale={NBEST_SCALE}")
    print("=" * 60)

    sp = spm.SentencePieceProcessor()
    bpe_path = args.model_dir / "data" / "lang_bpe_500" / "bpe.model"
    assert bpe_path.exists(), f"BPE model not found: {bpe_path}"
    sp.load(str(bpe_path))
    print(f"BPE vocab: {sp.get_piece_size()} tokens")

    model = load_model(args.model_dir, args.icefall_dir, device)
    utterances = load_all_utterances(args.data_dir, split)

    if args.num_utterances > 0:
        utterances = utterances[:args.num_utterances]
        print(f"Limited to {len(utterances)} utterances")

    import k2
    topo = k2.ctc_topo(max_token=MAX_TOKEN, modified=False, device=device)
    print(f"CTC topology: {topo.num_arcs} arcs")

    output_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "nbest_train_clean100_G16.jsonl"

    t0 = time.time()
    total_candidates = 0

    with open(out_path, "w") as f:
        for utt_idx, (utt_id, feats, ref_text) in enumerate(
            tqdm(utterances, desc=f"Generating N-best ({split})")
        ):
            feats_gpu = feats.unsqueeze(0).to(device)
            feat_lens = torch.tensor(
                [feats.shape[0]], dtype=torch.int64, device=device
            )

            with torch.no_grad():
                encoder_out, encoder_out_lens = model.forward_encoder(
                    feats_gpu, feat_lens
                )
                log_probs = model.ctc_output(encoder_out)

            log_probs_utt = log_probs[0]
            lattice = build_lattice(log_probs_utt, topo, device)

            log_probs_cpu = log_probs_utt.cpu()

            greedy_ids = log_probs_utt.argmax(dim=-1).tolist()
            greedy_collapsed = ctc_collapse(greedy_ids)
            greedy_text = sp.decode(greedy_collapsed).strip().lower()
            greedy_score = alignment_log_prob(greedy_ids, log_probs_cpu)

            candidates = extract_nbest_with_scores(
                lattice, NUM_PATHS_OVERSAMPLE, NBEST_SCALE, sp, log_probs_cpu
            )

            greedy_entry = None
            rest = []
            for c in candidates:
                if c["text"] == greedy_text and greedy_entry is None:
                    greedy_entry = c
                else:
                    rest.append(c)

            if greedy_entry is None:
                greedy_entry = {
                    "text": greedy_text,
                    "tokens": greedy_collapsed,
                    "ctc_log_prob": greedy_score,
                    "len_tokens": len(greedy_collapsed),
                    "len_chars": len(greedy_text),
                }
            else:
                greedy_entry["ctc_log_prob"] = greedy_score
                greedy_entry["tokens"] = greedy_collapsed

            candidates = [greedy_entry] + rest
            candidates = candidates[:G]

            for c in candidates:
                c["ctc_log_prob"] = round(c["ctc_log_prob"], 6)

            record = {
                "utt_id": utt_id,
                "ref_text": ref_text,
                "num_candidates": len(candidates),
                "candidates": candidates,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_candidates += len(candidates)

            del lattice, log_probs, encoder_out, feats_gpu
            torch.cuda.empty_cache()

            if (utt_idx + 1) % args.log_every == 0:
                elapsed = time.time() - t0
                rate = (utt_idx + 1) / elapsed
                eta = (len(utterances) - utt_idx - 1) / rate
                print(
                    f"  [{utt_idx+1}/{len(utterances)}] "
                    f"{rate:.1f} utt/s, ETA {eta:.0f}s"
                )

            if (utt_idx + 1) % args.save_every == 0:
                f.flush()

    elapsed = time.time() - t0
    print(f"\nDone: {len(utterances)} utterances, {total_candidates} total candidates")
    print(f"Avg candidates per utterance: {total_candidates / len(utterances):.1f}")
    print(f"Output: {out_path}")
    print(f"Time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()

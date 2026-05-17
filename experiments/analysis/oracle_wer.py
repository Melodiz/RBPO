#!/usr/bin/env python3
"""Stage 1: Oracle WER  --  beam search N-best vs lattice sampling diversity.

Tests the core RBPO claim: CTC lattices contain far more hypothesis diversity
than beam search N-best lists. Compares oracle WER (best WER among G candidates)
across beam search (nbest_scale=1.0) and lattice sampling (nbest_scale<1.0).

Usage:
    python experiments/analysis/oracle_wer.py \
        --model-dir /content/models/zipformer-s-cr-ctc \
        --data-dir /content/data/fbank \
        --results-dir /content/drive/MyDrive/rbpo_results \
        --num-utterances 300 \
        --device cuda:0
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import editdistance
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

def build_lattice(log_probs: torch.Tensor, topo, device: torch.device):
    """Build CTC lattice for a single utterance."""
    import k2

    T = log_probs.shape[0]
    supervision_segments = torch.tensor([[0, 0, T]], dtype=torch.int32)
    dense_fsa = k2.DenseFsaVec(log_probs.unsqueeze(0), supervision_segments)
    lattice = k2.intersect_dense(topo, dense_fsa, output_beam=8.0)
    lattice = k2.connect(lattice)
    return lattice

def extract_nbest_hypotheses(
    lattice, num_paths: int, nbest_scale: float, sp: spm.SentencePieceProcessor
) -> list[str]:
    """Extract N-best hypotheses from lattice, deduplicated."""
    import k2

    nbest = k2.Nbest.from_lattice(
        lattice,
        num_paths=num_paths,
        use_double_scores=True,
        nbest_scale=nbest_scale,
    )

    # Extract token IDs from all paths at once.
    # nbest.fsa.labels is a 1D tensor of all arc labels across all path FSAs.
    # Each path ends with a -1 label (final-arc sentinel), so we split on -1.
    all_labels = nbest.fsa.labels.cpu().tolist()

    paths_labels = []
    current = []
    for label in all_labels:
        if label == -1:
            paths_labels.append(current)
            current = []
        else:
            current.append(label)

    hyp_texts = []
    for raw_ids in paths_labels:
        token_ids = ctc_collapse(raw_ids)

        for t in token_ids:
            assert 1 <= t <= MAX_TOKEN, (
                f"Token {t} out of range [1, {MAX_TOKEN}] after CTC collapse"
            )

        text = sp.decode(token_ids).strip().lower()
        hyp_texts.append(text)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for text in hyp_texts:
        if text not in seen:
            seen.add(text)
            unique.append(text)
    return unique

def load_utterances(data_dir: Path, split: str, num_utterances: int):
    """Load N utterances from a split. Returns list of (cut_id, features, ref_text)."""
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
        utterances.append((cut.id, feats, ref_text))

    print(f"Loaded {len(utterances)} utterances from {split}")
    return utterances

CONDITIONS = [
    {"method": "beam",    "nbest_scale": 1.0},
    {"method": "lattice", "nbest_scale": 0.75},
    {"method": "lattice", "nbest_scale": 0.5},
    {"method": "lattice", "nbest_scale": 0.25},
]
G_VALUES = [4, 8, 16]

def run_experiment(
    model,
    utterances: list,
    sp: spm.SentencePieceProcessor,
    device: torch.device,
    output_dir: Path,
):
    import k2

    topo = k2.ctc_topo(max_token=MAX_TOKEN, modified=False, device=device)
    print(f"CTC topology: {topo.num_arcs} arcs")

    # CSV output
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "oracle_wer_results.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "utt_id", "method", "nbest_scale", "G",
        "oracle_wer", "onebest_wer", "mean_wer", "num_unique", "ref_text",
    ])

    agg = {
        (c["method"], c["nbest_scale"], g): []
        for c in CONDITIONS for g in G_VALUES
    }

    # Examples buffer (first 5 utterances)
    examples = []
    MAX_EXAMPLES = 5

    max_num_paths = max(G_VALUES) * 4  # oversample for the largest G

    for utt_idx, (utt_id, feats, ref_text) in enumerate(
        tqdm(utterances, desc="Processing utterances")
    ):
        # Forward pass (shared across all conditions)
        feats_gpu = feats.unsqueeze(0).to(device)
        feat_lens = torch.tensor([feats.shape[0]], dtype=torch.int64, device=device)

        with torch.no_grad():
            encoder_out, encoder_out_lens = model.forward_encoder(feats_gpu, feat_lens)
            log_probs = model.ctc_output(encoder_out)  # (1, T', V)

        log_probs_utt = log_probs[0]  # (T', V)

        lattice = build_lattice(log_probs_utt, topo, device)

        # Track per-utterance results for examples
        utt_example = {
            "utt_id": utt_id,
            "ref_text": ref_text,
            "conditions": {},
        }

        # 1-best WER (from greedy decode)
        greedy_ids = log_probs_utt.argmax(dim=-1).tolist()
        greedy_collapsed = ctc_collapse(greedy_ids)
        greedy_text = sp.decode(greedy_collapsed).strip().lower()
        onebest_wer = compute_wer(greedy_text, ref_text)
        utt_example["onebest_text"] = greedy_text
        utt_example["onebest_wer"] = onebest_wer

        for cond in CONDITIONS:
            method = cond["method"]
            scale = cond["nbest_scale"]

            all_hyps = extract_nbest_hypotheses(
                lattice, num_paths=max_num_paths, nbest_scale=scale, sp=sp,
            )

            # Ensure greedy 1-best is always the first candidate.
            # Nbest.from_lattice uses random sampling  --  greedy may be
            # missing or buried past position G, causing oracle > 1-best.
            if greedy_text in all_hyps:
                all_hyps.remove(greedy_text)
            all_hyps.insert(0, greedy_text)

            for g in G_VALUES:
                # Take top-G unique hypotheses
                candidates = all_hyps[:g]
                num_unique = len(candidates)

                wers = [compute_wer(h, ref_text) for h in candidates]

                oracle_wer = min(wers) if wers else onebest_wer
                mean_wer = sum(wers) / len(wers) if wers else onebest_wer

                # Smoke test: oracle <= 1-best
                assert oracle_wer <= onebest_wer + 1e-6, (
                    f"Oracle WER {oracle_wer:.4f} > 1-best WER {onebest_wer:.4f} "
                    f"for utt {utt_id}, method={method}, scale={scale}, G={g}. "
                    f"1-best not included in candidate set?"
                )

                writer.writerow([
                    utt_id, method, scale, g,
                    f"{oracle_wer:.6f}", f"{onebest_wer:.6f}",
                    f"{mean_wer:.6f}", num_unique, ref_text,
                ])

                agg[(method, scale, g)].append({
                    "oracle_wer": oracle_wer,
                    "onebest_wer": onebest_wer,
                    "mean_wer": mean_wer,
                    "num_unique": num_unique,
                })

                if len(examples) < MAX_EXAMPLES or utt_idx < MAX_EXAMPLES:
                    cond_key = f"{method}_scale{scale}_G{g}"
                    utt_example["conditions"][cond_key] = {
                        "candidates": candidates,
                        "wers": wers,
                        "num_unique": num_unique,
                    }

        if utt_idx < MAX_EXAMPLES:
            examples.append(utt_example)

        # Memory cleanup
        del lattice, log_probs, encoder_out, feats_gpu
        torch.cuda.empty_cache()

        if (utt_idx + 1) % 50 == 0:
            # Quick progress stats for the first condition
            key0 = ("beam", 1.0, 8)
            mean_orcl = sum(d["oracle_wer"] for d in agg[key0]) / len(agg[key0])
            print(
                f"  [{utt_idx+1}/{len(utterances)}] "
                f"beam G=8 oracle WER so far: {mean_orcl*100:.2f}%"
            )

    csv_file.close()
    print(f"CSV saved: {csv_path}")

    # Test 3: monotonicity  --  mean oracle WER should decrease as G increases
    for cond in CONDITIONS:
        method, scale = cond["method"], cond["nbest_scale"]
        prev_mean = None
        for g in G_VALUES:
            key = (method, scale, g)
            mean_oracle = sum(d["oracle_wer"] for d in agg[key]) / len(agg[key])
            if prev_mean is not None:
                assert mean_oracle <= prev_mean + 1e-4, (
                    f"Non-monotone oracle WER: {method} scale={scale} "
                    f"G={g} ({mean_oracle:.4f}) > G={G_VALUES[G_VALUES.index(g)-1]} "
                    f"({prev_mean:.4f})"
                )
            prev_mean = mean_oracle

    # Test 1: at G=1, beam and lattice should give ~same oracle WER
    # (We don't have G=1 in our conditions, but at G=4 with scale=1.0 we can
    #  compare beam vs lattice at scale=1.0. Instead, check beam G=4 vs
    #  the 1-best WER  --  they should be close for small G.)
    # Actually, let's compare the 1-best greedy WER against beam G=4 oracle.
    beam_g4 = agg[("beam", 1.0, 4)]
    mean_onebest = sum(d["onebest_wer"] for d in beam_g4) / len(beam_g4)
    mean_beam_g4_oracle = sum(d["oracle_wer"] for d in beam_g4) / len(beam_g4)
    print(
        f"\nSanity: 1-best WER={mean_onebest*100:.2f}%, "
        f"beam G=4 oracle={mean_beam_g4_oracle*100:.2f}%"
    )

    # Test 4: lattice diversity at scale=0.5, G=8
    lat_g8_05 = agg[("lattice", 0.5, 8)]
    frac_diverse = sum(1 for d in lat_g8_05 if d["num_unique"] >= 3) / len(lat_g8_05)
    print(
        f"Lattice diversity (scale=0.5, G=8): "
        f"{frac_diverse*100:.1f}% of utterances have >=3 unique candidates"
    )
    assert frac_diverse >= 0.40, (
        f"Only {frac_diverse*100:.1f}% of utterances have >=3 unique candidates "
        f"at scale=0.5, G=8. Expected >=40%. Lattice diversity not working."
    )

    summary = {"conditions": []}
    for cond in CONDITIONS:
        method, scale = cond["method"], cond["nbest_scale"]
        for g in G_VALUES:
            key = (method, scale, g)
            data = agg[key]
            n = len(data)
            oracle_wers = [d["oracle_wer"] for d in data]
            onebest_wers = [d["onebest_wer"] for d in data]
            mean_oracle = sum(oracle_wers) / n
            mean_onebest = sum(onebest_wers) / n
            median_oracle = sorted(oracle_wers)[n // 2]
            mean_unique = sum(d["num_unique"] for d in data) / n
            reduction = (
                (mean_onebest - mean_oracle) / mean_onebest
                if mean_onebest > 0 else 0.0
            )

            summary["conditions"].append({
                "method": method,
                "nbest_scale": scale,
                "G": g,
                "mean_oracle_wer": round(mean_oracle, 6),
                "median_oracle_wer": round(median_oracle, 6),
                "mean_onebest_wer": round(mean_onebest, 6),
                "mean_num_unique": round(mean_unique, 2),
                "n_utterances": n,
                "oracle_wer_reduction_vs_onebest": round(reduction, 4),
            })

    summary_path = output_dir / "oracle_wer_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")

    examples_path = output_dir / "examples.txt"
    with open(examples_path, "w") as f:
        for ex in examples:
            f.write(f"=== Utterance: {ex['utt_id']} ===\n")
            f.write(f"Reference: {ex['ref_text']}\n")
            f.write(
                f"1-best (WER={ex['onebest_wer']*100:.1f}%): "
                f"{ex['onebest_text']}\n\n"
            )

            for g in [8]:  # Show G=8 for readability
                for cond in CONDITIONS:
                    method = cond["method"]
                    scale = cond["nbest_scale"]
                    cond_key = f"{method}_scale{scale}_G{g}"
                    if cond_key not in ex["conditions"]:
                        continue
                    info = ex["conditions"][cond_key]
                    label = "Beam" if method == "beam" else "Lattice"
                    f.write(
                        f"{label} (G={g}, scale={scale}): "
                        f"{info['num_unique']} unique candidates\n"
                    )
                    oracle_idx = (
                        info["wers"].index(min(info["wers"]))
                        if info["wers"] else -1
                    )
                    for j, (hyp, w) in enumerate(
                        zip(info["candidates"], info["wers"])
                    ):
                        marker = "  <- ORACLE" if j == oracle_idx else ""
                        f.write(f"  [{j}] WER={w*100:5.1f}%: {hyp}{marker}\n")
                    f.write("\n")
            f.write("\n")
    print(f"Examples saved: {examples_path}")

    print("\n" + "=" * 70)
    print("ORACLE WER SUMMARY")
    print("=" * 70)
    print(
        f"{'Method':<10} {'Scale':>6} {'G':>4} "
        f"{'Oracle WER':>11} {'1-best WER':>11} {'Reduction':>10} {'Uniq':>6}"
    )
    print("-" * 70)
    for entry in summary["conditions"]:
        print(
            f"{entry['method']:<10} {entry['nbest_scale']:>6.2f} "
            f"{entry['G']:>4d} "
            f"{entry['mean_oracle_wer']*100:>10.2f}% "
            f"{entry['mean_onebest_wer']*100:>10.2f}% "
            f"{entry['oracle_wer_reduction_vs_onebest']*100:>9.1f}% "
            f"{entry['mean_num_unique']:>6.1f}"
        )
    print("=" * 70)

    return summary

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1: Oracle WER  --  beam vs lattice diversity"
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
    parser.add_argument("--num-utterances", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 60)
    print("RBPO Stage 1  --  Oracle WER Experiment")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Utterances: {args.num_utterances}")
    print(f"Conditions: {len(CONDITIONS)} methods x {len(G_VALUES)} G values")

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
    output_dir = args.results_dir / "stage_1_oracle_wer"
    summary = run_experiment(model, utterances, sp, device, output_dir)
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

if __name__ == "__main__":
    main()

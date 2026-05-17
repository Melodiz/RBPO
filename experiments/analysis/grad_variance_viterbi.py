#!/usr/bin/env python3
"""Stage 3b: Gradient variance  --  CTC-marginalized vs Viterbi vs Sampled alignment.

The genuine Rao-Blackwell test. Stage 3 showed that the "flat" MWER loss and
the explicit gamma-weighted loss produce identical gradients on the CTC projection
because CTC backward already implements RB internally. The actual "naive"
baseline is the Viterbi (or sampled) one-hot alignment estimator from Eq. 7
of the RBPO paper.

For each candidate y_i, computes three gradient estimates on the CTC output
projection:
  - CTC-marginalized: grad log P_CTC(y_i|x)  --  averages over all alignments via gamma
  - Viterbi:          grad sum_t log P(pi*_t | x_t)  --  best alignment, one-hot credit
  - Sampled:          grad sum_t log P(pi^_t | x_t)  --  random alignment ~ posterior

Predicted: Var(Sampled) >= Var(Viterbi) >= Var(CTC), with strict inequality
when the alignment posterior is not concentrated on a single path.

Usage:
    python experiments/grad_variance_viterbi.py \\
        --model-dir /content/icefall-asr-librispeech-zipformer-small-cr-ctc \\
        --icefall-dir /content/icefall \\
        --data-dir /content/librispeech_data \\
        --results-dir /content/drive/MyDrive/rbpo_results \\
        --num-utterances 50 --G 8 --device cuda:0
"""

import argparse
import csv
import json
import sys
import time
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
VOCAB_SIZE = 500

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

def find_ctc_projection(model) -> torch.nn.Linear:
    for attr in ["ctc_output_module", "ctc_output_proj"]:
        if hasattr(model, attr):
            module = getattr(model, attr)
            for m in module.modules():
                if isinstance(m, torch.nn.Linear) and m.out_features == VOCAB_SIZE:
                    return m
    candidates = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == VOCAB_SIZE:
            candidates.append((name, m))
    assert candidates, f"No Linear(*, {VOCAB_SIZE}) found in model"
    name, layer = candidates[-1]
    print(f"  Target layer (fallback): {name}")
    return layer

def build_outer_lattice(log_probs: torch.Tensor, topo, device: torch.device):
    """Build the full CTC lattice (over the topology) for sampling candidates."""
    import k2

    T = log_probs.shape[0]
    supervision_segments = torch.tensor([[0, 0, T]], dtype=torch.int32)
    dense_fsa = k2.DenseFsaVec(log_probs.unsqueeze(0), supervision_segments)
    lattice = k2.intersect_dense(topo, dense_fsa, output_beam=8.0)
    lattice = k2.connect(lattice)
    return lattice

def extract_nbest_with_tokens(
    lattice, num_paths: int, nbest_scale: float, sp: spm.SentencePieceProcessor
) -> list[tuple[str, list[int]]]:
    """Extract N-best (text, token_ids) pairs from outer lattice."""
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

    results = []
    seen_texts = set()
    for raw_ids in paths_labels:
        token_ids = ctc_collapse(raw_ids)
        if not token_ids:
            continue
        text = sp.decode(token_ids).strip().lower()
        if text not in seen_texts:
            seen_texts.add(text)
            results.append((text, token_ids))
    return results

def build_numerator_lattice(
    log_probs: torch.Tensor,
    token_ids: list[int],
    T: int,
    device: torch.device,
):
    """Build the numerator lattice  --  paths through CTC topology constrained
    to produce exactly the token sequence y_i. Returns the lattice."""
    import k2

    supervision_segments = torch.tensor([[0, 0, T]], dtype=torch.int32)
    dense_fsa = k2.DenseFsaVec(log_probs.unsqueeze(0), supervision_segments)
    ctc_graph = k2.ctc_graph([token_ids], modified=False, device=device)
    lattice = k2.intersect_dense(ctc_graph, dense_fsa, output_beam=10.0)
    return lattice

def viterbi_alignment(lattice, T: int) -> list[int]:
    """Extract the best (Viterbi) alignment path through the numerator lattice.

    Returns a list of T frame labels (each in {blank=0} union token_ids).
    """
    import k2

    best = k2.shortest_path(lattice, use_double_scores=True)
    labels = best.labels.cpu().tolist()
    alignment = [l for l in labels if l != -1]
    assert len(alignment) == T, (
        f"Viterbi path has {len(alignment)} frames, expected {T}"
    )
    return alignment

def sampled_alignment(lattice, T: int) -> list[int]:
    """Sample one alignment path proportional to its posterior probability."""
    import k2

    nbest = k2.Nbest.from_lattice(
        lattice,
        num_paths=1,
        use_double_scores=True,
        nbest_scale=1.0,
    )
    labels = nbest.fsa.labels.cpu().tolist()
    alignment = [l for l in labels if l != -1]
    assert len(alignment) == T, (
        f"Sampled path has {len(alignment)} frames, expected {T}"
    )
    return alignment

def compute_ctc_gradient(
    model,
    encoder_out: torch.Tensor,
    token_ids: list[int],
    T: int,
    advantage: float,
    target_layer: torch.nn.Linear,
    device: torch.device,
) -> torch.Tensor:
    """CTC-marginalized: loss = -A_hat * log P_CTC(y|x) via k2."""
    import k2

    model.zero_grad()
    log_probs = model.ctc_output(encoder_out)
    log_probs_T = log_probs[:, :T, :]

    supervision = torch.tensor([[0, 0, T]], dtype=torch.int32)
    dense_fsa = k2.DenseFsaVec(log_probs_T, supervision)
    ctc_graph = k2.ctc_graph([token_ids], modified=False, device=device)
    lattice = k2.intersect_dense(ctc_graph, dense_fsa, output_beam=10.0)
    log_p = lattice.get_tot_scores(
        log_semiring=True, use_double_scores=True
    )

    loss = -advantage * log_p
    loss.backward()
    return target_layer.weight.grad.clone()

def compute_one_hot_gradient(
    model,
    encoder_out: torch.Tensor,
    alignment: list[int],
    T: int,
    advantage: float,
    target_layer: torch.nn.Linear,
    device: torch.device,
) -> torch.Tensor:
    """One-hot alignment gradient: loss = -A_hat * sum_t log P(pi_t | x_t)."""
    model.zero_grad()
    log_probs = model.ctc_output(encoder_out)  # (1, T', V)

    indices = torch.tensor(alignment, device=device, dtype=torch.long)  # (T,)
    frame_ids = torch.arange(T, device=device)
    lp_path = log_probs[0, frame_ids, indices]  # (T,)

    loss = -advantage * lp_path.sum()
    loss.backward()
    return target_layer.weight.grad.clone()

def load_utterances(data_dir: Path, split: str, num_utterances: int):
    from lhotse import load_manifest_lazy

    cuts_path = data_dir / "cuts" / f"librispeech_cuts_{split}.jsonl.gz"
    assert cuts_path.exists(), f"CutSet not found: {cuts_path}"

    cuts = load_manifest_lazy(str(cuts_path))
    utterances = []
    for cut in cuts:
        if len(utterances) >= num_utterances:
            break
        feats = torch.from_numpy(cut.load_features())
        ref_text = " ".join(
            s.text for s in cut.supervisions if s.text
        ).strip().lower()
        if not ref_text:
            continue
        utterances.append((cut.id, feats, ref_text))

    print(f"Loaded {len(utterances)} utterances from {split}")
    return utterances

def run_experiment(
    model,
    utterances: list,
    sp: spm.SentencePieceProcessor,
    device: torch.device,
    output_dir: Path,
    G: int = 8,
):
    import k2

    topo = k2.ctc_topo(max_token=MAX_TOKEN, modified=False, device=device)
    target_layer = find_ctc_projection(model)
    print(
        f"Target layer: Linear({target_layer.in_features}, "
        f"{target_layer.out_features})  --  "
        f"{target_layer.weight.numel()} parameters"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "viterbi_variance_results.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "utt_id", "G_effective",
        "mean_var_ctc", "mean_var_viterbi", "mean_var_sampled",
        "ratio_viterbi_vs_ctc", "ratio_sampled_vs_ctc",
        "T", "L",
    ])

    all_results = []
    n_skipped = 0
    examples = []
    rb_worse_viterbi = 0
    rb_worse_sampled = 0

    for utt_idx, (utt_id, feats, ref_text) in enumerate(
        tqdm(utterances, desc="Viterbi variance")
    ):
        feats_gpu = feats.unsqueeze(0).to(device)
        feat_lens = torch.tensor(
            [feats.shape[0]], dtype=torch.int64, device=device
        )

        with torch.no_grad():
            encoder_out, encoder_out_lens = model.forward_encoder(
                feats_gpu, feat_lens
            )
            log_probs_init = model.ctc_output(encoder_out)

        T = encoder_out_lens[0].item()
        log_probs_utt = log_probs_init[0, :T]

        # Greedy 1-best
        greedy_ids = log_probs_utt.argmax(dim=-1).tolist()
        greedy_tokens = ctc_collapse(greedy_ids)
        greedy_text = (
            sp.decode(greedy_tokens).strip().lower() if greedy_tokens else ""
        )

        outer_lattice = build_outer_lattice(log_probs_utt, topo, device)
        candidates = extract_nbest_with_tokens(
            outer_lattice, num_paths=G * 4, nbest_scale=1.0, sp=sp
        )

        if greedy_tokens:
            candidates = [
                (t, ids) for t, ids in candidates if t != greedy_text
            ]
            candidates.insert(0, (greedy_text, greedy_tokens))
        candidates = candidates[:G]

        if len(candidates) < 2:
            n_skipped += 1
            del outer_lattice, log_probs_init, encoder_out, feats_gpu
            torch.cuda.empty_cache()
            continue

        wers = [compute_wer(text, ref_text) for text, _ in candidates]
        rewards = [-w for w in wers]
        mean_reward = sum(rewards) / len(rewards)
        advantages = [r - mean_reward for r in rewards]

        if len(set(wers)) <= 1:
            n_skipped += 1
            del outer_lattice, log_probs_init, encoder_out, feats_gpu
            torch.cuda.empty_cache()
            continue

        L = len(candidates[0][1])

        encoder_out_det = encoder_out.detach()
        log_probs_det = log_probs_init[:, :T, :].detach()

        grad_ctc_list = []
        grad_viterbi_list = []
        grad_sampled_list = []

        viterbi_examples = []  # for the first utterance only

        for i, ((_, token_ids_i), adv_i) in enumerate(
            zip(candidates, advantages)
        ):
            if not token_ids_i:
                z = torch.zeros_like(target_layer.weight)
                grad_ctc_list.append(z)
                grad_viterbi_list.append(z)
                grad_sampled_list.append(z)
                continue

            # Build numerator lattice (shared between Viterbi and Sampled)
            num_lattice = build_numerator_lattice(
                log_probs_det[0], token_ids_i, T, device
            )

            vit_align = viterbi_alignment(num_lattice, T)
            samp_align = sampled_alignment(num_lattice, T)

            # Smoke test: CTC-collapse of Viterbi alignment must equal token_ids
            vit_collapsed = ctc_collapse(vit_align)
            assert vit_collapsed == token_ids_i, (
                f"Viterbi path mismatch: collapse({vit_align[:20]}...) = "
                f"{vit_collapsed[:10]} vs expected {token_ids_i[:10]}"
            )

            # CTC-marginalized gradient
            g_ctc = compute_ctc_gradient(
                model, encoder_out_det, token_ids_i, T,
                adv_i, target_layer, device,
            )
            grad_ctc_list.append(g_ctc)

            # Viterbi gradient
            g_vit = compute_one_hot_gradient(
                model, encoder_out_det, vit_align, T,
                adv_i, target_layer, device,
            )
            grad_viterbi_list.append(g_vit)

            # Sampled gradient
            g_samp = compute_one_hot_gradient(
                model, encoder_out_det, samp_align, T,
                adv_i, target_layer, device,
            )
            grad_sampled_list.append(g_samp)

            if len(examples) == 0 and i < 3:
                viterbi_examples.append({
                    "candidate_idx": i,
                    "token_ids": token_ids_i[:20],
                    "viterbi_align_first50": vit_align[:50],
                    "sampled_align_first50": samp_align[:50],
                    "blank_post_first50": [
                        log_probs_utt[t, BLANK_ID].exp().item()
                        for t in range(min(50, T))
                    ],
                    "advantage": adv_i,
                    "grad_norm_ctc": g_ctc.norm().item(),
                    "grad_norm_viterbi": g_vit.norm().item(),
                    "grad_norm_sampled": g_samp.norm().item(),
                })

            del num_lattice

        # Stack & compute variance
        grads_ctc = torch.stack(grad_ctc_list)
        grads_vit = torch.stack(grad_viterbi_list)
        grads_samp = torch.stack(grad_sampled_list)

        var_ctc = grads_ctc.var(dim=0)
        var_vit = grads_vit.var(dim=0)
        var_samp = grads_samp.var(dim=0)

        mean_var_ctc = var_ctc.mean().item()
        mean_var_vit = var_vit.mean().item()
        mean_var_samp = var_samp.mean().item()

        ratio_vit = mean_var_vit / (mean_var_ctc + 1e-20)
        ratio_samp = mean_var_samp / (mean_var_ctc + 1e-20)

        # Smoke tests
        for g in grad_ctc_list + grad_viterbi_list + grad_sampled_list:
            assert g.isfinite().all(), "Non-finite gradient detected"

        if ratio_vit < 0.95:
            rb_worse_viterbi += 1
        if ratio_samp < 0.95:
            rb_worse_sampled += 1

        G_eff = len(candidates)
        result = {
            "utt_id": utt_id,
            "G_effective": G_eff,
            "mean_var_ctc": mean_var_ctc,
            "mean_var_viterbi": mean_var_vit,
            "mean_var_sampled": mean_var_samp,
            "ratio_viterbi_vs_ctc": ratio_vit,
            "ratio_sampled_vs_ctc": ratio_samp,
            "T": T,
            "L": L,
        }
        all_results.append(result)

        writer.writerow([
            utt_id, G_eff,
            f"{mean_var_ctc:.6e}", f"{mean_var_vit:.6e}",
            f"{mean_var_samp:.6e}",
            f"{ratio_vit:.6f}", f"{ratio_samp:.6f}",
            T, L,
        ])

        if len(examples) < 3:
            examples.append({
                "utt_id": utt_id,
                "ref_text": ref_text,
                "candidates": [(t, w) for (t, _), w in zip(candidates, wers)],
                "advantages": advantages,
                "mean_var_ctc": mean_var_ctc,
                "mean_var_viterbi": mean_var_vit,
                "mean_var_sampled": mean_var_samp,
                "ratio_viterbi_vs_ctc": ratio_vit,
                "ratio_sampled_vs_ctc": ratio_samp,
                "viterbi_details": viterbi_examples,
            })

        # Cleanup
        del (grads_ctc, grads_vit, grads_samp,
             var_ctc, var_vit, var_samp,
             grad_ctc_list, grad_viterbi_list, grad_sampled_list,
             outer_lattice, log_probs_init, log_probs_det,
             encoder_out, feats_gpu, encoder_out_det)
        torch.cuda.empty_cache()

        if (utt_idx + 1) % 10 == 0:
            ratios_v = [r["ratio_viterbi_vs_ctc"] for r in all_results]
            ratios_s = [r["ratio_sampled_vs_ctc"] for r in all_results]
            print(
                f"  [{utt_idx+1}/{len(utterances)}] "
                f"viterbi/ctc: {np.mean(ratios_v):.3f}, "
                f"sampled/ctc: {np.mean(ratios_s):.3f}"
            )

    csv_file.close()
    print(f"CSV saved: {csv_path}")

    if not all_results:
        print("ERROR: No valid utterances processed!")
        return None

    ratios_v = [r["ratio_viterbi_vs_ctc"] for r in all_results]
    ratios_s = [r["ratio_sampled_vs_ctc"] for r in all_results]

    summary = {
        "n_utterances": len(all_results),
        "n_skipped": n_skipped,
        "G": G,
        "mean_ratio_viterbi_vs_ctc": float(np.mean(ratios_v)),
        "median_ratio_viterbi_vs_ctc": float(np.median(ratios_v)),
        "mean_ratio_sampled_vs_ctc": float(np.mean(ratios_s)),
        "median_ratio_sampled_vs_ctc": float(np.median(ratios_s)),
        "min_ratio_viterbi": float(np.min(ratios_v)),
        "max_ratio_viterbi": float(np.max(ratios_v)),
        "min_ratio_sampled": float(np.min(ratios_s)),
        "max_ratio_sampled": float(np.max(ratios_s)),
        "rb_worse_viterbi_count": rb_worse_viterbi,
        "rb_worse_sampled_count": rb_worse_sampled,
        "target_layer_shape": list(target_layer.weight.shape),
    }

    summary_path = output_dir / "viterbi_variance_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")

    examples_path = output_dir / "examples.txt"
    with open(examples_path, "w") as f:
        for ex in examples:
            f.write(f"=== Utterance: {ex['utt_id']} ===\n")
            f.write(f"Reference: {ex['ref_text']}\n\n")
            f.write("Candidates:\n")
            for j, ((text, wer), adv) in enumerate(
                zip(ex["candidates"], ex["advantages"])
            ):
                f.write(
                    f"  [{j}] WER={wer*100:5.1f}% A_hat={adv:+.4f}: {text}\n"
                )
            f.write(f"\nMean variance:\n")
            f.write(f"  CTC-marginalized: {ex['mean_var_ctc']:.4e}\n")
            f.write(f"  Viterbi (1-hot):  {ex['mean_var_viterbi']:.4e}\n")
            f.write(f"  Sampled (1-hot):  {ex['mean_var_sampled']:.4e}\n")
            f.write(f"\nRatios:\n")
            f.write(f"  Viterbi / CTC: {ex['ratio_viterbi_vs_ctc']:.4f}\n")
            f.write(f"  Sampled / CTC: {ex['ratio_sampled_vs_ctc']:.4f}\n")

            if ex["viterbi_details"]:
                f.write("\nAlignment paths (first 50 frames):\n")
                for d in ex["viterbi_details"]:
                    f.write(
                        f"\n  Candidate [{d['candidate_idx']}] "
                        f"A_hat={d['advantage']:+.4f}\n"
                    )
                    f.write(f"    token_ids[:20]: {d['token_ids']}\n")
                    f.write(
                        f"    Viterbi[:50]:   {d['viterbi_align_first50']}\n"
                    )
                    f.write(
                        f"    Sampled[:50]:   {d['sampled_align_first50']}\n"
                    )
                    f.write(
                        f"    blank_P[:50]:   "
                        f"{[round(p, 2) for p in d['blank_post_first50']]}\n"
                    )
                    f.write(
                        f"    grad norms: ctc={d['grad_norm_ctc']:.4e} "
                        f"viterbi={d['grad_norm_viterbi']:.4e} "
                        f"sampled={d['grad_norm_sampled']:.4e}\n"
                    )
            f.write("\n\n")
    print(f"Examples saved: {examples_path}")

    print("\n" + "=" * 70)
    print("STAGE 3b SUMMARY: Viterbi vs CTC Variance Reduction")
    print("=" * 70)
    print(f"  Utterances analyzed:            {summary['n_utterances']}")
    print(f"  Skipped:                        {summary['n_skipped']}")
    print(f"  Group size G:                   {G}")
    print(f"  Target layer:                   Linear{tuple(summary['target_layer_shape'])}")
    print()
    print(f"  Viterbi/CTC ratio mean:         {summary['mean_ratio_viterbi_vs_ctc']:.4f}")
    print(f"  Viterbi/CTC ratio median:       {summary['median_ratio_viterbi_vs_ctc']:.4f}")
    print(f"  Viterbi/CTC min/max:            {summary['min_ratio_viterbi']:.4f} / {summary['max_ratio_viterbi']:.4f}")
    print()
    print(f"  Sampled/CTC ratio mean:         {summary['mean_ratio_sampled_vs_ctc']:.4f}")
    print(f"  Sampled/CTC ratio median:       {summary['median_ratio_sampled_vs_ctc']:.4f}")
    print(f"  Sampled/CTC min/max:            {summary['min_ratio_sampled']:.4f} / {summary['max_ratio_sampled']:.4f}")
    print()
    print(f"  RB-worse (Viterbi, ratio<0.95): {summary['rb_worse_viterbi_count']}")
    print(f"  RB-worse (Sampled, ratio<0.95): {summary['rb_worse_sampled_count']}")
    print("=" * 70)

    return summary

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 3b: Viterbi vs CTC-marginalized gradient variance"
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
    parser.add_argument("--num-utterances", type=int, default=50)
    parser.add_argument("--G", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 60)
    print("RBPO Stage 3b  --  Viterbi vs CTC Gradient Variance")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Utterances: {args.num_utterances}")
    print(f"Group size G: {args.G}")

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
    output_dir = args.results_dir / "stage_3b_viterbi_variance"
    summary = run_experiment(model, utterances, sp, device, output_dir, args.G)
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

if __name__ == "__main__":
    main()

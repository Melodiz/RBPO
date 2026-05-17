#!/usr/bin/env python3
"""Stage 3: Gradient variance  --  Rao-Blackwellized vs flat credit assignment.

Measures per-parameter gradient variance across G candidates for two estimators:
  - Flat: g_hat_i = A_hat_i * grad_theta log P_CTC(y_i|x) via k2 forward-backward
  - RB:   g_hat_i = A_hat_i * sum_t sum_k gamma_t(k|y_i) * grad_theta log P(k|x_t) via gamma-weighted sum

Both produce the same expected gradient (Proposition 4.1) but may differ in variance.
The Rao-Blackwell theorem predicts Var(RB) <= Var(flat).

Target layer: CTC output projection (last Linear before LogSoftmax).

Usage:
    python experiments/grad_variance.py \
        --model-dir /content/icefall-asr-librispeech-zipformer-small-cr-ctc \
        --data-dir /content/librispeech_data \
        --results-dir /content/drive/MyDrive/rbpo_results \
        --num-utterances 50 \
        --G 8 \
        --device cuda:0
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
    """Find the CTC output projection Linear layer."""
    # Try icefall's common attribute names
    for attr in ["ctc_output_module", "ctc_output_proj"]:
        if hasattr(model, attr):
            module = getattr(model, attr)
            for m in module.modules():
                if isinstance(m, torch.nn.Linear) and m.out_features == VOCAB_SIZE:
                    return m

    # Fallback: search all modules for last Linear with out_features=500
    candidates = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == VOCAB_SIZE:
            candidates.append((name, m))
    assert candidates, f"No Linear(*, {VOCAB_SIZE}) found in model"
    name, layer = candidates[-1]
    print(f"  Target layer (fallback): {name}")
    return layer

def build_lattice(log_probs: torch.Tensor, topo, device: torch.device):
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
    """Extract N-best hypotheses with token IDs."""
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

def extract_gamma(
    log_probs: torch.Tensor,
    token_ids: list[int],
    T: int,
    device: torch.device,
) -> torch.Tensor:
    """Extract gamma_t(k|y) via autograd. Returns (T, V) detached tensor."""
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
    gamma = lp_gamma.grad.squeeze(0)[:T]  # (T, V)
    return gamma.detach()

def compute_flat_gradient(
    model,
    encoder_out: torch.Tensor,
    token_ids: list[int],
    T: int,
    advantage: float,
    target_layer: torch.nn.Linear,
    device: torch.device,
) -> torch.Tensor:
    """Compute flat (MWER-style) gradient via k2 CTC forward-backward.

    Loss = -A_hat_i * log P_CTC(y_i|x)
    """
    import k2

    model.zero_grad()

    log_probs = model.ctc_output(encoder_out)  # (1, T', V)
    log_probs_T = log_probs[:, :T, :]  # (1, T, V)

    supervision_segments = torch.tensor([[0, 0, T]], dtype=torch.int32)
    dense_fsa = k2.DenseFsaVec(log_probs_T, supervision_segments)
    ctc_graph = k2.ctc_graph([token_ids], modified=False, device=device)
    lattice = k2.intersect_dense(ctc_graph, dense_fsa, output_beam=10.0)
    log_p = lattice.get_tot_scores(
        log_semiring=True, use_double_scores=True
    )

    loss = -advantage * log_p
    loss.backward()

    grad = target_layer.weight.grad.clone()
    return grad

def compute_rb_gradient(
    model,
    encoder_out: torch.Tensor,
    token_ids: list[int],
    T: int,
    advantage: float,
    target_layer: torch.nn.Linear,
    device: torch.device,
) -> torch.Tensor:
    """Compute RB (Rao-Blackwellized) gradient via gamma-weighted frame loss.

    Loss = -A_hat_i * sum_t sum_k gamma_t(k|y_i) * log P(k|x_t)
    """
    model.zero_grad()

    log_probs = model.ctc_output(encoder_out)  # (1, T', V)

    # Extract gamma_t from a detached copy (separate computation graph)
    gamma = extract_gamma(
        log_probs[:, :T, :].detach(), token_ids, T, device
    )  # (T, V), detached

    # gamma-weighted cross-entropy using live log_probs
    loss = -advantage * (gamma * log_probs[0, :T]).sum()
    loss.backward()

    grad = target_layer.weight.grad.clone()
    return grad

def verify_gradient_equivalence(
    grad_flat_list: list[torch.Tensor],
    grad_rb_list: list[torch.Tensor],
    advantages: list[float],
) -> float:
    """Verify that advantage-weighted mean gradients match (Prop 4.1).

    Under centered baseline (sum A_hat_i = 0), the advantage-weighted
    sum of gradients should be approximately equal for both estimators.
    """
    weighted_flat = sum(
        a * g for a, g in zip(advantages, grad_flat_list)
    )
    weighted_rb = sum(
        a * g for a, g in zip(advantages, grad_rb_list)
    )

    diff = (weighted_flat - weighted_rb).abs()
    mean_diff = diff.mean().item()

    scale = max(
        weighted_flat.abs().mean().item(),
        weighted_rb.abs().mean().item(),
        1e-10,
    )
    relative_diff = mean_diff / scale

    assert relative_diff < 0.1, (
        f"Prop 4.1 violation: relative gradient diff = {relative_diff:.6f} "
        f"(mean abs diff = {mean_diff:.2e}, scale = {scale:.2e})"
    )

    return relative_diff

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

    csv_path = output_dir / "grad_variance_results.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "utt_id", "G_effective", "num_unique_wer", "mean_var_flat",
        "mean_var_rb", "variance_ratio", "prop41_relative_diff",
        "T", "L", "mean_advantage_magnitude",
    ])

    all_results = []
    n_skipped = 0
    prop41_diffs = []
    examples = []

    for utt_idx, (utt_id, feats, ref_text) in enumerate(
        tqdm(utterances, desc="Gradient variance")
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
        log_probs_utt = log_probs_init[0, :T]  # (T, V)

        # Greedy 1-best
        greedy_ids = log_probs_utt.argmax(dim=-1).tolist()
        greedy_tokens = ctc_collapse(greedy_ids)
        greedy_text = sp.decode(greedy_tokens).strip().lower() if greedy_tokens else ""

        lattice = build_lattice(log_probs_utt, topo, device)
        candidates = extract_nbest_with_tokens(
            lattice, num_paths=G * 4, nbest_scale=1.0, sp=sp
        )

        # Ensure greedy 1-best is first
        if greedy_tokens:
            candidates = [
                (t, ids) for t, ids in candidates if t != greedy_text
            ]
            candidates.insert(0, (greedy_text, greedy_tokens))
        candidates = candidates[:G]

        if len(candidates) < 2:
            n_skipped += 1
            del lattice, log_probs_init, encoder_out, feats_gpu
            torch.cuda.empty_cache()
            continue

        # Compute WER rewards and advantages (Dr. GRPO)
        wers = [compute_wer(text, ref_text) for text, _ in candidates]
        rewards = [-w for w in wers]
        mean_reward = sum(rewards) / len(rewards)
        advantages = [r - mean_reward for r in rewards]

        # Skip if all WERs identical (zero advantage -> zero gradient)
        unique_wers = len(set(wers))
        if unique_wers <= 1:
            n_skipped += 1
            del lattice, log_probs_init, encoder_out, feats_gpu
            torch.cuda.empty_cache()
            continue

        L = len(candidates[0][1])  # token length of 1-best

        # encoder_out is detached from encoder but CTC linear still has grad
        encoder_out_det = encoder_out.detach()

        grad_flat_list = []
        grad_rb_list = []

        for i, ((_, token_ids_i), adv_i) in enumerate(
            zip(candidates, advantages)
        ):
            if not token_ids_i:
                # Empty candidate  --  use zero gradient
                zero_g = torch.zeros_like(target_layer.weight)
                grad_flat_list.append(zero_g)
                grad_rb_list.append(zero_g)
                continue

            # Flat gradient
            g_flat = compute_flat_gradient(
                model, encoder_out_det, token_ids_i, T,
                adv_i, target_layer, device,
            )
            grad_flat_list.append(g_flat)

            # RB gradient
            g_rb = compute_rb_gradient(
                model, encoder_out_det, token_ids_i, T,
                adv_i, target_layer, device,
            )
            grad_rb_list.append(g_rb)

        grads_flat = torch.stack(grad_flat_list)  # (G, D, V)
        grads_rb = torch.stack(grad_rb_list)      # (G, D, V)

        var_flat = grads_flat.var(dim=0)  # (D, V)
        var_rb = grads_rb.var(dim=0)      # (D, V)

        mean_var_flat = var_flat.mean().item()
        mean_var_rb = var_rb.mean().item()
        variance_ratio = mean_var_flat / (mean_var_rb + 1e-20)

        prop41_diff = verify_gradient_equivalence(
            grad_flat_list, grad_rb_list, advantages
        )
        prop41_diffs.append(prop41_diff)

        for g in grad_flat_list + grad_rb_list:
            assert g.isfinite().all(), "Non-finite gradient detected"

        adv_sum = abs(sum(advantages))
        assert adv_sum < 1e-6, f"Advantages don't sum to 0: sum={adv_sum}"

        # Record results
        G_eff = len(candidates)
        mean_adv_mag = sum(abs(a) for a in advantages) / len(advantages)

        result = {
            "utt_id": utt_id,
            "G_effective": G_eff,
            "num_unique_wer": unique_wers,
            "mean_var_flat": mean_var_flat,
            "mean_var_rb": mean_var_rb,
            "variance_ratio": variance_ratio,
            "prop41_relative_diff": prop41_diff,
            "T": T,
            "L": L,
            "mean_advantage_magnitude": mean_adv_mag,
        }
        all_results.append(result)

        writer.writerow([
            utt_id, G_eff, unique_wers,
            f"{mean_var_flat:.2e}", f"{mean_var_rb:.2e}",
            f"{variance_ratio:.6f}", f"{prop41_diff:.6f}",
            T, L, f"{mean_adv_mag:.6f}",
        ])

        if len(examples) < 3:
            examples.append({
                "utt_id": utt_id,
                "ref_text": ref_text,
                "candidates": [(t, w) for (t, _), w in zip(candidates, wers)],
                "advantages": advantages,
                "mean_var_flat": mean_var_flat,
                "mean_var_rb": mean_var_rb,
                "variance_ratio": variance_ratio,
            })

        # Cleanup
        del (grads_flat, grads_rb, var_flat, var_rb,
             grad_flat_list, grad_rb_list,
             lattice, log_probs_init, encoder_out, feats_gpu,
             encoder_out_det)
        torch.cuda.empty_cache()

        if (utt_idx + 1) % 10 == 0:
            ratios = [r["variance_ratio"] for r in all_results]
            print(
                f"  [{utt_idx+1}/{len(utterances)}] "
                f"mean ratio: {np.mean(ratios):.4f}, "
                f"prop41 max diff: {max(prop41_diffs):.2e}"
            )

    csv_file.close()
    print(f"CSV saved: {csv_path}")

    if not all_results:
        print("ERROR: No valid utterances processed!")
        return None

    ratios = [r["variance_ratio"] for r in all_results]
    rb_worse = sum(1 for r in ratios if r < 0.95)

    summary = {
        "n_utterances": len(all_results),
        "n_skipped_zero_variance": n_skipped,
        "mean_variance_ratio": float(np.mean(ratios)),
        "median_variance_ratio": float(np.median(ratios)),
        "min_variance_ratio": float(np.min(ratios)),
        "max_variance_ratio": float(np.max(ratios)),
        "std_variance_ratio": float(np.std(ratios)),
        "mean_prop41_diff": float(np.mean(prop41_diffs)),
        "max_prop41_diff": float(np.max(prop41_diffs)),
        "prop41_violations": sum(1 for d in prop41_diffs if d >= 0.1),
        "rb_worse_count": rb_worse,
        "G": G,
        "target_layer_shape": list(target_layer.weight.shape),
    }

    summary_path = output_dir / "grad_variance_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")

    examples_path = output_dir / "examples.txt"
    with open(examples_path, "w") as f:
        for ex in examples:
            f.write(f"=== Utterance: {ex['utt_id']} ===\n")
            f.write(f"Reference: {ex['ref_text']}\n\n")
            f.write("Candidates:\n")
            for i, ((text, wer), adv) in enumerate(
                zip(ex["candidates"], ex["advantages"])
            ):
                f.write(
                    f"  [{i}] WER={wer*100:5.1f}% A_hat={adv:+.4f}: {text}\n"
                )
            f.write(f"\nVariance (flat):  {ex['mean_var_flat']:.2e}\n")
            f.write(f"Variance (RB):    {ex['mean_var_rb']:.2e}\n")
            f.write(f"Ratio (flat/RB):  {ex['variance_ratio']:.4f}\n")
            f.write("\n\n")
    print(f"Examples saved: {examples_path}")

    print("\n" + "=" * 70)
    print("STAGE 3 SUMMARY: Gradient Variance Reduction")
    print("=" * 70)
    print(f"  Utterances analyzed:        {summary['n_utterances']}")
    print(f"  Skipped (zero variance):    {summary['n_skipped_zero_variance']}")
    print(f"  Target layer:               Linear{tuple(summary['target_layer_shape'])}")
    print(f"  Group size G:               {G}")
    print()
    print(f"  Mean variance ratio:        {summary['mean_variance_ratio']:.4f}")
    print(f"  Median variance ratio:      {summary['median_variance_ratio']:.4f}")
    print(f"  Min / Max ratio:            {summary['min_variance_ratio']:.4f} / {summary['max_variance_ratio']:.4f}")
    print(f"  Std of ratio:               {summary['std_variance_ratio']:.4f}")
    print()
    print(f"  Prop 4.1 mean rel diff:     {summary['mean_prop41_diff']:.2e}")
    print(f"  Prop 4.1 max rel diff:      {summary['max_prop41_diff']:.2e}")
    print(f"  Prop 4.1 violations (>0.1): {summary['prop41_violations']}")
    print(f"  RB worse count (ratio<0.95):{summary['rb_worse_count']}")
    print("=" * 70)

    return summary

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 3: Gradient variance  --  flat vs RB credit assignment"
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
    print("RBPO Stage 3  --  Gradient Variance Measurement")
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
    output_dir = args.results_dir / "stage_3_grad_variance"
    summary = run_experiment(model, utterances, sp, device, output_dir, args.G)
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

if __name__ == "__main__":
    main()

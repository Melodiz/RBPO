#!/usr/bin/env python3
"""R-PROP2: Proposition 4.2 verification expanded from n=50 to n=250.

Reuses Stage 3b logic (CTC-marginalized vs Viterbi vs Sampled gradient
variance) on 250 dev-other utterances. Tightens CIs by ~sqrt(5) and
rules out outlier-driven artifacts from the original 50-utterance run.

Optionally correlates variance ratio with per-frame alignment entropy
from Stage 2 gamma analysis (if overlapping utterances exist).

Usage (Colab):
    python /content/rbpo/scripts/prop42_expansion.py \
        --model-dir /content/icefall-asr-librispeech-zipformer-small-cr-ctc \
        --icefall-dir /content/icefall \
        --data-dir /content/librispeech_data \
        --output-dir /content/drive/MyDrive/rbpo_results/R_prop2_expansion \
        --n-utterances 250 --G 8 --device cuda:0

Output:
    prop42_results.json    --  aggregate stats + n=50 vs n=250 comparison
    prop42_per_utt.csv     --  per-utterance variance ratios
    R_prop2_expansion_report.md  --  stage report
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "rbpo"))

from experiments.analysis.grad_variance_viterbi import (
    BLANK_ID,
    MAX_TOKEN,
    VOCAB_SIZE,
    build_numerator_lattice,
    build_outer_lattice,
    compute_ctc_gradient,
    compute_one_hot_gradient,
    compute_wer,
    ctc_collapse,
    extract_nbest_with_tokens,
    find_ctc_projection,
    load_model,
    sampled_alignment,
    viterbi_alignment,
)

def load_utterances(data_dir: Path, split: str, num_utterances: int):
    """Load utterances with on-the-fly fbank extraction when pre-computed
    features are not available (common for cuts cached without features)."""
    from lhotse import Fbank, FbankConfig, load_manifest_lazy

    cuts_path = data_dir / "cuts" / f"librispeech_cuts_{split}.jsonl.gz"
    assert cuts_path.exists(), f"CutSet not found: {cuts_path}"

    cuts = load_manifest_lazy(str(cuts_path))

    first_cut = next(iter(load_manifest_lazy(str(cuts_path))))
    needs_onthefly = first_cut.load_features() is None

    fbank = None
    if needs_onthefly:
        fbank = Fbank(FbankConfig(num_mel_bins=80, dither=0.0))
        print("Features not pre-computed  --  extracting fbank on-the-fly")

    utterances = []
    for cut in cuts:
        if len(utterances) >= num_utterances:
            break

        if needs_onthefly:
            audio = cut.load_audio()  # (1, T_samples)
            feats_np = fbank.extract(
                torch.from_numpy(audio), cut.recording.sampling_rate
            )
            feats = feats_np if isinstance(feats_np, torch.Tensor) else torch.from_numpy(feats_np)
        else:
            feats_np = cut.load_features()
            if feats_np is None:
                continue
            feats = torch.from_numpy(feats_np)

        ref_text = " ".join(
            s.text for s in cut.supervisions if s.text
        ).strip().lower()
        if not ref_text:
            continue
        utterances.append((cut.id, feats, ref_text))

    print(f"Loaded {len(utterances)} utterances from {split}")
    return utterances

# Stage 3b reference values (n=50, G=8)
STAGE3B_REF = {
    "n": 50,
    "mean_viterbi_ctc": 2.7030,
    "median_viterbi_ctc": 2.4715,
    "min_viterbi_ctc": 1.1070,
    "max_viterbi_ctc": 8.1857,
    "mean_sampled_ctc": 3.6658,
    "median_sampled_ctc": 3.3820,
    "min_sampled_ctc": 1.1070,
    "max_sampled_ctc": 11.4148,
    "rb_worse_viterbi": 0,
    "rb_worse_sampled": 0,
}

def bootstrap_ci(values, n_boot=10000, seed=42, alpha=0.05):
    rng = np.random.RandomState(seed)
    n = len(values)
    boot_means = np.array([
        np.mean(rng.choice(values, size=n, replace=True))
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)

def load_gamma_entropy(gamma_csv_path: Path) -> dict:
    """Load per-utterance mean entropy from Stage 2 gamma analysis."""
    entropy_map = {}
    if not gamma_csv_path.exists():
        return entropy_map
    with open(gamma_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            entropy_map[row["utt_id"]] = float(row["mean_entropy"])
    return entropy_map

def run_prop42(
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

    all_results = []
    n_skipped = 0
    rb_worse_viterbi = 0
    rb_worse_sampled = 0

    for utt_idx, (utt_id, feats, ref_text) in enumerate(utterances):
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

        for i, ((_, token_ids_i), adv_i) in enumerate(
            zip(candidates, advantages)
        ):
            if not token_ids_i:
                z = torch.zeros_like(target_layer.weight)
                grad_ctc_list.append(z)
                grad_viterbi_list.append(z)
                grad_sampled_list.append(z)
                continue

            num_lattice = build_numerator_lattice(
                log_probs_det[0], token_ids_i, T, device
            )

            vit_align = viterbi_alignment(num_lattice, T)
            samp_align = sampled_alignment(num_lattice, T)

            vit_collapsed = ctc_collapse(vit_align)
            assert vit_collapsed == token_ids_i, (
                f"Viterbi path mismatch on utt {utt_id}, cand {i}"
            )

            g_ctc = compute_ctc_gradient(
                model, encoder_out_det, token_ids_i, T,
                adv_i, target_layer, device,
            )
            grad_ctc_list.append(g_ctc)

            g_vit = compute_one_hot_gradient(
                model, encoder_out_det, vit_align, T,
                adv_i, target_layer, device,
            )
            grad_viterbi_list.append(g_vit)

            g_samp = compute_one_hot_gradient(
                model, encoder_out_det, samp_align, T,
                adv_i, target_layer, device,
            )
            grad_sampled_list.append(g_samp)

            del num_lattice

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

        for g in grad_ctc_list + grad_viterbi_list + grad_sampled_list:
            assert g.isfinite().all(), f"Non-finite gradient on utt {utt_id}"

        if ratio_vit < 0.95:
            rb_worse_viterbi += 1
        if ratio_samp < 0.95:
            rb_worse_sampled += 1

        G_eff = len(candidates)
        result = {
            "utt_id": utt_id,
            "var_ctc": mean_var_ctc,
            "var_viterbi": mean_var_vit,
            "var_sampled": mean_var_samp,
            "ratio_viterbi": ratio_vit,
            "ratio_sampled": ratio_samp,
            "G_effective": G_eff,
            "T": T,
            "L": L,
        }
        all_results.append(result)

        del (grads_ctc, grads_vit, grads_samp,
             var_ctc, var_vit, var_samp,
             grad_ctc_list, grad_viterbi_list, grad_sampled_list,
             outer_lattice, log_probs_init, log_probs_det,
             encoder_out, feats_gpu, encoder_out_det)
        torch.cuda.empty_cache()

        if (utt_idx + 1) % 25 == 0:
            ratios_v = [r["ratio_viterbi"] for r in all_results]
            ratios_s = [r["ratio_sampled"] for r in all_results]
            print(
                f"  Step {utt_idx+1}/{len(utterances)}: "
                f"viterbi/ctc={np.mean(ratios_v):.3f}, "
                f"sampled/ctc={np.mean(ratios_s):.3f}, "
                f"valid={len(all_results)}, skipped={n_skipped}"
            )

    if not all_results:
        print("ERROR: No valid utterances processed!")
        return None

    csv_path = output_dir / "prop42_per_utt.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "utt_id", "var_ctc", "var_viterbi", "var_sampled",
            "ratio_viterbi", "ratio_sampled", "G_effective",
        ])
        for r in all_results:
            writer.writerow([
                r["utt_id"],
                f"{r['var_ctc']:.6e}",
                f"{r['var_viterbi']:.6e}",
                f"{r['var_sampled']:.6e}",
                f"{r['ratio_viterbi']:.6f}",
                f"{r['ratio_sampled']:.6f}",
                r["G_effective"],
            ])
    print(f"Per-utterance CSV: {csv_path}")

    ratios_v = np.array([r["ratio_viterbi"] for r in all_results])
    ratios_s = np.array([r["ratio_sampled"] for r in all_results])

    n_violations_ordering = sum(
        1 for r in all_results
        if r["var_ctc"] > r["var_viterbi"] or r["var_viterbi"] > r["var_sampled"]
    )

    ci_v_lo, ci_v_hi = bootstrap_ci(ratios_v)
    ci_s_lo, ci_s_hi = bootstrap_ci(ratios_s)

    summary = {
        "experiment": "R-PROP2",
        "description": "Proposition 4.2 verification expanded to n=250",
        "n_utts_requested": len(utterances),
        "n_utts_valid": len(all_results),
        "n_skipped": n_skipped,
        "G": G,
        "mean_ratio_viterbi": float(np.mean(ratios_v)),
        "sd_ratio_viterbi": float(np.std(ratios_v)),
        "ci_lower_viterbi": ci_v_lo,
        "ci_upper_viterbi": ci_v_hi,
        "median_ratio_viterbi": float(np.median(ratios_v)),
        "mean_ratio_sampled": float(np.mean(ratios_s)),
        "sd_ratio_sampled": float(np.std(ratios_s)),
        "ci_lower_sampled": ci_s_lo,
        "ci_upper_sampled": ci_s_hi,
        "median_ratio_sampled": float(np.median(ratios_s)),
        "n_violations": n_violations_ordering,
        "n_rb_worse_viterbi": rb_worse_viterbi,
        "n_rb_worse_sampled": rb_worse_sampled,
        "min_ratio_viterbi": float(np.min(ratios_v)),
        "max_ratio_viterbi": float(np.max(ratios_v)),
        "min_ratio_sampled": float(np.min(ratios_s)),
        "max_ratio_sampled": float(np.max(ratios_s)),
        "percentiles_viterbi": {
            "p5": float(np.percentile(ratios_v, 5)),
            "p10": float(np.percentile(ratios_v, 10)),
            "p25": float(np.percentile(ratios_v, 25)),
            "p50": float(np.percentile(ratios_v, 50)),
            "p75": float(np.percentile(ratios_v, 75)),
            "p90": float(np.percentile(ratios_v, 90)),
            "p95": float(np.percentile(ratios_v, 95)),
        },
        "percentiles_sampled": {
            "p5": float(np.percentile(ratios_s, 5)),
            "p10": float(np.percentile(ratios_s, 10)),
            "p25": float(np.percentile(ratios_s, 25)),
            "p50": float(np.percentile(ratios_s, 50)),
            "p75": float(np.percentile(ratios_s, 75)),
            "p90": float(np.percentile(ratios_s, 90)),
            "p95": float(np.percentile(ratios_s, 95)),
        },
        "stage3b_reference": STAGE3B_REF,
    }

    return summary, all_results

def entropy_correlation(all_results, gamma_csv_path: Path):
    """Compute Spearman correlation between variance ratio and alignment entropy."""
    from scipy.stats import spearmanr

    entropy_map = load_gamma_entropy(gamma_csv_path)
    if not entropy_map:
        return None

    paired = []
    for r in all_results:
        if r["utt_id"] in entropy_map:
            paired.append((
                entropy_map[r["utt_id"]],
                r["ratio_viterbi"],
                r["ratio_sampled"],
            ))

    if len(paired) < 10:
        return None

    entropies = [p[0] for p in paired]
    vit_ratios = [p[1] for p in paired]
    samp_ratios = [p[2] for p in paired]

    rho_vit, p_vit = spearmanr(entropies, vit_ratios)
    rho_samp, p_samp = spearmanr(entropies, samp_ratios)

    return {
        "n_overlap": len(paired),
        "spearman_entropy_vs_viterbi_ratio": float(rho_vit),
        "p_value_viterbi": float(p_vit),
        "spearman_entropy_vs_sampled_ratio": float(rho_samp),
        "p_value_sampled": float(p_samp),
        "scatter_data": [
            {"entropy": e, "viterbi_ctc_ratio": v, "sampled_ctc_ratio": s}
            for e, v, s in paired
        ],
    }

def write_report(report_path: Path, summary: dict, entropy_result: dict | None):
    ref = summary["stage3b_reference"]

    with open(report_path, "w") as f:
        f.write("# R-PROP2: Proposition 4.2 Verification Expanded (n=250)\n\n")

        f.write("## Summary\n\n")
        f.write(
            f"Expanded Prop 4.2 gradient variance verification from n={ref['n']} "
            f"to n={summary['n_utts_valid']} valid utterances (G={summary['G']}). "
        )
        f.write(
            f"Mean Viterbi/CTC ratio = {summary['mean_ratio_viterbi']:.4f} "
            f"(95% CI [{summary['ci_lower_viterbi']:.4f}, {summary['ci_upper_viterbi']:.4f}]). "
        )
        f.write(
            f"Mean Sampled/CTC ratio = {summary['mean_ratio_sampled']:.4f} "
            f"(95% CI [{summary['ci_lower_sampled']:.4f}, {summary['ci_upper_sampled']:.4f}]). "
        )
        f.write(
            f"Ordering violations (Var_CTC > Var_Viterbi or Var_Viterbi > Var_Sampled): "
            f"**{summary['n_violations']}**.\n\n"
        )

        f.write("## Comparison: n=50 vs n=250\n\n")
        f.write("| Metric | n=50 (Stage 3b) | n=250 (R-PROP2) |\n")
        f.write("|--------|-----------------|------------------|\n")
        f.write(
            f"| Mean Viterbi/CTC | {ref['mean_viterbi_ctc']:.4f} "
            f"| {summary['mean_ratio_viterbi']:.4f} |\n"
        )
        f.write(
            f"| Median Viterbi/CTC | {ref['median_viterbi_ctc']:.4f} "
            f"| {summary['median_ratio_viterbi']:.4f} |\n"
        )
        f.write(
            f"| SD Viterbi/CTC |  --  "
            f"| {summary['sd_ratio_viterbi']:.4f} |\n"
        )
        f.write(
            f"| 95% CI Viterbi/CTC |  --  "
            f"| [{summary['ci_lower_viterbi']:.4f}, {summary['ci_upper_viterbi']:.4f}] |\n"
        )
        f.write(
            f"| Min/Max Viterbi/CTC | {ref['min_viterbi_ctc']:.4f}/{ref['max_viterbi_ctc']:.4f} "
            f"| {summary['min_ratio_viterbi']:.4f}/{summary['max_ratio_viterbi']:.4f} |\n"
        )
        f.write(
            f"| Mean Sampled/CTC | {ref['mean_sampled_ctc']:.4f} "
            f"| {summary['mean_ratio_sampled']:.4f} |\n"
        )
        f.write(
            f"| Median Sampled/CTC | {ref['median_sampled_ctc']:.4f} "
            f"| {summary['median_ratio_sampled']:.4f} |\n"
        )
        f.write(
            f"| SD Sampled/CTC |  --  "
            f"| {summary['sd_ratio_sampled']:.4f} |\n"
        )
        f.write(
            f"| 95% CI Sampled/CTC |  --  "
            f"| [{summary['ci_lower_sampled']:.4f}, {summary['ci_upper_sampled']:.4f}] |\n"
        )
        f.write(
            f"| Min/Max Sampled/CTC | {ref['min_sampled_ctc']:.4f}/{ref['max_sampled_ctc']:.4f} "
            f"| {summary['min_ratio_sampled']:.4f}/{summary['max_ratio_sampled']:.4f} |\n"
        )
        f.write(
            f"| Ordering violations | {ref['rb_worse_viterbi']} "
            f"| {summary['n_violations']} |\n"
        )
        f.write(
            f"| Skipped |  --  | {summary['n_skipped']} |\n\n"
        )

        f.write("## Percentile Distribution (Viterbi/CTC ratio)\n\n")
        f.write("| Percentile | Value |\n")
        f.write("|------------|-------|\n")
        for pct, key in [
            ("5th", "p5"), ("10th", "p10"), ("25th", "p25"),
            ("50th", "p50"), ("75th", "p75"), ("90th", "p90"), ("95th", "p95"),
        ]:
            f.write(
                f"| {pct} | {summary['percentiles_viterbi'][key]:.4f} |\n"
            )
        f.write("\n")

        f.write("## Percentile Distribution (Sampled/CTC ratio)\n\n")
        f.write("| Percentile | Value |\n")
        f.write("|------------|-------|\n")
        for pct, key in [
            ("5th", "p5"), ("10th", "p10"), ("25th", "p25"),
            ("50th", "p50"), ("75th", "p75"), ("90th", "p90"), ("95th", "p95"),
        ]:
            f.write(
                f"| {pct} | {summary['percentiles_sampled'][key]:.4f} |\n"
            )
        f.write("\n")

        f.write("## Histogram (Viterbi/CTC ratio, 10 bins)\n\n")
        ratios_v = []
        for key in ["p5", "p25", "p50", "p75", "p95"]:
            ratios_v.append(summary["percentiles_viterbi"][key])
        f.write(
            f"Range: [{summary['min_ratio_viterbi']:.2f}, "
            f"{summary['max_ratio_viterbi']:.2f}], "
            f"IQR: [{summary['percentiles_viterbi']['p25']:.2f}, "
            f"{summary['percentiles_viterbi']['p75']:.2f}]\n\n"
        )
        f.write("(Histogram bin counts saved in prop42_results.json)\n\n")

        if entropy_result:
            f.write("## Entropy Correlation Analysis\n\n")
            f.write(
                f"Overlapping utterances with Stage 2 gamma analysis: "
                f"**{entropy_result['n_overlap']}**\n\n"
            )
            f.write("| Correlation | Spearman rho | p-value |\n")
            f.write("|-------------|-------------|----------|\n")
            f.write(
                f"| Entropy vs Viterbi/CTC ratio | "
                f"{entropy_result['spearman_entropy_vs_viterbi_ratio']:.4f} | "
                f"{entropy_result['p_value_viterbi']:.4e} |\n"
            )
            f.write(
                f"| Entropy vs Sampled/CTC ratio | "
                f"{entropy_result['spearman_entropy_vs_sampled_ratio']:.4f} | "
                f"{entropy_result['p_value_sampled']:.4e} |\n"
            )
            f.write(
                "\nPositive correlation expected: higher alignment entropy "
                "means more spread in gamma_t, so one-hot estimators lose "
                "more relative to the marginalized estimator.\n\n"
            )
        else:
            f.write("## Entropy Correlation Analysis\n\n")
            f.write(
                "Skipped: fewer than 10 overlapping utterances with "
                "Stage 2 gamma analysis, or gamma_stats.csv not found.\n\n"
            )

        f.write("## Verification Checklist\n\n")
        f.write(f"- [{'x' if summary['n_violations'] == 0 else ' '}] "
                f"Zero ordering violations (Var_CTC <= Var_Viterbi <= Var_Sampled)\n")
        vit_consistent = abs(summary["mean_ratio_viterbi"] - ref["mean_viterbi_ctc"]) < 1.0
        f.write(f"- [{'x' if vit_consistent else ' '}] "
                f"Mean Viterbi/CTC ratio consistent with Stage 3b "
                f"({ref['mean_viterbi_ctc']:.2f})\n")
        samp_consistent = abs(summary["mean_ratio_sampled"] - ref["mean_sampled_ctc"]) < 1.0
        f.write(f"- [{'x' if samp_consistent else ' '}] "
                f"Mean Sampled/CTC ratio consistent with Stage 3b "
                f"({ref['mean_sampled_ctc']:.2f})\n")
        ci_width = summary["ci_upper_viterbi"] - summary["ci_lower_viterbi"]
        f.write(f"- [ ] 95% CI width ({ci_width:.4f}) vs estimated Stage 3b "
                f"CI width (check ~sqrt(5) narrower)\n")
        f.write("\n")

        f.write("## Method\n\n")
        f.write("- Model: Zipformer-S CR-CTC (22M params, BPE-500)\n")
        f.write(f"- Data: first {summary['n_utts_requested']} utterances from "
                f"dev-other (by cut order)\n")
        f.write(f"- G = {summary['G']} candidates per utterance\n")
        f.write("- Three gradient estimators on CTC output projection:\n")
        f.write("  1. CTC-marginalized (gamma-weighted via k2 backward)\n")
        f.write("  2. Viterbi (best single alignment, one-hot credit)\n")
        f.write("  3. Sampled (one random alignment ~ posterior, one-hot credit)\n")
        f.write("- Variance = across-candidate variance of advantage-weighted gradients\n")
        f.write("- Bootstrap CI: B=10000, seed=42\n\n")

        f.write("## Paper Update\n\n")
        f.write(
            "Section 3.3 Proposition 2 verification paragraph should be updated: "
            f"n=50 -> n={summary['n_utts_valid']}, "
            f"mean Viterbi/CTC ratio = {summary['mean_ratio_viterbi']:.2f} "
            f"(95% CI [{summary['ci_lower_viterbi']:.2f}, "
            f"{summary['ci_upper_viterbi']:.2f}]), "
            f"mean Sampled/CTC ratio = {summary['mean_ratio_sampled']:.2f} "
            f"(95% CI [{summary['ci_lower_sampled']:.2f}, "
            f"{summary['ci_upper_sampled']:.2f}]), "
            f"{summary['n_violations']} ordering violations.\n"
        )

        f.write("\n## Bring-Back Files\n\n")
        f.write("```\n")
        f.write("results/R_prop2_expansion/prop42_results.json\n")
        f.write("results/R_prop2_expansion/prop42_per_utt.csv\n")
        f.write("reports/R_prop2_expansion_report.md\n")
        f.write("```\n")

def parse_args():
    parser = argparse.ArgumentParser(
        description="R-PROP2: Prop 4.2 verification expanded to n=250"
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
        "--output-dir", type=Path,
        default=Path("/content/drive/MyDrive/rbpo_results/R_prop2_expansion"),
    )
    parser.add_argument("--n-utterances", type=int, default=250)
    parser.add_argument("--G", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--gamma-csv", type=Path,
        default=None,
        help="Path to gamma_stats.csv from Stage 2 for entropy correlation",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 60)
    print("R-PROP2  --  Proposition 4.2 Expansion (n=250)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Utterances: {args.n_utterances}")
    print(f"Group size G: {args.G}")

    sp = spm.SentencePieceProcessor()
    bpe_path = args.model_dir / "data" / "lang_bpe_500" / "bpe.model"
    assert bpe_path.exists(), f"BPE model not found: {bpe_path}"
    sp.load(str(bpe_path))
    print(f"BPE vocab: {sp.get_piece_size()} tokens")

    model = load_model(args.model_dir, args.icefall_dir, device)
    utterances = load_utterances(args.data_dir, "dev-other", args.n_utterances)

    assert len(utterances) >= 50, (
        f"Need at least 50 utterances for Stage 3b comparison, got {len(utterances)}"
    )

    t0 = time.time()
    output_dir = Path(args.output_dir)
    result = run_prop42(model, utterances, sp, device, output_dir, args.G)
    elapsed = time.time() - t0

    if result is None:
        print("FAILED: no valid utterances processed")
        sys.exit(1)

    summary, all_results = result

    # Entropy correlation
    gamma_csv = args.gamma_csv
    if gamma_csv is None:
        for candidate in [
            output_dir.parent / "gap_covering" / "stage2" / "gamma_stats.csv",
            REPO_ROOT / "results" / "gap_covering" / "stage2" / "gamma_stats.csv",
            Path("/content/drive/MyDrive/rbpo_results/gap_covering/stage2/gamma_stats.csv"),
        ]:
            if candidate.exists():
                gamma_csv = candidate
                break

    entropy_result = None
    if gamma_csv and gamma_csv.exists():
        try:
            entropy_result = entropy_correlation(all_results, gamma_csv)
            if entropy_result:
                summary["entropy_correlation"] = {
                    k: v for k, v in entropy_result.items()
                    if k != "scatter_data"
                }
                print(
                    f"Entropy correlation (n={entropy_result['n_overlap']}): "
                    f"rho_vit={entropy_result['spearman_entropy_vs_viterbi_ratio']:.3f}, "
                    f"rho_samp={entropy_result['spearman_entropy_vs_sampled_ratio']:.3f}"
                )
        except ImportError:
            print("scipy not available  --  skipping entropy correlation")

    # Histogram bins for JSON
    ratios_v = np.array([r["ratio_viterbi"] for r in all_results])
    counts, edges = np.histogram(ratios_v, bins=10)
    summary["histogram_viterbi_10bins"] = [
        {"bin_start": float(edges[i]), "bin_end": float(edges[i + 1]), "count": int(counts[i])}
        for i in range(len(counts))
    ]

    ratios_s = np.array([r["ratio_sampled"] for r in all_results])
    counts_s, edges_s = np.histogram(ratios_s, bins=10)
    summary["histogram_sampled_10bins"] = [
        {"bin_start": float(edges_s[i]), "bin_end": float(edges_s[i + 1]), "count": int(counts_s[i])}
        for i in range(len(counts_s))
    ]

    json_path = output_dir / "prop42_results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results JSON: {json_path}")

    report_path = output_dir / "R_prop2_expansion_report.md"
    write_report(report_path, summary, entropy_result)
    print(f"Report: {report_path}")

    print(f"\nTotal time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print("\n" + "=" * 70)
    print("R-PROP2 SUMMARY")
    print("=" * 70)
    print(f"  Valid utterances:       {summary['n_utts_valid']}")
    print(f"  Skipped:                {summary['n_skipped']}")
    print(f"  Ordering violations:    {summary['n_violations']}")
    print()
    print(f"  Viterbi/CTC ratio:")
    print(f"    Mean +/- SD:            {summary['mean_ratio_viterbi']:.4f} +/- {summary['sd_ratio_viterbi']:.4f}")
    print(f"    95% CI:               [{summary['ci_lower_viterbi']:.4f}, {summary['ci_upper_viterbi']:.4f}]")
    print(f"    Median:               {summary['median_ratio_viterbi']:.4f}")
    print(f"    Min / Max:            {summary['min_ratio_viterbi']:.4f} / {summary['max_ratio_viterbi']:.4f}")
    print()
    print(f"  Sampled/CTC ratio:")
    print(f"    Mean +/- SD:            {summary['mean_ratio_sampled']:.4f} +/- {summary['sd_ratio_sampled']:.4f}")
    print(f"    95% CI:               [{summary['ci_lower_sampled']:.4f}, {summary['ci_upper_sampled']:.4f}]")
    print(f"    Median:               {summary['median_ratio_sampled']:.4f}")
    print(f"    Min / Max:            {summary['min_ratio_sampled']:.4f} / {summary['max_ratio_sampled']:.4f}")
    print("=" * 70)

    print("\nBring-back files:")
    print(f"  {json_path}")
    print(f"  {output_dir / 'prop42_per_utt.csv'}")
    print(f"  {report_path}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""E5: MC-Dropout Seed Variation  --  Is the 0.04pp improvement real?

Runs MC-Dropout T=4 + MBR-CER 5x with different random seeds to determine
if the marginal improvement (5.98% vs 6.02%) is reproducible or just noise.

For each seed: run MC-Dropout averaged-posterior decoding, record per-utterance
hypotheses, compute corpus WER, run paired bootstrap vs greedy.

Usage:
    python experiments/robustness/mc_dropout_seeds.py \
        --data-dir /content/librispeech_data \
        --model-dir /content/icefall-asr-librispeech-zipformer-small-cr-ctc \
        --icefall-dir /content/icefall \
        --output-dir /content/drive/MyDrive/rbpo_results/mc_dropout_seeds \
        --seeds 42,123,456,789,1024 \
        --T 4 \
        --n-bootstrap 10000
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import editdistance
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.significance_tests import paired_bootstrap_wer, corpus_wer

BLANK_ID = 0
MAX_TOKEN = 499
NUM_PATHS = 64
G = 16
NBEST_SCALE = 1.0
MBR_TAU = float("inf")


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
    return model


def ctc_collapse(token_ids):
    result = []
    prev = None
    for t in token_ids:
        if t != BLANK_ID and t != prev:
            result.append(t)
        prev = t
    return result


def build_lattice(log_probs, topo, device):
    import k2
    T = log_probs.shape[0]
    supervision_segments = torch.tensor([[0, 0, T]], dtype=torch.int32)
    dense_fsa = k2.DenseFsaVec(log_probs.unsqueeze(0), supervision_segments)
    lattice = k2.intersect_dense(topo, dense_fsa, output_beam=8.0)
    lattice = k2.connect(lattice)
    return lattice


def alignment_log_prob(label_seq, log_probs_cpu):
    T = log_probs_cpu.shape[0]
    if len(label_seq) != T:
        return float("-inf")
    idx = torch.tensor(label_seq, dtype=torch.long)
    return log_probs_cpu[torch.arange(T), idx].sum().item()


def extract_nbest_with_scores(lattice, num_paths, nbest_scale, sp, log_probs_cpu):
    import k2
    nbest = k2.Nbest.from_lattice(
        lattice, num_paths=num_paths,
        use_double_scores=True, nbest_scale=nbest_scale,
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
            "text": text, "tokens": token_ids,
            "ctc_log_prob": score,
            "len_tokens": len(token_ids), "len_chars": len(text),
        }
        if text in seen:
            if score > seen[text]["ctc_log_prob"]:
                seen[text] = entry
        else:
            seen[text] = entry

    return sorted(seen.values(), key=lambda c: c["ctc_log_prob"], reverse=True)


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


def char_distance(a, b):
    denom = max(len(a), len(b), 1)
    return editdistance.eval(list(a), list(b)) / denom


def mbr_cer_select(texts, log_probs, tau):
    n = len(texts)
    if n == 1:
        return 0
    if tau == float("inf"):
        weights = np.ones(n) / n
    else:
        a = np.array(log_probs, dtype=np.float64) / tau
        a -= a.max()
        weights = np.exp(a)
        weights /= weights.sum()
    scores = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                scores[i] += weights[j] * char_distance(texts[i], texts[j])
    return int(np.argmin(scores))


def enable_dropout_only(model):
    model.eval()
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()
            count += 1
    return count


def run_mc_dropout_seed(model, utterances, sp, topo, device, T_passes, seed):
    """Run MC-Dropout with a specific seed. Returns per-utterance hypotheses."""
    import k2

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    n_dropout = enable_dropout_only(model)
    if n_dropout == 0:
        print(f"  WARNING: No dropout layers found. Results will be deterministic.")

    per_utt = []

    for utt_id, feats, ref_text in tqdm(utterances, desc=f"  seed={seed}"):
        ref_w = ref_text.split()
        feats_gpu = feats.unsqueeze(0).to(device)
        feat_lens = torch.tensor([feats.shape[0]], dtype=torch.int64, device=device)

        with torch.no_grad():
            probs_sum = None
            for t in range(T_passes):
                encoder_out, enc_lens = model.forward_encoder(feats_gpu, feat_lens)
                log_probs = model.ctc_output(encoder_out)
                probs = log_probs.exp()
                if probs_sum is None:
                    probs_sum = probs
                else:
                    probs_sum = probs_sum + probs
                del encoder_out, log_probs

            avg_probs = probs_sum / T_passes
            avg_log_probs = avg_probs.clamp(min=1e-30).log()

        lp_utt = avg_log_probs[0]

        # Greedy from averaged posteriors
        greedy_ids = lp_utt.argmax(dim=-1).tolist()
        greedy_collapsed = ctc_collapse(greedy_ids)
        greedy_text = sp.decode(greedy_collapsed).strip().lower()
        greedy_score = alignment_log_prob(greedy_ids, lp_utt.cpu())

        # N-best
        lp_cpu = lp_utt.cpu()
        try:
            lattice = build_lattice(lp_utt, topo, device)
            candidates = extract_nbest_with_scores(
                lattice, NUM_PATHS, NBEST_SCALE, sp, lp_cpu
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
                    "text": greedy_text, "tokens": greedy_collapsed,
                    "ctc_log_prob": greedy_score,
                    "len_tokens": len(greedy_collapsed),
                    "len_chars": len(greedy_text),
                }
            candidates = [greedy_entry] + rest
            candidates = candidates[:G]
            del lattice
        except Exception:
            candidates = [{
                "text": greedy_text, "tokens": greedy_collapsed,
                "ctc_log_prob": greedy_score,
                "len_tokens": len(greedy_collapsed),
                "len_chars": len(greedy_text),
            }]

        # MBR-CER selection (uniform weights)
        texts = [c["text"] for c in candidates]
        lps = [c["ctc_log_prob"] for c in candidates]
        mbr_idx = mbr_cer_select(texts, lps, MBR_TAU)

        per_utt.append({
            "utt_id": utt_id,
            "ref_text": ref_text,
            "greedy_text": greedy_text,
            "mbr_text": texts[mbr_idx],
            "n_candidates": len(candidates),
        })

        del avg_log_probs, avg_probs, probs_sum
        torch.cuda.empty_cache()

    model.eval()
    return per_utt


def parse_args():
    parser = argparse.ArgumentParser(description="E5: MC-Dropout Seed Variation")
    parser.add_argument("--data-dir", type=Path, default=Path("/content/librispeech_data"))
    parser.add_argument("--model-dir", type=Path,
                        default=Path("/content/icefall-asr-librispeech-zipformer-small-cr-ctc"))
    parser.add_argument("--icefall-dir", type=Path, default=Path("/content/icefall"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/content/drive/MyDrive/rbpo_results/mc_dropout_seeds"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seeds", type=str, default="42,123,456,789,1024")
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed-bootstrap", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device(args.device)

    print("=" * 70)
    print("E5: MC-Dropout Seed Variation")
    print("=" * 70)
    print(f"  Seeds: {seeds}")
    print(f"  T passes: {args.T}")
    print(f"  Bootstrap: B={args.n_bootstrap}")

    import sentencepiece as spm
    import k2

    sp = spm.SentencePieceProcessor()
    bpe_path = args.model_dir / "data" / "lang_bpe_500" / "bpe.model"
    sp.load(str(bpe_path))

    model = load_model(args.model_dir, args.icefall_dir, device)
    utterances = load_all_utterances(args.data_dir, "dev-other")
    topo = k2.ctc_topo(max_token=MAX_TOKEN, modified=False, device=device)

    print(f"  Utterances: {len(utterances)}")
    print(f"  Dropout layers: {enable_dropout_only(model)}")
    model.eval()

    # First: get deterministic greedy baseline (no dropout)
    print("\n--- Greedy baseline (no dropout) ---")
    greedy_hyps = []
    ref_words_list = []
    model.eval()
    for utt_id, feats, ref_text in tqdm(utterances, desc="  Greedy"):
        feats_gpu = feats.unsqueeze(0).to(device)
        feat_lens = torch.tensor([feats.shape[0]], dtype=torch.int64, device=device)
        with torch.no_grad():
            encoder_out, _ = model.forward_encoder(feats_gpu, feat_lens)
            log_probs = model.ctc_output(encoder_out)
        greedy_ids = log_probs[0].argmax(dim=-1).tolist()
        greedy_collapsed = ctc_collapse(greedy_ids)
        greedy_text = sp.decode(greedy_collapsed).strip().lower()
        greedy_hyps.append(greedy_text)
        ref_words_list.append(ref_text.split())
        del encoder_out, log_probs
        torch.cuda.empty_cache()

    greedy_wer = corpus_wer(ref_words_list, [h.split() for h in greedy_hyps])
    print(f"  Greedy WER: {greedy_wer*100:.4f}%")

    seed_results = []
    for seed in seeds:
        print(f"\n--- MC-Dropout T={args.T}, seed={seed} ---")
        t0 = time.time()
        per_utt = run_mc_dropout_seed(model, utterances, sp, topo, device, args.T, seed)
        elapsed = time.time() - t0

        # Corpus WER for greedy-from-averaged and MBR
        mc_greedy_hyps = [r["greedy_text"] for r in per_utt]
        mc_mbr_hyps = [r["mbr_text"] for r in per_utt]

        mc_greedy_wer = corpus_wer(ref_words_list, [h.split() for h in mc_greedy_hyps])
        mc_mbr_wer = corpus_wer(ref_words_list, [h.split() for h in mc_mbr_hyps])

        print(f"  MC-Greedy WER: {mc_greedy_wer*100:.4f}%")
        print(f"  MC-MBR WER:    {mc_mbr_wer*100:.4f}%")

        # Bootstrap: MBR vs deterministic greedy
        bootstrap_res = paired_bootstrap_wer(
            ref_words_list,
            [h.split() for h in mc_mbr_hyps],
            [h.split() for h in greedy_hyps],
            n_bootstrap=args.n_bootstrap,
            seed=args.seed_bootstrap,
        )

        print(f"  Bootstrap: delta={bootstrap_res['delta']*100:+.4f}pp, "
              f"p={bootstrap_res['p_value']:.4f}, "
              f"CI=[{bootstrap_res['ci_lower']*100:+.3f}, {bootstrap_res['ci_upper']*100:+.3f}]")

        seed_result = {
            "seed": seed,
            "mc_greedy_wer": mc_greedy_wer,
            "mc_mbr_wer": mc_mbr_wer,
            "delta_pp": bootstrap_res["delta"] * 100,
            "p_value": bootstrap_res["p_value"],
            "ci_lower": bootstrap_res["ci_lower"] * 100,
            "ci_upper": bootstrap_res["ci_upper"] * 100,
            "significant_005": bootstrap_res["p_value"] < 0.05,
            "elapsed": elapsed,
        }
        seed_results.append(seed_result)

        jsonl_path = args.output_dir / f"per_utterance_seed_{seed}.jsonl"
        with open(jsonl_path, "w") as f:
            for r in per_utt:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary statistics
    wers = [r["mc_mbr_wer"] for r in seed_results]
    p_values = [r["p_value"] for r in seed_results]
    n_significant = sum(1 for r in seed_results if r["significant_005"])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Greedy baseline: {greedy_wer*100:.4f}%")
    print(f"  MC-MBR WERs: {[f'{w*100:.4f}%' for w in wers]}")
    print(f"  Mean +/- std: {np.mean(wers)*100:.4f}% +/- {np.std(wers)*100:.4f}%")
    print(f"  p-values: {[f'{p:.4f}' for p in p_values]}")
    print(f"  Significant at alpha=0.05: {n_significant}/{len(seeds)}")

    print("\n--- Writing outputs ---")

    # JSON
    out_json = {
        "metadata": {
            "T_passes": args.T,
            "G": G,
            "MBR_tau": "inf",
            "n_bootstrap": args.n_bootstrap,
            "n_utterances": len(utterances),
            "greedy_wer": greedy_wer,
        },
        "seed_results": seed_results,
        "summary": {
            "mean_wer": float(np.mean(wers)),
            "std_wer": float(np.std(wers)),
            "min_wer": float(np.min(wers)),
            "max_wer": float(np.max(wers)),
            "n_significant": n_significant,
            "n_seeds": len(seeds),
        },
    }
    p = args.output_dir / "mc_dropout_seed_results.json"
    with open(p, "w") as f:
        json.dump(out_json, f, indent=2, default=str)
    print(f"  Wrote {p}")

    # CSV
    p = args.output_dir / "mc_dropout_seed_summary.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "seed", "mc_greedy_wer", "mc_mbr_wer", "delta_pp",
            "p_value", "ci_lower", "ci_upper", "significant_005"
        ])
        w.writeheader()
        for r in seed_results:
            w.writerow({
                "seed": r["seed"],
                "mc_greedy_wer": f"{r['mc_greedy_wer']*100:.4f}",
                "mc_mbr_wer": f"{r['mc_mbr_wer']*100:.4f}",
                "delta_pp": f"{r['delta_pp']:+.4f}",
                "p_value": f"{r['p_value']:.4f}",
                "ci_lower": f"{r['ci_lower']:+.3f}",
                "ci_upper": f"{r['ci_upper']:+.3f}",
                "significant_005": r["significant_005"],
            })
    print(f"  Wrote {p}")

    # Report
    p = args.output_dir / "report_E5.md"
    lines = ["# E5: MC-Dropout Seed Variation  --  Stage Report", ""]
    lines.append(f"**Status:** Complete. {len(seeds)} seeds x T={args.T} passes, "
                 f"B={args.n_bootstrap} bootstrap.")
    lines.append("")
    lines.append("## TL;DR")
    lines.append("")
    if n_significant == 0:
        lines.append(f"**The MC-Dropout improvement is NOT reproducible.** "
                     f"Across {len(seeds)} random seeds, 0/{len(seeds)} reach significance "
                     f"at alpha=0.05. The 0.04pp improvement previously reported is within noise.")
    elif n_significant == len(seeds):
        lines.append(f"**The MC-Dropout improvement IS reproducible.** "
                     f"All {len(seeds)}/{len(seeds)} seeds reach significance at alpha=0.05.")
    else:
        lines.append(f"**Inconclusive:** {n_significant}/{len(seeds)} seeds reach significance. "
                     f"The effect is marginal and seed-dependent.")
    lines.append("")
    lines.append(f"Mean MC-MBR WER: **{np.mean(wers)*100:.4f}% +/- {np.std(wers)*100:.4f}%** "
                 f"(vs greedy {greedy_wer*100:.4f}%)")
    lines.append("")

    lines.append("## Per-Seed Results")
    lines.append("")
    lines.append("| Seed | MC-Greedy (%) | MC-MBR (%) | delta (pp) | p-value | CI (pp) | Sig? |")
    lines.append("|-----:|-------------:|-----------:|-------:|--------:|---------|:----:|")
    for r in seed_results:
        sig = "" if r["significant_005"] else " -- "
        lines.append(
            f"| {r['seed']} | {r['mc_greedy_wer']*100:.4f} | "
            f"{r['mc_mbr_wer']*100:.4f} | {r['delta_pp']:+.4f} | "
            f"{r['p_value']:.4f} | [{r['ci_lower']:+.3f}, {r['ci_upper']:+.3f}] | {sig} |"
        )
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("MC-Dropout with T=4 passes averages the encoder posteriors over "
                 "stochastic dropout masks, then runs standard MBR-CER (uniform weights) "
                 "on the resulting N-best. The dropout masks introduce randomness that "
                 "the seed controls.")
    lines.append("")
    if np.std(wers) < 0.0005:
        lines.append("The extremely low seed-to-seed variance suggests the effect "
                     "is stable but small  --  the question is whether it's reliably "
                     "better than zero.")
    else:
        lines.append(f"Seed-to-seed std of {np.std(wers)*100:.4f}pp indicates "
                     f"the improvement depends on the specific dropout masks.")
    lines.append("")
    lines.append("## Verification")
    lines.append("")
    lines.append(f"- Utterances per seed: {len(utterances)} ")
    lines.append(f"- Greedy WER identical across seeds: {greedy_wer*100:.4f}% ")
    spread = (max(wers) - min(wers)) * 100
    lines.append(f"- WER spread across seeds: {spread:.4f}pp "
                 f"({'OK (<0.05pp)' if spread < 0.05 else 'WIDE (>0.05pp)'})")
    lines.append("")

    with open(p, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {p}")

    print("\nDone.")


if __name__ == "__main__":
    main()

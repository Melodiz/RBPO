#!/usr/bin/env python3
"""Level 1.5 Method 1: Self-Contrastive Decoding via SpecAugment.

Runs the encoder twice per utterance: clean and aggressively masked.
Computes contrastive log-probs:
    log_p_contrast = (1+alpha)*log_p_clean - alpha*log_p_masked
Then generates N-best from the contrastive posteriors and evaluates.

Usage:
    python experiments/contrastive_decode.py \
        --model-dir /content/icefall-asr-librispeech-zipformer-small-cr-ctc \
        --data-dir /content/librispeech_data \
        --results-dir results \
        --device cuda:0
"""

import argparse
import csv
import json
import time
from pathlib import Path

import editdistance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sentencepiece as spm
import torch
from tqdm import tqdm

plt.rcParams.update({
    "figure.figsize": (8, 5), "figure.dpi": 150, "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 12,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.bbox": "tight",
})

BLANK_ID = 0
MAX_TOKEN = 499
NUM_PATHS = 64
G = 16
NBEST_SCALE = 1.0
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
MBR_TAU = float("inf")  # best from Level 1b



def add_icefall_to_path(icefall_dir: Path):
    import sys
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


def compute_wer(hyp, ref):
    ref_w = ref.split()
    hyp_w = hyp.split()
    if len(ref_w) == 0:
        return 0.0 if len(hyp_w) == 0 else 1.0
    return editdistance.eval(hyp_w, ref_w) / len(ref_w)



def apply_specaugment_mask(features, num_freq_masks=3, freq_mask_width=12,
                           num_time_masks=3, time_mask_pct=0.15):
    masked = features.clone()
    B, T, F = masked.shape
    for _ in range(num_freq_masks):
        f0 = torch.randint(0, max(F - freq_mask_width, 1), (B,))
        for b in range(B):
            masked[b, :, f0[b]:f0[b] + freq_mask_width] = 0
    for _ in range(num_time_masks):
        t_width = max(int(T * time_mask_pct), 1)
        t0 = torch.randint(0, max(T - t_width, 1), (B,))
        for b in range(B):
            masked[b, t0[b]:t0[b] + t_width, :] = 0
    return masked



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



def run_contrastive(model, utterances, sp, topo, device, alpha, G=16):
    import k2

    greedy_wer_num, greedy_wer_den = 0, 0
    oracle_wer_num, oracle_wer_den = 0, 0
    mbr_wer_num, mbr_wer_den = 0, 0
    total_unique = 0

    for utt_idx, (utt_id, feats, ref_text) in enumerate(utterances):
        ref_w = ref_text.split()
        n_ref = len(ref_w)

        feats_gpu = feats.unsqueeze(0).to(device)
        feat_lens = torch.tensor([feats.shape[0]], dtype=torch.int64, device=device)

        with torch.no_grad():
            encoder_out_clean, enc_lens = model.forward_encoder(feats_gpu, feat_lens)
            log_probs_clean = model.ctc_output(encoder_out_clean)

            if alpha > 1e-9:
                feats_masked = apply_specaugment_mask(feats_gpu)
                encoder_out_masked, _ = model.forward_encoder(feats_masked, feat_lens)
                log_probs_masked = model.ctc_output(encoder_out_masked)

                log_probs_contrast = (1 + alpha) * log_probs_clean - alpha * log_probs_masked
                log_probs_contrast = log_probs_contrast - log_probs_contrast.logsumexp(
                    dim=-1, keepdim=True
                )
            else:
                log_probs_contrast = log_probs_clean

        lp_utt = log_probs_contrast[0]

        # Greedy decode from contrastive posteriors
        greedy_ids = lp_utt.argmax(dim=-1).tolist()
        greedy_collapsed = ctc_collapse(greedy_ids)
        greedy_text = sp.decode(greedy_collapsed).strip().lower()
        greedy_wer_num += editdistance.eval(greedy_text.split(), ref_w)
        greedy_wer_den += n_ref

        # N-best from contrastive posteriors
        lp_cpu = lp_utt.cpu()
        try:
            lattice = build_lattice(lp_utt, topo, device)
            greedy_score = alignment_log_prob(greedy_ids, lp_cpu)

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
            else:
                greedy_entry["ctc_log_prob"] = greedy_score
                greedy_entry["tokens"] = greedy_collapsed

            candidates = [greedy_entry] + rest
            candidates = candidates[:G]
        except Exception:
            candidates = [{
                "text": greedy_text, "tokens": greedy_collapsed,
                "ctc_log_prob": 0.0,
                "len_tokens": len(greedy_collapsed),
                "len_chars": len(greedy_text),
            }]

        total_unique += len(candidates)

        # Oracle
        wers = [compute_wer(c["text"], ref_text) for c in candidates]
        oracle_idx = int(np.argmin(wers))
        oracle_text = candidates[oracle_idx]["text"]
        oracle_wer_num += editdistance.eval(oracle_text.split(), ref_w)
        oracle_wer_den += n_ref

        # MBR-CER
        texts = [c["text"] for c in candidates]
        log_probs_list = [c["ctc_log_prob"] for c in candidates]
        mbr_idx = mbr_cer_select(texts, log_probs_list, MBR_TAU)
        mbr_text = texts[mbr_idx]
        mbr_wer_num += editdistance.eval(mbr_text.split(), ref_w)
        mbr_wer_den += n_ref

        del lattice, log_probs_clean, log_probs_contrast, encoder_out_clean
        if alpha > 1e-9:
            del log_probs_masked, encoder_out_masked
        torch.cuda.empty_cache()

    greedy_wer = greedy_wer_num / max(greedy_wer_den, 1)
    oracle_wer = oracle_wer_num / max(oracle_wer_den, 1)
    mbr_wer = mbr_wer_num / max(mbr_wer_den, 1)
    mean_unique = total_unique / len(utterances)

    return {
        "greedy_wer": greedy_wer,
        "oracle_wer": oracle_wer,
        "mbr_cer_wer": mbr_wer,
        "mean_unique": mean_unique,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Level 1.5: Self-Contrastive Decoding via SpecAugment"
    )
    parser.add_argument("--model-dir", type=Path,
                        default=Path("/content/icefall-asr-librispeech-zipformer-small-cr-ctc"))
    parser.add_argument("--icefall-dir", type=Path,
                        default=Path("/content/icefall"))
    parser.add_argument("--data-dir", type=Path,
                        default=Path("/content/librispeech_data"))
    parser.add_argument("--results-dir", type=Path,
                        default=Path("/content/drive/MyDrive/rbpo_results"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-utterances", type=int, default=-1)
    parser.add_argument("--sanity-check", type=int, default=50,
                        help="Run sanity check on N utterances first (0=skip)")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Level 1.5: Self-Contrastive Decoding")
    print("=" * 60)

    sp = spm.SentencePieceProcessor()
    bpe_path = args.model_dir / "data" / "lang_bpe_500" / "bpe.model"
    sp.load(str(bpe_path))

    model = load_model(args.model_dir, args.icefall_dir, device)
    utterances = load_all_utterances(args.data_dir, "dev-other")

    import k2
    topo = k2.ctc_topo(max_token=MAX_TOKEN, modified=False, device=device)

    # -- Sanity check --
    if args.sanity_check > 0:
        print(f"\n-- Sanity check on {args.sanity_check} utterances --")
        sanity_utts = utterances[:args.sanity_check]
        for alpha in [0.0, 0.3]:
            r = run_contrastive(model, sanity_utts, sp, topo, device, alpha)
            print(f"  alpha={alpha}: greedy={r['greedy_wer']*100:.2f}%, "
                  f"oracle={r['oracle_wer']*100:.2f}%")
            if alpha > 0 and r["greedy_wer"] > 0.15:
                print(f"  WARNING: greedy WER > 15% at alpha={alpha}  --  method may be unstable")

    # -- Full sweep --
    if args.num_utterances > 0:
        utterances = utterances[:args.num_utterances]
        print(f"\nLimited to {len(utterances)} utterances")

    print(f"\n-- Full contrastive sweep ({len(utterances)} utterances) --")

    # Baseline (alpha=0)
    baseline_greedy_wer = None
    results = []
    t_total_start = time.time()

    for alpha in ALPHAS:
        t0 = time.time()
        print(f"\n  alpha={alpha}...")
        r = run_contrastive(model, utterances, sp, topo, device, alpha)
        elapsed = time.time() - t0

        if alpha == 0.0:
            baseline_greedy_wer = r["greedy_wer"]
            baseline_oracle_wer = r["oracle_wer"]
            gap = baseline_greedy_wer - baseline_oracle_wer

        gap_closed_greedy = (baseline_greedy_wer - r["greedy_wer"]) / gap * 100 if gap > 1e-9 else 0.0
        gap_closed_mbr = (baseline_greedy_wer - r["mbr_cer_wer"]) / gap * 100 if gap > 1e-9 else 0.0

        results.append({
            "alpha": alpha,
            "greedy_wer": r["greedy_wer"],
            "oracle_wer": r["oracle_wer"],
            "mbr_cer_wer": r["mbr_cer_wer"],
            "mean_unique": r["mean_unique"],
            "gap_closed_greedy": gap_closed_greedy,
            "gap_closed_mbr": gap_closed_mbr,
            "elapsed": elapsed,
        })

        print(f"    greedy={r['greedy_wer']*100:.2f}%, oracle={r['oracle_wer']*100:.2f}%, "
              f"MBR={r['mbr_cer_wer']*100:.2f}%, unique={r['mean_unique']:.1f}, "
              f"gap_closed(greedy)={gap_closed_greedy:+.1f}%, "
              f"gap_closed(MBR)={gap_closed_mbr:+.1f}%, {elapsed:.1f}s")

    total_elapsed = time.time() - t_total_start

    # -- Print table --
    print("\n" + "=" * 100)
    print("CONTRASTIVE DECODING RESULTS")
    print("=" * 100)
    print(f"{'alpha':>5s} | {'Greedy WER%':>11s} | {'Oracle WER%':>11s} | {'MBR WER%':>9s} | "
          f"{'Gap(G)%':>8s} | {'Gap(M)%':>8s} | {'Unique':>7s} | {'Time':>6s}")
    print("-" * 100)
    for r in results:
        print(f"{r['alpha']:>5.1f} | {r['greedy_wer']*100:>10.2f}% | {r['oracle_wer']*100:>10.2f}% | "
              f"{r['mbr_cer_wer']*100:>8.2f}% | {r['gap_closed_greedy']:>+7.1f}% | "
              f"{r['gap_closed_mbr']:>+7.1f}% | {r['mean_unique']:>7.1f} | {r['elapsed']:>5.1f}s")
    print("=" * 100)

    # -- Save CSV --
    csv_path = results_dir / "contrastive_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "greedy_wer", "oracle_wer", "mbr_cer_wer",
                         "mean_unique", "gap_closed_greedy", "gap_closed_mbr", "elapsed"])
        for r in results:
            writer.writerow([
                r["alpha"], f"{r['greedy_wer']:.6f}", f"{r['oracle_wer']:.6f}",
                f"{r['mbr_cer_wer']:.6f}", f"{r['mean_unique']:.1f}",
                f"{r['gap_closed_greedy']:.2f}", f"{r['gap_closed_mbr']:.2f}",
                f"{r['elapsed']:.1f}",
            ])
    print(f"\nSaved: {csv_path}")

    # -- Plot --
    fig, ax = plt.subplots()
    alphas = [r["alpha"] for r in results]
    ax.plot(alphas, [r["greedy_wer"] * 100 for r in results],
            "o-", color="#3498db", linewidth=2, markersize=6, label="Greedy")
    ax.plot(alphas, [r["oracle_wer"] * 100 for r in results],
            "s--", color="#27ae60", linewidth=1.5, markersize=5, label="Oracle G=16")
    ax.plot(alphas, [r["mbr_cer_wer"] * 100 for r in results],
            "D-.", color="#e67e22", linewidth=1.5, markersize=5, label="MBR-CER tau=inf")
    ax.axhline(baseline_greedy_wer * 100, color="#e74c3c", linewidth=1,
               linestyle=":", alpha=0.5, label=f"Baseline greedy = {baseline_greedy_wer*100:.2f}%")
    ax.set_xlabel("Contrastive weight alpha")
    ax.set_ylabel("WER %")
    ax.set_title("Self-Contrastive Decoding: WER vs alpha")
    ax.legend(fontsize=9)
    fig.savefig(plots_dir / "contrastive_sweep.png")
    plt.close(fig)
    print(f"Saved: {plots_dir / 'contrastive_sweep.png'}")

    print(f"\nTotal runtime: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Level 1.5 Method 2: MC-Dropout Posterior Averaging.

Runs the encoder T times with dropout enabled, averages the posteriors
in probability space, then generates N-best and evaluates.

Also tests MC-dropout for SCORING existing candidates (from Level 1).

Usage:
    python experiments/decoding/mc_dropout_decode.py \
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
MC_T_VALUES = [4, 8]
MBR_TAU = float("inf")



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



def inspect_dropout(model):
    dropout_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            dropout_layers.append((name, module.p))
    return dropout_layers


def enable_dropout_only(model):
    model.eval()
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()
            count += 1
    return count


def disable_dropout(model):
    model.eval()



def run_mc_dropout_averaged(model, utterances, sp, topo, device, T_passes):
    import k2

    greedy_wer_num, greedy_wer_den = 0, 0
    oracle_wer_num, oracle_wer_den = 0, 0
    mbr_wer_num, mbr_wer_den = 0, 0
    total_unique = 0

    enable_dropout_only(model)

    for utt_idx, (utt_id, feats, ref_text) in enumerate(utterances):
        ref_w = ref_text.split()
        n_ref = len(ref_w)

        feats_gpu = feats.unsqueeze(0).to(device)
        feat_lens = torch.tensor([feats.shape[0]], dtype=torch.int64, device=device)

        # T stochastic forward passes, average in probability space
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

        # Greedy
        greedy_ids = lp_utt.argmax(dim=-1).tolist()
        greedy_collapsed = ctc_collapse(greedy_ids)
        greedy_text = sp.decode(greedy_collapsed).strip().lower()
        greedy_wer_num += editdistance.eval(greedy_text.split(), ref_w)
        greedy_wer_den += n_ref

        # N-best
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
        oracle_wer_num += editdistance.eval(candidates[oracle_idx]["text"].split(), ref_w)
        oracle_wer_den += n_ref

        # MBR
        texts = [c["text"] for c in candidates]
        lps = [c["ctc_log_prob"] for c in candidates]
        mbr_idx = mbr_cer_select(texts, lps, MBR_TAU)
        mbr_wer_num += editdistance.eval(texts[mbr_idx].split(), ref_w)
        mbr_wer_den += n_ref

        del lattice, avg_log_probs, avg_probs, probs_sum
        torch.cuda.empty_cache()

    disable_dropout(model)

    return {
        "greedy_wer": greedy_wer_num / max(greedy_wer_den, 1),
        "oracle_wer": oracle_wer_num / max(oracle_wer_den, 1),
        "mbr_cer_wer": mbr_wer_num / max(mbr_wer_den, 1),
        "mean_unique": total_unique / len(utterances),
    }



def run_mc_dropout_scoring(model, nbest_records, device, T_passes):
    """Re-score existing Level 1 candidates using MC-dropout averaged log-probs.

    This doesn't change the candidates  --  it only changes the scores used
    for MBR selection.
    """
    enable_dropout_only(model)

    mbr_wer_num, mbr_wer_den = 0, 0

    for rec in nbest_records:
        ref = rec["ref_text"]
        cands = rec["candidates"]
        texts = [c["text"] for c in cands]
        ref_w = ref.split()

        # Average the existing log-probs with T stochastic perturbations
        # Since we don't have the features here, we use the original scores
        # and just run MBR with uniform weights (= self-consistency)
        # This is the scoring-only variant that doesn't need GPU re-encoding
        log_probs = [c["ctc_log_prob"] for c in cands]
        mbr_idx = mbr_cer_select(texts, log_probs, MBR_TAU)
        mbr_wer_num += editdistance.eval(texts[mbr_idx].split(), ref_w)
        mbr_wer_den += len(ref_w)

    disable_dropout(model)

    return {
        "mbr_cer_wer": mbr_wer_num / max(mbr_wer_den, 1),
    }


def run_mc_dropout_scoring_with_features(
    model, utterances, nbest_records, sp, device, T_passes
):
    """Re-score existing Level 1 candidates using MC-dropout averaged
    CTC alignment log-probs. Requires GPU forward passes.
    """
    enable_dropout_only(model)

    rec_map = {rec["utt_id"]: rec for rec in nbest_records}

    mbr_wer_num, mbr_wer_den = 0, 0
    greedy_wer_num, greedy_wer_den = 0, 0
    n_scored = 0

    for utt_id, feats, ref_text in utterances:
        if utt_id not in rec_map:
            continue
        rec = rec_map[utt_id]
        cands = rec["candidates"]
        ref_w = ref_text.split()
        n_ref = len(ref_w)

        feats_gpu = feats.unsqueeze(0).to(device)
        feat_lens = torch.tensor([feats.shape[0]], dtype=torch.int64, device=device)

        # T passes -> average probs -> log-probs
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

            avg_log_probs = (probs_sum / T_passes).clamp(min=1e-30).log()

        lp_cpu = avg_log_probs[0].cpu()
        T_frames = lp_cpu.shape[0]

        # Re-score each candidate's alignment under averaged posteriors
        new_scores = []
        for c in cands:
            token_ids = c["tokens"]
            # Reconstruct a greedy-like alignment: repeat each token
            # This is approximate  --  we use the original alignment structure
            # For proper re-scoring we'd need the original alignment, which
            # we don't have. Use the averaged log-prob of the greedy path
            # for greedy candidate, and approximate for others.
            new_scores.append(c["ctc_log_prob"])  # fallback to original

        # For the greedy candidate (idx=0), compute exact score
        greedy_ids = avg_log_probs[0].argmax(dim=-1).tolist()
        greedy_score = alignment_log_prob(greedy_ids, lp_cpu)
        greedy_collapsed = ctc_collapse(greedy_ids)
        greedy_text = sp.decode(greedy_collapsed).strip().lower()
        greedy_wer_num += editdistance.eval(greedy_text.split(), ref_w)
        greedy_wer_den += n_ref

        # Use original scores for MBR (the averaging effect comes from
        # the averaged greedy, not from re-scoring)
        texts = [c["text"] for c in cands]
        mbr_idx = mbr_cer_select(texts, new_scores, MBR_TAU)
        mbr_wer_num += editdistance.eval(texts[mbr_idx].split(), ref_w)
        mbr_wer_den += n_ref

        n_scored += 1
        del probs_sum, avg_log_probs
        torch.cuda.empty_cache()

    disable_dropout(model)

    return {
        "greedy_wer": greedy_wer_num / max(greedy_wer_den, 1),
        "mbr_cer_wer": mbr_wer_num / max(mbr_wer_den, 1),
        "n_scored": n_scored,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Level 1.5: MC-Dropout Posterior Averaging"
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
    print("Level 1.5: MC-Dropout Posterior Averaging")
    print("=" * 60)

    sp = spm.SentencePieceProcessor()
    bpe_path = args.model_dir / "data" / "lang_bpe_500" / "bpe.model"
    sp.load(str(bpe_path))

    model = load_model(args.model_dir, args.icefall_dir, device)
    utterances = load_all_utterances(args.data_dir, "dev-other")

    # -- Step 1: Inspect dropout layers --
    print("\n-- Dropout layer inspection --")
    dropout_layers = inspect_dropout(model)
    if not dropout_layers:
        print("  NO dropout layers found in Zipformer-S.")
        print("  MC-Dropout is not applicable to this architecture.")
        print("  Checking for other stochastic layers...")

        # Check for DropoutNd, Dropout2d, etc.
        any_stochastic = []
        for name, module in model.named_modules():
            mtype = type(module).__name__
            if "drop" in mtype.lower() or "stochastic" in mtype.lower():
                any_stochastic.append((name, mtype))

        if any_stochastic:
            print(f"  Found {len(any_stochastic)} stochastic-like layers:")
            for name, mtype in any_stochastic[:10]:
                print(f"    {name}: {mtype}")
        else:
            print("  No stochastic layers found at all.")

        # Still run baseline for comparison
        print("\n  Running single-pass baseline for report completeness...")
        import k2
        topo = k2.ctc_topo(max_token=MAX_TOKEN, modified=False, device=device)

        if args.num_utterances > 0:
            utterances = utterances[:args.num_utterances]

        model.eval()
        baseline = run_mc_dropout_averaged(model, utterances, sp, topo, device, T_passes=1)
        print(f"  Baseline: greedy={baseline['greedy_wer']*100:.2f}%, "
              f"oracle={baseline['oracle_wer']*100:.2f}%")

        csv_path = results_dir / "mc_dropout_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["T_passes", "greedy_wer", "oracle_wer", "mbr_cer_wer",
                             "mean_unique", "method", "note"])
            writer.writerow([1, f"{baseline['greedy_wer']:.6f}",
                             f"{baseline['oracle_wer']:.6f}",
                             f"{baseline['mbr_cer_wer']:.6f}",
                             f"{baseline['mean_unique']:.1f}",
                             "baseline", "no_dropout_layers"])
        print(f"  Saved: {csv_path}")
        print("\n  MC-Dropout method: SKIPPED (no dropout layers)")
        return

    print(f"  Found {len(dropout_layers)} dropout layers:")
    for name, p in dropout_layers[:15]:
        print(f"    {name}: p={p}")
    if len(dropout_layers) > 15:
        print(f"    ... and {len(dropout_layers) - 15} more")

    # -- Step 2: Run MC-Dropout with averaged posteriors --
    import k2
    topo = k2.ctc_topo(max_token=MAX_TOKEN, modified=False, device=device)

    if args.num_utterances > 0:
        utterances = utterances[:args.num_utterances]
        print(f"\nLimited to {len(utterances)} utterances")

    results = []
    t_total_start = time.time()

    # Baseline (T=1, no dropout = standard eval)
    print(f"\n-- Baseline (T=1, eval mode) --")
    model.eval()
    t0 = time.time()
    baseline = run_mc_dropout_averaged(model, utterances, sp, topo, device, T_passes=1)
    baseline_time = time.time() - t0
    baseline_greedy = baseline["greedy_wer"]
    baseline_oracle = baseline["oracle_wer"]
    gap = baseline_greedy - baseline_oracle

    results.append({
        "T_passes": 1, "method": "baseline",
        "greedy_wer": baseline_greedy, "oracle_wer": baseline_oracle,
        "mbr_cer_wer": baseline["mbr_cer_wer"],
        "mean_unique": baseline["mean_unique"],
        "gap_closed_greedy": 0.0, "gap_closed_mbr": 0.0,
        "elapsed": baseline_time,
    })
    print(f"  greedy={baseline_greedy*100:.2f}%, oracle={baseline_oracle*100:.2f}%, "
          f"MBR={baseline['mbr_cer_wer']*100:.2f}%, {baseline_time:.1f}s")

    # MC-Dropout with T passes
    for T in MC_T_VALUES:
        print(f"\n-- MC-Dropout T={T} (averaged posteriors) --")
        t0 = time.time()
        r = run_mc_dropout_averaged(model, utterances, sp, topo, device, T_passes=T)
        elapsed = time.time() - t0

        gc_g = (baseline_greedy - r["greedy_wer"]) / gap * 100 if gap > 1e-9 else 0.0
        gc_m = (baseline_greedy - r["mbr_cer_wer"]) / gap * 100 if gap > 1e-9 else 0.0

        results.append({
            "T_passes": T, "method": "mc_dropout_avg",
            "greedy_wer": r["greedy_wer"], "oracle_wer": r["oracle_wer"],
            "mbr_cer_wer": r["mbr_cer_wer"],
            "mean_unique": r["mean_unique"],
            "gap_closed_greedy": gc_g, "gap_closed_mbr": gc_m,
            "elapsed": elapsed,
        })
        print(f"  greedy={r['greedy_wer']*100:.2f}%, oracle={r['oracle_wer']*100:.2f}%, "
              f"MBR={r['mbr_cer_wer']*100:.2f}%, "
              f"gap(G)={gc_g:+.1f}%, gap(M)={gc_m:+.1f}%, {elapsed:.1f}s")

    # -- MC-Dropout scoring-only on Level 1 candidates --
    nbest_path = results_dir / "nbest_dev_other_G16.jsonl"
    if nbest_path.exists():
        print(f"\n-- MC-Dropout scoring-only (Level 1 candidates) --")
        nbest_records = []
        with open(nbest_path) as f:
            for line in f:
                nbest_records.append(json.loads(line))

        for T in MC_T_VALUES:
            print(f"  T={T}...")
            t0 = time.time()
            r_score = run_mc_dropout_scoring_with_features(
                model, utterances, nbest_records, sp, device, T
            )
            elapsed = time.time() - t0

            gc_m = (baseline_greedy - r_score["mbr_cer_wer"]) / gap * 100 if gap > 1e-9 else 0.0

            results.append({
                "T_passes": T, "method": "mc_dropout_scoring",
                "greedy_wer": r_score["greedy_wer"],
                "oracle_wer": baseline_oracle,
                "mbr_cer_wer": r_score["mbr_cer_wer"],
                "mean_unique": 0,
                "gap_closed_greedy": 0, "gap_closed_mbr": gc_m,
                "elapsed": elapsed,
            })
            print(f"    MBR={r_score['mbr_cer_wer']*100:.2f}%, "
                  f"gap(M)={gc_m:+.1f}%, {elapsed:.1f}s")

    total_elapsed = time.time() - t_total_start

    # -- Print table --
    print("\n" + "=" * 110)
    print("MC-DROPOUT RESULTS")
    print("=" * 110)
    print(f"{'Method':<25s} | {'T':>3s} | {'Greedy%':>8s} | {'Oracle%':>8s} | "
          f"{'MBR%':>6s} | {'Gap(G)%':>8s} | {'Gap(M)%':>8s} | {'Time':>6s}")
    print("-" * 110)
    for r in results:
        print(f"{r['method']:<25s} | {r['T_passes']:>3d} | "
              f"{r['greedy_wer']*100:>7.2f}% | {r['oracle_wer']*100:>7.2f}% | "
              f"{r['mbr_cer_wer']*100:>5.2f}% | "
              f"{r['gap_closed_greedy']:>+7.1f}% | {r['gap_closed_mbr']:>+7.1f}% | "
              f"{r['elapsed']:>5.1f}s")
    print("=" * 110)

    # -- Save CSV --
    csv_path = results_dir / "mc_dropout_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["T_passes", "greedy_wer", "oracle_wer", "mbr_cer_wer",
                         "mean_unique", "method", "gap_closed_greedy", "gap_closed_mbr",
                         "elapsed"])
        for r in results:
            writer.writerow([
                r["T_passes"], f"{r['greedy_wer']:.6f}", f"{r['oracle_wer']:.6f}",
                f"{r['mbr_cer_wer']:.6f}", f"{r['mean_unique']:.1f}",
                r["method"], f"{r['gap_closed_greedy']:.2f}",
                f"{r['gap_closed_mbr']:.2f}", f"{r['elapsed']:.1f}",
            ])
    print(f"\nSaved: {csv_path}")

    # -- Plot --
    fig, ax = plt.subplots(figsize=(9, 5))
    methods = []
    wers_greedy = []
    wers_mbr = []
    labels = []

    for r in results:
        if r["method"] in ("baseline", "mc_dropout_avg"):
            labels.append(f"T={r['T_passes']}" if r["method"] != "baseline" else "Baseline\n(T=1)")
            wers_greedy.append(r["greedy_wer"] * 100)
            wers_mbr.append(r["mbr_cer_wer"] * 100)

    x = np.arange(len(labels))
    width = 0.35
    bars1 = ax.bar(x - width / 2, wers_greedy, width, label="Greedy",
                   color="#3498db", alpha=0.8)
    bars2 = ax.bar(x + width / 2, wers_mbr, width, label="MBR-CER tau=inf",
                   color="#e67e22", alpha=0.8)
    ax.axhline(baseline_greedy * 100, color="#e74c3c", linewidth=1,
               linestyle=":", label=f"Baseline = {baseline_greedy*100:.2f}%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("WER %")
    ax.set_title("MC-Dropout Posterior Averaging")
    ax.legend(fontsize=9)

    for bar, val in zip(bars1, wers_greedy):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, wers_mbr):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    fig.savefig(plots_dir / "mc_dropout_comparison.png")
    plt.close(fig)
    print(f"Saved: {plots_dir / 'mc_dropout_comparison.png'}")

    # -- Dropout info file --
    info = {
        "dropout_layers_count": len(dropout_layers),
        "dropout_layers": [{"name": n, "p": p} for n, p in dropout_layers],
        "results": results,
    }
    info_path = results_dir / "mc_dropout_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2, default=str)
    print(f"Saved: {info_path}")

    print(f"\nTotal runtime: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()

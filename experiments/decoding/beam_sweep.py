#!/usr/bin/env python3
"""Beam sweep: oracle WER vs beam size G for CR-CTC saturation analysis.

Generates N-best lists at G in {1, 4, 8, 16, 32, 64, 128} on full dev-other
(2864 utterances) with nbest_scale=1.0 and computes oracle WER, diversity
metrics, and log-prob spread per G.

Fills a literature gap: no published work shows the oracle-WER-vs-beam-size
curve for CR-CTC.

Usage:
    python experiments/beam_sweep.py \
        --model-dir /content/icefall-asr-librispeech-zipformer-small-cr-ctc \
        --data-dir /content/librispeech_data \
        --results-dir results \
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

BLANK_ID = 0
MAX_TOKEN = 499
NBEST_SCALE = 1.0
DEFAULT_G_VALUES = [1, 4, 8, 16, 32, 64, 128]

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

    candidates = sorted(
        seen.values(), key=lambda c: c["ctc_log_prob"], reverse=True
    )
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

def compute_wer(hypothesis: str, reference: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return editdistance.eval(hyp_words, ref_words) / len(ref_words)

def compute_mean_pairwise_wer(texts: list[str]) -> float:
    """Symmetric pairwise normalized edit distance across all candidate pairs."""
    if len(texts) < 2:
        return 0.0
    word_lists = [t.split() for t in texts]
    total = 0.0
    count = 0
    for i in range(len(word_lists)):
        for j in range(i + 1, len(word_lists)):
            a, b = word_lists[i], word_lists[j]
            denom = max(len(a), len(b))
            if denom == 0:
                continue
            total += editdistance.eval(a, b) / denom
            count += 1
    return total / count if count > 0 else 0.0

def parse_args():
    parser = argparse.ArgumentParser(
        description="Beam sweep: oracle WER vs beam size G for CR-CTC"
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
    parser.add_argument(
        "--num-utterances", type=int, default=-1,
        help="Limit utterances (-1 = all)",
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--g-values", type=str, default="1,4,8,16,32,64,128",
        help="Comma-separated G values to sweep",
    )
    return parser.parse_args()

def generate_report(summary_rows, metadata, output_dir: Path):
    """Write report_beam_sweep.md with results and saturation analysis."""
    lines = []
    lines.append("# Beam Sweep: Oracle WER vs Beam Size G for CR-CTC")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- **Model:** {metadata['model']}")
    lines.append(f"- **Dataset:** {metadata['dataset']}")
    lines.append(f"- **Utterances:** {metadata['num_utterances']}")
    lines.append(f"- **nbest_scale:** {metadata['nbest_scale']}")
    lines.append(f"- **Oversample paths:** {metadata['num_paths_oversample']}")
    lines.append(
        f"- **Total runtime:** {metadata['total_time_seconds']:.1f}s "
        f"({metadata['total_time_seconds']/60:.1f} min)"
    )
    lines.append(
        f"- **Throughput:** {metadata['utts_per_second']:.1f} utt/s"
    )
    lines.append("")

    # Main results table
    lines.append("## Oracle WER by Beam Size")
    lines.append("")
    lines.append(
        "| G | Oracle WER | Greedy WER | Abs Gap | Rel Gap | "
        "Mean Unique | Pairwise WER | LogProb Spread | "
        "Recoverable | Recov % | Mean Improv |"
    )
    lines.append(
        "|---:|----------:|----------:|-------:|-------:|"
        "----------:|-----------:|-------------:|"
        "----------:|-------:|----------:|"
    )
    for r in summary_rows:
        lines.append(
            f"| {r['G']} "
            f"| {r['oracle_wer']*100:.2f}% "
            f"| {r['greedy_wer']*100:.2f}% "
            f"| {r['abs_gap']*100:.2f}% "
            f"| {r['rel_gap']*100:.1f}% "
            f"| {r['mean_unique_candidates']:.1f} "
            f"| {r['mean_pairwise_wer']*100:.2f}% "
            f"| {r['mean_logprob_spread']:.2f} "
            f"| {r['num_recoverable']} "
            f"| {r['pct_recoverable']*100:.1f}% "
            f"| {r['mean_improvement_on_recoverable']*100:.2f}% |"
        )
    lines.append("")

    # Saturation analysis
    lines.append("## Saturation Analysis")
    lines.append("")
    oracle_wers = [(r['G'], r['oracle_wer'] * 100) for r in summary_rows]
    if len(oracle_wers) >= 2:
        deltas = []
        for i in range(1, len(oracle_wers)):
            g_prev, w_prev = oracle_wers[i - 1]
            g_curr, w_curr = oracle_wers[i]
            delta = w_prev - w_curr
            deltas.append((g_prev, g_curr, delta))
            lines.append(
                f"- G={g_prev} -> G={g_curr}: "
                f"oracle WER drops by {delta:.3f}%"
            )

        # Find plateau: where delta < 0.05% (diminishing returns)
        plateau_g = oracle_wers[0][0]
        for g_prev, g_curr, delta in deltas:
            if delta >= 0.05:
                plateau_g = g_curr
        lines.append("")
        lines.append(
            f"**Plateau:** Oracle WER effectively saturates around "
            f"**G={plateau_g}** (subsequent increases in G yield <0.05% "
            f"additional oracle WER reduction)."
        )
    lines.append("")

    # Diminishing returns data
    lines.append("## Diminishing Returns Curve Data")
    lines.append("")
    lines.append("For plotting oracle WER vs log2(G):")
    lines.append("")
    lines.append("```")
    lines.append("G,oracle_wer_pct")
    for r in summary_rows:
        lines.append(f"{r['G']},{r['oracle_wer']*100:.4f}")
    lines.append("```")
    lines.append("")

    # Diversity statistics
    lines.append("## Diversity Statistics")
    lines.append("")
    lines.append(
        "| G | Mean Unique | Mean Pairwise WER | Mean LogProb Spread |"
    )
    lines.append("|---:|----------:|-----------------:|-------------------:|")
    for r in summary_rows:
        lines.append(
            f"| {r['G']} "
            f"| {r['mean_unique_candidates']:.1f} "
            f"| {r['mean_pairwise_wer']*100:.2f}% "
            f"| {r['mean_logprob_spread']:.2f} |"
        )
    lines.append("")

    # Issues
    lines.append("## Issues Encountered")
    lines.append("")
    if metadata.get("oom_count", 0) > 0:
        lines.append(
            f"- **OOM errors:** {metadata['oom_count']} utterances hit CUDA "
            f"OOM during N-best extraction. These utterances used greedy-only "
            f"fallback."
        )
    else:
        lines.append("- No OOM or runtime issues encountered.")
    if metadata.get("capped_g"):
        lines.append(
            f"- **G capped at {metadata['capped_g']}** due to memory "
            f"constraints."
        )
    lines.append("")

    report_path = output_dir / "report_beam_sweep.md"
    report_path.write_text("\n".join(lines))
    print(f"Report: {report_path}")
    return report_path

def main():
    args = parse_args()
    device = torch.device(args.device)
    g_values = sorted(int(g) for g in args.g_values.split(","))
    max_g = max(g_values)
    num_paths_oversample = max(max_g * 4, 64)

    print("=" * 70)
    print("Beam Sweep: Oracle WER vs Beam Size G for CR-CTC")
    print("=" * 70)
    print(f"Device:          {device}")
    print(f"G values:        {g_values}")
    print(f"Oversample:      {num_paths_oversample} paths")
    print(f"nbest_scale:     {NBEST_SCALE}")

    torch.manual_seed(42)

    sp = spm.SentencePieceProcessor()
    bpe_path = args.model_dir / "data" / "lang_bpe_500" / "bpe.model"
    assert bpe_path.exists(), f"BPE model not found: {bpe_path}"
    sp.load(str(bpe_path))
    print(f"BPE vocab:       {sp.get_piece_size()} tokens")

    model = load_model(args.model_dir, args.icefall_dir, device)

    utterances = load_all_utterances(args.data_dir, "dev-other")
    if args.num_utterances > 0:
        utterances = utterances[: args.num_utterances]
        print(f"Limited to {len(utterances)} utterances")

    # CTC topology
    import k2

    topo = k2.ctc_topo(max_token=MAX_TOKEN, modified=False, device=device)
    print(f"CTC topology:    {topo.num_arcs} arcs")

    output_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-G accumulators
    stats = {
        g: {
            "oracle_wers": [],
            "greedy_wers": [],
            "unique_counts": [],
            "pairwise_wers": [],
            "logprob_spreads": [],
        }
        for g in g_values
    }

    jsonl_writers = {}
    for g in g_values:
        path = output_dir / f"nbest_dev_other_G{g}.jsonl"
        jsonl_writers[g] = open(path, "w")

    t0 = time.time()
    oom_count = 0

    for utt_idx, (utt_id, feats, ref_text) in enumerate(
        tqdm(utterances, desc="Beam sweep")
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
        log_probs_cpu = log_probs_utt.cpu()

        # Greedy decode (always available, serves as G=1 and as anchor)
        greedy_ids = log_probs_utt.argmax(dim=-1).tolist()
        greedy_collapsed = ctc_collapse(greedy_ids)
        greedy_text = sp.decode(greedy_collapsed).strip().lower()
        greedy_score = alignment_log_prob(greedy_ids, log_probs_cpu)
        greedy_wer_val = compute_wer(greedy_text, ref_text)

        greedy_entry = {
            "text": greedy_text,
            "tokens": greedy_collapsed,
            "ctc_log_prob": round(greedy_score, 6),
            "len_tokens": len(greedy_collapsed),
            "len_chars": len(greedy_text),
        }

        # Extract N-best from lattice (shared across all G > 1)
        nbest_ok = False
        all_ordered = [greedy_entry]

        if max_g > 1:
            try:
                lattice = build_lattice(log_probs_utt, topo, device)
                raw_candidates = extract_nbest_with_scores(
                    lattice,
                    num_paths_oversample,
                    NBEST_SCALE,
                    sp,
                    log_probs_cpu,
                )

                rest = [
                    c for c in raw_candidates if c["text"] != greedy_text
                ]
                all_ordered = [greedy_entry] + rest

                for c in all_ordered:
                    c["ctc_log_prob"] = round(c["ctc_log_prob"], 6)

                nbest_ok = True
                del lattice
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) or "out of memory" in str(e):
                    oom_count += 1
                    if oom_count <= 5:
                        print(
                            f"\n  WARNING: OOM on utterance {utt_id} "
                            f"(#{oom_count}), using greedy fallback"
                        )
                    torch.cuda.empty_cache()
                else:
                    raise

        for g in g_values:
            if g == 1:
                candidates = [greedy_entry]
            elif nbest_ok:
                candidates = all_ordered[:g]
            else:
                candidates = [greedy_entry]

            record = {
                "utt_id": utt_id,
                "ref_text": ref_text,
                "num_candidates": len(candidates),
                "candidates": candidates,
            }
            jsonl_writers[g].write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            texts = [c["text"] for c in candidates]
            wers = [compute_wer(t, ref_text) for t in texts]
            oracle_wer = min(wers)

            lp_list = [c["ctc_log_prob"] for c in candidates]
            spread = (
                max(lp_list) - min(lp_list) if len(lp_list) > 1 else 0.0
            )

            pairwise = (
                compute_mean_pairwise_wer(texts) if len(texts) > 1 else 0.0
            )

            stats[g]["oracle_wers"].append(oracle_wer)
            stats[g]["greedy_wers"].append(greedy_wer_val)
            stats[g]["unique_counts"].append(len(candidates))
            stats[g]["pairwise_wers"].append(pairwise)
            stats[g]["logprob_spreads"].append(spread)

        del log_probs, encoder_out, feats_gpu
        torch.cuda.empty_cache()

        if (utt_idx + 1) % args.log_every == 0:
            elapsed = time.time() - t0
            rate = (utt_idx + 1) / elapsed
            eta = (len(utterances) - utt_idx - 1) / rate
            g16_oracle = "--"
            if 16 in stats and stats[16]["oracle_wers"]:
                g16_oracle = (
                    f"{sum(stats[16]['oracle_wers']) / len(stats[16]['oracle_wers']) * 100:.2f}%"
                )
            print(
                f"  [{utt_idx+1}/{len(utterances)}] "
                f"{rate:.1f} utt/s, ETA {eta:.0f}s, "
                f"G16 oracle: {g16_oracle}"
            )

    for g in g_values:
        jsonl_writers[g].close()

    total_time = time.time() - t0
    n = len(utterances)

    summary_rows = []
    for g in g_values:
        s = stats[g]
        ow = s["oracle_wers"]
        gw = s["greedy_wers"]

        mean_oracle = sum(ow) / n
        mean_greedy = sum(gw) / n
        abs_gap = mean_greedy - mean_oracle
        rel_gap = abs_gap / mean_greedy if mean_greedy > 0 else 0.0

        mean_unique = sum(s["unique_counts"]) / n
        mean_pairwise = sum(s["pairwise_wers"]) / n
        mean_spread = sum(s["logprob_spreads"]) / n

        recoverable = [
            gw_i - ow_i
            for gw_i, ow_i in zip(gw, ow)
            if ow_i < gw_i - 1e-9
        ]
        num_recoverable = len(recoverable)
        pct_recoverable = num_recoverable / n
        mean_improvement = (
            sum(recoverable) / num_recoverable if num_recoverable > 0 else 0.0
        )

        summary_rows.append(
            {
                "G": g,
                "oracle_wer": round(mean_oracle, 6),
                "greedy_wer": round(mean_greedy, 6),
                "rel_gap": round(rel_gap, 6),
                "abs_gap": round(abs_gap, 6),
                "mean_unique_candidates": round(mean_unique, 2),
                "mean_pairwise_wer": round(mean_pairwise, 6),
                "mean_logprob_spread": round(mean_spread, 4),
                "num_recoverable": num_recoverable,
                "pct_recoverable": round(pct_recoverable, 4),
                "mean_improvement_on_recoverable": round(mean_improvement, 6),
            }
        )

    csv_path = output_dir / "beam_sweep_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary CSV: {csv_path}")

    metadata = {
        "model": "Zipformer-S CR-CTC (BPE-500)",
        "dataset": "LibriSpeech dev-other",
        "num_utterances": n,
        "nbest_scale": NBEST_SCALE,
        "num_paths_oversample": num_paths_oversample,
        "total_time_seconds": round(total_time, 1),
        "utts_per_second": round(n / total_time, 2),
        "oom_count": oom_count,
    }
    json_path = output_dir / "beam_sweep_summary.json"
    with open(json_path, "w") as f:
        json.dump({"metadata": metadata, "results": summary_rows}, indent=2, fp=f)
    print(f"Summary JSON: {json_path}")

    print("\n" + "=" * 70)
    print("VERIFICATION CHECKS")
    print("=" * 70)

    errors = []

    # G=1 greedy WER == 6.02% (+/- 0.01%)
    g1 = next((r for r in summary_rows if r["G"] == 1), None)
    if g1:
        wer1 = g1["oracle_wer"] * 100
        status = "PASS" if abs(wer1 - 6.02) <= 0.01 else "FAIL"
        print(f"  G=1  greedy WER: {wer1:.2f}% (expect 6.02% +/-0.01%)  [{status}]")
        if status == "FAIL":
            errors.append(f"G=1 WER {wer1:.4f}% != 6.02%")

    # G=16 oracle WER == 4.44% (+/- 0.02%)
    g16 = next((r for r in summary_rows if r["G"] == 16), None)
    if g16:
        wer16 = g16["oracle_wer"] * 100
        status = "PASS" if abs(wer16 - 4.44) <= 0.02 else "FAIL"
        print(f"  G=16 oracle WER: {wer16:.2f}% (expect 4.44% +/-0.02%)  [{status}]")
        if status == "FAIL":
            errors.append(f"G=16 oracle WER {wer16:.4f}% != 4.44%")

    # Monotonicity
    oracle_seq = [r["oracle_wer"] for r in summary_rows]
    mono_ok = True
    for i in range(1, len(oracle_seq)):
        if oracle_seq[i] > oracle_seq[i - 1] + 1e-6:
            mono_ok = False
            errors.append(
                f"Non-monotone: G={summary_rows[i]['G']} oracle "
                f"{oracle_seq[i]*100:.2f}% > G={summary_rows[i-1]['G']} "
                f"oracle {oracle_seq[i-1]*100:.2f}%"
            )
    print(f"  Monotonicity:    {'PASS' if mono_ok else 'FAIL'}")

    # Unique candidates <= G
    uniq_ok = True
    for r in summary_rows:
        if r["mean_unique_candidates"] > r["G"] + 0.01:
            uniq_ok = False
            errors.append(
                f"G={r['G']}: mean unique {r['mean_unique_candidates']:.1f} > G"
            )
    print(f"  Unique <= G:     {'PASS' if uniq_ok else 'FAIL'}")

    if errors:
        print("\n  WARNINGS:")
        for e in errors:
            print(f"    - {e}")
    else:
        print("\n  All checks PASSED")

    print("\n" + "=" * 90)
    print("BEAM SWEEP RESULTS")
    print("=" * 90)
    print(
        f"{'G':>5} {'Oracle%':>9} {'Greedy%':>9} {'AbsGap':>8} "
        f"{'RelGap%':>8} {'Uniq':>6} {'PairWER':>8} {'Spread':>8} "
        f"{'Recov':>6} {'Recov%':>7}"
    )
    print("-" * 90)
    for r in summary_rows:
        print(
            f"{r['G']:>5d} "
            f"{r['oracle_wer']*100:>8.2f}% "
            f"{r['greedy_wer']*100:>8.2f}% "
            f"{r['abs_gap']*100:>7.2f}% "
            f"{r['rel_gap']*100:>7.1f}% "
            f"{r['mean_unique_candidates']:>6.1f} "
            f"{r['mean_pairwise_wer']*100:>7.2f}% "
            f"{r['mean_logprob_spread']:>8.2f} "
            f"{r['num_recoverable']:>6d} "
            f"{r['pct_recoverable']*100:>6.1f}%"
        )
    print("=" * 90)
    print(f"\nTotal time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"Rate: {n / total_time:.1f} utt/s")

    if oom_count:
        print(f"OOM fallbacks: {oom_count}")

    # Output file sizes
    print("\nOutput files:")
    for g in g_values:
        path = output_dir / f"nbest_dev_other_G{g}.jsonl"
        if path.exists():
            size_mb = path.stat().st_size / 1e6
            print(f"  {path.name}: {size_mb:.1f} MB")
    print(f"  {csv_path.name}")
    print(f"  {json_path.name}")

    generate_report(summary_rows, metadata, output_dir)

if __name__ == "__main__":
    main()

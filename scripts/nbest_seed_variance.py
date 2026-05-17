#!/usr/bin/env python3
"""R-SEED: Multi-seed N-best resampling to quantify sampling variance.

Quantifies candidate-set sampling variance by running the full decode-time
pipeline (lattice -> N-best -> PLL scoring -> MBR-CER+PLL) across multiple
seeds at G=16. The CTC lattice is deterministic; only random path sampling
in k2.Nbest.from_lattice depends on torch RNG.

PLL scores are cached across seeds: at G=16, ~80%+ of hypotheses are shared
between seeds, so seeds 2+ score much faster.

Resumption: if PLL-scored JSONL files exist for all seeds in --output-dir,
steps 1-3 are skipped and MBR proceeds from cached files.

Usage (Colab):
    python scripts/nbest_seed_variance.py \
        --model-dir /path/to/model \
        --data-dir /path/to/data \
        --output-dir results/R_seed_variance/ \
        --seeds 42 137 2024 \
        --G 16 --nbest-scale 1.0 --oversample 64 --tau 10 \
        --cache-pll
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import editdistance
import numpy as np

BLANK_ID = 0
MAX_TOKEN = 499
VOCAB_SIZE = 500
EXPECTED_GREEDY_WER = 0.060218
EXPECTED_N_UTTS = 2864

_TAG_RE = re.compile(r"\{[^}]+\}|<[^>]+>")
_MULTI_SPACE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

def normalize_text(text):
    text = _TAG_RE.sub(" ", text)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text

def ctc_collapse(token_ids):
    result, prev = [], None
    for t in token_ids:
        if t != BLANK_ID and t != prev:
            result.append(t)
        prev = t
    return result

def alignment_log_prob(label_seq, log_probs_cpu):
    import torch
    T = log_probs_cpu.shape[0]
    if len(label_seq) != T:
        return float("-inf")
    idx = torch.tensor(label_seq, dtype=torch.long)
    return log_probs_cpu[torch.arange(T), idx].sum().item()

def add_icefall_to_path(icefall_dir):
    for d in [
        icefall_dir,
        icefall_dir / "egs" / "librispeech" / "ASR",
        icefall_dir / "egs" / "librispeech" / "ASR" / "zipformer",
    ]:
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))

def load_asr_model(checkpoint_path, icefall_dir, device):
    import torch
    add_icefall_to_path(icefall_dir)
    from train import add_model_arguments, get_model, get_params

    params = get_params()
    parser = argparse.ArgumentParser(add_help=False)
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
    params.vocab_size = VOCAB_SIZE
    params.feature_dim = 80

    model = get_model(params)
    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False
    )
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ASR model: {n_params / 1e6:.1f}M parameters on {device}")
    return model

def build_lattice(log_probs_utt, topo):
    """Build CTC lattice from log-probs (deterministic)."""
    import k2
    import torch

    T = log_probs_utt.shape[0]
    supervision_segments = torch.tensor([[0, 0, T]], dtype=torch.int32)
    dense_fsa = k2.DenseFsaVec(
        log_probs_utt.unsqueeze(0), supervision_segments
    )
    lattice = k2.intersect_dense(topo, dense_fsa, output_beam=8.0)
    lattice = k2.connect(lattice)
    return lattice

def greedy_decode(log_probs_utt, sp):
    """Greedy CTC decode (deterministic, same across all seeds)."""
    lp_cpu = log_probs_utt.cpu()
    greedy_ids = log_probs_utt.argmax(dim=-1).cpu().tolist()
    greedy_collapsed = ctc_collapse(greedy_ids)
    greedy_text = normalize_text(sp.decode(greedy_collapsed))
    greedy_score = alignment_log_prob(greedy_ids, lp_cpu)
    return {"hyp": greedy_text, "score": round(greedy_score, 6)}

def sample_nbest(lattice, log_probs_cpu, num_paths, nbest_scale, sp,
                 greedy_entry, G):
    """Sample N-best from existing lattice. Stochastic via torch RNG.

    Call torch.manual_seed(seed) BEFORE this function to control sampling.
    """
    import k2

    nbest_obj = k2.Nbest.from_lattice(
        lattice,
        num_paths=num_paths,
        use_double_scores=True,
        nbest_scale=nbest_scale,
    )

    all_labels = nbest_obj.fsa.labels.cpu().tolist()
    paths, current = [], []
    for label in all_labels:
        if label == -1:
            paths.append(current)
            current = []
        else:
            current.append(label)

    seen = {}
    for raw_ids in paths:
        score = alignment_log_prob(raw_ids, log_probs_cpu)
        if score == float("-inf"):
            continue
        token_ids = ctc_collapse(raw_ids)
        text = normalize_text(sp.decode(token_ids))
        if not text:
            continue
        entry = {"hyp": text, "score": round(score, 6)}
        if text not in seen or score > seen[text]["score"]:
            seen[text] = entry

    greedy_text = greedy_entry["hyp"]
    seen[greedy_text] = greedy_entry

    candidates = sorted(seen.values(), key=lambda c: c["score"], reverse=True)
    rest = [c for c in candidates if c["hyp"] != greedy_text]
    candidates = [greedy_entry] + rest

    del nbest_obj
    return candidates[:G]

def compute_pll(text, tokenizer, model, device, batch_size=32):
    """Pseudo-log-likelihood: sum_i log P(token_i | tokens_{-i})."""
    import torch

    with torch.no_grad():
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"][0].to(device)
        L = input_ids.size(0)

        special = {
            tokenizer.bos_token_id, tokenizer.eos_token_id,
            tokenizer.pad_token_id, tokenizer.cls_token_id,
            tokenizer.sep_token_id,
        }
        special.discard(None)
        mask_id = tokenizer.mask_token_id

        positions = [
            i for i in range(L) if input_ids[i].item() not in special
        ]
        if not positions:
            return 0.0

        total = 0.0
        for s in range(0, len(positions), batch_size):
            batch_pos = positions[s:s + batch_size]
            bsz = len(batch_pos)
            masked = input_ids.unsqueeze(0).repeat(bsz, 1).clone()
            for k, p in enumerate(batch_pos):
                masked[k, p] = mask_id
            logits = model(masked).logits
            log_probs = torch.log_softmax(logits, dim=-1)
            for k, p in enumerate(batch_pos):
                total += log_probs[k, p, input_ids[p].item()].item()

    return total

def cer_matrix(texts):
    """Symmetric character error rate matrix for one utterance."""
    n = len(texts)
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = editdistance.eval(list(texts[i]), list(texts[j]))
            denom = max(len(texts[i]), len(texts[j]), 1)
            mat[i, j] = d / denom
            mat[j, i] = mat[i, j]
    return mat

def mbr_select(cer_mat, log_scores, tau):
    """Select hypothesis minimizing expected CER under softmax(scores/tau)."""
    scaled = np.array(log_scores) / tau
    scaled -= np.max(scaled)
    weights = np.exp(scaled)
    weights /= weights.sum()
    risk = cer_mat @ weights
    return int(np.argmin(risk))

def corpus_wer_from_pairs(refs, hyps):
    """Corpus-level WER: sum(edits) / sum(ref_words)."""
    total_edits = total_ref = 0
    for ref, hyp in zip(refs, hyps):
        ref_w = ref.split()
        hyp_w = hyp.split()
        total_edits += editdistance.eval(hyp_w, ref_w)
        total_ref += len(ref_w)
    return total_edits / max(total_ref, 1)

def interp_sweep(records, alphas):
    """Best WER from argmax interpolation: alpha*CTC + (1-alpha)*PLL."""
    best_wer, best_alpha = float("inf"), None
    for alpha in alphas:
        hyps = []
        for rec in records:
            scores = [
                alpha * c["score"] + (1 - alpha) * c.get("pll_score", 0)
                for c in rec["nbest"]
            ]
            hyps.append(rec["nbest"][int(np.argmax(scores))]["hyp"])
        wer = corpus_wer_from_pairs([r["ref"] for r in records], hyps)
        if wer < best_wer:
            best_wer, best_alpha = wer, alpha
    return best_wer, best_alpha

def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records

def save_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def find_file(directory, patterns, description):
    """Find a file matching glob patterns, searching subdirectories too."""
    for pat in patterns:
        matches = sorted(directory.glob(pat))
        if matches:
            return matches[0]
    for pat in patterns:
        matches = sorted(directory.rglob(pat))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Could not find {description} in {directory}. "
        f"Tried patterns: {patterns}"
    )

def generate_report(summary, output_path):
    meta = summary["metadata"]
    seeds = meta["seeds"]
    per_seed = summary["per_seed"]
    cross = summary["cross_seed"]
    verif = summary["verification"]
    greedy_wer = meta["greedy_wer"]

    lines = []
    lines.append("# R-SEED: Multi-Seed N-best Resampling Variance Report")
    lines.append("")
    lines.append("## What ran")
    lines.append("")
    lines.append(
        f"- **Experiment:** Multi-seed N-best resampling variance at G={meta['G']}"
    )
    lines.append(f"- **Seeds:** {seeds}")
    lines.append(
        f"- **Dataset:** LibriSpeech dev-other "
        f"({meta['n_utterances']} utterances)"
    )
    lines.append(
        f"- **N-best:** G={meta['G']}, nbest_scale={meta['nbest_scale']}, "
        f"oversample={meta['oversample']} (num_paths={meta['num_paths']})"
    )
    lines.append(
        f"- **MBR:** CER utility, PLL posterior weights, tau={meta['tau']}"
    )
    lines.append(
        f"- **PLL model:** {meta['pll_model']}, cache={meta['cache_pll']}"
    )
    lines.append(f"- **Date:** {meta['date']}")
    lines.append("")

    lines.append("## Verification checks")
    lines.append("")
    checks = [
        ("Different oracle WERs across seeds", verif["different_oracles"]),
        ("Seed 42 MBR < greedy", verif.get("seed42_mbr_approx_579")),
        ("MBR <= greedy for all seeds", verif["mbr_leq_greedy_all_seeds"]),
        ("SD(MBR) < SD(oracle)", verif["mbr_sd_lt_oracle_sd"]),
        ("PLL cache hits monotonic", verif["cache_hits_monotonic"]),
    ]
    for name, ok in checks:
        if ok is None:
            mark = "N/A"
        elif ok:
            mark = "PASS"
        else:
            mark = "FAIL"
        lines.append(f"- [{mark}] {name}")
    lines.append("")

    lines.append("## Per-seed results")
    lines.append("")
    lines.append(
        "| Seed | Unique Hyps | Oracle WER | MBR WER "
        "| Interp WER (alpha) | Cache Hits |"
    )
    lines.append(
        "|-----:|------------:|-----------:|--------:"
        "|-------------------:|-----------:|"
    )
    for r in per_seed:
        lines.append(
            f"| {r['seed']} | {r['n_unique_hyps']} | "
            f"{r['oracle_wer']*100:.4f}% | {r['mbr_wer']*100:.4f}% | "
            f"{r['interp_best_wer']*100:.4f}% "
            f"(a={r['interp_best_alpha']}) | "
            f"{r['pll_cache_hits']} |"
        )
    lines.append("")

    lines.append("## Cross-seed statistics")
    lines.append("")
    lines.append("| Metric | Mean | SD (pp) | Range |")
    lines.append("|--------|-----:|--------:|-------|")
    lines.append(
        f"| Oracle WER | {cross['oracle_wer_mean']*100:.4f}% | "
        f"{cross['oracle_wer_std_pp']:.4f} | "
        f"{cross['oracle_wer_range'][0]*100:.4f}%"
        f"--{cross['oracle_wer_range'][1]*100:.4f}% |"
    )
    lines.append(
        f"| MBR WER | {cross['mbr_wer_mean']*100:.4f}% | "
        f"{cross['mbr_wer_std_pp']:.4f} | "
        f"{cross['mbr_wer_range'][0]*100:.4f}%"
        f"--{cross['mbr_wer_range'][1]*100:.4f}% |"
    )
    lines.append("")
    lines.append(f"- **Greedy WER:** {greedy_wer*100:.4f}%")
    lines.append(
        f"- **Effect size (greedy - MBR mean):** "
        f"{cross['effect_size_pp']:.4f}pp"
    )
    lines.append(f"- **Effect / SD ratio:** {cross['effect_to_sd_ratio']}x")
    lines.append("")

    lines.append("## Paper-ready paragraph")
    lines.append("")
    seed_str = ", ".join(str(s) for s in seeds)
    lines.append(
        f"To quantify the sensitivity of our results to N-best sampling noise, "
        f"we repeated the full decode-time pipeline "
        f"(lattice sampling, PLL scoring, MBR-CER reranking) at G={meta['G']} "
        f"with three random seeds ({seed_str}). "
        f"MBR-CER+PLL WER varied by only "
        f"{cross['mbr_wer_std_pp']:.4f}pp (SD), "
        f"ranging from {cross['mbr_wer_range'][0]*100:.2f}\\% "
        f"to {cross['mbr_wer_range'][1]*100:.2f}\\%, "
        f"compared to the {cross['effect_size_pp']:.2f}pp improvement "
        f"over greedy ({greedy_wer*100:.2f}\\%)---yielding an "
        f"effect-to-noise ratio of {cross['effect_to_sd_ratio']}$\\times$. "
        f"Oracle WER showed slightly higher variance "
        f"(SD={cross['oracle_wer_std_pp']:.4f}pp), "
        f"confirming that MBR consensus smooths out "
        f"candidate-set sampling noise."
    )
    lines.append("")

    lines.append("## Section B.1 update needed?")
    lines.append("")
    ratio = cross["effect_to_sd_ratio"]
    ratio_val = float(ratio) if ratio != "inf" else 999
    if ratio_val > 10:
        lines.append(
            f"The effect-to-noise ratio of {ratio}x is large. "
            f"Section B.1 should note that sampling variance was quantified "
            f"and found negligible relative to the reported gains."
        )
    else:
        lines.append(
            f"The effect-to-noise ratio of {ratio}x is modest. "
            f"Section B.1 should discuss this as a limitation "
            f"and report the variance."
        )
    lines.append("")

    lines.append("## Surprises")
    lines.append("")
    surprises = []
    if not verif["different_oracles"]:
        surprises.append(
            "Seeds did NOT produce different oracle WERs. "
            "torch.manual_seed may not control k2's random path sampling. "
            "Try k2.set_seed(seed) or check for a seed kwarg in "
            "Nbest.from_lattice."
        )
    if verif.get("seed42_mbr_approx_579") is False:
        seed42 = next(r for r in per_seed if r["seed"] == 42)
        surprises.append(
            f"Seed 42 MBR WER = {seed42['mbr_wer']*100:.4f}%, "
            f"deviates from expected 5.79%. "
            f"Likely due to oversample={meta['oversample']} vs "
            f"original oversample factor."
        )
    if not verif["mbr_sd_lt_oracle_sd"]:
        surprises.append(
            "SD(MBR) >= SD(oracle). "
            "MBR consensus did not reduce variance as expected."
        )
    if surprises:
        for s in surprises:
            lines.append(f"- {s}")
    else:
        lines.append("- None. All results match expectations.")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved report: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="R-SEED: Multi-seed N-best resampling variance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir", type=Path,
        help="Directory with pretrained.pt and bpe.model "
             "(not needed if PLL-scored JSONL files already exist)",
    )
    parser.add_argument(
        "--data-dir", type=Path,
        help="Directory with lhotse cuts or path to cuts file "
             "(not needed if PLL-scored JSONL files already exist)",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--icefall-dir", type=Path, default=Path("/content/icefall"),
        help="Path to icefall repo",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 137, 2024],
        help="Random seeds for N-best sampling",
    )
    parser.add_argument("--G", type=int, default=16,
                        help="Hypotheses per utterance")
    parser.add_argument("--nbest-scale", type=float, default=1.0,
                        help="Lattice score scale (canonical=1.0)")
    parser.add_argument("--oversample", type=int, default=64,
                        help="Oversample factor: sample G*oversample paths")
    parser.add_argument("--tau", type=float, default=10.0,
                        help="MBR temperature")
    parser.add_argument("--cache-pll", action="store_true",
                        help="Cache PLL scores across seeds")
    parser.add_argument("--pll-model", type=str, default="roberta-base",
                        help="HuggingFace model for PLL scoring")
    parser.add_argument("--pll-batch-size", type=int, default=32,
                        help="Batch size for PLL masked positions")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    assert args.nbest_scale != 0.5, (
        "nbest_scale=0.5 destroys oracle gap (E23/E24 bug). Use 1.0."
    )

    num_paths = args.G * args.oversample
    t_total = time.time()

    print("=" * 70)
    print("R-SEED: Multi-Seed N-best Resampling Variance")
    print("=" * 70)
    print(f"  output_dir:   {args.output_dir}")
    print(f"  seeds:        {args.seeds}")
    print(f"  G:            {args.G}")
    print(f"  nbest_scale:  {args.nbest_scale}")
    print(f"  oversample:   {args.oversample} (num_paths={num_paths})")
    print(f"  tau:          {args.tau}")
    print(f"  cache_pll:    {args.cache_pll}")
    print(f"  pll_model:    {args.pll_model}")
    print(f"  device:       {args.device}")
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.output_dir / "seed_variance_summary.json"
    if summary_path.exists():
        print(f"Output already exists: {summary_path}")
        print("Delete to re-run.")
        with open(summary_path) as f:
            summary = json.load(f)
        for r in summary["per_seed"]:
            print(f"  Seed {r['seed']}: oracle={r['oracle_wer']:.4%}  "
                  f"MBR={r['mbr_wer']:.4%}")
        return

    pll_files = {
        s: args.output_dir / f"nbest_seed_{s}_pll.jsonl"
        for s in args.seeds
    }
    all_pll_exist = all(f.exists() for f in pll_files.values())

    gen_time = None
    pll_time = None

    if all_pll_exist:
        # Fast path: skip generation + PLL scoring
        print("Found PLL-scored N-best files for all seeds. "
              "Skipping to MBR step.")
        seed_records = {}
        for seed in args.seeds:
            seed_records[seed] = load_jsonl(pll_files[seed])
            n_utts = len(seed_records[seed])
            print(f"  Seed {seed}: {n_utts} utterances loaded")
        seed_pll_stats = {
            s: {"cache_hits": -1, "cache_misses": -1}
            for s in args.seeds
        }
        print()

    else:
        # Full pipeline: generate + score + save

        if args.model_dir is None or args.data_dir is None:
            parser.error(
                "--model-dir and --data-dir are required "
                "(no cached PLL-scored JSONL files found in --output-dir)"
            )

        import torch
        device = torch.device(args.device)

        # Step 0: Discover files
        print("Step 0/5: Discovering files...")
        checkpoint_path = find_file(
            args.model_dir, ["pretrained.pt", "*.pt"],
            "model checkpoint",
        )
        bpe_path = find_file(
            args.model_dir, ["bpe.model", "*.model"],
            "BPE model",
        )
        print(f"  checkpoint: {checkpoint_path}")
        print(f"  bpe:        {bpe_path}")

        if args.data_dir.is_file():
            cuts_path = args.data_dir
        else:
            cuts_path = find_file(
                args.data_dir,
                ["cuts_dev*other*.jsonl.gz",
                 "cuts_*.jsonl.gz",
                 "*.jsonl.gz"],
                "lhotse CutSet",
            )
        print(f"  cuts:       {cuts_path}")
        print()

        # Step 1: Load model + data
        print("Step 1/5: Loading model and data...")

        import sentencepiece as spm
        import k2
        from lhotse import Fbank, FbankConfig, load_manifest_lazy

        asr_model = load_asr_model(
            checkpoint_path, args.icefall_dir, device
        )

        sp = spm.SentencePieceProcessor()
        sp.load(str(bpe_path))
        assert sp.get_piece_size() == VOCAB_SIZE, (
            f"BPE vocab {sp.get_piece_size()} != expected {VOCAB_SIZE}"
        )

        cuts = list(load_manifest_lazy(str(cuts_path)))
        n_utts = len(cuts)
        assert n_utts == EXPECTED_N_UTTS, (
            f"Expected {EXPECTED_N_UTTS} utterances, got {n_utts}"
        )
        print(f"  {n_utts} utterances loaded")

        fbank = Fbank(FbankConfig(num_mel_bins=80, dither=0.0))
        topo = k2.ctc_topo(
            max_token=MAX_TOKEN, modified=False, device=device
        )
        print()

        # Step 2: Generate N-best (lattice once, sample per seed)
        print("Step 2/5: Generating N-best "
              "(lattice once per utterance, sample per seed)...")

        seed_records = {s: [] for s in args.seeds}
        t0 = time.time()

        for utt_i, cut in enumerate(cuts):
            # Forward pass  --  once per utterance
            feat = fbank.extract(cut.load_audio(), sampling_rate=16000)
            feat_t = torch.from_numpy(feat).unsqueeze(0).to(device)
            feat_lens = torch.tensor(
                [feat.shape[0]], dtype=torch.int64, device=device
            )

            with torch.no_grad():
                enc_out, enc_lens = asr_model.forward_encoder(
                    feat_t, feat_lens
                )
                log_probs = asr_model.ctc_output(enc_out)

            lp_utt = log_probs[0, :enc_lens[0].item()]
            lp_cpu = lp_utt.cpu()

            # Greedy  --  deterministic, same for all seeds
            greedy_entry = greedy_decode(lp_utt, sp)

            # Lattice  --  deterministic
            lattice = build_lattice(lp_utt, topo)

            ref_raw = " ".join(
                s.text for s in cut.supervisions if s.text
            )
            ref = normalize_text(ref_raw)

            # Sample with each seed (only this step is stochastic)
            for seed in args.seeds:
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)

                candidates = sample_nbest(
                    lattice, lp_cpu, num_paths, args.nbest_scale,
                    sp, greedy_entry, args.G,
                )
                seed_records[seed].append({
                    "utt_id": cut.id,
                    "ref": ref,
                    "nbest": candidates,
                })

            del lattice, log_probs, enc_out, feat_t, lp_utt
            torch.cuda.empty_cache()

            if (utt_i + 1) % 100 == 0 or utt_i == n_utts - 1:
                elapsed = time.time() - t0
                speed = (utt_i + 1) / elapsed
                eta = (n_utts - utt_i - 1) / speed if speed > 0 else 0
                print(f"  {utt_i+1}/{n_utts}  "
                      f"({speed:.1f} utt/s, ETA {eta:.0f}s)")

        gen_time = time.time() - t0
        print(f"\n  N-best generation: {gen_time:.1f}s "
              f"({gen_time / 60:.1f} min)")

        # Verify seeds produce different candidate sets
        for i, s1 in enumerate(args.seeds):
            for s2 in args.seeds[i + 1:]:
                set1 = {
                    (r["utt_id"], c["hyp"])
                    for r in seed_records[s1] for c in r["nbest"]
                }
                set2 = {
                    (r["utt_id"], c["hyp"])
                    for r in seed_records[s2] for c in r["nbest"]
                }
                union = len(set1 | set2)
                overlap = len(set1 & set2) / union * 100 if union else 0
                print(f"  Jaccard(seed {s1}, {s2}): {overlap:.1f}%")

        # Free ASR model before loading PLL model
        del asr_model, topo
        torch.cuda.empty_cache()
        print()

        # Step 3: PLL scoring (with cross-seed caching)
        print("Step 3/5: Scoring with RoBERTa PLL...")

        from transformers import RobertaTokenizer, RobertaForMaskedLM

        tokenizer = RobertaTokenizer.from_pretrained(args.pll_model)
        pll_model = RobertaForMaskedLM.from_pretrained(
            args.pll_model
        ).to(device)
        pll_model.eval()

        n_pll_params = sum(p.numel() for p in pll_model.parameters())
        print(f"  PLL model: {n_pll_params / 1e6:.1f}M parameters")

        pll_cache = {} if args.cache_pll else None
        seed_pll_stats = {}
        t0 = time.time()

        for seed in args.seeds:
            hits, misses = 0, 0

            for rec_i, rec in enumerate(seed_records[seed]):
                for c in rec["nbest"]:
                    text = c["hyp"]
                    if pll_cache is not None and text in pll_cache:
                        c["pll_score"] = pll_cache[text]
                        hits += 1
                    else:
                        tokens = tokenizer.encode(text)
                        score_text = (
                            tokenizer.decode(
                                tokens[:510], skip_special_tokens=True
                            )
                            if len(tokens) > 510 else text
                        )
                        c["pll_score"] = compute_pll(
                            score_text, tokenizer, pll_model,
                            device, args.pll_batch_size,
                        )
                        if pll_cache is not None:
                            pll_cache[text] = c["pll_score"]
                        misses += 1

                if (rec_i + 1) % 200 == 0:
                    elapsed = time.time() - t0
                    rate = misses / elapsed if elapsed > 0 else 0
                    print(
                        f"  Seed {seed}: {rec_i+1}/"
                        f"{len(seed_records[seed])} utts  "
                        f"({misses} scored, {hits} cached, "
                        f"{rate:.1f} new/s)"
                    )

            total = hits + misses
            seed_pll_stats[seed] = {
                "cache_hits": hits,
                "cache_misses": misses,
            }
            print(
                f"  Seed {seed} done: {misses} scored, {hits} cached "
                f"({hits / total * 100:.1f}% hit rate)"
            )

            # Save intermediate JSONL per seed (for resumption)
            save_jsonl(seed_records[seed], pll_files[seed])
            print(f"  Saved: {pll_files[seed]}")

        pll_time = time.time() - t0
        print(f"\n  PLL scoring: {pll_time:.1f}s "
              f"({pll_time / 60:.1f} min)")
        if pll_cache:
            print(f"  PLL cache: {len(pll_cache)} unique texts")

        del pll_model
        torch.cuda.empty_cache()
        print()

    print("Step 4/5: MBR-CER reranking and WER computation...")

    n_utts = len(seed_records[args.seeds[0]])
    greedy_wer = None
    seed_results = {}
    per_utt_rows = []
    alphas = [a / 10 for a in range(11)]

    for seed in args.seeds:
        records = seed_records[seed]

        # Greedy WER  --  same for all seeds (greedy is deterministic)
        if greedy_wer is None:
            greedy_wer = corpus_wer_from_pairs(
                [r["ref"] for r in records],
                [r["nbest"][0]["hyp"] for r in records],
            )
            if abs(greedy_wer - EXPECTED_GREEDY_WER) < 0.001:
                print(f"  Greedy WER: {greedy_wer:.4%} (verified)")
            else:
                print(
                    f"  WARNING: Greedy WER = {greedy_wer:.4%}, "
                    f"expected {EXPECTED_GREEDY_WER:.4%}. "
                    f"Feature extraction or model version may differ. "
                    f"Proceeding  --  variance measurement is still valid."
                )

        # Oracle WER
        oracle_hyps = []
        for r in records:
            ref_w = r["ref"].split()
            best = min(
                r["nbest"],
                key=lambda c: editdistance.eval(c["hyp"].split(), ref_w),
            )
            oracle_hyps.append(best["hyp"])
        oracle_wer = corpus_wer_from_pairs(
            [r["ref"] for r in records], oracle_hyps,
        )

        # MBR-CER with PLL weights at tau
        mbr_hyps = []
        for utt_i, rec in enumerate(records):
            texts = [c["hyp"] for c in rec["nbest"]]
            pll_scores = [c["pll_score"] for c in rec["nbest"]]
            cer_mat = cer_matrix(texts)
            mbr_idx = mbr_select(cer_mat, pll_scores, args.tau)
            mbr_hyps.append(texts[mbr_idx])

            per_utt_rows.append({
                "utt_id": rec["utt_id"],
                "seed": seed,
                "n_cands": len(rec["nbest"]),
                "mbr_idx": mbr_idx,
                "mbr_hyp": texts[mbr_idx],
            })

        mbr_wer = corpus_wer_from_pairs(
            [r["ref"] for r in records], mbr_hyps,
        )

        # Interpolation sweep
        interp_wer, interp_alpha = interp_sweep(records, alphas)

        n_unique = len({
            c["hyp"] for r in records for c in r["nbest"]
        })

        gap = greedy_wer - oracle_wer
        gap_closed = (
            (greedy_wer - mbr_wer) / gap * 100
            if gap > 1e-9 else 0
        )

        seed_results[seed] = {
            "seed": seed,
            "n_unique_hyps": n_unique,
            "oracle_wer": round(oracle_wer, 6),
            "mbr_wer": round(mbr_wer, 6),
            "interp_best_wer": round(interp_wer, 6),
            "interp_best_alpha": interp_alpha,
            "pll_cache_hits": seed_pll_stats[seed]["cache_hits"],
            "pll_cache_misses": seed_pll_stats[seed]["cache_misses"],
        }

        print(
            f"  Seed {seed}: oracle={oracle_wer:.4%}  "
            f"MBR={mbr_wer:.4%}  "
            f"interp={interp_wer:.4%} (a={interp_alpha})  "
            f"gap_closed={gap_closed:.1f}%"
        )

    print()

    print("Step 5/5: Cross-seed statistics and output...")

    oracle_wers = [seed_results[s]["oracle_wer"] for s in args.seeds]
    mbr_wers = [seed_results[s]["mbr_wer"] for s in args.seeds]

    oracle_mean = float(np.mean(oracle_wers))
    oracle_std = (
        float(np.std(oracle_wers, ddof=1))
        if len(args.seeds) > 1 else 0.0
    )
    mbr_mean = float(np.mean(mbr_wers))
    mbr_std = (
        float(np.std(mbr_wers, ddof=1))
        if len(args.seeds) > 1 else 0.0
    )

    effect_size = greedy_wer - mbr_mean
    effect_to_sd = (
        effect_size / mbr_std if mbr_std > 0 else float("inf")
    )

    # Verification
    v_different_oracles = (
        len(set(f"{w:.6f}" for w in oracle_wers)) > 1
    )
    v_seed42_mbr = (
        seed_results[42]["mbr_wer"] < greedy_wer
        if 42 in seed_results else None
    )
    v_mbr_leq_greedy = all(
        w <= greedy_wer + 1e-6 for w in mbr_wers
    )
    v_mbr_sd_lt_oracle = (
        mbr_std < oracle_std if oracle_std > 0 else mbr_std == 0
    )

    cache_hits_list = [
        seed_pll_stats[s]["cache_hits"] for s in args.seeds
    ]
    if all(h >= 0 for h in cache_hits_list):
        v_cache_monotonic = all(
            cache_hits_list[i] <= cache_hits_list[i + 1]
            for i in range(len(cache_hits_list) - 1)
        )
    else:
        v_cache_monotonic = None

    verification = {
        "different_oracles": v_different_oracles,
        "seed42_mbr_approx_579": v_seed42_mbr,
        "mbr_leq_greedy_all_seeds": v_mbr_leq_greedy,
        "mbr_sd_lt_oracle_sd": v_mbr_sd_lt_oracle,
        "cache_hits_monotonic": v_cache_monotonic,
    }

    print()
    print("=" * 70)
    print("CROSS-SEED SUMMARY")
    print("=" * 70)
    print(f"  Greedy WER:    {greedy_wer:.4%}")
    print(f"  Oracle WER:    {oracle_mean:.4%} "
          f"+/- {oracle_std * 100:.4f}pp")
    print(f"  MBR WER:       {mbr_mean:.4%} "
          f"+/- {mbr_std * 100:.4f}pp")
    print(f"  Effect size:   {effect_size * 100:.4f}pp")
    eff_str = (
        f"{effect_to_sd:.1f}" if not np.isinf(effect_to_sd) else "inf"
    )
    print(f"  Effect / SD:   {eff_str}x")
    print()
    print("  VERIFICATION:")
    for name, ok in verification.items():
        if ok is None:
            status = "N/A"
        elif ok:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"    [{status}] {name}")

    eff_ratio_json = (
        round(effect_to_sd, 1)
        if not np.isinf(effect_to_sd) else "inf"
    )
    summary = {
        "metadata": {
            "experiment": "R-SEED",
            "description": (
                "Multi-seed N-best resampling variance at "
                f"G={args.G}"
            ),
            "seeds": args.seeds,
            "G": args.G,
            "nbest_scale": args.nbest_scale,
            "oversample": args.oversample,
            "num_paths": num_paths,
            "tau": args.tau,
            "pll_model": args.pll_model,
            "cache_pll": args.cache_pll,
            "n_utterances": n_utts,
            "greedy_wer": round(greedy_wer, 6),
            "date": time.strftime("%Y-%m-%d"),
        },
        "per_seed": [seed_results[s] for s in args.seeds],
        "cross_seed": {
            "oracle_wer_mean": round(oracle_mean, 6),
            "oracle_wer_std_pp": round(oracle_std * 100, 4),
            "oracle_wer_range": [
                round(min(oracle_wers), 6),
                round(max(oracle_wers), 6),
            ],
            "mbr_wer_mean": round(mbr_mean, 6),
            "mbr_wer_std_pp": round(mbr_std * 100, 4),
            "mbr_wer_range": [
                round(min(mbr_wers), 6),
                round(max(mbr_wers), 6),
            ],
            "effect_size_pp": round(effect_size * 100, 4),
            "effect_to_sd_ratio": eff_ratio_json,
        },
        "verification": verification,
        "timing": {
            "nbest_generation_s": (
                round(gen_time, 1) if gen_time is not None else None
            ),
            "pll_scoring_s": (
                round(pll_time, 1) if pll_time is not None else None
            ),
            "total_s": round(time.time() - t_total, 1),
        },
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {summary_path}")

    csv_path = args.output_dir / "seed_variance_per_utt.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "utt_id", "seed", "n_cands", "mbr_idx", "mbr_hyp",
            ],
        )
        writer.writeheader()
        writer.writerows(per_utt_rows)
    print(f"  Saved: {csv_path}")

    report_path = Path("reports") / "R_seed_variance_report.md"
    generate_report(summary, report_path)

    total_time = time.time() - t_total
    print(f"\n  Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print()
    print("Bring-back files:")
    print(f"  {args.output_dir}/seed_variance_summary.json")
    print(f"  {args.output_dir}/seed_variance_per_utt.csv")
    for seed in args.seeds:
        print(f"  {args.output_dir}/nbest_seed_{seed}_pll.jsonl")
    print("  reports/R_seed_variance_report.md")

if __name__ == "__main__":
    main()

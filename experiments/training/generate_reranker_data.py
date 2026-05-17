#!/usr/bin/env python3
"""E21: Generate N-best training data for discriminative reranker.

For each train-clean-100 utterance: emit a JSONL line with the reference
text and a list of N-best hypotheses, each tagged with its CTC log-prob
and word-level edit distance to the reference.

NO PLL/BERT scoring here  --  too expensive for ~450K hypotheses. The
downstream reranker training will compute its own representations.

Same model, BPE, and feature pipeline as E11/E18/E19/E20.
CTC-only N-best (no HLG  --  that's broken in k2 1.24, see E20).

Usage (Colab T4):
    python generate_reranker_data.py \\
        --model-dir /content/icefall-asr-librispeech-zipformer-small-cr-ctc \\
        --data-dir /content/librispeech_data \\
        --icefall-dir /content/icefall \\
        --output-dir /content/drive/MyDrive/rbpo_results/reranker_training_data \\
        --device cuda:0 \\
        --num-paths 64 \\
        --nbest-scale 0.5 \\
        --max-keep 16

Expected runtime: 1-3 hours on T4 for 28,539 utterances.
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import editdistance
import sentencepiece as spm
import torch

# Suppress noisy k2 C++ warnings
os.environ["K2_VERBOSE_LEVEL"] = "0"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


BLANK_ID = 0
MAX_TOKEN = 499
VOCAB_SIZE = 500


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
        model_dir / "exp" / "pretrained.pt",
        map_location="cpu", weights_only=False,
    )
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {n_params/1e6:.1f}M parameters")
    return model


def load_cuts(data_dir: Path, split: str, cuts_path: Path = None):
    """Load lhotse CutSet. If cuts_path is given, use it directly;
    otherwise default to librispeech naming.
    """
    from lhotse import load_manifest_lazy
    if cuts_path is None:
        cuts_path = data_dir / "cuts" / f"librispeech_cuts_{split}.jsonl.gz"
    if not cuts_path.exists():
        raise RuntimeError(
            f"CutSet not found at {cuts_path}. "
            f"Run prepare_train_clean_100.py or prepare_tedlium3.py first."
        )
    return load_manifest_lazy(str(cuts_path))


def extract_features_batch(cuts_batch, device):
    """Use pre-computed lhotse features. Critical: do NOT recompute from
    audio with torchaudio  --  different params give 99.97% WER (E20 lesson).
    """
    features_list = []
    lengths = []
    for cut in cuts_batch:
        feat = torch.from_numpy(cut.load_features())
        features_list.append(feat)
        lengths.append(feat.shape[0])
    max_len = max(lengths)
    batch = torch.zeros(len(features_list), max_len, 80)
    for i, feat in enumerate(features_list):
        batch[i, :feat.shape[0]] = feat
    return batch.to(device), torch.tensor(lengths, dtype=torch.int64).to(device)


def ctc_collapse(ids):
    """CTC collapse: drop blanks and consecutive duplicates."""
    out = []
    prev = None
    for t in ids:
        if t != BLANK_ID and t != prev:
            out.append(t)
        prev = t
    return out


def generate_nbest_batch(
    model, cuts_batch, sp, topo, device, num_paths, nbest_scale, max_keep,
    return_greedy=False, output_beam=8.0,
):
    """Generate N-best for a batch. Encoder runs batched; lattice ops per-utt.

    If return_greedy=True, also returns true greedy (argmax) text per utt.
    output_beam controls k2.intersect_dense pruning (default 8.0).
    """
    import k2

    features, lengths = extract_features_batch(cuts_batch, device)
    with torch.no_grad():
        encoder_out, encoder_out_lens = model.forward_encoder(features, lengths)
        log_probs = model.ctc_output(encoder_out)

    if log_probs.shape[0] != len(cuts_batch):
        raise RuntimeError(
            f"Encoder output batch {log_probs.shape[0]} != input {len(cuts_batch)}"
        )

    results = []
    greedy_texts = []
    for i, cut in enumerate(cuts_batch):
        T = encoder_out_lens[i].item()
        lp = log_probs[i, :T]
        lp_cpu = lp.cpu()

        supervision_segments = torch.tensor([[0, 0, T]], dtype=torch.int32)
        dense_fsa = k2.DenseFsaVec(lp.unsqueeze(0), supervision_segments)
        lattice = k2.intersect_dense(topo, dense_fsa, output_beam=output_beam)
        lattice = k2.connect(lattice)

        nbest = k2.Nbest.from_lattice(
            lattice,
            num_paths=num_paths,
            use_double_scores=True,
            nbest_scale=nbest_scale,
        )

        all_labels = nbest.fsa.labels.cpu().tolist()
        paths = []
        current = []
        for label in all_labels:
            if label == -1:
                paths.append(current)
                current = []
            else:
                current.append(label)

        # Compute greedy path. We always inject it as a candidate so the
        # N-best is guaranteed to include the model's best guess; without
        # this, diverse sampling (low nbest_scale) often misses it and
        # the reranker has no good target to rank up.
        greedy_ids_full = lp.argmax(dim=-1).cpu().tolist()
        greedy_collapsed = ctc_collapse(greedy_ids_full)
        greedy_text = sp.decode(greedy_collapsed).strip().lower()
        greedy_score = sum(
            lp_cpu[t_idx, tok].item() for t_idx, tok in enumerate(greedy_ids_full)
        )
        if return_greedy:
            greedy_texts.append(greedy_text)

        seen = {}
        if greedy_text:
            seen[greedy_text] = {
                "text": greedy_text,
                "tokens": greedy_collapsed,
                "ctc_log_prob": greedy_score,
            }

        for raw_ids in paths:
            if len(raw_ids) != T:
                continue
            collapsed = ctc_collapse(raw_ids)
            text = sp.decode(collapsed).strip().lower()
            if not text:
                continue
            score = sum(lp_cpu[t_idx, tok].item() for t_idx, tok in enumerate(raw_ids))
            entry = {
                "text": text,
                "tokens": collapsed,
                "ctc_log_prob": score,
            }
            if text in seen:
                if score > seen[text]["ctc_log_prob"]:
                    seen[text] = entry
            else:
                seen[text] = entry

        cands = sorted(seen.values(), key=lambda c: c["ctc_log_prob"], reverse=True)
        results.append(cands[:max_keep])

    if return_greedy:
        return results, greedy_texts
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--icefall-dir", type=Path, default=Path("/content/icefall"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", default="train-clean-100")
    parser.add_argument("--cuts-path", type=Path, default=None,
                        help="Direct path to a lhotse cuts .jsonl.gz file. "
                             "If set, overrides --split-based librispeech path.")
    parser.add_argument("--output-jsonl-name", default=None,
                        help="Override output JSONL basename (default: "
                             "{split}_G{max_keep}.jsonl).")
    parser.add_argument("--smoke-greedy-max", type=float, default=0.10,
                        help="Smoke test fails if greedy WER > this. "
                             "0.10 for clean LibriSpeech; raise to ~0.30 for "
                             "cross-domain TED-LIUM zero-shot.")
    parser.add_argument("--num-paths", type=int, default=64,
                        help="Paths to sample per utterance (oversample then dedup)")
    parser.add_argument("--nbest-scale", type=float, default=0.5,
                        help="Path sampling temperature (lower = more diverse)")
    parser.add_argument("--max-keep", type=int, default=16,
                        help="Max unique hypotheses to keep per utterance")
    parser.add_argument("--output-beam", type=float, default=8.0,
                        help="k2.intersect_dense output_beam. Wider = more "
                             "candidates retained at lattice construction. "
                             "Default 8.0 matches LibriSpeech setup.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Encoder batch size (lattice ops still per-utt)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only first N utterances (0 = all)")
    parser.add_argument("--checkpoint-every", type=int, default=5000,
                        help="Save partial summary every N utterances")
    parser.add_argument("--batch-size-sweep", action="store_true",
                        help="Try multiple batch sizes on first 50 utts and report "
                             "the largest that fits + per-utt rate. Use this to "
                             "pick --batch-size for the full run.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"E21: Reranker training data  --  N-best generation")
    print("=" * 60)
    print(f"  Split:       {args.split}")
    print(f"  Device:      {device}")
    print(f"  Output:      {args.output_dir}")
    print(f"  num_paths:   {args.num_paths} (oversample)")
    print(f"  nbest_scale: {args.nbest_scale}")
    print(f"  max_keep:    {args.max_keep}")
    print(f"  batch_size:  {args.batch_size}")
    if args.limit > 0:
        print(f"   LIMIT: only first {args.limit} utterances")
    print()

    # Setup
    add_icefall_to_path(args.icefall_dir)

    print("Loading BPE tokenizer...")
    sp = spm.SentencePieceProcessor()
    sp.load(str(args.model_dir / "data" / "lang_bpe_500" / "bpe.model"))

    print("Loading model...")
    model = load_model(args.model_dir, args.icefall_dir, device)

    print(f"Loading cuts for {args.split}...")
    cuts = list(load_cuts(args.data_dir, args.split, cuts_path=args.cuts_path))
    if args.limit > 0:
        cuts = cuts[:args.limit]
    n_total = len(cuts)
    print(f"  {n_total} utterances")

    # CTC topology  --  built once on GPU
    import k2
    topo = k2.ctc_topo(max_token=MAX_TOKEN, modified=False, device=device)

    # Smoke test on first 10 utterances
    # CRITICAL: with nbest_scale=0.5 (diverse sampling), the top-1 by CTC
    # score from Nbest.from_lattice is NOT the greedy path  --  it's the
    # highest-scoring among the *sampled* diverse paths. Bad-looking top-1
    # is expected and useful for MWER training (the reranker learns to
    # demote it). The right pass criterion is GREEDY WER and ORACLE WER.
    print("\n=== Smoke test (first 10 utterances) ===")
    smoke_results, smoke_greedy = generate_nbest_batch(
        model, cuts[:10], sp, topo, device,
        args.num_paths, args.nbest_scale, args.max_keep,
        return_greedy=True, output_beam=args.output_beam,
    )

    greedy_edits = 0       # true argmax CTC
    oracle_edits = 0       # min edits across all N-best
    top1_edits = 0         # top-1 by CTC score from N-best (informational)
    total_ref_words = 0
    candidate_counts = []

    for i, (cut, cands, greedy_text) in enumerate(
        zip(cuts[:10], smoke_results, smoke_greedy)
    ):
        ref = ""
        for sup in cut.supervisions:
            ref = sup.text.lower().strip()
        ref_words = ref.split()
        ref_len = len(ref_words)
        total_ref_words += ref_len

        greedy_edits += editdistance.eval(greedy_text.split(), ref_words)
        top1 = cands[0]["text"] if cands else ""
        top1_edits += editdistance.eval(top1.split(), ref_words)
        if cands:
            cand_edits = [editdistance.eval(c["text"].split(), ref_words) for c in cands]
            oracle_edits += min(cand_edits)
        else:
            oracle_edits += ref_len

        candidate_counts.append(len(cands))

        print(f"  REF:    {ref[:70]}")
        print(f"  GREEDY: {greedy_text[:70]}")
        print(f"  N-best: {top1[:70]}  ({len(cands)} unique)")
        print()

    greedy_wer = greedy_edits / max(1, total_ref_words)
    oracle_wer = oracle_edits / max(1, total_ref_words)
    top1_wer = top1_edits / max(1, total_ref_words)
    mean_unique = sum(candidate_counts) / len(candidate_counts)

    print(f"  Greedy WER:    {greedy_wer*100:6.2f}%  (true argmax)")
    print(f"  Oracle WER:    {oracle_wer*100:6.2f}%  (best of {args.max_keep} N-best)")
    print(f"  N-best top-1:  {top1_wer*100:6.2f}%  (informational; high is OK)")
    print(f"  Mean unique:   {mean_unique:.1f}/{args.max_keep}")

    # Pass criteria: greedy must be sane, oracle must improve on greedy
    if greedy_wer > args.smoke_greedy_max:
        print(f"   SMOKE TEST FAILED: greedy WER = {greedy_wer*100:.2f}% > "
              f"{args.smoke_greedy_max*100:.0f}%. "
              f"Model/feature pipeline is broken. ABORTING.")
        sys.exit(1)
    if oracle_wer > greedy_wer + 1e-9:
        print(f"   WARNING: oracle ({oracle_wer*100:.2f}%) > greedy "
              f"({greedy_wer*100:.2f}%). N-best is missing the greedy path.")
    print(f"   Smoke test passed")

    # Optional: batch-size sweep
    if args.batch_size_sweep:
        print("\n=== Batch-size sweep on 50 utts ===")
        # Use last 50 utts (more variable lengths than first 50)
        sweep_cuts = cuts[:50] if len(cuts) >= 50 else cuts
        candidates = [16, 32, 48, 64, 96, 128, 160]
        results_table = []
        max_safe = None
        max_safe_rate = 0.0
        for bs in candidates:
            torch.cuda.empty_cache()
            try:
                t0 = time.time()
                for batch_start in range(0, len(sweep_cuts), bs):
                    batch = sweep_cuts[batch_start:batch_start + bs]
                    _ = generate_nbest_batch(
                        model, batch, sp, topo, device,
                        args.num_paths, args.nbest_scale, args.max_keep,
                        output_beam=args.output_beam,
                    )
                elapsed = time.time() - t0
                rate = len(sweep_cuts) / elapsed
                results_table.append((bs, "OK", rate, elapsed))
                print(f"  batch={bs:>4d}   OK   {rate:>5.1f} utt/s  "
                      f"({elapsed:.1f}s for {len(sweep_cuts)} utts)")
                max_safe = bs
                max_safe_rate = rate
            except torch.cuda.OutOfMemoryError as e:
                results_table.append((bs, "OOM", 0.0, 0.0))
                print(f"  batch={bs:>4d}   OOM")
                torch.cuda.empty_cache()
                break  # larger sizes will also fail
            except Exception as e:
                print(f"  batch={bs:>4d}   ERROR: {str(e)[:120]}")
                torch.cuda.empty_cache()
                break

        print()
        if max_safe:
            full_eta_min = n_total / max_safe_rate / 60
            print(f"  Recommended batch_size = {max_safe} "
                  f"({max_safe_rate:.1f} utt/s, full ETA ~ {full_eta_min:.0f} min)")
            print(f"  Run full with: --batch-size {max_safe}")
        else:
            print(f"   Even batch=16 failed. Try --batch-size 8 or 4.")
        print(f"\n  Sweep complete. Skipping full generation.")
        return

    # Full N-best generation
    if args.output_jsonl_name:
        output_jsonl = args.output_dir / args.output_jsonl_name
    else:
        output_jsonl = args.output_dir / f"{args.split}_G{args.max_keep}.jsonl"
    print(f"\n=== Generating N-best for {n_total} utterances ===")
    print(f"  Output: {output_jsonl}")

    t0 = time.time()
    total_edits_greedy = 0    # true argmax CTC
    total_edits_top1 = 0      # top-1 by CTC score from N-best
    total_edits_oracle = 0    # min over N-best
    total_ref_words = 0
    total_unique = 0
    wer_stds = []
    n_zero_var = 0
    written = 0

    with open(output_jsonl, "w") as f:
        for batch_start in range(0, n_total, args.batch_size):
            batch_cuts = cuts[batch_start:batch_start + args.batch_size]
            try:
                batch_cands, batch_greedy = generate_nbest_batch(
                    model, batch_cuts, sp, topo, device,
                    args.num_paths, args.nbest_scale, args.max_keep,
                    return_greedy=True, output_beam=args.output_beam,
                )
            except torch.cuda.OutOfMemoryError:
                # Fall back to per-utterance for this batch
                print(f"   OOM at batch {batch_start}, falling back to batch=1...")
                torch.cuda.empty_cache()
                batch_cands = []
                batch_greedy = []
                for cut in batch_cuts:
                    cands_i, greedy_i = generate_nbest_batch(
                        model, [cut], sp, topo, device,
                        args.num_paths, args.nbest_scale, args.max_keep,
                        return_greedy=True, output_beam=args.output_beam,
                    )
                    batch_cands.extend(cands_i)
                    batch_greedy.extend(greedy_i)

            for cut, cands, greedy_text in zip(batch_cuts, batch_cands, batch_greedy):
                ref = ""
                for sup in cut.supervisions:
                    ref = sup.text.lower().strip()
                ref_words = ref.split()
                ref_len = len(ref_words)

                # True greedy WER (argmax CTC)
                greedy_edits = editdistance.eval(greedy_text.split(), ref_words)

                hypotheses = []
                wer_per_cand = []
                for c in cands:
                    edits = editdistance.eval(c["text"].split(), ref_words)
                    hypotheses.append({
                        "text": c["text"],
                        "tokens": c["tokens"],
                        "ctc_log_prob": c["ctc_log_prob"],
                        "wer_edits": edits,
                        "wer_ref_len": ref_len,
                    })
                    wer_per_cand.append(edits / max(1, ref_len))

                # Stats
                total_edits_greedy += greedy_edits
                total_ref_words += ref_len
                if hypotheses:
                    top1_edits = hypotheses[0]["wer_edits"]
                    oracle_edits = min(h["wer_edits"] for h in hypotheses)
                    total_edits_top1 += top1_edits
                    total_edits_oracle += oracle_edits
                    total_unique += len(hypotheses)
                    if len(wer_per_cand) > 1:
                        std = statistics.pstdev(wer_per_cand)
                        wer_stds.append(std)
                        if std == 0:
                            n_zero_var += 1
                    else:
                        # Single hypothesis = no variance, no MWER signal
                        n_zero_var += 1
                else:
                    # No hypotheses = oracle defaults to greedy
                    total_edits_oracle += greedy_edits
                    n_zero_var += 1

                f.write(json.dumps({
                    "utterance_id": cut.id,
                    "reference": ref,
                    "greedy_text": greedy_text,
                    "greedy_wer_edits": greedy_edits,
                    "hypotheses": hypotheses,
                }) + "\n")
                written += 1

                # Progress every 500
                if written % 500 == 0:
                    elapsed = time.time() - t0
                    rate = written / elapsed
                    eta_min = (n_total - written) / rate / 60 if rate > 0 else 0
                    greedy_wer = total_edits_greedy / max(1, total_ref_words) * 100
                    top1_wer = total_edits_top1 / max(1, total_ref_words) * 100
                    oracle_wer = total_edits_oracle / max(1, total_ref_words) * 100
                    print(
                        f"  {written}/{n_total}  "
                        f"({rate:.1f}/s, ETA {eta_min:.1f}min) "
                        f"greedy={greedy_wer:.2f}% top1={top1_wer:.2f}% "
                        f"oracle={oracle_wer:.2f}%"
                    )

                # Checkpoint summary
                if written % args.checkpoint_every == 0:
                    ckpt = {
                        "num_utterances": written,
                        "running_greedy_wer": total_edits_greedy / max(1, total_ref_words),
                        "running_top1_wer": total_edits_top1 / max(1, total_ref_words),
                        "running_oracle_wer": total_edits_oracle / max(1, total_ref_words),
                        "mean_hypotheses_per_utt": total_unique / written,
                        "zero_variance_count": n_zero_var,
                    }
                    with open(args.output_dir / f"checkpoint_{written}.json", "w") as cf:
                        json.dump(ckpt, cf, indent=2)
                    print(f"   Checkpoint saved: {written} utterances")

    # Final summary
    elapsed = time.time() - t0
    summary = {
        "split": args.split,
        "num_utterances": written,
        "num_paths_oversample": args.num_paths,
        "nbest_scale": args.nbest_scale,
        "max_keep": args.max_keep,
        "mean_hypotheses_per_utt": total_unique / max(1, written),
        "mean_greedy_wer": total_edits_greedy / max(1, total_ref_words),
        "mean_top1_nbest_wer": total_edits_top1 / max(1, total_ref_words),
        "mean_oracle_wer": total_edits_oracle / max(1, total_ref_words),
        "mean_wer_std_per_utt": (
            sum(wer_stds) / max(1, len(wer_stds))
        ),
        "utterances_with_zero_variance": n_zero_var,
        "total_edits_greedy": total_edits_greedy,
        "total_edits_top1": total_edits_top1,
        "total_edits_oracle": total_edits_oracle,
        "total_ref_words": total_ref_words,
        "elapsed_seconds": elapsed,
        "output_jsonl": str(output_jsonl),
        "output_size_bytes": output_jsonl.stat().st_size,
    }

    summary_path = args.output_dir / "data_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 60)
    print(f"DONE in {elapsed/60:.1f} min")
    print("=" * 60)
    print(f"  Greedy WER:      {summary['mean_greedy_wer']*100:.2f}%  (true argmax)")
    print(f"  N-best top-1:    {summary['mean_top1_nbest_wer']*100:.2f}%  (sampled, informational)")
    print(f"  Oracle WER:      {summary['mean_oracle_wer']*100:.2f}%  (best of N-best)")
    print(f"  Mean unique:     {summary['mean_hypotheses_per_utt']:.1f}/{args.max_keep}")
    print(f"  Mean WER std:    {summary['mean_wer_std_per_utt']*100:.2f}pp")
    print(f"  Zero-variance:   {n_zero_var}/{written} "
          f"({n_zero_var/max(1,written)*100:.1f}%)")
    print(f"  Output JSONL:    {output_jsonl}")
    print(f"                   {summary['output_size_bytes']/1e6:.0f} MB")
    print(f"  Summary:         {summary_path}")


if __name__ == "__main__":
    main()

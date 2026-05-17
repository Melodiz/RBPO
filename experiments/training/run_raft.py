#!/usr/bin/env python3
"""E26-RAFT: Best-of-N distillation on standard CTC.

Phases:
  1   Generate training N-best (or load existing) and extract oracle
      transcripts. Saves raft_oracle_selection.json + oracle JSONL.
  2   Run RAFT training with one or more lambda values.
      Loss: L = (1-lam) * L_CTC(y_oracle) + lam * L_CTC(y_gold)
      All CTC losses are LENGTH-NORMALIZED (E26 fix is mandatory).
  3   Final eval + paired bootstrap (B=10000, seed=42) vs original
      standard CTC baseline on dev-other.
  4   Diagnostic G=16 oracle/Spearman on the RAFT model (only if any
      lam improves dev-other WER).

Usage (Colab, 5000-utt run):
    python experiments/training/run_raft.py \
        --checkpoint /content/standard_ctc_model/exp/pretrained.pt \
        --bpe /content/standard_ctc_model/data/lang_bpe_500/bpe.model \
        --icefall-dir /content/icefall \
        --data-dir /content/librispeech_data \
        --output-dir /content/drive/MyDrive/rbpo_results/standard_ctc \
        --recipe zipformer --model-size small \
        --train-subset 5000 \
        --lambda-values 0.5,0.3,0.0 \
        --phase 1,2,3
"""

import argparse
import copy
import json
import math
import re
import sys
import time
from pathlib import Path

BLANK_ID = 0
MAX_TOKEN = 499
_TAG_RE = re.compile(r"\{[^}]+\}|<[^>]+>")
_MULTI_SPACE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

MODEL_PRESETS = {
    "small": {
        "num_encoder_layers": "2,2,2,2,2,2",
        "encoder_dim": "192,256,256,256,256,256",
        "encoder_unmasked_dim": "192,192,192,192,192,192",
        "feedforward_dim": "512,768,768,768,768,768",
    },
    "medium": {
        "num_encoder_layers": "2,2,3,4,3,2",
        "encoder_dim": "384,384,384,384,384,384",
        "encoder_unmasked_dim": "256,256,256,256,256,256",
        "feedforward_dim": "1536,1536,1536,1536,1536,1536",
    },
    "large": {
        "num_encoder_layers": "2,2,4,5,4,2",
        "encoder_dim": "512,512,512,512,512,512",
        "encoder_unmasked_dim": "384,384,384,384,384,384",
        "feedforward_dim": "2048,2048,2048,2048,2048,2048",
    },
}

def normalize_text(text: str) -> str:
    text = _TAG_RE.sub(" ", text)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text

def ctc_collapse(token_ids):
    result = []
    prev = None
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

def compute_wer(hypothesis: str, reference: str) -> float:
    import editdistance
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return editdistance.eval(hyp_words, ref_words) / len(ref_words)

def corpus_wer_from_lists(hyps, refs):
    """Corpus-level WER: sum(edits) / sum(ref_words)."""
    import editdistance
    total_edits = 0
    total_ref = 0
    n = 0
    for h, r in zip(hyps, refs):
        if not r.strip():
            continue
        rw = r.split()
        total_edits += editdistance.eval(h.split(), rw)
        total_ref += len(rw)
        n += 1
    return total_edits / max(1, total_ref), n

def greedy_ctc_decode_batch(log_probs):
    argmax_ids = log_probs.argmax(dim=-1)
    out = []
    for seq in argmax_ids:
        toks = []
        prev = -1
        for t in seq.tolist():
            if t != 0 and t != prev:
                toks.append(t)
            prev = t
        out.append(toks)
    return out

def add_icefall_to_path(icefall_dir: Path, recipe: str):
    dirs = [
        icefall_dir,
        icefall_dir / "egs" / "librispeech" / "ASR",
        icefall_dir / "egs" / "librispeech" / "ASR" / recipe,
    ]
    for d in dirs:
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))

def load_model(args, device):
    import torch

    add_icefall_to_path(Path(args.icefall_dir), args.recipe)
    import train as train_module
    add_model_arguments = train_module.add_model_arguments
    get_params = train_module.get_params
    _get_model = getattr(train_module, "get_ctc_model",
                         getattr(train_module, "get_model", None))
    assert _get_model is not None

    params = get_params()
    parser = argparse.ArgumentParser(add_help=False)
    add_model_arguments(parser)
    model_args = parser.parse_args([])
    for k, v in vars(model_args).items():
        params[k] = v

    preset = MODEL_PRESETS[args.model_size]
    params.num_encoder_layers = preset["num_encoder_layers"]
    params.encoder_dim = preset["encoder_dim"]
    params.encoder_unmasked_dim = preset["encoder_unmasked_dim"]
    params.feedforward_dim = preset["feedforward_dim"]
    params.vocab_size = args.vocab_size
    params.feature_dim = 80
    for flag, val in [
        ("use_transducer", False),
        ("use_ctc", True),
        ("use_cr_ctc", args.use_cr_ctc),
        ("use_attention_decoder", False),
    ]:
        if hasattr(params, flag):
            setattr(params, flag, val)

    model = _get_model(params)
    checkpoint = torch.load(
        str(args.checkpoint), map_location="cpu", weights_only=False
    )
    state_dict = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"  INFO: {len(unexpected)} unexpected keys ignored")

    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params / 1e6:.1f}M params, "
          f"{'CR-CTC' if args.use_cr_ctc else 'standard CTC'}, {args.model_size}")
    return model

def log_p_ctc(log_probs, token_ids, T, output_beam, device):
    """Differentiable log P_CTC(y|x) via k2 numerator lattice."""
    import k2
    import torch

    sup = torch.tensor([[0, 0, T]], dtype=torch.int32)
    dense = k2.DenseFsaVec(log_probs[:, :T, :], sup)
    ctc_graph = k2.ctc_graph([token_ids], modified=False, device=device)
    lat = k2.intersect_dense(ctc_graph, dense, output_beam=output_beam)
    return lat.get_tot_scores(log_semiring=True, use_double_scores=True)[0]

def ce_loss_normalized(log_probs, token_ids, T, output_beam, device):
    raw = -log_p_ctc(log_probs, token_ids, T, output_beam, device)
    return raw / max(len(token_ids), 1)

def generate_nbest_for_utt(log_probs_utt, topo, num_paths, sp, device):
    import k2
    import torch

    T = log_probs_utt.shape[0]
    lp_cpu = log_probs_utt.cpu()

    greedy_ids = log_probs_utt.argmax(dim=-1).cpu().tolist()
    greedy_collapsed = ctc_collapse(greedy_ids)
    greedy_text = normalize_text(sp.decode(greedy_collapsed))
    greedy_score = alignment_log_prob(greedy_ids, lp_cpu)

    sup = torch.tensor([[0, 0, T]], dtype=torch.int32)
    dense = k2.DenseFsaVec(log_probs_utt.unsqueeze(0), sup)
    lattice = k2.intersect_dense(topo, dense, output_beam=8.0)
    lattice = k2.connect(lattice)

    nbest = k2.Nbest.from_lattice(
        lattice, num_paths=num_paths,
        use_double_scores=True, nbest_scale=1.0,
    )
    all_labels = nbest.fsa.labels.cpu().tolist()
    paths, current = [], []
    for label in all_labels:
        if label == -1:
            paths.append(current)
            current = []
        else:
            current.append(label)

    seen = {}
    for raw_ids in paths:
        score = alignment_log_prob(raw_ids, lp_cpu)
        if score == float("-inf"):
            continue
        token_ids = ctc_collapse(raw_ids)
        text = normalize_text(sp.decode(token_ids))
        if not text:
            continue
        entry = {"hyp": text, "score": round(score, 6)}
        if text not in seen or score > seen[text]["score"]:
            seen[text] = entry

    greedy_entry = {"hyp": greedy_text, "score": round(greedy_score, 6)}
    seen[greedy_text] = greedy_entry
    candidates = sorted(seen.values(), key=lambda c: c["score"], reverse=True)
    rest = [c for c in candidates if c["hyp"] != greedy_text]
    candidates = [greedy_entry] + rest

    del lattice, nbest
    return candidates

def phase1_oracle_selection(args, model, sp, device):
    nbest_path = Path(args.output_dir) / f"nbest_train_g{args.G}_n{args.train_subset}.jsonl"
    oracle_path = Path(args.output_dir) / f"raft_oracle_transcripts_n{args.train_subset}.jsonl"
    stats_path = Path(args.output_dir) / "raft_oracle_selection.json"

    if oracle_path.exists() and stats_path.exists() and not args.force:
        print(f"Phase 1: SKIP (oracle file exists: {oracle_path})")
        return

    print("\n" + "=" * 70)
    print("Phase 1: Oracle extraction from training N-best")
    print("=" * 70)

    import torch
    import k2
    import editdistance
    from lhotse import Fbank, FbankConfig, load_manifest_lazy

    cuts_path = Path(args.data_dir) / "cuts" / "librispeech_cuts_train-clean-100.jsonl.gz"
    assert cuts_path.exists(), f"Train cuts not found: {cuts_path}"
    cuts = list(load_manifest_lazy(str(cuts_path)))
    cuts = cuts[:args.train_subset] if args.train_subset > 0 else cuts
    print(f"  {len(cuts)} utterances, G={args.G}, scale=1.0")

    # Generate N-best if not present (~12 min for 5000 utts on T4)
    if nbest_path.exists() and not args.force:
        print(f"  N-best JSONL exists, loading: {nbest_path}")
        records = []
        with open(nbest_path) as f:
            for line in f:
                records.append(json.loads(line))
    else:
        print(f"  Generating N-best lists...")
        fbank = Fbank(FbankConfig(num_mel_bins=80, dither=0.0))
        topo = k2.ctc_topo(max_token=MAX_TOKEN, modified=False, device=device)
        records = []
        nbest_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        model.eval()

        with open(nbest_path, "w") as f_out:
            for i, cut in enumerate(cuts):
                audio = cut.load_audio()
                feat = fbank.extract(audio, sampling_rate=16000)
                feat_t = torch.from_numpy(feat).unsqueeze(0).to(device)
                feat_lens = torch.tensor(
                    [feat.shape[0]], dtype=torch.int64, device=device
                )

                with torch.no_grad():
                    enc_out, enc_lens = model.forward_encoder(feat_t, feat_lens)
                    log_probs = model.ctc_output(enc_out)

                lp_utt = log_probs[0, :enc_lens[0].item()]
                cands = generate_nbest_for_utt(
                    lp_utt, topo, args.G * 4, sp, device
                )
                cands = cands[:args.G]

                ref_raw = " ".join(s.text for s in cut.supervisions if s.text)
                ref = normalize_text(ref_raw)
                rec = {"utt_id": cut.id, "ref": ref, "nbest": cands}
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                records.append(rec)

                del log_probs, enc_out, feat_t
                torch.cuda.empty_cache()

                if (i + 1) % 200 == 0 or i == len(cuts) - 1:
                    el = time.time() - t0
                    print(f"    {i+1}/{len(cuts)} ({(i+1)/el:.1f} utt/s, "
                          f"ETA {(len(cuts)-i-1)/((i+1)/el):.0f}s)")

        print(f"  N-best done: {len(records)} utts in {time.time()-t0:.0f}s")

    print("\n  Selecting oracle transcripts (min-WER per utterance)...")
    n_recoverable = 0
    n_eq_gold = 0
    n_neq_gold = 0
    total_oracle_edits = 0
    total_greedy_edits = 0
    total_ref_words = 0
    oracle_records = []

    for rec in records:
        ref = rec["ref"]
        ref_words = ref.split()
        if not ref_words:
            continue
        n_ref = len(ref_words)
        total_ref_words += n_ref

        nbest = rec["nbest"]
        greedy_hyp = nbest[0]["hyp"]
        greedy_edits = editdistance.eval(greedy_hyp.split(), ref_words)
        total_greedy_edits += greedy_edits

        best_edits = greedy_edits
        best_hyp = greedy_hyp
        for c in nbest:
            e = editdistance.eval(c["hyp"].split(), ref_words)
            if e < best_edits:
                best_edits = e
                best_hyp = c["hyp"]
        total_oracle_edits += best_edits

        if best_hyp != greedy_hyp:
            n_recoverable += 1
        if best_hyp == ref:
            n_eq_gold += 1
        else:
            n_neq_gold += 1

        oracle_records.append({
            "utt_id": rec["utt_id"],
            "ref": ref,
            "oracle_hyp": best_hyp,
            "greedy_hyp": greedy_hyp,
            "oracle_edits": best_edits,
            "greedy_edits": greedy_edits,
            "n_ref_words": n_ref,
            "recoverable": best_hyp != greedy_hyp,
        })

    greedy_wer = total_greedy_edits / max(1, total_ref_words)
    oracle_wer = total_oracle_edits / max(1, total_ref_words)

    with open(oracle_path, "w") as f_out:
        for r in oracle_records:
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "n_utterances": len(oracle_records),
        "n_recoverable": n_recoverable,
        "recoverable_pct": round(n_recoverable / max(1, len(oracle_records)) * 100, 2),
        "n_oracle_eq_gold": n_eq_gold,
        "n_oracle_neq_gold": n_neq_gold,
        "mean_oracle_wer": oracle_wer,
        "mean_greedy_wer": greedy_wer,
        "oracle_gap_pp": (greedy_wer - oracle_wer) * 100,
        "G": args.G,
        "train_subset": args.train_subset,
        "oracle_jsonl_path": str(oracle_path),
        "nbest_jsonl_path": str(nbest_path),
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n  Greedy WER:      {greedy_wer:.4%}")
    print(f"  Oracle WER:      {oracle_wer:.4%}")
    print(f"  Oracle gap:      {(greedy_wer - oracle_wer) * 100:.3f} pp")
    print(f"  Recoverable:     {n_recoverable}/{len(oracle_records)} "
          f"({n_recoverable/max(1,len(oracle_records))*100:.1f}%)")
    print(f"  Oracle == gold:  {n_eq_gold}")
    print(f"  Oracle != gold:  {n_neq_gold}")
    print(f"\n  Saved oracle transcripts: {oracle_path}")
    print(f"  Saved stats: {stats_path}")

def phase2_raft_train(args, lambda_val, model, sp, device, original_baseline):
    tag = f"lambda{int(lambda_val * 100):02d}"
    cfg_path = Path(args.output_dir) / f"raft_{tag}_config.json"
    log_path = Path(args.output_dir) / f"raft_{tag}_training_log.jsonl"
    report_path = Path(args.output_dir) / f"raft_{tag}_smoke_report.json"
    ckpt_path = Path(args.output_dir) / f"raft_{tag}_checkpoint.pt"

    if report_path.exists() and not args.force:
        print(f"\nPhase 2 [{tag}]: SKIP (exists: {report_path})")
        return json.load(open(report_path))

    print("\n" + "=" * 70)
    print(f"Phase 2 [{tag}]: RAFT training (lam={lambda_val})")
    print("=" * 70)

    import torch
    import editdistance
    from lhotse import Fbank, FbankConfig, load_manifest_lazy
    from torch.optim import AdamW

    config = {
        "lambda": lambda_val,
        "loss": f"L = {1-lambda_val:.2f} * L_CTC(oracle) + {lambda_val:.2f} * L_CTC(gold), length-normalized",
        "lr": args.raft_lr,
        "G": args.G,
        "grad_accum": args.raft_grad_accum,
        "max_grad_norm": 5.0,
        "model_size": args.model_size,
        "use_cr_ctc": args.use_cr_ctc,
        "train_subset": args.train_subset,
        "eval_every_steps": args.raft_eval_every,
        "max_steps": args.raft_max_steps,
        "variant": args.raft_variant,
    }
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)

    oracle_path = Path(args.output_dir) / f"raft_oracle_transcripts_n{args.train_subset}.jsonl"
    assert oracle_path.exists(), f"Run Phase 1 first: {oracle_path} missing"
    oracle_records = []
    with open(oracle_path) as f:
        for line in f:
            oracle_records.append(json.loads(line))
    oracle_by_id = {r["utt_id"]: r for r in oracle_records}
    print(f"  Loaded {len(oracle_records)} oracle records")

    # Variant filter (optional)
    if args.raft_variant == "filtered":
        recoverable_ids = {r["utt_id"] for r in oracle_records if r["recoverable"]}
        non_rec = [r["utt_id"] for r in oracle_records if not r["recoverable"]]
        # Sample equal-sized non-recoverable subset for anchoring
        import random
        random.seed(42)
        sampled_non_rec = random.sample(non_rec, min(len(recoverable_ids), len(non_rec)))
        active_ids = recoverable_ids | set(sampled_non_rec)
        print(f"  Variant=filtered: {len(active_ids)} active utts "
              f"({len(recoverable_ids)} recoverable + {len(sampled_non_rec)} sampled)")
    else:
        active_ids = set(oracle_by_id.keys())
        print(f"  Variant=A (all): {len(active_ids)} utts")

    cuts_path = Path(args.data_dir) / "cuts" / "librispeech_cuts_train-clean-100.jsonl.gz"
    train_cuts = list(load_manifest_lazy(str(cuts_path)))
    train_cuts = [c for c in train_cuts if c.id in active_ids]
    print(f"  Training on {len(train_cuts)} cuts")

    fbank = Fbank(FbankConfig(num_mel_bins=80, dither=0.0))

    # Eval helper
    def eval_dev(split):
        sp_path = Path(args.data_dir) / "cuts" / f"librispeech_cuts_{split}.jsonl.gz"
        if not sp_path.exists():
            return None, 0
        eval_cuts = list(load_manifest_lazy(str(sp_path)))
        if args.raft_eval_utts > 0:
            eval_cuts = eval_cuts[:args.raft_eval_utts]
        hyps, refs_e = [], []
        model.eval()
        for ec in eval_cuts:
            audio = ec.load_audio()
            feat = fbank.extract(audio, sampling_rate=16000)
            ft = torch.from_numpy(feat).unsqueeze(0).to(device)
            fl = torch.tensor([feat.shape[0]], dtype=torch.int64, device=device)
            with torch.no_grad():
                eo, _ = model.forward_encoder(ft, fl)
                lp = model.ctc_output(eo)
            toks = greedy_ctc_decode_batch(lp)
            hyps.append(sp.decode(toks[0]).strip().lower())
            refs_e.append(
                " ".join(s.text for s in ec.supervisions if s.text).strip().lower()
            )
            del eo, lp, ft
            torch.cuda.empty_cache()
        return corpus_wer_from_lists(hyps, refs_e)

    # Baseline eval
    print(f"\n  Baseline eval (epoch 0)...")
    baseline_metrics = {}
    for split in ["dev-other", "dev-clean"]:
        wer, n = eval_dev(split)
        baseline_metrics[split] = {"wer": wer, "num_utts": n}
        if wer is not None:
            print(f"    {split}: WER={wer:.4%} ({n} utts)")

    # Training setup
    optimizer = AdamW(model.parameters(), lr=args.raft_lr, weight_decay=0.01)

    log_f = open(log_path, "w")
    step = 0
    acc_count = 0
    accum_total = []
    accum_oracle = []
    accum_gold = []
    eval_log = []
    skipped = 0
    t_epoch = time.time()

    optimizer.zero_grad()
    epoch = 0
    reached_max = False

    while not reached_max:
      epoch += 1
      if epoch > 1:
          print(f"\n  --- Epoch {epoch} (step {step}) ---")

      for utt_idx, cut in enumerate(train_cuts):
        ref_raw = " ".join(s.text for s in cut.supervisions if s.text)
        ref = normalize_text(ref_raw)
        if not ref:
            continue
        oracle_rec = oracle_by_id.get(cut.id)
        if oracle_rec is None:
            continue
        oracle_hyp = oracle_rec["oracle_hyp"]

        oracle_token_ids = sp.encode(oracle_hyp, out_type=int)
        gold_token_ids = sp.encode(ref, out_type=int)
        if not oracle_token_ids or not gold_token_ids:
            skipped += 1
            continue

        audio = cut.load_audio()
        feat = fbank.extract(audio, sampling_rate=16000)
        feat_t = torch.from_numpy(feat).unsqueeze(0).to(device)
        feat_lens = torch.tensor([feat.shape[0]], dtype=torch.int64, device=device)

        try:
            enc_out, enc_lens = model.forward_encoder(feat_t, feat_lens)
            log_probs = model.ctc_output(enc_out)
            T = enc_lens[0].item()

            loss_oracle = ce_loss_normalized(
                log_probs, oracle_token_ids, T, 8.0, device
            )

            if lambda_val > 0:
                loss_gold = ce_loss_normalized(
                    log_probs, gold_token_ids, T, 8.0, device
                )
                total_loss = (1 - lambda_val) * loss_oracle + lambda_val * loss_gold
                gold_val = loss_gold.item()
            else:
                total_loss = loss_oracle
                gold_val = 0.0

            if not torch.isfinite(total_loss):
                skipped += 1
                continue

            (total_loss / args.raft_grad_accum).backward()
            acc_count += 1
            accum_total.append(total_loss.item())
            accum_oracle.append(loss_oracle.item())
            accum_gold.append(gold_val)

        except Exception as e:
            print(f"  [utt {utt_idx}] ERROR: {e}")
            torch.cuda.empty_cache()
            continue

        del log_probs, enc_out, feat_t
        torch.cuda.empty_cache()

        if acc_count >= args.raft_grad_accum:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=5.0
            ).item()
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            agg_total = sum(accum_total) / len(accum_total)
            agg_oracle = sum(accum_oracle) / len(accum_oracle)
            agg_gold = sum(accum_gold) / len(accum_gold)

            log_line = {
                "step": step, "loss_total": round(agg_total, 4),
                "loss_oracle": round(agg_oracle, 4),
                "loss_gold": round(agg_gold, 4),
                "grad_norm": round(grad_norm, 4),
                "skipped": skipped,
            }
            log_f.write(json.dumps(log_line) + "\n")
            log_f.flush()

            if step == 1 or step % 50 == 0:
                print(f"  [step {step}] L={agg_total:.3f} "
                      f"L_oracle={agg_oracle:.3f} L_gold={agg_gold:.3f} "
                      f"g={grad_norm:.2f}")

            # Mid-eval
            if args.raft_eval_every > 0 and step % args.raft_eval_every == 0:
                print(f"\n  Eval at step {step}...")
                wer, n = eval_dev("dev-other")
                eval_log.append({"step": step, "dev-other": {"wer": wer, "num_utts": n}})
                print(f"    dev-other: WER={wer:.4%} ({n} utts)")

            acc_count = 0
            accum_total, accum_oracle, accum_gold = [], [], []
            skipped = 0

            if args.raft_max_steps > 0 and step >= args.raft_max_steps:
                print(f"\n  Stopping at max_steps={args.raft_max_steps}")
                reached_max = True
                break

      if not reached_max and args.raft_max_steps <= 0:
          break  # single epoch when no max_steps set

    log_f.close()
    epoch_time = time.time() - t_epoch

    # Final eval
    print(f"\n  Final eval after {step} steps ({epoch_time:.0f}s)...")
    final_metrics = {}
    for split in ["dev-other", "dev-clean"]:
        wer, n = eval_dev(split)
        final_metrics[split] = {"wer": wer, "num_utts": n}
        if wer is not None:
            print(f"    {split}: WER={wer:.4%} ({n} utts)")

    torch.save({"model": model.state_dict(), "step": step, "lambda": lambda_val},
               ckpt_path)
    print(f"  Checkpoint: {ckpt_path}")

    base_do = baseline_metrics.get("dev-other", {}).get("wer")
    final_do = final_metrics.get("dev-other", {}).get("wer")
    rel_change = None
    if base_do and final_do:
        rel_change = (final_do - base_do) / base_do * 100

    report = {
        "lambda": lambda_val,
        "config": config,
        "total_steps": step,
        "epoch_time_s": round(epoch_time, 1),
        "baseline_eval": baseline_metrics,
        "final_eval": final_metrics,
        "eval_log": eval_log,
        "relative_change_pct": round(rel_change, 2) if rel_change else None,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  lam={lambda_val} summary:")
    print(f"    dev-other: {base_do:.4%} -> {final_do:.4%} "
          f"({rel_change:+.2f}%)" if rel_change else "")
    print(f"    Saved: {report_path}")
    return report

def phase3_eval_bootstrap(args, sp, device):
    out_path = Path(args.output_dir) / "raft_eval_bootstrap.json"
    if out_path.exists() and not args.force:
        print(f"Phase 3: SKIP (exists: {out_path})")
        return

    print("\n" + "=" * 70)
    print("Phase 3: Per-utterance eval + paired bootstrap")
    print("=" * 70)

    import torch
    import editdistance
    import numpy as np
    from lhotse import Fbank, FbankConfig, load_manifest_lazy

    fbank = Fbank(FbankConfig(num_mel_bins=80, dither=0.0))

    def per_utt_decode(model):
        sp_path = Path(args.data_dir) / "cuts" / "librispeech_cuts_dev-other.jsonl.gz"
        eval_cuts = list(load_manifest_lazy(str(sp_path)))
        edits, refs_n = [], []
        model.eval()
        for ec in eval_cuts:
            audio = ec.load_audio()
            feat = fbank.extract(audio, sampling_rate=16000)
            ft = torch.from_numpy(feat).unsqueeze(0).to(device)
            fl = torch.tensor([feat.shape[0]], dtype=torch.int64, device=device)
            with torch.no_grad():
                eo, _ = model.forward_encoder(ft, fl)
                lp = model.ctc_output(eo)
            toks = greedy_ctc_decode_batch(lp)
            hyp = sp.decode(toks[0]).strip().lower()
            ref = " ".join(s.text for s in ec.supervisions if s.text).strip().lower()
            ref_w = ref.split()
            edits.append(editdistance.eval(hyp.split(), ref_w))
            refs_n.append(len(ref_w))
            del eo, lp, ft
            torch.cuda.empty_cache()
        return np.array(edits), np.array(refs_n, dtype=np.float64)

    # Original baseline
    print("\n  Decoding with ORIGINAL baseline...")
    model = load_model(args, device)
    base_edits, ref_n = per_utt_decode(model)
    base_wer = base_edits.sum() / ref_n.sum()
    print(f"  Baseline dev-other WER = {base_wer:.4%}")

    results = {
        "original": {"wer": float(base_wer), "n_utts": int(len(ref_n))},
    }

    # Each lambda checkpoint
    rng = np.random.default_rng(42)
    B = 10000
    for lam in args.lambda_values:
        tag = f"lambda{int(lam * 100):02d}"
        ckpt = Path(args.output_dir) / f"raft_{tag}_checkpoint.pt"
        if not ckpt.exists():
            print(f"  SKIP {tag}: no checkpoint")
            continue
        print(f"\n  Decoding with RAFT {tag}...")
        state = torch.load(str(ckpt), map_location="cpu")["model"]
        model.load_state_dict(state, strict=False)
        model.to(device)
        raft_edits, _ = per_utt_decode(model)
        raft_wer = raft_edits.sum() / ref_n.sum()

        # Paired bootstrap
        n = len(ref_n)
        boot_deltas = np.zeros(B)
        for b in range(B):
            idx = rng.integers(0, n, size=n)
            base_b = base_edits[idx].sum() / ref_n[idx].sum()
            raft_b = raft_edits[idx].sum() / ref_n[idx].sum()
            boot_deltas[b] = base_b - raft_b
        p_value = float((boot_deltas <= 0).mean())

        results[tag] = {
            "lambda": lam,
            "wer": float(raft_wer),
            "delta_pp": float((base_wer - raft_wer) * 100),
            "p_value": round(p_value, 4),
            "significant_005": p_value < 0.05,
            "n_utts": n,
            "bootstrap_B": B,
        }
        sig = "*" if p_value < 0.05 else " "
        print(f"  RAFT {tag}: WER={raft_wer:.4%} "
              f"(delta={(base_wer - raft_wer)*100:+.3f}pp, p={p_value:.4f} {sig})")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="E26-RAFT: Best-of-N distillation on standard CTC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bpe", type=Path, required=True)
    parser.add_argument("--icefall-dir", type=Path, default=Path("/content/icefall"))
    parser.add_argument("--recipe", type=str, default="zipformer",
                        choices=["zipformer", "zipformer_ctc"])
    parser.add_argument("--model-size", type=str, default="small",
                        choices=["small", "medium", "large"])
    parser.add_argument("--use-cr-ctc", action="store_true", default=False)
    parser.add_argument("--vocab-size", type=int, default=500)

    # Data
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    # Phase
    parser.add_argument("--phase", type=str, default="1,2,3")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")

    # Phase 1
    parser.add_argument("--G", type=int, default=16)
    parser.add_argument("--train-subset", type=int, default=5000,
                        help="Use first N train-clean-100 utts (0=all)")

    # Phase 2 (RAFT)
    parser.add_argument("--lambda-values", type=str, default="0.5,0.3,0.0",
                        help="Comma-separated list of lam values to sweep")
    parser.add_argument("--raft-lr", type=float, default=1e-6)
    parser.add_argument("--raft-grad-accum", type=int, default=4)
    parser.add_argument("--raft-eval-every", type=int, default=200)
    parser.add_argument("--raft-eval-utts", type=int, default=0,
                        help="Mid-training eval on first N utts (0=all)")
    parser.add_argument("--raft-max-steps", type=int, default=0,
                        help="Stop training after N steps (0=full subset)")
    parser.add_argument("--raft-variant", type=str, default="A",
                        choices=["A", "filtered"],
                        help="A=all utts, filtered=recoverable+sampled")

    args = parser.parse_args()
    args.lambda_values = [float(x) for x in args.lambda_values.split(",")]

    print("=" * 70)
    print("E26-RAFT: Best-of-N distillation on standard CTC")
    print("=" * 70)
    print(f"  checkpoint:    {args.checkpoint}")
    print(f"  recipe:        {args.recipe}")
    print(f"  model_size:    {args.model_size}")
    print(f"  train_subset:  {args.train_subset}")
    print(f"  lambda values: {args.lambda_values}")
    print(f"  variant:       {args.raft_variant}")
    print(f"  phase:         {args.phase}")
    print()

    phases = set(args.phase.split(","))

    import torch
    import sentencepiece as spm

    device = torch.device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor()
    sp.load(str(args.bpe))
    assert sp.get_piece_size() == args.vocab_size

    # Phase 1
    if "1" in phases:
        print("Loading model for Phase 1...")
        model = load_model(args, device)
        phase1_oracle_selection(args, model, sp, device)
        del model
        torch.cuda.empty_cache()

    # Phase 2  --  one lam at a time, fresh model each time
    if "2" in phases:
        for lam in args.lambda_values:
            print(f"\nLoading FRESH model for lam={lam}...")
            model = load_model(args, device)
            original_baseline = None
            phase2_raft_train(args, lam, model, sp, device, original_baseline)
            del model
            torch.cuda.empty_cache()

    # Phase 3  --  eval all checkpoints with paired bootstrap
    if "3" in phases:
        phase3_eval_bootstrap(args, sp, device)

    print("\n" + "=" * 70)
    print("E26-RAFT phases complete.")
    print(f"  Results in: {args.output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""E26: Standard CTC  --  Phases 5-6 (MWER training + conditional RAFT).

Phase 5a: Training N-best statistics
Phase 5b: MWER training (1 epoch, match exp_F3 config)
Phase 5c: Post-training evaluation
Phase 6:  RAFT/BoN distillation (only if train oracle gap > 0.1 pp)

Usage (Colab):
    python experiments/training/run_mwer.py \
        --checkpoint /content/standard_ctc_model/exp/pretrained.pt \
        --bpe /content/standard_ctc_model/data/lang_bpe_500/bpe.model \
        --icefall-dir /content/icefall \
        --data-dir /content/librispeech_data \
        --output-dir /content/drive/MyDrive/rbpo_results/standard_ctc \
        --recipe zipformer_ctc \
        --model-size medium \
        --phase 5a,5b,5c
"""

import argparse
import copy
import json
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
    assert _get_model is not None, (
        f"Neither get_ctc_model nor get_model found in {args.recipe}/train.py"
    )

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
    if missing:
        print(f"  WARNING: {len(missing)} missing keys")
    if unexpected:
        print(f"  INFO: {len(unexpected)} unexpected keys ignored")

    model.eval().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params / 1e6:.1f}M params, "
          f"{'CR-CTC' if args.use_cr_ctc else 'standard CTC'}, "
          f"{args.model_size}")
    return model

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
    wer = total_edits / max(1, total_ref)
    return wer, n

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

def generate_candidates_beam(log_probs, topo, sp, G, output_beam, device):
    import k2

    T = log_probs.shape[0]
    sup = __import__("torch").tensor([[0, 0, T]], dtype=__import__("torch").int32)
    dense = k2.DenseFsaVec(log_probs.unsqueeze(0), sup)
    lattice = k2.intersect_dense(topo, dense, output_beam=output_beam)
    lattice = k2.connect(lattice)

    nbest = k2.Nbest.from_lattice(
        lattice, num_paths=max(G * 4, 16),
        use_double_scores=True, nbest_scale=1.0,
    )
    all_labels = nbest.fsa.labels.cpu().tolist()
    paths, cur = [], []
    for lbl in all_labels:
        if lbl == -1:
            paths.append(cur)
            cur = []
        else:
            cur.append(lbl)

    seen = set()
    out = []
    for raw in paths:
        toks = ctc_collapse(raw)
        if not toks:
            continue
        text = sp.decode(toks).strip().lower()
        if text in seen:
            continue
        seen.add(text)
        out.append((text, toks))

    greedy_raw = log_probs.argmax(dim=-1).tolist()
    greedy_toks = ctc_collapse(greedy_raw)
    if greedy_toks:
        greedy_text = sp.decode(greedy_toks).strip().lower()
        out = [(t, ids) for t, ids in out if t != greedy_text]
        out.insert(0, (greedy_text, greedy_toks))

    return out[:G]

def log_p_ctc(log_probs, token_ids, T, output_beam, device):
    import k2
    import torch

    sup = torch.tensor([[0, 0, T]], dtype=torch.int32)
    dense = k2.DenseFsaVec(log_probs[:, :T, :], sup)
    ctc_graph = k2.ctc_graph([token_ids], modified=False, device=device)
    lat = k2.intersect_dense(ctc_graph, dense, output_beam=output_beam)
    return lat.get_tot_scores(log_semiring=True, use_double_scores=True)[0]

def phase5a_train_stats(args, model, sp, device):
    out_path = Path(args.output_dir) / "train_data_summary.json"
    if out_path.exists() and not args.force:
        print(f"Phase 5a: SKIP (exists: {out_path})")
        return json.load(open(out_path))

    print("\n" + "=" * 70)
    print("Phase 5a: Training N-best statistics")
    print("=" * 70)

    import torch
    import k2
    import editdistance
    import numpy as np
    from lhotse import Fbank, FbankConfig, load_manifest_lazy

    G = 16
    cuts_path = Path(args.data_dir) / "cuts" / "librispeech_cuts_train-clean-100.jsonl.gz"
    assert cuts_path.exists(), f"Train cuts not found: {cuts_path}"
    cuts = list(load_manifest_lazy(str(cuts_path)))
    print(f"  {len(cuts)} utterances, G={G}")

    fbank = Fbank(FbankConfig(num_mel_bins=80, dither=0.0))
    topo = k2.ctc_topo(max_token=MAX_TOKEN, modified=False, device=device)

    total_greedy_edits = 0
    total_oracle_edits = 0
    total_ref_words = 0
    wer_stds = []
    n_recoverable = 0
    t0 = time.time()

    max_sample = args.train_stats_max_utts or len(cuts)
    sample_cuts = cuts[:max_sample]

    for i, cut in enumerate(sample_cuts):
        audio = cut.load_audio()
        feat = fbank.extract(audio, sampling_rate=16000)
        feat_t = torch.from_numpy(feat).unsqueeze(0).to(device)
        feat_lens = torch.tensor([feat.shape[0]], dtype=torch.int64, device=device)

        with torch.no_grad():
            enc_out, enc_lens = model.forward_encoder(feat_t, feat_lens)
            log_probs = model.ctc_output(enc_out)

        lp_utt = log_probs[0, :enc_lens[0].item()]
        cands = generate_candidates_beam(
            lp_utt, topo, sp, G, 8.0, device
        )

        ref_raw = " ".join(s.text for s in cut.supervisions if s.text)
        ref = normalize_text(ref_raw)
        ref_words = ref.split()
        if not ref_words:
            continue
        n_ref = len(ref_words)
        total_ref_words += n_ref

        wers = [compute_wer(t, ref) for t, _ in cands]
        greedy_edits = editdistance.eval(cands[0][0].split(), ref_words)
        total_greedy_edits += greedy_edits

        best_edits = min(
            editdistance.eval(t.split(), ref_words) for t, _ in cands
        )
        total_oracle_edits += best_edits
        if best_edits < greedy_edits:
            n_recoverable += 1

        if len(wers) >= 2:
            wer_stds.append(float(np.std(wers)))

        del log_probs, enc_out, feat_t
        torch.cuda.empty_cache()

        if (i + 1) % 500 == 0 or i == len(sample_cuts) - 1:
            elapsed = time.time() - t0
            print(f"    {i+1}/{len(sample_cuts)} ({(i+1)/elapsed:.1f} utt/s)")

    elapsed = time.time() - t0
    greedy_wer = total_greedy_edits / max(1, total_ref_words)
    oracle_wer = total_oracle_edits / max(1, total_ref_words)
    gap_pp = greedy_wer - oracle_wer

    result = {
        "split": "train-clean-100",
        "num_utterances": len(sample_cuts),
        "G": G,
        "nbest_scale": 1.0,
        "greedy_wer": greedy_wer,
        "oracle_wer": oracle_wer,
        "gap_pp": gap_pp,
        "recoverable_count": n_recoverable,
        "recoverable_pct": n_recoverable / max(1, len(sample_cuts)) * 100,
        "mean_wer_std_per_utt": round(float(np.mean(wer_stds)), 4)
            if wer_stds else None,
        "wall_time_s": round(elapsed, 1),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Train greedy WER:  {greedy_wer:.4%}")
    print(f"  Train oracle WER:  {oracle_wer:.4%}")
    print(f"  Train oracle gap:  {gap_pp*100:.3f} pp")
    print(f"  Recoverable:       {n_recoverable}/{len(sample_cuts)}")
    if wer_stds:
        print(f"  Mean WER std/utt:  {np.mean(wer_stds)*100:.2f} pp")

    print(f"\n  CR-CTC reference: greedy 1.09%, oracle 1.08%, gap 0.007 pp")
    print(f"  Standard CTC:     greedy {greedy_wer:.2%}, oracle {oracle_wer:.2%}, "
          f"gap {gap_pp*100:.3f} pp")

    print(f"\n  Saved: {out_path}")
    return result

def phase5b_mwer_train(args, model, sp, device):
    mwer_log_path = Path(args.output_dir) / "mwer_training_log.jsonl"
    mwer_config_path = Path(args.output_dir) / "mwer_config.json"
    mwer_report_path = Path(args.output_dir) / "mwer_smoke_report.json"

    if mwer_report_path.exists() and not args.force:
        print(f"Phase 5b: SKIP (exists: {mwer_report_path})")
        return

    print("\n" + "=" * 70)
    print("Phase 5b: MWER training (1 epoch)")
    print("=" * 70)

    import torch
    import k2
    import editdistance
    import numpy as np
    from lhotse import Fbank, FbankConfig, load_manifest_lazy
    from torch.optim import AdamW

    lr = args.mwer_lr
    ce_weight = args.mwer_ce_weight
    G = args.mwer_G
    grad_accum = args.mwer_grad_accum
    output_beam = 8.0

    config = {
        "lr": lr,
        "ce_weight": ce_weight,
        "G": G,
        "grad_accum": grad_accum,
        "num_epochs": 1,
        "mode": "mwer",
        "nbest_scale": 1.0,
        "output_beam": output_beam,
        "model_size": args.model_size,
        "use_cr_ctc": args.use_cr_ctc,
        "eval_every_steps": args.mwer_eval_every,
        "max_grad_norm": 5.0,
        "note": "Match exp_F3 config: pure MWER, no clipping",
    }
    with open(mwer_config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config saved: {mwer_config_path}")

    cuts_path = Path(args.data_dir) / "cuts" / "librispeech_cuts_train-clean-100.jsonl.gz"
    assert cuts_path.exists(), f"Train cuts not found: {cuts_path}"

    fbank = Fbank(FbankConfig(num_mel_bins=80, dither=0.0))
    topo = k2.ctc_topo(max_token=MAX_TOKEN, modified=False, device=device)

    # Baseline eval (epoch 0)
    print("\n  Baseline eval (epoch 0)...")
    baseline_metrics = {}
    for split in ["dev-other", "dev-clean"]:
        sp_path = Path(args.data_dir) / "cuts" / f"librispeech_cuts_{split}.jsonl.gz"
        if not sp_path.exists():
            continue
        eval_cuts = list(load_manifest_lazy(str(sp_path)))
        if args.mwer_eval_utts > 0:
            eval_cuts = eval_cuts[:args.mwer_eval_utts]

        hyps, refs = [], []
        for cut in eval_cuts:
            audio = cut.load_audio()
            feat = fbank.extract(audio, sampling_rate=16000)
            feat_t = torch.from_numpy(feat).unsqueeze(0).to(device)
            feat_lens = torch.tensor([feat.shape[0]], dtype=torch.int64, device=device)
            with torch.no_grad():
                enc_out, _ = model.forward_encoder(feat_t, feat_lens)
                lp = model.ctc_output(enc_out)
            toks = greedy_ctc_decode_batch(lp)
            hyps.append(sp.decode(toks[0]).strip().lower())
            ref = " ".join(s.text for s in cut.supervisions if s.text).strip().lower()
            refs.append(ref)
            del enc_out, lp, feat_t
            torch.cuda.empty_cache()

        wer, n_eval = corpus_wer_from_lists(hyps, refs)
        baseline_metrics[split] = {"wer": wer, "num_utts": n_eval}
        print(f"    {split}: WER={wer:.4%} ({n_eval} utts)")

    # Training setup
    model.eval()  # dropout off, but gradients flow
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    train_cuts = list(load_manifest_lazy(str(cuts_path)))
    print(f"\n  Training: {len(train_cuts)} utterances, G={G}, lr={lr}, "
          f"ce_weight={ce_weight}")

    log_f = open(mwer_log_path, "w")
    step = 0
    acc_count = 0
    accum_losses = []
    accum_mwer_losses = []
    accum_ce_losses = []
    skipped = 0
    eval_log = []
    t_epoch = time.time()

    optimizer.zero_grad()

    for utt_idx, cut in enumerate(train_cuts):
        ref_raw = " ".join(s.text for s in cut.supervisions if s.text)
        ref = normalize_text(ref_raw)
        if not ref:
            continue

        ref_token_ids = sp.encode(ref, out_type=int)
        if not ref_token_ids:
            continue

        audio = cut.load_audio()
        feat = fbank.extract(audio, sampling_rate=16000)
        feat_t = torch.from_numpy(feat).unsqueeze(0).to(device)
        feat_lens = torch.tensor([feat.shape[0]], dtype=torch.int64, device=device)

        try:
            enc_out, enc_lens = model.forward_encoder(feat_t, feat_lens)
            log_probs = model.ctc_output(enc_out)
            T = enc_lens[0].item()

            with torch.no_grad():
                lp_det = log_probs[:, :T, :].detach().clone()
                cands = generate_candidates_beam(
                    lp_det[0], topo, sp, G, output_beam, device
                )

            if len(cands) < 2:
                skipped += 1
                continue

            wers = [compute_wer(t, ref) for t, _ in cands]
            if len(set(wers)) <= 1:
                skipped += 1
                continue

            rewards = torch.tensor(
                [-w for w in wers], dtype=torch.float32, device=device
            )
            advantages = rewards - rewards.mean()

            log_p_curr = []
            for _, ids in cands:
                if not ids:
                    log_p_curr.append(torch.tensor(-1e9, device=device))
                    continue
                log_p_curr.append(
                    log_p_ctc(log_probs, ids, T, output_beam, device)
                )
            log_p_curr_t = torch.stack(log_p_curr)

            mwer_loss = -(advantages * log_p_curr_t).mean()

            # CE loss: -log P(y*|x). Length-normalize by reference token count
            # so magnitude is comparable across models (CR-CTC's peaked posterior
            # gives ~50-200, standard CTC's broader posterior gives ~600-1900;
            # normalizing puts both in ~1-5 per-token range).
            ce_raw = -log_p_ctc(
                log_probs, ref_token_ids, T, output_beam, device
            )
            ce_loss = ce_raw / max(len(ref_token_ids), 1)
            total_loss = mwer_loss + ce_weight * ce_loss

            assert torch.isfinite(total_loss), (
                f"Non-finite loss at utt {utt_idx}: {total_loss.item()}"
            )

            loss_scaled = total_loss / grad_accum
            loss_scaled.backward()
            acc_count += 1
            accum_losses.append(total_loss.item())
            accum_mwer_losses.append(mwer_loss.item())
            accum_ce_losses.append(ce_loss.item())

        except Exception as e:
            print(f"  [utt {utt_idx}] ERROR: {e}")
            torch.cuda.empty_cache()
            continue

        del log_probs, enc_out, feat_t
        torch.cuda.empty_cache()

        if acc_count >= grad_accum:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=5.0
            ).item()
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            agg_loss = sum(accum_losses) / len(accum_losses)
            agg_mwer = sum(accum_mwer_losses) / len(accum_mwer_losses)
            agg_ce = sum(accum_ce_losses) / len(accum_ce_losses)

            log_line = {
                "step": step, "loss": round(agg_loss, 4),
                "mwer_loss": round(agg_mwer, 4),
                "ce_loss": round(agg_ce, 4),
                "grad_norm": round(grad_norm, 4),
                "utt_idx": utt_idx,
                "skipped": skipped,
            }
            log_f.write(json.dumps(log_line) + "\n")
            log_f.flush()

            if step == 1 or step % 50 == 0:
                print(f"  [step {step}] loss={agg_loss:.3f} "
                      f"mwer={agg_mwer:.3f} ce={agg_ce:.2f} "
                      f"g={grad_norm:.2f} skip={skipped}")

            # Mid-training eval
            if args.mwer_eval_every > 0 and step % args.mwer_eval_every == 0:
                print(f"\n  Eval at step {step}...")
                step_eval = {"step": step}
                for split in ["dev-other"]:
                    sp_path = (Path(args.data_dir) / "cuts" /
                               f"librispeech_cuts_{split}.jsonl.gz")
                    if not sp_path.exists():
                        continue
                    eval_cuts_s = list(load_manifest_lazy(str(sp_path)))
                    if args.mwer_eval_utts > 0:
                        eval_cuts_s = eval_cuts_s[:args.mwer_eval_utts]

                    hyps, refs_e = [], []
                    for ec in eval_cuts_s:
                        audio = ec.load_audio()
                        feat = fbank.extract(audio, sampling_rate=16000)
                        ft = torch.from_numpy(feat).unsqueeze(0).to(device)
                        fl = torch.tensor(
                            [feat.shape[0]], dtype=torch.int64, device=device
                        )
                        with torch.no_grad():
                            eo, _ = model.forward_encoder(ft, fl)
                            lp = model.ctc_output(eo)
                        toks = greedy_ctc_decode_batch(lp)
                        hyps.append(sp.decode(toks[0]).strip().lower())
                        ref_e = " ".join(
                            s.text for s in ec.supervisions if s.text
                        ).strip().lower()
                        refs_e.append(ref_e)
                        del eo, lp, ft
                        torch.cuda.empty_cache()

                    mw, n_ev = corpus_wer_from_lists(hyps, refs_e)
                    step_eval[split] = {"wer": mw, "num_utts": n_ev}
                    print(f"    {split}: WER={mw:.4%} ({n_ev} utts)")
                eval_log.append(step_eval)
                model.eval()

            acc_count = 0
            accum_losses = []
            accum_mwer_losses = []
            accum_ce_losses = []
            skipped = 0

            # Smoke-test early stop
            if args.mwer_max_steps > 0 and step >= args.mwer_max_steps:
                print(f"\n  Stopping at max_steps={args.mwer_max_steps}")
                break

    log_f.close()
    epoch_time = time.time() - t_epoch

    # Final eval
    print(f"\n  Final eval after 1 epoch ({step} steps, {epoch_time:.0f}s)...")
    final_metrics = {}
    for split in ["dev-other", "dev-clean"]:
        sp_path = Path(args.data_dir) / "cuts" / f"librispeech_cuts_{split}.jsonl.gz"
        if not sp_path.exists():
            continue
        eval_cuts = list(load_manifest_lazy(str(sp_path)))
        if args.mwer_eval_utts > 0:
            eval_cuts = eval_cuts[:args.mwer_eval_utts]

        hyps, refs_f = [], []
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
            ref_f = " ".join(
                s.text for s in ec.supervisions if s.text
            ).strip().lower()
            refs_f.append(ref_f)
            del eo, lp, ft
            torch.cuda.empty_cache()

        mw, n_ev = corpus_wer_from_lists(hyps, refs_f)
        final_metrics[split] = {"wer": mw, "num_utts": n_ev}
        print(f"    {split}: WER={mw:.4%} ({n_ev} utts)")

    ckpt_path = Path(args.output_dir) / "mwer_checkpoint.pt"
    torch.save({"model": model.state_dict(), "step": step}, ckpt_path)
    print(f"  Checkpoint: {ckpt_path}")

    # Smoke report
    baseline_dev_other = baseline_metrics.get("dev-other", {}).get("wer")
    final_dev_other = final_metrics.get("dev-other", {}).get("wer")

    rel_change = None
    if baseline_dev_other and final_dev_other and baseline_dev_other > 0:
        rel_change = (final_dev_other - baseline_dev_other) / baseline_dev_other * 100

    report = {
        "mode": "mwer",
        "config": config,
        "total_steps": step,
        "epoch_time_s": round(epoch_time, 1),
        "baseline_eval": baseline_metrics,
        "final_eval": final_metrics,
        "eval_log": eval_log,
        "relative_change_pct": round(rel_change, 1) if rel_change else None,
    }

    with open(mwer_report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  " + "=" * 60)
    print(f"  MWER training summary:")
    print(f"  Steps: {step}, Time: {epoch_time:.0f}s")
    if baseline_dev_other and final_dev_other:
        print(f"  dev-other baseline: {baseline_dev_other:.4%}")
        print(f"  dev-other final:    {final_dev_other:.4%}")
        print(f"  Relative change:    {rel_change:+.1f}%")
        print()
        print(f"  CR-CTC (F3) reference: 6.67% -> 15.30% (+129%)")
        print(f"  Standard CTC:          {baseline_dev_other:.2%} -> "
              f"{final_dev_other:.2%} ({rel_change:+.1f}%)")

        if rel_change > 50:
            print("  -> Outcome A: MWER degrades similarly (CTC-general failure)")
        elif rel_change > 10:
            print("  -> Outcome B: MWER degrades LESS (CR-CTC specific vulnerability)")
        elif rel_change < -5:
            print("  -> Outcome C: MWER IMPROVES (higher train WER = real signal)")
        else:
            print("  -> Outcome D: MWER is neutral (inconclusive)")

    print(f"\n  Saved: {mwer_report_path}")
    return report

def phase6_raft(args, model, sp, device, train_summary):
    out_path = Path(args.output_dir) / "raft_report.json"
    if out_path.exists() and not args.force:
        print(f"Phase 6: SKIP (exists: {out_path})")
        return

    print("\n" + "=" * 70)
    print("Phase 6: RAFT/BoN distillation (conditional)")
    print("=" * 70)

    gap_pp = train_summary.get("gap_pp", 0)
    if gap_pp < 0.001:
        reason = (f"Training oracle gap ({gap_pp*100:.3f} pp) < 0.1 pp threshold. "
                  f"Same regime as CR-CTC  --  near-optimal training WER leaves "
                  f"no room for BoN distillation.")
        print(f"  SKIP: {reason}")
        result = {"skipped": True, "reason": reason, "gap_pp": gap_pp}
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        return

    print(f"  Training oracle gap = {gap_pp*100:.3f} pp > 0.1 pp  --  proceeding")

    import torch
    import k2
    import editdistance
    from lhotse import Fbank, FbankConfig, load_manifest_lazy
    from torch.optim import AdamW

    # Re-load fresh model (RAFT starts from pre-trained, not MWER-trained)
    print("  Re-loading pre-trained model for RAFT...")
    model = load_model(args, device)
    model.eval()

    fbank = Fbank(FbankConfig(num_mel_bins=80, dither=0.0))
    topo = k2.ctc_topo(max_token=MAX_TOKEN, modified=False, device=device)

    cuts_path = Path(args.data_dir) / "cuts" / "librispeech_cuts_train-clean-100.jsonl.gz"
    train_cuts = list(load_manifest_lazy(str(cuts_path)))

    G = 16
    lam = 0.5  # anti-forgetting weight
    lr = 1e-6

    # Baseline eval
    print("  Baseline eval...")
    baseline_wer = None
    sp_path = Path(args.data_dir) / "cuts" / "librispeech_cuts_dev-other.jsonl.gz"
    if sp_path.exists():
        eval_cuts = list(load_manifest_lazy(str(sp_path)))
        if args.mwer_eval_utts > 0:
            eval_cuts = eval_cuts[:args.mwer_eval_utts]
        hyps, refs = [], []
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
            refs.append(
                " ".join(s.text for s in ec.supervisions if s.text).strip().lower()
            )
            del eo, lp, ft
            torch.cuda.empty_cache()
        baseline_wer, _ = corpus_wer_from_lists(hyps, refs)
        print(f"    dev-other baseline: {baseline_wer:.4%}")

    # RAFT training: for each utterance, select WER-best from N-best,
    # fine-tune on oracle transcripts with anti-forgetting CTC loss
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    step = 0
    raft_log = []
    t0 = time.time()

    for utt_idx, cut in enumerate(train_cuts):
        ref_raw = " ".join(s.text for s in cut.supervisions if s.text)
        ref = normalize_text(ref_raw)
        if not ref:
            continue
        ref_token_ids = sp.encode(ref, out_type=int)
        if not ref_token_ids:
            continue

        audio = cut.load_audio()
        feat = fbank.extract(audio, sampling_rate=16000)
        feat_t = torch.from_numpy(feat).unsqueeze(0).to(device)
        feat_lens = torch.tensor([feat.shape[0]], dtype=torch.int64, device=device)

        try:
            enc_out, enc_lens = model.forward_encoder(feat_t, feat_lens)
            log_probs = model.ctc_output(enc_out)
            T = enc_lens[0].item()

            with torch.no_grad():
                lp_det = log_probs[:, :T, :].detach().clone()
                cands = generate_candidates_beam(
                    lp_det[0], topo, sp, G, 8.0, device
                )

            if not cands:
                continue

            best_wer = float("inf")
            best_ids = ref_token_ids
            for text, toks in cands:
                w = compute_wer(text, ref)
                if w < best_wer:
                    best_wer = w
                    best_ids = toks

            # RAFT loss: (1-lam)*CTC(oracle) + lam*CTC(gold)
            ce_oracle = -log_p_ctc(log_probs, best_ids, T, 8.0, device)
            ce_gold = -log_p_ctc(log_probs, ref_token_ids, T, 8.0, device)
            loss = (1 - lam) * ce_oracle + lam * ce_gold

            if not torch.isfinite(loss):
                continue

            loss.backward()
            step += 1

            if step % args.mwer_grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad()

            if step % 500 == 0:
                raft_log.append({"step": step, "loss": round(loss.item(), 4)})
                print(f"  [step {step}] loss={loss.item():.3f}")

        except Exception as e:
            print(f"  [utt {utt_idx}] ERROR: {e}")
            torch.cuda.empty_cache()
            continue

        del log_probs, enc_out, feat_t
        torch.cuda.empty_cache()

    elapsed = time.time() - t0

    # Final eval
    print(f"\n  RAFT training done: {step} steps, {elapsed:.0f}s")
    final_wer = None
    if sp_path.exists():
        eval_cuts = list(load_manifest_lazy(str(sp_path)))
        if args.mwer_eval_utts > 0:
            eval_cuts = eval_cuts[:args.mwer_eval_utts]
        hyps, refs = [], []
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
            refs.append(
                " ".join(s.text for s in ec.supervisions if s.text).strip().lower()
            )
            del eo, lp, ft
            torch.cuda.empty_cache()
        final_wer, _ = corpus_wer_from_lists(hyps, refs)
        print(f"    dev-other final: {final_wer:.4%}")

    result = {
        "skipped": False,
        "gap_pp": gap_pp,
        "lambda": lam,
        "lr": lr,
        "G": G,
        "total_steps": step,
        "elapsed_s": round(elapsed, 1),
        "baseline_dev_other_wer": baseline_wer,
        "final_dev_other_wer": final_wer,
        "log": raft_log,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="E26: Standard CTC  --  MWER training + RAFT (Phases 5-6)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bpe", type=Path, required=True)
    parser.add_argument("--icefall-dir", type=Path, default=Path("/content/icefall"))
    parser.add_argument("--recipe", type=str, default="zipformer",
                        choices=["zipformer", "zipformer_ctc"])
    parser.add_argument("--model-size", type=str, default="medium",
                        choices=["small", "medium", "large"])
    parser.add_argument("--use-cr-ctc", action="store_true", default=False)
    parser.add_argument("--vocab-size", type=int, default=500)

    # Data
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    # Phase selection
    parser.add_argument("--phase", type=str, default="5a,5b,5c",
                        help="Phases: 5a, 5b, 5c, 6, or comma-separated")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")

    # Phase 5a
    parser.add_argument("--train-stats-max-utts", type=int, default=0,
                        help="Max utterances for train stats (0=all)")

    # Phase 5b MWER
    parser.add_argument("--mwer-lr", type=float, default=1e-6)
    parser.add_argument("--mwer-ce-weight", type=float, default=0.01)
    parser.add_argument("--mwer-G", type=int, default=16)
    parser.add_argument("--mwer-grad-accum", type=int, default=4)
    parser.add_argument("--mwer-eval-every", type=int, default=500,
                        help="Eval dev-other every N steps (0=disabled)")
    parser.add_argument("--mwer-eval-utts", type=int, default=0,
                        help="Max eval utterances (0=all)")
    parser.add_argument("--mwer-max-steps", type=int, default=0,
                        help="Stop training after N steps (0=full epoch). "
                             "Use for smoke testing.")

    args = parser.parse_args()

    print("=" * 70)
    print("E26: Standard CTC  --  MWER training (Phases 5-6)")
    print("=" * 70)
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  recipe:     {args.recipe}")
    print(f"  model_size: {args.model_size}")
    print(f"  phase:      {args.phase}")
    print()

    phases = set(args.phase.split(","))

    import torch
    import sentencepiece as spm

    device = torch.device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor()
    sp.load(str(args.bpe))
    assert sp.get_piece_size() == args.vocab_size

    print("Loading model...")
    model = load_model(args, device)

    train_summary = None

    if "5a" in phases:
        train_summary = phase5a_train_stats(args, model, sp, device)

    if "5b" in phases or "5c" in phases:
        # Re-load fresh model for training (5a may have been inference-only)
        if "5a" in phases:
            print("\n  Re-loading fresh model for MWER training...")
            model = load_model(args, device)
        phase5b_mwer_train(args, model, sp, device)

    if "6" in phases:
        if train_summary is None:
            summary_path = Path(args.output_dir) / "train_data_summary.json"
            if summary_path.exists():
                train_summary = json.load(open(summary_path))
            else:
                print("Phase 6: SKIP (no train_data_summary.json  --  run 5a first)")
                return

        phase6_raft(args, model, sp, device, train_summary)

    print("\n" + "=" * 70)
    print("E26 MWER phases complete.")
    print(f"  Results in: {args.output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()

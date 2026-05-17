#!/usr/bin/env python3
"""Stage 4: RBPO training script.

Three modes selected via --mode:
  - mwer:                 standard MWER (A_hat * log P_CTC, no clipping)
  - mwer_clipped:         MWER + sequence-level clipped IS ratio (RBPO Eq. 15)
  - mwer_clipped_lattice: mwer_clipped with candidates from lattice sampling

For Stage 4 the goal is a 10-step smoke test confirming forward+backward+step
all work without NaNs. Experiments A/B/C reuse this script with longer runs.

Usage:
    python training/train.py \\
        --mode mwer_clipped \\
        --model-dir /content/icefall-asr-librispeech-zipformer-small-cr-ctc \\
        --icefall-dir /content/icefall \\
        --data-dir /content/librispeech_data \\
        --results-dir /content/drive/MyDrive/rbpo_results \\
        --run-name smoke_test \\
        --num-epochs 1 \\
        --max-utterances-per-epoch 20 \\
        --G 4 --grad-accum 2
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import editdistance
import sentencepiece as spm
import torch
from torch.optim import AdamW

# Allow running from any CWD: utils live at rbpo/utils
THIS_FILE = Path(__file__).resolve()
RBPO_ROOT = THIS_FILE.parent.parent
if str(RBPO_ROOT) not in sys.path:
    sys.path.insert(0, str(RBPO_ROOT))

from utils.advantages import group_relative_advantages
from utils.clipping import clip_surrogate, length_normalize_ratio

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
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params / 1e6:.1f}M parameters")
    return model

def ctc_collapse(token_ids: list[int]) -> list[int]:
    out = []
    prev = None
    for t in token_ids:
        if t != BLANK_ID and t != prev:
            out.append(t)
        prev = t
    return out

def compute_wer(hypothesis: str, reference: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return editdistance.eval(hyp_words, ref_words) / len(ref_words)

def greedy_ctc_decode_batch(log_probs: torch.Tensor) -> list[list[int]]:
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

def generate_candidates_beam(
    log_probs: torch.Tensor,
    topo,
    sp: spm.SentencePieceProcessor,
    G: int,
    output_beam: float,
    device: torch.device,
) -> list[tuple[str, list[int]]]:
    """Score-proportional sampling at scale=1.0 (~ beam-style).

    Despite the name, we use Nbest.from_lattice with nbest_scale=1.0 so the
    same code path serves both 'mwer' (acts ~ beam top-G) and the lattice
    variants. Greedy 1-best is forced into position 0.
    """
    import k2

    T = log_probs.shape[0]
    sup = torch.tensor([[0, 0, T]], dtype=torch.int32)
    dense = k2.DenseFsaVec(log_probs.unsqueeze(0), sup)
    lattice = k2.intersect_dense(topo, dense, output_beam=output_beam)
    lattice = k2.connect(lattice)
    return _extract_candidates(lattice, log_probs, sp, G, nbest_scale=1.0)

def generate_candidates_lattice(
    log_probs: torch.Tensor,
    topo,
    sp: spm.SentencePieceProcessor,
    G: int,
    output_beam: float,
    nbest_scale: float,
    device: torch.device,
) -> list[tuple[str, list[int]]]:
    """Lattice sampling at the given nbest_scale (Stage 1 winner: 1.0)."""
    import k2

    T = log_probs.shape[0]
    sup = torch.tensor([[0, 0, T]], dtype=torch.int32)
    dense = k2.DenseFsaVec(log_probs.unsqueeze(0), sup)
    lattice = k2.intersect_dense(topo, dense, output_beam=output_beam)
    lattice = k2.connect(lattice)
    return _extract_candidates(lattice, log_probs, sp, G, nbest_scale=nbest_scale)

def _extract_candidates(lattice, log_probs, sp, G, nbest_scale):
    """Sample paths, dedup, force greedy 1-best to position 0."""
    import k2

    nbest = k2.Nbest.from_lattice(
        lattice,
        num_paths=max(G * 4, 16),
        use_double_scores=True,
        nbest_scale=nbest_scale,
    )
    all_labels = nbest.fsa.labels.cpu().tolist()
    paths = []
    cur = []
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

    # Force greedy 1-best to position 0
    greedy_raw = log_probs.argmax(dim=-1).tolist()
    greedy_toks = ctc_collapse(greedy_raw)
    if greedy_toks:
        greedy_text = sp.decode(greedy_toks).strip().lower()
        out = [(t, ids) for t, ids in out if t != greedy_text]
        out.insert(0, (greedy_text, greedy_toks))

    return out[:G]

def log_p_ctc(
    log_probs: torch.Tensor,
    token_ids: list[int],
    T: int,
    output_beam: float,
    device: torch.device,
) -> torch.Tensor:
    """Differentiable log P_CTC(y|x) via k2 numerator lattice."""
    import k2

    sup = torch.tensor([[0, 0, T]], dtype=torch.int32)
    dense = k2.DenseFsaVec(log_probs[:, :T, :], sup)
    ctc_graph = k2.ctc_graph([token_ids], modified=False, device=device)
    lat = k2.intersect_dense(ctc_graph, dense, output_beam=output_beam)
    return lat.get_tot_scores(log_semiring=True, use_double_scores=True)[0]

def compute_ce_loss(
    log_probs: torch.Tensor,
    ref_token_ids: list[int],
    T: int,
    output_beam: float,
    device: torch.device,
) -> torch.Tensor:
    return -log_p_ctc(log_probs, ref_token_ids, T, output_beam, device)

def load_train_cuts(data_dir: Path):
    from lhotse import load_manifest_lazy

    path = data_dir / "cuts" / "librispeech_cuts_train-clean-100.jsonl.gz"
    assert path.exists(), (
        f"Train cuts not found: {path}\n"
        "Run scripts/download_train_data.sh first."
    )
    return load_manifest_lazy(str(path))

def iter_train_utterances(cuts, max_utts: int, sp):
    """Yield (cut_id, feats, ref_text, ref_token_ids) for up to max_utts cuts."""
    n = 0
    for cut in cuts:
        if max_utts and n >= max_utts:
            break
        ref_text = " ".join(
            s.text for s in cut.supervisions if s.text
        ).strip().lower()
        if not ref_text:
            continue
        feats = torch.from_numpy(cut.load_features())
        ref_token_ids = sp.encode(ref_text, out_type=int)
        if not ref_token_ids:
            continue
        yield cut.id, feats, ref_text, ref_token_ids
        n += 1

@torch.no_grad()
def eval_split(
    model,
    data_dir: Path,
    split: str,
    sp: spm.SentencePieceProcessor,
    device: torch.device,
    max_utts: int = 0,
    batch_size: int = 8,
) -> dict:
    from lhotse import load_manifest_lazy

    path = data_dir / "cuts" / f"librispeech_cuts_{split}.jsonl.gz"
    if not path.exists():
        return {"wer": None, "num_utts": 0, "skipped": "cuts not found"}
    cuts = load_manifest_lazy(str(path))

    model.eval()
    hyps = []
    refs = []
    batch_cuts = []

    def flush():
        if not batch_cuts:
            return
        feats_list = [torch.from_numpy(c.load_features()) for c in batch_cuts]
        lengths = [f.shape[0] for f in feats_list]
        max_len = max(lengths)
        bf = torch.zeros(len(batch_cuts), max_len, feats_list[0].shape[1])
        for i, f in enumerate(feats_list):
            bf[i, : f.shape[0]] = f
        bf = bf.to(device)
        feat_lens = torch.tensor(lengths, dtype=torch.int64, device=device)

        encoder_out, _ = model.forward_encoder(bf, feat_lens)
        log_probs = model.ctc_output(encoder_out)
        token_ids_batch = greedy_ctc_decode_batch(log_probs)
        for c, toks in zip(batch_cuts, token_ids_batch):
            text = sp.decode(toks).strip().lower()
            hyps.append(text)
            ref = " ".join(s.text for s in c.supervisions if s.text).strip().lower()
            refs.append(ref)
        del bf, encoder_out, log_probs
        torch.cuda.empty_cache()

    n = 0
    for cut in cuts:
        batch_cuts.append(cut)
        n += 1
        if len(batch_cuts) >= batch_size:
            flush()
            batch_cuts = []
        if max_utts and n >= max_utts:
            break
    flush()

    if not refs:
        return {"wer": None, "num_utts": 0}

    wer_vals = [compute_wer(h, r) for h, r in zip(hyps, refs) if r.strip()]
    mean_wer = sum(wer_vals) / len(wer_vals)
    return {"wer": mean_wer, "num_utts": len(wer_vals)}

def compute_step_loss(
    model,
    feats: torch.Tensor,
    ref_text: str,
    ref_token_ids: list[int],
    sp: spm.SentencePieceProcessor,
    topo,
    args,
    device: torch.device,
    ref_model=None,
) -> tuple[torch.Tensor, dict]:
    """Forward + candidate gen + loss for one utterance.

    Returns (loss, metrics_dict). loss has gradient, NOT yet backwarded.
    """
    feats_gpu = feats.unsqueeze(0).to(device)
    feat_lens = torch.tensor(
        [feats.shape[0]], dtype=torch.int64, device=device
    )

    # Forward in eval mode for stable batchnorm/dropout? icefall expects
    # train mode for proper handling. We're in train mode, no dropout disabled
    #  --  that's the standard MWER setup.
    encoder_out, encoder_out_lens = model.forward_encoder(feats_gpu, feat_lens)
    log_probs = model.ctc_output(encoder_out)  # (1, T', V), live grad
    T = encoder_out_lens[0].item()

    # Candidate generation runs on detached log_probs (we don't backward through it)
    with torch.no_grad():
        log_probs_det = log_probs[:, :T, :].detach().clone()
        if args.mode == "mwer_clipped_lattice":
            cands = generate_candidates_lattice(
                log_probs_det[0], topo, sp, args.G,
                args.output_beam, args.nbest_scale, device,
            )
        else:
            cands = generate_candidates_beam(
                log_probs_det[0], topo, sp, args.G, args.output_beam, device,
            )

    if len(cands) < 2:
        return None, {"skipped": "too_few_candidates", "G_unique": len(cands)}

    wers = [compute_wer(t, ref_text) for t, _ in cands]
    if len(set(wers)) <= 1:
        # No reward signal  --  skip. This still happens on easy utterances.
        return None, {
            "skipped": "zero_advantage",
            "G_unique": len(cands),
            "wers": wers,
        }

    # Dr. GRPO advantage: r = -WER, A_hat = r - mean(r)
    rewards = torch.tensor([-w for w in wers], dtype=torch.float32, device=device)
    advantages = group_relative_advantages(rewards).to(device)

    # Per-candidate log P_CTC under current theta
    log_p_curr = []
    for _, ids in cands:
        if not ids:
            log_p_curr.append(torch.tensor(-1e9, device=device))
            continue
        log_p_curr.append(
            log_p_ctc(log_probs, ids, T, args.output_beam, device)
        )
    log_p_curr_t = torch.stack(log_p_curr)  # (G,)

    # MWER loss term
    if args.mode == "mwer":
        # L = -(1/G) sum_i A_hat_i * log P_CTC(y_i|x)
        mwer_loss = -(advantages * log_p_curr_t).mean()
        rho_log = torch.zeros_like(advantages)
        clip_frac = 0.0
    else:
        # Compute log P under theta_old (frozen ref_model)
        assert ref_model is not None, "Clipped mode requires ref_model"
        with torch.no_grad():
            ref_encoder_out, ref_lens = ref_model.forward_encoder(
                feats_gpu, feat_lens
            )
            ref_log_probs = ref_model.ctc_output(ref_encoder_out)
            T_ref = ref_lens[0].item()
            assert T_ref == T, f"Encoder length mismatch: {T_ref} vs {T}"
            log_p_old_list = []
            for _, ids in cands:
                if not ids:
                    log_p_old_list.append(
                        torch.tensor(-1e9, device=device)
                    )
                    continue
                log_p_old_list.append(
                    log_p_ctc(ref_log_probs, ids, T, args.output_beam, device)
                )
            log_p_old_t = torch.stack(log_p_old_list).detach()
            del ref_encoder_out, ref_log_probs

        # Length-normalized importance ratio (GSPO-style)
        lengths = torch.tensor(
            [max(len(ids), 1) for _, ids in cands],
            dtype=torch.float32, device=device,
        )
        log_rho = (log_p_curr_t.detach() - log_p_old_t)
        rho = length_normalize_ratio(log_rho, lengths)  # exp(log_rho / L)

        # Stop-gradient surrogate weights
        weights = clip_surrogate(
            rho, advantages, eps_low=args.eps_low, eps_high=args.eps_high
        ).detach()
        clipped_rho = torch.clamp(
            rho, 1.0 - args.eps_low, 1.0 + args.eps_high
        )
        clip_frac = (rho != clipped_rho).float().mean().item()

        # The implicit RBPO form (paper Eq. 15): backward through log_p_curr only,
        # weighted by the (detached) clipped surrogate scalar per candidate.
        mwer_loss = -(weights * log_p_curr_t).mean()
        rho_log = rho.detach()

    # CE regularization on the reference
    ce_loss = compute_ce_loss(
        log_probs, ref_token_ids, T, args.output_beam, device
    )

    total_loss = mwer_loss + args.ce_weight * ce_loss

    metrics = {
        "loss": total_loss.item(),
        "mwer_loss": mwer_loss.item(),
        "ce_loss": ce_loss.item(),
        "mean_rho": float(rho_log.mean().item()) if isinstance(rho_log, torch.Tensor) else 1.0,
        "clip_frac": clip_frac,
        "mean_adv_abs": float(advantages.abs().mean().item()),
        "G_unique": len(cands),
        "wers": wers,
        "T": T,
        "L_ref": len(ref_token_ids),
    }

    # Cleanup before backward (caller will backward)
    del log_probs, encoder_out, feats_gpu, log_probs_det, log_p_curr_t
    return total_loss, metrics

def train(args):
    device = torch.device(args.device)
    out_dir = Path(args.results_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = out_dir / "config.json"
    with open(cfg_path, "w") as f:
        json.dump(vars(args) | {
            k: str(v) for k, v in vars(args).items() if isinstance(v, Path)
        }, f, indent=2, default=str)
    print(f"Config saved: {cfg_path}")

    # SentencePiece + model
    sp = spm.SentencePieceProcessor()
    bpe_path = Path(args.model_dir) / "data" / "lang_bpe_500" / "bpe.model"
    sp.load(str(bpe_path))
    print(f"BPE vocab: {sp.get_piece_size()}")

    model = load_model(Path(args.model_dir), Path(args.icefall_dir), device)
    # Keep dropout disabled during fine-tuning so the IS ratio reflects real
    # policy change, not Monte-Carlo dropout noise. Zipformer uses LayerNorm
    # (no running stats), so eval mode here only differs from train mode by
    # disabling dropout. Gradients still flow normally.
    model.eval()

    # Reference model for clipped modes
    ref_model = None
    if args.mode in ("mwer_clipped", "mwer_clipped_lattice"):
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        print("Reference model (theta_old) initialized as deepcopy")

    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    import k2
    topo = k2.ctc_topo(max_token=MAX_TOKEN, modified=False, device=device)

    train_cuts = load_train_cuts(Path(args.data_dir))

    print("\n-- Baseline eval (epoch 0) --")
    base_metrics = {}
    for split in ["dev-clean", "dev-other"]:
        m = eval_split(
            model, Path(args.data_dir), split, sp, device, args.eval_utts
        )
        base_metrics[split] = m
        print(f"  {split}: WER={m.get('wer')} ({m['num_utts']} utts)")
    # Stay in eval mode (dropout off) for the training loop  --  see note above.
    model.eval()

    log_path = out_dir / "training_log.jsonl"
    log_f = open(log_path, "w")

    loss_inc = 0
    prev_loss = None
    rho_logged = False

    ep_summaries = []

    for epoch in range(1, args.num_epochs + 1):
        print(f"\n-- Epoch {epoch}/{args.num_epochs} --")
        epoch_start = time.time()
        step = 0
        acc_count = 0
        acc_metrics = []
        skipped = 0
        utt_iter = iter_train_utterances(
            train_cuts, args.max_utterances_per_epoch, sp
        )

        optimizer.zero_grad()

        for utt_idx, (cut_id, feats, ref_text, ref_token_ids) in enumerate(
            utt_iter
        ):
            t0 = time.time()
            try:
                loss, metrics = compute_step_loss(
                    model, feats, ref_text, ref_token_ids, sp, topo,
                    args, device, ref_model=ref_model,
                )
            except Exception as e:
                print(f"  [utt {utt_idx} {cut_id}] EXCEPTION: {e}")
                torch.cuda.empty_cache()
                continue

            if loss is None:
                skipped += 1
                continue

            assert torch.isfinite(loss), (
                f"Non-finite loss at utt {utt_idx} ({cut_id}): {loss.item()}"
            )

            loss_scaled = loss / args.grad_accum
            loss_scaled.backward()
            acc_count += 1
            acc_metrics.append(metrics)

            if acc_count >= args.grad_accum:
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.max_grad_norm
                ).item()
                assert torch.isfinite(torch.tensor(grad_norm)), (
                    f"Non-finite grad norm at step {step}"
                )

                optimizer.step()
                optimizer.zero_grad()
                step += 1

                # Aggregate metrics over the accumulation window
                agg_loss = sum(m["loss"] for m in acc_metrics) / len(acc_metrics)
                agg_mwer = sum(m["mwer_loss"] for m in acc_metrics) / len(acc_metrics)
                agg_ce = sum(m["ce_loss"] for m in acc_metrics) / len(acc_metrics)
                agg_rho = sum(m["mean_rho"] for m in acc_metrics) / len(acc_metrics)
                agg_clip = sum(m["clip_frac"] for m in acc_metrics) / len(acc_metrics)
                agg_adv = sum(m["mean_adv_abs"] for m in acc_metrics) / len(acc_metrics)
                agg_G = sum(m["G_unique"] for m in acc_metrics) / len(acc_metrics)
                agg_oracle = sum(min(m["wers"]) for m in acc_metrics) / len(acc_metrics)
                step_time = time.time() - t0

                # -- Smoke-test assertions --
                if args.mode != "mwer":
                    if step == 1 and not rho_logged:
                        rho_logged = True
                        # At first step theta ~ theta_old, so rho should be ~1.0
                        if not (0.999 <= agg_rho <= 1.001):
                            print(
                                f"WARNING: step 1 mean rho={agg_rho:.6f} "
                                f"outside [0.999, 1.001] (theta != theta_old?)"
                            )
                    assert 0.5 <= agg_rho <= 2.0, (
                        f"rho out of bounds at step {step}: {agg_rho:.4f}"
                    )

                if prev_loss is not None and agg_loss > prev_loss:
                    loss_inc += 1
                else:
                    loss_inc = 0
                if loss_inc >= 5:
                    print(
                        f"WARNING: loss increased {loss_inc} "
                        f"consecutive steps (possible divergence)"
                    )
                prev_loss = agg_loss

                # -- Update reference model --
                if (
                    ref_model is not None
                    and args.ref_update_interval > 0
                    and step % args.ref_update_interval == 0
                ):
                    with torch.no_grad():
                        for src, dst in zip(
                            model.parameters(), ref_model.parameters()
                        ):
                            dst.data.copy_(src.data)

                # -- Logging --
                log_line = {
                    "epoch": epoch, "step": step,
                    "loss": round(agg_loss, 4),
                    "mwer_loss": round(agg_mwer, 4),
                    "ce_loss": round(agg_ce, 4),
                    "mean_rho": round(agg_rho, 6),
                    "clip_frac": round(agg_clip, 4),
                    "mean_adv": round(agg_adv, 4),
                    "grad_norm": round(grad_norm, 4),
                    "G_unique": round(agg_G, 2),
                    "oracle_wer": round(agg_oracle, 4),
                    "step_time": round(step_time, 2),
                    "skipped_in_window": skipped,
                }
                log_f.write(json.dumps(log_line) + "\n")
                log_f.flush()
                if step == 1 or step % args.log_every == 0:
                    print(
                        f"  [ep {epoch} step {step}] "
                        f"loss={agg_loss:.3f} mwer={agg_mwer:.3f} ce={agg_ce:.2f} "
                        f"rho={agg_rho:.4f} clip={agg_clip:.2f} "
                        f"|A_hat|={agg_adv:.3f} g={grad_norm:.2f} "
                        f"G={agg_G:.1f} ow={agg_oracle*100:.1f}% "
                        f"t={step_time:.1f}s"
                    )

                acc_count = 0
                acc_metrics = []
                skipped = 0
                torch.cuda.empty_cache()

        epoch_time = time.time() - epoch_start
        epoch_eval = {}
        if args.eval_interval > 0 and epoch % args.eval_interval == 0:
            print(f"\n  Evaluating after epoch {epoch}...")
            for split in ["dev-clean", "dev-other"]:
                m = eval_split(
                    model, Path(args.data_dir), split, sp,
                    device, args.eval_utts,
                )
                epoch_eval[split] = m
                wer_str = f"{m['wer']*100:.2f}%" if m.get("wer") is not None else "n/a"
                print(f"    {split}: WER={wer_str} ({m['num_utts']} utts)")
            # Stay in eval mode for the next training epoch  --  see note above.
            model.eval()

        epoch_summary = {
            "epoch": epoch,
            "steps": step,
            "time_seconds": round(epoch_time, 1),
            "eval": epoch_eval,
        }
        ep_summaries.append(epoch_summary)

        # Checkpoint
        ckpt_path = out_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
        }, ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")

    log_f.close()

    # Final smoke report
    smoke_report = {
        "mode": args.mode,
        "run_name": args.run_name,
        "num_epochs": args.num_epochs,
        "baseline_eval": {k: v for k, v in base_metrics.items()},
        "epochs": ep_summaries,
        "first_step_rho_ok": rho_logged,
    }
    with open(out_dir / "smoke_report.json", "w") as f:
        json.dump(smoke_report, f, indent=2, default=str)

    print(f"\n Training complete. Output: {out_dir}")
    return smoke_report

def parse_args():
    p = argparse.ArgumentParser(description="RBPO training (Stage 4)")
    # Mode + paths
    p.add_argument(
        "--mode", choices=["mwer", "mwer_clipped", "mwer_clipped_lattice"],
        default="mwer",
    )
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--icefall-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--run-name", type=str, default="run_001")
    # Training
    p.add_argument("--num-epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-utterances-per-epoch", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-grad-norm", type=float, default=5.0)
    # Candidate gen
    p.add_argument("--G", type=int, default=4)
    p.add_argument("--nbest-scale", type=float, default=1.0)
    p.add_argument("--output-beam", type=float, default=8.0)
    # MWER
    p.add_argument("--ce-weight", type=float, default=0.01)
    # Clipping
    p.add_argument("--eps-low", type=float, default=3e-4)
    p.add_argument("--eps-high", type=float, default=6e-4)
    p.add_argument("--ref-update-interval", type=int, default=2)
    # Eval
    p.add_argument("--eval-interval", type=int, default=1)
    p.add_argument("--eval-utts", type=int, default=0)
    # Logging
    p.add_argument(
        "--log-every", type=int, default=10,
        help="Print step log every N steps (JSONL still gets every step)",
    )
    # Misc
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()

def main():
    args = parse_args()
    print("=" * 60)
    print(f"RBPO Stage 4  --  Training ({args.mode})")
    print("=" * 60)
    print(f"Run name: {args.run_name}")
    print(f"Mode: {args.mode}")
    print(f"Epochs: {args.num_epochs} | utts/epoch: {args.max_utterances_per_epoch}")
    print(f"G: {args.G} | grad_accum: {args.grad_accum} | lr: {args.lr}")
    train(args)

if __name__ == "__main__":
    main()

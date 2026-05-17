#!/usr/bin/env python3
"""E16b: Zipformer-M G=128  --  cross-model x cross-G scaling verification.

Tests whether the G scaling finding (F8) replicates on a second architecture:
does MBR-CER+PLL tau=10 keep improving from G=16 -> G=128 on Zipformer-M while
linear interpolation plateaus?

Key differences vs E16 (G=16):
  - G=128, oversample=512 (matches Zipformer-S G=128 protocol)
  - Batched PLL (~30x faster per hypothesis than per-position masking)
  - Checkpointing every 200 utterances during PLL scoring (resume on disconnect)
  - Adds alpha=0.8 interp and tau=50 MBR+PLL methods (G=128 spec)

Pipeline (resumable via --steps):
  discover   --  Inventory existing files
  generate   --  N-best for G=128 (~10-15 min on T4)
  score      --  Batched RoBERTa PLL + GPT-2 (~30-45 min on T4)
  evaluate   --  All methods + bootstrap + Spearman + cross-model table

Usage (Colab T4):
    python experiments/evaluation/eval_zipformer_m_g128.py \
        --data-dir /content/drive/MyDrive/rbpo_results \
        --librispeech-dir /content/librispeech_data \
        --model-dir /content/icefall-asr-librispeech-zipformer-medium-cr-ctc \
        --icefall-dir /content/icefall \
        --output-dir /content/drive/MyDrive/rbpo_results/zipformer_m_g128 \
        --model-config zipformer-m-cr-ctc \
        --steps all
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import editdistance
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.eval_zipformer_m import (
    MODEL_CONFIGS,
    add_icefall_to_path,
    load_model,
    ctc_collapse,
    build_lattice,
    alignment_log_prob,
    extract_nbest_with_scores,
    load_all_utterances,
    load_jsonl,
    compute_cer_matrix,
    select_greedy,
    select_oracle,
    select_interp,
    per_utt_spearman,
)
from experiments.significance_tests import paired_bootstrap_wer, corpus_wer

# G=128 constants (match Zipformer-S G=128 protocol)
G = 128
NUM_PATHS_OVERSAMPLE = 512
NBEST_SCALE = 1.0
BLANK_ID = 0
MAX_TOKEN = 499

# Methods evaluated (G=128 spec adds alpha=0.8 and tau=50)
INTERP_ALPHAS = [0.7, 0.8]
MBR_PLL_TAUS = [10.0, 50.0]

# How often to flush partial PLL scores to disk
CHECKPOINT_EVERY_N_UTTS = 200

def select_mbr_pll(records, tau, score_field="roberta_pll"):
    """MBR-CER selection with PLL softmax weights at given tau. G=128 version
    (re-implements local copy because compute_cer_matrix scales O(G^2))."""
    out = []
    for rec in records:
        cands = [c for c in rec["candidates"] if c["text"].strip()]
        if not cands:
            cands = rec["candidates"]
        n = len(cands)
        texts = [c["text"] for c in cands]
        log_scores = np.array([c[score_field] for c in cands])
        if math.isinf(tau):
            weights = np.ones(n) / n
        else:
            scaled = log_scores / tau
            scaled -= np.max(scaled)
            weights = np.exp(scaled)
            weights /= weights.sum()
        cer = compute_cer_matrix(texts)
        risk = cer @ weights
        out.append(texts[int(np.argmin(risk))])
    return out

def select_mbr_ctc(records, tau):
    return select_mbr_pll(records, tau, score_field="ctc_log_prob")

def step_discover(args):
    print("\n" + "=" * 70)
    print("STEP 0: DISCOVER  --  Inventory existing files")
    print("=" * 70)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    nbest_path = args.output_dir / "nbest_zipformer_m_G128.jsonl"
    scored_path = args.output_dir / "neural_lm_scores_zipformer_m_G128.jsonl"

    print(f"\n  Output dir: {args.output_dir}")
    if nbest_path.exists() and nbest_path.stat().st_size > 0:
        n = sum(1 for _ in open(nbest_path))
        print(f"  N-best:  EXISTS ({nbest_path.name}, {n} records, "
              f"{nbest_path.stat().st_size/1e6:.1f} MB)")
    else:
        print(f"  N-best:  MISSING (need to generate)")

    if scored_path.exists() and scored_path.stat().st_size > 0:
        with open(scored_path) as f:
            first = json.loads(f.readline())
        c0 = first["candidates"][0]
        has_pll = "roberta_pll" in c0
        has_gpt = "gpt2_ll" in c0
        n = sum(1 for _ in open(scored_path))
        flags = []
        if has_pll:
            flags.append("PLL")
        if has_gpt:
            flags.append("GPT-2")
        flag_str = "+".join(flags) if flags else "none"
        print(f"  Scored:  EXISTS ({scored_path.name}, {n} records, "
              f"{scored_path.stat().st_size/1e6:.1f} MB, has: {flag_str})")
    else:
        print(f"  Scored:  MISSING (need to score)")

    if args.model_dir.exists():
        print(f"\n  Model dir: {args.model_dir} (exists)")
    else:
        print(f"\n  Model dir: {args.model_dir} (DOES NOT EXIST)")

def step_generate(args, config):
    print("\n" + "=" * 70)
    print(f"STEP 1: GENERATE  --  Build G={G} N-best (oversample={NUM_PATHS_OVERSAMPLE})")
    print("=" * 70)

    out_path = args.output_dir / "nbest_zipformer_m_G128.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0:
        n = sum(1 for _ in open(out_path))
        print(f"  SKIP: {out_path.name} exists ({n} records)")
        return

    import torch
    import sentencepiece as spm
    import k2
    from tqdm import tqdm

    device = torch.device(args.device)

    sp = spm.SentencePieceProcessor()
    bpe_path = args.model_dir / "data" / "lang_bpe_500" / "bpe.model"
    assert bpe_path.exists(), f"BPE model not found: {bpe_path}"
    sp.load(str(bpe_path))
    print(f"  BPE: {sp.get_piece_size()} tokens")

    print(f"\n  Loading model from {args.model_dir}")
    model = load_model(args.model_dir, args.icefall_dir, device, config)

    utterances = load_all_utterances(args.librispeech_dir, "dev-other")
    topo = k2.ctc_topo(max_token=MAX_TOKEN, modified=False, device=device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    total_candidates = 0

    print(f"\n  Generating N-best for {len(utterances)} utterances "
          f"(G={G}, oversample={NUM_PATHS_OVERSAMPLE})...")
    with open(out_path, "w") as f:
        for utt_id, feats, ref_text in tqdm(utterances, desc="  Generate"):
            feats_gpu = feats.unsqueeze(0).to(device)
            feat_lens = torch.tensor([feats.shape[0]], dtype=torch.int64, device=device)

            with torch.no_grad():
                encoder_out, _ = model.forward_encoder(feats_gpu, feat_lens)
                log_probs = model.ctc_output(encoder_out)

            lp_utt = log_probs[0]
            lp_cpu = lp_utt.cpu()

            greedy_ids = lp_utt.argmax(dim=-1).tolist()
            greedy_collapsed = ctc_collapse(greedy_ids)
            greedy_text = sp.decode(greedy_collapsed).strip().lower()
            greedy_score = alignment_log_prob(greedy_ids, lp_cpu)

            try:
                lattice = build_lattice(lp_utt, topo, device)
                candidates = extract_nbest_with_scores(
                    lattice, NUM_PATHS_OVERSAMPLE, NBEST_SCALE, sp, lp_cpu
                )
                del lattice
            except Exception:
                candidates = []

            greedy_entry = None
            rest = []
            for c in candidates:
                if c["text"] == greedy_text and greedy_entry is None:
                    greedy_entry = c
                else:
                    rest.append(c)
            if greedy_entry is None:
                greedy_entry = {
                    "text": greedy_text,
                    "tokens": greedy_collapsed,
                    "ctc_log_prob": greedy_score,
                    "len_tokens": len(greedy_collapsed),
                    "len_chars": len(greedy_text),
                }
            else:
                greedy_entry["ctc_log_prob"] = greedy_score
                greedy_entry["tokens"] = greedy_collapsed

            candidates = [greedy_entry] + rest
            candidates = [c for c in candidates if c["text"].strip()][:G]
            if not candidates:
                candidates = [greedy_entry]

            for c in candidates:
                c["ctc_log_prob"] = round(c["ctc_log_prob"], 6)

            record = {
                "utt_id": utt_id,
                "ref_text": ref_text,
                "num_candidates": len(candidates),
                "candidates": candidates,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_candidates += len(candidates)

            del log_probs, encoder_out, feats_gpu
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    avg = total_candidates / len(utterances)
    print(f"\n  Done: {total_candidates} candidates ({avg:.1f}/utt)")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Output: {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")

def compute_pll_batched(text, tokenizer, model, device, batch_size=64):
    """Pseudo-log-likelihood: sum_i log P(token_i | tokens_{-i}).

    Batched: scores up to batch_size masked positions in one forward pass.
    ~30x faster than per-position naive scoring for sentences with ~30 tokens.
    """
    import torch

    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"][0].to(device)
    L = input_ids.size(0)

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    cls = tokenizer.cls_token_id
    sep = tokenizer.sep_token_id
    mask_id = tokenizer.mask_token_id

    special = {bos, eos, pad, cls, sep}
    special.discard(None)

    positions = [i for i in range(L) if input_ids[i].item() not in special]
    if not positions:
        return 0.0

    total = 0.0
    with torch.no_grad():
        for s in range(0, len(positions), batch_size):
            e = min(s + batch_size, len(positions))
            batch_pos = positions[s:e]
            bsz = len(batch_pos)

            masked = input_ids.unsqueeze(0).repeat(bsz, 1).clone()
            for k, p in enumerate(batch_pos):
                masked[k, p] = mask_id

            logits = model(masked).logits  # (bsz, L, V)
            log_probs = torch.log_softmax(logits, dim=-1)
            for k, p in enumerate(batch_pos):
                total += log_probs[k, p, input_ids[p].item()].item()
    return total

def save_records(records, path):
    """Atomic write: write to .tmp then rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)

def step_score(args):
    print("\n" + "=" * 70)
    print("STEP 2: SCORE  --  Batched RoBERTa PLL + GPT-2 LL (with checkpointing)")
    print("=" * 70)

    nbest_path = args.output_dir / "nbest_zipformer_m_G128.jsonl"
    scored_path = args.output_dir / "neural_lm_scores_zipformer_m_G128.jsonl"

    if not nbest_path.exists():
        print(f"  ERROR: N-best file missing: {nbest_path}")
        print(f"  Run --steps generate first")
        return

    # If scored file exists with both fields, skip
    if scored_path.exists() and scored_path.stat().st_size > 0:
        with open(scored_path) as f:
            first = json.loads(f.readline())
        c0 = first["candidates"][0]
        if "roberta_pll" in c0 and "gpt2_ll" in c0:
            n = sum(1 for _ in open(scored_path))
            print(f"  SKIP: {scored_path.name} fully scored ({n} records)")
            return

    # Resume from partial scored file if it exists; else start from N-best
    source_path = scored_path if scored_path.exists() and scored_path.stat().st_size > 0 else nbest_path
    print(f"\n  Loading: {source_path}")
    records = load_jsonl(source_path)
    total_hyps = sum(len(r["candidates"]) for r in records)
    print(f"  {len(records)} utterances, {total_hyps} hypotheses")

    needs_pll = not all("roberta_pll" in c for r in records for c in r["candidates"])
    needs_gpt = not all("gpt2_ll" in c for r in records for c in r["candidates"])

    import torch
    device = torch.device(args.device)

    if needs_pll:
        print(f"\n  Loading RoBERTa-base...")
        from transformers import RobertaTokenizer, RobertaForMaskedLM
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        rob_model = RobertaForMaskedLM.from_pretrained("roberta-base").to(device)
        rob_model.eval()
        n_params = sum(p.numel() for p in rob_model.parameters())
        print(f"  {n_params/1e6:.1f}M parameters")

        # Sanity check
        print("\n  PLL sanity check:")
        sample_texts = []
        for rec in records[:3]:
            for c in rec["candidates"][:1]:
                if 3 <= len(c["text"].split()) <= 12:
                    sample_texts.append(c["text"])
        for t in sample_texts[:3]:
            pll = compute_pll_batched(t, tokenizer, rob_model, device, batch_size=64)
            print(f"    PLL={pll:8.2f}  text={t!r}")

        print(f"\n  Scoring with RoBERTa PLL (batched, checkpoint every "
              f"{CHECKPOINT_EVERY_N_UTTS} utts)...")
        t0 = time.time()
        n_scored = 0
        n_skipped = 0
        from tqdm import tqdm
        for i, rec in enumerate(tqdm(records, desc="  RoBERTa PLL")):
            for cand in rec["candidates"]:
                if "roberta_pll" in cand:
                    n_skipped += 1
                    continue
                text = cand["text"]
                if not text.strip():
                    cand["roberta_pll"] = -999.0
                    continue
                try:
                    pll = compute_pll_batched(
                        text, tokenizer, rob_model, device, batch_size=64
                    )
                    cand["roberta_pll"] = round(pll, 4)
                except Exception as ex:
                    cand["roberta_pll"] = -999.0
                    if n_scored < 5:
                        print(f"    PLL error on {rec['utt_id']!r}: {ex}")
                n_scored += 1

            # Periodic checkpoint
            if (i + 1) % CHECKPOINT_EVERY_N_UTTS == 0:
                save_records(records, scored_path)
                rate = n_scored / max(time.time() - t0, 1e-6)
                tqdm.write(f"    [checkpoint @ utt {i+1}] scored {n_scored} hyps, "
                           f"{rate:.1f} hyps/s, saved")

        save_records(records, scored_path)
        elapsed = time.time() - t0
        print(f"  RoBERTa done in {elapsed:.1f}s ({elapsed/60:.1f} min). "
              f"Scored {n_scored}, resumed {n_skipped}.")
        del rob_model
        torch.cuda.empty_cache()

    if needs_gpt:
        print(f"\n  Loading GPT-2...")
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        gpt_tok = GPT2Tokenizer.from_pretrained("gpt2")
        gpt_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        gpt_model.eval()
        n_params = sum(p.numel() for p in gpt_model.parameters())
        print(f"  {n_params/1e6:.1f}M parameters")

        print(f"  Scoring with GPT-2 LL...")
        t0 = time.time()
        from tqdm import tqdm
        for i, rec in enumerate(tqdm(records, desc="  GPT-2 LL")):
            for cand in rec["candidates"]:
                if "gpt2_ll" in cand:
                    continue
                text = cand["text"]
                if not text.strip():
                    cand["gpt2_ll"] = -999.0
                    continue
                enc = gpt_tok(text, return_tensors="pt").to(device)
                ids = enc["input_ids"]
                with torch.no_grad():
                    out = gpt_model(ids, labels=ids)
                    n_tok = ids.shape[1] - 1
                    ll = -out.loss.item() * n_tok
                cand["gpt2_ll"] = round(ll, 4)
            if (i + 1) % CHECKPOINT_EVERY_N_UTTS == 0:
                save_records(records, scored_path)
        save_records(records, scored_path)
        elapsed = time.time() - t0
        print(f"  GPT-2 done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        del gpt_model
        torch.cuda.empty_cache()

    print(f"\n  Final: {scored_path} "
          f"({scored_path.stat().st_size/1e6:.1f} MB)")

def step_evaluate(args, config):
    print("\n" + "=" * 70)
    print("STEP 3: EVALUATE  --  All methods + bootstrap + cross-model x cross-G")
    print("=" * 70)

    scored_path = args.output_dir / "neural_lm_scores_zipformer_m_G128.jsonl"
    if not scored_path.exists():
        print(f"  ERROR: {scored_path} missing. Run --steps score first.")
        return

    print(f"\nLoading: {scored_path}")
    records = load_jsonl(scored_path)
    n_utts = len(records)
    avg_cands = np.mean([r["num_candidates"] for r in records])
    print(f"  {n_utts} utterances, avg {avg_cands:.1f} candidates")

    ref_words = [r["ref_text"].split() for r in records]

    # Methods
    print("\nRunning method selection...")
    methods = {}
    print("  greedy...")
    methods["greedy"] = select_greedy(records)
    print("  oracle...")
    methods["oracle"] = select_oracle(records)
    for alpha in INTERP_ALPHAS:
        name = f"roberta_interp_a{alpha}"
        print(f"  {name}...")
        methods[name] = select_interp(records, alpha, "roberta_pll")
    for tau in MBR_PLL_TAUS:
        name = f"mbr_cer_pll_tau{int(tau)}"
        print(f"  {name}... (this takes a while at G=128)")
        t0 = time.time()
        methods[name] = select_mbr_pll(records, tau)
        print(f"    done in {time.time()-t0:.1f}s")
    print("  mbr_cer_ctc_tau_inf (uniform)...")
    t0 = time.time()
    methods["mbr_cer_ctc_tau_inf"] = select_mbr_ctc(records, float("inf"))
    print(f"    done in {time.time()-t0:.1f}s")

    # WERs
    wers = {}
    for name, hyps in methods.items():
        hyp_w = [h.split() for h in hyps]
        wers[name] = corpus_wer(ref_words, hyp_w)
    print("\nCorpus WERs:")
    for name, w in wers.items():
        print(f"  {name:30s}: {w*100:.4f}%")

    greedy_wer = wers["greedy"]
    oracle_wer = wers["oracle"]

    print(f"\nBootstrap (B={args.n_bootstrap}) vs greedy...")
    greedy_w = [h.split() for h in methods["greedy"]]
    bootstrap = {}
    boot_methods = [k for k in methods if k not in ("greedy", "oracle")]
    for name in boot_methods:
        hyp_w = [h.split() for h in methods[name]]
        res = paired_bootstrap_wer(
            ref_words, hyp_w, greedy_w,
            n_bootstrap=args.n_bootstrap, seed=args.seed
        )
        bootstrap[name] = {
            "wer": res["wer_a"],
            "delta_pp": res["delta"] * 100,
            "p_value": res["p_value"],
            "ci_lower": res["ci_lower"] * 100,
            "ci_upper": res["ci_upper"] * 100,
        }
        print(f"  {name:30s}: delta={res['delta']*100:+.4f}pp, "
              f"p={res['p_value']:.4f}")

    # Spearman
    print(f"\nSpearman rho at G={G}...")
    rho_ctc = per_utt_spearman(records, lambda c: c["ctc_log_prob"])
    rho_pll = per_utt_spearman(records, lambda c: c["roberta_pll"])
    print(f"  CTC rho:        {rho_ctc:.4f}")
    print(f"  RoBERTa PLL rho: {rho_pll:.4f}")

    # Verification
    print("\n--- Verification ---")
    expected_greedy = config["expected_dev_other_wer"]
    if abs(greedy_wer * 100 - expected_greedy) < 0.5:
        print(f"  [PASS] Greedy {greedy_wer*100:.4f}% ~ {expected_greedy}% "
              f"(model card)")
    else:
        print(f"  [WARN] Greedy {greedy_wer*100:.4f}% vs expected "
              f"{expected_greedy}%")
    if oracle_wer < greedy_wer:
        print(f"  [PASS] Oracle ({oracle_wer*100:.4f}%) < Greedy "
              f"({greedy_wer*100:.4f}%)")
    if avg_cands > 100:
        print(f"  [PASS] Avg candidates {avg_cands:.1f} > 100 (G=128 spec)")
    else:
        print(f"  [WARN] Avg candidates {avg_cands:.1f} below 100")
    print(f"  [INFO] Utterance count: {n_utts}")

    # Compare to E16 G=16 numbers (greedy must match  --  G-independent)
    e16_greedy = 4.7755  # from results/zipformer_m/zipformer_m_results.json
    if abs(greedy_wer * 100 - e16_greedy) < 0.001:
        print(f"  [PASS] Greedy matches E16 G=16: {greedy_wer*100:.4f}% "
              f"vs {e16_greedy}%")
    else:
        print(f"  [WARN] Greedy {greedy_wer*100:.4f}% != E16 {e16_greedy}%")
    e16_oracle = 3.4427
    if oracle_wer * 100 < e16_oracle:
        print(f"  [PASS] Oracle G=128 ({oracle_wer*100:.4f}%) < "
              f"Oracle G=16 ({e16_oracle}%)")
    else:
        print(f"  [WARN] Oracle G=128 should be < Oracle G=16")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("\n--- Writing outputs ---")

    results = {
        "experiment": "E16b_zipformer_m_g128",
        "model_config": config["name"],
        "model_dir": str(args.model_dir),
        "n_utterances": n_utts,
        "G": G,
        "avg_candidates": float(avg_cands),
        "wers": {k: float(v) for k, v in wers.items()},
        "oracle_gap_pp": (greedy_wer - oracle_wer) * 100,
        "oracle_gap_relative": (greedy_wer - oracle_wer) / greedy_wer * 100,
        "bootstrap": bootstrap,
        "spearman": {"ctc": rho_ctc, "roberta_pll": rho_pll},
        "n_bootstrap": args.n_bootstrap,
    }
    p = args.output_dir / "zipformer_m_g128_results.json"
    with open(p, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Wrote {p}")

    p = args.output_dir / "zipformer_m_g128_bootstrap.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "method", "wer_pct", "delta_pp", "p_value", "ci_lower", "ci_upper"
        ])
        w.writeheader()
        for name, b in bootstrap.items():
            w.writerow({
                "method": name,
                "wer_pct": f"{b['wer']*100:.4f}",
                "delta_pp": f"{b['delta_pp']:+.4f}",
                "p_value": f"{b['p_value']:.4f}",
                "ci_lower": f"{b['ci_lower']:+.4f}",
                "ci_upper": f"{b['ci_upper']:+.4f}",
            })
    print(f"  Wrote {p}")

    p = args.output_dir / "zipformer_m_g128_spearman.json"
    with open(p, "w") as f:
        json.dump({"ctc": rho_ctc, "roberta_pll": rho_pll,
                   "G": G, "n_utterances": n_utts}, f, indent=2)
    print(f"  Wrote {p}")

    # Cross-model x cross-G table
    write_cross_model_g_scaling(args, config, wers, bootstrap)

    # Stage report
    write_report(args, config, wers, bootstrap, rho_ctc, rho_pll, n_utts,
                 avg_cands)

def write_cross_model_g_scaling(args, config, wers_m_g128, bootstrap_m_g128):
    """THE key deliverable: 4-row table comparing model x G."""
    p = args.output_dir / "cross_model_g_scaling.md"

    # Hardcoded reference numbers from prior experiments
    # (Zipformer-S G=16 from E5; Zipformer-S G=128 from E11; Zipformer-M G=16 from E16)
    s_g16 = {"greedy": 6.0218, "oracle": 4.4418, "mbr10": 5.7902,
             "best_interp": 5.92, "best_alpha": 0.7}
    s_g128 = {"greedy": 6.0218, "oracle": 3.535, "mbr10": 5.5292,
              "best_interp": 5.89, "best_alpha": 0.8}
    m_g16 = {"greedy": 4.7755, "oracle": 3.4427, "mbr10": 4.5556,
             "best_interp": 4.7185, "best_alpha": 0.7}

    g_m = wers_m_g128["greedy"] * 100
    o_m = wers_m_g128["oracle"] * 100
    mbr10_m = wers_m_g128["mbr_cer_pll_tau10"] * 100
    interps = [(a, wers_m_g128[f"roberta_interp_a{a}"] * 100)
               for a in INTERP_ALPHAS]
    best_alpha_m, best_interp_m = min(interps, key=lambda t: t[1])

    def gap_closed(g, o, mbr):
        return (g - mbr) / max(g - o, 0.01) * 100

    rows = [
        ("Zipformer-S (22M)", 16, s_g16["greedy"], s_g16["oracle"],
         s_g16["mbr10"], s_g16["best_interp"], s_g16["best_alpha"],
         gap_closed(s_g16["greedy"], s_g16["oracle"], s_g16["mbr10"])),
        ("Zipformer-S (22M)", 128, s_g128["greedy"], s_g128["oracle"],
         s_g128["mbr10"], s_g128["best_interp"], s_g128["best_alpha"],
         gap_closed(s_g128["greedy"], s_g128["oracle"], s_g128["mbr10"])),
        ("Zipformer-M (65M)", 16, m_g16["greedy"], m_g16["oracle"],
         m_g16["mbr10"], m_g16["best_interp"], m_g16["best_alpha"],
         gap_closed(m_g16["greedy"], m_g16["oracle"], m_g16["mbr10"])),
        ("Zipformer-M (65M)", 128, g_m, o_m,
         mbr10_m, best_interp_m, best_alpha_m,
         gap_closed(g_m, o_m, mbr10_m)),
    ]

    lines = ["# Cross-Model x Cross-G Scaling Table", ""]
    lines.append("Does MBR-CER+PLL tau=10 scale with G on both architectures?")
    lines.append("")
    lines.append("| Model | G | Greedy | Oracle | MBR+PLL tau=10 | "
                 "Best Interp (alpha) | Gap Closed (MBR) |")
    lines.append("|-------|--:|-------:|-------:|-------------:|"
                 "----------------:|-----------------:|")
    for name, g_val, gw, ow, mbr, interp, alpha, gc in rows:
        lines.append(f"| {name} | {g_val} | {gw:.2f}% | {ow:.2f}% | "
                     f"{mbr:.2f}% | {interp:.2f}% (alpha={alpha}) | {gc:.1f}% |")
    lines.append("")

    # Per-model G=16 -> G=128 deltas
    lines.append("## Per-Model G Scaling")
    lines.append("")
    s_mbr_delta = s_g16["mbr10"] - s_g128["mbr10"]
    s_interp_delta = s_g16["best_interp"] - s_g128["best_interp"]
    m_mbr_delta = m_g16["mbr10"] - mbr10_m
    m_interp_delta = m_g16["best_interp"] - best_interp_m

    lines.append("| Model | MBR+PLL: G=16 -> G=128 | Best Interp: G=16 -> G=128 |")
    lines.append("|-------|----------------------:|--------------------------:|")
    lines.append(f"| Zipformer-S | {s_g16['mbr10']:.2f}% -> {s_g128['mbr10']:.2f}% "
                 f"({s_mbr_delta:+.2f}pp) | "
                 f"{s_g16['best_interp']:.2f}% -> {s_g128['best_interp']:.2f}% "
                 f"({s_interp_delta:+.2f}pp) |")
    lines.append(f"| Zipformer-M | {m_g16['mbr10']:.2f}% -> {mbr10_m:.2f}% "
                 f"({m_mbr_delta:+.2f}pp) | "
                 f"{m_g16['best_interp']:.2f}% -> {best_interp_m:.2f}% "
                 f"({m_interp_delta:+.2f}pp) |")
    lines.append("")

    # Verdict
    lines.append("## Verdict")
    lines.append("")
    s_mbr_scales = s_mbr_delta > 0.05
    m_mbr_scales = m_mbr_delta > 0.05
    s_interp_flat = abs(s_interp_delta) < 0.1
    m_interp_flat = abs(m_interp_delta) < 0.1

    if s_mbr_scales and m_mbr_scales:
        lines.append(f"**MBR scales on both architectures.** Zipformer-S gains "
                     f"{s_mbr_delta:.2f}pp from G=16->G=128; Zipformer-M gains "
                     f"{m_mbr_delta:.2f}pp. The G-scaling property is structural, "
                     "not specific to the small model.")
    elif m_mbr_scales:
        lines.append(f"MBR scales on Zipformer-M ({m_mbr_delta:.2f}pp from "
                     "G=16->G=128). Confirms the F8 finding generalizes.")
    else:
        lines.append(f"MBR did not scale meaningfully on Zipformer-M "
                     f"({m_mbr_delta:+.2f}pp). The larger model's smaller absolute "
                     "oracle gap may have left less room.")
    lines.append("")

    if s_interp_flat and m_interp_flat:
        lines.append("**Linear interpolation plateaus on both architectures.** "
                     f"Zipformer-S: {s_interp_delta:+.2f}pp; "
                     f"Zipformer-M: {m_interp_delta:+.2f}pp. "
                     "argmax-based methods cannot exploit larger candidate sets  --  "
                     "this is the central asymmetry the paper documents.")
    lines.append("")

    with open(p, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {p}")

def write_report(args, config, wers, bootstrap, rho_ctc, rho_pll, n_utts,
                 avg_cands):
    p = args.output_dir / "report_E16b.md"
    g = wers["greedy"] * 100
    o = wers["oracle"] * 100
    mbr10 = wers["mbr_cer_pll_tau10"] * 100
    mbr50 = wers.get("mbr_cer_pll_tau50", float("nan")) * 100
    mbr_inf = wers["mbr_cer_ctc_tau_inf"] * 100

    p_mbr10 = bootstrap["mbr_cer_pll_tau10"]["p_value"]

    lines = ["# Report E16b: Zipformer-M G=128  --  Cross-Model Scaling", ""]
    lines.append(f"**Model:** {config['name']}")
    lines.append(f"**Status:** Complete. {n_utts} utterances, G={G}, "
                 f"B={args.n_bootstrap} bootstrap.")
    lines.append("")
    lines.append("## What Ran")
    lines.append("")
    lines.append(f"- Pipeline: discover -> generate -> score -> evaluate")
    lines.append(f"- N-best: G={G}, oversample={NUM_PATHS_OVERSAMPLE}, "
                 f"avg {avg_cands:.1f} candidates per utterance")
    lines.append(f"- Methods: greedy, oracle, RoBERTa interp alphain{INTERP_ALPHAS}, "
                 f"MBR+PLL tauin{[int(t) for t in MBR_PLL_TAUS]}, MBR uniform")
    lines.append(f"- Bootstrap: B={args.n_bootstrap}, paired vs greedy")
    lines.append(f"- Optimizations: batched PLL (~30x speedup), "
                 "checkpointing every 200 utts")
    lines.append("")
    lines.append("## Key Results")
    lines.append("")
    lines.append("| Method | WER (%) | delta vs greedy (pp) | p-value |")
    lines.append("|--------|--------:|-----------------:|--------:|")
    lines.append(f"| Greedy | {g:.4f} | 0 |  --  |")
    for name, b in bootstrap.items():
        lines.append(f"| {name} | {b['wer']*100:.4f} | "
                     f"{b['delta_pp']:+.4f} | {b['p_value']:.4f} |")
    lines.append(f"| Oracle | {o:.4f} | {(o-g):+.4f} |  --  |")
    lines.append("")
    lines.append("## Spearman Correlations at G=128")
    lines.append("")
    lines.append(f"- CTC log-prob rho: **{rho_ctc:.4f}** "
                 f"(Zipformer-S G=128: ~-0.30 typical)")
    lines.append(f"- RoBERTa PLL rho: **{rho_pll:.4f}** "
                 f"(Zipformer-S G=128: ~-0.46)")
    lines.append("")
    lines.append("## Comparison vs Zipformer-M G=16 (E16)")
    lines.append("")
    lines.append("| Metric | G=16 (E16) | G=128 (E16b) | delta |")
    lines.append("|--------|-----------:|-------------:|----:|")
    lines.append(f"| Greedy | 4.7755% | {g:.4f}% | {(g-4.7755):+.4f}pp |")
    lines.append(f"| Oracle | 3.4427% | {o:.4f}% | {(o-3.4427):+.4f}pp |")
    lines.append(f"| MBR+PLL tau=10 | 4.5556% | {mbr10:.4f}% | "
                 f"{(mbr10-4.5556):+.4f}pp |")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    if mbr10 < g and p_mbr10 < 0.05:
        gain_g16 = 4.7755 - 4.5556
        gain_g128 = g - mbr10
        if gain_g128 > gain_g16:
            lines.append(f"**MBR scales with G on Zipformer-M.** "
                         f"G=16 gain: -{gain_g16:.2f}pp; G=128 gain: -{gain_g128:.2f}pp. "
                         "The MBR-vs-interp asymmetry replicates on a second architecture.")
        else:
            lines.append(f"MBR remains significant at G=128 (p={p_mbr10:.4f}) "
                         f"but doesn't scale further (G=16 gain {gain_g16:.2f}pp, "
                         f"G=128 gain {gain_g128:.2f}pp).")
    else:
        lines.append(f"MBR+PLL tau=10 does not show significant improvement at G=128 "
                     f"(p={p_mbr10:.4f}). Investigate whether the oracle gap "
                     "permits larger gains.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("| File | Purpose |")
    lines.append("|------|---------|")
    lines.append("| `zipformer_m_g128_results.json` | Full results |")
    lines.append("| `zipformer_m_g128_bootstrap.csv` | Bootstrap p-values |")
    lines.append("| `zipformer_m_g128_spearman.json` | Per-utt Spearman rho |")
    lines.append("| `cross_model_g_scaling.md` | **THE** 4-row comparison table |")
    lines.append("| `report_E16b.md` | This stage report |")
    with open(p, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {p}")

def parse_args():
    parser = argparse.ArgumentParser(description="E16b: Zipformer-M G=128")
    parser.add_argument("--data-dir", type=Path,
                        default=Path("/content/drive/MyDrive/rbpo_results"))
    parser.add_argument("--librispeech-dir", type=Path,
                        default=Path("/content/librispeech_data"))
    parser.add_argument("--model-dir", type=Path,
                        default=Path("/content/icefall-asr-librispeech-zipformer-medium-cr-ctc"))
    parser.add_argument("--icefall-dir", type=Path,
                        default=Path("/content/icefall"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/content/drive/MyDrive/rbpo_results/zipformer_m_g128"))
    parser.add_argument("--model-config", type=str, default="zipformer-m-cr-ctc",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--steps", type=str, default="all",
                        help="Comma-separated: discover,generate,score,evaluate or 'all'")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = MODEL_CONFIGS[args.model_config]

    steps = (args.steps.lower().split(",") if args.steps != "all"
             else ["discover", "generate", "score", "evaluate"])

    print("=" * 70)
    print(f"E16b: Zipformer-M G={G}  --  Cross-Model Scaling")
    print("=" * 70)
    print(f"  Model dir:    {args.model_dir}")
    print(f"  Output dir:   {args.output_dir}")
    print(f"  Steps:        {steps}")
    print(f"  G:            {G} (oversample={NUM_PATHS_OVERSAMPLE})")
    print(f"  Bootstrap:    B={args.n_bootstrap}")

    t0 = time.time()
    if "discover" in steps:
        step_discover(args)
    if "generate" in steps:
        step_generate(args, config)
    if "score" in steps:
        step_score(args)
    if "evaluate" in steps:
        step_evaluate(args, config)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"DONE. Total: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()

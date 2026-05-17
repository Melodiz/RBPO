#!/usr/bin/env python3
"""E16: Cross-Model Verification  --  Zipformer-M (or other second CTC model).

Tests whether MBR-CER + RoBERTa PLL tau=10 generalizes to a different model
size. The script is designed to be flexible: it accepts model configs via
preset names or explicit CLI overrides.

Pipeline (all steps resumable, output to Drive):
  discover   --  Inventory existing N-best/scored files
  generate   --  Build G=16 N-best for the new model (GPU)
  score      --  RoBERTa PLL + GPT-2 LL (GPU)
  evaluate   --  All methods + bootstrap + Spearman + cross-model comparison

Model selection (use --model-config or override individually):
  zipformer-s-cr-ctc   --  Small (22M)   layers=2,2,2,2,2,2 dim=192,256,256,256,256,256
  zipformer-m-cr-ctc   --  Medium (~65M) layers=2,2,3,4,3,2 dim=192,256,384,512,384,256
  zipformer-l-cr-ctc   --  Large (~150M) layers=2,2,4,5,4,2 dim=192,256,512,768,512,256

Available CR-CTC checkpoints (HuggingFace, by Zengwei):
  Medium: Zengwei/icefall-asr-librispeech-zipformer-medium-cr-ctc-20241018  (64M, test-other 4.61%)
  Large:  Zengwei/icefall-asr-librispeech-zipformer-large-cr-ctc-20241018   (147M, test-other 4.35%)
  Small:  (not on HF  --  may need to train locally or use a non-CR-CTC small model)

Each repo's structure:
  exp/pretrained.pt            # 257 MB for medium
  data/lang_bpe_500/bpe.model  # SentencePiece tokenizer
  exp/train.sh, decode.sh      # original training/decoding commands

Usage (Colab T4):
    python experiments/evaluation/eval_zipformer_m.py \
        --data-dir /content/drive/MyDrive/rbpo_results \
        --librispeech-dir /content/librispeech_data \
        --model-dir /content/icefall-asr-librispeech-zipformer-medium-cr-ctc \
        --icefall-dir /content/icefall \
        --output-dir /content/drive/MyDrive/rbpo_results/zipformer_m \
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

from experiments.significance_tests import paired_bootstrap_wer, corpus_wer

BLANK_ID = 0
MAX_TOKEN = 499
NBEST_SCALE = 1.0
G = 16
NUM_PATHS_OVERSAMPLE = 64

# Model configs by name. Use --model-config to pick, override with explicit flags.
MODEL_CONFIGS = {
    "zipformer-s-cr-ctc": {
        "name": "Zipformer-S CR-CTC",
        "num_encoder_layers": "2,2,2,2,2,2",
        "encoder_dim": "192,256,256,256,256,256",
        "encoder_unmasked_dim": "192,192,192,192,192,192",
        "feedforward_dim": "512,768,768,768,768,768",
        "expected_params_m": 22.0,
        "expected_dev_other_wer": 6.02,
    },
    "zipformer-m-cr-ctc": {
        "name": "Zipformer-M CR-CTC",
        "num_encoder_layers": "2,2,3,4,3,2",
        "encoder_dim": "192,256,384,512,384,256",
        "encoder_unmasked_dim": "192,192,256,256,256,192",
        "feedforward_dim": "512,768,1024,1536,1024,768",
        "expected_params_m": 65.0,
        "expected_dev_other_wer": 4.50,
    },
    "zipformer-l-cr-ctc": {
        "name": "Zipformer-L CR-CTC",
        "num_encoder_layers": "2,2,4,5,4,2",
        "encoder_dim": "192,256,512,768,512,256",
        "encoder_unmasked_dim": "192,192,256,256,256,192",
        "feedforward_dim": "512,768,1536,2048,1536,768",
        "expected_params_m": 148.0,
        "expected_dev_other_wer": 4.10,
    },
}

INTERP_ALPHAS = [0.5, 0.6, 0.7, 0.8, 0.9]
MBR_PLL_TAUS = [5.0, 10.0, 50.0, float("inf")]

def add_icefall_to_path(icefall_dir: Path):
    for d in [
        icefall_dir,
        icefall_dir / "egs" / "librispeech" / "ASR",
        icefall_dir / "egs" / "librispeech" / "ASR" / "zipformer",
    ]:
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))

def load_model(model_dir: Path, icefall_dir: Path, device, config: dict):
    import torch
    add_icefall_to_path(icefall_dir)
    from train import add_model_arguments, get_model, get_params

    params = get_params()
    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    model_args = parser.parse_args([])
    for k, v in vars(model_args).items():
        params[k] = v

    params.num_encoder_layers = config["num_encoder_layers"]
    params.encoder_dim = config["encoder_dim"]
    params.encoder_unmasked_dim = config["encoder_unmasked_dim"]
    params.feedforward_dim = config["feedforward_dim"]
    params.use_transducer = False
    params.use_ctc = True
    params.use_cr_ctc = True
    params.use_attention_decoder = False
    params.vocab_size = 500
    params.feature_dim = 80

    print(f"  Architecture: {config['name']}")
    print(f"    layers: {params.num_encoder_layers}")
    print(f"    dim:    {params.encoder_dim}")
    print(f"    ff:     {params.feedforward_dim}")

    model = get_model(params)

    # Try multiple checkpoint locations
    ckpt_candidates = [
        model_dir / "exp" / "pretrained.pt",
        model_dir / "pretrained.pt",
        model_dir / "exp" / "epoch-30.pt",
    ]
    ckpt_path = next((p for p in ckpt_candidates if p.exists()), None)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No checkpoint found in {model_dir}. Tried: {ckpt_candidates}"
        )
    print(f"  Loading: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  WARN: {len(missing)} missing keys")
    if unexpected:
        print(f"  WARN: {len(unexpected)} unexpected keys")
    model.eval()
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_params_m = n_params / 1e6
    print(f"  Parameters: {n_params_m:.1f}M (expected ~{config['expected_params_m']}M)")
    if abs(n_params_m - config["expected_params_m"]) > config["expected_params_m"] * 0.2:
        print(f"  WARN: parameter count differs >20% from expected")
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
    import torch
    T = log_probs.shape[0]
    supervision_segments = torch.tensor([[0, 0, T]], dtype=torch.int32)
    dense_fsa = k2.DenseFsaVec(log_probs.unsqueeze(0), supervision_segments)
    lattice = k2.intersect_dense(topo, dense_fsa, output_beam=8.0)
    lattice = k2.connect(lattice)
    return lattice

def alignment_log_prob(label_seq, log_probs_cpu):
    import torch
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
    return sorted(seen.values(), key=lambda c: c["ctc_log_prob"], reverse=True)

def load_all_utterances(librispeech_dir: Path, split: str):
    import torch
    from lhotse import load_manifest_lazy
    cuts_path = librispeech_dir / "cuts" / f"librispeech_cuts_{split}.jsonl.gz"
    assert cuts_path.exists(), f"CutSet not found: {cuts_path}"
    cuts = load_manifest_lazy(str(cuts_path))
    utterances = []
    for cut in cuts:
        feats = torch.from_numpy(cut.load_features())
        ref_text = " ".join(s.text for s in cut.supervisions if s.text).strip().lower()
        if not ref_text:
            continue
        utterances.append((cut.id, feats, ref_text))
    print(f"  Loaded {len(utterances)} utterances from {split}")
    return utterances

def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records

def step_discover(args):
    print("\n" + "=" * 70)
    print("STEP 0: DISCOVER  --  Inventory existing data")
    print("=" * 70)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    nbest_path = args.output_dir / "nbest_zipformer_m_G16.jsonl"
    scored_path = args.output_dir / "neural_lm_scores_zipformer_m.jsonl"

    print(f"\n  Output dir: {args.output_dir}")
    if nbest_path.exists() and nbest_path.stat().st_size > 0:
        n = sum(1 for _ in open(nbest_path))
        print(f"  N-best: EXISTS ({nbest_path.name}, {n} records, "
              f"{nbest_path.stat().st_size/1e6:.1f} MB)")
    else:
        print(f"  N-best: MISSING (need to generate)")

    if scored_path.exists() and scored_path.stat().st_size > 0:
        n = sum(1 for _ in open(scored_path))
        print(f"  Scored: EXISTS ({scored_path.name}, {n} records, "
              f"{scored_path.stat().st_size/1e6:.1f} MB)")
    else:
        print(f"  Scored: MISSING (need to score)")

    if args.model_dir.exists():
        print(f"\n  Model dir: {args.model_dir} (exists)")
    else:
        print(f"\n  Model dir: {args.model_dir} (DOES NOT EXIST)")
        print("  Download with: hf download <model-name> --local-dir <model-dir>")

def step_generate(args, config):
    print("\n" + "=" * 70)
    print("STEP 1: GENERATE  --  Build G=16 N-best")
    print("=" * 70)

    out_path = args.output_dir / "nbest_zipformer_m_G16.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0:
        n = sum(1 for _ in open(out_path))
        print(f"  SKIP: {out_path} exists ({n} records)")
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

    # Verify greedy WER on first 50 utterances before committing to full run
    print(f"\n  Verifying greedy WER on first 50 utterances...")
    verify_errors = 0
    verify_words = 0
    for i, (utt_id, feats, ref_text) in enumerate(utterances[:50]):
        feats_gpu = feats.unsqueeze(0).to(device)
        feat_lens = torch.tensor([feats.shape[0]], dtype=torch.int64, device=device)
        with torch.no_grad():
            encoder_out, _ = model.forward_encoder(feats_gpu, feat_lens)
            log_probs = model.ctc_output(encoder_out)
        greedy_ids = log_probs[0].argmax(dim=-1).tolist()
        greedy_text = sp.decode(ctc_collapse(greedy_ids)).strip().lower()
        ref_w = ref_text.split()
        verify_errors += editdistance.eval(greedy_text.split(), ref_w)
        verify_words += len(ref_w)
        del log_probs, encoder_out
    verify_wer = verify_errors / max(verify_words, 1) * 100
    expected = config["expected_dev_other_wer"]
    print(f"  Greedy WER on 50 utts: {verify_wer:.2f}% (full-set expected ~{expected}%)")
    if abs(verify_wer - expected) > 2.0:
        print(f"  WARN: WER differs from expected by >2pp. Model config may be wrong.")
        print(f"  Continuing anyway  --  verify final corpus WER in evaluate step.")

    print(f"\n  Generating N-best for {len(utterances)} utterances...")
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

def step_score(args):
    print("\n" + "=" * 70)
    print("STEP 2: SCORE  --  RoBERTa PLL + GPT-2 LL")
    print("=" * 70)

    nbest_path = args.output_dir / "nbest_zipformer_m_G16.jsonl"
    scored_path = args.output_dir / "neural_lm_scores_zipformer_m.jsonl"

    if scored_path.exists() and scored_path.stat().st_size > 0:
        # Verify it has both score fields
        with open(scored_path) as f:
            first = json.loads(f.readline())
        c = first["candidates"][0]
        if "roberta_pll" in c and "gpt2_ll" in c:
            n = sum(1 for _ in open(scored_path))
            print(f"  SKIP: {scored_path.name} already scored ({n} records)")
            return
        print(f"  Existing scored file is incomplete; re-scoring")

    if not nbest_path.exists():
        print(f"  ERROR: N-best file missing: {nbest_path}")
        print(f"  Run --steps generate first")
        return

    import torch
    from tqdm import tqdm
    device = torch.device(args.device)

    print(f"\n  Loading {nbest_path}")
    records = load_jsonl(nbest_path)
    total_hyps = sum(len(r["candidates"]) for r in records)
    print(f"  {len(records)} utterances, {total_hyps} hypotheses")

    # RoBERTa PLL
    print(f"\n  Loading RoBERTa-base...")
    from transformers import RobertaTokenizer, RobertaForMaskedLM
    rob_tok = RobertaTokenizer.from_pretrained("roberta-base")
    rob_model = RobertaForMaskedLM.from_pretrained("roberta-base").to(device)
    rob_model.eval()
    n_params = sum(p.numel() for p in rob_model.parameters())
    print(f"  {n_params/1e6:.1f}M parameters")

    print(f"  Scoring with RoBERTa PLL...")
    t0 = time.time()
    for rec in tqdm(records, desc="  RoBERTa"):
        for cand in rec["candidates"]:
            text = cand["text"]
            if not text.strip():
                cand["roberta_pll"] = -999.0
                continue
            enc = rob_tok(text, return_tensors="pt").to(device)
            input_ids = enc["input_ids"][0]
            n_tok = len(input_ids) - 2
            if n_tok <= 0:
                cand["roberta_pll"] = -999.0
                continue
            pll = 0.0
            for i in range(1, n_tok + 1):
                masked = input_ids.clone()
                masked[i] = rob_tok.mask_token_id
                with torch.no_grad():
                    out = rob_model(masked.unsqueeze(0),
                                    attention_mask=enc["attention_mask"])
                logits = out.logits[0, i]
                lp = torch.log_softmax(logits, dim=-1)
                pll += lp[input_ids[i]].item()
            cand["roberta_pll"] = round(pll, 4)
    print(f"  RoBERTa done in {time.time()-t0:.1f}s")
    del rob_model
    torch.cuda.empty_cache()

    # GPT-2 LL
    print(f"\n  Loading GPT-2...")
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    gpt_tok = GPT2Tokenizer.from_pretrained("gpt2")
    gpt_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    gpt_model.eval()
    n_params = sum(p.numel() for p in gpt_model.parameters())
    print(f"  {n_params/1e6:.1f}M parameters")

    print(f"  Scoring with GPT-2 LL...")
    t0 = time.time()
    for rec in tqdm(records, desc="  GPT-2"):
        for cand in rec["candidates"]:
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
    print(f"  GPT-2 done in {time.time()-t0:.1f}s")
    del gpt_model
    torch.cuda.empty_cache()

    # Save
    with open(scored_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n  Wrote {scored_path} "
          f"({scored_path.stat().st_size/1e6:.1f} MB)")

def compute_cer_matrix(texts):
    n = len(texts)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = editdistance.eval(list(texts[i]), list(texts[j]))
            denom = max(len(texts[i]), len(texts[j]), 1)
            mat[i, j] = d / denom
            mat[j, i] = mat[i, j]
    return mat

def select_greedy(records):
    return [r["candidates"][0]["text"] for r in records]

def select_oracle(records):
    out = []
    for rec in records:
        ref = rec["ref_text"]
        best = min(rec["candidates"],
                   key=lambda c: editdistance.eval(c["text"].split(), ref.split()))
        out.append(best["text"])
    return out

def select_interp(records, alpha, score_field):
    out = []
    for rec in records:
        cands = [c for c in rec["candidates"] if c["text"].strip()]
        if not cands:
            cands = rec["candidates"]
        scores = [alpha * c["ctc_log_prob"] + (1 - alpha) * c[score_field]
                  for c in cands]
        out.append(cands[int(np.argmax(scores))]["text"])
    return out

def select_mbr_pll(records, tau):
    out = []
    for rec in records:
        cands = [c for c in rec["candidates"] if c["text"].strip()]
        if not cands:
            cands = rec["candidates"]
        n = len(cands)
        texts = [c["text"] for c in cands]
        log_scores = np.array([c["roberta_pll"] for c in cands])
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
    out = []
    for rec in records:
        cands = [c for c in rec["candidates"] if c["text"].strip()]
        if not cands:
            cands = rec["candidates"]
        n = len(cands)
        texts = [c["text"] for c in cands]
        log_scores = np.array([c["ctc_log_prob"] for c in cands])
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

def per_utt_spearman(records, score_fn):
    from scipy import stats as scipy_stats
    rhos = []
    for rec in records:
        cands = rec["candidates"]
        if len(cands) < 3:
            continue
        ref_w = rec["ref_text"].split()
        wers = [editdistance.eval(c["text"].split(), ref_w) / max(len(ref_w), 1)
                for c in cands]
        scores = [score_fn(c) for c in cands]
        if len(set(scores)) < 2 or len(set(wers)) < 2:
            continue
        rho, _ = scipy_stats.spearmanr(scores, wers)
        if not np.isnan(rho):
            rhos.append(rho)
    return float(np.mean(rhos)) if rhos else float("nan")

def step_evaluate(args, config):
    print("\n" + "=" * 70)
    print("STEP 3: EVALUATE  --  All methods + bootstrap + cross-model")
    print("=" * 70)

    scored_path = args.output_dir / "neural_lm_scores_zipformer_m.jsonl"
    if not scored_path.exists():
        print(f"  ERROR: {scored_path} missing. Run --steps score first.")
        return

    print(f"\nLoading: {scored_path}")
    records = load_jsonl(scored_path)
    n_utts = len(records)
    print(f"  {n_utts} utterances")

    ref_words = [r["ref_text"].split() for r in records]

    # Methods
    print("\nRunning method selection...")
    methods = {}
    print("  greedy...")
    methods["greedy"] = select_greedy(records)
    print("  oracle...")
    methods["oracle"] = select_oracle(records)
    print("  RoBERTa interp alpha=0.7...")
    methods["roberta_interp_a0.7"] = select_interp(records, 0.7, "roberta_pll")
    print("  MBR-CER + PLL tau=10...")
    methods["mbr_cer_pll_tau10"] = select_mbr_pll(records, 10.0)
    print("  MBR-CER tau=inf (uniform)...")
    methods["mbr_cer_ctc_tau_inf"] = select_mbr_ctc(records, float("inf"))

    # WERs
    wers = {}
    for name, hyps in methods.items():
        hyp_w = [h.split() for h in hyps]
        wers[name] = corpus_wer(ref_words, hyp_w)
        print(f"  {name}: WER={wers[name]*100:.4f}%")

    greedy_wer = wers["greedy"]
    oracle_wer = wers["oracle"]
    gap = greedy_wer - oracle_wer

    print(f"\nBootstrap (B={args.n_bootstrap})...")
    greedy_w = [h.split() for h in methods["greedy"]]
    bootstrap = {}
    for name in ["roberta_interp_a0.7", "mbr_cer_pll_tau10", "mbr_cer_ctc_tau_inf"]:
        hyp_w = [h.split() for h in methods[name]]
        res = paired_bootstrap_wer(ref_words, hyp_w, greedy_w,
                                   n_bootstrap=args.n_bootstrap, seed=args.seed)
        bootstrap[name] = {
            "wer": res["wer_a"],
            "delta_pp": res["delta"] * 100,
            "p_value": res["p_value"],
            "ci_lower": res["ci_lower"] * 100,
            "ci_upper": res["ci_upper"] * 100,
        }
        print(f"  {name}: delta={res['delta']*100:+.4f}pp, p={res['p_value']:.4f}")

    # Sanity: greedy vs greedy -> delta=0, p=1.0 (one-sided test degenerates when A=B)
    boot_self = paired_bootstrap_wer(ref_words, greedy_w, greedy_w,
                                     n_bootstrap=1000, seed=args.seed)
    print(f"  greedy vs greedy: delta={boot_self['delta']:.6f}, "
          f"p={boot_self['p_value']:.4f} (expect delta=0, p=1.0)")

    # Spearman
    print(f"\nSpearman correlations...")
    rho_ctc = per_utt_spearman(records, lambda c: c["ctc_log_prob"])
    rho_pll = per_utt_spearman(records, lambda c: c["roberta_pll"])
    print(f"  CTC rho:        {rho_ctc:.4f}")
    print(f"  RoBERTa PLL rho: {rho_pll:.4f}")

    # Verification
    print("\n--- Verification ---")
    expected_wer = config["expected_dev_other_wer"]
    if abs(greedy_wer * 100 - expected_wer) < 0.5:
        print(f"  [PASS] Greedy WER {greedy_wer*100:.4f}% ~ {expected_wer}% (model card)")
    else:
        print(f"  [WARN] Greedy WER {greedy_wer*100:.4f}% differs from expected "
              f"{expected_wer}% (>0.5pp)  --  check model config")
    if oracle_wer < greedy_wer:
        print(f"  [PASS] Oracle ({oracle_wer*100:.4f}%) < Greedy ({greedy_wer*100:.4f}%)")
    else:
        print(f"  [FAIL] Oracle should be < Greedy")
    print(f"  [INFO] Utterance count: {n_utts}")
    if boot_self["delta"] == 0 and boot_self["p_value"] > 0.99:
        print(f"  [PASS] greedy vs greedy: delta=0, p={boot_self['p_value']:.4f}")
    else:
        print(f"  [WARN] greedy-self bootstrap unexpected: "
              f"delta={boot_self['delta']:.6f}, p={boot_self['p_value']:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("\n--- Writing outputs ---")

    # 1. Results JSON
    results = {
        "experiment": "E16_zipformer_m",
        "model_config": config["name"],
        "model_dir": str(args.model_dir),
        "n_utterances": n_utts,
        "G": G,
        "wers": {k: float(v) for k, v in wers.items()},
        "oracle_gap_pp": (greedy_wer - oracle_wer) * 100,
        "oracle_gap_relative": (greedy_wer - oracle_wer) / greedy_wer * 100,
        "bootstrap": bootstrap,
        "spearman": {
            "ctc": rho_ctc,
            "roberta_pll": rho_pll,
        },
        "n_bootstrap": args.n_bootstrap,
    }
    p = args.output_dir / "zipformer_m_results.json"
    with open(p, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Wrote {p}")

    # 2. Bootstrap CSV
    p = args.output_dir / "zipformer_m_bootstrap.csv"
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

    # 3. Spearman JSON
    p = args.output_dir / "zipformer_m_spearman.json"
    with open(p, "w") as f:
        json.dump({"ctc": rho_ctc, "roberta_pll": rho_pll,
                   "n_utterances": n_utts}, f, indent=2)
    print(f"  Wrote {p}")

    # 4. Cross-model comparison MD
    write_cross_model_comparison(args, config, wers, bootstrap, rho_ctc, rho_pll, n_utts)

    # 5. Stage report
    write_report(args, config, wers, bootstrap, rho_ctc, rho_pll, n_utts)

def write_cross_model_comparison(args, config, wers, bootstrap, rho_ctc, rho_pll, n_utts):
    """Side-by-side with Zipformer-S baseline."""
    p = args.output_dir / "cross_model_comparison.md"
    greedy = wers["greedy"] * 100
    oracle = wers["oracle"] * 100
    mbr10 = wers["mbr_cer_pll_tau10"] * 100
    interp = wers["roberta_interp_a0.7"] * 100
    gap_rel = (greedy - oracle) / greedy * 100
    gap_closed = (greedy - mbr10) / max(greedy - oracle, 0.01) * 100
    p_mbr = bootstrap["mbr_cer_pll_tau10"]["p_value"]

    lines = ["# Cross-Model Comparison: Zipformer-S vs " + config["name"], ""]
    lines.append(f"Both models evaluated on dev-other, G=16, same N-best/scoring pipeline.")
    lines.append("")
    lines.append("| Metric | Zipformer-S (22M) | " + config["name"] + " |")
    lines.append("|--------|:-----------------:|:-----------------:|")
    lines.append(f"| Greedy WER | 6.02% | {greedy:.2f}% |")
    lines.append(f"| Oracle G=16 | 4.44% | {oracle:.2f}% |")
    lines.append(f"| Oracle gap (relative) | 26.2% | {gap_rel:.1f}% |")
    lines.append(f"| RoBERTa interp alpha=0.7 | 5.92% | {interp:.2f}% |")
    lines.append(f"| MBR+PLL tau=10 WER | 5.79% | {mbr10:.2f}% |")
    lines.append(f"| MBR+PLL tau=10 p-value | <0.0001 | {p_mbr:.4f} |")
    lines.append(f"| Gap closed (MBR+PLL) | 14.7% | {gap_closed:.1f}% |")
    lines.append(f"| CTC Spearman rho | -0.347 | {rho_ctc:.3f} |")
    lines.append(f"| PLL Spearman rho | -0.484 | {rho_pll:.3f} |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if mbr10 < greedy and p_mbr < 0.05:
        lines.append("**Generalization confirmed.** MBR-CER + RoBERTa PLL tau=10 produces "
                     f"a {(greedy-mbr10):.2f}pp WER reduction (p={p_mbr:.4f}) on the larger "
                     "model, replicating the qualitative finding from Zipformer-S. "
                     "The information bottleneck is not a Zipformer-S quirk.")
    elif mbr10 < greedy:
        lines.append(f"MBR+PLL improves WER by {(greedy-mbr10):.2f}pp on the larger model, "
                     f"but the improvement is not significant (p={p_mbr:.4f}). The smaller "
                     "absolute oracle gap on a stronger acoustic model gives less room for "
                     "linguistic re-ranking to help.")
    else:
        lines.append("MBR+PLL does not improve over greedy on the larger model. "
                     "The stronger acoustic model may already exhaust the available signal.")
    lines.append("")
    with open(p, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {p}")

def write_report(args, config, wers, bootstrap, rho_ctc, rho_pll, n_utts):
    p = args.output_dir / "report_E16.md"
    greedy = wers["greedy"] * 100
    oracle = wers["oracle"] * 100
    mbr10 = wers["mbr_cer_pll_tau10"] * 100
    p_mbr = bootstrap["mbr_cer_pll_tau10"]["p_value"]

    lines = ["# Report E16: Cross-Model Verification", ""]
    lines.append(f"**Model:** {config['name']}")
    lines.append(f"**Status:** Complete. {n_utts} utterances, G=16, "
                 f"B={args.n_bootstrap} bootstrap.")
    lines.append("")
    lines.append("## What Ran")
    lines.append("")
    lines.append(f"- Pipeline: discover -> generate -> score -> evaluate")
    lines.append(f"- Config: layers={config['num_encoder_layers']}, "
                 f"dim={config['encoder_dim']}")
    lines.append(f"- Methods: greedy, oracle, RoBERTa interp alpha=0.7, "
                 f"MBR+PLL tau=10, MBR uniform")
    lines.append(f"- Bootstrap: B={args.n_bootstrap}, paired vs greedy")
    lines.append("")
    lines.append("## Key Results")
    lines.append("")
    lines.append("| Method | WER (%) | delta vs greedy (pp) | p-value |")
    lines.append("|--------|--------:|-----------------:|--------:|")
    lines.append(f"| Greedy | {greedy:.4f} | 0 |  --  |")
    for name in ["roberta_interp_a0.7", "mbr_cer_pll_tau10", "mbr_cer_ctc_tau_inf"]:
        b = bootstrap[name]
        lines.append(f"| {name} | {b['wer']*100:.4f} | "
                     f"{b['delta_pp']:+.4f} | {b['p_value']:.4f} |")
    lines.append(f"| Oracle | {oracle:.4f} | "
                 f"{(oracle-greedy):+.4f} |  --  |")
    lines.append("")
    lines.append("## Spearman Correlations")
    lines.append("")
    lines.append(f"- CTC log-prob rho: **{rho_ctc:.4f}** "
                 f"(Zipformer-S: -0.347)")
    lines.append(f"- RoBERTa PLL rho: **{rho_pll:.4f}** "
                 f"(Zipformer-S: -0.484)")
    lines.append("")
    lines.append("## Generalization Verdict")
    lines.append("")
    if mbr10 < greedy and p_mbr < 0.05:
        lines.append(f"**Confirmed.** MBR+PLL tau=10 reduces WER by "
                     f"{(greedy-mbr10):.2f}pp (p={p_mbr:.4f}). The method generalizes "
                     "across model sizes  --  closes the 'single architecture' critique.")
    elif mbr10 < greedy:
        lines.append(f"MBR+PLL improves by {(greedy-mbr10):.2f}pp but "
                     f"p={p_mbr:.4f} is not below 0.05. The stronger model has less "
                     "absolute oracle gap, leaving less room for LM re-ranking.")
    else:
        lines.append("MBR+PLL does not help on this larger model. The acoustic "
                     "model may have saturated the available signal.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("| File | Purpose |")
    lines.append("|------|---------|")
    lines.append("| `zipformer_m_results.json` | Full results |")
    lines.append("| `zipformer_m_bootstrap.csv` | Bootstrap p-values |")
    lines.append("| `zipformer_m_spearman.json` | Per-utt Spearman rho |")
    lines.append("| `cross_model_comparison.md` | Side-by-side with Zipformer-S |")
    lines.append("| `report_E16.md` | This stage report |")

    with open(p, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {p}")

def parse_args():
    parser = argparse.ArgumentParser(description="E16: Cross-model verification")
    parser.add_argument("--data-dir", type=Path,
                        default=Path("/content/drive/MyDrive/rbpo_results"))
    parser.add_argument("--librispeech-dir", type=Path,
                        default=Path("/content/librispeech_data"))
    parser.add_argument("--model-dir", type=Path,
                        default=Path("/content/icefall-asr-librispeech-zipformer-medium-cr-ctc"))
    parser.add_argument("--icefall-dir", type=Path,
                        default=Path("/content/icefall"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/content/drive/MyDrive/rbpo_results/zipformer_m"))
    parser.add_argument("--model-config", type=str, default="zipformer-m-cr-ctc",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model architecture preset")
    parser.add_argument("--num-encoder-layers", type=str, default=None,
                        help="Override config preset")
    parser.add_argument("--encoder-dim", type=str, default=None)
    parser.add_argument("--encoder-unmasked-dim", type=str, default=None)
    parser.add_argument("--feedforward-dim", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--steps", type=str, default="all",
                        help="Comma-separated: discover,generate,score,evaluate or 'all'")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = dict(MODEL_CONFIGS[args.model_config])
    if args.num_encoder_layers:
        config["num_encoder_layers"] = args.num_encoder_layers
    if args.encoder_dim:
        config["encoder_dim"] = args.encoder_dim
    if args.encoder_unmasked_dim:
        config["encoder_unmasked_dim"] = args.encoder_unmasked_dim
    if args.feedforward_dim:
        config["feedforward_dim"] = args.feedforward_dim

    steps = (args.steps.lower().split(",") if args.steps != "all"
             else ["discover", "generate", "score", "evaluate"])

    print("=" * 70)
    print(f"E16: Cross-Model Verification  --  {config['name']}")
    print("=" * 70)
    print(f"  Model dir:   {args.model_dir}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"  Steps:       {steps}")
    print(f"  Bootstrap:   B={args.n_bootstrap}")

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

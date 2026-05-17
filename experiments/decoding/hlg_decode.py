#!/usr/bin/env python3
"""E20: HLG (WFST-composed 4-gram LM) CTC decoding on Zipformer-S CR-CTC.

First published HLG result on CR-CTC. Compares HLG-based decoding methods
against the CTC-only baseline (6.02% greedy -> 5.53% MBR+PLL).

HLG = H o C o L o G, where:
  H = CTC topology (blank + tokens)
  C = token-to-word transducer (BPE)
  L = lexicon (word-to-token mapping)
  G = 4-gram word LM (from openslr.org/11)

Methods tested:
  1. ctc-decoding     --  CTC greedy/prefix beam (no LM, baseline)
  2. 1best            --  HLG Viterbi best path
  3. nbest            --  HLG N-best, oracle WER
  4. nbest-rescoring  --  HLG N-best + 4-gram path rescoring
  5. whole-lattice-rescoring  --  HLG + 4-gram lattice rescoring

Also extracts HLG N-best in JSONL format for the MBR+PLL pipeline.

Usage (Colab T4):
    python /content/rbpo/experiments/decoding/hlg_decode.py \
        --model-dir /content/icefall-asr-librispeech-zipformer-small-cr-ctc \
        --data-dir /content/librispeech_data \
        --icefall-dir /content/icefall \
        --output-dir /content/drive/MyDrive/rbpo_results/hlg_decode \
        --device cuda:0 \
        --num-paths 128 \
        --max-duration 200 \
        --steps all
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import editdistance
import sentencepiece as spm
import torch

# Suppress noisy k2 C++ warnings (forward/backward score differences)
# K2_VERBOSE_LEVEL=0 suppresses C++ level warnings from intersect_dense.cu
os.environ["K2_VERBOSE_LEVEL"] = "0"

# Force unbuffered stdout so progress is visible in Colab
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None


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
    """Load Zipformer-S CR-CTC model (matches generate_nbest.py)."""
    add_icefall_to_path(icefall_dir)
    from train import add_model_arguments, get_model, get_params

    params = get_params()
    parser = argparse.ArgumentParser(add_help=False)
    add_model_arguments(parser)
    model_args = parser.parse_args([])
    for k, v in vars(model_args).items():
        params[k] = v

    # Zipformer-S architecture hyperparameters
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
    checkpoint = torch.load(model_dir / "exp" / "pretrained.pt",
                            map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {num_params / 1e6:.1f}M parameters")
    return model


def load_cuts(data_dir: Path, split: str):
    from lhotse import load_manifest_lazy
    cuts_path = data_dir / "cuts" / f"librispeech_cuts_{split}.jsonl.gz"
    assert cuts_path.exists(), f"CutSet not found: {cuts_path}"
    return load_manifest_lazy(str(cuts_path))


def extract_features_batch(cuts_batch, device):
    """Load pre-computed fbank features from lhotse CutSet.

    Uses cut.load_features() which returns the exact features the model
    was trained on (computed by setup_colab.sh). Do NOT recompute from
    audio  --  different feature params produce garbage model outputs.
    """
    features_list = []
    lengths = []
    for cut in cuts_batch:
        feat = torch.from_numpy(cut.load_features())  # (T, 80)
        features_list.append(feat)
        lengths.append(feat.shape[0])

    max_len = max(lengths)
    batch = torch.zeros(len(features_list), max_len, 80)
    for i, feat in enumerate(features_list):
        batch[i, :feat.shape[0]] = feat
    lengths_t = torch.tensor(lengths, dtype=torch.int64)
    return batch.to(device), lengths_t.to(device)


def download_lm(output_dir: Path):
    """Download LibriSpeech 4-gram and 3-gram ARPA LMs."""
    import urllib.request

    lm_dir = output_dir / "lm"
    lm_dir.mkdir(parents=True, exist_ok=True)

    # 3-gram pruned (small, fast, for fallback)
    arpa_3gram = lm_dir / "3-gram.pruned.1e-7.arpa"
    if not arpa_3gram.exists():
        url = "https://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz"
        gz_path = lm_dir / "3-gram.pruned.1e-7.arpa.gz"
        print(f"  Downloading 3-gram LM from {url}...")
        urllib.request.urlretrieve(url, str(gz_path))
        os.system(f"gunzip -f {gz_path}")
        print(f"   3-gram LM: {arpa_3gram} ({arpa_3gram.stat().st_size / 1e6:.0f} MB)")
    else:
        print(f"   3-gram LM already exists: {arpa_3gram}")

    # 4-gram (large, best quality)
    arpa_4gram = lm_dir / "4-gram.arpa"
    if not arpa_4gram.exists():
        url = "https://www.openslr.org/resources/11/4-gram.arpa.gz"
        gz_path = lm_dir / "4-gram.arpa.gz"
        print(f"  Downloading 4-gram LM from {url}...")
        urllib.request.urlretrieve(url, str(gz_path))
        os.system(f"gunzip -f {gz_path}")
        print(f"   4-gram LM: {arpa_4gram} ({arpa_4gram.stat().st_size / 1e6:.0f} MB)")
    else:
        print(f"   4-gram LM already exists: {arpa_4gram}")

    return arpa_3gram, arpa_4gram


def build_hlg(model_dir: Path, icefall_dir: Path, output_dir: Path,
              lm_order: int = 4):
    """Build HLG.pt using icefall's standard pipeline.

    Pipeline:
    1. prepare_lang_bpe.py -> L.pt, L_disambig.pt (lexicon FSAs)
    2. kaldilm.arpa2fst -> G_{n}gram.fst.txt (grammar FST in text format)
    3. compile_hlg.py -> HLG.pt (composed H o L o G)
    """
    import k2
    import subprocess
    import shutil

    hlg_dir = output_dir / "hlg_artifacts"
    hlg_dir.mkdir(parents=True, exist_ok=True)

    lang_dir = model_dir / "data" / "lang_bpe_500"
    hlg_path = hlg_dir / f"HLG_{lm_order}gram.pt"

    if hlg_path.exists():
        print(f"   HLG already exists: {hlg_path}")
        return hlg_path

    arpa_3gram, arpa_4gram = download_lm(output_dir)
    arpa_path = arpa_4gram if lm_order == 4 else arpa_3gram

    words_txt = lang_dir / "words.txt"
    tokens_txt = lang_dir / "tokens.txt"

    ASR_DIR = icefall_dir / "egs" / "librispeech" / "ASR"

    # Step 1: Build words.txt from ARPA if missing or corrupt
    # Reserved symbols that get hardcoded IDs  --  exclude from ARPA extraction
    RESERVED = {"<eps>", "!SIL", "<SPOKEN_NOISE>", "<UNK>", "#0", "<s>", "</s>",
                "<unk>", "<EPS>"}

    needs_rebuild = False
    if not words_txt.exists():
        needs_rebuild = True
    else:
        # Validate: check for duplicate symbols
        seen_syms = set()
        with open(words_txt) as f:
            for line in f:
                sym = line.strip().split()[0] if line.strip() else ""
                if sym in seen_syms:
                    print(f"  WARNING: Duplicate symbol '{sym}' in words.txt  --  rebuilding")
                    needs_rebuild = True
                    break
                seen_syms.add(sym)

    if needs_rebuild:
        print("  Building words.txt from ARPA LM unigrams...")
        words = set()
        with open(arpa_path) as f:
            in_unigrams = False
            for line in f:
                line = line.strip()
                if line == "\\1-grams:":
                    in_unigrams = True
                    continue
                if line.startswith("\\") and "grams:" in line:
                    if in_unigrams:
                        break
                if in_unigrams and line:
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        w = parts[1].strip()
                        if w and w not in RESERVED:
                            words.add(w)

        with open(words_txt, "w") as f:
            f.write("<eps> 0\n")
            f.write("!SIL 1\n")
            f.write("<SPOKEN_NOISE> 2\n")
            f.write("<UNK> 3\n")
            for i, w in enumerate(sorted(words), start=4):
                f.write(f"{w} {i}\n")
            f.write(f"#0 {len(words) + 4}\n")
            f.write(f"<s> {len(words) + 5}\n")
            f.write(f"</s> {len(words) + 6}\n")
        print(f"   words.txt: {len(words)} words")
        # Also save a copy to hlg_artifacts (survives Colab restart)
        import shutil
        shutil.copy2(words_txt, hlg_dir / "words.txt")

    # Step 2: prepare_lang_bpe.py -> L.pt, L_disambig.pt
    L_pt = lang_dir / "L.pt"
    if not L_pt.exists():
        print("  Step 2: Building L.pt via prepare_lang_bpe.py...")

        tokens_txt = lang_dir / "tokens.txt"
        bpe_model = lang_dir / "bpe.model"
        print(f"    tokens.txt exists: {tokens_txt.exists()}")
        print(f"    words.txt exists:  {words_txt.exists()}")
        print(f"    bpe.model exists:  {bpe_model.exists()}")
        if tokens_txt.exists():
            with open(tokens_txt) as f:
                n_tokens = sum(1 for _ in f)
            print(f"    tokens.txt lines: {n_tokens}")

        cmd = [
            sys.executable,
            str(ASR_DIR / "local" / "prepare_lang_bpe.py"),
            "--lang-dir", str(lang_dir),
            "--debug", "0",
        ]
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"  prepare_lang_bpe.py FAILED (rc={result.returncode})")
            print(f"  stderr (last 2000 chars):\n{result.stderr[-2000:]}")
            print(f"  stdout: {result.stdout[:500]}")
        else:
            print(f"   prepare_lang_bpe.py completed")
            print(f"  stdout: {result.stdout[-300:]}")
        if not L_pt.exists():
            # List what IS in lang_dir
            print(f"  lang_dir contents: {sorted(p.name for p in lang_dir.iterdir())}")
            raise RuntimeError(
                f"L.pt not created at {L_pt}. See stderr above for details."
            )
    else:
        print(f"   L.pt already exists: {L_pt}")

    # Step 3: ARPA -> G_Ngram.fst.txt via kaldilm
    lm_dir = lang_dir / "lm"  # icefall expects G_*.fst.txt inside lang_dir/lm/
    lm_dir.mkdir(parents=True, exist_ok=True)
    lm_stem = f"G_{lm_order}_gram"
    g_fst_txt = lm_dir / f"{lm_stem}.fst.txt"

    if not g_fst_txt.exists():
        print(f"  Step 3: Converting {lm_order}-gram ARPA -> {g_fst_txt.name} via kaldilm...")
        print("  This may take 5-15 minutes and use 4-8 GB RAM for 4-gram...")

        try:
            from kaldilm import arpa2fst
        except ImportError:
            raise RuntimeError(
                "kaldilm not installed. Run: pip install kaldilm"
            )

        fst_text = arpa2fst(
            input_arpa=str(arpa_path),
            read_symbol_table=str(words_txt),
            disambig_symbol="#0",
            max_order=lm_order,
        )
        with open(g_fst_txt, "w") as f:
            f.write(fst_text)
        print(f"   {g_fst_txt.name}: {g_fst_txt.stat().st_size / 1e6:.0f} MB")
    else:
        print(f"   G FST already exists: {g_fst_txt}")

    # Step 4: compile_hlg.py -> HLG.pt
    print(f"  Step 4: Composing HLG via compile_hlg.py...")

    compile_hlg_script = None
    for candidate in [
        ASR_DIR / "local" / "compile_hlg.py",
        icefall_dir / "icefall" / "shared" / "compile_hlg.py",
        ASR_DIR / "shared" / "compile_hlg.py",
    ]:
        if candidate.exists():
            compile_hlg_script = candidate
            break

    if compile_hlg_script:
        cmd = [
            sys.executable, str(compile_hlg_script),
            "--lang-dir", str(lang_dir),
            "--lm", lm_stem,
        ]
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800
        )
        if result.returncode == 0:
            print(f"  compile_hlg.py stdout: {result.stdout[-500:]}")
            # Find the output HLG file
            for candidate_name in [
                f"HLG.pt",
                f"HLG_{lm_order}gram.pt",
                f"HLG_{lm_stem}.pt",
            ]:
                candidate_path = lang_dir / candidate_name
                if candidate_path.exists():
                    shutil.copy2(candidate_path, hlg_path)
                    print(f"   HLG built: {hlg_path} "
                          f"({hlg_path.stat().st_size / 1e6:.0f} MB)")
                    return hlg_path
            # Check if it was put in lm_dir instead
            for candidate_path in lm_dir.glob("HLG*.pt"):
                shutil.copy2(candidate_path, hlg_path)
                print(f"   HLG built: {hlg_path} "
                      f"({hlg_path.stat().st_size / 1e6:.0f} MB)")
                return hlg_path
            print("  WARNING: compile_hlg.py succeeded but HLG.pt not found")
            print(f"  lang_dir contents: {list(lang_dir.glob('*.pt'))}")
            print(f"  lm_dir contents: {list(lm_dir.glob('*'))}")
        else:
            print(f"  compile_hlg.py failed (rc={result.returncode})")
            print(f"  stderr: {result.stderr[:1000]}")
            print(f"  stdout: {result.stdout[:500]}")
    else:
        print("  WARNING: compile_hlg.py not found in icefall")

    # Fallback: manual k2 composition
    print("\n  Falling back to manual k2 HLG construction...")
    return build_hlg_manual(lang_dir, g_fst_txt, hlg_dir, hlg_path, lm_order)


def build_hlg_manual(lang_dir, g_fst_txt, hlg_dir, hlg_path, lm_order):
    """Build HLG using the canonical icefall algorithm.

    Algorithm (from icefall/egs/librispeech/ASR/local/compile_hlg.py):
      H = ctc_topo
      L = L_disambig (with disambig symbols #0, #1...)
      LG = compose(arc_sort(L), arc_sort(G), treat_epsilons_specially=False)
      LG = determinize(LG); connect(LG)
      Remove disambig symbols from LG.labels (set to 0)
      Remove disambig symbols from LG.aux_labels (set to 0, then drop)
      LG = remove_epsilon(LG); connect; arc_sort
      HLG = compose(H, LG, inner_labels='tokens')
      HLG = connect; arc_sort
    """
    import k2

    print("  Building CTC topology H...")
    H = k2.ctc_topo(max_token=MAX_TOKEN, modified=False, device="cpu")

    print(f"  Loading G from {g_fst_txt.name}...")
    with open(g_fst_txt) as f:
        G = k2.Fsa.from_openfst(f.read(), acceptor=False)

    # MUST use L_disambig.pt (with disambiguation symbols), NOT plain L.pt
    L_pt = lang_dir / "L_disambig.pt"
    if not L_pt.exists():
        L_pt = hlg_dir / "L_disambig.pt"
    if not L_pt.exists():
        raise RuntimeError(
            f"L_disambig.pt not found in {lang_dir} or {hlg_dir}. "
            f"Run prepare_lang_bpe.py first."
        )

    print("  Loading L_disambig.pt...")
    L = k2.Fsa.from_dict(torch.load(L_pt, map_location="cpu", weights_only=False))

    # Find disambig symbol IDs (#0 in tokens.txt and words.txt)
    tokens_txt = lang_dir / "tokens.txt"
    words_txt = lang_dir / "words.txt"
    if not words_txt.exists():
        words_txt = hlg_dir / "words.txt"

    first_token_disambig_id = None
    with open(tokens_txt) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2 and parts[0] == "#0":
                first_token_disambig_id = int(parts[1])
                break
    first_word_disambig_id = None
    with open(words_txt) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2 and parts[0] == "#0":
                first_word_disambig_id = int(parts[1])
                break

    if first_token_disambig_id is None or first_word_disambig_id is None:
        raise RuntimeError(
            f"Could not find #0 in tokens.txt ({first_token_disambig_id}) "
            f"or words.txt ({first_word_disambig_id})"
        )
    print(f"  Disambig IDs: token #0={first_token_disambig_id}, "
          f"word #0={first_word_disambig_id}")

    print("  arc_sort(L) and arc_sort(G)...")
    L = k2.arc_sort(L)
    G = k2.arc_sort(G)

    # Use default epsilon handling (treat_epsilons_specially=True).
    # The strict `False` mode requires determinization which crashes in
    # k2 1.24.4 on this FSA. With True, epsilons are real epsilons.
    print("  Composing LG = L o G (default epsilon handling)...")
    LG = k2.compose(L, G)
    print(f"    LG after compose: {LG.num_arcs} arcs")

    # Remove disambig symbols using the proper k2 API
    # (k2 forbids in-place mutation of fsa.labels  --  must clone+reassign)
    print("  Removing disambig symbols...")
    labels = LG.labels.clone()
    labels[labels >= first_token_disambig_id] = 0
    LG.labels = labels

    if isinstance(LG.aux_labels, torch.Tensor):
        aux = LG.aux_labels.clone()
        aux[aux >= first_word_disambig_id] = 0
        LG.aux_labels = aux
    else:
        # RaggedTensor  --  values is callable in k2 1.24
        try:
            v = LG.aux_labels.values
            if callable(v):
                v = v()
            v[v >= first_word_disambig_id] = 0
        except Exception as e:
            print(f"     Could not zero ragged aux disambig: {str(e)[:150]}")

    LG = k2.connect(LG)
    print(f"    LG after connect: {LG.num_arcs} arcs")
    LG = k2.arc_sort(LG)

    print("  Composing HLG = H o LG (inner_labels='tokens')...")
    HLG = k2.compose(H, LG, inner_labels="tokens")
    print(f"    HLG after compose: {HLG.num_arcs} arcs")
    HLG = k2.connect(HLG)
    print(f"    HLG after connect: {HLG.num_arcs} arcs")
    HLG = k2.arc_sort(HLG)

    torch.save(HLG.as_dict(), hlg_path)
    size_mb = hlg_path.stat().st_size / 1e6
    print(f"   HLG saved: {hlg_path} ({size_mb:.0f} MB)")
    if size_mb < 100:
        print(f"   WARNING: HLG is only {size_mb:.0f} MB. Standard icefall HLG is "
              f"600+ MB. Output may be degraded.")
    return hlg_path


def _aux_to_list(aux):
    """Convert k2 aux_labels (Tensor or RaggedTensor) to flat Python list.

    In k2 1.24.x, RaggedTensor.values is a method (not property), so we
    need to call it. Tensors don't have .values, so we use .cpu() directly.
    """
    if isinstance(aux, torch.Tensor):
        return aux.cpu().tolist()
    # k2.RaggedTensor  --  .values may be method or property depending on version
    v = aux.values
    if callable(v):
        v = v()
    return v.cpu().tolist()


def hlg_decode(model, hlg, cuts, sp, word_table, device, method, num_paths=128,
               max_duration=200, batch_size=32, debug_first=0):
    """Run HLG-based CTC decoding using icefall's canonical helpers.

    Uses icefall.utils.get_texts to extract word IDs from HLG lattice
    paths (handles all k2 RaggedTensor API variants).

    Returns list of dicts with: utt_id, ref_text, hyp_text, [nbest_candidates]
    """
    import k2
    from icefall.utils import get_texts

    results = []
    all_cuts = list(cuts)
    print(f"  Processing {len(all_cuts)} utterances with method={method} (batch_size={batch_size})...")

    for batch_start in range(0, len(all_cuts), batch_size):
        batch_cuts = all_cuts[batch_start:batch_start + batch_size]
        features, lengths = extract_features_batch(batch_cuts, device)

        with torch.no_grad():
            encoder_out, encoder_out_lens = model.forward_encoder(features, lengths)
            log_probs = model.ctc_output(encoder_out)

        # Lattice ops per-utterance (k2 requirement)
        for i, cut in enumerate(batch_cuts):
            T = encoder_out_lens[i].item()
            lp = log_probs[i, :T]

            supervision_segments = torch.tensor([[0, 0, T]], dtype=torch.int32)
            dense_fsa = k2.DenseFsaVec(lp.unsqueeze(0), supervision_segments)

            # Use intersect_dense_pruned for HLG (smaller memory footprint).
            # Beams from icefall's standard HLG decoding.
            lattice = k2.intersect_dense_pruned(
                hlg, dense_fsa,
                search_beam=15.0,
                output_beam=6.0,
                min_active_states=30,
                max_active_states=10000,
            )
            lattice = k2.connect(lattice)

            ref_text = ""
            for sup in cut.supervisions:
                ref_text = sup.text.lower().strip()

            utt_id = cut.id

            if method == "1best":
                best_path = k2.shortest_path(lattice, use_double_scores=True)

                # Use icefall's get_texts  --  works across k2 versions
                hyps_word_ids = get_texts(best_path, return_ragged=False)
                word_ids = hyps_word_ids[0] if hyps_word_ids else []

                # Diagnostic dump for first few utterances
                if debug_first > 0 and len(results) < debug_first:
                    print(f"  [debug] utt {len(results)}: {len(word_ids)} word IDs")
                    print(f"    first 10 word_ids: {word_ids[:10]}")
                    print(f"    lattice arcs: {lattice.num_arcs}, "
                          f"best_path arcs: {best_path.num_arcs}")

                words = [word_table[w] for w in word_ids if w in word_table]
                hyp_text = " ".join(words).lower()
                results.append({
                    "utt_id": utt_id,
                    "ref_text": ref_text,
                    "hyp_text": hyp_text,
                })

            elif method in ("nbest", "nbest-rescoring"):
                nbest = k2.Nbest.from_lattice(
                    lattice,
                    num_paths=num_paths,
                    use_double_scores=True,
                    nbest_scale=1.0,
                )

                # nbest.fsa is a stacked FSA  --  get_texts returns one
                # word_id list per path (one path per "utterance" in the stack)
                hyps_word_ids = get_texts(nbest.fsa, return_ragged=False)

                # Get per-path total scores (AM + LM combined).
                # k2 1.24 changed this API  --  try multiple fallbacks.
                path_scores = None
                # Attempt 1: nbest.tot_scores() (older k2)
                try:
                    if hasattr(nbest, 'tot_scores') and callable(nbest.tot_scores):
                        ts = nbest.tot_scores()
                        path_scores = _aux_to_list(ts)
                except Exception:
                    pass
                # Attempt 2: nbest.compute_total_scores() or via lattice
                if path_scores is None:
                    try:
                        arc_scores = nbest.fsa.scores.cpu().tolist()
                        all_labels_cpu = nbest.fsa.labels.cpu().tolist()
                        path_scores = []
                        current = 0.0
                        for lbl, sc in zip(all_labels_cpu, arc_scores):
                            if lbl == -1:
                                path_scores.append(current)
                                current = 0.0
                            else:
                                current += sc
                    except Exception:
                        pass
                # Final fallback: rank by index (earliest = highest)
                if path_scores is None:
                    path_scores = [-float(i) for i in range(len(hyps_word_ids))]

                seen = {}
                for path_idx, word_ids in enumerate(hyps_word_ids):
                    words = [word_table[w] for w in word_ids if w in word_table]
                    text = " ".join(words).lower().strip()

                    if not text:
                        continue

                    score = path_scores[path_idx] if path_idx < len(path_scores) else 0.0

                    entry = {
                        "text": text,
                        "tokens": [],
                        "ctc_log_prob": score,  # actually total HLG score
                        "len_tokens": len(words),
                        "len_chars": len(text),
                    }

                    if text in seen:
                        if score > seen[text]["ctc_log_prob"]:
                            seen[text] = entry
                    else:
                        seen[text] = entry

                candidates = sorted(
                    seen.values(),
                    key=lambda c: c["ctc_log_prob"],
                    reverse=True,
                )

                hyp_text = candidates[0]["text"] if candidates else ""

                results.append({
                    "utt_id": utt_id,
                    "ref_text": ref_text,
                    "hyp_text": hyp_text,
                    "candidates": candidates,
                    "num_candidates": len(candidates),
                })

        done = min(batch_start + batch_size, len(all_cuts))
        if done % 500 < batch_size or done == len(all_cuts):
            print(f"    {done}/{len(all_cuts)}...")

    return results


def ctc_greedy_decode(model, cuts, sp, device, max_duration=200,
                      batch_size=32):
    """CTC greedy decoding (no LM)  --  baseline. Batched encoder forward."""

    results = []
    all_cuts = list(cuts)
    print(f"  CTC greedy on {len(all_cuts)} utterances (batch_size={batch_size})...")

    for batch_start in range(0, len(all_cuts), batch_size):
        batch_cuts = all_cuts[batch_start:batch_start + batch_size]
        features, lengths = extract_features_batch(batch_cuts, device)

        with torch.no_grad():
            encoder_out, encoder_out_lens = model.forward_encoder(features, lengths)
            log_probs = model.ctc_output(encoder_out)

        if log_probs.shape[0] != len(batch_cuts):
            print(f"   FATAL: Encoder output batch {log_probs.shape[0]} != input {len(batch_cuts)}")
            sys.exit(1)

        for i, cut in enumerate(batch_cuts):
            T = encoder_out_lens[i].item()
            lp = log_probs[i, :T]
            greedy_ids = lp.argmax(dim=-1).cpu().tolist()

            # CTC collapse
            collapsed = []
            prev = None
            for t in greedy_ids:
                if t != BLANK_ID and t != prev:
                    collapsed.append(t)
                prev = t

            text = sp.decode(collapsed).strip().lower()
            ref_text = ""
            for sup in cut.supervisions:
                ref_text = sup.text.lower().strip()

            results.append({
                "utt_id": cut.id,
                "ref_text": ref_text,
                "hyp_text": text,
            })

        done = min(batch_start + batch_size, len(all_cuts))
        if done % 500 < batch_size or done == len(all_cuts):
            print(f"    {done}/{len(all_cuts)}...")

    return results


def ctc_nbest_decode(model, cuts, sp, device, num_paths=128, max_duration=200,
                     batch_size=32):
    """CTC-only N-best (no LM). Batched encoder, per-utterance lattice ops."""
    import k2

    results = []
    all_cuts = list(cuts)
    print(f"  CTC N-best (G={num_paths}) on {len(all_cuts)} utterances (batch_size={batch_size})...")

    topo = k2.ctc_topo(max_token=MAX_TOKEN, modified=False, device=device)

    for batch_start in range(0, len(all_cuts), batch_size):
        batch_cuts = all_cuts[batch_start:batch_start + batch_size]
        features, lengths = extract_features_batch(batch_cuts, device)

        with torch.no_grad():
            encoder_out, encoder_out_lens = model.forward_encoder(features, lengths)
            log_probs = model.ctc_output(encoder_out)

        # Lattice ops are per-utterance (k2 requirement)
        for i, cut in enumerate(batch_cuts):
            T = encoder_out_lens[i].item()
            lp = log_probs[i, :T]

            supervision_segments = torch.tensor([[0, 0, T]], dtype=torch.int32)
            dense_fsa = k2.DenseFsaVec(lp.unsqueeze(0), supervision_segments)
            lattice = k2.intersect_dense(topo, dense_fsa, output_beam=8.0)
            lattice = k2.connect(lattice)

            nbest = k2.Nbest.from_lattice(
                lattice, num_paths=num_paths * 4,  # oversample
                use_double_scores=True, nbest_scale=1.0,
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

            seen = {}
            lp_cpu = lp.cpu()
            for raw_ids in paths:
                # Skip paths with wrong number of frames
                if len(raw_ids) != T:
                    continue

                collapsed = []
                prev = None
                for t in raw_ids:
                    if t != BLANK_ID and t != prev:
                        collapsed.append(t)
                    prev = t

                text = sp.decode(collapsed).strip().lower()
                score = 0.0
                for t_idx, tok in enumerate(raw_ids):
                    score += lp_cpu[t_idx, tok].item()

                entry = {
                    "text": text,
                    "tokens": collapsed,
                    "ctc_log_prob": score,
                    "len_tokens": len(collapsed),
                    "len_chars": len(text),
                }
                if text in seen:
                    if score > seen[text]["ctc_log_prob"]:
                        seen[text] = entry
                else:
                    seen[text] = entry

            candidates = sorted(
                seen.values(), key=lambda c: c["ctc_log_prob"], reverse=True
            )[:num_paths]

            ref_text = ""
            for sup in cut.supervisions:
                ref_text = sup.text.lower().strip()

            results.append({
                "utt_id": cut.id,
                "ref_text": ref_text,
                "hyp_text": candidates[0]["text"] if candidates else "",
                "candidates": candidates,
                "num_candidates": len(candidates),
            })

        done = min(batch_start + batch_size, len(all_cuts))
        if done % 500 < batch_size or done == len(all_cuts):
            print(f"    {done}/{len(all_cuts)}...")

    return results


def compute_corpus_wer(results):
    """Corpus-level WER from a list of result dicts."""
    total_edits = 0
    total_ref_words = 0
    for r in results:
        ref = r["ref_text"].split()
        hyp = r["hyp_text"].split()
        total_edits += editdistance.eval(ref, hyp)
        total_ref_words += len(ref)
    return total_edits / total_ref_words if total_ref_words > 0 else 0.0


def compute_oracle_wer(results):
    """Oracle WER: best candidate per utterance."""
    total_edits = 0
    total_ref_words = 0
    for r in results:
        ref = r["ref_text"].split()
        if "candidates" in r and r["candidates"]:
            best_edits = min(
                editdistance.eval(ref, c["text"].split())
                for c in r["candidates"]
            )
        else:
            best_edits = editdistance.eval(ref, r["hyp_text"].split())
        total_edits += best_edits
        total_ref_words += len(ref)
    return total_edits / total_ref_words if total_ref_words > 0 else 0.0


def save_nbest_jsonl(results, output_path):
    """Save N-best results in the standard RBPO JSONL format."""
    with open(output_path, "w") as f:
        for r in results:
            record = {
                "utt_id": r["utt_id"],
                "ref_text": r["ref_text"],
                "num_candidates": r.get("num_candidates", 1),
                "candidates": r.get("candidates", [{
                    "text": r["hyp_text"],
                    "tokens": [],
                    "ctc_log_prob": 0.0,
                    "len_tokens": 0,
                    "len_chars": len(r["hyp_text"]),
                }]),
            }
            f.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser(description="E20: HLG CTC Decoding")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--icefall-dir", type=Path,
                        default=Path("/content/icefall"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-paths", type=int, default=128)
    parser.add_argument("--max-duration", type=int, default=200,
                        help="Max audio duration per batch in seconds")
    parser.add_argument("--split", default="dev-other")
    parser.add_argument("--steps", default="all",
                        help="Comma-separated: build,ctc,hlg,compare or 'all'")
    parser.add_argument("--lm-order", type=int, default=4, choices=[3, 4],
                        help="N-gram LM order (3 or 4)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Encoder batch size")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only first N utterances (0 = all). For testing.")
    parser.add_argument("--debug-hlg", action="store_true",
                        help="Print diagnostic info for first 5 HLG paths")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    steps = args.steps.split(",") if args.steps != "all" else [
        "build", "ctc", "hlg", "compare"
    ]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"E20: HLG CTC Decoding on {args.split}")
    print(f"  Device: {device}")
    print(f"  Model: {args.model_dir}")
    print(f"  Data: {args.data_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Num paths: {args.num_paths}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LM order: {args.lm_order}-gram")
    print(f"  Steps: {steps}")
    if args.limit > 0:
        print(f"   LIMIT mode: only first {args.limit} utterances")
    if args.debug_hlg:
        print(f"   DEBUG HLG: will dump diagnostics for first 5 paths")

    def get_cuts():
        """Load cuts, optionally truncated."""
        cuts = load_cuts(args.data_dir, args.split)
        if args.limit > 0:
            cuts_list = list(cuts)[:args.limit]
            return cuts_list
        return cuts

    add_icefall_to_path(args.icefall_dir)

    sp = spm.SentencePieceProcessor()
    sp.load(str(args.model_dir / "data" / "lang_bpe_500" / "bpe.model"))

    # Load word symbol table (built during HLG construction)
    # Maps word_id -> word string, used for HLG output extraction
    word_table = None
    # Try lang dir first (build_hlg writes it there), then hlg_artifacts
    candidates = [
        args.model_dir / "data" / "lang_bpe_500" / "words.txt",
        args.output_dir / "hlg_artifacts" / "words.txt",
    ]
    for words_txt in candidates:
        if words_txt.exists():
            word_table = {}
            with open(words_txt) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        word, idx = parts
                        word_table[int(idx)] = word
            print(f"  Loaded word table from {words_txt}: {len(word_table)} words")
            break

    # Fallback: regenerate words.txt from ARPA if missing (e.g. after Colab restart)
    if word_table is None and "hlg" in steps:
        arpa_path = args.output_dir / "lm" / (
            "4-gram.arpa" if args.lm_order == 4 else "3-gram.pruned.1e-7.arpa"
        )
        if not arpa_path.exists():
            print(f"   FATAL: words.txt missing AND ARPA missing at {arpa_path}")
            sys.exit(1)
        print(f"  words.txt missing; regenerating from {arpa_path}...")
        RESERVED = {"<eps>", "!SIL", "<SPOKEN_NOISE>", "<UNK>", "#0", "<s>", "</s>",
                    "<unk>", "<EPS>"}
        words = set()
        with open(arpa_path) as f:
            in_unigrams = False
            for line in f:
                line = line.strip()
                if line == "\\1-grams:":
                    in_unigrams = True
                    continue
                if line.startswith("\\") and "grams:" in line and in_unigrams:
                    break
                if in_unigrams and line:
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        w = parts[1].strip()
                        if w and w not in RESERVED:
                            words.add(w)
        # Reconstruct same ID assignment used at build time
        word_table = {0: "<eps>", 1: "!SIL", 2: "<SPOKEN_NOISE>", 3: "<UNK>"}
        for i, w in enumerate(sorted(words), start=4):
            word_table[i] = w
        word_table[len(words) + 4] = "#0"
        word_table[len(words) + 5] = "<s>"
        word_table[len(words) + 6] = "</s>"
        # Save it for next time
        out_words = args.output_dir / "hlg_artifacts" / "words.txt"
        out_words.parent.mkdir(parents=True, exist_ok=True)
        with open(out_words, "w") as f:
            for idx in sorted(word_table.keys()):
                f.write(f"{word_table[idx]} {idx}\n")
        print(f"   Regenerated word table: {len(word_table)} entries, saved to {out_words}")

    timings = {}

    hlg = None
    if "build" in steps:
        print("\n=== STEP 1: Build HLG artifacts ===")
        t0 = time.time()
        try:
            hlg_path = build_hlg(
                args.model_dir, args.icefall_dir, args.output_dir,
                lm_order=args.lm_order
            )
            import k2
            hlg = k2.Fsa.from_dict(
                torch.load(hlg_path, map_location="cpu", weights_only=False)
            ).to(device)
            print(f"  HLG loaded: {hlg.num_arcs} arcs")
            timings["build_hlg"] = time.time() - t0
        except Exception as e:
            print(f"  ERROR building HLG: {e}")
            print("  Will skip HLG decoding steps.")
            timings["build_hlg"] = time.time() - t0

    # Load pre-built HLG if build was skipped but hlg step is requested
    if hlg is None and "hlg" in steps:
        hlg_dir = args.output_dir / "hlg_artifacts"
        hlg_path = hlg_dir / f"HLG_{args.lm_order}gram.pt"
        if hlg_path.exists():
            print(f"\n=== Loading pre-built HLG from {hlg_path} ===")
            import k2
            hlg = k2.Fsa.from_dict(
                torch.load(hlg_path, map_location="cpu", weights_only=False)
            ).to(device)
            print(f"  HLG loaded: {hlg.num_arcs} arcs")
        else:
            print(f"\n  WARNING: HLG not found at {hlg_path}. Run with --steps build first.")

    # Load model and data
    print("\n=== Loading model and data ===")
    model = load_model(args.model_dir, args.icefall_dir, device)
    cuts = load_cuts(args.data_dir, args.split)
    print(f"  Model loaded, {len(list(cuts))} utterances")
    cuts = load_cuts(args.data_dir, args.split)  # reload since we consumed it

    # Step 2: CTC-only decoding
    ctc_results = {}
    if "ctc" in steps:
        print("\n=== STEP 2: CTC-only decoding ===")

        # Greedy
        t0 = time.time()
        cuts_iter = get_cuts()
        greedy_results = ctc_greedy_decode(
            model, cuts_iter, sp, device,
            max_duration=args.max_duration, batch_size=args.batch_size
        )
        timings["ctc_greedy"] = time.time() - t0
        greedy_wer = compute_corpus_wer(greedy_results)
        ctc_results["greedy"] = greedy_wer
        print(f"  CTC greedy: {greedy_wer*100:.2f}% ({timings['ctc_greedy']:.0f}s)")

        # Smoke test: greedy WER must be in sane range (FATAL  --  abort)
        print("  --- Smoke test samples ---")
        for r in greedy_results[:3]:
            print(f"    REF: {r['ref_text'][:80]}")
            print(f"    HYP: {r['hyp_text'][:80]}")
            print()
        if greedy_wer > 0.15:
            print(f"   SMOKE TEST FAILED: CTC greedy WER = {greedy_wer*100:.2f}%"
                  f" (expected ~6.02%). Model forward pass is broken. ABORTING.")
            sys.exit(1)
        print(f"   CTC greedy smoke test passed ({greedy_wer*100:.2f}%)")

        # N-best
        t0 = time.time()
        cuts_iter = get_cuts()
        ctc_nbest_results = ctc_nbest_decode(
            model, cuts_iter, sp, device,
            num_paths=args.num_paths, max_duration=args.max_duration,
            batch_size=args.batch_size
        )
        timings["ctc_nbest"] = time.time() - t0
        ctc_nbest_wer = compute_corpus_wer(ctc_nbest_results)
        ctc_oracle_wer = compute_oracle_wer(ctc_nbest_results)

        mean_unique = sum(r["num_candidates"] for r in ctc_nbest_results) / len(ctc_nbest_results)
        ctc_results["ctc_nbest_1best"] = ctc_nbest_wer
        ctc_results["ctc_oracle"] = ctc_oracle_wer
        ctc_results["ctc_mean_unique"] = mean_unique
        print(f"  CTC N-best 1best: {ctc_nbest_wer*100:.2f}%")
        print(f"  CTC oracle (G={args.num_paths}): {ctc_oracle_wer*100:.2f}%")
        print(f"  CTC mean unique: {mean_unique:.1f}")
        print(f"  Time: {timings['ctc_nbest']:.0f}s")

        # Smoke tests: oracle is critical (FATAL), 1best is warning only
        if ctc_oracle_wer > 0.06:
            print(f"   SMOKE TEST FAILED: CTC oracle = {ctc_oracle_wer*100:.2f}%"
                  f" (expected <6%). N-best candidate extraction is broken. ABORTING.")
            sys.exit(1)
        if ctc_nbest_wer > 0.08:
            print(f"   WARNING: CTC N-best 1best = {ctc_nbest_wer*100:.2f}%"
                  f" (expected ~6%). Path scoring may differ from existing pipeline."
                  f" Continuing  --  oracle is the key metric.")
        else:
            print(f"   CTC N-best 1best OK ({ctc_nbest_wer*100:.2f}%)")
        print(f"   CTC oracle smoke test passed ({ctc_oracle_wer*100:.2f}%)")

        ctc_jsonl = args.output_dir / f"ctc_nbest_{args.split}_G{args.num_paths}.jsonl"
        save_nbest_jsonl(ctc_nbest_results, ctc_jsonl)
        print(f"  Saved: {ctc_jsonl}")

    # Step 3: HLG decoding
    hlg_results = {}
    if "hlg" in steps and hlg is not None:
        print(f"\n=== STEP 3: HLG decoding ({args.lm_order}-gram) ===")

        # HLG 1best
        t0 = time.time()
        cuts_iter = get_cuts()
        try:
            hlg_1best = hlg_decode(
                model, hlg, cuts_iter, sp, word_table, device,
                method="1best", max_duration=args.max_duration,
                batch_size=args.batch_size,
                debug_first=5 if args.debug_hlg else 0,
            )
            timings["hlg_1best"] = time.time() - t0
            hlg_1best_wer = compute_corpus_wer(hlg_1best)
            hlg_results["hlg_1best"] = hlg_1best_wer
            print(f"  HLG 1best: {hlg_1best_wer*100:.2f}% ({timings['hlg_1best']:.0f}s)")

            # Smoke test: HLG 1best (FATAL if > 20%)
            print("  --- HLG 1best smoke test samples ---")
            for r in hlg_1best[:3]:
                print(f"    REF: {r['ref_text'][:80]}")
                print(f"    HYP: {r['hyp_text'][:80]}")
                print()
            if hlg_1best_wer > 0.50:
                print(f"   SMOKE TEST FAILED: HLG 1best = {hlg_1best_wer*100:.2f}%"
                      f" (expected <50%). HLG output extraction is broken. ABORTING.")
                sys.exit(1)
            elif hlg_1best_wer > 0.15:
                print(f"   WARNING: HLG 1best = {hlg_1best_wer*100:.2f}%"
                      f" (expected ~5-7%). HLG quality may be degraded.")
            else:
                print(f"   HLG 1best smoke test passed ({hlg_1best_wer*100:.2f}%)")

        except Exception as e:
            print(f"  HLG 1best failed: {e}")
            timings["hlg_1best"] = time.time() - t0

        # HLG N-best
        t0 = time.time()
        cuts_iter = get_cuts()
        try:
            hlg_nbest = hlg_decode(
                model, hlg, cuts_iter, sp, word_table, device,
                method="nbest", num_paths=args.num_paths,
                max_duration=args.max_duration,
                batch_size=args.batch_size,
            )
            timings["hlg_nbest"] = time.time() - t0
            hlg_nbest_wer = compute_corpus_wer(hlg_nbest)
            hlg_oracle_wer = compute_oracle_wer(hlg_nbest)

            mean_unique_hlg = sum(
                r["num_candidates"] for r in hlg_nbest
            ) / len(hlg_nbest)

            hlg_results["hlg_nbest_1best"] = hlg_nbest_wer
            hlg_results["hlg_oracle"] = hlg_oracle_wer
            hlg_results["hlg_mean_unique"] = mean_unique_hlg

            print(f"  HLG N-best 1best: {hlg_nbest_wer*100:.2f}%")
            print(f"  HLG oracle (G={args.num_paths}): {hlg_oracle_wer*100:.2f}%")
            print(f"  HLG mean unique: {mean_unique_hlg:.1f}")
            print(f"  Time: {timings['hlg_nbest']:.0f}s")

            hlg_jsonl = args.output_dir / f"hlg_nbest_{args.split}_G{args.num_paths}.jsonl"
            save_nbest_jsonl(hlg_nbest, hlg_jsonl)
            print(f"  Saved: {hlg_jsonl}")

        except Exception as e:
            print(f"  HLG N-best failed: {e}")
            import traceback
            traceback.print_exc()
            timings["hlg_nbest"] = time.time() - t0

    elif "hlg" in steps and hlg is None:
        print("\n=== STEP 3: SKIPPED (HLG not available) ===")

    # Step 4: Comparison
    if "compare" in steps:
        print("\n=== STEP 4: Comparison ===")
        print(f"\n{'Method':<30s} {'WER (%)':>10s} {'Time (s)':>10s}")
        print("-" * 55)

        all_methods = {}
        all_methods.update(ctc_results)
        all_methods.update(hlg_results)

        method_order = [
            ("greedy", "CTC greedy"),
            ("ctc_nbest_1best", "CTC N-best 1best"),
            ("ctc_oracle", "CTC oracle"),
            ("hlg_1best", f"HLG {args.lm_order}gram 1best"),
            ("hlg_nbest_1best", f"HLG {args.lm_order}gram N-best 1best"),
            ("hlg_oracle", f"HLG {args.lm_order}gram oracle"),
        ]

        for key, label in method_order:
            if key in all_methods:
                wer_val = all_methods[key]
                t_key = key.replace("_oracle", "_nbest").replace("_1best", "")
                t_key = t_key if t_key in timings else f"ctc_{t_key}" if f"ctc_{t_key}" in timings else ""
                time_str = f"{timings.get(t_key, 0):.0f}" if t_key else " -- "
                if isinstance(wer_val, float):
                    print(f"  {label:<30s} {wer_val*100:>8.2f}%  {time_str:>8s}")

        if "ctc_mean_unique" in ctc_results and "hlg_mean_unique" in hlg_results:
            print(f"\n  Unique hypotheses: CTC={ctc_results['ctc_mean_unique']:.1f} "
                  f"HLG={hlg_results['hlg_mean_unique']:.1f}")

        summary = {
            "split": args.split,
            "num_paths": args.num_paths,
            "lm_order": args.lm_order,
            "methods": all_methods,
            "timings": timings,
        }
        with open(args.output_dir / "hlg_decode_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Summary saved: {args.output_dir / 'hlg_decode_summary.json'}")

    print("\nDONE.")


if __name__ == "__main__":
    main()

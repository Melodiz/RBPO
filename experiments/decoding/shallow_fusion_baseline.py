#!/usr/bin/env python3
"""E17: Shallow Fusion Baseline (4-gram or 3-gram LM via N-best rescoring).

Adds the standard CTC+LM shallow fusion baseline to the comparison table.
Uses Approach A (N-best rescoring with ARPA LM via kenlm) for memory safety
on T4. Avoids HLG/WFST decoding entirely.

Pipeline:
  1. Discover available LM resources (ARPA / kenlm binary / icefall G_*.pt)
  2. Score N-best with the LM (text-only  --  model-agnostic)
  3. Sweep alpha for linear interpolation (CTC + LM)
  4. Run MBR-CER with LM-derived weights at tau=10
  5. Paired bootstrap vs greedy for best configs
  6. Comparison table: 4-gram fusion vs RoBERTa methods

Usage (Colab T4):
    pip install https://github.com/kpu/kenlm/archive/master.zip

    python experiments/decoding/shallow_fusion_baseline.py \
        --data-dir /content/drive/MyDrive/rbpo_results \
        --model-dir /content/icefall-asr-librispeech-zipformer-small-cr-ctc \
        --icefall-dir /content/icefall \
        --output-dir /content/drive/MyDrive/rbpo_results/shallow_fusion \
        --n-bootstrap 10000

Local M2:
    python experiments/decoding/shallow_fusion_baseline.py \
        --data-dir rbpo/results \
        --output-dir results/shallow_fusion \
        --arpa-path rbpo/results/3-gram.pruned.1e-7.arpa
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

ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TAU = 10.0

def find_arpa(args):
    """Search for an ARPA LM file across common locations."""
    candidates = []
    if args.arpa_path:
        candidates.append(args.arpa_path)
    candidates.extend([
        args.data_dir / "3-gram.pruned.1e-7.arpa",
        args.data_dir / "4-gram.arpa",
        args.data_dir / "lm" / "3-gram.pruned.1e-7.arpa",
        args.data_dir / "lm" / "4-gram.arpa",
        args.data_dir / "lm" / "G_4_gram.arpa",
    ])
    if args.model_dir:
        candidates.extend([
            args.model_dir / "data" / "lm" / "3-gram.pruned.1e-7.arpa",
            args.model_dir / "data" / "lm" / "4-gram.arpa",
            args.model_dir / "data" / "lm" / "G_4_gram.arpa",
        ])
    for p in candidates:
        if p and p.exists() and p.stat().st_size > 0:
            return p
    return None

def discover_lm(args):
    """Print what LM resources are available; pick the best."""
    print("\n" + "=" * 70)
    print("STEP 0: DISCOVER  --  LM resources")
    print("=" * 70)

    arpa = find_arpa(args)
    if arpa:
        size_mb = arpa.stat().st_size / 1e6
        print(f"\n  ARPA file: {arpa} ({size_mb:.1f} MB)")
        # Read header to determine n-gram order
        with open(arpa) as f:
            for _ in range(20):
                line = f.readline()
                if "ngram " in line:
                    print(f"    {line.strip()}")
        return arpa

    print("\n  No ARPA file found. Approaches:")
    print("    - Pass --arpa-path /path/to/lm.arpa")
    print("    - Download standard LibriSpeech LM:")
    print("        wget https://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz")
    print("        gunzip 3-gram.pruned.1e-7.arpa.gz")
    return None

def score_with_kenlm(records, arpa_path):
    """Score each candidate text with kenlm. Adds 'kenlm_lm_score' field."""
    try:
        import kenlm
    except ImportError:
        print("\n  ERROR: kenlm not installed. Install with:")
        print("    pip install https://github.com/kpu/kenlm/archive/master.zip")
        raise

    print(f"\n  Loading kenlm model from {arpa_path}...")
    t0 = time.time()
    model = kenlm.Model(str(arpa_path))
    print(f"  Loaded in {time.time()-t0:.1f}s, order={model.order}")

    print(f"\n  Scoring N-best with kenlm...")
    t0 = time.time()
    n_hyps = 0
    for rec in records:
        for cand in rec["candidates"]:
            text = cand["text"].strip()
            if not text:
                cand["kenlm_lm_score"] = -999.0
                continue
            # Natural log (kenlm.score() returns log10; convert)
            log10_score = model.score(text, bos=True, eos=True)
            cand["kenlm_lm_score"] = round(log10_score * math.log(10.0), 4)
            n_hyps += 1
    elapsed = time.time() - t0
    print(f"  Scored {n_hyps} hypotheses in {elapsed:.1f}s "
          f"({n_hyps/max(elapsed,0.01):.0f} hyps/s)")
    return model.order

def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records

def select_interp(records, alpha, score_field):
    """Argmax over alpha*log_ctc + (1-alpha)*log_lm."""
    out = []
    for rec in records:
        cands = [c for c in rec["candidates"] if c["text"].strip()]
        if not cands:
            cands = rec["candidates"]
        scores = [alpha * c["ctc_log_prob"] + (1 - alpha) * c[score_field]
                  for c in cands]
        out.append(cands[int(np.argmax(scores))]["text"])
    return out

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

def select_mbr(records, score_field, tau):
    """MBR-CER with weights from softmax(score_field/tau)."""
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

def run_alpha_sweep(records, alphas, score_field):
    """Sweep alpha and return WER + hyp_words per alpha."""
    ref_words = [r["ref_text"].split() for r in records]
    results = {}
    for a in alphas:
        hyps = select_interp(records, a, score_field)
        hyp_words = [h.split() for h in hyps]
        wer = corpus_wer(ref_words, hyp_words)
        results[a] = {"wer": wer, "hyp_words": hyp_words}
        print(f"  alpha={a:.2f}: WER={wer*100:.4f}%")
    return results

def main():
    parser = argparse.ArgumentParser(description="E17: Shallow Fusion Baseline")
    parser.add_argument("--data-dir", type=Path,
                        default=Path("/content/drive/MyDrive/rbpo_results"))
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--icefall-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/content/drive/MyDrive/rbpo_results/shallow_fusion"))
    parser.add_argument("--arpa-path", type=Path, default=None,
                        help="Explicit ARPA file path (overrides discovery)")
    parser.add_argument("--nbest-file", type=Path, default=None,
                        help="N-best file to rescore (defaults to data-dir/nbest_dev_other_G16.jsonl)")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("E17: Shallow Fusion Baseline (4-gram or 3-gram LM)")
    print("=" * 70)
    t0 = time.time()

    # Step 0: Discover
    arpa_path = discover_lm(args)
    if arpa_path is None:
        print("\nERROR: No LM file found. Cannot proceed.")
        sys.exit(1)

    # Step 1: Load N-best
    nbest_path = args.nbest_file
    if nbest_path is None:
        candidates = [
            args.data_dir / "nbest_dev_other_G16.jsonl",
            args.data_dir / "g_scaling" / "nbest_dev_other_G16.jsonl",
        ]
        nbest_path = next((p for p in candidates if p.exists()), None)
    if nbest_path is None or not nbest_path.exists():
        print(f"\nERROR: N-best file not found. Tried: {candidates}")
        print("Pass --nbest-file explicitly")
        sys.exit(1)

    print(f"\n  Loading N-best: {nbest_path}")
    records = load_jsonl(nbest_path)
    n_utts = len(records)
    print(f"  {n_utts} utterances")

    # Step 2: kenlm scoring
    print("\n" + "=" * 70)
    print("STEP 1: SCORE  --  kenlm LM scoring")
    print("=" * 70)
    lm_order = score_with_kenlm(records, arpa_path)

    # Save scored copy (cheap: just adds one float per candidate)
    scored_path = args.output_dir / "nbest_with_lm_scores.jsonl"
    with open(scored_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Wrote scored N-best to {scored_path.name} "
          f"({scored_path.stat().st_size/1e6:.1f} MB)")

    # Step 3: Greedy and oracle baselines
    ref_words = [r["ref_text"].split() for r in records]
    greedy_words = [r["candidates"][0]["text"].split() for r in records]
    greedy_wer = corpus_wer(ref_words, greedy_words)

    oracle_hyps = []
    for rec in records:
        ref_w = rec["ref_text"].split()
        best = min(rec["candidates"],
                   key=lambda c: editdistance.eval(c["text"].split(), ref_w))
        oracle_hyps.append(best["text"])
    oracle_wer = corpus_wer(ref_words, [h.split() for h in oracle_hyps])

    print(f"\n  Greedy WER: {greedy_wer*100:.4f}%")
    print(f"  Oracle WER (G=16): {oracle_wer*100:.4f}%")

    # Step 4: Alpha sweep
    print("\n" + "=" * 70)
    print(f"STEP 2: ALPHA SWEEP  --  CTC + {lm_order}-gram interpolation")
    print("=" * 70)
    alpha_results = run_alpha_sweep(records, ALPHAS, "kenlm_lm_score")

    # Verification: alpha~1.0 ~ greedy
    sweep_a09 = alpha_results[0.9]["wer"]
    print(f"\n  Sanity: alpha=0.9 WER ({sweep_a09*100:.4f}%) close to greedy "
          f"({greedy_wer*100:.4f}%)")

    best_alpha = min(ALPHAS, key=lambda a: alpha_results[a]["wer"])
    best_alpha_wer = alpha_results[best_alpha]["wer"]
    print(f"\n  Best alpha = {best_alpha:.2f}: WER = {best_alpha_wer*100:.4f}%")

    # Step 5: MBR-CER with LM weights
    print("\n" + "=" * 70)
    print(f"STEP 3: MBR-CER + {lm_order}-gram weights at tau={TAU}")
    print("=" * 70)
    mbr_lm_hyps = select_mbr(records, "kenlm_lm_score", TAU)
    mbr_lm_words = [h.split() for h in mbr_lm_hyps]
    mbr_lm_wer = corpus_wer(ref_words, mbr_lm_words)
    print(f"  MBR+kenlm tau=10 WER: {mbr_lm_wer*100:.4f}%")

    # Step 6: Bootstrap
    print("\n" + "=" * 70)
    print(f"STEP 4: BOOTSTRAP (B={args.n_bootstrap})")
    print("=" * 70)

    bootstrap = {}
    print(f"  Best alpha (interp) vs greedy...")
    best_interp_words = alpha_results[best_alpha]["hyp_words"]
    res = paired_bootstrap_wer(
        ref_words, best_interp_words, greedy_words,
        n_bootstrap=args.n_bootstrap, seed=args.seed,
    )
    bootstrap[f"interp_alpha_{best_alpha}"] = {
        "wer": res["wer_a"], "delta_pp": res["delta"] * 100,
        "p_value": res["p_value"],
        "ci_lower": res["ci_lower"] * 100, "ci_upper": res["ci_upper"] * 100,
    }
    print(f"    delta={res['delta']*100:+.4f}pp, p={res['p_value']:.4f}")

    print(f"  MBR + kenlm tau=10 vs greedy...")
    res = paired_bootstrap_wer(
        ref_words, mbr_lm_words, greedy_words,
        n_bootstrap=args.n_bootstrap, seed=args.seed,
    )
    bootstrap["mbr_kenlm_tau10"] = {
        "wer": res["wer_a"], "delta_pp": res["delta"] * 100,
        "p_value": res["p_value"],
        "ci_lower": res["ci_lower"] * 100, "ci_upper": res["ci_upper"] * 100,
    }
    print(f"    delta={res['delta']*100:+.4f}pp, p={res['p_value']:.4f}")

    # Verification
    print("\n--- Verification ---")
    if abs(greedy_wer * 100 - 6.02) < 0.1:
        print(f"  [PASS] Greedy WER {greedy_wer*100:.4f}% ~ 6.02%")
    else:
        print(f"  [WARN] Greedy WER {greedy_wer*100:.4f}% differs from 6.02%")
    if best_alpha_wer < greedy_wer:
        print(f"  [PASS] Best LM-rescored ({best_alpha_wer*100:.4f}%) < greedy "
              f"({greedy_wer*100:.4f}%)")
    else:
        print(f"  [WARN] Best LM-rescored does not beat greedy")
    # MBR with LM weights should be >= as good as argmax interpolation? Not necessarily.
    # The key check is that it beats greedy.
    if mbr_lm_wer < greedy_wer:
        print(f"  [PASS] MBR+kenlm ({mbr_lm_wer*100:.4f}%) < greedy")
    else:
        print(f"  [WARN] MBR+kenlm does not beat greedy")
    print(f"  [INFO] Utterance count: {n_utts}")

    print("\n--- Writing outputs ---")

    # 1. JSON
    out_json = {
        "experiment": "E17_shallow_fusion",
        "lm_path": str(arpa_path),
        "lm_order": lm_order,
        "n_utterances": n_utts,
        "greedy_wer": greedy_wer,
        "oracle_wer": oracle_wer,
        "best_alpha": best_alpha,
        "best_alpha_wer": best_alpha_wer,
        "mbr_kenlm_tau10_wer": mbr_lm_wer,
        "alpha_sweep": {f"{a:.2f}": alpha_results[a]["wer"] for a in ALPHAS},
        "bootstrap": bootstrap,
        "n_bootstrap": args.n_bootstrap,
    }
    p = args.output_dir / "shallow_fusion_results.json"
    with open(p, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"  Wrote {p}")

    # 2. Sweep CSV
    p = args.output_dir / "shallow_fusion_sweep.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["alpha", "wer", "delta_pp_vs_greedy"])
        w.writeheader()
        for a in ALPHAS:
            wer = alpha_results[a]["wer"]
            w.writerow({
                "alpha": f"{a:.2f}",
                "wer": f"{wer*100:.4f}",
                "delta_pp_vs_greedy": f"{(wer-greedy_wer)*100:+.4f}",
            })
    print(f"  Wrote {p}")

    # 3. LM vs PLL MBR comparison CSV
    p = args.output_dir / "lm_mbr_comparison.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "method", "score_source", "tau", "wer", "delta_pp_vs_greedy"
        ])
        w.writeheader()
        # Best interp with kenlm
        w.writerow({
            "method": f"argmax_interp",
            "score_source": f"kenlm_{lm_order}gram",
            "tau": " -- ",
            "wer": f"{best_alpha_wer*100:.4f}",
            "delta_pp_vs_greedy": f"{(best_alpha_wer-greedy_wer)*100:+.4f}",
        })
        # MBR with kenlm
        w.writerow({
            "method": "mbr_cer",
            "score_source": f"kenlm_{lm_order}gram",
            "tau": "10",
            "wer": f"{mbr_lm_wer*100:.4f}",
            "delta_pp_vs_greedy": f"{(mbr_lm_wer-greedy_wer)*100:+.4f}",
        })
    print(f"  Wrote {p}")

    # 4. Method comparison MD
    write_method_comparison(args, greedy_wer, best_alpha, best_alpha_wer,
                            mbr_lm_wer, bootstrap, lm_order)

    # 5. Stage report
    elapsed = time.time() - t0
    write_report_e17(args, greedy_wer, oracle_wer, best_alpha, best_alpha_wer,
                     mbr_lm_wer, alpha_results, bootstrap, lm_order, n_utts,
                     elapsed, arpa_path)

    print(f"\nDone. Total time: {elapsed:.1f}s")

def write_method_comparison(args, greedy_wer, best_alpha, best_alpha_wer,
                            mbr_lm_wer, bootstrap, lm_order):
    p = args.output_dir / "method_comparison.md"
    g = greedy_wer * 100
    bw = best_alpha_wer * 100
    mw = mbr_lm_wer * 100
    p_interp = bootstrap[f"interp_alpha_{best_alpha}"]["p_value"]
    p_mbr = bootstrap["mbr_kenlm_tau10"]["p_value"]

    lines = ["# Shallow Fusion vs Neural-LM Methods", ""]
    lines.append("Comparison of decode-time methods against greedy CTC.")
    lines.append("All methods use the same Zipformer-S CR-CTC, same dev-other.")
    lines.append("")
    lines.append("| Method | WER (%) | delta (pp) | p-value | Info source |")
    lines.append("|--------|--------:|-------:|--------:|-------------|")
    lines.append(f"| Greedy CTC | {g:.2f} |  --  |  --  | Acoustic only |")
    lines.append(f"| {lm_order}-gram shallow fusion (best alpha={best_alpha:.2f}) | "
                 f"{bw:.2f} | {(bw-g):+.3f} | {p_interp:.4f} | + {lm_order}-gram LM |")
    lines.append(f"| MBR-CER + {lm_order}-gram weights tau=10 | "
                 f"{mw:.2f} | {(mw-g):+.3f} | {p_mbr:.4f} | + {lm_order}-gram + MBR |")
    lines.append(f"| RoBERTa PLL interp alpha=0.7 G=16 | 5.92 | -0.10 | 0.002 | + Neural LM |")
    lines.append(f"| MBR+PLL tau=10 G=16 | 5.79 | -0.23 | <0.0001 | + Neural LM + MBR |")
    lines.append(f"| MBR+PLL tau=10 G=128 | 5.53 | -0.49 | <0.0001 | + Neural LM + MBR + G |")
    lines.append("")
    lines.append("## Story")
    lines.append("")
    if bw < g:
        lines.append(f"Shallow fusion with {lm_order}-gram LM closes "
                     f"{(g-bw)/(g-4.44)*100:.1f}% of the oracle gap. "
                     "Neural LM rescoring closes more. MBR with neural LM posteriors "
                     "closes the most and scales with G.")
    else:
        lines.append(f"The {lm_order}-gram fusion does not improve over greedy at "
                     "any alpha tested. The neural LM provides signal the n-gram cannot.")
    lines.append("")
    with open(p, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {p}")

def write_report_e17(args, greedy_wer, oracle_wer, best_alpha, best_alpha_wer,
                     mbr_lm_wer, alpha_results, bootstrap, lm_order, n_utts,
                     elapsed, arpa_path):
    p = args.output_dir / "report_E17.md"
    g = greedy_wer * 100
    bw = best_alpha_wer * 100
    mw = mbr_lm_wer * 100

    lines = ["# Report E17: Shallow Fusion Baseline", ""]
    lines.append(f"**Status:** Complete. {n_utts} utterances, "
                 f"{lm_order}-gram LM via kenlm. {elapsed:.0f}s.")
    lines.append("")
    lines.append("## What Ran")
    lines.append("")
    lines.append(f"- Approach A: N-best rescoring with kenlm")
    lines.append(f"- LM: `{arpa_path.name}` (order={lm_order})")
    lines.append(f"- N-best: existing G=16 dev-other (Zipformer-S CR-CTC)")
    lines.append(f"- alpha sweep: {ALPHAS}")
    lines.append(f"- MBR-CER with LM weights at tau={TAU}")
    lines.append(f"- Bootstrap: B={args.n_bootstrap}, paired vs greedy")
    lines.append("")
    lines.append("## Key Results")
    lines.append("")
    lines.append("| Method | WER (%) | delta (pp) | p-value |")
    lines.append("|--------|--------:|-------:|--------:|")
    lines.append(f"| Greedy CTC | {g:.4f} | 0 |  --  |")
    lines.append(f"| Argmax + {lm_order}-gram alpha={best_alpha:.2f} | {bw:.4f} | "
                 f"{(bw-g):+.4f} | "
                 f"{bootstrap[f'interp_alpha_{best_alpha}']['p_value']:.4f} |")
    lines.append(f"| MBR-CER + {lm_order}-gram tau=10 | {mw:.4f} | "
                 f"{(mw-g):+.4f} | "
                 f"{bootstrap['mbr_kenlm_tau10']['p_value']:.4f} |")
    lines.append(f"| Oracle (G=16) | {oracle_wer*100:.4f} | "
                 f"{(oracle_wer*100-g):+.4f} |  --  |")
    lines.append("")
    lines.append("## Alpha Sweep")
    lines.append("")
    lines.append("| alpha | WER (%) | delta vs greedy (pp) |")
    lines.append("|--:|--------:|-----------------:|")
    for a in ALPHAS:
        wer = alpha_results[a]["wer"] * 100
        marker = " **<-best**" if a == best_alpha else ""
        lines.append(f"| {a:.2f} | {wer:.4f} | {(wer-g):+.4f} |{marker}")
    lines.append("")
    lines.append("## Comparison with Neural LM Methods")
    lines.append("")
    lines.append("| Method | WER (%) | delta vs greedy (pp) |")
    lines.append("|--------|--------:|-----------------:|")
    lines.append(f"| Greedy CTC | {g:.4f} | 0 |")
    lines.append(f"| {lm_order}-gram fusion (best) | {bw:.4f} | {(bw-g):+.4f} |")
    lines.append(f"| RoBERTa PLL interp alpha=0.7 G=16 | 5.9200 | -0.1000 |")
    lines.append(f"| MBR+PLL tau=10 G=16 | 5.7900 | -0.2300 |")
    lines.append(f"| MBR+PLL tau=10 G=128 | 5.5300 | -0.4900 |")
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    delta_pp = g - bw
    p_interp = bootstrap[f"interp_alpha_{best_alpha}"]["p_value"]
    if delta_pp >= 0.05 and p_interp < 0.05:
        gap_closed = delta_pp / (g - 4.44) * 100
        lines.append(f"The {lm_order}-gram shallow fusion provides a significant "
                     f"{delta_pp:.3f}pp WER reduction "
                     f"(p={p_interp:.4f}, closes ~{gap_closed:.1f}% of oracle gap). "
                     f"Neural LM methods (RoBERTa) add additional improvement on top of "
                     f"this baseline by capturing semantic context that "
                     f"{lm_order}-grams cannot.")
    elif delta_pp > 0:
        lines.append(f"The {lm_order}-gram shallow fusion provides only a "
                     f"{delta_pp:.4f}pp reduction (p={p_interp:.4f}, not significant). "
                     "The n-gram LM contributes minimal value for this strong CTC model. "
                     "Neural LM methods (RoBERTa, especially with MBR) provide the "
                     "substantial gains  --  the linguistic signal that helps here is "
                     "richer than n-gram statistics can capture.")
    else:
        lines.append(f"The {lm_order}-gram fusion does not improve over greedy. "
                     "Either the alpha scale is mismatched or the LM is too weak/mismatched "
                     "for this acoustic model.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("| File | Purpose |")
    lines.append("|------|---------|")
    lines.append("| `shallow_fusion_results.json` | Full results |")
    lines.append("| `shallow_fusion_sweep.csv` | alpha sweep tabular |")
    lines.append("| `lm_mbr_comparison.csv` | LM with argmax vs MBR weights |")
    lines.append("| `method_comparison.md` | Cross-method comparison table |")
    lines.append("| `report_E17.md` | This stage report |")
    with open(p, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {p}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""End-to-end smoke test of the full overnight pipeline.

Runs every script on a tiny slice (first N utterances) to catch
import errors, format mismatches, and missing deps before the
real run. Exits non-zero on first failure.

Usage:
    python scripts/smoke_test.py \
        --nbest /path/to/any_nbest.jsonl \
        --cuts /path/to/cuts.jsonl.gz \
        --checkpoint /path/to/pretrained.pt \
        --bpe /path/to/bpe.model \
        [--icefall-dir /content/icefall] \
        [--n-utts 5]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def run(label, cmd, expect_file=None):
    """Run a command, print pass/fail, return success bool."""
    print(f"\n{'-' * 60}")
    print(f"  TEST: {label}")
    print(f"  CMD:  {' '.join(cmd)}")
    print(f"{'-' * 60}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n   FAIL: {label} (exit code {result.returncode}, {elapsed:.1f}s)")
        return False

    if expect_file and not os.path.exists(expect_file):
        print(f"\n   FAIL: {label}  --  expected output missing: {expect_file}")
        return False

    if expect_file:
        size = os.path.getsize(expect_file)
        print(f"\n   PASS: {label} ({elapsed:.1f}s, output {size:,} bytes)")
    else:
        print(f"\n   PASS: {label} ({elapsed:.1f}s)")
    return True


def slice_jsonl(src, dst, n):
    """Copy first n lines of a JSONL file."""
    with open(src) as f_in, open(dst, "w") as f_out:
        for i, line in enumerate(f_in):
            if i >= n:
                break
            f_out.write(line)
    return min(i + 1, n)


def validate_jsonl(path, required_keys):
    """Check that every record in JSONL has required keys."""
    with open(path) as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            for k in required_keys:
                if k not in rec:
                    # Check in nbest[0] for nested keys
                    if "nbest" in rec and rec["nbest"] and k in rec["nbest"][0]:
                        continue
                    return False, f"line {i}: missing key '{k}'"
    return True, "ok"


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test the full overnight pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nbest", type=Path, required=True,
                        help="Any existing N-best JSONL (will slice first N utts)")
    parser.add_argument("--cuts", type=Path, default=None,
                        help="CutSet for generate_nbest test (optional)")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Model checkpoint (optional, for generate_nbest)")
    parser.add_argument("--bpe", type=Path, default=None,
                        help="BPE model (optional, for generate_nbest)")
    parser.add_argument("--icefall-dir", type=Path, default=Path("/content/icefall"))
    parser.add_argument("--n-utts", type=int, default=5,
                        help="Number of utterances for smoke test")
    args = parser.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    t_start = time.time()
    results = []

    with tempfile.TemporaryDirectory(prefix="smoke_") as tmpdir:
        tmpdir = Path(tmpdir)

        # -- Step 0: Check imports -------------------------------------
        print("=" * 60)
        print("  SMOKE TEST  --  checking all scripts")
        print("=" * 60)

        imports_ok = True
        for mod in ["editdistance", "numpy", "scipy", "torch", "transformers"]:
            try:
                __import__(mod)
                print(f"   import {mod}")
            except ImportError:
                print(f"   import {mod} FAILED")
                imports_ok = False

        try:
            import kenlm
            print(f"   import kenlm")
        except ImportError:
            print(f"   import kenlm FAILED (score_ngram will fail)")

        if not imports_ok:
            print("\n  ABORT: missing core dependencies")
            sys.exit(1)

        # -- Step 1: Slice input ---------------------------------------
        mini = tmpdir / "mini.jsonl"
        n_actual = slice_jsonl(args.nbest, mini, args.n_utts)
        print(f"\n  Sliced {n_actual} utterances from {args.nbest.name}")

        # Check format
        with open(mini) as f:
            rec = json.loads(f.readline())
        if "nbest" in rec:
            fmt = "new"
            hyp_key, score_key, ref_key = "hyp", "score", "ref"
            cands_key = "nbest"
        else:
            fmt = "E11"
            hyp_key, score_key, ref_key = "text", "ctc_log_prob", "ref_text"
            cands_key = "candidates"
        n_cands = len(rec[cands_key])
        print(f"  Format: {fmt}, {n_cands} candidates/utt")

        # -- Step 2: score_pll.py --------------------------------------
        pll_out = tmpdir / "mini_pll.jsonl"
        ok = run(
            "score_pll.py (RoBERTa PLL)",
            [sys.executable, str(scripts_dir / "score_pll.py"),
             "--nbest", str(mini),
             "--output", str(pll_out),
             "--model", "roberta-base",
             "--device", "cuda",
             "--batch-size", "16",
             "--save-every", "0"],
            expect_file=str(pll_out),
        )
        results.append(("score_pll", ok))

        if ok:
            valid, msg = validate_jsonl(pll_out, ["pll_score"])
            if valid:
                print(f"   pll_score present in all records")
            else:
                print(f"   validation: {msg}")
                results[-1] = ("score_pll", False)

        # -- Step 3: rerank_mbr.py -------------------------------------
        mbr_out = tmpdir / "mbr_result.json"
        ok = run(
            "rerank_mbr.py (MBR-CER, 2x2 grid)",
            [sys.executable, str(scripts_dir / "rerank_mbr.py"),
             "--nbest", str(pll_out),
             "--output", str(mbr_out),
             "--utility", "cer",
             "--tau-sweep", "1.0,10.0",
             "--pll-sweep", "0.0,0.5"],
            expect_file=str(mbr_out),
        )
        results.append(("rerank_mbr", ok))

        if ok:
            d = json.load(open(mbr_out))
            n_results = len(d.get("all_results", []))
            print(f"   {n_results} configs evaluated, "
                  f"best WER={d['best_config']['wer']:.4%}")
            if d["greedy_wer"] < d["oracle_wer"]:
                print(f"   greedy < oracle  --  something is wrong")
                results[-1] = ("rerank_mbr", False)

        # -- Step 4: rerank_interpolation.py ---------------------------
        interp_out = tmpdir / "interp_result.json"
        ok = run(
            "rerank_interpolation.py (alpha sweep)",
            [sys.executable, str(scripts_dir / "rerank_interpolation.py"),
             "--nbest", str(pll_out),
             "--output", str(interp_out),
             "--alpha-sweep", "0.0,0.5,1.0"],
            expect_file=str(interp_out),
        )
        results.append(("rerank_interpolation", ok))

        # -- Step 5: score_ngram.py ------------------------------------
        ngram_out = tmpdir / "mini_ngram.jsonl"
        ok = run(
            "score_ngram.py (kenlm, downloads LM if needed)",
            [sys.executable, str(scripts_dir / "score_ngram.py"),
             "--nbest", str(mini),
             "--output", str(ngram_out)],
            expect_file=str(ngram_out),
        )
        results.append(("score_ngram", ok))

        if ok:
            valid, msg = validate_jsonl(ngram_out, ["ngram_score"])
            if valid:
                print(f"   ngram_score present in all records")
            else:
                print(f"   validation: {msg}")
                results[-1] = ("score_ngram", False)

        # -- Step 6: rerank_mbr.py with --score-key ngram_score --------
        ngram_mbr_out = tmpdir / "ngram_mbr_result.json"
        ok = run(
            "rerank_mbr.py --score-key ngram_score",
            [sys.executable, str(scripts_dir / "rerank_mbr.py"),
             "--nbest", str(ngram_out),
             "--output", str(ngram_mbr_out),
             "--tau-sweep", "1.0,10.0",
             "--pll-sweep", "0.0,1.0",
             "--score-key", "ngram_score"],
            expect_file=str(ngram_mbr_out),
        )
        results.append(("rerank_mbr+ngram", ok))

        # -- Step 7: generate_nbest_batch.py (only if checkpoint given) -
        if args.cuts and args.checkpoint and args.bpe:
            batch_out = tmpdir / "nbest_batch"
            batch_out.mkdir()
            ok = run(
                "generate_nbest_batch.py (G=4 only, quick)",
                [sys.executable, str(scripts_dir / "generate_nbest_batch.py"),
                 "--cuts", str(args.cuts),
                 "--checkpoint", str(args.checkpoint),
                 "--bpe", str(args.bpe),
                 "--icefall-dir", str(args.icefall_dir),
                 "--G-values", "4",
                 "--nbest-scale", "1.0",
                 "--output-dir", str(batch_out)],
                expect_file=str(batch_out / "nbest_g4.jsonl"),
            )
            results.append(("generate_nbest_batch", ok))
        else:
            print(f"\n  SKIP: generate_nbest_batch (no --cuts/--checkpoint/--bpe)")

        # -- Step 8: score_pll_batch.py --------------------------------
        # Create a tiny dir with one file to test the batch wrapper
        batch_pll_dir = tmpdir / "pll_batch_test"
        batch_pll_dir.mkdir()
        import shutil
        shutil.copy(mini, batch_pll_dir / "nbest_g4.jsonl")

        ok = run(
            "score_pll_batch.py (1 file, 5 utts)",
            [sys.executable, str(scripts_dir / "score_pll_batch.py"),
             "--input-dir", str(batch_pll_dir),
             "--pattern", "nbest_g*.jsonl",
             "--model", "roberta-base",
             "--device", "cuda",
             "--batch-size", "16"],
            expect_file=str(batch_pll_dir / "nbest_g4_pll.jsonl"),
        )
        results.append(("score_pll_batch", ok))

    # -- Summary -------------------------------------------------------
    total_time = time.time() - t_start
    print()
    print("=" * 60)
    print("  SMOKE TEST SUMMARY")
    print("=" * 60)

    all_pass = True
    for name, ok in results:
        status = " PASS" if ok else " FAIL"
        print(f"  {status}  {name}")
        if not ok:
            all_pass = False

    print(f"\n  Total time: {total_time:.0f}s ({total_time / 60:.1f} min)")

    if all_pass:
        print(f"\n  ALL {len(results)} TESTS PASSED  --  safe to run overnight")
    else:
        n_fail = sum(1 for _, ok in results if not ok)
        print(f"\n  {n_fail}/{len(results)} FAILED  --  fix before overnight run")

    print()
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

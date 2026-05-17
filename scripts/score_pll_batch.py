#!/usr/bin/env python3
"""Run score_pll.py on all matching JSONL files in a directory.

Finds files matching --pattern (without "_pll" suffix), scores each,
and outputs {basename}_pll.jsonl alongside.

Usage:
    python scripts/score_pll_batch.py \
        --input-dir /path/to/dir/ \
        --pattern "nbest_g*.jsonl" \
        [--skip-existing] \
        [--model roberta-base] \
        [--device cuda] \
        [--batch-size 32]
"""

import argparse
import fnmatch
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Batch PLL scoring of N-best JSONL files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory containing N-best JSONL files")
    parser.add_argument("--pattern", type=str, default="nbest_g*.jsonl",
                        help="Glob pattern for input files")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip if _pll output already exists")
    parser.add_argument("--model", type=str, default="roberta-base",
                        help="HuggingFace model name")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    print("=" * 70)
    print("score_pll_batch.py  --  batch PLL scoring")
    print("=" * 70)
    print(f"  input_dir: {args.input_dir}")
    print(f"  pattern:   {args.pattern}")
    print(f"  model:     {args.model}")
    print(f"  device:    {args.device}")
    print()

    all_files = sorted(os.listdir(args.input_dir))
    candidates = []
    for f in all_files:
        if fnmatch.fnmatch(f, args.pattern) and "_pll" not in f:
            candidates.append(f)

    if not candidates:
        print(f"  No files matching '{args.pattern}' (without _pll) found")
        return

    work = []
    for fname in candidates:
        input_path = args.input_dir / fname
        stem = fname.rsplit(".jsonl", 1)[0]
        output_path = args.input_dir / f"{stem}_pll.jsonl"

        if args.skip_existing and output_path.exists() and output_path.stat().st_size > 0:
            print(f"  SKIP {fname}: {output_path.name} exists "
                  f"({output_path.stat().st_size / 1e6:.1f} MB)")
            continue
        work.append((input_path, output_path))

    if not work:
        print("  Nothing to do (all files already scored)")
        return

    print(f"\n  Files to score: {len(work)}")
    for inp, out in work:
        print(f"    {inp.name} -> {out.name}")
    print()

    script = Path(__file__).resolve().parent / "score_pll.py"
    assert script.exists(), f"score_pll.py not found at {script}"

    t_total = time.time()
    completed = []
    failed = []

    for inp, out in work:
        print(f"\n{'=' * 60}")
        print(f"  Scoring: {inp.name}")
        print(f"{'=' * 60}")

        t0 = time.time()
        cmd = [
            sys.executable, str(script),
            "--nbest", str(inp),
            "--output", str(out),
            "--model", args.model,
            "--device", args.device,
            "--batch-size", str(args.batch_size),
        ]

        result = subprocess.run(cmd)
        elapsed = time.time() - t0
        cumulative = time.time() - t_total

        if result.returncode == 0 and out.exists():
            size = out.stat().st_size / 1e6
            print(f"\n  Done: {elapsed / 60:.1f} min, {size:.1f} MB, "
                  f"cumulative {cumulative / 60:.1f} min")
            completed.append(inp.name)
        else:
            print(f"\n  FAILED: {inp.name} (exit code {result.returncode})")
            failed.append(inp.name)

    # GPU memory summary
    try:
        import torch
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"\n  Peak GPU memory (this process): {peak:.2f} GB")
    except ImportError:
        pass

    # Summary
    total_time = time.time() - t_total
    print()
    print("=" * 60)
    print("  BATCH PLL SUMMARY")
    print("=" * 60)
    print(f"  Completed: {len(completed)} files")
    for f in completed:
        print(f"    {f}")
    if failed:
        print(f"  Failed:    {len(failed)} files")
        for f in failed:
            print(f"    {f}")
    print(f"  Total time: {total_time / 60:.1f} min ({total_time / 3600:.1f} h)")

    # List all scored files
    print()
    print("  All PLL-scored files:")
    for f in sorted(os.listdir(args.input_dir)):
        if "_pll.jsonl" in f:
            fp = args.input_dir / f
            print(f"    {f:40s} {fp.stat().st_size / 1e6:>8.1f} MB")
    print()

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

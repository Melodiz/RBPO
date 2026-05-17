#!/usr/bin/env python3
"""Run generate_nbest.py at multiple G values sequentially.

Wrapper for batch N-best generation with crash recovery.
Output files named: nbest_g{G}.jsonl

Usage:
    python scripts/generate_nbest_batch.py \
        --cuts /path/to/cuts.jsonl.gz \
        --checkpoint /path/to/pretrained.pt \
        --bpe /path/to/bpe.model \
        --G-values 4,8,16,32,64,128 \
        --nbest-scale 1.0 \
        --output-dir /path/to/output/ \
        [--icefall-dir /content/icefall] \
        [--skip-existing]
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Batch N-best generation at multiple G values",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cuts", type=Path, required=True,
                        help="CutSet path (jsonl.gz)")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Model checkpoint path")
    parser.add_argument("--bpe", type=Path, required=True,
                        help="BPE model path")
    parser.add_argument("--G-values", type=str, required=True,
                        help="Comma-separated G values (e.g. 4,8,16,32,64,128)")
    parser.add_argument("--nbest-scale", type=float, default=1.0,
                        help="Lattice sampling scale")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for nbest_g{G}.jsonl files")
    parser.add_argument("--icefall-dir", type=Path, default=Path("/content/icefall"),
                        help="Path to icefall installation")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip G values whose output file already exists")
    args = parser.parse_args()

    g_values = [int(g) for g in args.G_values.split(",")]

    print("=" * 70)
    print("generate_nbest_batch.py  --  batch N-best generation")
    print("=" * 70)
    print(f"  cuts:        {args.cuts}")
    print(f"  checkpoint:  {args.checkpoint}")
    print(f"  bpe:         {args.bpe}")
    print(f"  G values:    {g_values}")
    print(f"  nbest_scale: {args.nbest_scale}")
    print(f"  output_dir:  {args.output_dir}")
    print(f"  skip_exist:  {args.skip_existing}")
    print()

    assert args.nbest_scale != 0.5, "nbest_scale=0.5 destroys oracle gap  --  use 1.0"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    script = Path(__file__).resolve().parent / "generate_nbest.py"
    assert script.exists(), f"generate_nbest.py not found at {script}"

    t_total = time.time()
    completed = []
    skipped = []
    failed = []

    for g in g_values:
        output_file = args.output_dir / f"nbest_g{g}.jsonl"

        if args.skip_existing and output_file.exists() and output_file.stat().st_size > 0:
            print(f"  SKIP G={g}: {output_file} exists "
                  f"({output_file.stat().st_size / 1e6:.1f} MB)")
            skipped.append(g)
            continue

        print(f"\n{'=' * 60}")
        print(f"  G={g}")
        print(f"{'=' * 60}")

        t0 = time.time()
        cmd = [
            sys.executable, str(script),
            "--cuts", str(args.cuts),
            "--checkpoint", str(args.checkpoint),
            "--bpe", str(args.bpe),
            "--icefall-dir", str(args.icefall_dir),
            "--G", str(g),
            "--nbest-scale", str(args.nbest_scale),
            "--output", str(output_file),
        ]

        result = subprocess.run(cmd)
        elapsed = time.time() - t0
        cumulative = time.time() - t_total

        if result.returncode == 0:
            size = output_file.stat().st_size / 1e6 if output_file.exists() else 0
            print(f"\n  G={g} done: {elapsed:.0f}s ({elapsed / 60:.1f} min), "
                  f"{size:.1f} MB, cumulative {cumulative / 60:.1f} min")
            completed.append(g)
        else:
            print(f"\n  G={g} FAILED (exit code {result.returncode})")
            failed.append(g)

    # Summary
    total_time = time.time() - t_total
    print()
    print("=" * 60)
    print("  BATCH SUMMARY")
    print("=" * 60)
    print(f"  Completed: {completed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Failed:    {failed}")
    print(f"  Total time: {total_time / 60:.1f} min")

    if args.skip_existing:
        print()
        print("  Output files:")
        for g in sorted(set(completed + skipped)):
            f = args.output_dir / f"nbest_g{g}.jsonl"
            if f.exists():
                print(f"    {f.name:30s} {f.stat().st_size / 1e6:>8.1f} MB")
    print()

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

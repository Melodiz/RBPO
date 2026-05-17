#!/usr/bin/env python3
"""E23 prep: download TED-LIUM 3 and compute lhotse fbank features for
dev + test (legacy splits  --  pre-segmented utterances suitable for CTC).

Uses the SAME lhotse Fbank(FbankConfig(num_mel_bins=80)) extractor as
setup_colab.sh / prepare_train_clean_100.py  --  required for feature
compatibility with the Zipformer-S CR-CTC checkpoint.

Idempotent: skips download / feature compute if outputs exist.

Usage:
    python prepare_tedlium3.py \\
        --audio-dir /content/tedlium_audio \\
        --data-dir /content/tedlium_data \\
        --num-jobs 4

Expected runtime on Colab T4:
- Download: 30-60 min (~25 GB tarball from OpenSLR-51)
- Manifests: ~3-5 min
- Feature compute (dev + test only): ~5-15 min total
- Total: ~40-80 min one-time
"""

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

# Suppress lhotse INFO logs and tqdm progress bars in Colab output
os.environ["TQDM_DISABLE"] = "1"
logging.disable(logging.INFO)
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path,
                        default=Path("/content/tedlium_audio"))
    parser.add_argument("--data-dir", type=Path,
                        default=Path("/content/tedlium_data"))
    parser.add_argument("--num-jobs", type=int, default=4)
    parser.add_argument("--parts", nargs="+", default=["test", "dev"],
                        help="Subsets to prepare. Default: test+dev (zero-shot eval).")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip lhotse's download step (use if you've "
                             "manually wget'd or extracted the corpus already).")
    args = parser.parse_args()

    args.audio_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    (args.data_dir / "cuts").mkdir(exist_ok=True)
    (args.data_dir / "manifests").mkdir(exist_ok=True)

    # Skip-all check: if every requested part already has a cuts file, exit
    all_done = all(
        (args.data_dir / "cuts" / f"tedlium3_cuts_{p}.jsonl.gz").exists()
        for p in args.parts
    )
    if all_done:
        print(f" All requested splits already prepared: {args.parts}")
        for p in args.parts:
            cp = args.data_dir / "cuts" / f"tedlium3_cuts_{p}.jsonl.gz"
            print(f"   {cp.name}: {cp.stat().st_size / 1e6:.0f} MB")
        return

    # -- Step 1: Download (~25 GB tarball) -------------------------------
    libri_root = args.audio_dir / "TEDLIUM_release-3"
    if args.skip_download:
        if not libri_root.exists():
            print(f" FATAL: --skip-download set but {libri_root} not found.")
            print(f"  Manually download via wget first. See E23 README.")
            sys.exit(1)
        print(f" Skipping download (--skip-download). Using {libri_root}")
    elif not libri_root.exists() or not (libri_root / "legacy").exists():
        print(f"Step 1: Downloading TED-LIUM 3 to {args.audio_dir}...")
        print("  ~25 GB tarball, ~30-60 min on Colab")
        t0 = time.time()
        try:
            from lhotse.recipes.tedlium import download_tedlium
            download_tedlium(args.audio_dir)
        except Exception as e:
            print(f"\n lhotse download failed: {e}")
            print("\nFallback: download manually via wget, then re-run with --skip-download:")
            print(f"  mkdir -p {args.audio_dir}")
            print(f"  cd {args.audio_dir}")
            print("  wget -c https://www.openslr.org/resources/51/TEDLIUM_release-3.tgz")
            print("  tar xzf TEDLIUM_release-3.tgz")
            print("\nIf OpenSLR is down, try the HuggingFace fallback in the README.")
            sys.exit(1)
        print(f"   Downloaded in {(time.time()-t0)/60:.1f} min")
    else:
        print(f" Audio already downloaded: {libri_root}")

    # -- Step 2: Prepare manifests ---------------------------------------
    print(f"Step 2: Preparing manifests for {args.parts}...")
    t0 = time.time()
    from lhotse.recipes.tedlium import prepare_tedlium
    manifests = prepare_tedlium(
        tedlium_root=libri_root,
        output_dir=args.data_dir / "manifests",
        dataset_parts=args.parts,
        num_jobs=args.num_jobs,
    )
    print(f"   Manifests in {(time.time()-t0)/60:.1f} min")
    for p in args.parts:
        if p in manifests:
            n = len(manifests[p].get("supervisions", []))
            print(f"    {p}: {n} supervisions")

    # -- Step 3: Compute fbank features per split ------------------------
    from lhotse import CutSet, Fbank, FbankConfig
    fbank = Fbank(FbankConfig(num_mel_bins=80))

    for part in args.parts:
        cuts_path = args.data_dir / "cuts" / f"tedlium3_cuts_{part}.jsonl.gz"
        if cuts_path.exists():
            print(f"   {part}: features already exist ({cuts_path.name}, "
                  f"{cuts_path.stat().st_size / 1e6:.0f} MB)")
            continue

        print(f"Step 3 [{part}]: Computing features...")
        t0 = time.time()
        cuts = CutSet.from_manifests(
            recordings=manifests[part]["recordings"],
            supervisions=manifests[part]["supervisions"],
        )
        # trim_to_supervisions: one cut per supervision (utterance)
        # Required for our CTC pipeline  --  same as LibriSpeech prep
        cuts = cuts.trim_to_supervisions()
        n_utts = len(cuts)
        print(f"  {n_utts} utterances after trim_to_supervisions")

        cuts = cuts.compute_and_store_features(
            extractor=fbank,
            storage_path=str(args.data_dir / "cuts" / f"feats_tedlium3_{part}"),
            num_jobs=args.num_jobs,
        )
        cuts.to_file(str(cuts_path))
        elapsed = time.time() - t0
        print(f"   {part}: {len(cuts)} cuts in {elapsed/60:.1f} min "
              f"-> {cuts_path.name} ({cuts_path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()

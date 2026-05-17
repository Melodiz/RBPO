#!/usr/bin/env python3
"""E21 prep: download train-clean-100 audio and compute lhotse fbank features.

Uses the SAME lhotse Fbank(FbankConfig(num_mel_bins=80)) extractor as
setup_colab.sh  --  required for feature compatibility with the model.

Idempotent: skips download + computation if outputs exist.

Usage:
    python prepare_train_clean_100.py \\
        --audio-dir /content/librispeech_audio \\
        --data-dir /content/librispeech_data \\
        --num-jobs 4

Expected runtime on Colab T4:
- Download: ~10 min (6.3 GB)
- Manifests: ~2 min
- Feature computation: ~30-60 min for 100 hours of audio
"""

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

# Suppress tqdm progress bars + verbose lhotse logging (matches setup_colab.sh)
os.environ["TQDM_DISABLE"] = "1"
logging.disable(logging.INFO)
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path, default=Path("/content/librispeech_audio"))
    parser.add_argument("--data-dir", type=Path, default=Path("/content/librispeech_data"))
    parser.add_argument("--num-jobs", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0,
                        help="If >0, compute features for only first N utterances "
                             "and save with -smoke suffix (for fast smoke testing). "
                             "Audio download is always full (single 6GB tarball).")
    args = parser.parse_args()

    suffix = "-smoke" if args.limit > 0 else ""
    audio_dir = args.audio_dir
    data_dir = args.data_dir
    cuts_path = data_dir / "cuts" / f"librispeech_cuts_train-clean-100{suffix}.jsonl.gz"
    manifests_dir = data_dir / "manifests"
    cuts_dir = data_dir / "cuts"
    feats_dir = cuts_dir / f"feats_train-clean-100{suffix}"

    cuts_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    if cuts_path.exists() and cuts_path.stat().st_size > 0:
        print(f" Features already exist: {cuts_path}")
        print(f"  Size: {cuts_path.stat().st_size / 1e6:.0f} MB")
        return

    # -- Step 1: Download audio (~6 GB) ------------------------------
    libri_root = audio_dir / "LibriSpeech" / "train-clean-100"
    if not libri_root.exists():
        print(f"Step 1: Downloading train-clean-100 to {audio_dir}...")
        t0 = time.time()
        from lhotse.recipes.librispeech import download_librispeech
        download_librispeech(audio_dir, dataset_parts=["train-clean-100"])
        print(f"   Downloaded in {(time.time()-t0)/60:.1f} min")
    else:
        print(f" Audio already downloaded: {libri_root}")

    # -- Step 2: Prepare manifests -----------------------------------
    rec_path = manifests_dir / "librispeech_recordings_train-clean-100.jsonl.gz"
    if not rec_path.exists():
        print("Step 2: Preparing manifests...")
        t0 = time.time()
        from lhotse.recipes.librispeech import prepare_librispeech
        prepare_librispeech(
            corpus_dir=audio_dir / "LibriSpeech",
            dataset_parts=["train-clean-100"],
            output_dir=manifests_dir,
            num_jobs=args.num_jobs,
        )
        print(f"   Manifests prepared in {(time.time()-t0)/60:.1f} min")
    else:
        print(f" Manifests already prepared")

    # -- Step 3: Compute fbank features (SAME as setup_colab.sh) -----
    if args.limit > 0:
        print(f"Step 3: Computing fbank features for first {args.limit} utts "
              f"(num_jobs={args.num_jobs})...")
        print(f"  Output suffix: '-smoke'  (fast smoke testing only)")
    else:
        print(f"Step 3: Computing fbank features (num_jobs={args.num_jobs})...")
        print("  This takes ~30-60 minutes for 100 hours of audio.")
    t0 = time.time()

    from lhotse import CutSet, Fbank, FbankConfig, load_manifest

    recordings = load_manifest(manifests_dir / "librispeech_recordings_train-clean-100.jsonl.gz")
    supervisions = load_manifest(manifests_dir / "librispeech_supervisions_train-clean-100.jsonl.gz")
    cuts = CutSet.from_manifests(recordings=recordings, supervisions=supervisions)

    # Trim to supervisions: each cut becomes one supervision span (utterance).
    cuts = cuts.trim_to_supervisions()

    if args.limit > 0:
        # Subset to first N utterances BEFORE feature computation
        cuts = cuts.subset(first=args.limit)
        print(f"  Subset to {len(cuts)} cuts")

    fbank = Fbank(FbankConfig(num_mel_bins=80))
    cuts = cuts.compute_and_store_features(
        extractor=fbank,
        storage_path=str(feats_dir),
        num_jobs=args.num_jobs,
    )
    cuts.to_file(str(cuts_path))

    elapsed = time.time() - t0
    print(f"   Features computed in {elapsed/60:.1f} min")
    print(f"   {len(cuts)} cuts saved to {cuts_path}")
    print(f"  Cuts file size: {cuts_path.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()

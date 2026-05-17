#!/usr/bin/env python3
"""E23 prep (HuggingFace fallback): TED-LIUM 3 dev+test from HF datasets.

Use this when OpenSLR-51 is down (404 on TEDLIUM_release-3.tgz).
HF only downloads the splits you ask for, so this is much smaller (~2 GB
total for dev+test vs 25 GB for the lhotse path).

Outputs lhotse cuts files in the same path/naming as prepare_tedlium3.py:
  tedlium3_cuts_test.jsonl.gz
  tedlium3_cuts_dev.jsonl.gz

So all downstream scripts (generate_reranker_data, score_neural_lm,
mbr_eval) work unchanged.

Usage:
    pip install -q datasets soundfile
    python prepare_tedlium3_hf.py \\
        --audio-dir /content/tedlium_audio_hf \\
        --data-dir /content/tedlium_data \\
        --num-jobs 4
"""

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

# Suppress lhotse INFO + tqdm
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
# Redirect HF cache to /content/ (Colab's larger partition).
# /root/.cache/huggingface fills the system disk on big audio datasets.
if "HF_DATASETS_CACHE" not in os.environ:
    os.environ["HF_DATASETS_CACHE"] = "/content/hf_cache"
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "/content/hf_cache"
os.makedirs("/content/hf_cache", exist_ok=True)
logging.disable(logging.INFO)
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


# HF "validation" = TED-LIUM "dev" in our naming
HF_TO_LHOTSE = {"test": "test", "validation": "dev"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path,
                        default=Path("/content/tedlium_audio_hf"))
    parser.add_argument("--data-dir", type=Path,
                        default=Path("/content/tedlium_data"))
    parser.add_argument("--num-jobs", type=int, default=4)
    parser.add_argument("--hf-dataset", default="LIUM/tedlium",
                        help="HF dataset path (e.g. LIUM/tedlium, "
                             "distil-whisper/tedlium-long-form, "
                             "mozilla-foundation/common_voice_17_0)")
    parser.add_argument("--hf-config", default="release3",
                        help="HF dataset config name. Examples: "
                             "release3 (LIUM/tedlium); en (common_voice).")
    parser.add_argument("--hf-splits", nargs="+",
                        default=["test", "validation"],
                        help="HF split names to download. "
                             "validation->dev, test->test in our naming.")
    parser.add_argument("--text-field", default="text",
                        help="Field name holding the transcript "
                             "(text, sentence, transcription, ...)")
    parser.add_argument("--max-utts-per-split", type=int, default=0,
                        help="If >0, subset to first N utterances per split "
                             "(useful for huge datasets like Common Voice).")
    parser.add_argument("--cuts-prefix", default="tedlium3_cuts",
                        help="Prefix for output cuts files. e.g. "
                             "voxpopuli_cuts -> voxpopuli_cuts_test.jsonl.gz")
    parser.add_argument("--feats-prefix", default="feats_tedlium3",
                        help="Prefix for feature storage dirs.")
    parser.add_argument("--streaming", action="store_true",
                        help="Stream HF dataset (no Arrow cache). Saves ~50-100 "
                             "GB peak disk for big audio sets like VoxPopuli "
                             "en. Sequential only  --  no len() / no ds[i].")
    args = parser.parse_args()

    args.audio_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    (args.data_dir / "cuts").mkdir(exist_ok=True)

    # Skip-all check
    all_done = all(
        (args.data_dir / "cuts" / f"{args.cuts_prefix}_{HF_TO_LHOTSE[s]}.jsonl.gz").exists()
        for s in args.hf_splits
    )
    if all_done:
        print(" All requested splits already prepared:")
        for s in args.hf_splits:
            cp = args.data_dir / "cuts" / f"{args.cuts_prefix}_{HF_TO_LHOTSE[s]}.jsonl.gz"
            print(f"   {cp.name}: {cp.stat().st_size / 1e6:.0f} MB")
        return

    # -- Imports ---------------------------------------------------------
    print("Loading dependencies (datasets, soundfile, lhotse)...")
    import numpy as np
    import soundfile as sf
    from datasets import load_dataset, Audio
    from lhotse import (
        CutSet, Fbank, FbankConfig,
        Recording, RecordingSet, SupervisionSegment, SupervisionSet,
    )

    fbank = Fbank(FbankConfig(num_mel_bins=80))

    for hf_split in args.hf_splits:
        lhotse_split = HF_TO_LHOTSE[hf_split]
        cuts_path = args.data_dir / "cuts" / f"tedlium3_cuts_{lhotse_split}.jsonl.gz"
        if cuts_path.exists():
            print(f" {lhotse_split}: cuts already exist")
            continue

        print(f"\n=== HF split '{hf_split}' -> lhotse '{lhotse_split}' ===")

        # -- Step 1: Load HF dataset (this downloads if not cached) ------
        print(f"  Loading HF dataset {args.hf_dataset} "
              f"config={args.hf_config} split={hf_split} "
              f"(streaming={args.streaming})...")
        t0 = time.time()
        ds = load_dataset(
            args.hf_dataset, args.hf_config,
            split=hf_split, streaming=args.streaming,
        )
        # Force 16 kHz (the model was trained at 16 kHz)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        if args.streaming:
            if args.max_utts_per_split > 0:
                ds = ds.take(args.max_utts_per_split)
            print(f"  Streaming mode  --  total count unknown until iteration ends")
            print(f"  (~{(time.time()-t0):.1f}s to start streaming)")
        else:
            if args.max_utts_per_split > 0 and len(ds) > args.max_utts_per_split:
                ds = ds.select(range(args.max_utts_per_split))
            print(f"  Loaded {len(ds)} utterances in {(time.time()-t0)/60:.1f} min")
            print(f"  First record fields: {list(ds[0].keys())}")

        # -- Step 2: Save audio + build lhotse manifests -----------------
        print(f"  Saving WAV files + building manifests...")
        t0 = time.time()
        wav_dir = args.audio_dir / lhotse_split
        wav_dir.mkdir(parents=True, exist_ok=True)

        recordings = []
        supervisions = []
        n_skipped = 0
        for i, ex in enumerate(ds):
            audio = ex["audio"]["array"]
            sr = ex["audio"]["sampling_rate"]
            # Try the configured text field, else common alternatives
            text = ex.get(args.text_field) or ex.get("text") or \
                   ex.get("sentence") or ex.get("transcription") or \
                   ex.get("transcript") or ""
            text = str(text).strip()
            if not text or len(audio) < 1600:  # < 0.1s or empty text
                n_skipped += 1
                continue

            # ID from HF if available; else synthesize
            uid = ex.get("id") or ex.get("file") or f"{lhotse_split}_{i:06d}"
            uid = str(uid).replace("/", "_").replace(" ", "_")

            wav_path = wav_dir / f"{uid}.wav"
            if not wav_path.exists():
                # Convert to int16 if float (sf handles both)
                sf.write(str(wav_path), audio.astype(np.float32), sr)

            rec = Recording.from_file(wav_path)
            recordings.append(rec)
            sup = SupervisionSegment(
                id=uid,
                recording_id=rec.id,
                start=0.0,
                duration=rec.duration,
                channel=0,
                text=text.lower(),
                language="English",
            )
            supervisions.append(sup)

            if (i + 1) % 200 == 0:
                if args.streaming:
                    print(f"    {i+1} written (streaming)")
                else:
                    print(f"    {i+1}/{len(ds)} written")

        rec_set = RecordingSet.from_recordings(recordings)
        sup_set = SupervisionSet.from_segments(supervisions)
        print(f"   Built {len(rec_set)} recordings, "
              f"{len(sup_set)} supervisions  "
              f"({n_skipped} skipped)  in {(time.time()-t0)/60:.1f} min")

        # -- Step 3: Compute features ------------------------------------
        print(f"  Computing fbank features (num_jobs={args.num_jobs})...")
        t0 = time.time()
        cuts = CutSet.from_manifests(recordings=rec_set, supervisions=sup_set)
        cuts = cuts.trim_to_supervisions()  # cut == supervision span
        cuts = cuts.compute_and_store_features(
            extractor=fbank,
            storage_path=str(args.data_dir / "cuts" / f"{args.feats_prefix}_{lhotse_split}"),
            num_jobs=args.num_jobs,
        )
        cuts.to_file(str(cuts_path))
        elapsed = time.time() - t0
        print(f"   {lhotse_split}: {len(cuts)} cuts in {elapsed/60:.1f} min "
              f"-> {cuts_path.name} ({cuts_path.stat().st_size / 1e6:.0f} MB)")

    # -- Final summary ---------------------------------------------------
    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)
    for s in args.hf_splits:
        cp = args.data_dir / "cuts" / f"{args.cuts_prefix}_{HF_TO_LHOTSE[s]}.jsonl.gz"
        if cp.exists():
            print(f"  {cp.name}: {cp.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()

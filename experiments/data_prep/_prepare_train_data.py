#!/usr/bin/env python3
"""Prepares manifests, optionally subsets, and
computes fbank features for LibriSpeech train-clean-100.

Lives in a real .py file (not a heredoc) so lhotse's multiprocessing can
re-import __main__ cleanly when needed. Stick to num_jobs=1 anyway for
robustness on Colab Python 3.12 + spawn context.

Can be invoked directly or via Colab notebooks.
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--subset", type=int, default=0,
                   help="0 = full set; N = first N cuts sorted by duration")
    args = p.parse_args()

    # Silence lhotse's tqdm to avoid newline spam in piped/teed shells.
    os.environ.setdefault("TQDM_DISABLE", "1")

    from lhotse import CutSet, Fbank, FbankConfig, load_manifest
    from lhotse.recipes.librispeech import (
        download_librispeech, prepare_librispeech,
    )

    corpus_dir = args.data_dir
    manifest_dir = corpus_dir / "manifests"
    cuts_dir = corpus_dir / "cuts"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    cuts_dir.mkdir(parents=True, exist_ok=True)

    split = "train-clean-100"

    if not (corpus_dir / "LibriSpeech" / split).exists():
        print(f"[1/3] Downloading {split} (~6.3 GB tar)...", flush=True)
        download_librispeech(
            target_dir=str(corpus_dir),
            dataset_parts=[split],
        )
    else:
        print(f"[1/3] {split} audio already present, skipping download", flush=True)

    print(f"[2/3] Preparing manifests for {split}...", flush=True)
    prepare_librispeech(
        corpus_dir=corpus_dir / "LibriSpeech",
        dataset_parts=[split],
        output_dir=str(manifest_dir),
        num_jobs=1,
    )

    print(f"[3/3] Building cuts and computing fbank features...", flush=True)
    recordings = load_manifest(
        manifest_dir / f"librispeech_recordings_{split}.jsonl.gz"
    )
    supervisions = load_manifest(
        manifest_dir / f"librispeech_supervisions_{split}.jsonl.gz"
    )
    cuts = CutSet.from_manifests(
        recordings=recordings, supervisions=supervisions
    )

    if args.subset:
        sorted_cuts = sorted(cuts, key=lambda c: c.duration)
        cuts = CutSet.from_cuts(sorted_cuts[: args.subset])
        print(
            f"      Subset to first {args.subset} cuts (sorted by duration)",
            flush=True,
        )
    else:
        print(f"      Using full {split} ({len(cuts)} cuts)", flush=True)

    fbank = Fbank(FbankConfig(num_mel_bins=80))
    storage = cuts_dir / f"feats_{split}"
    output_jsonl = cuts_dir / f"librispeech_cuts_{split}.jsonl.gz"

    print(f"      Computing fbank -> {storage} (num_jobs=1, may take a while)",
          flush=True)
    cuts_with_feats = cuts.compute_and_store_features(
        extractor=fbank,
        storage_path=str(storage),
        num_jobs=1,
    )
    cuts_with_feats.to_file(str(output_jsonl))
    print(
        f"      done  --  {len(cuts_with_feats)} cuts saved to {output_jsonl}",
        flush=True,
    )


if __name__ == "__main__":
    main()

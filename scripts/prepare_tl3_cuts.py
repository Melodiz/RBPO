#!/usr/bin/env python3
"""Download and prepare TED-LIUM 3 test set as a lhotse CutSet.

Handles TL3-specific issues:
  - Downloads from HuggingFace mirror (OpenSLR-51 is dead)
  - Strips {NOISE}, {BREATH}, <unk> tags from supervision text
  - SPH files need sph2pipe (installed via lhotse)
  - Channel forced to 0 (SPH is mono despite STM saying channel 1)

Usage:
    python scripts/prepare_tl3_cuts.py \
        --output-dir /path/to/output \
        [--hf-repo kfajdsl/tedlium]
"""

import argparse
import glob
import os
import re
import subprocess
import sys
from pathlib import Path


_TAG_RE = re.compile(r"\{[^}]+\}|<[^>]+>")
_MULTI_SPACE = re.compile(r"\s+")


def clean_tl3_text(text):
    text = _TAG_RE.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


def parse_stm(stm_path: str):
    """Parse STM file -> list of (file_id, start, end, text)."""
    segments = []
    with open(stm_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";;"):
                continue
            parts = line.split(None, 6)
            if len(parts) < 7:
                continue
            file_id = parts[0]
            start = float(parts[3])
            end = float(parts[4])
            text = clean_tl3_text(parts[6])
            if text and text.lower() != "ignore_time_segment_in_scoring":
                segments.append((file_id, start, end, text))
    return segments


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare TED-LIUM 3 test CutSet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for downloaded data and output CutSet")
    parser.add_argument("--hf-repo", type=str, default="kfajdsl/tedlium",
                        help="HuggingFace dataset repo for TL3")
    args = parser.parse_args()

    print("=" * 70)
    print("prepare_tl3_cuts.py  --  TED-LIUM 3 test set")
    print("=" * 70)
    print(f"  output_dir: {args.output_dir}")
    print(f"  hf_repo:    {args.hf_repo}")
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.output_dir / "raw"
    data_dir.mkdir(exist_ok=True)

    # Download test-only tarball from HuggingFace
    test_tar = data_dir / "test.tar.gz"
    if not test_tar.exists():
        url = f"https://huggingface.co/datasets/{args.hf_repo}/resolve/main/TEDLIUM_release3/legacy/test.tar.gz"
        print(f"Downloading TL3 test set from {url}...")
        subprocess.run(
            ["wget", "-q", "--show-progress", "-L", "-O", str(test_tar), url],
            check=True,
        )
    else:
        print(f"Already downloaded: {test_tar}")

    # Extract (test.tar.gz extracts to test/ with sph/ and stm/ subdirs)
    extracted_marker = data_dir / ".extracted"
    if not extracted_marker.exists():
        print("Extracting...")
        subprocess.run(["tar", "xzf", str(test_tar), "-C", str(data_dir)], check=True)
        extracted_marker.touch()
    else:
        print("Already extracted")

    sph_files = sorted(glob.glob(str(data_dir / "**/*.sph"), recursive=True))
    stm_files = sorted(glob.glob(str(data_dir / "**/*.stm"), recursive=True))
    print(f"  SPH files: {len(sph_files)}")
    print(f"  STM files: {len(stm_files)}")

    test_sph = [f for f in sph_files if "/test/" in f]
    test_stm = [f for f in stm_files if "/test/" in f]
    if not test_sph:
        test_sph = sph_files
        test_stm = stm_files
    print(f"  Test SPH: {len(test_sph)}")
    print(f"  Test STM: {len(test_stm)}")

    assert test_sph, "No SPH files found"
    assert test_stm, "No STM files found"

    # Install sph2pipe if needed
    try:
        subprocess.run(["sph2pipe", "-h"], capture_output=True)
    except FileNotFoundError:
        print("Installing sph2pipe...")
        subprocess.run(
            [sys.executable, "-m", "lhotse", "install-sph2pipe"],
            check=True,
        )

    from lhotse import CutSet, Recording, SupervisionSegment

    # Map file_id -> sph_path
    sph_map = {}
    for sph in test_sph:
        fid = Path(sph).stem
        sph_map[fid] = sph

    all_segments = []
    for stm in test_stm:
        all_segments.extend(parse_stm(stm))

    print(f"  Total STM segments: {len(all_segments)}")

    recordings = {}
    supervisions = []

    for file_id, start, end, text in all_segments:
        if file_id not in sph_map:
            continue

        if file_id not in recordings:
            recordings[file_id] = Recording.from_file(
                sph_map[file_id], recording_id=file_id
            )

        sup_id = f"{file_id}-{start:.3f}-{end:.3f}".replace(".", "_")
        supervisions.append(SupervisionSegment(
            id=sup_id,
            recording_id=file_id,
            start=start,
            duration=end - start,
            channel=0,
            text=text,
        ))

    recordings_list = list(recordings.values())
    print(f"  Recordings: {len(recordings_list)}")
    print(f"  Supervisions: {len(supervisions)}")

    from lhotse import SupervisionSet, RecordingSet
    rec_set = RecordingSet.from_recordings(recordings_list)
    sup_set = SupervisionSet.from_segments(supervisions)
    cuts = CutSet.from_manifests(recordings=rec_set, supervisions=sup_set)
    cuts = cuts.trim_to_supervisions().to_eager()

    cuts_filtered = []
    for cut in cuts:
        text = " ".join(s.text for s in cut.supervisions if s.text).strip()
        if text and text.lower() != "ignore_time_segment_in_scoring":
            cuts_filtered.append(cut)
    cuts = CutSet.from_cuts(cuts_filtered)

    # Save
    cuts_path = args.output_dir / "cuts_tl3_test.jsonl.gz"
    cuts.to_file(str(cuts_path))

    # Summary
    total_dur = sum(c.duration for c in cuts)
    print()
    print(f"  CutSet saved: {cuts_path}")
    print(f"  Utterances:   {len(cuts)}")
    print(f"  Duration:     {total_dur / 3600:.2f} hours")
    print(f"  Mean length:  {total_dur / len(cuts):.1f}s")
    print()

    # Sample transcripts
    print("  Sample transcripts:")
    for cut in list(cuts)[:3]:
        text = " ".join(s.text for s in cut.supervisions)
        print(f"    [{cut.id}] {text[:80]}")
    print()


if __name__ == "__main__":
    main()

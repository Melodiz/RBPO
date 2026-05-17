#!/usr/bin/env python3
"""Create noise-augmented LibriSpeech dev-other CutSet.

Loads clean LS dev-other CutSet, applies additive MUSAN noise at a target SNR,
and saves the augmented audio as a new CutSet with on-the-fly mixing.

For generate_nbest.py: the augmented CutSet produces noisy fbank features
when load_audio() is called during on-the-fly feature extraction.

Usage:
    python scripts/prepare_musan_cuts.py \
        --ls-cuts /path/to/ls_devother_cuts.jsonl.gz \
        --musan-dir /path/to/musan/noise \
        --snr 0 \
        --output /path/to/cuts_0dB.jsonl.gz \
        [--seed 42]
"""

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Create noise-augmented LS dev-other CutSet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ls-cuts", type=Path, required=True,
                        help="Clean LS dev-other CutSet (jsonl.gz)")
    parser.add_argument("--musan-dir", type=Path, required=True,
                        help="Path to MUSAN noise directory (musan/noise/)")
    parser.add_argument("--snr", type=float, default=0.0,
                        help="Target SNR in dB")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output CutSet path (jsonl.gz)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download-musan", action="store_true",
                        help="Download MUSAN if --musan-dir doesn't exist")
    args = parser.parse_args()

    print("=" * 70)
    print("prepare_musan_cuts.py  --  noise-augmented CutSet")
    print("=" * 70)
    print(f"  ls_cuts:    {args.ls_cuts}")
    print(f"  musan_dir:  {args.musan_dir}")
    print(f"  snr:        {args.snr} dB")
    print(f"  output:     {args.output}")
    print(f"  seed:       {args.seed}")
    print()

    import torch
    import torchaudio
    from lhotse import CutSet, Recording, SupervisionSegment, load_manifest_lazy

    # Download MUSAN if needed
    noise_dir = args.musan_dir
    if not noise_dir.exists() and args.download_musan:
        print("Downloading MUSAN...")
        musan_tar = noise_dir.parent / "musan.tar.gz"
        subprocess.run([
            "wget", "-q", "--show-progress", "-O", str(musan_tar),
            "https://www.openslr.org/resources/17/musan.tar.gz",
        ], check=True)
        print("Extracting noise subset...")
        subprocess.run([
            "tar", "xzf", str(musan_tar), "-C", str(noise_dir.parent),
            "musan/noise/",
        ], check=True)
        os.remove(musan_tar)

    noise_files = sorted(glob.glob(str(noise_dir / "**/*.wav"), recursive=True))
    if not noise_files:
        noise_files = sorted(glob.glob(
            str(noise_dir.parent / "musan" / "noise" / "**/*.wav"), recursive=True
        ))
    assert noise_files, f"No noise WAV files found in {noise_dir}"
    print(f"  Noise files: {len(noise_files)}")

    print("Loading clean CutSet...")
    clean_cuts = list(load_manifest_lazy(str(args.ls_cuts)))
    print(f"  Clean utterances: {len(clean_cuts)}")

    rng = np.random.default_rng(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    augmented_wav_dir = args.output.parent / f"augmented_wav_{args.snr}dB"
    augmented_wav_dir.mkdir(exist_ok=True)

    new_recordings = {}
    new_cuts_data = []
    actual_snrs = []

    for i, cut in enumerate(clean_cuts):
        audio = cut.load_audio()  # (1, T)
        T = audio.shape[-1]

        noise_path = rng.choice(noise_files)
        noise_wav, sr = torchaudio.load(noise_path)
        if sr != 16000:
            noise_wav = torchaudio.functional.resample(noise_wav, sr, 16000)
        noise = noise_wav.numpy()
        if noise.shape[0] > 1:
            noise = noise[:1]

        # Loop if shorter
        if noise.shape[-1] < T:
            reps = T // noise.shape[-1] + 1
            noise = np.tile(noise, (1, reps))

        # Random crop
        if noise.shape[-1] > T:
            start = rng.integers(0, noise.shape[-1] - T)
            noise = noise[:, start:start + T]

        # Scale to target SNR
        sig_power = np.mean(audio ** 2)
        noise_power = np.mean(noise ** 2)
        if noise_power < 1e-10 or sig_power < 1e-10:
            noisy = audio
            actual_snr = float("inf")
        else:
            target_noise_power = sig_power / (10 ** (args.snr / 10))
            scale = np.sqrt(target_noise_power / noise_power)
            noisy = audio + scale * noise
            actual_snr = 10 * np.log10(sig_power / (np.mean((scale * noise) ** 2) + 1e-10))

        actual_snrs.append(actual_snr)

        wav_path = augmented_wav_dir / f"{cut.id}.wav"
        torchaudio.save(
            str(wav_path),
            torch.from_numpy(noisy).float(),
            sample_rate=16000,
        )

        rec = Recording.from_file(str(wav_path), recording_id=cut.id)
        new_recordings[cut.id] = rec

        # Copy supervisions
        for sup in cut.supervisions:
            new_cuts_data.append({
                "recording": rec,
                "supervision": sup,
            })

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(clean_cuts)} augmented")

    from lhotse import RecordingSet, SupervisionSet
    rec_set = RecordingSet.from_recordings(list(new_recordings.values()))

    # Re-attach supervisions
    new_sups = []
    for cut in clean_cuts:
        for sup in cut.supervisions:
            new_sups.append(SupervisionSegment(
                id=sup.id,
                recording_id=cut.id,
                start=sup.start,
                duration=sup.duration,
                channel=sup.channel,
                text=sup.text,
            ))

    sup_set = SupervisionSet.from_segments(new_sups)
    new_cuts = CutSet.from_manifests(recordings=rec_set, supervisions=sup_set)
    new_cuts = new_cuts.trim_to_supervisions().to_eager()

    new_cuts.to_file(str(args.output))

    actual_snrs_finite = [s for s in actual_snrs if np.isfinite(s)]
    mean_snr = np.mean(actual_snrs_finite) if actual_snrs_finite else float("nan")

    print()
    print(f"  Augmented CutSet saved: {args.output}")
    print(f"  Utterances: {len(new_cuts)}")
    print(f"  Target SNR: {args.snr} dB")
    print(f"  Actual SNR: {mean_snr:.1f} dB (mean)")
    print(f"  WAV dir:    {augmented_wav_dir}")

    # Sample verification
    sample = new_cuts[0] if len(new_cuts) > 0 else None
    if sample:
        audio = sample.load_audio()
        print(f"  Sample: {sample.id}, shape={audio.shape}, "
              f"power={np.mean(audio**2):.6f}")
    print()


if __name__ == "__main__":
    main()

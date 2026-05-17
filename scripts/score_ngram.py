#!/usr/bin/env python3
"""Score N-best hypotheses with an n-gram language model.

Uses kenlm for ARPA model scoring. If no LM path is provided,
downloads the LibriSpeech 3-gram from OpenSLR.

Usage:
    python scripts/score_ngram.py \
        --nbest /path/to/nbest.jsonl \
        --output /path/to/nbest_ngram.jsonl \
        [--lm-path /path/to/3-gram.arpa] \
        [--order 3]
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def _normalize_record(raw):
    if "nbest" in raw:
        return raw
    return {
        "utt_id": raw["utt_id"],
        "ref": raw.get("ref", raw.get("ref_text", "")),
        "nbest": [
            {"hyp": c.get("hyp", c.get("text", "")),
             "score": c.get("score", c.get("ctc_log_prob", 0.0)),
             **{k: v for k, v in c.items()
                if k not in ("hyp", "text", "score", "ctc_log_prob")}}
            for c in raw.get("nbest", raw.get("candidates", []))
        ],
    }


def load_nbest(path: Path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(_normalize_record(json.loads(line)))
    return records


def save_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def download_ls_3gram(target_dir: Path):
    """Download LibriSpeech pruned 3-gram ARPA from OpenSLR."""
    target_dir.mkdir(parents=True, exist_ok=True)
    arpa_gz = target_dir / "3-gram.pruned.1e-7.arpa.gz"
    arpa = target_dir / "3-gram.pruned.1e-7.arpa"

    if arpa.exists():
        print(f"  Already exists: {arpa}")
        return arpa

    url = "https://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz"
    print(f"  Downloading {url}...")
    subprocess.run(
        ["wget", "-q", "--show-progress", "-O", str(arpa_gz), url],
        check=True,
    )
    print("  Decompressing...")
    subprocess.run(["gunzip", "-f", str(arpa_gz)], check=True)
    print(f"  Saved: {arpa} ({arpa.stat().st_size / 1e6:.1f} MB)")
    return arpa


def main():
    parser = argparse.ArgumentParser(
        description="Score N-best with n-gram LM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nbest", type=Path, required=True,
                        help="Input N-best JSONL")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSONL with ngram_score added")
    parser.add_argument("--lm-path", type=Path, default=None,
                        help="Path to ARPA LM. If omitted, downloads LS 3-gram")
    parser.add_argument("--order", type=int, default=3,
                        help="N-gram order (for display only; kenlm auto-detects)")
    args = parser.parse_args()

    print("=" * 70)
    print("score_ngram.py  --  n-gram LM scoring")
    print("=" * 70)
    print(f"  nbest:   {args.nbest}")
    print(f"  output:  {args.output}")
    print(f"  lm_path: {args.lm_path or '(will download LS 3-gram)'}")
    print()

    import kenlm

    if args.lm_path and args.lm_path.exists():
        lm_path = args.lm_path
    else:
        lm_dir = args.output.parent / "lm_data"
        lm_path = download_ls_3gram(lm_dir)

    print(f"Loading LM: {lm_path}...")
    model = kenlm.Model(str(lm_path))
    print(f"  Order: {model.order}")

    records = load_nbest(args.nbest)
    n_utts = len(records)
    n_hyps = sum(len(r["nbest"]) for r in records)
    print(f"Loaded {n_utts} utterances, {n_hyps} hypotheses")

    # Score
    t0 = time.time()
    n_scored = 0
    for i, rec in enumerate(records):
        for c in rec["nbest"]:
            text = c["hyp"].lower()
            c["ngram_score"] = model.score(text, bos=True, eos=True)
            n_scored += 1
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n_utts} utterances scored")

    elapsed = time.time() - t0

    # Sanity check
    print(f"\n  Sanity check (first 3 utterances, rank-0):")
    for rec in records[:3]:
        c = rec["nbest"][0]
        print(f"    ngram={c['ngram_score']:8.2f}  ctc={c['score']:8.2f}  "
              f"{c['hyp'][:60]}")

    # Save
    save_jsonl(records, args.output)

    print(f"\n  Scored {n_scored} hypotheses in {elapsed:.1f}s "
          f"({n_scored / max(elapsed, 1):.0f} hyps/s)")
    print(f"  Output: {args.output}")
    print()


if __name__ == "__main__":
    main()

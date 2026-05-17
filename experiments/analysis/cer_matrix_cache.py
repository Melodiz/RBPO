#!/usr/bin/env python3
"""Shared CER matrix computation with pickle caching.

The CER matrix (GxG per utterance) is tau-independent and expensive to compute.
This module computes it once and caches to disk for reuse across E12/E14/E15.

Cache location: results/.cache/cer_matrix_g128.pkl (gitignored)
"""

import hashlib
import json
import pickle
import sys
from pathlib import Path

import editdistance
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "results" / ".cache"


def compute_cer_matrix_single(texts):
    """Compute symmetric CER matrix for one utterance's candidates."""
    n = len(texts)
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = editdistance.eval(list(texts[i]), list(texts[j]))
            denom = max(len(texts[i]), len(texts[j]), 1)
            mat[i, j] = d / denom
            mat[j, i] = mat[i, j]
    return mat


def _cache_key(data_path):
    p = Path(data_path)
    stat = p.stat()
    raw = f"{p.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def compute_or_load_cer_matrices(records, data_path=None, cache_name=None, verbose=True):
    """Compute CER matrices for all utterances, with optional disk caching."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if cache_name:
        cache_file = CACHE_DIR / f"{cache_name}.pkl"
    elif data_path:
        key = _cache_key(data_path)
        cache_file = CACHE_DIR / f"cer_matrix_{key}.pkl"
    else:
        cache_file = None

    if cache_file and cache_file.exists():
        if verbose:
            size_mb = cache_file.stat().st_size / 1e6
            print(f"  Loading CER matrices from cache: {cache_file.name} ({size_mb:.1f} MB)")
        with open(cache_file, "rb") as f:
            matrices = pickle.load(f)
        if len(matrices) == len(records):
            if verbose:
                print(f"  Loaded {len(matrices)} matrices")
            return matrices
        if verbose:
            print(f"  Cache size mismatch ({len(matrices)} vs {len(records)}), recomputing...")

    if verbose:
        print(f"  Computing CER matrices for {len(records)} utterances...")

    matrices = []
    for i, rec in enumerate(records):
        cands = rec["candidates"]
        texts = [c["text"] for c in cands]
        mat = compute_cer_matrix_single(texts)
        matrices.append(mat)
        if verbose and (i + 1) % 500 == 0:
            print(f"    {i + 1}/{len(records)} utterances done")

    if verbose:
        print(f"  Done: {len(matrices)} matrices computed")

    if cache_file:
        with open(cache_file, "wb") as f:
            pickle.dump(matrices, f, protocol=4)
        size_mb = cache_file.stat().st_size / 1e6
        if verbose:
            print(f"  Cached to {cache_file.name} ({size_mb:.1f} MB)")

    return matrices


def mbr_select(cer_matrix, log_scores, tau):
    """Select hypothesis by MBR-CER with softmax(log_scores/tau) weights."""
    n = len(log_scores)
    if np.isinf(tau):
        weights = np.ones(n) / n
    else:
        scaled = log_scores / tau
        scaled -= np.max(scaled)
        weights = np.exp(scaled)
        weights /= weights.sum()
    risk = cer_matrix @ weights
    return int(np.argmin(risk))

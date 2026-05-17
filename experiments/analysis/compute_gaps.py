#!/usr/bin/env python3
"""Compute Gap E25 bootstrap + Gap F VoxPopuli MBR punct-strip WER.

Inputs (all repo-local, no Drive access needed):
- results/tl3_rerun/nbest_g{16,128}_pll.jsonl
- results/musan_rerun/nbest_{0,5,10,20}dB_g16_pll.jsonl
- results/voxpopuli/test_G128_scored.jsonl

For each: runs MBR-CER + RoBERTa PLL tau=10 selection per utterance,
then computes corpus-level WER and paired bootstrap (B=10000 seed=42)
for MBR vs greedy. VoxPopuli also reports punct-strip WER for greedy,
oracle, and MBR.

Outputs JSON to kb_updates/gap_e25_bootstrap.json and
kb_updates/gap_f_voxpopuli_punct_strip.json.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import editdistance
import numpy as np

PUNCT_RE = re.compile(r"[.,;?!:'\"\-]")
WS_RE = re.compile(r"\s+")

def strip_punct(s: str) -> str:
    s = s.lower()
    s = PUNCT_RE.sub(" ", s)
    s = WS_RE.sub(" ", s).strip()
    return s

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def cer_pair(a, b):
    return editdistance.eval(a, b)

def hyp_text(h):
    return h.get("text") or h.get("hyp") or ""

def hyp_pll(h: dict) -> float:
    """Try canonical 'roberta_pll', then 'pll_score' fallback."""
    if "roberta_pll" in h:
        return float(h["roberta_pll"])
    if "pll_score" in h:
        return float(h["pll_score"])
    return 0.0

def hyp_ctc(h: dict) -> float:
    if "ctc_log_prob" in h:
        return float(h["ctc_log_prob"])
    if "score" in h:
        return float(h["score"])
    return 0.0

def mbr_select(hyps: list[dict], score_key: str = "roberta_pll", tau: float = 10.0) -> int:
    """Return index of MBR-selected hypothesis.

    weights[i] propto exp(score[i] / tau)
    risk[i] = sum_j weights[j] * CER(hyp_i, hyp_j)
    selected = argmin risk
    """
    n = len(hyps)
    if n == 0:
        return -1
    if n == 1:
        return 0

    if score_key == "roberta_pll":
        scores = np.array([hyp_pll(h) for h in hyps], dtype=np.float64)
    else:
        scores = np.array([h.get(score_key, 0.0) for h in hyps], dtype=np.float64)
    # softmax(scores / tau) for posterior weights
    s = scores / max(tau, 1e-8)
    s -= s.max()
    w = np.exp(s)
    w /= w.sum()

    texts = [hyp_text(h) for h in hyps]

    # CER matrix (in chars). CER pair (i, j) symmetric.
    n = len(texts)
    risks = np.zeros(n)
    for i in range(n):
        ti = texts[i]
        risk = 0.0
        for j in range(n):
            if i == j:
                continue
            d = cer_pair(ti, texts[j])
            risk += w[j] * d
        risks[i] = risk

    return int(np.argmin(risks))

def greedy_index(hyps: list[dict]) -> int:
    """Return index of greedy hypothesis (highest CTC log-prob).

    For repo nbest files we observed two layouts:
      - voxpopuli/test_G128_scored.jsonl  : key is 'ctc_log_prob'
      - tl3_rerun/, musan_rerun/ nbest_*_pll: key is 'score' (CTC score)
    Greedy injection means index 0 should be greedy in the new pipeline,
    but we still pick by max CTC score to be safe.
    """
    if not hyps:
        return -1
    if "ctc_log_prob" in hyps[0]:
        return int(np.argmax([float(h.get("ctc_log_prob", -1e9)) for h in hyps]))
    if "score" in hyps[0]:
        return int(np.argmax([float(h.get("score", -1e9)) for h in hyps]))
    return 0

def oracle_index(hyps: list[dict]) -> int:
    """Return index with min wer_edits / wer_ref_len."""
    if not hyps:
        return -1
    # min by raw edits (ref_len constant per utt)
    return int(np.argmin([h.get("wer_edits", 999999) for h in hyps]))

def paired_bootstrap_wer(
    refs: list[list[str]],
    hyps_a: list[list[str]],
    hyps_b: list[list[str]],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """One-sided test: is A's WER < B's WER (B = greedy baseline)?"""
    n = len(refs)
    errors_a = np.array([editdistance.eval(hyps_a[i], refs[i]) for i in range(n)])
    errors_b = np.array([editdistance.eval(hyps_b[i], refs[i]) for i in range(n)])
    ref_lens = np.array([len(refs[i]) for i in range(n)])

    wer_a = errors_a.sum() / max(ref_lens.sum(), 1)
    wer_b = errors_b.sum() / max(ref_lens.sum(), 1)
    delta = wer_a - wer_b  # negative = A is better

    rng = np.random.default_rng(seed)
    deltas = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sum_err_a = errors_a[idx].sum()
        sum_err_b = errors_b[idx].sum()
        sum_ref = ref_lens[idx].sum()
        if sum_ref == 0:
            deltas[b] = 0.0
        else:
            deltas[b] = (sum_err_a - sum_err_b) / sum_ref

    # one-sided p-value: probability that A is NOT better (delta >= 0)
    p_value = float(np.mean(deltas >= 0))
    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))

    return {
        "wer_a_pct": float(wer_a) * 100,
        "wer_b_pct": float(wer_b) * 100,
        "delta_pp": float(delta) * 100,
        "p_value": p_value,
        "ci_lower_pp": ci_lower * 100,
        "ci_upper_pp": ci_upper * 100,
        "n_utts": n,
        "n_bootstrap": n_bootstrap,
    }

def run_e25_condition(name: str, jsonl_path: Path, score_key: str = "roberta_pll") -> dict:
    print(f"  [{name}] loading {jsonl_path.name} ...", end=" ", flush=True)
    rows = load_jsonl(jsonl_path)
    print(f"{len(rows)} utts")

    refs_words = []
    greedy_words = []
    mbr_words = []
    n_skipped = 0

    for r in rows:
        # ref/hyp keys vary; try variants
        ref = r.get("ref") or r.get("reference") or r.get("text") or ""
        hyps = r.get("nbest") or r.get("hypotheses") or r.get("hyps") or []
        if not hyps or not ref:
            n_skipped += 1
            continue
        # If hypotheses is a list of strings (older format), wrap
        if isinstance(hyps[0], str):
            n_skipped += 1
            continue

        gi = greedy_index(hyps)
        mi = mbr_select(hyps, score_key=score_key, tau=10.0)

        refs_words.append(ref.split())
        greedy_words.append(hyp_text(hyps[gi]).split())
        mbr_words.append(hyp_text(hyps[mi]).split())

    print(f"  [{name}] n_used={len(refs_words)} n_skipped={n_skipped}")
    if not refs_words:
        return {"name": name, "error": "no usable utterances", "n_skipped": n_skipped}

    bs = paired_bootstrap_wer(refs_words, mbr_words, greedy_words, n_bootstrap=10000, seed=42)
    bs["name"] = name
    bs["score_key"] = score_key
    return bs

def run_voxpopuli_punct_strip(jsonl_path: Path) -> dict:
    print(f"  [voxpopuli] loading {jsonl_path.name} ...", end=" ", flush=True)
    rows = load_jsonl(jsonl_path)
    print(f"{len(rows)} utts")

    # Collect refs + 3 hyp variants per utterance (raw + stripped versions)
    refs_raw_w = []
    refs_strip_w = []
    greedy_raw_w = []
    greedy_strip_w = []
    oracle_raw_w = []
    oracle_strip_w = []
    mbr_raw_w = []
    mbr_strip_w = []

    for i, r in enumerate(rows):
        ref = r.get("reference") or r.get("ref") or ""
        hyps = r.get("hypotheses") or r.get("nbest") or []
        if not ref or not hyps:
            continue

        gi = greedy_index(hyps)
        oi = oracle_index(hyps)
        mi = mbr_select(hyps, score_key="roberta_pll", tau=10.0)

        ref_raw = ref.lower().strip()
        ref_strip = strip_punct(ref)

        g_text = hyp_text(hyps[gi]).lower().strip()
        o_text = hyp_text(hyps[oi]).lower().strip()
        m_text = hyp_text(hyps[mi]).lower().strip()

        g_strip = strip_punct(g_text)
        o_strip = strip_punct(o_text)
        m_strip = strip_punct(m_text)

        refs_raw_w.append(ref_raw.split())
        refs_strip_w.append(ref_strip.split())

        greedy_raw_w.append(g_text.split())
        greedy_strip_w.append(g_strip.split())

        oracle_raw_w.append(o_text.split())
        oracle_strip_w.append(o_strip.split())

        mbr_raw_w.append(m_text.split())
        mbr_strip_w.append(m_strip.split())

        if (i + 1) % 200 == 0:
            print(f"    processed {i+1}/{len(rows)}")

    def corpus(refs, hyps):
        e = sum(editdistance.eval(h, r) for h, r in zip(hyps, refs))
        n = sum(len(r) for r in refs)
        return e / max(n, 1)

    out = {
        "n_utts": len(refs_raw_w),
        "raw": {
            "greedy_pct": corpus(refs_raw_w, greedy_raw_w) * 100,
            "oracle_pct": corpus(refs_raw_w, oracle_raw_w) * 100,
            "mbr_pll_tau10_pct": corpus(refs_raw_w, mbr_raw_w) * 100,
        },
        "punct_strip": {
            "greedy_pct": corpus(refs_strip_w, greedy_strip_w) * 100,
            "oracle_pct": corpus(refs_strip_w, oracle_strip_w) * 100,
            "mbr_pll_tau10_pct": corpus(refs_strip_w, mbr_strip_w) * 100,
        },
    }
    # Derived gaps (in pp)
    out["gaps_pp"] = {
        "raw_greedy_minus_oracle": out["raw"]["greedy_pct"] - out["raw"]["oracle_pct"],
        "raw_mbr_minus_greedy": out["raw"]["mbr_pll_tau10_pct"] - out["raw"]["greedy_pct"],
        "strip_greedy_minus_oracle": out["punct_strip"]["greedy_pct"] - out["punct_strip"]["oracle_pct"],
        "strip_mbr_minus_greedy": out["punct_strip"]["mbr_pll_tau10_pct"] - out["punct_strip"]["greedy_pct"],
    }
    return out

def main():
    repo = Path(__file__).resolve().parents[1]
    out_dir = repo / "kb_updates"
    out_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Phase 3: E25 bootstrap (TL3 G=16/128 + MUSAN 0/20 dB)")
    print("=" * 60)

    e25_bootstraps = {}
    conditions = [
        ("tl3_g16",  repo / "results/tl3_rerun/nbest_g16_pll.jsonl"),
        ("tl3_g128", repo / "results/tl3_rerun/nbest_g128_pll.jsonl"),
        ("musan_0dB_g16",  repo / "results/musan_rerun/nbest_0dB_g16_pll.jsonl"),
        ("musan_5dB_g16",  repo / "results/musan_rerun/nbest_5dB_g16_pll.jsonl"),
        ("musan_10dB_g16", repo / "results/musan_rerun/nbest_10dB_g16_pll.jsonl"),
        ("musan_20dB_g16", repo / "results/musan_rerun/nbest_20dB_g16_pll.jsonl"),
    ]
    for name, path in conditions:
        if not path.exists():
            print(f"  [{name}] MISSING: {path}")
            e25_bootstraps[name] = {"error": "file missing", "path": str(path)}
            continue
        e25_bootstraps[name] = run_e25_condition(name, path, score_key="roberta_pll")

    e25_path = out_dir / "gap_e25_bootstrap.json"
    e25_path.write_text(json.dumps(e25_bootstraps, indent=2))
    print(f"\nSaved {e25_path}\n")

    print("=" * 60)
    print("Phase 2 Gap F: VoxPopuli MBR punct-strip recompute")
    print("=" * 60)

    vox_path = repo / "results/voxpopuli/test_G128_scored.jsonl"
    if vox_path.exists():
        vox_out = run_voxpopuli_punct_strip(vox_path)
        vox_json = out_dir / "gap_f_voxpopuli_punct_strip.json"
        vox_json.write_text(json.dumps(vox_out, indent=2))
        print(f"\nSaved {vox_json}")
        print(json.dumps(vox_out, indent=2))
    else:
        print(f"MISSING: {vox_path}")

if __name__ == "__main__":
    main()

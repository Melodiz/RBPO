#!/usr/bin/env python3
"""E23 verification: is the 0.42pp oracle gap real or a bug?

Runs Checks 1, 2, 3, 5, 6 from a scored N-best JSONL (CPU-only).
Check 4 (beam sweep) is in a separate Colab cell since it needs GPU.

Usage:
    python e23_verify.py \\
        --voxpopuli-jsonl /content/drive/MyDrive/rbpo_results/voxpopuli/test_G128_scored.jsonl \\
        --librispeech-jsonl /content/rbpo/results/g_scaling/neural_lm_scores_G128.jsonl \\
        --output-dir /content/drive/MyDrive/rbpo_results/voxpopuli/verify
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import editdistance



def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def get_candidates(rec):
    return rec.get("candidates") or rec.get("hypotheses") or []


def get_ref(rec):
    return rec.get("ref_text") or rec.get("reference") or ""


def cand_wer_edits_reflen(c, ref_words):
    edits = editdistance.eval(c["text"].split(), ref_words)
    return edits, len(ref_words)


def corpus_wer(records):
    total_edits = 0
    total_ref = 0
    for r in records:
        ref_words = get_ref(r).split()
        cands = get_candidates(r)
        if not cands or not ref_words:
            continue
        # Greedy = first candidate (by E21 convention; first sorted by ctc_log_prob desc)
        e, rl = cand_wer_edits_reflen(cands[0], ref_words)
        total_edits += e
        total_ref += rl
    return total_edits / max(1, total_ref)


def oracle_corpus_wer(records):
    total_edits = 0
    total_ref = 0
    for r in records:
        ref_words = get_ref(r).split()
        cands = get_candidates(r)
        if not cands or not ref_words:
            continue
        e_min = min(editdistance.eval(c["text"].split(), ref_words) for c in cands)
        total_edits += e_min
        total_ref += len(ref_words)
    return total_edits / max(1, total_ref)



PUNCT_RE = re.compile(r"[^\w\s']")
DIGIT_RE = re.compile(r"\d+")

NUM_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
    "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen",
    "18": "eighteen", "19": "nineteen", "20": "twenty",
}

ABBREV = {
    "eu": "european union",
    "us": "united states",
    "uk": "united kingdom",
    "uno": "united nations",
    "un": "united nations",
    "nato": "nato",
    "etc": "et cetera",
    "vs": "versus",
    "mr": "mister",
    "mrs": "missus",
    "ms": "miss",
    "dr": "doctor",
    "st": "saint",
}


def normalize_text(s, lowercase=True, strip_punct=True, expand_digits=False, expand_abbrev=False):
    if not s:
        return s
    if lowercase:
        s = s.lower()
    if strip_punct:
        s = PUNCT_RE.sub(" ", s)
    if expand_digits:
        # Replace digit sequences with spelled-out form (best-effort)
        def _replace(m):
            d = m.group(0)
            if d in NUM_WORDS:
                return NUM_WORDS[d]
            # For multi-digit, leave as-is (safer than wrong expansion)
            return d
        s = DIGIT_RE.sub(_replace, s)
    if expand_abbrev:
        words = s.split()
        s = " ".join(ABBREV.get(w, w) for w in words)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def corpus_wer_with_norm(records, **norm_kwargs):
    total_edits = 0
    total_ref = 0
    for r in records:
        ref_norm = normalize_text(get_ref(r), **norm_kwargs)
        cands = get_candidates(r)
        if not cands or not ref_norm:
            continue
        ref_words = ref_norm.split()
        # Greedy
        hyp_norm = normalize_text(cands[0]["text"], **norm_kwargs)
        e = editdistance.eval(hyp_norm.split(), ref_words)
        total_edits += e
        total_ref += len(ref_words)
    return total_edits / max(1, total_ref)


def oracle_corpus_wer_with_norm(records, **norm_kwargs):
    total_edits = 0
    total_ref = 0
    for r in records:
        ref_norm = normalize_text(get_ref(r), **norm_kwargs)
        cands = get_candidates(r)
        if not cands or not ref_norm:
            continue
        ref_words = ref_norm.split()
        e_min = min(
            editdistance.eval(normalize_text(c["text"], **norm_kwargs).split(), ref_words)
            for c in cands
        )
        total_edits += e_min
        total_ref += len(ref_words)
    return total_edits / max(1, total_ref)



def align_words(ref, hyp):
    """Return list of (op, ref_word, hyp_word) where op in {'=', 'S', 'I', 'D'}."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],     # deletion
                                    dp[i][j - 1],     # insertion
                                    dp[i - 1][j - 1]) # substitution
    # Backtrack
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            ops.append(("=", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("S", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("D", ref[i - 1], "_"))
            i = i - 1
        else:
            ops.append(("I", "_", hyp[j - 1]))
            j = j - 1
    ops.reverse()
    return ops


def fmt_alignment(ops, max_chars=120):
    """Render alignment as two lines: ref_words / hyp_words, with ops marked."""
    ref_line = []
    hyp_line = []
    op_line = []
    for op, r, h in ops:
        w = max(len(r), len(h))
        ref_line.append(r.ljust(w))
        hyp_line.append(h.ljust(w))
        marker = " " if op == "=" else op
        op_line.append(marker.ljust(w))
    return (
        "  REF: " + " ".join(ref_line)[:max_chars],
        "  HYP: " + " ".join(hyp_line)[:max_chars],
        "  OPS: " + " ".join(op_line)[:max_chars],
    )



def check1_inspect(records, n=20):
    print("=" * 70)
    print("CHECK 1: REFERENCE TEXT INSPECTION (first 20 utterances)")
    print("=" * 70)
    for r in records[:n]:
        ref = get_ref(r)
        ref_words = ref.split()
        cands = get_candidates(r)
        if not cands:
            continue
        greedy = cands[0]
        # Oracle = candidate with min edit distance
        cand_edits = [editdistance.eval(c["text"].split(), ref_words) for c in cands]
        oi = cand_edits.index(min(cand_edits))
        oracle = cands[oi]
        ge = cand_edits[0]
        oe = cand_edits[oi]
        print(f"\n  utt={r.get('utt_id', r.get('utterance_id', '?'))}")
        print(f"  REF [{len(ref_words):3d}w]: {ref}")
        print(f"  GRD [edits={ge:3d}, wer={ge/max(1,len(ref_words))*100:.1f}%]: "
              f"{greedy['text']}")
        if oi != 0:
            print(f"  ORC [edits={oe:3d}, wer={oe/max(1,len(ref_words))*100:.1f}%, "
                  f"rank={oi}]: {oracle['text']}")
        else:
            print(f"  ORC = greedy (no recoverable alternative)")

    # Pattern detection
    print("\n  --- Pattern checks ---")
    n_lower_ref = sum(1 for r in records[:200] if get_ref(r).islower() or not any(c.isalpha() for c in get_ref(r)))
    n_lower_hyp = sum(
        1 for r in records[:200]
        for c in get_candidates(r)[:1]
        if c["text"].islower() or not any(ch.isalpha() for ch in c["text"])
    )
    print(f"  Lowercase refs (first 200):   {n_lower_ref}/200")
    print(f"  Lowercase greedies (first 200): {n_lower_hyp}/200")

    has_punct_ref = sum(1 for r in records[:200]
                        if PUNCT_RE.search(get_ref(r)))
    has_punct_hyp = sum(1 for r in records[:200]
                        for c in get_candidates(r)[:1]
                        if PUNCT_RE.search(c["text"]))
    print(f"  Refs with non-word punctuation:    {has_punct_ref}/200")
    print(f"  Greedies with non-word punctuation: {has_punct_hyp}/200")

    has_digit_ref = sum(1 for r in records[:200] if DIGIT_RE.search(get_ref(r)))
    has_digit_hyp = sum(1 for r in records[:200]
                        for c in get_candidates(r)[:1]
                        if DIGIT_RE.search(c["text"]))
    print(f"  Refs with digits:    {has_digit_ref}/200")
    print(f"  Greedies with digits: {has_digit_hyp}/200")


def check1_norm_sweep(records):
    print("\n=" * 35)
    print("CHECK 1 (cont): WER under normalization variants")
    print("=" * 70)

    variants = [
        ("baseline (lowercase only)", dict(lowercase=True, strip_punct=False, expand_digits=False, expand_abbrev=False)),
        ("+ strip punctuation",        dict(lowercase=True, strip_punct=True,  expand_digits=False, expand_abbrev=False)),
        ("+ expand single digits",     dict(lowercase=True, strip_punct=True,  expand_digits=True,  expand_abbrev=False)),
        ("+ expand abbreviations",     dict(lowercase=True, strip_punct=True,  expand_digits=True,  expand_abbrev=True)),
    ]
    print(f"\n  {'variant':<35}  {'greedy WER':>10}  {'oracle WER':>10}  {'gap (pp)':>8}")
    print(f"  {'-'*35}  {'-'*10}  {'-'*10}  {'-'*8}")
    rows = []
    for name, kw in variants:
        gw = corpus_wer_with_norm(records, **kw)
        ow = oracle_corpus_wer_with_norm(records, **kw)
        gap_pp = (gw - ow) * 100
        rows.append({"variant": name, "greedy_wer": gw, "oracle_wer": ow, "gap_pp": gap_pp})
        print(f"  {name:<35}  {gw*100:>9.3f}%  {ow*100:>9.3f}%  {gap_pp:>7.3f}")
    return rows


def check2_alignment(records, n=10):
    print("\n" + "=" * 70)
    print(f"CHECK 2: MANUAL WER VERIFICATION ({n} utterances, word alignment)")
    print("=" * 70)
    for r in records[:n]:
        ref = get_ref(r)
        ref_words = ref.split()
        cands = get_candidates(r)
        if not cands or not ref_words:
            continue
        greedy = cands[0]
        hyp_words = greedy["text"].split()
        ops = align_words(ref_words, hyp_words)
        recomputed_edits = sum(1 for op, _, _ in ops if op != "=")
        de = editdistance.eval(ref_words, hyp_words)
        stored_wer = greedy.get("wer")
        print(f"\n  utt={r.get('utt_id', r.get('utterance_id', '?'))}")
        a, b, c = fmt_alignment(ops)
        print(a)
        print(b)
        print(c)
        print(f"  recomputed edits = {recomputed_edits}, "
              f"editdistance.eval = {de}, "
              f"stored 'wer' field = {stored_wer}")
        ref_len = len(ref_words)
        recomputed_wer = de / max(1, ref_len)
        if stored_wer is not None and abs(stored_wer - recomputed_wer) > 0.001:
            print(f"   MISMATCH: stored {stored_wer*100:.2f}% vs recomputed {recomputed_wer*100:.2f}%")
        else:
            print(f"   stored matches recomputed")


def check3_oracle(records, n=10):
    print("\n" + "=" * 70)
    print(f"CHECK 3: ORACLE COMPUTATION VERIFICATION ({n} recoverable utterances)")
    print("=" * 70)
    recoverable = []
    for r in records:
        ref_words = get_ref(r).split()
        cands = get_candidates(r)
        if len(cands) < 2 or not ref_words:
            continue
        edits = [editdistance.eval(c["text"].split(), ref_words) for c in cands]
        if min(edits) < edits[0]:
            recoverable.append((r, edits))
    print(f"  Found {len(recoverable)} recoverable utterances")

    for r, edits in recoverable[:n]:
        ref_words = get_ref(r).split()
        ref_len = len(ref_words)
        cands = get_candidates(r)
        # Sort by edits ascending (lowest WER first)
        sorted_idx = sorted(range(len(cands)), key=lambda i: edits[i])
        print(f"\n  utt={r.get('utt_id', r.get('utterance_id', '?'))}  ref_len={ref_len}")
        print(f"  REF: {get_ref(r)}")
        print(f"  Top-5 by edit-count:")
        for k in range(min(5, len(sorted_idx))):
            i = sorted_idx[k]
            mark = " <- greedy" if i == 0 else ""
            print(f"    rank {k+1}: edits={edits[i]:3d} "
                  f"wer={edits[i]/max(1,ref_len)*100:5.2f}%  "
                  f"i={i:3d} text='{cands[i]['text'][:80]}'{mark}")
        print(f"  Bottom-3 by edit-count:")
        for k in range(max(0, len(sorted_idx) - 3), len(sorted_idx)):
            i = sorted_idx[k]
            print(f"    rank {k+1}: edits={edits[i]:3d} "
                  f"wer={edits[i]/max(1,ref_len)*100:5.2f}%  "
                  f"i={i:3d} text='{cands[i]['text'][:80]}'")
        zero_idx = [i for i, e in enumerate(edits) if e == 0]
        if zero_idx:
            print(f"   ZERO-EDIT candidates exist at indices: {zero_idx[:10]}")


def check5_compare(librispeech_records, voxpopuli_records, n=5):
    print("\n" + "=" * 70)
    print(f"CHECK 5: SAME ORACLE COMPUTATION ON BOTH DATASETS")
    print("=" * 70)
    # Compute full corpus oracle on both via the same code path
    print("\n  LibriSpeech dev-other (same code path):")
    g_ls = corpus_wer(librispeech_records)
    o_ls = oracle_corpus_wer(librispeech_records)
    print(f"    n_utts={len(librispeech_records)}  "
          f"greedy={g_ls*100:.3f}%  oracle={o_ls*100:.3f}%  "
          f"gap={(g_ls-o_ls)*100:.3f}pp")

    print("\n  VoxPopuli en/test (same code path):")
    g_vp = corpus_wer(voxpopuli_records)
    o_vp = oracle_corpus_wer(voxpopuli_records)
    print(f"    n_utts={len(voxpopuli_records)}  "
          f"greedy={g_vp*100:.3f}%  oracle={o_vp*100:.3f}%  "
          f"gap={(g_vp-o_vp)*100:.3f}pp")

    # Print 5 LibriSpeech recoverable utterances for visual sanity
    print("\n  5 LibriSpeech recoverable utterances (oracle != greedy):")
    n_shown = 0
    for r in librispeech_records:
        ref_words = get_ref(r).split()
        cands = get_candidates(r)
        if len(cands) < 2 or not ref_words:
            continue
        edits = [editdistance.eval(c["text"].split(), ref_words) for c in cands]
        if min(edits) < edits[0]:
            improvement = edits[0] - min(edits)
            print(f"\n    utt={r.get('utt_id', '?')}  improvement={improvement} edits")
            print(f"    REF:    {get_ref(r)[:80]}")
            print(f"    GREEDY: {cands[0]['text'][:80]} ({edits[0]} edits)")
            best_i = edits.index(min(edits))
            print(f"    ORACLE: {cands[best_i]['text'][:80]} ({edits[best_i]} edits)")
            n_shown += 1
            if n_shown >= n:
                break
    return {
        "librispeech": {"greedy": g_ls, "oracle": o_ls, "gap_pp": (g_ls - o_ls) * 100},
        "voxpopuli":   {"greedy": g_vp, "oracle": o_vp, "gap_pp": (g_vp - o_vp) * 100},
    }


def check6_distribution(records):
    print("\n" + "=" * 70)
    print("CHECK 6: PER-UTTERANCE GAP DISTRIBUTION")
    print("=" * 70)
    gaps = []  # (greedy_wer - oracle_wer) in pp per utt
    improvements_edits = []  # raw edit improvements per utt
    n_zero_gap = 0
    n_with_improvement = 0
    for r in records:
        ref_words = get_ref(r).split()
        cands = get_candidates(r)
        if not cands or not ref_words:
            continue
        edits = [editdistance.eval(c["text"].split(), ref_words) for c in cands]
        improvement = edits[0] - min(edits)
        improvements_edits.append(improvement)
        gap_pp = (improvement / max(1, len(ref_words))) * 100
        gaps.append(gap_pp)
        if improvement == 0:
            n_zero_gap += 1
        else:
            n_with_improvement += 1

    n = len(gaps)
    print(f"  Total utterances analyzed: {n}")
    print(f"  Zero-gap (greedy = oracle): {n_zero_gap} ({n_zero_gap/max(1,n)*100:.1f}%)")
    print(f"  With ANY improvement: {n_with_improvement} ({n_with_improvement/max(1,n)*100:.1f}%)")
    if n_with_improvement > 0:
        improvements_only = [i for i in improvements_edits if i > 0]
        print(f"\n  Improvements (edits): "
              f"mean={sum(improvements_only)/len(improvements_only):.2f} "
              f"max={max(improvements_only)}")
        sorted_imp = sorted(improvements_only)
        median_imp = sorted_imp[len(sorted_imp) // 2]
        print(f"  median={median_imp}")

    # Histogram of per-utt gap (in pp)
    hist_bins = [0, 0.01, 1, 5, 10, 20, 50, 100]
    print(f"\n  Histogram of per-utt gap (in pp):")
    for lo, hi in zip(hist_bins[:-1], hist_bins[1:]):
        bucket = sum(1 for g in gaps if lo <= g < hi)
        bar = "#" * (bucket * 60 // max(1, n))
        print(f"    [{lo:>5.2f}, {hi:>5.2f}): {bucket:5d}  {bar}")
    over_top = sum(1 for g in gaps if g >= hist_bins[-1])
    if over_top:
        print(f"    [{hist_bins[-1]:>5.2f}, inf):   {over_top:5d}")

    return {
        "n_utts": n,
        "n_zero_gap": n_zero_gap,
        "n_with_improvement": n_with_improvement,
        "improvements_total_edits": sum(improvements_edits),
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voxpopuli-jsonl", type=Path, required=True)
    parser.add_argument("--librispeech-jsonl", type=Path, default=None,
                        help="Optional: path to LibriSpeech G=128 N-best with same format")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inspect-n", type=int, default=20)
    parser.add_argument("--align-n", type=int, default=10)
    parser.add_argument("--oracle-n", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading VoxPopuli: {args.voxpopuli_jsonl}")
    vp = load_jsonl(args.voxpopuli_jsonl)
    print(f"  {len(vp)} utterances")

    # Quick aggregate check upfront
    g = corpus_wer(vp)
    o = oracle_corpus_wer(vp)
    print(f"\n  Re-computed greedy WER:  {g*100:.4f}%")
    print(f"  Re-computed oracle WER:  {o*100:.4f}%")
    print(f"  Re-computed gap:         {(g-o)*100:.4f}pp")

    check1_inspect(vp, n=args.inspect_n)
    norm_rows = check1_norm_sweep(vp)
    check2_alignment(vp, n=args.align_n)
    check3_oracle(vp, n=args.oracle_n)

    compare_results = None
    if args.librispeech_jsonl and args.librispeech_jsonl.exists():
        print(f"\nLoading LibriSpeech: {args.librispeech_jsonl}")
        ls = load_jsonl(args.librispeech_jsonl)
        print(f"  {len(ls)} utterances")
        compare_results = check5_compare(ls, vp, n=5)

    distrib = check6_distribution(vp)

    summary = {
        "voxpopuli_n": len(vp),
        "voxpopuli_greedy_wer": g,
        "voxpopuli_oracle_wer": o,
        "voxpopuli_gap_pp": (g - o) * 100,
        "normalization_sweep": norm_rows,
        "compare_results": compare_results,
        "distribution": distrib,
    }
    out = args.output_dir / "verify_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n\n Saved summary: {out}")


if __name__ == "__main__":
    main()

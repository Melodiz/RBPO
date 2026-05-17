#!/usr/bin/env python3
"""Decode-time scoring strategy comparison on cached N-best data.

Reads nbest_dev_other_G16.jsonl and applies 9 selection strategies:
  1. Greedy (1-best)          --  always pick candidates[0]
  2. Argmax P_CTC             --  pick highest ctc_log_prob
  3. Length-norm (tokens)      --  ctc_log_prob / len_tokens
  4. Length-norm (chars)       --  ctc_log_prob / len_chars
  5. MBR-CER                  --  minimum Bayes risk, char edit distance
  6. MBR-WER                  --  minimum Bayes risk, word edit distance
  7. MBR-token                --  minimum Bayes risk, BPE token edit distance
  8. Self-consistency          --  MBR with uniform weights (ignores P_CTC)
  9. Oracle                   --  pick candidate with lowest WER to reference

Usage:
    python experiments/scoring_strategies.py \
        --nbest-file results/nbest_dev_other_G16.jsonl \
        --results-dir results
"""

import argparse
import csv
import json
import time
from pathlib import Path

import editdistance
import numpy as np


def compute_wer(hypothesis: str, reference: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return editdistance.eval(hyp_words, ref_words) / len(ref_words)


def compute_cer(hypothesis: str, reference: str) -> float:
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    return editdistance.eval(list(hypothesis), list(reference)) / len(reference)


def log_softmax(log_probs: list[float]) -> np.ndarray:
    a = np.array(log_probs, dtype=np.float64)
    max_a = np.max(a)
    log_sum = max_a + np.log(np.sum(np.exp(a - max_a)))
    return a - log_sum


def mbr_select(candidates, log_probs, distance_fn, uniform=False):
    """Select candidate minimizing expected distance under the posterior.

    If uniform=True, ignores log_probs and uses 1/G for all candidates
    (self-consistency / ROVER-style majority voting).
    """
    n = len(candidates)
    if n == 1:
        return 0

    if uniform:
        weights = np.ones(n, dtype=np.float64) / n
    else:
        log_p = log_softmax(log_probs)
        weights = np.exp(log_p)

    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            scores[i] += weights[j] * distance_fn(candidates[i], candidates[j])

    return int(np.argmin(scores))


def word_distance(a, b):
    wa = a.split()
    wb = b.split()
    denom = max(len(wa), len(wb), 1)
    return editdistance.eval(wa, wb) / denom


def char_distance(a, b):
    denom = max(len(a), len(b), 1)
    return editdistance.eval(list(a), list(b)) / denom


def token_distance(a_tokens: list[int], b_tokens: list[int]) -> float:
    denom = max(len(a_tokens), len(b_tokens), 1)
    return editdistance.eval(a_tokens, b_tokens) / denom


def load_nbest(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} utterances from {path}")
    if not records:
        raise SystemExit(
            f"ERROR: {path} is empty. Did generate_nbest.py succeed?"
        )
    return records


STRATEGIES = [
    "greedy",
    "argmax_p",
    "length_norm_tok",
    "length_norm_char",
    "mbr_cer",
    "mbr_wer",
    "mbr_token",
    "self_consistency",
    "oracle",
]


def run_strategies(records: list[dict]):
    per_utt = []
    strategy_totals = {s: {"wer_num": 0, "wer_den": 0, "cer_num": 0, "cer_den": 0}
                       for s in STRATEGIES}

    for rec in records:
        ref = rec["ref_text"]
        cands = rec["candidates"]
        ref_words = ref.split()
        ref_chars = list(ref)
        n_ref_words = len(ref_words)
        n_ref_chars = len(ref_chars)

        texts = [c["text"] for c in cands]
        log_probs = [c["ctc_log_prob"] for c in cands]
        token_seqs = [c["tokens"] for c in cands]
        len_toks = [c["len_tokens"] for c in cands]
        len_chars = [c["len_chars"] for c in cands]
        n = len(cands)

        selected = {}

        # 1. Greedy: always first candidate
        selected["greedy"] = 0

        # 2. Argmax P_CTC
        selected["argmax_p"] = int(np.argmax(log_probs))

        # 3. Length-norm (tokens)
        scores_lt = [lp / max(lt, 1) for lp, lt in zip(log_probs, len_toks)]
        selected["length_norm_tok"] = int(np.argmax(scores_lt))

        # 4. Length-norm (chars)
        scores_lc = [lp / max(lc, 1) for lp, lc in zip(log_probs, len_chars)]
        selected["length_norm_char"] = int(np.argmax(scores_lc))

        # 5. MBR-CER
        selected["mbr_cer"] = mbr_select(texts, log_probs, char_distance)

        # 6. MBR-WER
        selected["mbr_wer"] = mbr_select(texts, log_probs, word_distance)

        # 7. MBR-token
        selected["mbr_token"] = mbr_select(
            token_seqs, log_probs,
            lambda a, b: token_distance(a, b),
        )

        # 8. Self-consistency (uniform MBR)
        selected["self_consistency"] = mbr_select(
            texts, log_probs, word_distance, uniform=True
        )

        # 9. Oracle
        wers = [compute_wer(t, ref) for t in texts]
        selected["oracle"] = int(np.argmin(wers))

        utt_result = {
            "utt_id": rec["utt_id"],
            "ref_text": ref,
            "num_unique_candidates": n,
            "ctc_log_probs": log_probs,
            "candidate_lengths_tokens": len_toks,
            "candidate_lengths_chars": len_chars,
            "greedy_is_oracle": selected["greedy"] == selected["oracle"],
            "strategy_wers": {},
        }

        for s in STRATEGIES:
            idx = selected[s]
            hyp = texts[idx]
            w = compute_wer(hyp, ref)
            c = compute_cer(hyp, ref)

            hyp_words = hyp.split()
            hyp_chars = list(hyp)
            strategy_totals[s]["wer_num"] += editdistance.eval(hyp_words, ref_words)
            strategy_totals[s]["wer_den"] += n_ref_words
            strategy_totals[s]["cer_num"] += editdistance.eval(hyp_chars, ref_chars)
            strategy_totals[s]["cer_den"] += n_ref_chars

            utt_result["strategy_wers"][s] = round(w, 6)

        utt_result["greedy_wer"] = utt_result["strategy_wers"]["greedy"]
        utt_result["oracle_wer"] = utt_result["strategy_wers"]["oracle"]

        per_utt.append(utt_result)

    results = {}
    for s in STRATEGIES:
        t = strategy_totals[s]
        wer = t["wer_num"] / max(t["wer_den"], 1)
        cer = t["cer_num"] / max(t["cer_den"], 1)
        results[s] = {"wer": wer, "cer": cer}

    greedy_wer = results["greedy"]["wer"]
    oracle_wer = results["oracle"]["wer"]
    gap = greedy_wer - oracle_wer

    for s in STRATEGIES:
        if gap > 0:
            closed = (greedy_wer - results[s]["wer"]) / gap * 100
        else:
            closed = 100.0 if results[s]["wer"] <= greedy_wer else 0.0
        results[s]["gap_closed_pct"] = closed

    return results, per_utt


def print_table(results):
    print("\n" + "=" * 80)
    print("SCORING STRATEGY COMPARISON  --  dev-other G=16")
    print("=" * 80)
    print(f"{'Strategy':<22} {'WER%':>8} {'CER%':>8} {'Gap Closed%':>12}")
    print("-" * 80)
    for s in STRATEGIES:
        r = results[s]
        label = s.replace("_", " ").title()
        print(f"{label:<22} {r['wer']*100:>7.2f}% {r['cer']*100:>7.2f}% {r['gap_closed_pct']:>11.1f}%")
    print("=" * 80)

    greedy_wer = results["greedy"]["wer"]
    oracle_wer = results["oracle"]["wer"]
    print(f"\nGreedy WER: {greedy_wer*100:.2f}%")
    print(f"Oracle WER: {oracle_wer*100:.2f}%")
    if greedy_wer > 0:
        print(f"Relative oracle gap: {(greedy_wer - oracle_wer) / greedy_wer * 100:.1f}%")
    else:
        print("Relative oracle gap: N/A (greedy WER is 0)")

    if abs(results["greedy"]["wer"] - results["argmax_p"]["wer"]) < 1e-6:
        print("\nGreedy == Argmax P_CTC: YES (as expected  --  candidates[0] has highest CTC prob)")
    else:
        print(f"\nGreedy != Argmax P_CTC: greedy={results['greedy']['wer']*100:.2f}% vs "
              f"argmax={results['argmax_p']['wer']*100:.2f}%")
        print("  This means candidates[0] is not always the highest-scoring path in the lattice.")


def save_outputs(results, per_utt, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    json_out = {}
    for s in STRATEGIES:
        r = results[s]
        json_out[s] = {
            "wer": round(r["wer"], 6),
            "cer": round(r["cer"], 6),
            "gap_closed_pct": round(r["gap_closed_pct"], 2),
        }

    json_path = output_dir / "scoring_results.json"
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"Saved: {json_path}")

    csv_path = output_dir / "scoring_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "wer", "cer", "gap_closed_pct"])
        for s in STRATEGIES:
            r = results[s]
            writer.writerow([s, f"{r['wer']:.6f}", f"{r['cer']:.6f}",
                             f"{r['gap_closed_pct']:.2f}"])
    print(f"Saved: {csv_path}")

    per_utt_path = output_dir / "per_utterance.jsonl"
    with open(per_utt_path, "w") as f:
        for rec in per_utt:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved: {per_utt_path}")


def generate_report(results, per_utt, elapsed_scoring, output_dir: Path):
    greedy_wer = results["greedy"]["wer"]
    oracle_wer = results["oracle"]["wer"]
    n_utts = len(per_utt)
    n_greedy_opt = sum(1 for u in per_utt if u["greedy_is_oracle"])

    lines = []
    lines.append("# Level 1: Decode-Time Scoring Strategy Comparison\n")
    lines.append(f"**Dataset:** LibriSpeech dev-other ({n_utts} utterances)")
    lines.append(f"**Model:** Zipformer-S CR-CTC, BPE-500")
    lines.append(f"**N-best:** G=16, nbest_scale=1.0\n")

    lines.append("## Summary Table\n")
    lines.append(f"| {'Strategy':<22} | {'WER%':>8} | {'CER%':>8} | {'Gap Closed%':>12} |")
    lines.append(f"|{'-'*24}|{'-'*10}|{'-'*10}|{'-'*14}|")
    for s in STRATEGIES:
        r = results[s]
        label = s.replace("_", " ").title()
        lines.append(f"| {label:<22} | {r['wer']*100:>7.2f}% | {r['cer']*100:>7.2f}% | {r['gap_closed_pct']:>11.1f}% |")

    lines.append(f"\n**Greedy WER:** {greedy_wer*100:.2f}%")
    lines.append(f"**Oracle WER:** {oracle_wer*100:.2f}%")
    rel_gap = (greedy_wer - oracle_wer) / greedy_wer * 100 if greedy_wer > 0 else 0.0
    lines.append(f"**Relative oracle gap:** {rel_gap:.1f}%")
    lines.append(f"**Greedy is oracle:** {n_greedy_opt}/{n_utts} "
                 f"({n_greedy_opt/n_utts*100:.1f}%) utterances\n")

    lines.append("## Greedy vs Argmax P_CTC\n")
    if abs(results["greedy"]["wer"] - results["argmax_p"]["wer"]) < 1e-6:
        lines.append("Greedy (candidates[0]) and Argmax P_CTC produce identical WER. "
                      "This confirms that candidates are sorted by descending CTC log-probability "
                      "and the 1-best from the lattice matches the greedy argmax path.\n")
    else:
        lines.append(f"Greedy WER = {results['greedy']['wer']*100:.2f}%, "
                      f"Argmax WER = {results['argmax_p']['wer']*100:.2f}%.\n")
        lines.append("These differ because `candidates[0]` is the greedy CTC decode "
                      "(frame-by-frame argmax), while Argmax P_CTC selects the candidate with "
                      "the highest total CTC log-probability from the lattice. The greedy path "
                      "is not guaranteed to have the highest total probability  --  CTC marginalizes "
                      "over alignments, so a different token sequence can accumulate more "
                      "probability mass across all its alignments.\n")

    lines.append("## Findings\n")

    best_non_oracle = None
    best_name = None
    for s in STRATEGIES:
        if s in ("greedy", "argmax_p", "oracle"):
            continue
        if best_non_oracle is None or results[s]["wer"] < best_non_oracle:
            best_non_oracle = results[s]["wer"]
            best_name = s

    if best_name:
        lines.append(f"- **Best strategy:** {best_name.replace('_', ' ').title()} "
                      f"(WER {best_non_oracle*100:.2f}%, "
                      f"closes {results[best_name]['gap_closed_pct']:.1f}% of oracle gap)")

    worse_than_greedy = [s for s in STRATEGIES
                         if s not in ("greedy", "oracle")
                         and results[s]["wer"] > greedy_wer + 1e-6]
    if worse_than_greedy:
        lines.append(f"- **Worse than greedy:** {', '.join(worse_than_greedy)}  --  "
                      "these strategies hurt rather than help.")

    mbr_cer_gap = results["mbr_cer"]["gap_closed_pct"]
    sc_gap = results["self_consistency"]["gap_closed_pct"]
    if abs(mbr_cer_gap - sc_gap) < 5.0:
        lines.append(f"- **MBR-CER vs Self-consistency:** similar gap closure "
                      f"({mbr_cer_gap:.1f}% vs {sc_gap:.1f}%), suggesting CTC probabilities "
                      "add little value for candidate selection  --  diversity matters more than scoring.")
    elif mbr_cer_gap > sc_gap:
        lines.append(f"- **MBR-CER vs Self-consistency:** MBR-CER closes {mbr_cer_gap:.1f}% vs "
                      f"{sc_gap:.1f}%  --  CTC probabilities meaningfully improve selection beyond "
                      "simple consensus.")
    else:
        lines.append(f"- **MBR-CER vs Self-consistency:** Self-consistency closes MORE gap "
                      f"({sc_gap:.1f}% vs {mbr_cer_gap:.1f}%)  --  CTC posterior weighting actually "
                      "hurts. The model's probabilities may be miscalibrated.")

    mbr_wer_gap = results["mbr_wer"]["gap_closed_pct"]
    if mbr_wer_gap > mbr_cer_gap + 2.0:
        lines.append(f"- **MBR-WER > MBR-CER:** {mbr_wer_gap:.1f}% vs {mbr_cer_gap:.1f}%  --  "
                      "flagged: using WER as both utility and evaluation metric is metric gaming. "
                      "MBR-CER is the fairer comparison.")

    lines.append(f"\n## Runtime\n")
    lines.append(f"- Scoring (CPU): {elapsed_scoring:.1f}s")

    lines.append(f"\n## Output Files\n")
    lines.append(f"- `nbest_dev_other_G16.jsonl`  --  cached N-best data")
    lines.append(f"- `scoring_results.json`  --  aggregate results")
    lines.append(f"- `scoring_results.csv`  --  for plotting")
    lines.append(f"- `per_utterance.jsonl`  --  per-utterance strategy WERs for Level 2 analysis")
    lines.append(f"- `level1_report.md`  --  this report")

    report_path = output_dir / "level1_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {report_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scoring strategy comparison on cached N-best data"
    )
    parser.add_argument(
        "--nbest-file", type=Path,
        default=Path("results/nbest_dev_other_G16.jsonl"),
    )
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path("results"),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Level 1: Scoring Strategy Comparison")
    print("=" * 60)

    records = load_nbest(args.nbest_file)

    t0 = time.time()
    results, per_utt = run_strategies(records)
    elapsed = time.time() - t0
    print(f"\nScoring completed in {elapsed:.1f}s")

    print_table(results)
    save_outputs(results, per_utt, args.results_dir)
    generate_report(results, per_utt, elapsed, args.results_dir)


if __name__ == "__main__":
    main()

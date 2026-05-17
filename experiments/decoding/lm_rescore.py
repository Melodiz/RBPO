#!/usr/bin/env python3
"""Level 4: N-gram LM rescoring of CTC N-best lists.

Scores N-best candidates with a kenlm n-gram language model, performs
interpolated rescoring sweeps, MBR with LM-adjusted weights, and trains
an enhanced feature rescorer with LM features.

Usage (Colab):
    pip install kenlm editdistance scikit-learn matplotlib tqdm

    # Download LM (run once)
    wget https://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz
    gunzip 3-gram.pruned.1e-7.arpa.gz

    python experiments/lm_rescore.py \
        --lm-path 3-gram.pruned.1e-7.arpa \
        --nbest-file results/nbest_dev_other_G16.jsonl \
        --results-dir results

    # With 4-gram:
    python experiments/lm_rescore.py \
        --lm-path 4-gram.arpa \
        --nbest-file results/nbest_dev_other_G16.jsonl \
        --results-dir results --lm-name 4gram
"""

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import editdistance
import kenlm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


LN10 = math.log(10)

ORIG_FEATURE_NAMES = [
    "ctc_log_prob", "ctc_log_prob_per_token", "ctc_log_prob_per_char",
    "ctc_rank", "len_tokens", "len_chars", "len_words", "len_deviation",
    "mean_cer_to_others", "mean_wer_to_others", "agrees_with_majority",
    "log_prob_gap", "ptilde", "entropy_of_group",
]

LM_FEATURE_NAMES = [
    "lm_log_prob", "lm_log_prob_per_word", "lm_perplexity", "lm_rank",
]

ALL_FEATURE_NAMES = ORIG_FEATURE_NAMES + LM_FEATURE_NAMES


def compute_wer(hyp: str, ref: str) -> float:
    ref_w = ref.split()
    hyp_w = hyp.split()
    if len(ref_w) == 0:
        return 0.0 if len(hyp_w) == 0 else 1.0
    return editdistance.eval(hyp_w, ref_w) / len(ref_w)


def corpus_wer_from_selections(records, selections):
    total_edits = 0
    total_ref_words = 0
    for rec in records:
        uid = rec["utt_id"]
        sel = selections.get(uid, 0)
        sel = min(sel, len(rec["candidates"]) - 1)
        hyp = rec["candidates"][sel]["text"]
        ref = rec["ref_text"]
        ref_w = ref.split()
        hyp_w = hyp.split()
        total_edits += editdistance.eval(hyp_w, ref_w)
        total_ref_words += len(ref_w)
    return total_edits / max(total_ref_words, 1)


def load_nbest(path: Path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} utterances from {path}")
    return records


def score_candidates_with_lm(records, lm_model):
    """Add LM scores to each candidate in-place. Returns records."""
    for rec in tqdm(records, desc="LM scoring"):
        lm_scores = []
        for cand in rec["candidates"]:
            text = cand["text"].strip()
            n_words = max(len(text.split()), 1)
            lm_log10 = lm_model.score(text, bos=True, eos=True)
            lm_ln = lm_log10 * LN10
            lm_per_word = lm_ln / n_words
            lm_ppl = 10.0 ** (-lm_log10 / n_words)
            cand["lm_log_prob"] = lm_ln
            cand["lm_log_prob_per_word"] = lm_per_word
            cand["lm_perplexity"] = lm_ppl
            lm_scores.append(lm_ln)

        ranked = np.argsort(lm_scores)[::-1]
        for rank, idx in enumerate(ranked):
            rec["candidates"][idx]["lm_rank"] = rank

    return records



def interpolation_sweep_normalized(records, lambdas):
    """score = (1-lam)*ctc_per_word + lam*lm_per_word (both in nats)."""
    results = []
    for lam in lambdas:
        sels = {}
        for rec in records:
            best_score = -float("inf")
            best_idx = 0
            for i, cand in enumerate(rec["candidates"]):
                n_words = max(len(cand["text"].split()), 1)
                ctc_pw = cand["ctc_log_prob"] / n_words
                lm_pw = cand["lm_log_prob_per_word"]
                score = (1 - lam) * ctc_pw + lam * lm_pw
                if score > best_score:
                    best_score = score
                    best_idx = i
            sels[rec["utt_id"]] = best_idx

        wer = corpus_wer_from_selections(records, sels)
        results.append({"lambda": lam, "wer": wer, "method": "normalized"})
        print(f"  lam={lam:.1f}  WER={wer*100:.2f}%")
    return results


def interpolation_sweep_unnormalized(records, betas):
    """score = ctc_log_prob + beta*lm_log_prob (unnormalized)."""
    results = []
    for beta in betas:
        sels = {}
        for rec in records:
            best_score = -float("inf")
            best_idx = 0
            for i, cand in enumerate(rec["candidates"]):
                score = cand["ctc_log_prob"] + beta * cand["lm_log_prob"]
                if score > best_score:
                    best_score = score
                    best_idx = i
            sels[rec["utt_id"]] = best_idx

        wer = corpus_wer_from_selections(records, sels)
        results.append({"beta": beta, "wer": wer, "method": "unnormalized"})
        print(f"  beta={beta:.1f}  WER={wer*100:.2f}%")
    return results



def mbr_cer_with_lm(records, beta, tau):
    """MBR-CER selection with LM-adjusted posterior weights."""
    sels = {}
    for rec in records:
        cands = rec["candidates"]
        n = len(cands)
        texts = [c["text"] for c in cands]

        log_scores = np.array([
            c["ctc_log_prob"] + beta * c["lm_log_prob"] for c in cands
        ])

        if tau == float("inf"):
            weights = np.ones(n) / n
        else:
            scaled = log_scores / tau
            scaled -= np.max(scaled)
            weights = np.exp(scaled)
            weights /= weights.sum()

        cer_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = editdistance.eval(list(texts[i]), list(texts[j]))
                denom = max(len(texts[i]), len(texts[j]), 1)
                cer_matrix[i, j] = d / denom
                cer_matrix[j, i] = cer_matrix[i, j]

        expected_risk = cer_matrix @ weights
        sels[rec["utt_id"]] = int(np.argmin(expected_risk))

    wer = corpus_wer_from_selections(records, sels)
    return wer


def mbr_sweep(records, betas, taus):
    results = []
    for beta in betas:
        for tau in taus:
            wer = mbr_cer_with_lm(records, beta, tau)
            results.append({
                "beta": beta, "tau": tau if tau != float("inf") else "inf",
                "wer": wer, "method": "mbr_lm",
            })
            tau_str = "inf" if tau == float("inf") else f"{tau:.0f}"
            print(f"  MBR beta={beta:.1f} tau={tau_str}  WER={wer*100:.2f}%")
    return results



def augment_features_csv(csv_path: Path, records, output_path: Path):
    """Add LM features to an existing feature CSV. Match by utt_id + candidate_idx."""
    lm_lookup = {}
    for rec in records:
        uid = rec["utt_id"]
        for i, cand in enumerate(rec["candidates"]):
            lm_lookup[(uid, i)] = {
                "lm_log_prob": cand["lm_log_prob"],
                "lm_log_prob_per_word": cand["lm_log_prob_per_word"],
                "lm_perplexity": cand["lm_perplexity"],
                "lm_rank": cand["lm_rank"],
            }

    rows_out = []
    matched = 0
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        orig_fields = reader.fieldnames
        out_fields = list(orig_fields) + LM_FEATURE_NAMES

        for row in reader:
            uid = row["utt_id"]
            cidx = int(row["candidate_idx"])
            key = (uid, cidx)
            if key in lm_lookup:
                for fn in LM_FEATURE_NAMES:
                    row[fn] = f"{lm_lookup[key][fn]:.6f}"
                matched += 1
            else:
                for fn in LM_FEATURE_NAMES:
                    row[fn] = "0.0"
            rows_out.append(row)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Augmented {csv_path} -> {output_path}: {matched} matched, "
          f"{len(rows_out) - matched} unmatched")
    return output_path


def load_features(path: Path, feature_names):
    utt_ids = []
    cand_idxs = []
    X_rows = []
    y_rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            utt_ids.append(row["utt_id"])
            cand_idxs.append(int(row["candidate_idx"]))
            feats = [float(row[fn]) for fn in feature_names]
            X_rows.append(feats)
            y_rows.append(float(row["wer"]))
    return utt_ids, cand_idxs, np.array(X_rows), np.array(y_rows)


def group_by_utterance(utt_ids, X, y):
    groups = defaultdict(list)
    for i, uid in enumerate(utt_ids):
        groups[uid].append(i)
    result = {}
    for uid, idxs in groups.items():
        idxs = np.array(idxs)
        result[uid] = (idxs, X[idxs], y[idxs])
    return result


def train_and_eval_rescorer(X_train, y_train, X_dev, y_dev,
                            utt_ids_dev, nbest_path, feature_names,
                            model_type="ridge"):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_dv = scaler.transform(X_dev)

    if model_type == "ridge":
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
    else:
        from sklearn.neural_network import MLPRegressor
        model = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500,
                             early_stopping=True, validation_fraction=0.1,
                             random_state=42)

    model.fit(X_tr, y_train)

    groups = group_by_utterance(utt_ids_dev, X_dev, y_dev)
    sels = {}
    for uid, (idxs, X_g, y_g) in groups.items():
        X_scaled = scaler.transform(X_g)
        pred = model.predict(X_scaled)
        sels[uid] = int(np.argmin(pred))

    records = load_nbest(nbest_path)
    wer = corpus_wer_from_selections(records, sels)

    importance = None
    if model_type == "ridge":
        importance = sorted(
            zip(feature_names, model.coef_),
            key=lambda x: abs(x[1]), reverse=True,
        )

    return wer, importance, model, scaler


def run_rescorer_ablation(X_train, y_train, X_dev, y_dev,
                          utt_ids_train, utt_ids_dev, nbest_path):
    """Train rescopers on various feature subsets including LM."""
    feature_groups = {
        "lm_only": LM_FEATURE_NAMES,
        "ctc_plus_lm": [
            "ctc_log_prob", "ctc_log_prob_per_token",
            "ctc_log_prob_per_char", "ctc_rank",
        ] + LM_FEATURE_NAMES,
        "agreement_plus_lm": [
            "mean_cer_to_others", "mean_wer_to_others", "agrees_with_majority",
        ] + LM_FEATURE_NAMES,
        "all_18": ALL_FEATURE_NAMES,
    }

    results = {}
    for group_name, feat_list in feature_groups.items():
        feat_indices = [ALL_FEATURE_NAMES.index(f) for f in feat_list]
        X_tr_sub = X_train[:, feat_indices]
        X_dv_sub = X_dev[:, feat_indices]

        wer, importance, _, _ = train_and_eval_rescorer(
            X_tr_sub, y_train, X_dv_sub, y_dev,
            utt_ids_dev, nbest_path, feat_list, "ridge",
        )
        results[group_name] = {"wer": wer, "features": feat_list}
        print(f"  Ridge ({group_name}): WER {wer*100:.2f}%")

    # MLP on all 18
    feat_indices = list(range(len(ALL_FEATURE_NAMES)))
    wer_mlp, _, _, _ = train_and_eval_rescorer(
        X_train, y_train, X_dev, y_dev,
        utt_ids_dev, nbest_path, ALL_FEATURE_NAMES, "mlp",
    )
    results["mlp_all_18"] = {"wer": wer_mlp}
    print(f"  MLP (all_18): WER {wer_mlp*100:.2f}%")

    return results



def plot_lambda_sweep(norm_results, unnorm_results, greedy_wer, oracle_wer,
                      output_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    lams = [r["lambda"] for r in norm_results]
    wers = [r["wer"] * 100 for r in norm_results]
    ax1.plot(lams, wers, "o-", color="#2ecc71", linewidth=2, markersize=6)
    ax1.axhline(greedy_wer * 100, color="red", linestyle="--", label=f"Greedy {greedy_wer*100:.2f}%")
    ax1.axhline(oracle_wer * 100, color="blue", linestyle="--", label=f"Oracle {oracle_wer*100:.2f}%")
    ax1.set_xlabel("lam (LM weight)")
    ax1.set_ylabel("WER (%)")
    ax1.set_title("Normalized Interpolation: (1-lam)*CTC/w + lam*LM/w")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    betas = [r["beta"] for r in unnorm_results]
    wers2 = [r["wer"] * 100 for r in unnorm_results]
    ax2.plot(betas, wers2, "s-", color="#e67e22", linewidth=2, markersize=6)
    ax2.axhline(greedy_wer * 100, color="red", linestyle="--", label=f"Greedy {greedy_wer*100:.2f}%")
    ax2.axhline(oracle_wer * 100, color="blue", linestyle="--", label=f"Oracle {oracle_wer*100:.2f}%")
    ax2.set_xlabel("beta (LM weight)")
    ax2.set_ylabel("WER (%)")
    ax2.set_title("Unnormalized: CTC + beta*LM")
    ax2.set_xscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_rescorer_ablation(ablation_results, greedy_wer, oracle_wer,
                           output_path: Path):
    names = sorted(ablation_results.keys(), key=lambda k: ablation_results[k]["wer"])
    wers = [ablation_results[n]["wer"] * 100 for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ecc71" if w < greedy_wer * 100 else "#e74c3c" for w in wers]
    bars = ax.barh(range(len(names)), wers, color=colors, edgecolor="black", linewidth=0.5)

    ax.axvline(greedy_wer * 100, color="red", linestyle="--", linewidth=1,
               label=f"Greedy {greedy_wer*100:.2f}%")
    ax.axvline(oracle_wer * 100, color="blue", linestyle="--", linewidth=1,
               label=f"Oracle {oracle_wer*100:.2f}%")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("_", " ") for n in names])
    ax.set_xlabel("WER (%)")
    ax.set_title("Rescorer Ablation with LM Features")
    ax.legend()

    for bar, w in zip(bars, wers):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{w:.2f}%", va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_master_comparison(all_methods, output_path: Path):
    names = sorted(all_methods.keys(), key=lambda k: all_methods[k])
    wers = [all_methods[n] * 100 for n in names]

    fig, ax = plt.subplots(figsize=(12, 7))

    greedy_wer = all_methods.get("Greedy", 0.0602) * 100
    oracle_wer = all_methods.get("Oracle", 0.0444) * 100

    colors = []
    for n, w in zip(names, wers):
        if n == "Oracle":
            colors.append("#3498db")
        elif n == "Greedy":
            colors.append("#95a5a6")
        elif w < greedy_wer:
            colors.append("#2ecc71")
        else:
            colors.append("#e74c3c")

    bars = ax.barh(range(len(names)), wers, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(greedy_wer, color="red", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("WER (%)")
    ax.set_title("Master Comparison  --  All Levels (dev-other)")

    for bar, w in zip(bars, wers):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{w:.2f}%", va="center", fontsize=7)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")



def generate_report(norm_results, unnorm_results, mbr_results,
                    ablation_results, all_methods, greedy_wer, oracle_wer,
                    lm_name, elapsed, output_dir: Path):
    gap = greedy_wer - oracle_wer

    def gc(wer):
        return (greedy_wer - wer) / gap * 100 if gap > 0 else 0.0

    best_norm = min(norm_results, key=lambda r: r["wer"])
    best_unnorm = min(unnorm_results, key=lambda r: r["wer"])
    best_mbr = min(mbr_results, key=lambda r: r["wer"]) if mbr_results else None

    lines = []
    lines.append(f"# Level 4: N-gram LM Rescoring\n")
    lines.append(f"**Dataset:** LibriSpeech dev-other (2864 utterances)")
    lines.append(f"**Model:** Zipformer-S CR-CTC, BPE-500")
    lines.append(f"**N-best:** G=16, nbest_scale=1.0")
    lines.append(f"**LM:** {lm_name}")
    lines.append(f"**Greedy WER:** {greedy_wer*100:.2f}%")
    lines.append(f"**Oracle WER:** {oracle_wer*100:.2f}%")
    lines.append(f"**Oracle gap:** {gap*100:.2f} pp ({gap/greedy_wer*100:.1f}% relative)\n")

    # Normalized interpolation
    lines.append("## Step 3: Normalized Interpolation\n")
    lines.append("Score = (1-lam)*CTC_per_word + lam*LM_per_word\n")
    lines.append("| lam | WER% | Gap Closed |")
    lines.append("|--:|-----:|-----------:|")
    for r in norm_results:
        lines.append(f"| {r['lambda']:.1f} | {r['wer']*100:.2f}% | {gc(r['wer']):+.1f}% |")
    lines.append(f"\n**Best:** lam={best_norm['lambda']:.1f} -> WER={best_norm['wer']*100:.2f}% "
                 f"(gap closed: {gc(best_norm['wer']):.1f}%)\n")

    # Unnormalized
    lines.append("## Unnormalized Interpolation\n")
    lines.append("Score = CTC_log_prob + beta*LM_log_prob\n")
    lines.append("| beta | WER% | Gap Closed |")
    lines.append("|--:|-----:|-----------:|")
    for r in unnorm_results:
        lines.append(f"| {r['beta']:.1f} | {r['wer']*100:.2f}% | {gc(r['wer']):+.1f}% |")
    lines.append(f"\n**Best:** beta={best_unnorm['beta']:.1f} -> WER={best_unnorm['wer']*100:.2f}% "
                 f"(gap closed: {gc(best_unnorm['wer']):.1f}%)\n")

    # MBR
    if mbr_results:
        lines.append("## Step 4: MBR-CER with LM-adjusted Weights\n")
        lines.append("| beta | tau | WER% | Gap Closed |")
        lines.append("|--:|--:|-----:|-----------:|")
        for r in mbr_results:
            lines.append(f"| {r['beta']:.1f} | {r['tau']} | {r['wer']*100:.2f}% | {gc(r['wer']):+.1f}% |")
        if best_mbr:
            lines.append(f"\n**Best MBR:** beta={best_mbr['beta']:.1f} tau={best_mbr['tau']} -> "
                         f"WER={best_mbr['wer']*100:.2f}% (gap closed: {gc(best_mbr['wer']):.1f}%)\n")

    # Rescorer ablation
    if ablation_results:
        lines.append("## Step 5: Enhanced Feature Rescorer\n")
        lines.append("| Method | WER% | Gap Closed |")
        lines.append("|--------|-----:|-----------:|")
        for name in sorted(ablation_results.keys(), key=lambda k: ablation_results[k]["wer"]):
            r = ablation_results[name]
            label = name.replace("_", " ")
            lines.append(f"| {label} | {r['wer']*100:.2f}% | {gc(r['wer']):+.1f}% |")
        lines.append("")

    # Master comparison
    lines.append("## Master Comparison (All Levels)\n")
    lines.append("| Method | WER% | Gap Closed | External? |")
    lines.append("|--------|-----:|-----------:|-----------|")
    for name in sorted(all_methods.keys(), key=lambda k: all_methods[k]):
        wer = all_methods[name]
        ext = " -- " if name in ("Oracle", "Greedy") else (
            "Yes" if "LM" in name or "lm" in name or "4gram" in name or "3gram" in name
            else "No"
        )
        lines.append(f"| {name} | {wer*100:.2f}% | {gc(wer):+.1f}% | {ext} |")

    # Analysis
    lines.append("\n## Analysis\n")
    best_overall_wer = min(all_methods[k] for k in all_methods if k != "Oracle")
    best_overall_name = min(
        (k for k in all_methods if k != "Oracle"),
        key=lambda k: all_methods[k],
    )
    lines.append(f"**Best method overall:** {best_overall_name} "
                 f"(WER={best_overall_wer*100:.2f}%, gap closed: {gc(best_overall_wer):.1f}%)\n")

    if gc(best_norm["wer"]) > 0:
        lines.append("**LM interpolation succeeds where CTC-only methods failed.** "
                     "Simple n-gram rescoring closes a substantial portion of the oracle gap, "
                     "confirming that the information needed for correct hypothesis selection "
                     "is external to the CTC model.\n")
    else:
        lines.append("**Surprising result:** LM interpolation did not improve over greedy. "
                     "This may indicate the N-best list diversity is insufficient or the "
                     "LM is not well-matched to the domain.\n")

    lines.append(f"\n## Runtime\n")
    lines.append(f"- Total: {elapsed:.1f}s ({elapsed/60:.1f} min)\n")

    lines.append("## Generated Files\n")
    lines.append("- `lm_rescore_results.csv`  --  interpolation and MBR sweep data")
    lines.append("- `lm_features_dev.csv`  --  dev features with LM columns")
    lines.append("- `lm_features_train.csv`  --  train features with LM columns")
    lines.append("- `lm_rescorer_results.json`  --  rescorer ablation results")
    lines.append("- `plots/lm_lambda_sweep.png`  --  WER vs lam/beta")
    lines.append("- `plots/lm_rescorer_ablation.png`  --  ablation bar chart")
    lines.append("- `plots/lm_master_comparison.png`  --  all levels comparison")
    lines.append("- `level4_report.md`  --  this report")

    report_path = output_dir / "level4_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {report_path}")



def parse_args():
    parser = argparse.ArgumentParser(description="Level 4: LM rescoring")
    parser.add_argument("--lm-path", type=Path, required=True,
                        help="Path to ARPA LM file")
    parser.add_argument("--nbest-file", type=Path,
                        default=Path("results/nbest_dev_other_G16.jsonl"))
    parser.add_argument("--train-nbest-file", type=Path, default=None,
                        help="Train N-best JSONL for feature augmentation")
    parser.add_argument("--train-features", type=Path,
                        default=Path("results/level3_results/features_train.csv"))
    parser.add_argument("--dev-features", type=Path,
                        default=Path("results/level3_results/features_dev.csv"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--lm-name", type=str, default="3gram",
                        help="Label for LM (used in filenames/report)")
    parser.add_argument("--skip-rescorer", action="store_true",
                        help="Skip feature rescorer (Steps 5)")
    parser.add_argument("--skip-mbr", action="store_true",
                        help="Skip MBR sweep (Step 4)")
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    print("=" * 60)
    print(f"Level 4: N-gram LM Rescoring ({args.lm_name})")
    print("=" * 60)

    output_dir = args.results_dir
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading LM: {args.lm_path}")
    lm = kenlm.Model(str(args.lm_path))
    print(f"LM order: {lm.order}")

    records = load_nbest(args.nbest_file)

    # Baselines
    greedy_sels = {rec["utt_id"]: 0 for rec in records}
    oracle_sels = {}
    for rec in records:
        wers = [compute_wer(c["text"], rec["ref_text"]) for c in rec["candidates"]]
        oracle_sels[rec["utt_id"]] = int(np.argmin(wers))

    greedy_wer = corpus_wer_from_selections(records, greedy_sels)
    oracle_wer = corpus_wer_from_selections(records, oracle_sels)
    gap = greedy_wer - oracle_wer
    print(f"\nGreedy WER: {greedy_wer*100:.2f}%")
    print(f"Oracle WER: {oracle_wer*100:.2f}%")
    print(f"Gap: {gap*100:.2f} pp ({gap/greedy_wer*100:.1f}% relative)\n")

    # Score with LM
    print("Scoring candidates with LM...")
    score_candidates_with_lm(records, lm)

    # Quick LM sanity check
    sample = records[0]["candidates"][0]
    print(f"  Sample: '{sample['text'][:60]}...'")
    print(f"  LM log-prob: {sample['lm_log_prob']:.2f}")
    print(f"  LM ppl: {sample['lm_perplexity']:.1f}")

    print("\n" + "=" * 40)
    print("NORMALIZED INTERPOLATION SWEEP")
    print("=" * 40)
    lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    norm_results = interpolation_sweep_normalized(records, lambdas)

    print("\n" + "=" * 40)
    print("UNNORMALIZED INTERPOLATION SWEEP")
    print("=" * 40)
    betas = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    unnorm_results = interpolation_sweep_unnormalized(records, betas)

    mbr_results = []
    if not args.skip_mbr:
        print("\n" + "=" * 40)
        print("MBR-CER WITH LM-ADJUSTED WEIGHTS")
        print("=" * 40)
        mbr_betas = [0.0, 0.5, 1.0, 2.0]
        mbr_taus = [1.0, 5.0, 50.0, float("inf")]
        mbr_results = mbr_sweep(records, mbr_betas, mbr_taus)

    csv_path = output_dir / "lm_rescore_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "method", "lambda", "beta", "tau", "wer", "gap_closed_pct",
        ])
        writer.writeheader()
        for r in norm_results:
            gc = (greedy_wer - r["wer"]) / gap * 100 if gap > 0 else 0.0
            writer.writerow({
                "method": "normalized", "lambda": r["lambda"],
                "beta": "", "tau": "", "wer": f"{r['wer']:.6f}",
                "gap_closed_pct": f"{gc:.2f}",
            })
        for r in unnorm_results:
            gc = (greedy_wer - r["wer"]) / gap * 100 if gap > 0 else 0.0
            writer.writerow({
                "method": "unnormalized", "lambda": "",
                "beta": r["beta"], "tau": "", "wer": f"{r['wer']:.6f}",
                "gap_closed_pct": f"{gc:.2f}",
            })
        for r in mbr_results:
            gc = (greedy_wer - r["wer"]) / gap * 100 if gap > 0 else 0.0
            writer.writerow({
                "method": "mbr_lm", "lambda": "",
                "beta": r["beta"], "tau": r["tau"],
                "wer": f"{r['wer']:.6f}", "gap_closed_pct": f"{gc:.2f}",
            })
    print(f"\nSaved: {csv_path}")

    ablation_results = {}
    if not args.skip_rescorer:
        print("\n" + "=" * 40)
        print("ENHANCED FEATURE RESCORER (LM features)")
        print("=" * 40)

        # Augment dev features
        dev_feat_path = args.dev_features
        lm_dev_path = output_dir / "lm_features_dev.csv"
        if dev_feat_path.exists():
            augment_features_csv(dev_feat_path, records, lm_dev_path)
        else:
            print(f"WARNING: {dev_feat_path} not found, skipping rescorer")
            args.skip_rescorer = True

        # Augment train features
        train_feat_path = args.train_features
        lm_train_path = output_dir / "lm_features_train.csv"
        if not args.skip_rescorer and train_feat_path.exists():
            # Score train nbest with LM if available
            if args.train_nbest_file and args.train_nbest_file.exists():
                train_records = load_nbest(args.train_nbest_file)
                score_candidates_with_lm(train_records, lm)
                augment_features_csv(train_feat_path, train_records, lm_train_path)
            else:
                print("No train N-best JSONL  --  scoring train texts from features CSV directly")
                score_train_features_with_lm(train_feat_path, lm, lm_train_path)
        elif not args.skip_rescorer:
            print(f"WARNING: {train_feat_path} not found, skipping rescorer")
            args.skip_rescorer = True

    if not args.skip_rescorer and lm_train_path.exists() and lm_dev_path.exists():
        print("\nTraining rescopers with LM features...")
        utt_ids_train, _, X_train, y_train = load_features(lm_train_path, ALL_FEATURE_NAMES)
        utt_ids_dev, _, X_dev, y_dev = load_features(lm_dev_path, ALL_FEATURE_NAMES)
        print(f"Train: {X_train.shape}, Dev: {X_dev.shape}")

        ablation_results = run_rescorer_ablation(
            X_train, y_train, X_dev, y_dev,
            utt_ids_train, utt_ids_dev, args.nbest_file,
        )

        # Feature importance for all-18 ridge
        wer_full, importance_full, _, _ = train_and_eval_rescorer(
            X_train, y_train, X_dev, y_dev,
            utt_ids_dev, args.nbest_file, ALL_FEATURE_NAMES, "ridge",
        )

        if importance_full:
            print("\nFeature importance (Ridge, all 18 features):")
            for name, coef in importance_full:
                lm_tag = " [LM]" if name in LM_FEATURE_NAMES else ""
                print(f"  {name:<28} {coef:+.4f}{lm_tag}")

        rescorer_json = {name: {"wer": round(r["wer"], 6)} for name, r in ablation_results.items()}
        json_path = output_dir / "lm_rescorer_results.json"
        with open(json_path, "w") as f:
            json.dump(rescorer_json, f, indent=2)
        print(f"Saved: {json_path}")

        plot_rescorer_ablation(ablation_results, greedy_wer, oracle_wer,
                               plot_dir / "lm_rescorer_ablation.png")

    gc_fn = lambda w: (greedy_wer - w) / gap * 100 if gap > 0 else 0.0

    all_methods = {
        "Oracle": oracle_wer,
        "Greedy": greedy_wer,
        "MBR-CER tau=50 (L1b)": 0.0599,
        "MC-Dropout T=4 MBR (L1.5)": 0.0598,
        "MLP rescorer CTC (L3)": 0.0605,
    }

    best_norm_r = min(norm_results, key=lambda r: r["wer"])
    all_methods[f"LM interp best (lam={best_norm_r['lambda']:.1f})"] = best_norm_r["wer"]

    best_unnorm_r = min(unnorm_results, key=lambda r: r["wer"])
    all_methods[f"LM unnorm best (beta={best_unnorm_r['beta']:.1f})"] = best_unnorm_r["wer"]

    if mbr_results:
        best_mbr_r = min(mbr_results, key=lambda r: r["wer"])
        all_methods[f"LM-MBR best (beta={best_mbr_r['beta']:.1f} tau={best_mbr_r['tau']})"] = best_mbr_r["wer"]

    if ablation_results:
        best_ab_name = min(ablation_results.keys(), key=lambda k: ablation_results[k]["wer"])
        best_ab_wer = ablation_results[best_ab_name]["wer"]
        all_methods[f"Rescorer {best_ab_name} (L4b)"] = best_ab_wer

    plot_lambda_sweep(norm_results, unnorm_results, greedy_wer, oracle_wer,
                      plot_dir / "lm_lambda_sweep.png")
    plot_master_comparison(all_methods, plot_dir / "lm_master_comparison.png")

    elapsed = time.time() - t0

    generate_report(norm_results, unnorm_results, mbr_results,
                    ablation_results, all_methods, greedy_wer, oracle_wer,
                    args.lm_name, elapsed, output_dir)

    print(f"\n{'='*60}")
    print(f"Level 4 complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}")


def score_train_features_with_lm(train_csv_path: Path, lm, output_path: Path):
    """Score train features using text reconstructed from the CSV.

    When we don't have the train N-best JSONL, we need to get the candidate
    texts. The feature CSV doesn't contain text, so we need to load the
    train nbest JSONL. If it's not available, we score using the dev nbest
    approach  --  but we need the texts.

    Fallback: create LM features from what we can. We'll read the existing
    feature CSV, and for rows where we can find the text in the train nbest,
    add LM scores. For others, use 0.
    """
    # Try loading train nbest
    possible_paths = [
        train_csv_path.parent / "nbest_train_clean100_G16.jsonl",
        Path("results/nbest_train_clean100_G16.jsonl"),
        Path("results/level3_results/nbest_train_clean100_G16.jsonl"),
    ]

    train_records = None
    for p in possible_paths:
        if p.exists():
            train_records = load_nbest(p)
            break

    if train_records is not None:
        score_candidates_with_lm(train_records, lm)
        augment_features_csv(train_csv_path, train_records, output_path)
        return

    # No train JSONL available  --  cannot score. Copy CSV with zero LM features.
    print("WARNING: No train N-best JSONL found. LM features will be zero for train set.")
    print("This means the rescorer cannot learn to use LM features effectively.")
    print("To fix: provide --train-nbest-file pointing to the train JSONL.")

    rows = []
    with open(train_csv_path) as f:
        reader = csv.DictReader(f)
        orig_fields = reader.fieldnames
        out_fields = list(orig_fields) + LM_FEATURE_NAMES
        for row in reader:
            for fn in LM_FEATURE_NAMES:
                row[fn] = "0.0"
            rows.append(row)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved (zero LM features): {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train and evaluate a discriminative feature rescorer on CTC N-best lists.

Trains Ridge regression, MLP, and pairwise ranking models on
train-clean-100 features, evaluates on dev-other, and runs ablation
studies over feature subsets.

Usage:
    python experiments/train_rescorer.py \
        --train-features results/features_train.csv \
        --dev-features results/features_dev.csv \
        --results-dir results
"""

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import editdistance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FEATURE_NAMES = [
    "ctc_log_prob",
    "ctc_log_prob_per_token",
    "ctc_log_prob_per_char",
    "ctc_rank",
    "len_tokens",
    "len_chars",
    "len_words",
    "len_deviation",
    "mean_cer_to_others",
    "mean_wer_to_others",
    "agrees_with_majority",
    "log_prob_gap",
    "ptilde",
    "entropy_of_group",
]

FEATURE_GROUPS = {
    "ctc_only": ["ctc_log_prob", "ctc_log_prob_per_token",
                 "ctc_log_prob_per_char", "ctc_rank"],
    "length_only": ["len_tokens", "len_chars", "len_words", "len_deviation"],
    "agreement_only": ["mean_cer_to_others", "mean_wer_to_others",
                       "agrees_with_majority"],
    "prob_only": ["log_prob_gap", "ptilde", "entropy_of_group"],
    "ctc_plus_agreement": ["ctc_log_prob", "ctc_log_prob_per_token",
                           "ctc_log_prob_per_char", "ctc_rank",
                           "mean_cer_to_others", "mean_wer_to_others",
                           "agrees_with_majority"],
    "all_features": FEATURE_NAMES,
}

def load_features(path: Path):
    """Load feature CSV. Returns utt_ids, candidate_idxs, X, y."""
    utt_ids = []
    cand_idxs = []
    X_rows = []
    y_rows = []

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            utt_ids.append(row["utt_id"])
            cand_idxs.append(int(row["candidate_idx"]))
            feats = [float(row[fn]) for fn in FEATURE_NAMES]
            X_rows.append(feats)
            y_rows.append(float(row["wer"]))

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_rows, dtype=np.float64)
    print(f"Loaded {path}: {len(utt_ids)} rows, {X.shape[1]} features")
    return utt_ids, cand_idxs, X, y

def group_by_utterance(utt_ids, cand_idxs, X, y):
    """Group rows by utterance. Returns dict: utt_id -> (indices, X_group, y_group)."""
    groups = defaultdict(list)
    for i, uid in enumerate(utt_ids):
        groups[uid].append(i)

    result = {}
    for uid, idxs in groups.items():
        idxs = np.array(idxs)
        result[uid] = (idxs, X[idxs], y[idxs])
    return result

def compute_corpus_wer_from_selections(groups, y_all, nbest_file_path: Path = None,
                                       selections: dict = None):
    """Compute corpus-level WER from per-utterance candidate selections.

    If nbest_file_path is provided, uses actual texts and references for
    accurate corpus WER. Otherwise, returns mean of per-utterance WERs
    (approximation).
    """
    if nbest_file_path and nbest_file_path.exists() and selections:
        return _corpus_wer_from_nbest(nbest_file_path, selections)

    total_wer = 0.0
    n = 0
    for uid, (idxs, X_g, y_g) in groups.items():
        if uid in selections:
            best = selections[uid]
        else:
            best = int(np.argmin(y_g))
        total_wer += y_g[best]
        n += 1
    return total_wer / max(n, 1)

def _corpus_wer_from_nbest(nbest_path: Path, selections: dict):
    """Compute true corpus WER using original texts."""
    total_edits = 0
    total_ref_words = 0

    with open(nbest_path) as f:
        for line in f:
            rec = json.loads(line)
            uid = rec["utt_id"]
            if uid not in selections:
                sel_idx = 0
            else:
                sel_idx = min(selections[uid], len(rec["candidates"]) - 1)

            hyp = rec["candidates"][sel_idx]["text"]
            ref = rec["ref_text"]
            ref_words = ref.split()
            hyp_words = hyp.split()
            total_edits += editdistance.eval(hyp_words, ref_words)
            total_ref_words += len(ref_words)

    return total_edits / max(total_ref_words, 1)

def select_by_model(groups, X_all, model, scaler, feature_indices=None):
    """For each utterance, select the candidate with lowest predicted WER."""
    selections = {}
    for uid, (idxs, X_g, y_g) in groups.items():
        if feature_indices is not None:
            X_input = X_g[:, feature_indices]
        else:
            X_input = X_g
        X_scaled = scaler.transform(X_input)
        pred = model.predict(X_scaled)
        best_local = int(np.argmin(pred))
        selections[uid] = best_local
    return selections

def select_greedy(groups):
    return {uid: 0 for uid in groups}

def select_oracle(groups):
    """Select best candidate (lowest WER) for each utterance."""
    selections = {}
    for uid, (idxs, X_g, y_g) in groups.items():
        selections[uid] = int(np.argmin(y_g))
    return selections

def train_ridge(X_train, y_train, alpha=1.0):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = Ridge(alpha=alpha)
    model.fit(X_scaled, y_train)
    return model, scaler

def train_mlp(X_train, y_train):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    model.fit(X_scaled, y_train)
    return model, scaler

def train_pairwise(X_train, y_train, utt_ids_train):
    """Train a pairwise ranking model using logistic regression on feature diffs."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    groups = defaultdict(list)
    for i, uid in enumerate(utt_ids_train):
        groups[uid].append(i)

    X_pairs = []
    y_pairs = []

    rng = np.random.RandomState(42)
    for uid, idxs in groups.items():
        if len(idxs) < 2:
            continue
        n = len(idxs)
        max_pairs = min(n * (n - 1) // 2, 20)

        pairs_added = 0
        for a_local in range(n):
            for b_local in range(a_local + 1, n):
                if pairs_added >= max_pairs:
                    break
                ia, ib = idxs[a_local], idxs[b_local]
                diff = X_train[ia] - X_train[ib]
                if y_train[ia] < y_train[ib]:
                    label = 1
                elif y_train[ia] > y_train[ib]:
                    label = 0
                else:
                    continue
                X_pairs.append(diff)
                y_pairs.append(label)
                pairs_added += 1
            if pairs_added >= max_pairs:
                break

    if len(X_pairs) < 10:
        return None, None

    X_pairs = np.array(X_pairs)
    y_pairs = np.array(y_pairs)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pairs)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y_pairs)

    print(f"Pairwise: {len(X_pairs)} pairs, accuracy={model.score(X_scaled, y_pairs):.3f}")
    return model, scaler

def select_by_pairwise(groups, X_all, model, scaler, feature_indices=None):
    """Select candidate that wins the most pairwise comparisons."""
    selections = {}
    for uid, (idxs, X_g, y_g) in groups.items():
        n = len(idxs)
        if n == 1:
            selections[uid] = 0
            continue

        if feature_indices is not None:
            X_input = X_g[:, feature_indices]
        else:
            X_input = X_g

        wins = np.zeros(n)
        for i in range(n):
            for j in range(i + 1, n):
                diff = X_input[i] - X_input[j]
                diff_scaled = scaler.transform(diff.reshape(1, -1))
                prob_i_better = model.predict_proba(diff_scaled)[0, 1]
                if prob_i_better > 0.5:
                    wins[i] += 1
                else:
                    wins[j] += 1

        selections[uid] = int(np.argmax(wins))
    return selections

def cross_validate_ridge(X, y, utt_ids, n_folds=5, alpha=1.0):
    """5-fold CV on train data grouped by utterance to check overfitting."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    unique_utts = list(set(utt_ids))
    rng = np.random.RandomState(42)
    rng.shuffle(unique_utts)

    fold_size = len(unique_utts) // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(unique_utts)
        folds.append(set(unique_utts[start:end]))

    utt_to_idx = defaultdict(list)
    for i, uid in enumerate(utt_ids):
        utt_to_idx[uid].append(i)

    r2_scores = []
    fold_wers = []

    for fold_i in range(n_folds):
        val_utts = folds[fold_i]
        val_idxs = []
        train_idxs = []
        for uid, idxs in utt_to_idx.items():
            if uid in val_utts:
                val_idxs.extend(idxs)
            else:
                train_idxs.extend(idxs)

        train_idxs = np.array(train_idxs)
        val_idxs = np.array(val_idxs)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idxs])
        X_va = scaler.transform(X[val_idxs])

        model = Ridge(alpha=alpha)
        model.fit(X_tr, y[train_idxs])

        pred = model.predict(X_va)
        ss_res = np.sum((y[val_idxs] - pred) ** 2)
        ss_tot = np.sum((y[val_idxs] - np.mean(y[val_idxs])) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
        r2_scores.append(r2)

        val_groups = defaultdict(list)
        for idx in val_idxs:
            val_groups[utt_ids[idx]].append(idx)

        fold_wer_sum = 0.0
        fold_n = 0
        for uid, group_idxs in val_groups.items():
            group_idxs = np.array(group_idxs)
            X_g = scaler.transform(X[group_idxs])
            p = model.predict(X_g)
            best = int(np.argmin(p))
            fold_wer_sum += y[group_idxs[best]]
            fold_n += 1
        fold_wers.append(fold_wer_sum / max(fold_n, 1))

    return {
        "r2_mean": float(np.mean(r2_scores)),
        "r2_std": float(np.std(r2_scores)),
        "r2_per_fold": [float(r) for r in r2_scores],
        "wer_mean": float(np.mean(fold_wers)),
        "wer_std": float(np.std(fold_wers)),
        "wer_per_fold": [float(w) for w in fold_wers],
    }

def per_utterance_analysis(groups_dev, selections, greedy_sels, oracle_sels):
    """Classify utterances as improved / degraded / same vs greedy."""
    improved = 0
    degraded = 0
    same = 0
    recoverable_total = 0
    recoverable_recovered = 0

    utt_details = []

    for uid, (idxs, X_g, y_g) in groups_dev.items():
        greedy_wer = y_g[greedy_sels[uid]]
        oracle_wer = y_g[oracle_sels[uid]]
        model_wer = y_g[selections[uid]]

        is_recoverable = abs(greedy_wer - oracle_wer) > 1e-6

        if model_wer < greedy_wer - 1e-6:
            improved += 1
        elif model_wer > greedy_wer + 1e-6:
            degraded += 1
        else:
            same += 1

        if is_recoverable:
            recoverable_total += 1
            if model_wer < greedy_wer - 1e-6:
                recoverable_recovered += 1

        utt_details.append({
            "utt_id": uid,
            "greedy_wer": float(greedy_wer),
            "oracle_wer": float(oracle_wer),
            "model_wer": float(model_wer),
            "improved": model_wer < greedy_wer - 1e-6,
            "degraded": model_wer > greedy_wer + 1e-6,
        })

    return {
        "improved": improved,
        "degraded": degraded,
        "same": same,
        "total": improved + degraded + same,
        "recoverable_total": recoverable_total,
        "recoverable_recovered": recoverable_recovered,
        "recovery_rate": recoverable_recovered / max(recoverable_total, 1),
        "details": utt_details,
    }

def plot_feature_importance(model, scaler, feature_names, output_path: Path):
    """Plot Ridge coefficient magnitudes (after standardization)."""
    coefs = model.coef_
    abs_coefs = np.abs(coefs)
    order = np.argsort(abs_coefs)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    names = [feature_names[i] for i in order]
    values = [coefs[i] for i in order]
    colors = ["#2ecc71" if v < 0 else "#e74c3c" for v in values]

    ax.barh(range(len(names)), [abs(v) for v in values], color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Absolute coefficient (standardized)")
    ax.set_title("Ridge Rescorer: Feature Importance\n(green = lower coeff -> lower predicted WER)")

    for i, (n, v) in enumerate(zip(names, values)):
        ax.text(abs(v) + 0.001, i, f"{v:+.4f}", va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")

def plot_rescorer_comparison(results_dict, output_path: Path):
    """Bar chart comparing all rescoring methods."""
    names = list(results_dict.keys())
    wers = [results_dict[n]["wer"] * 100 for n in names]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(names)), wers, color="#3498db", edgecolor="black", linewidth=0.5)

    greedy_wer = results_dict.get("greedy", {}).get("wer", 0) * 100
    oracle_wer = results_dict.get("oracle", {}).get("wer", 0) * 100
    if greedy_wer > 0:
        ax.axhline(y=greedy_wer, color="red", linestyle="--", linewidth=1, label=f"Greedy: {greedy_wer:.2f}%")
    if oracle_wer > 0:
        ax.axhline(y=oracle_wer, color="green", linestyle="--", linewidth=1, label=f"Oracle: {oracle_wer:.2f}%")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("WER (%)")
    ax.set_title("Rescorer Comparison  --  dev-other G=16")
    ax.legend()

    for bar, wer_val in zip(bars, wers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{wer_val:.2f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")

def plot_per_utterance(utt_details, output_path: Path):
    """Scatter plot: greedy WER vs model WER per utterance."""
    greedy = [u["greedy_wer"] for u in utt_details]
    model = [u["model_wer"] for u in utt_details]

    fig, ax = plt.subplots(figsize=(8, 8))

    max_val = max(max(greedy), max(model)) * 1.05
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=0.5, alpha=0.5, label="y=x")

    improved = [(g, m) for g, m in zip(greedy, model) if m < g - 1e-6]
    degraded = [(g, m) for g, m in zip(greedy, model) if m > g + 1e-6]
    same = [(g, m) for g, m in zip(greedy, model) if abs(m - g) <= 1e-6]

    if same:
        ax.scatter(*zip(*same), alpha=0.3, s=10, c="gray", label=f"Same ({len(same)})")
    if improved:
        ax.scatter(*zip(*improved), alpha=0.5, s=15, c="green", label=f"Improved ({len(improved)})")
    if degraded:
        ax.scatter(*zip(*degraded), alpha=0.5, s=15, c="red", label=f"Degraded ({len(degraded)})")

    ax.set_xlabel("Greedy WER")
    ax.set_ylabel("Rescorer WER")
    ax.set_title("Per-Utterance: Greedy vs Rescorer WER")
    ax.legend()
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")

def compute_corpus_wer(nbest_path: Path, groups, selections):
    """Compute corpus WER using original JSONL texts."""
    if not nbest_path.exists():
        wer_sum = 0.0
        n = 0
        for uid, (idxs, X_g, y_g) in groups.items():
            sel = selections.get(uid, 0)
            wer_sum += y_g[sel]
            n += 1
        return wer_sum / max(n, 1)

    return _corpus_wer_from_nbest(nbest_path, selections)

def generate_report(all_results, cv_results, per_utt, feature_importance,
                    elapsed, output_dir: Path):
    greedy_wer = all_results["greedy"]["wer"]
    oracle_wer = all_results["oracle"]["wer"]
    gap = greedy_wer - oracle_wer

    best_name = None
    best_wer = greedy_wer
    for name, res in all_results.items():
        if name in ("greedy", "oracle"):
            continue
        if res["wer"] < best_wer:
            best_wer = res["wer"]
            best_name = name

    lines = []
    lines.append("# Level 3: Discriminative Feature Rescorer\n")
    lines.append(f"**Dataset:** LibriSpeech dev-other (2864 utterances)")
    lines.append(f"**Training data:** train-clean-100")
    lines.append(f"**Model:** Zipformer-S CR-CTC, BPE-500")
    lines.append(f"**N-best:** G=16, nbest_scale=1.0\n")

    lines.append("## Approach\n")
    lines.append("A learned value function (discriminative rescorer) that predicts")
    lines.append("hypothesis quality from CTC-derived features. For each utterance,")
    lines.append("the rescorer predicts WER for all G candidates and selects the one")
    lines.append("with the lowest predicted WER. This is equivalent to the Q-function")
    lines.append("of the one-step selection MDP.\n")
    lines.append("14 features are extracted per candidate: CTC log-probability variants (4),")
    lines.append("length statistics (4), inter-candidate agreement scores (3), and")
    lines.append("probability-distribution features (3).\n")

    lines.append("## Results\n")
    lines.append(f"| {'Method':<28} | {'WER%':>8} | {'Gap Closed%':>12} |")
    lines.append(f"|{'-'*30}|{'-'*10}|{'-'*14}|")

    for name in sorted(all_results.keys(), key=lambda n: all_results[n]["wer"]):
        r = all_results[name]
        label = name.replace("_", " ").title()
        gc = (greedy_wer - r["wer"]) / gap * 100 if gap > 0 else 0.0
        lines.append(f"| {label:<28} | {r['wer']*100:>7.2f}% | {gc:>11.1f}% |")

    lines.append(f"\n**Greedy WER:** {greedy_wer*100:.2f}%")
    lines.append(f"**Oracle WER:** {oracle_wer*100:.2f}%")
    lines.append(f"**Oracle gap:** {gap*100:.2f} pp ({(gap/greedy_wer*100):.1f}% relative)")

    if best_name:
        best_gc = (greedy_wer - best_wer) / gap * 100 if gap > 0 else 0.0
        lines.append(f"\n**Best rescorer:** {best_name} (WER {best_wer*100:.2f}%, "
                      f"closes {best_gc:.1f}% of oracle gap)")
    else:
        lines.append(f"\n**Best rescorer:** None improved over greedy")

    lines.append("\n## Feature Importance (Ridge)\n")
    if feature_importance:
        lines.append(f"| {'Rank':>4} | {'Feature':<28} | {'Coefficient':>12} |")
        lines.append(f"|{'-'*6}|{'-'*30}|{'-'*14}|")
        for rank, (name, coef) in enumerate(feature_importance, 1):
            lines.append(f"| {rank:>4} | {name:<28} | {coef:>+11.4f} |")
        lines.append("")
        lines.append("Positive coefficients increase predicted WER (bad features).")
        lines.append("Negative coefficients decrease predicted WER (good features).")

    lines.append("\n## Ablation Study\n")
    lines.append("Feature subset ablations show which signal sources contribute:\n")
    for group_name in ["ctc_only", "length_only", "agreement_only",
                       "prob_only", "ctc_plus_agreement", "all_features"]:
        key = f"ridge_{group_name}"
        if key in all_results:
            r = all_results[key]
            gc = (greedy_wer - r["wer"]) / gap * 100 if gap > 0 else 0.0
            lines.append(f"- **{group_name}:** WER {r['wer']*100:.2f}% "
                          f"(gap closed: {gc:.1f}%)")

    lines.append("\n## Cross-Validation (train-clean-100)\n")
    if cv_results:
        lines.append(f"- **R^2 mean:** {cv_results['r2_mean']:.4f} +/- {cv_results['r2_std']:.4f}")
        lines.append(f"- **R^2 per fold:** {', '.join(f'{r:.4f}' for r in cv_results['r2_per_fold'])}")
        lines.append(f"- **WER mean (CV):** {cv_results['wer_mean']*100:.2f}% +/- {cv_results['wer_std']*100:.2f}%")
        r2 = cv_results["r2_mean"]
        if r2 > 0.3:
            lines.append(f"\nR^2 = {r2:.3f} indicates features have strong predictive power for WER.")
        elif r2 > 0.1:
            lines.append(f"\nR^2 = {r2:.3f} indicates features have moderate predictive power.")
        else:
            lines.append(f"\nR^2 = {r2:.3f} is low  --  features weakly predict WER. "
                          "The rescorer may not generalize well.")

    lines.append("\n## Per-Utterance Analysis\n")
    if per_utt:
        lines.append(f"- **Improved:** {per_utt['improved']} utterances")
        lines.append(f"- **Degraded:** {per_utt['degraded']} utterances")
        lines.append(f"- **Same:** {per_utt['same']} utterances")
        lines.append(f"- **Net:** {per_utt['improved'] - per_utt['degraded']:+d} utterances")
        lines.append(f"- **Recoverable utterances:** {per_utt['recoverable_total']}")
        lines.append(f"- **Recovered:** {per_utt['recoverable_recovered']} "
                      f"({per_utt['recovery_rate']*100:.1f}%)")

    lines.append("\n## Master Comparison (All Levels)\n")
    lines.append("Including results from Level 1 and Level 1b:\n")
    lines.append(f"| {'Method':<28} | {'WER%':>8} | {'Gap Closed%':>12} | {'Source':>10} |")
    lines.append(f"|{'-'*30}|{'-'*10}|{'-'*14}|{'-'*12}|")

    master = {}
    master["Greedy"] = (greedy_wer, 0.0, "L1")
    master["Oracle"] = (oracle_wer, 100.0, "L1")

    for name, r in all_results.items():
        if name in ("greedy", "oracle"):
            continue
        gc = (greedy_wer - r["wer"]) / gap * 100 if gap > 0 else 0.0
        label = name.replace("_", " ").title()
        source = "L3" if "ridge" in name or "mlp" in name or "pairwise" in name else "L1"
        master[label] = (r["wer"], gc, source)

    for label in sorted(master.keys(), key=lambda k: master[k][0]):
        wer_val, gc, source = master[label]
        lines.append(f"| {label:<28} | {wer_val*100:>7.2f}% | {gc:>11.1f}% | {source:>10} |")

    lines.append(f"\n## Runtime\n")
    lines.append(f"- Feature extraction + training + evaluation: {elapsed:.1f}s")

    lines.append(f"\n## Output Files\n")
    lines.append(f"- `features_train.csv`  --  training features")
    lines.append(f"- `features_dev.csv`  --  dev features")
    lines.append(f"- `rescorer_results.json`  --  all model performances")
    lines.append(f"- `plots/feature_importance.png`")
    lines.append(f"- `plots/rescorer_comparison.png`")
    lines.append(f"- `plots/rescorer_per_utterance.png`")
    lines.append(f"- `level3_report.md`  --  this report")

    report_path = output_dir / "level3_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {report_path}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate discriminative rescorer"
    )
    parser.add_argument(
        "--train-features", type=Path,
        default=Path("results/features_train.csv"),
    )
    parser.add_argument(
        "--dev-features", type=Path,
        default=Path("results/features_dev.csv"),
    )
    parser.add_argument(
        "--dev-nbest", type=Path,
        default=Path("results/nbest_dev_other_G16.jsonl"),
    )
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path("results"),
    )
    parser.add_argument(
        "--skip-mlp", action="store_true",
        help="Skip MLP training (faster)",
    )
    parser.add_argument(
        "--skip-pairwise", action="store_true",
        help="Skip pairwise ranking (faster)",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    t0 = time.time()

    print("=" * 60)
    print("Level 3: Discriminative Feature Rescorer")
    print("=" * 60)

    utt_ids_train, cand_idxs_train, X_train, y_train = load_features(args.train_features)
    utt_ids_dev, cand_idxs_dev, X_dev, y_dev = load_features(args.dev_features)

    groups_train = group_by_utterance(utt_ids_train, cand_idxs_train, X_train, y_train)
    groups_dev = group_by_utterance(utt_ids_dev, cand_idxs_dev, X_dev, y_dev)

    nbest_path = args.dev_nbest
    all_results = {}

    greedy_sels = select_greedy(groups_dev)
    oracle_sels = select_oracle(groups_dev)

    greedy_wer = compute_corpus_wer(nbest_path, groups_dev, greedy_sels)
    oracle_wer = compute_corpus_wer(nbest_path, groups_dev, oracle_sels)

    all_results["greedy"] = {"wer": greedy_wer}
    all_results["oracle"] = {"wer": oracle_wer}
    print(f"\nGreedy WER: {greedy_wer*100:.2f}%")
    print(f"Oracle WER: {oracle_wer*100:.2f}%")
    print(f"Gap: {(greedy_wer - oracle_wer)*100:.2f} pp\n")

    print("=" * 40)
    print("ABLATION STUDY")
    print("=" * 40)

    for group_name, feat_list in FEATURE_GROUPS.items():
        feat_indices = [FEATURE_NAMES.index(f) for f in feat_list]
        X_tr_sub = X_train[:, feat_indices]
        X_dev_sub = X_dev[:, feat_indices]

        model, scaler = train_ridge(X_tr_sub, y_train)

        from sklearn.preprocessing import StandardScaler as SS
        scaler_for_select = SS()
        scaler_for_select.fit(X_tr_sub)

        sels = select_by_model(groups_dev, X_dev, model, scaler, feat_indices)
        wer = compute_corpus_wer(nbest_path, groups_dev, sels)

        gap = greedy_wer - oracle_wer
        gc = (greedy_wer - wer) / gap * 100 if gap > 0 else 0.0

        key = f"ridge_{group_name}"
        all_results[key] = {"wer": wer, "features": feat_list}
        print(f"  Ridge ({group_name}): WER {wer*100:.2f}% (gap closed: {gc:.1f}%)")

    print("\n" + "=" * 40)
    print("FULL RIDGE MODEL")
    print("=" * 40)

    model_ridge, scaler_ridge = train_ridge(X_train, y_train)
    sels_ridge = select_by_model(groups_dev, X_dev, model_ridge, scaler_ridge)
    wer_ridge = compute_corpus_wer(nbest_path, groups_dev, sels_ridge)
    print(f"Ridge (all): WER {wer_ridge*100:.2f}%")

    coefs = model_ridge.coef_
    importance = sorted(
        zip(FEATURE_NAMES, coefs),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    print("\nFeature importance (by |coefficient|):")
    for name, coef in importance:
        print(f"  {name:<28} {coef:+.4f}")

    wer_mlp = None
    if not args.skip_mlp:
        print("\n" + "=" * 40)
        print("MLP MODEL")
        print("=" * 40)

        model_mlp, scaler_mlp = train_mlp(X_train, y_train)
        sels_mlp = select_by_model(groups_dev, X_dev, model_mlp, scaler_mlp)
        wer_mlp = compute_corpus_wer(nbest_path, groups_dev, sels_mlp)
        all_results["mlp_all"] = {"wer": wer_mlp}
        print(f"MLP (all): WER {wer_mlp*100:.2f}%")

    if not args.skip_pairwise:
        print("\n" + "=" * 40)
        print("PAIRWISE RANKING MODEL")
        print("=" * 40)

        model_pw, scaler_pw = train_pairwise(X_train, y_train, utt_ids_train)
        if model_pw is not None:
            sels_pw = select_by_pairwise(groups_dev, X_dev, model_pw, scaler_pw)
            wer_pw = compute_corpus_wer(nbest_path, groups_dev, sels_pw)
            all_results["pairwise_all"] = {"wer": wer_pw}
            print(f"Pairwise (all): WER {wer_pw*100:.2f}%")
        else:
            print("Pairwise: not enough training pairs, skipped")

    print("\n" + "=" * 40)
    print("CROSS-VALIDATION (5-fold, train-clean-100)")
    print("=" * 40)

    cv_results = cross_validate_ridge(X_train, y_train, utt_ids_train)
    print(f"R^2 mean: {cv_results['r2_mean']:.4f} +/- {cv_results['r2_std']:.4f}")
    print(f"WER mean: {cv_results['wer_mean']*100:.2f}% +/- {cv_results['wer_std']*100:.2f}%")

    print("\n" + "=" * 40)
    print("PER-UTTERANCE ANALYSIS")
    print("=" * 40)

    best_model_name = "ridge_all_features"
    best_sels = sels_ridge

    if wer_mlp is not None and wer_mlp < wer_ridge:
        best_model_name = "mlp_all"
        best_sels = sels_mlp
    if "pairwise_all" in all_results and all_results["pairwise_all"]["wer"] < all_results.get(best_model_name, {}).get("wer", 1.0):
        best_model_name = "pairwise_all"
        best_sels = sels_pw

    per_utt = per_utterance_analysis(groups_dev, best_sels, greedy_sels, oracle_sels)
    print(f"Best model: {best_model_name}")
    print(f"Improved: {per_utt['improved']}, Degraded: {per_utt['degraded']}, Same: {per_utt['same']}")
    print(f"Net: {per_utt['improved'] - per_utt['degraded']:+d}")
    print(f"Recoverable: {per_utt['recoverable_total']}, Recovered: {per_utt['recoverable_recovered']} "
          f"({per_utt['recovery_rate']*100:.1f}%)")

    print("\n" + "=" * 40)
    print("VERIFICATION")
    print("=" * 40)

    ctc_only_wer = all_results.get("ridge_ctc_only", {}).get("wer", 0)
    print(f"1. CTC-only rescorer WER: {ctc_only_wer*100:.2f}% (greedy: {greedy_wer*100:.2f}%)")
    if abs(ctc_only_wer - greedy_wer) < 0.005:
        print("    Matches greedy (as expected)")
    else:
        print(f"    Differs from greedy by {abs(ctc_only_wer - greedy_wer)*100:.2f} pp")

    agr_wer = all_results.get("ridge_agreement_only", {}).get("wer", 0)
    print(f"2. Agreement-only WER: {agr_wer*100:.2f}%")

    full_wer = all_results.get("ridge_all_features", {}).get("wer", greedy_wer)
    print(f"3. Full rescorer WER: {full_wer*100:.2f}% vs greedy: {greedy_wer*100:.2f}%")
    if full_wer <= greedy_wer + 1e-6:
        print("    At least as good as greedy")
    else:
        print(f"    Worse than greedy by {(full_wer - greedy_wer)*100:.2f} pp")

    print(f"4. CV R^2 = {cv_results['r2_mean']:.4f}")

    len_only_wer = all_results.get("ridge_length_only", {}).get("wer", 0)
    print(f"5. Length-only WER: {len_only_wer*100:.2f}% (checks rescorer isn't just pick-shortest/longest)")

    elapsed = time.time() - t0

    output_dir = Path(args.results_dir)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    results_json = {}
    for name, r in all_results.items():
        results_json[name] = {
            "wer": round(r["wer"], 6),
            "gap_closed_pct": round(
                (greedy_wer - r["wer"]) / max(greedy_wer - oracle_wer, 1e-10) * 100, 2
            ),
        }
    results_json["cv_r2_mean"] = round(cv_results["r2_mean"], 4)
    results_json["cv_r2_std"] = round(cv_results["r2_std"], 4)

    json_path = output_dir / "rescorer_results.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved: {json_path}")

    plot_feature_importance(model_ridge, scaler_ridge, FEATURE_NAMES,
                            plot_dir / "feature_importance.png")
    plot_rescorer_comparison(all_results, plot_dir / "rescorer_comparison.png")
    plot_per_utterance(per_utt["details"], plot_dir / "rescorer_per_utterance.png")

    generate_report(all_results, cv_results, per_utt, importance, elapsed, output_dir)

    print(f"\nTotal time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")

if __name__ == "__main__":
    main()

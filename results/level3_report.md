# Level 3: Discriminative Feature Rescorer

**Dataset:** LibriSpeech dev-other (2864 utterances)
**Training data:** train-clean-100, shortest 10,000 utterances (subset)
**Model:** Zipformer-S CR-CTC, BPE-500
**N-best:** G=16, nbest_scale=1.0

## Headline finding

**A learned rescorer with 14 hand-crafted features fails to beat greedy
on dev-other.** The best model (MLP on all features) achieves 6.05% WER
vs. greedy 6.02% — a small *regression*, not an improvement. Every Ridge
ablation also lands above greedy. This continues the pattern from
Levels 1–2: at `nbest_scale=1.0`, the N-best list does not contain enough
exploitable diversity for any decode-time selection method to close the
oracle gap (1.58 pp absolute, 26.2% relative).

The features *do* predict WER in-distribution: 5-fold CV on
train-clean-100 gives R² = 0.76. But that signal does not transfer to
dev-other selection — the rescorer learns a function of (CTC log-prob,
agreement, length) whose argmin per utterance group is, on average,
neither the greedy nor the oracle pick.

## Results

| Method                       |     WER% |  Gap Closed% |
|------------------------------|----------|--------------|
| Oracle                       |    4.44% |       100.0% |
| **Greedy (baseline)**        |  **6.02%** |     **0.0%** |
| MLP (all features)           |    6.05% |        −1.5% |
| Pairwise rank (all)          |    6.07% |        −3.1% |
| Ridge CTC + agreement        |    6.10% |        −4.7% |
| Ridge agreement only         |    6.10% |        −5.2% |
| Ridge all features           |    6.14% |        −7.2% |
| Ridge CTC only               |    6.22% |       −12.5% |
| Ridge prob only              |    6.24% |       −13.5% |
| Ridge length only            |   12.21% |      −391.7% |

Per-utterance breakdown (best model = MLP):
- Improved: 54
- Degraded: 53
- Same:     2757
- Net:      +1 utterance (well within noise)
- Recoverable utterances (greedy ≠ oracle): 665
- Of those, recovered by rescorer: 54 (8.1%)

## Caveats

1. **Train subset is biased.** We used `--subset 10000` of train-clean-100,
   which selects the *shortest* 10k utterances by duration (sorted, not
   sampled). Short utterances tend to be easier and may have systematically
   different feature distributions than dev-other (which contains the full
   length range). A random 10k sample would be a fairer test, and the full
   28,539-utterance set fairer still.

2. **CV R² ≠ corpus WER.** The cross-validation R² of 0.76 measures how well
   the model predicts WER for individual candidates, *not* whether its
   per-utterance argmin recovers the oracle. The CV "WER mean" of 1.48%
   reported below is a per-utterance mean (dragged down by the many
   zero-WER utterances); the honest comparison is the corpus WER on
   dev-other (6.14% for Ridge), which is reported in the table above.

3. **Length-only ablation (12.21%) confirms the rescorer can't trivially
   game the metric** by always picking the shortest or longest candidate.
   That ablation is catastrophically bad, so the small gains/losses of the
   other ablations are real signal, not length artifacts.

4. **CTC-only Ridge (6.22%) is *worse* than greedy (6.02%).** Greedy = first
   candidate by CTC log-prob, so a Ridge model trained on CTC features
   should at minimum match greedy. It doesn't, because the linear model
   adds noise from the per-token / per-char / rank features that have weak
   correlation with WER on dev-other.

## Approach

A learned value function (discriminative rescorer) that predicts hypothesis
quality from CTC-derived features. For each dev-other utterance the rescorer
predicts WER for all G=16 candidates and selects the one with the lowest
predicted WER. This is the Q-function of the one-step selection MDP, but
that framing is decorative — operationally it is supervised regression on
N-best candidate features.

14 features per candidate:
- **CTC-derived (4):** ctc_log_prob, per-token, per-char, rank
- **Length (4):** len_tokens, len_chars, len_words, len_deviation
- **Agreement (3):** mean_cer_to_others, mean_wer_to_others,
  agrees_with_majority
- **Probability (3):** log_prob_gap, ptilde, entropy_of_group

No external LM; all features come from the model's own N-best output.

## Feature Importance (full Ridge model)

| Rank | Feature                      |  Coefficient |
|------|------------------------------|--------------|
|    1 | mean_wer_to_others           |     +0.0734 |
|    2 | log_prob_gap                 |     +0.0297 |
|    3 | ctc_log_prob                 |     −0.0265 |
|    4 | len_words                    |     +0.0174 |
|    5 | len_chars                    |     −0.0124 |
|    6 | ptilde                       |     −0.0110 |
|    7 | ctc_rank                     |     +0.0100 |
|    8 | ctc_log_prob_per_char        |     +0.0066 |
|    9 | entropy_of_group             |     +0.0062 |
|   10 | len_tokens                   |     +0.0060 |
|   11 | mean_cer_to_others           |     −0.0043 |
|   12 | ctc_log_prob_per_token       |     −0.0030 |
|   13 | agrees_with_majority         |     −0.0023 |
|   14 | len_deviation                |     −0.0019 |

Positive coefficient → higher feature value predicts higher WER (worse
candidate). The single most useful feature is `mean_wer_to_others` — i.e.
this is essentially the MBR-WER signal — but combining it with CTC
log-prob doesn't outperform either alone.

## Ablation Study

| Feature subset             |   WER% | Δ vs greedy |
|----------------------------|--------|-------------|
| ctc_only                   |  6.22% |       −0.20 |
| length_only                | 12.21% |       −6.19 |
| agreement_only             |  6.10% |       −0.08 |
| prob_only                  |  6.24% |       −0.22 |
| ctc + agreement            |  6.10% |       −0.08 |
| all_features               |  6.14% |       −0.12 |

`agreement_only` and `ctc + agreement` are tied as the best Ridge subsets,
both still slightly below greedy. Adding length and prob features makes
things worse — they introduce overfitting noise without contributing
signal.

## Cross-Validation (train-clean-100, 5 folds)

- R² mean: **0.7601 ± 0.0271**
- R² per fold: 0.7511, 0.7766, 0.7849, 0.7109, 0.7769
- Per-utt WER mean (CV, see caveat 2): 1.48% ± 0.05%

The high R² confirms the features are not random — they predict candidate
WER well in-distribution. The fact that this in-distribution skill does
not produce dev-other improvement suggests the limitation is the N-best
list itself (Level 1–2 finding) rather than the rescorer architecture.

## Conclusion for the coursework

This is a **negative result** that strengthens the Level 1–2 conclusion:
the bottleneck is not "CTC log-prob is one feature, more features will
help" — it is that at `nbest_scale=1.0` the N-best candidates are
near-duplicate alignment variants of the same hypothesis, leaving no
recoverable error mass for *any* selection rule (single feature or
learned combination) to exploit.

To get a positive rescorer result we would need either (a) more diverse
N-best lists (lower `nbest_scale`, or contrastive/MC-dropout sampling
from Level 1.5), or (b) features that look outside the N-best list
(external LM scores, acoustic confidence, etc.). Both are out of scope
for a single-feature, decoder-only setup.

## Master Comparison (all levels)

| Method                       |     WER% |  Gap Closed% | Source |
|------------------------------|----------|--------------|--------|
| Oracle                       |    4.44% |       100.0% |   L1   |
| **Greedy**                   |  **6.02%** |     **0.0%** |   L1   |
| MBR-CER (L1)                 |    6.02% |         0.0% |   L1   |
| MLP rescorer (all features)  |    6.05% |        −1.5% |   L3   |
| Pairwise rescorer            |    6.07% |        −3.1% |   L3   |
| Ridge agreement only         |    6.10% |        −5.2% |   L3   |
| Ridge all features           |    6.14% |        −7.2% |   L3   |
| Ridge length only            |   12.21% |      −391.7% |   L3   |

(L1 numbers are from `level1_report.md`; all rescorer rows are L3.)

## Runtime

- N-best generation, train-clean-100 (10k utt): ~30 min on T4
- Feature extraction (train + dev): ~2 min CPU
- Training + ablation + evaluation: ~3 min CPU

## Output Files

- `features_train.csv` — training features (kept on Colab Drive, 22 MB)
- `features_dev.csv` — dev features (committed)
- `rescorer_results.json` — all model performances
- `plots/feature_importance.png`
- `plots/rescorer_comparison.png`
- `plots/rescorer_per_utterance.png`
- `level3_report.md` — this report

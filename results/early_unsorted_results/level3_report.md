# Level 3: Discriminative Feature Rescorer

**Dataset:** LibriSpeech dev-other (2864 utterances)
**Training data:** train-clean-100
**Model:** Zipformer-S CR-CTC, BPE-500
**N-best:** G=16, nbest_scale=1.0

## Approach

A learned value function (discriminative rescorer) that predicts
hypothesis quality from CTC-derived features. For each utterance,
the rescorer predicts WER for all G candidates and selects the one
with the lowest predicted WER. This is equivalent to the Q-function
of the one-step selection MDP.

14 features are extracted per candidate: CTC log-probability variants (4),
length statistics (4), inter-candidate agreement scores (3), and
probability-distribution features (3).

## Results

| Method                       |     WER% |  Gap Closed% |
|------------------------------|----------|--------------|
| Oracle                       |    4.44% |       100.0% |
| Greedy                       |    6.02% |         0.0% |
| Mlp All                      |    6.05% |        -1.5% |
| Pairwise All                 |    6.07% |        -3.1% |
| Ridge Ctc Plus Agreement     |    6.10% |        -4.7% |
| Ridge Agreement Only         |    6.10% |        -5.2% |
| Ridge All Features           |    6.14% |        -7.2% |
| Ridge Ctc Only               |    6.22% |       -12.5% |
| Ridge Prob Only              |    6.24% |       -13.5% |
| Ridge Length Only            |   12.21% |      -391.7% |

**Greedy WER:** 6.02%
**Oracle WER:** 4.44%
**Oracle gap:** 1.58 pp (26.2% relative)

**Best rescorer:** None improved over greedy

## Feature Importance (Ridge)

| Rank | Feature                      |  Coefficient |
|------|------------------------------|--------------|
|    1 | mean_wer_to_others           |     +0.0734 |
|    2 | log_prob_gap                 |     +0.0297 |
|    3 | ctc_log_prob                 |     -0.0265 |
|    4 | len_words                    |     +0.0174 |
|    5 | len_chars                    |     -0.0124 |
|    6 | ptilde                       |     -0.0110 |
|    7 | ctc_rank                     |     +0.0100 |
|    8 | ctc_log_prob_per_char        |     +0.0066 |
|    9 | entropy_of_group             |     +0.0062 |
|   10 | len_tokens                   |     +0.0060 |
|   11 | mean_cer_to_others           |     -0.0043 |
|   12 | ctc_log_prob_per_token       |     -0.0030 |
|   13 | agrees_with_majority         |     -0.0023 |
|   14 | len_deviation                |     -0.0019 |

Positive coefficients increase predicted WER (bad features).
Negative coefficients decrease predicted WER (good features).

## Ablation Study

Feature subset ablations show which signal sources contribute:

- **ctc_only:** WER 6.22% (gap closed: -12.5%)
- **length_only:** WER 12.21% (gap closed: -391.7%)
- **agreement_only:** WER 6.10% (gap closed: -5.2%)
- **prob_only:** WER 6.24% (gap closed: -13.5%)
- **ctc_plus_agreement:** WER 6.10% (gap closed: -4.7%)
- **all_features:** WER 6.14% (gap closed: -7.2%)

## Cross-Validation (train-clean-100)

- **R² mean:** 0.7601 ± 0.0271
- **R² per fold:** 0.7511, 0.7766, 0.7849, 0.7109, 0.7769
- **WER mean (CV):** 1.48% ± 0.05%

R² = 0.760 indicates features have strong predictive power for WER.

## Per-Utterance Analysis

- **Improved:** 54 utterances
- **Degraded:** 53 utterances
- **Same:** 2757 utterances
- **Net:** +1 utterances
- **Recoverable utterances:** 665
- **Recovered:** 54 (8.1%)

## Master Comparison (All Levels)

Including results from Level 1 and Level 1b:

| Method                       |     WER% |  Gap Closed% |     Source |
|------------------------------|----------|--------------|------------|
| Oracle                       |    4.44% |       100.0% |         L1 |
| Greedy                       |    6.02% |         0.0% |         L1 |
| Mlp All                      |    6.05% |        -1.5% |         L3 |
| Pairwise All                 |    6.07% |        -3.1% |         L3 |
| Ridge Ctc Plus Agreement     |    6.10% |        -4.7% |         L3 |
| Ridge Agreement Only         |    6.10% |        -5.2% |         L3 |
| Ridge All Features           |    6.14% |        -7.2% |         L3 |
| Ridge Ctc Only               |    6.22% |       -12.5% |         L3 |
| Ridge Prob Only              |    6.24% |       -13.5% |         L3 |
| Ridge Length Only            |   12.21% |      -391.7% |         L3 |

## Runtime

- Feature extraction + training + evaluation: 172.5s

## Output Files

- `features_train.csv` — training features
- `features_dev.csv` — dev features
- `rescorer_results.json` — all model performances
- `plots/feature_importance.png`
- `plots/rescorer_comparison.png`
- `plots/rescorer_per_utterance.png`
- `level3_report.md` — this report

# R-PROP2: Proposition 4.2 Verification Expanded (n=250)

## Summary

Expanded Prop 4.2 gradient variance verification from n=50 to n=250 valid utterances (G=8). Mean Viterbi/CTC ratio = 2.9620 (95% CI [2.7402, 3.2031]). Mean Sampled/CTC ratio = 3.8692 (95% CI [3.5535, 4.2138]). Ordering violations (Var_CTC > Var_Viterbi or Var_Viterbi > Var_Sampled): **30**.

## Comparison: n=50 vs n=250

| Metric | n=50 (Stage 3b) | n=250 (R-PROP2) |
|--------|-----------------|------------------|
| Mean Viterbi/CTC | 2.7030 | 2.9620 |
| Median Viterbi/CTC | 2.4715 | 2.3755 |
| SD Viterbi/CTC | — | 1.8826 |
| 95% CI Viterbi/CTC | — | [2.7402, 3.2031] |
| Min/Max Viterbi/CTC | 1.1070/8.1857 | 1.0963/15.4335 |
| Mean Sampled/CTC | 3.6658 | 3.8692 |
| Median Sampled/CTC | 3.3820 | 3.2455 |
| SD Sampled/CTC | — | 2.6779 |
| 95% CI Sampled/CTC | — | [3.5535, 4.2138] |
| Min/Max Sampled/CTC | 1.1070/11.4148 | 1.0678/24.2995 |
| Ordering violations | 0 | 30 |
| Skipped | — | 0 |

## Percentile Distribution (Viterbi/CTC ratio)

| Percentile | Value |
|------------|-------|
| 5th | 1.3095 |
| 10th | 1.4545 |
| 25th | 1.7881 |
| 50th | 2.3755 |
| 75th | 3.4335 |
| 90th | 4.7970 |
| 95th | 6.8644 |

## Percentile Distribution (Sampled/CTC ratio)

| Percentile | Value |
|------------|-------|
| 5th | 1.5212 |
| 10th | 1.7545 |
| 25th | 2.2579 |
| 50th | 3.2455 |
| 75th | 4.4695 |
| 90th | 6.3108 |
| 95th | 8.4167 |

## Histogram (Viterbi/CTC ratio, 10 bins)

Range: [1.10, 15.43], IQR: [1.79, 3.43]

(Histogram bin counts saved in prop42_results.json)

## Entropy Correlation Analysis

Skipped: fewer than 10 overlapping utterances with Stage 2 gamma analysis, or gamma_stats.csv not found.

## Verification Checklist

- [ ] Zero ordering violations (Var_CTC <= Var_Viterbi <= Var_Sampled)
- [x] Mean Viterbi/CTC ratio consistent with Stage 3b (2.70)
- [x] Mean Sampled/CTC ratio consistent with Stage 3b (3.67)
- [ ] 95% CI width (0.4629) vs estimated Stage 3b CI width (check ~sqrt(5) narrower)

## Method

- Model: Zipformer-S CR-CTC (22M params, BPE-500)
- Data: first 250 utterances from dev-other (by cut order)
- G = 8 candidates per utterance
- Three gradient estimators on CTC output projection:
  1. CTC-marginalized (gamma-weighted via k2 backward)
  2. Viterbi (best single alignment, one-hot credit)
  3. Sampled (one random alignment ~ posterior, one-hot credit)
- Variance = across-candidate variance of advantage-weighted gradients
- Bootstrap CI: B=10000, seed=42

## Paper Update

Section 3.3 Proposition 2 verification paragraph should be updated: n=50 -> n=250, mean Viterbi/CTC ratio = 2.96 (95% CI [2.74, 3.20]), mean Sampled/CTC ratio = 3.87 (95% CI [3.55, 4.21]), 30 ordering violations.

## Bring-Back Files

```
results/R_prop2_expansion/prop42_results.json
results/R_prop2_expansion/prop42_per_utt.csv
reports/R_prop2_expansion_report.md
```

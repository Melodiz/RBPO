# Cross-Split Comparison: dev-clean vs dev-other vs test-other

All results use the same model (Zipformer-S CR-CTC 22.1M) and G=16.
Hyperparameters tuned on dev-other; dev-clean and test-other are held-out.

## Greedy Baseline

| Split | Utterances | Greedy WER (%) | Oracle WER (%) | Oracle Gap (pp) |
|-------|------------|---------------:|---------------:|----------------:|
| dev-clean | ~2703 | 2.37 | 1.54 | 0.83 |
| dev-other | 2864 | 6.02 | — | — |
| test-other | 2939 | 5.96 | — | — |

## Best Method: MBR-CER + RoBERTa PLL τ=10

| Split | Greedy (%) | MBR+PLL (%) | Δ (pp) | p-value | Significant? |
|-------|----------:|-----------:|-------:|--------:|:------------:|
| dev-clean | 2.37 | 2.28 | -0.08 | 0.0080 | ✓ |
| dev-other | 6.02 | 5.79 | -0.23 | <0.0001 | ✓ |
| test-other | 5.96 | 5.77 | -0.19 | 0.0003 | ✓ |

## All Methods (dev-clean)

| Method | WER (%) | Δ vs Greedy (pp) | p-value | α=0.05 |
|--------|--------:|-----------------:|--------:|:------:|
| Greedy | 2.3676 | — | — | — |
| Oracle | 1.5404 | -0.827 | — | — |
| MBR-CER + RoBERTa PLL τ=10 | 2.2830 | -0.085 | 0.0080 | ✓ |
| MBR-CER τ=50 | 2.3400 | -0.028 | 0.1119 | — |
| MBR-CER τ=∞ | 2.3510 | -0.017 | 0.2734 | — |
| GPT-2 interp α=0.8 | 2.3547 | -0.013 | 0.0496 | ✓ |
| RoBERTa PLL interp α=0.7 | 2.3657 | -0.002 | 0.4868 | — |

## Interpretation

- dev-clean has lower absolute WER (~2.4%) vs dev/test-other (~6%)
- The improvement margin (in pp) is expected to be smaller on cleaner data
- Consistent direction of improvement across splits confirms generalization

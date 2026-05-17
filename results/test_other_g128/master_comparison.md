# Master Comparison: G=16 vs G=128, dev-other vs test-other

All methods evaluated against greedy CTC baseline (B=10,000 bootstrap).

## Headline WERs

| Split | G | Greedy | Oracle | Gap (pp) | Rel gap |
|-------|--:|-------:|-------:|---------:|--------:|
| dev-other  | 16  | 6.02% | 4.44% | 1.58 | 26.2% |
| test-other | 16  | 5.96% | 4.41% | 1.55 | 26.0% |
| dev-other  | 128 | 6.02% | 3.53% | 2.49 | 41.3% |
| test-other | 128 | 5.96% | 3.37% | 2.58 | 43.4% |

## Per-Method Results (sorted by best WER)

| Method | G16 dev | G16 test | G128 dev | G128 test | Best config |
|--------|--------:|---------:|---------:|----------:|:-----------:|
| MBR-CER + PLL τ=10 | 5.79% | 5.77% | 5.53% | 5.42% | G128 test |
| RoBERTa PLL interp α=0.7 | 5.92% | 5.85% | 5.96% | 5.92% | G16 test |
| RoBERTa PLL interp α=0.8 | — | — | 5.89% | 5.86% | G128 test |
| GPT-2 interp α=0.8 | 5.99% | 5.91% | 6.00% | 5.92% | G16 test |
| GPT-2 interp α=0.7 | — | — | 6.00% | 6.04% | G128 dev |
| MBR-CER τ=50 (CTC) | 5.99% | 5.92% | 5.94% | 5.84% | G128 test |
| MBR-CER τ=∞ (CTC) | 5.99% | 5.92% | 5.93% | 5.84% | G128 test |
| MBR-CER + PLL τ=5 | — | — | 6.00% | 5.97% | G128 test |
| MBR-CER + PLL τ=50 | — | — | 5.82% | 5.73% | G128 test |
| MBR-CER + PLL τ=∞ | — | — | 5.93% | 5.84% | G128 test |

## Significance Summary (α=0.05)

| Method | G16 dev | G16 test | G128 dev | G128 test |
|--------|:-------:|:--------:|:--------:|:---------:|
| MBR-CER + PLL τ=10 | ✓ | ✓ | ✓ | ✓ |
| RoBERTa PLL interp α=0.7 | ✓ | ✓ | — | — |
| RoBERTa PLL interp α=0.8 | — | — | ✓ | ✓ |
| GPT-2 interp α=0.8 | ✓ | ✓ | — | ✓ |
| GPT-2 interp α=0.7 | — | — | — | — |
| MBR-CER τ=50 (CTC) | — | — | ✓ | ✓ |
| MBR-CER τ=∞ (CTC) | — | — | ✓ | ✓ |
| MBR-CER + PLL τ=5 | — | — | — | — |
| MBR-CER + PLL τ=50 | — | — | ✓ | ✓ |
| MBR-CER + PLL τ=∞ | — | — | ✓ | ✓ |

## Spearman ρ Comparison (G=16 vs G=128)

| Scorer | G16 dev | G16 test | G128 dev | G128 test |
|--------|--------:|---------:|---------:|----------:|
| CTC log-prob | -0.3474 | -0.3385 | -0.2697 | -0.2727 |
| RoBERTa PLL | -0.4844 | -0.4747 | -0.4609 | -0.4507 |
| GPT-2 LL | -0.4005 | -0.3934 | -0.3614 | -0.3529 |
| Interpolated (α=0.6 CTC + 0.4 PLL) | -0.5270 | -0.5165 | -0.4923 | -0.4855 |

## Key Finding

> **The headline result generalizes.** MBR-CER + RoBERTa PLL τ=10 achieves 5.42% WER on test-other G=128 (p=<0.0001), confirming the dev-other finding (5.53%) is not overfit.

# Test-Other Evaluation — Summary

Split: **test-other** (2939 utterances, 52343 ref words). Bootstrap B=10000, seed=42.

## Corpus WERs

| Method | WER (%) |
|--------|--------:|
| Greedy | 5.9569 |
| Oracle | 4.4113 |
| MBR-CER τ=50 | 5.9206 |
| MBR-CER τ=∞ | 5.9225 |
| RoBERTa PLL interp α=0.7 | 5.8499 |
| GPT-2 interp α=0.8 | 5.9129 |
| MBR-CER + RoBERTa PLL τ=10 | 5.7696 |

## Paired Bootstrap vs Greedy

| Method | WER (%) | Δ (pp) | p-value | 95% CI (pp) | α=0.05 | α=0.01 |
|--------|--------:|-------:|--------:|------------:|:------:|:------:|
| MBR-CER + RoBERTa PLL τ=10 | 5.7696 | -0.187 | 0.0003 | [-0.288, -0.086] | ✓ | ✓ |
| RoBERTa PLL interp α=0.7 | 5.8499 | -0.107 | 0.0007 | [-0.176, -0.040] | ✓ | ✓ |
| GPT-2 interp α=0.8 | 5.9129 | -0.044 | 0.0015 | [-0.075, -0.015] | ✓ | ✓ |
| MBR-CER τ=50 | 5.9206 | -0.036 | 0.1535 | [-0.103, +0.030] | — | — |
| MBR-CER τ=∞ | 5.9225 | -0.034 | 0.1706 | [-0.102, +0.033] | — | — |

## Spearman ρ (corpus)

| Scorer | ρ | 95% CI | N |
|--------|---:|--------|---:|
| CTC log-prob | -0.3385 | [-0.3467, -0.3302] | 2930 |
| RoBERTa PLL | -0.4747 | [-0.4837, -0.4656] | 2930 |
| Interpolated (α=0.6 CTC + 0.4 PLL) | -0.5165 | [-0.5249, -0.5082] | 2930 |
| GPT-2 LL | -0.3934 | [-0.4031, -0.3838] | 2930 |

## Verification

- [✓] test-other utt count ≈ 2939: measured=2939, expected=2939
- [✓] Greedy WER ≈ 6.03%: measured=0.05956861471447949, expected=0.0603
- [✓] Oracle < Greedy: measured=0.044112870870985615, expected=0.05956861471447949
- [✓] Record count consistent: measured=2939, expected=2939
- [✓] No empty candidates: measured=0, expected=0
- [✓] PLL sign convention: measured=1.0, expected=0.95
- [✓] Total ref words: measured=52343, expected=None

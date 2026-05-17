# Dev-Clean Evaluation — Summary

Split: **dev-clean** (2703 utterances, 54402 ref words). Bootstrap B=10000, seed=42.

## Corpus WERs

| Method | WER (%) |
|--------|--------:|
| Greedy | 2.3676 |
| Oracle | 1.5404 |
| MBR-CER τ=50 | 2.3400 |
| MBR-CER τ=∞ | 2.3510 |
| RoBERTa PLL interp α=0.7 | 2.3657 |
| GPT-2 interp α=0.8 | 2.3547 |
| MBR-CER + RoBERTa PLL τ=10 | 2.2830 |

## Paired Bootstrap vs Greedy

| Method | WER (%) | Δ (pp) | p-value | 95% CI (pp) | α=0.05 | α=0.01 |
|--------|--------:|-------:|--------:|------------:|:------:|:------:|
| MBR-CER + RoBERTa PLL τ=10 | 2.2830 | -0.085 | 0.0080 | [-0.156, -0.016] | ✓ | ✓ |
| MBR-CER τ=50 | 2.3400 | -0.028 | 0.1119 | [-0.071, +0.015] | — | — |
| MBR-CER τ=∞ | 2.3510 | -0.017 | 0.2734 | [-0.068, +0.033] | — | — |
| GPT-2 interp α=0.8 | 2.3547 | -0.013 | 0.0496 | [-0.029, +0.002] | ✓ | — |
| RoBERTa PLL interp α=0.7 | 2.3657 | -0.002 | 0.4868 | [-0.045, +0.041] | — | — |

## Spearman ρ (corpus)

| Scorer | ρ | 95% CI | N |
|--------|---:|--------|---:|
| CTC log-prob | -0.3651 | [-0.3735, -0.3568] | 2697 |
| RoBERTa PLL | -0.4959 | [-0.5045, -0.4872] | 2697 |
| Interpolated (α=0.6 CTC + 0.4 PLL) | -0.5371 | [-0.5449, -0.5293] | 2697 |
| GPT-2 LL | -0.4251 | [-0.4343, -0.4158] | 2697 |

## Verification

- [✓] dev-clean utt count ≈ 2703: measured=2703, expected=2703
- [✓] Greedy WER ≈ 2.37%: measured=0.023675600161758757, expected=0.0237
- [✓] Oracle < Greedy: measured=0.015403845446858572, expected=0.023675600161758757
- [✓] Record count consistent: measured=2703, expected=2703
- [✓] No empty candidates: measured=0, expected=0
- [✓] PLL sign convention: measured=1.0, expected=0.95
- [✓] Total ref words: measured=54402, expected=None

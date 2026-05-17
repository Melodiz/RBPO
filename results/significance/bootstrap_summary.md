# Paired Bootstrap Significance Tests — WER

Baseline: Greedy (1-best CTC). One-sided test: is method better than greedy?
Bootstrap samples: 10,000. Dataset: LibriSpeech dev-other (2864 utts, ~50,948 ref words).

## Results

| Method | WER (%) | Δ (pp) | p-value | 95% CI | Sig α=0.05 | Sig α=0.01 |
|--------|--------:|-------:|--------:|--------|:----------:|:----------:|
| MBR-CER + RoBERTa PLL τ=10 | 5.79 | -0.232 | 0.0000 | [-0.327, -0.138] | ✓ | ✓ |
| RoBERTa PLL interp α=0.7 | 5.92 | -0.104 | 0.0019 | [-0.170, -0.039] | ✓ | ✓ |
| MBR-CER τ=50 | 5.99 | -0.035 | 0.1630 | [-0.102, +0.033] | — | — |
| MBR-CER τ=∞ (uniform) | 5.99 | -0.033 | 0.1812 | [-0.101, +0.035] | — | — |
| GPT-2 interp α=0.8 | 5.99 | -0.033 | 0.0238 | [-0.067, -0.002] | ✓ | — |
| Argmax P_CTC | 6.02 | +0.000 | 1.0000 | [+0.000, +0.000] | — | — |
| Length-norm (tokens) | 6.02 | +0.000 | 0.6517 | [-0.006, +0.006] | — | — |
| Length-norm (chars) | 6.02 | +0.000 | 1.0000 | [+0.000, +0.000] | — | — |
| MBR-WER | 6.02 | +0.000 | 1.0000 | [+0.000, +0.000] | — | — |
| MBR-CER τ=1 | 6.03 | +0.006 | 1.0000 | [+0.000, +0.016] | — | — |
| Self-consistency | 6.04 | +0.018 | 0.7213 | [-0.047, +0.083] | — | — |

## Interpretation

**3 method(s) achieve statistical significance at α=0.05:**

- MBR-CER + RoBERTa PLL τ=10: WER 5.79%, Δ=-0.232pp, p=0.0000
- RoBERTa PLL interp α=0.7: WER 5.92%, Δ=-0.104pp, p=0.0019
- GPT-2 interp α=0.8: WER 5.99%, Δ=-0.033pp, p=0.0238

## Verification Checks

- [✓] Greedy WER == 6.02%: measured=0.06021826175708565, expected=0.0602
- [✓] Oracle WER == 4.44%: measured=0.04441783779539923, expected=0.0444
- [✓] Greedy vs Greedy delta=0: measured=0.0, expected=0.0
- [✓] N utterances == 2864: measured=2864, expected=2864
- [✓] Total ref words ~50948: measured=50948, expected=50948

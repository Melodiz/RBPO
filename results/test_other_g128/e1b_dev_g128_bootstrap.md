# E1b: dev-other G=128 Paired Bootstrap Significance Tests

**Status:** Complete. B=10,000, seed=42.

## Key Finding

With G=128 candidates (vs G=16), **two new patterns emerge:**

1. **MBR-CER + PLL τ=10 is overwhelmingly significant** — CI [-0.586, -0.403] pp, the entire CI is far from zero. This is the strongest result in the project.
2. **CTC-internal MBR now reaches significance** (τ=50: p=0.004, τ=∞: p=0.003). With 128 candidates, CTC weights have enough diversity for MBR to help — the information bottleneck is partially relieved by broader lattice exploration.

### dev-other G=128 vs Greedy

| Method | WER (%) | Δ (pp) | p-value | 95% CI (pp) | α=0.05 | α=0.01 | N differ |
|--------|--------:|-------:|--------:|------------:|:------:|:------:|---------:|
| MBR-CER + PLL τ=10 | 5.53 | -0.493 | <0.0001 | [-0.586, -0.403] | ✓ | ✓ | 662 |
| MBR-CER + PLL τ=50 | 5.82 | -0.202 | <0.0001 | [-0.271, -0.135] | ✓ | ✓ | 446 |
| RoBERTa PLL interp α=0.8 | 5.89 | -0.134 | <0.0001 | [-0.190, -0.079] | ✓ | ✓ | 169 |
| MBR-CER τ=∞ (CTC) | 5.93 | -0.088 | 0.0034 | [-0.150, -0.024] | ✓ | ✓ | 430 |
| MBR-CER + PLL τ=∞ | 5.93 | -0.088 | 0.0034 | [-0.150, -0.024] | ✓ | ✓ | 430 |
| MBR-CER τ=50 (CTC) | 5.94 | -0.084 | 0.0040 | [-0.145, -0.023] | ✓ | ✓ | 422 |
| RoBERTa PLL interp α=0.7 | 5.96 | -0.057 | 0.0960 | [-0.142, +0.026] | — | — | 393 |
| GPT-2 interp α=0.8 | 6.00 | -0.024 | 0.0501 | [-0.052, +0.002] | — | — | 74 |
| GPT-2 interp α=0.7 | 6.00 | -0.022 | 0.2437 | [-0.082, +0.037] | — | — | 243 |
| MBR-CER + PLL τ=5 | 6.00 | -0.022 | 0.3828 | [-0.159, +0.119] | — | — | 1105 |

## Spearman ρ (corpus)

| Scorer | ρ | 95% CI |
|--------|---:|--------|
| CTC log-prob | -0.2697 | [-0.2745, -0.2647] |
| RoBERTa PLL | -0.4609 | [-0.4673, -0.4543] |
| Interpolated (α=0.6 CTC + 0.4 PLL) | -0.4923 | [-0.4982, -0.4863] |
| GPT-2 LL | -0.3614 | [-0.3682, -0.3544] |

## Comparison with G=16 dev-other (E1)

| Method | G=16 WER | G=128 WER | G=16 p | G=128 p | G=16 sig? | G=128 sig? |
|--------|--------:|--------:|--------:|--------:|:---------:|:----------:|
| MBR-CER + PLL τ=10 | 5.79% | 5.53% | <0.0001 | <0.0001 | ✓ | ✓ |
| RoBERTa PLL interp α=0.7 | 5.92% | 5.96% | 0.0019 | 0.0960 | ✓ | — |
| RoBERTa PLL interp α=0.8 | — | 5.89% | — | <0.0001 | — | ✓ |
| GPT-2 interp α=0.8 | 5.99% | 6.00% | 0.0238 | 0.0501 | ✓ | — |
| MBR-CER τ=50 (CTC) | 5.99% | 5.94% | 0.1630 | 0.0040 | — | ✓ |
| MBR-CER τ=∞ (CTC) | 5.99% | 5.93% | 0.1812 | 0.0034 | — | ✓ |

## Interpretation

**G=128 changes the significance landscape:**
- The **headline result** (MBR-CER+PLL τ=10) doubles its effect size: -0.232pp (G=16) → -0.493pp (G=128). More candidates + PLL weighting = more room to pick a better hypothesis.
- **CTC-internal MBR** crosses the significance threshold. At G=16 the lattice is too narrow for MBR to help. At G=128 there is enough diversity that even peaked CTC weights can steer toward lower-error candidates.
- **Linear interpolation α=0.7** *loses* significance. Why? At G=128, the best α shifts from 0.7 to 0.8 (as confirmed by the sweep in report_neural_lm.md). The hyperparameter is slightly stale.
- **GPT-2 interp α=0.8** is marginal (p=0.05) — its weaker signal struggles against the noise of 128 candidates.

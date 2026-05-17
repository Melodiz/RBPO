# E1b: dev-other G=128 Paired Bootstrap Significance Tests — Stage Report

**Status:** Complete. B=10,000, seed=42. Run locally + confirmed on Colab.

## TL;DR

- **MBR-CER + PLL τ=10 at G=128 is the project's strongest result:** 5.53% WER, p<0.0001, CI=[-0.586, -0.403]pp. The entire CI is ~0.4pp below zero.
- **CTC-internal MBR crosses significance at G=128** (τ=50: p=0.004, τ=∞: p=0.003). The information bottleneck is partially relieved by broader lattice exploration — with 128 candidates, even peaked CTC weights have enough diversity for MBR to consistently find a better hypothesis.
- **6 of 10 methods significant at α=0.05** (vs 3/11 at G=16). More candidates → more methods work.

## Bootstrap Results

| Method | WER (%) | Δ (pp) | p-value | 95% CI (pp) | α=0.05 | α=0.01 |
|--------|--------:|-------:|--------:|------------:|:------:|:------:|
| MBR-CER + PLL τ=10 | 5.53 | -0.493 | <0.0001 | [-0.586, -0.403] | ✓ | ✓ |
| MBR-CER + PLL τ=50 | 5.82 | -0.202 | <0.0001 | [-0.271, -0.135] | ✓ | ✓ |
| RoBERTa PLL interp α=0.8 | 5.89 | -0.134 | <0.0001 | [-0.190, -0.079] | ✓ | ✓ |
| MBR-CER τ=∞ (CTC) | 5.93 | -0.088 | 0.0034 | [-0.150, -0.024] | ✓ | ✓ |
| MBR-CER + PLL τ=∞ | 5.93 | -0.088 | 0.0034 | [-0.150, -0.024] | ✓ | ✓ |
| MBR-CER τ=50 (CTC) | 5.94 | -0.084 | 0.0040 | [-0.145, -0.023] | ✓ | ✓ |
| RoBERTa PLL interp α=0.7 | 5.96 | -0.057 | 0.0960 | [-0.142, +0.026] | — | — |
| GPT-2 interp α=0.8 | 6.00 | -0.024 | 0.0501 | [-0.052, +0.002] | — | — |
| GPT-2 interp α=0.7 | 6.00 | -0.022 | 0.2437 | [-0.082, +0.037] | — | — |
| MBR-CER + PLL τ=5 | 6.00 | -0.022 | 0.3828 | [-0.159, +0.119] | — | — |

## Comparison with G=16 (E1)

| Method | G=16 WER | G=128 WER | G=16 p | G=128 p | Interpretation |
|--------|--------:|---------:|--------:|--------:|----------------|
| MBR-CER + PLL τ=10 | 5.79% | 5.53% | <0.0001 | <0.0001 | Effect doubles: -0.232→-0.493pp |
| RoBERTa PLL interp α=0.7 | 5.92% | 5.96% | 0.0019 | 0.0960 | α stale; optimal shifts to 0.8 |
| RoBERTa PLL interp α=0.8 | — | 5.89% | — | <0.0001 | New best interp at G=128 |
| GPT-2 interp α=0.8 | 5.99% | 6.00% | 0.0238 | 0.0501 | Marginal at both G values |
| MBR-CER τ=50 (CTC) | 5.99% | 5.94% | 0.1630 | 0.0040 | **Newly significant** |
| MBR-CER τ=∞ (CTC) | 5.99% | 5.93% | 0.1812 | 0.0034 | **Newly significant** |

## Spearman ρ (corpus-level)

| Scorer | G=16 ρ | G=128 ρ | Δ |
|--------|-------:|--------:|---:|
| CTC log-prob | -0.3474 | -0.2697 | +0.078 |
| RoBERTa PLL | -0.4844 | -0.4609 | +0.024 |
| GPT-2 LL | -0.4005 | -0.3614 | +0.039 |
| Interpolated (0.6·CTC + 0.4·PLL) | -0.5270 | -0.4923 | +0.035 |

**CTC ρ drops most** (22% weaker) at G=128 — more candidates dilute CTC's ability to rank. PLL degrades gracefully (5% weaker). This explains why MBR-CER with PLL weights succeeds while linear interpolation at the wrong α fails: MBR can exploit PLL's still-strong ranking over a large candidate set.

## Key Insights

1. **G=128 changes the significance landscape.** Methods that failed at G=16 (CTC MBR) now pass because more candidates provide the diversity needed for consensus decoding to outperform greedy.

2. **τ=10 is optimal across G values.** The same temperature works at G=16 (-0.232pp) and G=128 (-0.493pp) — it was not overfit to the G=16 lattice geometry.

3. **τ=5 is too sharp, τ=50 too flat.** At τ=5 (1105 utterances differ but p=0.38), the PLL distribution is so peaked that MBR collapses to argmax-PLL, losing the consensus benefit. At τ=50, the weighting is nearly uniform — you get the same result as CTC-uniform MBR.

4. **The information bottleneck is *partially* — not fully — relieved by G=128.** CTC-internal MBR now helps, but its effect (-0.088pp) is still 5.6× smaller than PLL-weighted MBR (-0.493pp). External linguistic information remains the primary driver.

## Verification

- Greedy WER: 6.0218% (matches all prior reports exactly)
- Oracle G=128: 3.5350% (matches beam-sweep report)
- 2864 utterances, 50,948 ref words
- Empty candidates: 2 (known bug, filtered by all methods)

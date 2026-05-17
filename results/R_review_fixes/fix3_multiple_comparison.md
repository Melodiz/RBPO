# Fix 3: Multiple Comparison Correction

## Context

The paper reports 13 paired bootstrap significance tests across conditions
(Table A.1). A reviewer requested family-wise error rate (FWER) and
false discovery rate (FDR) corrections.

We apply Holm-Bonferroni (controls FWER) and Benjamini-Hochberg
(controls FDR at α=0.05) to all 13 raw p-values simultaneously.

**Note:** Several p-values are reported as "<0.0001" from B=10,000 bootstrap.
We use p=0.0001 as a conservative upper bound; true corrected values would
be even smaller.

## Raw and Corrected p-values

| # | Condition | Raw p | Holm p | Sig? | BH-FDR p | Sig? |
|---|-----------|------:|-------:|:----:|---------:|:----:|
| 1 | dev-other G=16 MBR+PLL | <0.0001 | 0.0013 | ✓ | 0.0002 | ✓ |
| 2 | dev-other G=128 MBR+PLL | <0.0001 | 0.0013 | ✓ | 0.0002 | ✓ |
| 3 | test-other G=128 MBR+PLL | <0.0001 | 0.0013 | ✓ | 0.0002 | ✓ |
| 4 | dev-clean G=16 MBR+PLL | 0.0080 | 0.0192 | ✓ | 0.0087 | ✓ |
| 5 | dev-clean G=128 MBR+PLL | <0.0001 | 0.0013 | ✓ | 0.0002 | ✓ |
| 6 | Zipformer-M G=16 MBR+PLL | <0.0001 | 0.0013 | ✓ | 0.0002 | ✓ |
| 7 | Zipformer-M G=128 MBR+PLL | <0.0001 | 0.0013 | ✓ | 0.0002 | ✓ |
| 8 | TL3 G=16 MBR+PLL (full 1155) | <0.0001 | 0.0013 | ✓ | 0.0002 | ✓ |
| 9 | TL3 G=128 MBR+PLL | <0.0001 | 0.0013 | ✓ | 0.0002 | ✓ |
| 10 | MUSAN 20dB G=16 | 0.0064 | 0.0192 | ✓ | 0.0076 | ✓ |
| 11 | MUSAN 10dB G=16 | 0.0030 | 0.0120 | ✓ | 0.0039 | ✓ |
| 12 | MUSAN 5dB G=16 | 0.0009 | 0.0045 | ✓ | 0.0013 | ✓ |
| 13 | MUSAN 0dB G=16 | 0.6458 | 0.6458 | — | 0.6458 | — |

## Summary

- **Holm-Bonferroni (FWER):** 12 of 13 conditions remain significant at α=0.05.
- **Benjamini-Hochberg (FDR):** 12 of 13 conditions remain significant at α=0.05.
- **Only failure:** MUSAN 0dB G=16 (raw p=0.6458), which was already reported
  as non-significant. At SNR=0dB, speech is essentially masked by noise and
  MBR+PLL cannot improve over greedy.
- The correction does not change any conclusion in the paper.

## Paper-Ready Paragraph

> To control for multiple testing across the 13 conditions in Table A.1,
> we applied both Holm-Bonferroni (family-wise error rate) and
> Benjamini-Hochberg (false discovery rate) corrections at α = 0.05.
> All 12 conditions reported as significant retain significance under
> both corrections (worst-case adjusted p = 0.019 for dev-clean G=16
> and MUSAN 20 dB). The sole non-significant condition (MUSAN 0 dB,
> p = 0.646) remains non-significant, as expected. Note that the
> conservative bound p = 0.0001 was used for all conditions originally
> reported as p < 0.0001 (B = 10,000); true adjusted values are smaller.

## Method

```python
from statsmodels.stats.multitest import multipletests
reject_holm, pvals_holm, _, _ = multipletests(pvals, method='holm')
reject_bh, pvals_bh, _, _ = multipletests(pvals, method='fdr_bh')
```

Source p-values: project master results table (paired bootstrap,
B=10,000, seed=42, one-sided test vs greedy).

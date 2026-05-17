# S2 — Test-Other G=16 Verification

## Source

`results/test_other/test_other_results.json` — E2c cross-split evaluation.
G=16, nbest_scale=1.0, oversample=64, B=10000, seed=42.

## Full result row

| Field | Value |
|-------|-------|
| Split | test-other |
| G | 16 |
| Method | MBR-CER + RoBERTa PLL τ=10 |
| WER (%) | **5.7696** |
| Greedy WER (%) | 5.9569 |
| Oracle WER (%) | 4.4113 |
| Δ (pp) | −0.1872 |
| p-value | 0.0003 |
| CI (95%) | [−0.2882, −0.0855] |
| Significant (α=0.05) | Yes |
| Significant (α=0.01) | Yes |
| N utterances differ | 652 |
| N utterances total | 2939 |
| N ref words | 52343 |

## Gap closed

- Oracle gap: 5.9569 − 4.4113 = 1.5456 pp
- MBR gap closed: 5.9569 − 5.7696 = 0.1872 pp
- Fraction: 0.1872 / 1.5456 = **12.1%**

## Cross-check with task specification

| Field | Expected | Found | Match |
|-------|----------|-------|-------|
| WER | 5.77% | 5.7696% | ✓ (within ±0.01%) |
| Greedy | 5.96% | 5.9569% | ✓ |
| Oracle | 4.41% | 4.4113% | ✓ |
| Δ | −0.187 pp | −0.1872 pp | ✓ |
| p-value | 0.0003 | 0.0003 | ✓ exact |
| Gap closed | 12.1% | 12.1% | ✓ |

## Data provenance

Source file: `results/test_other/test_other_results.json`
Bootstrap CSV: `results/test_other/test_other_results.csv`
Per-utterance N-best: `results/test_other/nbest_test_other_G16.jsonl`
PLL scores: `results/test_other/neural_lm_scores_test_other.jsonl`

All data is local in the repo — no Google Drive retrieval needed.

## G=128 comparison (same split)

From `results/test_other_g128/test_other_g128_results.json`:

| G | Greedy | Oracle | MBR-CER+PLL τ=10 | Δ (pp) | p | Gap closed |
|---|--------|--------|-------------------|--------|---|------------|
| 16 | 5.9569 | 4.4113 | 5.7696 | −0.187 | 0.0003 | 12.1% |
| 128 | 5.9569 | 3.3720 | 5.4219 | −0.535 | 0.0000 | 20.7% |

Increasing G from 16 to 128 on test-other: MBR gain grows from 0.19 to
0.54 pp, with gap-closed increasing from 12.1% to 20.7%.

## Paper-ready text

> On test-other (G=16), MBR-CER+PLL at τ=10 achieves 5.77% WER,
> a statistically significant −0.19 pp reduction from the 5.96% greedy
> baseline (p=0.0003, paired bootstrap B=10000, 95% CI [−0.29, −0.09]),
> closing 12.1% of the oracle gap.

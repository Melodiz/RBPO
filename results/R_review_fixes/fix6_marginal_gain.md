# Fix 6: Marginal Gain per G-Doubling

## Source

All data from `results/g_scaling/scaling_curve.csv` (E11 beam-sweep experiment).
Dataset: LibriSpeech dev-other, 2864 utterances, Zipformer-S CR-CTC.

## MBR-CER + PLL τ=10 (Primary Method)

| G transition | WER from (%) | WER to (%) | Δ WER (pp) | Note |
|:-------------|:------------:|:----------:|:----------:|:-----|
| greedy → 4 | 6.0218 | 6.1985 | +0.177 | MBR hurts — too few candidates for consensus |
| 4 → 8 | 6.1985 | 5.8864 | −0.312 | Largest single gain; crosses below greedy |
| 8 → 16 | 5.8864 | 5.7902 | −0.096 | Diminishing returns begin |
| 16 → 32 | 5.7902 | 5.7078 | −0.082 | Continued diminishing |
| 32 → 64 | 5.7078 | 5.6450 | −0.063 | Smallest marginal gain |
| 64 → 128 | 5.6450 | 5.5292 | −0.116 | Anomalous bump — discussed below |

**Total gain (G=8→128):** 5.8864 → 5.5292 = −0.357 pp
**Total gain (greedy→G=128):** 6.0218 → 5.5292 = −0.493 pp

**Verification:** Sum of deltas from greedy→128: +0.177 − 0.312 − 0.096 − 0.082 − 0.063 − 0.116 = −0.493 pp ✓

## Oracle WER (Upper Bound on Achievable Improvement)

| G transition | Oracle from (%) | Oracle to (%) | Δ Oracle (pp) |
|:-------------|:---------------:|:-------------:|:--------------:|
| greedy → 4 | 6.0218 | 5.2485 | −0.773 |
| 4 → 8 | 5.2485 | 4.8265 | −0.422 |
| 8 → 16 | 4.8265 | 4.4418 | −0.385 |
| 16 → 32 | 4.4418 | 4.2612 | −0.181 |
| 32 → 64 | 4.2612 | 3.9001 | −0.361 |
| 64 → 128 | 3.9001 | 3.5350 | −0.365 |

Oracle does NOT show diminishing returns — each doubling adds 0.18–0.42 pp
of new headroom. The diminishing marginal gain in MBR is therefore a
limitation of the selection method, not of the candidate pool.

## Best Interpolation (RoBERTa PLL, best α per G)

| G transition | WER from (%) | WER to (%) | Δ WER (pp) |
|:-------------|:------------:|:----------:|:----------:|
| greedy → 4 | 6.0218 | 5.9315 | −0.090 |
| 4 → 8 | 5.9315 | 5.8884 | −0.043 |
| 8 → 16 | 5.8884 | 5.9178 | +0.029 |
| 16 → 32 | 5.9178 | 5.8903 | −0.028 |
| 32 → 64 | 5.8903 | 5.8884 | −0.002 |
| 64 → 128 | 5.8884 | 5.8884 | ±0.000 |

Linear interpolation gains are negligible beyond G=8.
Confirms that argmax-based selection cannot exploit larger candidate pools.

## The G=64→128 Anomaly

MBR-CER+PLL shows a marginal gain of −0.116 pp at G=64→128, larger than the
preceding −0.063 pp at G=32→64. This reversal of the diminishing-returns
trend likely reflects MBR's consensus mechanism: at G=128, the pairwise CER
matrix has 128² = 16,384 entries per utterance (vs 64² = 4,096), providing
substantially more evidence for the consensus vote. The threshold for
robust consensus may be in the G=64–128 range.

## Paper-Ready Paragraph

> Table X reports the marginal WER reduction per doubling of beam size G
> for MBR-CER+PLL (τ = 10). Beyond the initial G = 4 → 8 transition
> (−0.31 pp), returns diminish monotonically through G = 32 → 64
> (−0.06 pp), with a partial rebound at G = 64 → 128 (−0.12 pp).
> In contrast, oracle WER decreases steadily at each doubling
> (0.18–0.42 pp per step), indicating that the candidate pool continues
> to improve even where the selection method saturates. Linear
> interpolation shows near-zero marginal gains beyond G = 8. These
> patterns suggest that practical deployments should weigh the
> ~7.5× scoring cost of G = 128 vs G = 16 against the 0.26 pp WER
> reduction (5.79% → 5.53%).

# S1 — PLL-Argmax Baseline Decomposition

## Source

`results/diagnostics/mbr_sweep_results.csv`, rows with `sweep == argmax_baseline`.

### Raw CSV rows (verification)

```
argmax_baseline,none,argmax_ctc,0,6.0218
argmax_baseline,none,argmax_pll,0,9.7452
argmax_baseline,none,argmax_gpt2,0,11.1329
argmax_baseline,none,argmax_interp_0.8,0,5.8884
argmax_baseline,none,argmax_interp_0.7,0,5.9649
```

## Decomposition

| Method | WER (%) | Source |
|--------|---------|--------|
| Greedy (argmax CTC) | 6.0218 | `argmax_ctc` row |
| Argmax PLL | 9.7452 | `argmax_pll` row |
| Best non-MBR interpolation (α=0.8) | 5.8884 | `argmax_interp_0.8` row |
| Best MBR (MBR-CER+PLL τ=10) | 5.5292 | `tau` sweep, τ=10 row |

### Improvement breakdown

| Component | WER (%) | Δ (pp) | Share |
|-----------|---------|--------|-------|
| Greedy baseline | 6.0218 | — | — |
| Interpolation α=0.8 | 5.8884 | −0.1334 | 27.1% |
| MBR-CER+PLL τ=10 | 5.5292 | −0.3592 | 72.9% |
| **Total improvement** | — | **−0.4926** | **100.0%** |

Verification: 0.1334 + 0.3592 = 0.4926 ✓
Verification: 27.1% + 72.9% = 100.0% ✓

## Key finding: PLL alone is degenerate

Pure PLL argmax (9.75%) is **worse** than greedy (6.02%) by +3.72 pp.
This demonstrates that PLL without CTC anchor produces degenerate rankings:
the CTC score provides the coarse ordering, PLL provides refinement only.

Interpolation at α=0.8 (80% CTC + 20% PLL) gives a modest −0.13 pp
improvement over greedy, confirming the CTC dominance.

MBR consensus decoding contributes the remaining 72.9% of the total
improvement (−0.36 pp), showing that the averaging effect of MBR over
the hypothesis space is the primary driver of gains.

## Paper-ready text

> Decomposing the total WER reduction from greedy decoding (6.02%) to
> MBR-CER+PLL τ=10 (5.53%), we find that simple score interpolation
> (argmax at α=0.8) accounts for only 27.1% of the gain (−0.13 pp),
> while MBR consensus selection contributes the remaining 72.9%
> (−0.36 pp). Notably, pure PLL reranking without CTC anchor
> (argmax PLL = 9.75%) degrades WER by +3.72 pp relative to greedy,
> confirming that the CTC posterior provides essential coarse ordering
> that PLL alone cannot supply.

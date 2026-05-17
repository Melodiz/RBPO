# Cross-Model × Cross-G Scaling Table

Does MBR-CER+PLL τ=10 scale with G on both architectures?

| Model | G | Greedy | Oracle | MBR+PLL τ=10 | Best Interp (α) | Gap Closed (MBR) |
|-------|--:|-------:|-------:|-------------:|----------------:|-----------------:|
| Zipformer-S (22M) | 16 | 6.02% | 4.44% | 5.79% | 5.92% (α=0.7) | 14.7% |
| Zipformer-S (22M) | 128 | 6.02% | 3.54% | 5.53% | 5.89% (α=0.8) | 19.8% |
| Zipformer-M (65M) | 16 | 4.78% | 3.44% | 4.56% | 4.72% (α=0.7) | 16.5% |
| Zipformer-M (65M) | 128 | 4.78% | 2.73% | 4.43% | 4.71% (α=0.8) | 16.8% |

## Per-Model G Scaling

| Model | MBR+PLL: G=16 → G=128 | Best Interp: G=16 → G=128 |
|-------|----------------------:|--------------------------:|
| Zipformer-S | 5.79% → 5.53% (+0.26pp) | 5.92% → 5.89% (+0.03pp) |
| Zipformer-M | 4.56% → 4.43% (+0.12pp) | 4.72% → 4.71% (+0.01pp) |

## Verdict

**MBR scales on both architectures.** Zipformer-S gains 0.26pp from G=16→G=128; Zipformer-M gains 0.12pp. The G-scaling property is structural, not specific to the small model.

**Linear interpolation plateaus on both architectures.** Zipformer-S: +0.03pp; Zipformer-M: +0.01pp. argmax-based methods cannot exploit larger candidate sets — this is the central asymmetry the paper documents.

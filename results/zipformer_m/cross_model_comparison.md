# Cross-Model Comparison: Zipformer-S vs Zipformer-M CR-CTC

Both models evaluated on dev-other, G=16, same N-best/scoring pipeline.

| Metric | Zipformer-S (22M) | Zipformer-M CR-CTC |
|--------|:-----------------:|:-----------------:|
| Greedy WER | 6.02% | 4.78% |
| Oracle G=16 | 4.44% | 3.44% |
| Oracle gap (relative) | 26.2% | 27.9% |
| RoBERTa interp α=0.7 | 5.92% | 4.72% |
| MBR+PLL τ=10 WER | 5.79% | 4.56% |
| MBR+PLL τ=10 p-value | <0.0001 | 0.0000 |
| Gap closed (MBR+PLL) | 14.7% | 16.5% |
| CTC Spearman ρ | -0.347 | -0.350 |
| PLL Spearman ρ | -0.484 | -0.482 |

## Interpretation

**Generalization confirmed.** MBR-CER + RoBERTa PLL τ=10 produces a 0.22pp WER reduction (p=0.0000) on the larger model, replicating the qualitative finding from Zipformer-S. The information bottleneck is not a Zipformer-S quirk.

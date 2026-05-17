# Shallow Fusion vs Neural-LM Methods

Comparison of decode-time methods against greedy CTC.
All methods use the same Zipformer-S CR-CTC, same dev-other.

| Method | WER (%) | Δ (pp) | p-value | Info source |
|--------|--------:|-------:|--------:|-------------|
| Greedy CTC | 6.02 | — | — | Acoustic only |
| 3-gram shallow fusion (best α=0.90) | 6.02 | -0.004 | 0.3682 | + 3-gram LM |
| MBR-CER + 3-gram weights τ=10 | 6.00 | -0.020 | 0.3083 | + 3-gram + MBR |
| RoBERTa PLL interp α=0.7 G=16 | 5.92 | -0.10 | 0.002 | + Neural LM |
| MBR+PLL τ=10 G=16 | 5.79 | -0.23 | <0.0001 | + Neural LM + MBR |
| MBR+PLL τ=10 G=128 | 5.53 | -0.49 | <0.0001 | + Neural LM + MBR + G |

## Story

Shallow fusion with 3-gram LM closes 0.2% of the oracle gap. Neural LM rescoring closes more. MBR with neural LM posteriors closes the most and scales with G.

# Report E17: Shallow Fusion Baseline

**Status:** Complete. 2864 utterances, 3-gram LM via kenlm. 4s.

## What Ran

- Approach A: N-best rescoring with kenlm
- LM: `3-gram.pruned.1e-7.arpa` (order=3)
- N-best: existing G=16 dev-other (Zipformer-S CR-CTC)
- α sweep: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
- MBR-CER with LM weights at τ=10.0
- Bootstrap: B=10000, paired vs greedy

## Key Results

| Method | WER (%) | Δ (pp) | p-value |
|--------|--------:|-------:|--------:|
| Greedy CTC | 6.0218 | 0 | — |
| Argmax + 3-gram α=0.90 | 6.0179 | -0.0039 | 0.3682 |
| MBR-CER + 3-gram τ=10 | 6.0022 | -0.0196 | 0.3083 |
| Oracle (G=16) | 4.4418 | -1.5800 | — |

## Alpha Sweep

| α | WER (%) | Δ vs greedy (pp) |
|--:|--------:|-----------------:|
| 0.10 | 13.4451 | +7.4233 |
| 0.20 | 8.7285 | +2.7067 |
| 0.30 | 6.4517 | +0.4299 |
| 0.40 | 6.0866 | +0.0648 |
| 0.50 | 6.0532 | +0.0314 |
| 0.60 | 6.0336 | +0.0118 |
| 0.70 | 6.0238 | +0.0020 |
| 0.80 | 6.0258 | +0.0039 |
| 0.90 | 6.0179 | -0.0039 | **←best**

## Comparison with Neural LM Methods

| Method | WER (%) | Δ vs greedy (pp) |
|--------|--------:|-----------------:|
| Greedy CTC | 6.0218 | 0 |
| 3-gram fusion (best) | 6.0179 | -0.0039 |
| RoBERTa PLL interp α=0.7 G=16 | 5.9200 | -0.1000 |
| MBR+PLL τ=10 G=16 | 5.7900 | -0.2300 |
| MBR+PLL τ=10 G=128 | 5.5300 | -0.4900 |

## Conclusion

The 3-gram shallow fusion provides only a 0.0039pp reduction (p=0.3682, not significant). The n-gram LM contributes minimal value for this strong CTC model. Neural LM methods (RoBERTa, especially with MBR) provide the substantial gains — the linguistic signal that helps here is richer than n-gram statistics can capture.

## Files

| File | Purpose |
|------|---------|
| `shallow_fusion_results.json` | Full results |
| `shallow_fusion_sweep.csv` | α sweep tabular |
| `lm_mbr_comparison.csv` | LM with argmax vs MBR weights |
| `method_comparison.md` | Cross-method comparison table |
| `report_E17.md` | This stage report |
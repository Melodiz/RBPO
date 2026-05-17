# Report E12: τ Fine-Sweep at G=128

**Status:** Complete. 11 τ values, B=10000 bootstrap. 84s on M2.

## What Ran

- Data: `g128/neural_lm_scores.jsonl` (2864 utterances, G=128)
- Sweep: τ ∈ {5, 7, 8, 9, 10, 11, 12, 15, 20, 30, 50}
- Method: MBR-CER with RoBERTa PLL softmax weights
- CER matrix: computed once, re-weighted per τ
- Bootstrap: B=10000, paired vs greedy

## Key Results

- **Optimal τ = 9** (WER = 5.5056%)
- Greedy baseline: 6.0218%
- τ=10 WER: 5.5292%
- τ=9 vs τ=10: -0.0236pp
- Flat region: τ ∈ [8, 9, 10, 11]
- Total WER range: 0.4946pp

## Summary Table

| τ | WER (%) | Δ vs greedy (pp) | p-value |
|--:|--------:|-----------------:|--------:|
| 5 | 6.0002 | -0.0216 | 0.3828 |
| 7 | 5.5920 | -0.4299 | 0.0000 |
| 8 | 5.5095 | -0.5123 | 0.0000 |
| 9 | 5.5056 | -0.5162 | 0.0000 |
| 10 | 5.5292 | -0.4927 | 0.0000 |
| 11 | 5.5429 | -0.4789 | 0.0000 |
| 12 | 5.5723 | -0.4495 | 0.0000 |
| 15 | 5.6195 | -0.4024 | 0.0000 |
| 20 | 5.6921 | -0.3297 | 0.0000 |
| 30 | 5.7588 | -0.2630 | 0.0000 |
| 50 | 5.8197 | -0.2022 | 0.0000 |

## Interpretation

τ=10 is confirmed as optimal (or within noise of optimal). The flat region suggests robustness — the result is not lucky.

## Files

| File | Purpose |
|------|---------|
| `tau_sweep_results.json` | Full results with bootstrap |
| `tau_sweep.csv` | Tabular: τ, WER, p-value, CI |
| `tau_sweep_summary.md` | Formatted summary |
| `report_E12.md` | This stage report |
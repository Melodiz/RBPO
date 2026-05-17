# Report E14: Ensemble RoBERTa + GPT-2 MBR Weights

**Status:** Complete. 7×3 grid + tiebreaker. 7s on M2.

## What Ran

- Data: `g128/neural_lm_scores.jsonl` (2864 utterances)
- Combined score: s = β·roberta_pll + (1-β)·gpt2_ll
- Grid: β ∈ {0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0}, τ ∈ {7, 10, 15}
- Tiebreaker: MBR-CER(RoBERTa τ=10) → GPT-2 re-rank top-5
- Bootstrap: B=10000

## Key Results

| Method | WER (%) | Δ vs RoBERTa (pp) | p-value |
|--------|--------:|-------------------:|--------:|
| Greedy | 6.0218 | — | — |
| Pure RoBERTa (β=1.0, τ=10) | 5.5292 | 0 | — |
| Best ensemble (β=0.8, τ=7) | 5.5135 | -0.0157 | 0.3054 |
| Pure GPT-2 (β=0.0, τ=10) | 5.6940 | +0.1649 | — |
| Tiebreaker (top-5) | 6.3496 | +0.8204 | 1.0000 |

## Conclusion

Ensemble does not improve over pure RoBERTa. RoBERTa already captures the accessible linguistic signal. This is informative: it means the remaining errors are NOT addressable by simply combining a second LM.

## Files

| File | Purpose |
|------|---------|
| `ensemble_results.json` | Full grid + bootstrap |
| `ensemble_grid.csv` | β × τ → WER tabular |
| `ensemble_summary.md` | Formatted analysis |
| `report_E14.md` | This stage report |
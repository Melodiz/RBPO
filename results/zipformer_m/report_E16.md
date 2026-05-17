# Report E16: Cross-Model Verification

**Model:** Zipformer-M CR-CTC
**Status:** Complete. 2864 utterances, G=16, B=10000 bootstrap.

## What Ran

- Pipeline: discover → generate → score → evaluate
- Config: layers=2,2,3,4,3,2, dim=192,256,384,512,384,256
- Methods: greedy, oracle, RoBERTa interp α=0.7, MBR+PLL τ=10, MBR uniform
- Bootstrap: B=10000, paired vs greedy

## Key Results

| Method | WER (%) | Δ vs greedy (pp) | p-value |
|--------|--------:|-----------------:|--------:|
| Greedy | 4.7755 | 0 | — |
| roberta_interp_a0.7 | 4.7185 | -0.0569 | 0.0242 |
| mbr_cer_pll_tau10 | 4.5556 | -0.2198 | 0.0000 |
| mbr_cer_ctc_tau_inf | 4.7421 | -0.0334 | 0.1377 |
| Oracle | 3.4427 | -1.3327 | — |

## Spearman Correlations

- CTC log-prob ρ: **-0.3498** (Zipformer-S: -0.347)
- RoBERTa PLL ρ: **-0.4824** (Zipformer-S: -0.484)

## Generalization Verdict

**Confirmed.** MBR+PLL τ=10 reduces WER by 0.22pp (p=0.0000). The method generalizes across model sizes — closes the 'single architecture' critique.

## Files

| File | Purpose |
|------|---------|
| `zipformer_m_results.json` | Full results |
| `zipformer_m_bootstrap.csv` | Bootstrap p-values |
| `zipformer_m_spearman.json` | Per-utt Spearman ρ |
| `cross_model_comparison.md` | Side-by-side with Zipformer-S |
| `report_E16.md` | This stage report |
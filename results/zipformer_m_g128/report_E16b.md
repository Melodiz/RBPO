# Report E16b: Zipformer-M G=128 — Cross-Model Scaling

**Model:** Zipformer-M CR-CTC
**Status:** Complete. 2864 utterances, G=128, B=10000 bootstrap.

## What Ran

- Pipeline: discover → generate → score → evaluate
- N-best: G=128, oversample=512, avg 109.7 candidates per utterance
- Methods: greedy, oracle, RoBERTa interp α∈[0.7, 0.8], MBR+PLL τ∈[10, 50], MBR uniform
- Bootstrap: B=10000, paired vs greedy
- Optimizations: batched PLL (~30× speedup), checkpointing every 200 utts

## Key Results

| Method | WER (%) | Δ vs greedy (pp) | p-value |
|--------|--------:|-----------------:|--------:|
| Greedy | 4.7755 | 0 | — |
| roberta_interp_a0.7 | 4.8049 | +0.0294 | 0.8061 |
| roberta_interp_a0.8 | 4.7126 | -0.0628 | 0.0007 |
| mbr_cer_pll_tau10 | 4.4320 | -0.3435 | 0.0000 |
| mbr_cer_pll_tau50 | 4.6145 | -0.1609 | 0.0000 |
| mbr_cer_ctc_tau_inf | 4.6950 | -0.0805 | 0.0007 |
| Oracle | 2.7263 | -2.0491 | — |

## Spearman Correlations at G=128

- CTC log-prob ρ: **-0.2882** (Zipformer-S G=128: ~-0.30 typical)
- RoBERTa PLL ρ: **-0.4594** (Zipformer-S G=128: ~-0.46)

## Comparison vs Zipformer-M G=16 (E16)

| Metric | G=16 (E16) | G=128 (E16b) | Δ |
|--------|-----------:|-------------:|----:|
| Greedy | 4.7755% | 4.7755% | -0.0000pp |
| Oracle | 3.4427% | 2.7263% | -0.7164pp |
| MBR+PLL τ=10 | 4.5556% | 4.4320% | -0.1236pp |

## Verdict

**MBR scales with G on Zipformer-M.** G=16 gain: -0.22pp; G=128 gain: -0.34pp. The MBR-vs-interp asymmetry replicates on a second architecture.

## Files

| File | Purpose |
|------|---------|
| `zipformer_m_g128_results.json` | Full results |
| `zipformer_m_g128_bootstrap.csv` | Bootstrap p-values |
| `zipformer_m_g128_spearman.json` | Per-utt Spearman ρ |
| `cross_model_g_scaling.md` | **THE** 4-row comparison table |
| `report_E16b.md` | This stage report |
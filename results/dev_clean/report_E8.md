# E8: Dev-Clean Evaluation Pipeline — Stage Report

**Status:** Complete. Full pipeline (generate + score + evaluate). Run on Colab T4, 2026-05-04.

## TL;DR

- **MBR-CER + RoBERTa PLL tau=10 is significant on dev-clean:** 2.28% WER (p=0.008), a 0.085pp reduction from the 2.37% greedy baseline.
- The effect is **smaller in absolute terms** than dev-other (0.085pp vs 0.232pp) but proportionally similar (~3.6% relative reduction on both).
- Only 13.9% of utterances are recoverable (have a better candidate in the N-best list), vs ~45% on dev-other — dev-clean is already near-ceiling for this model.
- Spearman correlations are **stronger** on dev-clean than dev-other, confirming scoring quality is consistent.

## Key Numbers

| Metric | dev-clean | dev-other | Ratio |
|--------|-----------|-----------|-------|
| Utterances | 2703 | 2864 | — |
| Ref words | 54,402 | 50,948 | — |
| Greedy WER | 2.3676% | 6.0218% | 2.5x harder |
| Oracle WER (G=16) | 1.5404% | 4.4400% | — |
| Gap (pp) | 0.827 | 1.582 | 1.9x |
| Best method WER | 2.2830% | 5.7900% | — |
| Best Delta (pp) | -0.085 | -0.232 | 2.7x |
| Best p-value | 0.008 | <0.0001 | — |
| Gap closed | 10.3% | 14.7% | — |

## Bootstrap Results (B=10,000, seed=42)

| Method | WER (%) | Delta (pp) | p-value | 95% CI (pp) | Sig? |
|--------|--------:|-------:|--------:|-------------|:----:|
| MBR-CER + RoBERTa PLL tau=10 | 2.2830 | -0.085 | 0.0080 | [-0.156, -0.016] | ** |
| GPT-2 interp alpha=0.8 | 2.3547 | -0.013 | 0.0496 | [-0.029, +0.002] | * |
| MBR-CER tau=50 | 2.3400 | -0.028 | 0.1119 | [-0.071, +0.015] | — |
| MBR-CER tau=inf | 2.3510 | -0.017 | 0.2734 | [-0.068, +0.033] | — |
| RoBERTa PLL interp alpha=0.7 | 2.3657 | -0.002 | 0.4868 | [-0.045, +0.041] | — |

## Spearman Rank Correlations (with bootstrap CI)

| Scorer | rho | 95% CI | N |
|--------|----:|--------|---:|
| CTC log-prob | -0.3651 | [-0.3735, -0.3568] | 2697 |
| GPT-2 LL | -0.4251 | [-0.4343, -0.4158] | 2697 |
| RoBERTa PLL | -0.4959 | [-0.5045, -0.4872] | 2697 |
| Interpolated (0.6 CTC + 0.4 PLL) | -0.5371 | [-0.5449, -0.5293] | 2697 |

## Cross-Split Comparison

| Scorer | dev-other rho | dev-clean rho | Delta |
|--------|-------------:|-------------:|------:|
| CTC log-prob | -0.3474 | -0.3651 | -0.018 |
| RoBERTa PLL | -0.4844 | -0.4959 | -0.012 |
| GPT-2 LL | -0.4005 | -0.4251 | -0.025 |
| Interpolated | -0.5270 | -0.5371 | -0.010 |

All correlations are **stronger** (more negative) on dev-clean. This makes sense: cleaner audio produces more reliable CTC posteriors, making rank ordering by any scorer more consistent.

## Interpretation

1. **The method generalizes to easy data.** MBR-CER + PLL tau=10 is the only method that reaches strong significance on dev-clean where the baseline is already very low (2.37%).

2. **The ceiling effect is real.** Only 13.9% of utterances have room for improvement (vs ~45% on dev-other). The greedy decoder is already correct on 86% of utterances.

3. **GPT-2 alpha=0.8 is borderline** (p=0.0496, CI touches zero). This mirrors its marginal behavior on dev-other — GPT-2 is consistently the weaker LM for reranking.

4. **CTC-internal MBR is not significant on dev-clean** (p=0.11-0.27). On easy data where greedy is already strong, consensus decoding from CTC alone cannot find improvements. This reinforces the information bottleneck thesis — when acoustic decoding is already near-correct, only linguistic disambiguation from an external LM helps.

## Paper-Ready Statement

> On LibriSpeech dev-clean, MBR-CER weighted by RoBERTa PLL at tau=10 achieves 2.28% WER (G=16, p=0.008, 95% CI=[-0.156, -0.016]pp), reducing the 2.37% greedy baseline by 3.6% relative. The effect is proportionally consistent with dev-other (-3.9% relative), confirming the method generalizes across difficulty levels.

## Verification

| Check | Expected | Measured | Status |
|-------|----------|----------|:------:|
| Utterance count | ~2703 | 2703 | PASS |
| Greedy WER | ~2.37% | 2.3676% | PASS |
| Oracle < Greedy | yes | 1.54% < 2.37% | PASS |
| Empty candidates | 0 | 0 | PASS |
| RoBERTa PLL sign | >95% negative | 100% | PASS |
| Ref words | ~54k | 54,402 | PASS |

## Runtime

| Step | Time |
|------|------|
| N-best generation (G=16) | Skipped (cached) |
| RoBERTa PLL scoring (41,695 hyps) | 52 min 44s |
| GPT-2 LL scoring (41,695 hyps) | 2 min 11s |
| Evaluation + bootstrap | ~2s |

## Files Produced

| File | Purpose |
|------|---------|
| `dev_clean_results.json` | Machine-readable full results |
| `dev_clean_bootstrap.csv` | Bootstrap table |
| `dev_clean_spearman.json` | Spearman correlations with CIs |
| `dev_clean_summary.md` | Formatted summary |
| `cross_split_comparison.md` | dev-clean vs dev-other comparison |
| `report_E8.md` | This report |
| `nbest_dev_clean_G16.jsonl` | N-best lists (on Drive, ~50MB) |
| `neural_lm_scores_dev_clean.jsonl` | Scored N-best (on Drive, ~18MB) |

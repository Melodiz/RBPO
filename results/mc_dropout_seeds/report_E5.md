# E5: MC-Dropout Seed Variation — Stage Report

**Status:** Complete. 5 seeds x T=4 passes, B=10,000 bootstrap. Run on Colab T4, 2026-05-04.

## TL;DR

**The MC-Dropout improvement is NOT reproducible.** Across 5 random seeds, 0/5 reach significance at alpha=0.05. The 0.04pp improvement previously reported is within noise. MC-Dropout with T=4 passes does not reliably improve over deterministic greedy decoding on this model.

## Key Numbers

| Metric | Value |
|--------|-------|
| Greedy baseline WER | 6.0218% |
| Mean MC-MBR WER | 6.0301% +/- 0.0197% |
| Best seed (789) | 5.9983% (p=0.274) |
| Worst seed (123) | 6.0493% (p=0.771) |
| Seeds significant at alpha=0.05 | 0/5 |

## Per-Seed Results

| Seed | MC-Greedy (%) | MC-MBR (%) | Delta (pp) | p-value | 95% CI (pp) | Sig? |
|-----:|-------------:|-----------:|-------:|--------:|-------------|:----:|
| 42 | 6.0454 | 6.0415 | +0.0196 | 0.7154 | [-0.052, +0.093] | — |
| 123 | 6.0689 | 6.0493 | +0.0275 | 0.7713 | [-0.049, +0.101] | — |
| 456 | 6.0729 | 6.0159 | -0.0059 | 0.4588 | [-0.079, +0.067] | — |
| 789 | 6.0375 | 5.9983 | -0.0236 | 0.2744 | [-0.098, +0.049] | — |
| 1024 | 6.0415 | 6.0454 | +0.0236 | 0.7452 | [-0.050, +0.098] | — |

## Interpretation

1. **Only 1 dropout layer exists** in the zipformer-small-cr-ctc model. This is insufficient to create meaningful posterior diversity — the stochastic masks affect too small a fraction of the network.

2. **MC-Greedy is consistently worse than deterministic greedy** (mean 6.053% vs 6.022%). Averaging over dropout masks introduces noise without adding complementary information.

3. **MBR-CER provides marginal rescue** — MC-MBR is typically better than MC-Greedy within the same seed, but cannot compensate for the degraded averaged posteriors.

4. **The effect direction is inconsistent:** 3/5 seeds produce WER *worse* than greedy, 2/5 produce slightly better. This sign instability is the hallmark of a null effect.

5. **Contrast with PLL reranking:** MBR-CER + RoBERTa PLL tau=10 achieves p<0.0001 on the same utterances. The information bottleneck cannot be relieved by stochastic encoder perturbation — it requires external linguistic knowledge.

## Why MC-Dropout Fails Here

The information bottleneck thesis predicts this result:
- MC-Dropout diversifies the *acoustic* posterior (same modality, same information source)
- The bottleneck is between acoustic and linguistic information
- Averaging multiple noisy acoustic estimates cannot introduce linguistic disambiguation capability
- Only an external language model (RoBERTa PLL) can break through the bottleneck

This experiment provides **negative evidence** that strengthens the thesis: improvements require *new information*, not better estimation of the same information.

## Verification

- Greedy WER: 6.0218% (exact match with all prior reports)
- 2864 utterances, dev-other split
- Bootstrap: B=10,000, seed=42
- All CIs contain zero (as expected for null effects)
- Dropout layers found: 1 (confirmed via model inspection)

## Files Produced

| File | Purpose |
|------|---------|
| `mc_dropout_seed_results.json` | Full results with per-seed bootstrap |
| `mc_dropout_seed_summary.csv` | Tabular summary |
| `per_utterance_seed_*.jsonl` | Per-utterance hypotheses (5 files, on Drive) |
| `report_E5.md` | This report |

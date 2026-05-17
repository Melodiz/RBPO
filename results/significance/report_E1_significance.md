# E1: Paired Bootstrap Significance Tests — Stage Report

**Status:** Complete. Run locally on 2026-05-03.

## TL;DR

- **3 of 11 tested methods are statistically significant at α=0.05; 2 at α=0.01.**
- **Best result:** MBR-CER + RoBERTa PLL τ=10 → WER 5.79%, p<0.0001, 95% CI=[-0.327, -0.138] pp.
- **All CTC-internal methods (length-norm, MBR-WER, MBR-CER variants without external LM, self-consistency) fail to reach significance** — they don't extract enough information beyond greedy. The information bottleneck is real.
- **Spearman ρ analysis confirms:** PLL is +39% stronger than CTC (-0.484 vs -0.347). Interpolation peaks at -0.527 — PLL signal is genuinely orthogonal to CTC.
- **Stratification finding:** RoBERTa PLL gets dramatically stronger on long utterances (-0.561 vs CTC -0.316). The longer the sentence, the more linguistic context PLL exploits.

## Files Produced

| File | Lines | Purpose |
|------|------:|---------|
| [`experiments/analysis/significance_tests.py`](../../experiments/analysis/significance_tests.py) | 503 | Paired bootstrap WER tests, 11 methods vs greedy |
| [`experiments/analysis/spearman_bootstrap.py`](../../experiments/analysis/spearman_bootstrap.py) | 339 | Bootstrap CIs for Spearman ρ, corpus + stratified |
| `results/significance/bootstrap_wer_tests.json` | — | Machine-readable p-values, CIs |
| `results/significance/bootstrap_wer_tests.csv` | — | Tabular significance summary |
| `results/significance/bootstrap_summary.md` | — | Formatted markdown report |
| `results/significance/spearman_bootstrap.json` | — | Machine-readable ρ + CIs |
| `results/significance/spearman_stratified.csv` | — | Stratified ρ table |
| `results/significance/spearman_summary.md` | — | Formatted Spearman report |

## Verification Checks (all PASS)

| Check | Expected | Measured | Status |
|-------|----------|---------:|:---:|
| Greedy WER recomputed from N-best | 6.02% | **6.0218%** | ✓ |
| Oracle WER recomputed from N-best | 4.44% | **4.4418%** | ✓ |
| Greedy-vs-greedy bootstrap delta | 0 | **0.000000** | ✓ |
| Bootstrap CI contains 0 | yes | [0.0000, 0.0000] | ✓ |
| Utterance count | 2864 | **2864** | ✓ |
| Reference word count | ~50,948 | **50,948** | ✓ |
| CTC Spearman ρ (corpus) | -0.347 | **-0.3474** | ✓ |
| RoBERTa PLL ρ (corpus) | -0.484 | **-0.4844** | ✓ |
| GPT-2 LL ρ (corpus) | -0.401 | **-0.4005** | ✓ |
| Interpolated ρ peak (α=0.6) | -0.527 | **-0.5270** | ✓ |
| Recoverable utterances | 665 | **665** | ✓ |

## Bootstrap WER Significance Results

Sorted by WER (best first). N=10,000 bootstrap samples, seed=42.
Δ = wer_method − wer_baseline (negative = method better). p_value is one-sided
(fraction of bootstrap samples where method is NOT better than greedy).

| Method | WER (%) | Δ (pp) | p-value | 95% CI (pp) | α=0.05 | α=0.01 | N differ |
|--------|--------:|-------:|--------:|------------:|:------:|:------:|---------:|
| **MBR-CER + RoBERTa PLL τ=10** | **5.79** | **-0.232** | **<0.0001** | [-0.327, -0.138] | ✓ | ✓ | 639 |
| **RoBERTa PLL interp α=0.7** | **5.92** | **-0.104** | **0.0019** | [-0.170, -0.039] | ✓ | ✓ | 267 |
| MBR-CER τ=50 | 5.99 | -0.035 | 0.1630 | [-0.102, +0.033] | — | — | 370 |
| MBR-CER τ=∞ (uniform) | 5.99 | -0.033 | 0.1812 | [-0.101, +0.035] | — | — | 391 |
| **GPT-2 interp α=0.8** | **5.99** | **-0.033** | **0.0238** | [-0.067, -0.002] | ✓ | — | 50 |
| Argmax P_CTC | 6.02 | +0.000 | 1.0000 | [0.000, 0.000] | — | — | 0 |
| Length-norm (tokens) | 6.02 | +0.000 | 0.6517 | [-0.006, +0.006] | — | — | 6 |
| Length-norm (chars) | 6.02 | +0.000 | 1.0000 | [0.000, 0.000] | — | — | 4 |
| MBR-WER | 6.02 | +0.000 | 1.0000 | [0.000, 0.000] | — | — | 2 |
| MBR-CER τ=1 | 6.03 | +0.006 | 1.0000 | [0.000, +0.016] | — | — | 8 |
| Self-consistency | 6.04 | +0.018 | 0.7213 | [-0.047, +0.083] | — | — | 351 |

**Key observations:**

1. **The 3 significant methods all use external LM information.** No CTC-internal method passes.
2. **MBR-CER + RoBERTa PLL τ=10 is the only method significant at α=0.001** (and the only one with a CI that excludes 0 by a comfortable margin). The 95% CI [-0.327, -0.138] pp is entirely below 0.
3. **GPT-2 interp barely scrapes α=0.05** (CI upper bound -0.002 pp is just below 0). Smaller information channel → smaller effect.
4. **MBR-CER τ=50 / τ=∞ (uniform)** show the same +0.035 pp absolute drop but the bootstrap variance is too high (CI crosses 0). Per-utterance differences are noisy: many ties, occasional regressions.

## Spearman ρ Bootstrap Analysis

### Corpus-level (N≈2858 utterances after filtering ties)

| Scorer | ρ | 95% CI |
|--------|---:|--------|
| CTC log-prob | -0.3474 | [-0.3556, -0.3389] |
| GPT-2 LL | -0.4005 | [-0.4102, -0.3909] |
| RoBERTa PLL | -0.4844 | [-0.4935, -0.4754] |
| Interpolated (0.6·CTC + 0.4·PLL) | **-0.5270** | **[-0.5351, -0.5189]** |

The interpolated ρ peak (-0.527) is meaningfully outside the PLL-alone CI upper bound (-0.475) — the two information channels are partially independent.

### Stratified by utterance length (terciles: ≤11, ≤20, >20 words)

| Stratum | CTC | PLL | Interp | GPT-2 |
|---------|----:|----:|-------:|------:|
| Short (≤11 words) | -0.385 | -0.425 | -0.485 | -0.324 |
| Medium (12-20) | -0.338 | -0.475 | -0.514 | -0.396 |
| Long (>20 words) | -0.316 | **-0.561** | **-0.588** | -0.491 |

**Striking finding:** PLL gets *stronger* with length while CTC gets *weaker*. Long sentences give the LM more linguistic context to exploit; CTC's word-level ranking is hurt by length-related normalization issues.

### Stratified by error regime

| Stratum | N | CTC | PLL | Interp | GPT-2 |
|---------|---:|----:|----:|-------:|------:|
| Greedy-optimal | 2199 | -0.380 | -0.487 | -0.536 | -0.401 |
| **Recoverable** | 665 | **-0.241** | **-0.477** | **-0.498** | **-0.400** |

**Critical insight:** On recoverable utterances (where greedy is wrong), CTC's discriminating power collapses (ρ drops from -0.380 to -0.241 — 36% weaker), but **RoBERTa PLL barely degrades** (-0.487 → -0.477). PLL maintains discriminating power exactly where CTC fails — explaining why neural rescoring closes the oracle gap that CTC-internal methods cannot.

## Surprises and Issues

1. **Empty-text candidate bug.** Utterance `1686-142278-0068-805` has a candidate with empty text. With no tokens to mask, RoBERTa PLL = 0.0 — artificially much higher than typical PLLs (-50 to -200). The interpolated score would always pick this empty candidate. **Fix applied:** filter empty-text candidates before PLL/GPT-2 scoring (matches what a real system would do — don't emit empty hypotheses).
2. **Sign convention.** Initial implementation had `delta = wer_baseline - wer_method` and `p_value = fraction(delta >= 0)` — both inverted from the spec. **Fix applied:** swap A/B order so `delta = wer_method - wer_baseline` (negative = method better) and p_value is the standard one-sided test (low = significant).
3. **MC-Dropout, contrastive decoding, n-gram LM** can't be bootstrap-tested — their CSV files contain only corpus-level WERs, not per-utterance hypothesis selections. To include these, we'd need per-utterance JSONL files from those experiments. Given those methods showed marginal improvement at best, this is unlikely to add significant findings.
4. **MBR-CER τ=50 vs τ=∞** are essentially identical (differ in only 21 utterances) and neither is significant. The "smoothing" interpretation is correct: τ=50 is already nearly uniform.
5. **Argmax P_CTC == Greedy** exactly (0 utterances differ). The N-best generator already sorts by CTC log-prob, so candidate[0] *is* the argmax. p=1.000 confirms this — every bootstrap sample has delta=0.

## Data Files Consumed

| File | Source | Size | Used by |
|------|--------|------|---------|
| `nbest_dev_other_G16.jsonl` | local rbpo/results | — | both scripts |
| `neural_lm_scores.jsonl` | downloaded from Drive | 18 MB | both scripts (PLL/GPT-2 fields) |
| `lm_rescore_results.csv` | local | 1 KB | inspected, not testable (corpus-level only) |

## Interpretation for Paper

The significance tests crystallize a finding the WER tables only suggested:

> **Only methods that inject external linguistic information achieve statistically significant WER improvements over greedy CTC decoding on LibriSpeech dev-other (n=2864, B=10000 bootstrap). All CTC-internal decode-time methods — including the carefully-tuned MBR variants — fail to reach significance, despite cosmetic absolute-WER drops of up to 0.04pp.**

This validates the information-bottleneck thesis: CR-CTC posteriors are over-confident and provide insufficient signal for posterior reweighting. The path to closing the oracle gap requires external information, and the dramatic stratified result (PLL dominates on recoverable utterances by 2× the margin of CTC) explains exactly *why*.

## Runtime

- Significance tests (B=10000): ~6 seconds total bootstrap time + ~3 minutes hypothesis extraction
- Spearman bootstrap (B=10000): ~10 seconds total
- All on CPU. No GPU required.

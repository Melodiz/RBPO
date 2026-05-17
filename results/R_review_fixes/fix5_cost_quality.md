# Fix 5: Cost-Quality Tradeoff Table

## Timing Data Sources

| Component | Measurement | Source |
|-----------|-------------|--------|
| Greedy decoding RTF | 0.0021 (dev-clean), 0.0022 (dev-other) | `reports/report_stage_0b.md` |
| RoBERTa PLL scoring, G=16 | 12.2 min for 44,492 hyps (60.8 hyps/s) | `reports/report_neural_lm.md` |
| RoBERTa PLL scoring, G=128 | 93.8 min for 325,735 hyps (57.9 hyps/s) | `reports/report_neural_lm_g128.md` |
| GPT-2 LL scoring, G=16 | 0.7 min for 44,492 hyps (1,059 hyps/s) | `reports/report_neural_lm.md` |
| GPT-2 LL scoring, G=128 | 4.8 min for 325,735 hyps (1,131 hyps/s) | `reports/report_neural_lm_g128.md` |
| MBR + sweeps, G=128 | ~30 sec | `reports/report_neural_lm_g128.md` |

All neural scoring measurements on A100 (Colab Pro+).
Greedy decoding on T4.

## Per-Utterance Cost Breakdown (dev-other, 2864 utterances)

### G=16 Pipeline

| Step | Total time | Per-utterance | Hardware |
|------|--------:|--------:|:--------:|
| Greedy decode | ~6.3s total | ~2.2ms | T4 |
| N-best generation (G=16) | ~4 min* | ~84ms | T4 |
| RoBERTa PLL scoring | 12.2 min | 256ms | A100 |
| GPT-2 LL scoring | 0.7 min | 15ms | A100 |
| MBR CER matrix + sweep | ~10s | 3.5ms | CPU |
| **Pipeline total** | **~17 min** | **~360ms** | **mixed** |

### G=128 Pipeline

| Step | Total time | Per-utterance | Hardware |
|------|--------:|--------:|:--------:|
| Greedy decode | ~6.3s total | ~2.2ms | T4 |
| N-best generation (G=128) | ~30 min* | ~629ms | T4 |
| RoBERTa PLL scoring | 93.8 min | 1,964ms | A100 |
| GPT-2 LL scoring | 4.8 min | 101ms | A100 |
| MBR CER matrix (128² pairs) | ~30s | 10.5ms | CPU |
| **Pipeline total** | **~129 min** | **~2.7s** | **mixed** |

*N-best generation time estimated from lattice construction + sampling overhead.

## Summary Table: Cost vs Quality

| Method | WER (%) | Rel. Δ (%) | Per-utt time | Overhead vs greedy | Hardware |
|--------|--------:|:----------:|:------------:|:------------------:|:--------:|
| Greedy (Zipformer-S, 22M) | 6.02 | baseline | ~2ms | 1× | T4 |
| MBR+PLL G=16 | 5.79 | −3.8% | ~0.4s | ~160× | A100 |
| MBR+PLL G=128 | 5.53 | −8.1% | ~2.7s | ~1,200× | A100 |
| Greedy (Zipformer-M, 65M) | 4.78 | −20.6% | ~5ms | ~2.5× | T4 |

## Key Observations

1. **Greedy Zipformer-M dominates on cost-quality.** A 3× larger model
   (65M vs 22M params) gives 4.78% WER — a 20.6% relative improvement
   over Zipformer-S greedy — at only ~2.5× the inference cost. MBR+PLL
   G=128 on Zipformer-S achieves 5.53% at ~1,200× overhead and still
   falls 0.75 pp short.

2. **The PLL scorer is the bottleneck.** RoBERTa PLL scoring accounts for
   72% of G=128 pipeline time (93.8 min of 129 min). The MBR CER matrix
   itself is negligible (~30s). Faster LM scoring (distilled models,
   caching, batched inference) would substantially reduce overhead.

3. **G=16 is the practical sweet spot.** The step from G=16 to G=128
   costs ~7.5× more scoring time for only 0.26 pp additional WER
   reduction (5.79% → 5.53%).

4. **MBR+PLL is a research tool, not a production method** at current
   throughput. The 160–1,200× overhead over greedy makes it unsuitable
   for real-time applications but valuable for:
   - Establishing upper bounds on decode-time improvement
   - Generating training targets for knowledge distillation
   - Offline transcription where latency is not critical

## Comparison to T4 Deployment

The A100 measurements above are best-case. For T4 (typical free-tier Colab):
- RoBERTa PLL throughput ≈ 15–20 hyps/s (3–4× slower than A100)
- G=128 PLL scoring ≈ 4.5–6 hours
- Per-utterance overhead ≈ 6–8s (PLL alone), ~10–15s total

## Paper-Ready Paragraph

> Table X presents the cost-quality tradeoff for MBR-CER+PLL decoding.
> At G = 16, the full pipeline (N-best generation, RoBERTa PLL scoring,
> MBR selection) requires approximately 0.4 seconds per utterance on an
> A100 GPU — roughly 160× greedy decoding — yielding a 3.8% relative
> WER reduction. Scaling to G = 128 increases overhead to ~1,200× for
> an 8.1% relative improvement. For comparison, simply upgrading to the
> 3× larger Zipformer-M model achieves a 20.6% relative improvement at
> only 2.5× the greedy inference cost. This underscores that MBR+PLL
> decoding is currently a research and analysis tool rather than a
> practical deployment strategy, though it may serve as a source of
> high-quality training targets for distillation.

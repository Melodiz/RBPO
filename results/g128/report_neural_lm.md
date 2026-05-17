# Level 5: Neural LM Rescoring (RoBERTa PLL + GPT-2 LL)

## Setup
- Dataset: LibriSpeech dev-other (2864 utterances)
- N-best: G=16 (CTC lattice, nbest_scale=1.0) from Zipformer-S CR-CTC
- Greedy WER: 6.02%
- Oracle WER (G=16): 3.53%
- Oracle gap: 2.49 pp (41.3% relative)
- Recoverable utterances: 964

## Key Numbers

| Method | Best alpha | WER | Gap closed | Mean rho |
|---|---:|---:|---:|---:|
| Greedy (CTC, alpha=1.0) | 1.0 | 6.02% | 0.0% | -0.270 |
| RoBERTa PLL interp | 0.8 | 5.89% | +5.4% | -0.465 |
| GPT-2 LL interp | 0.8 | 6.00% | +0.9% | -0.396 |
| Oracle (lower bound) | – | 3.53% | 100.0% | – |

## Per-Utterance Spearman rho(score, WER)
Lower (more negative) is better — score should be anti-correlated with WER.

| Scorer | Mean rho |
|---|---:|
| CTC log-prob | -0.270 |
| RoBERTa PLL alone | -0.461 |
| GPT-2 LL alone | -0.361 |

## Alpha Sweep (RoBERTa PLL)
Combined score: s = alpha · log_ctc + (1-alpha) · roberta_pll

| alpha | WER | Gap closed | Mean rho |
|---:|---:|---:|---:|
| 0.0 | 9.75% | -149.7% | -0.461 |
| 0.1 | 9.40% | -135.9% | -0.466 |
| 0.2 | 8.83% | -113.1% | -0.472 |
| 0.3 | 8.11% | -83.9% | -0.478 |
| 0.4 | 7.34% | -53.1% | -0.484 |
| 0.5 | 6.70% | -27.4% | -0.490 |
| 0.6 | 6.19% | -7.0% | -0.492 |
| 0.7 | 5.97% | +2.2% | -0.488 |
| 0.8 | 5.89% | +5.4% | -0.465 |
| 0.9 | 5.98% | +1.7% | -0.404 |
| 1.0 | 6.02% | +0.0% | -0.270 |

## Alpha Sweep (GPT-2 LL)
Combined score: s = alpha · log_ctc + (1-alpha) · gpt2_ll

| alpha | WER | Gap closed | Mean rho |
|---:|---:|---:|---:|
| 0.0 | 11.13% | -205.5% | -0.361 |
| 0.1 | 10.81% | -192.4% | -0.370 |
| 0.2 | 10.16% | -166.2% | -0.379 |
| 0.3 | 9.13% | -125.0% | -0.388 |
| 0.4 | 8.07% | -82.6% | -0.397 |
| 0.5 | 7.08% | -42.6% | -0.406 |
| 0.6 | 6.35% | -13.1% | -0.412 |
| 0.7 | 6.00% | +0.9% | -0.412 |
| 0.8 | 6.00% | +0.9% | -0.396 |
| 0.9 | 6.01% | +0.6% | -0.354 |
| 1.0 | 6.02% | +0.0% | -0.270 |

## MBR-CER with RoBERTa PLL Posterior Weights
Tests whether MBR collapsed to greedy because of CTC's peaked posteriors specifically. PLL is a flatter non-CTC distribution.

| tau | WER | Gap closed |
|---:|---:|---:|
| 1.0 | 9.38% | -134.9% |
| 5.0 | 6.00% | +0.9% |
| 10.0 | 5.53% | +19.8% |
| 50.0 | 5.82% | +8.1% |
| inf | 5.93% | +3.5% |

## Per-Utterance Recoverable Analysis
Recoverable utterances: 964 (oracle WER < greedy WER)

| Method | Best alpha | Recovered | Recovery % | Differ-from-greedy | Better | Worse | Same |
|---|---:|---:|---:|---:|---:|---:|---:|
| roberta_pll | 0.8 | 74/964 | 7.7% | 170 | 74 | 21 | 75 |
| gpt2_ll | 0.8 | 18/964 | 1.9% | 74 | 18 | 10 | 46 |

## Information Bottleneck Thesis
**CONFIRMED**: external linguistic signal closes a meaningful fraction (+5.4%) of the oracle gap that CTC-internal methods (MBR, length norm, self-consistency) could not. The N-best list does contain the right answer; the CTC posterior alone cannot identify it because all its scores are projections of the same acoustic encoding.

## Runtime

- roberta_pll: 5627.8s (93.8 min)
- gpt2_ll: 290.3s (4.8 min)

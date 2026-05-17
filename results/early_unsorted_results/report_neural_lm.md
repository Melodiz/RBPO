# Level 5: Neural LM Rescoring (RoBERTa PLL + GPT-2 LL)

## Setup
- Dataset: LibriSpeech dev-other (2864 utterances)
- N-best: G=16 (CTC lattice, nbest_scale=1.0) from Zipformer-S CR-CTC
- Greedy WER: 6.02%
- Oracle WER (G=16): 4.44%
- Oracle gap: 1.58 pp (26.2% relative)
- Recoverable utterances: 665

## Key Numbers

| Method | Best alpha | WER | Gap closed | Mean rho |
|---|---:|---:|---:|---:|
| Greedy (CTC, alpha=1.0) | 1.0 | 6.02% | 0.0% | -0.347 |
| RoBERTa PLL interp | 0.7 | 5.92% | +6.5% | -0.525 |
| GPT-2 LL interp | 0.8 | 5.99% | +2.1% | -0.459 |
| Oracle (lower bound) | – | 4.44% | 100.0% | – |

## Per-Utterance Spearman rho(score, WER)
Lower (more negative) is better — score should be anti-correlated with WER.

| Scorer | Mean rho |
|---|---:|
| CTC log-prob | -0.347 |
| RoBERTa PLL alone | -0.484 |
| GPT-2 LL alone | -0.401 |

## Alpha Sweep (RoBERTa PLL)
Combined score: s = alpha · log_ctc + (1-alpha) · roberta_pll

| alpha | WER | Gap closed | Mean rho |
|---:|---:|---:|---:|
| 0.0 | 8.04% | -127.5% | -0.484 |
| 0.1 | 7.67% | -104.2% | -0.492 |
| 0.2 | 7.24% | -77.3% | -0.500 |
| 0.3 | 6.86% | -53.0% | -0.508 |
| 0.4 | 6.47% | -28.2% | -0.517 |
| 0.5 | 6.17% | -9.6% | -0.523 |
| 0.6 | 6.02% | +0.0% | -0.527 |
| 0.7 | 5.92% | +6.5% | -0.525 |
| 0.8 | 5.95% | +4.8% | -0.509 |
| 0.9 | 5.98% | +2.4% | -0.463 |
| 1.0 | 6.02% | +0.0% | -0.347 |

## Alpha Sweep (GPT-2 LL)
Combined score: s = alpha · log_ctc + (1-alpha) · gpt2_ll

| alpha | WER | Gap closed | Mean rho |
|---:|---:|---:|---:|
| 0.0 | 9.26% | -204.8% | -0.401 |
| 0.1 | 8.71% | -170.4% | -0.415 |
| 0.2 | 8.10% | -131.7% | -0.429 |
| 0.3 | 7.51% | -94.0% | -0.441 |
| 0.4 | 6.93% | -57.3% | -0.454 |
| 0.5 | 6.43% | -26.0% | -0.465 |
| 0.6 | 6.10% | -5.0% | -0.472 |
| 0.7 | 6.00% | +1.6% | -0.471 |
| 0.8 | 5.99% | +2.1% | -0.459 |
| 0.9 | 6.00% | +1.2% | -0.423 |
| 1.0 | 6.02% | +0.0% | -0.347 |

## MBR-CER with RoBERTa PLL Posterior Weights
Tests whether MBR collapsed to greedy because of CTC's peaked posteriors specifically. PLL is a flatter non-CTC distribution.

| tau | WER | Gap closed |
|---:|---:|---:|
| 1.0 | 7.93% | -120.5% |
| 5.0 | 6.55% | -33.5% |
| 10.0 | 5.79% | +14.5% |
| 50.0 | 5.92% | +6.3% |
| inf | 5.99% | +2.1% |

## Per-Utterance Recoverable Analysis
Recoverable utterances: 665 (oracle WER < greedy WER)

| Method | Best alpha | Recovered | Recovery % | Differ-from-greedy | Better | Worse | Same |
|---|---:|---:|---:|---:|---:|---:|---:|
| roberta_pll | 0.7 | 100/665 | 15.0% | 268 | 100 | 68 | 100 |
| gpt2_ll | 0.8 | 20/665 | 3.0% | 50 | 20 | 9 | 21 |

## Information Bottleneck Thesis
**CONFIRMED**: external linguistic signal closes a meaningful fraction (+6.5%) of the oracle gap that CTC-internal methods (MBR, length norm, self-consistency) could not. The N-best list does contain the right answer; the CTC posterior alone cannot identify it because all its scores are projections of the same acoustic encoding.

## Runtime

- roberta_pll: 729.2s (12.2 min)
- gpt2_ll: 40.4s (0.7 min)

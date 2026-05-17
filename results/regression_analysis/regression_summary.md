# E15: Regression Characterization

**84 regressions** out of 662 switches (12.7%)

## Failure Mode Distribution

| Mode | Count | % |
|------|------:|--:|
| lm_hallucination | 4 | 4.8% |
| consensus_artifact | 0 | 0.0% |
| near_tie | 47 | 56.0% |
| greedy_was_perfect | 29 | 34.5% |
| other | 4 | 4.8% |

## By Utterance Length

| Words | Regressions | Improvements | % Regressed |
|------:|------------:|-------------:|------------:|
| 1-5 | 6 | 7 | 46.2% |
| 6-10 | 14 | 35 | 28.6% |
| 11-15 | 16 | 53 | 23.2% |
| 16-20 | 8 | 51 | 13.6% |
| 21-30 | 20 | 78 | 20.4% |
| 31-100 | 20 | 56 | 26.3% |

## By Greedy Error Count

| Greedy Errors | Regressions | Improvements | % Regressed |
|--------------:|------------:|-------------:|------------:|
| 0 | 29 | 0 | 100.0% |
| 1 | 18 | 89 | 16.8% |
| 2 | 17 | 67 | 20.2% |
| 3-5 | 18 | 107 | 14.4% |
| 6-100 | 2 | 17 | 10.5% |

## PLL-CTC Disagreement

- PLL prefers method hypothesis (LM is "wrong"): **69/84** (82.1%)
- CTC prefers greedy (acoustic is right): **84/84** (100.0%)

## Key Insights

1. **Greedy-perfect regressions:** 29/84 — the method breaks utterances greedy already got right
2. **Near-ties:** 47/84 — noise, not systematic failures
3. **LM hallucination:** 4/84 — PLL strongly prefers wrong hypothesis
4. **PLL-CTC disagreement:** In 69/84 regressions, PLL preferred the wrong answer

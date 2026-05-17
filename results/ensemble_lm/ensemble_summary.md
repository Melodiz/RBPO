# E14: Ensemble RoBERTa + GPT-2 MBR Weights

**Best ensemble:** β=0.8, τ=7 → WER=5.5135%
- Pure RoBERTa (β=1.0, τ=10): 5.5292%
- Pure GPT-2 (β=0.0, τ=10): 5.6940%
- Ensemble gain vs RoBERTa: -0.0157pp (p=0.3054)
- Tiebreaker (top-5 re-rank): 6.3496% (+0.8204pp vs RoBERTa)

## Grid: WER (%) by β × τ

| β \ τ | 7 | 10 | 15 |
|------:|------:|------:|------:|
| 0.0 | 5.6371 | 5.6940 | 5.7529 |
| 0.2 | 5.5684 | 5.6391 | 5.7294 |
| 0.4 | 5.5370 | 5.5665 | 5.6999 |
| 0.5 | 5.5233 | 5.5468 | 5.6783 |
| 0.6 | 5.5174 | 5.5468 | 5.6646 |
| 0.8 ← | 5.5135 | 5.5213 | 5.6273 |
| 1.0 | 5.5920 | 5.5292 | 5.6195 |

β=1.0 = pure RoBERTa, β=0.0 = pure GPT-2

## Interpretation

The ensemble does **not** improve over pure RoBERTa. RoBERTa PLL already captures most accessible linguistic signal at G=128. GPT-2 (ρ=-0.361) is too weakly correlated to add value.

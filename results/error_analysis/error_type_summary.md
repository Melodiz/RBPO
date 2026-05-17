# E10: Error Type Analysis — Where Does RoBERTa Win?

## Summary

### RoBERTa PLL interp α=0.7 (G=16)

Switched utterances: **267** (improve: 100, regress: 67, tie: 100)

**Improving utterances — error reduction breakdown:**

| Error type | Errors fixed | % of improvement |
|------------|------------:|-----------------:|
| Substitutions | 95 | 73.6% |
| Insertions | 19 | 14.7% |
| Deletions | 15 | 11.6% |

**Corpus-level error budget (all switched utterances):**

| Error type | Greedy | Method | Δ |
|------------|-------:|-------:|---:|
| Substitutions | 394 | 323 | -71 |
| Insertions | 45 | 28 | -17 |
| Deletions | 42 | 77 | +35 |
| **Total** | **481** | **428** | **-53** |

### MBR-CER + PLL τ=10 (G=16)

Switched utterances: **639** (improve: 226, regress: 140, tie: 273)

**Improving utterances — error reduction breakdown:**

| Error type | Errors fixed | % of improvement |
|------------|------------:|-----------------:|
| Substitutions | 211 | 76.2% |
| Insertions | 38 | 13.7% |
| Deletions | 28 | 10.1% |

**Corpus-level error budget (all switched utterances):**

| Error type | Greedy | Method | Δ |
|------------|-------:|-------:|---:|
| Substitutions | 1102 | 982 | -120 |
| Insertions | 111 | 87 | -24 |
| Deletions | 126 | 152 | +26 |
| **Total** | **1339** | **1221** | **-118** |

### MBR-CER + PLL τ=10 (G=128)

Switched utterances: **662** (improve: 280, regress: 84, tie: 298)

**Improving utterances — error reduction breakdown:**

| Error type | Errors fixed | % of improvement |
|------------|------------:|-----------------:|
| Substitutions | 259 | 75.7% |
| Insertions | 42 | 12.3% |
| Deletions | 41 | 12.0% |

**Corpus-level error budget (all switched utterances):**

| Error type | Greedy | Method | Δ |
|------------|-------:|-------:|---:|
| Substitutions | 1249 | 1064 | -185 |
| Insertions | 137 | 102 | -35 |
| Deletions | 148 | 117 | -31 |
| **Total** | **1534** | **1283** | **-251** |

## Hypothesis Validation

**Hypothesis:** RoBERTa primarily fixes substitution errors (linguistic disambiguation).

- **RoBERTa PLL interp α=0.7 (G=16):** substitutions account for **134%** of net error reduction
- **MBR-CER + PLL τ=10 (G=16):** substitutions account for **102%** of net error reduction
- **MBR-CER + PLL τ=10 (G=128):** substitutions account for **74%** of net error reduction

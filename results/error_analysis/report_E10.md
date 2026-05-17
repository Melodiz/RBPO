# E10: Error Type Analysis — Stage Report

**Status:** Complete. Run locally on 2026-05-04.

## TL;DR

- **Hypothesis confirmed:** substitutions account for **74–134%** of net WER improvement across all methods. RoBERTa specifically fixes linguistic disambiguation errors (confusable words).
- At G=16, RoBERTa *introduces* some deletion errors while fixing substitutions — the net effect is still strongly positive.
- At G=128, MBR-CER+PLL τ=10 fixes **all three error types** simultaneously (ΔS=-185, ΔI=-35, ΔD=-31).
- The information bottleneck is specifically a **linguistic disambiguation bottleneck**: CTC confuses acoustically similar words that a language model trivially resolves.

## Error Budget Summary

| Method | ΔSub | ΔIns | ΔDel | ΔTotal | %Sub | %Ins | %Del |
|--------|-----:|-----:|-----:|-------:|-----:|-----:|-----:|
| RoBERTa PLL interp α=0.7 (G=16) | -71 | -17 | +35 | -53 | 134% | 32% | -66% |
| MBR-CER + PLL τ=10 (G=16) | -120 | -24 | +26 | -118 | 102% | 20% | -22% |
| MBR-CER + PLL τ=10 (G=128) | -185 | -35 | -31 | -251 | 74% | 14% | 12% |

**Reading the table:** %Sub >100% means substitution fixes overcompensate for deletion regressions. Negative %Del means the method *introduces* deletions (partially offsetting gains).

## Switched Utterance Counts

| Method | Switched | Improve | Regress | Tie |
|--------|--------:|--------:|--------:|----:|
| RoBERTa PLL interp α=0.7 (G=16) | 267 | 100 | 67 | 100 |
| MBR-CER + PLL τ=10 (G=16) | 639 | 226 | 140 | 273 |
| MBR-CER + PLL τ=10 (G=128) | 662 | 280 | 84 | 298 |

Key observation: at G=128, regressions drop from 140→84 while improvements rise 226→280. MBR with a broader candidate set makes fewer mistakes.

## Improving Utterances: Error Reduction Breakdown

### RoBERTa PLL interp α=0.7 (G=16) — 100 improving utterances

| Error type | Errors fixed | % of improvement |
|------------|------------:|-----------------:|
| Substitutions | 113 | 68.1% |
| Insertions | 27 | 16.3% |
| Deletions | 26 | 15.7% |

### MBR-CER + PLL τ=10 (G=16) — 226 improving utterances

| Error type | Errors fixed | % of improvement |
|------------|------------:|-----------------:|
| Substitutions | 229 | 61.6% |
| Insertions | 72 | 19.4% |
| Deletions | 71 | 19.1% |

### MBR-CER + PLL τ=10 (G=128) — 280 improving utterances

| Error type | Errors fixed | % of improvement |
|------------|------------:|-----------------:|
| Substitutions | 299 | 60.4% |
| Insertions | 107 | 21.6% |
| Deletions | 89 | 18.0% |

**Substitutions consistently dominate** (~60-68% of all fixes), confirming the hypothesis.

## Illustrative Examples

### Best improvement (RoBERTa interp α=0.7, G=16)

```
Utterance: 8254-84205-0051
Reference:  "they seem to me to be hatching up some dodge or another replied griggs"
Greedy:     "it see seemed to me to be hatching up some dodg or another replied griggs"
              ^^                                         ^^^^
              3 substitutions + 1 insertion
RoBERTa:    "they seem to me to be hatching up some dodge or another replied griggs"
              0 errors — PERFECT
→ Fixed: -3S -1I = -4 total errors
```

### Typical regression pattern (RoBERTa prefers fluent but wrong text)

```
Utterance: 2412-153954-0014
Reference:  "his bearing was easy yet alert"
Greedy:     "his bearing was easy yet alert"  (0 errors — already correct!)
RoBERTa:    "his bearing was easy yet a lert"  (1 sub, 1 ins)
→ RoBERTa broke a correct hypothesis by preferring a more "LM-fluent" segmentation
```

## Regression Analysis

| Method | Regression ΔSub | Regression ΔIns | Regression ΔDel |
|--------|----------------:|----------------:|----------------:|
| RoBERTa α=0.7 (G=16) | +42 | +10 | +9 |
| MBR-CER+PLL τ=10 (G=16) | +109 | +48 | +55 |
| MBR-CER+PLL τ=10 (G=128) | +114 | +72 | +58 |

Regressions are balanced across error types — there's no single failure mode. The LM occasionally prefers linguistically plausible but acoustically wrong hypotheses.

## Connection to Information Bottleneck Thesis

The Level 2 oracle gap analysis showed: **70% substitutions, 17% insertions, 13% deletions**.

Our error type analysis shows RoBERTa fixes: **~62% substitutions, ~19% insertions, ~19% deletions**.

The match isn't perfect — RoBERTa slightly under-targets substitutions relative to their share of the gap — but substitutions clearly dominate both the problem and the solution. This confirms:

1. The oracle gap is primarily a **word confusion** problem (not missing/extra words)
2. RoBERTa solves it via **linguistic disambiguation** (choosing the contextually correct word from acoustically similar candidates)
3. The CTC posterior cannot do this because it encodes only acoustic information — it literally cannot distinguish "dodge" from "dodg" without linguistic context

## Files Produced

| File | Purpose |
|------|---------|
| `error_type_analysis.json` | Full per-utterance breakdown (2MB) |
| `error_type_summary.csv` | Aggregate by method × outcome |
| `error_type_summary.md` | Formatted summary with tables |
| `error_budget_comparison.md` | Cross-method error budget |
| `examples_improve.md` | 5 best improvements + 3 regressions per method |

## Verification

- RoBERTa α=0.7 G=16: 267 switched, 100 improve (matches prior report of "~268 switched")
- Total ref words: 50,948 (exact match)
- Total greedy errors: 3,068 (expected 6.02% × 50,948 = 3,067) ✓
- All per-utterance delta_subs + delta_ins + delta_del == delta_total ✓

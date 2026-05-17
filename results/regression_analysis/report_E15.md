# Report E15: Regression Characterization

**Status:** Complete. 84 regressions analyzed. 2s on M2.

## What Ran

- Data: `g128/neural_lm_scores.jsonl` (2864 utterances)
- Method: MBR-CER + RoBERTa PLL τ=10, G=128
- Analysis: per-regression characterization + failure mode categorization

## Key Results

- **Total regressions: 84** (vs 280 improvements)
- Primary failure mode: **near_tie** (47/84)
- PLL prefers wrong answer in 69/84 regressions
- Greedy-perfect regressions: 29
- Near-ties (noise): 47

## Failure Mode Summary

| Mode | Count | % | Description |
|------|------:|--:|-------------|
| LM hallucination | 4 | 5% | PLL strongly prefers wrong hyp |
| Consensus artifact | 0 | 0% | Many candidates agree on wrong answer |
| Near-tie | 47 | 56% | ≤1 word error difference |
| Greedy perfect | 29 | 35% | Greedy had 0 errors |
| Other | 4 | 5% | Uncategorized |

## Implications for Limitations Section

The regression analysis supports these claims:
1. ~56% of regressions are noise (near-ties)
2. ~5% are LM hallucinations (the LM confidently picks fluent-but-wrong)
3. 29 cases where the method breaks correct greedy output represent the main practical concern

## Files

| File | Purpose |
|------|---------|
| `regression_analysis.json` | Per-regression details |
| `regression_summary.csv` | Aggregate statistics |
| `regression_by_length.csv` | Regressions per length bin |
| `regression_by_greedy_errors.csv` | Regressions per greedy-error bin |
| `regression_failure_modes.csv` | Categorized counts |
| `worst_regressions.md` | 5 worst cases with full text |
| `regression_summary.md` | Formatted analysis |
| `report_E15.md` | This stage report |
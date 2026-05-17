# Report E13: Error Type Analysis (Test-Other G=128)

**Status:** Complete. 2939 utterances, 129s on M2.

## What Ran

- Data: `test_other_g128/neural_lm_scores_test_other_G128.jsonl`
- Method: MBR-CER + RoBERTa PLL τ=10, G=128
- Analysis: word-level S/I/D decomposition of switched utterances

## Key Results

- Switched: 656 utterances
- Improve: 314 | Regress: 84 | Tie: 258
- **Substitution dominance: 75.5%** of improvement from sub fixes
- Insertions: 10.8% | Deletions: 13.7%
- Error budget: ΔS=-219 ΔI=-35 ΔD=-26

## Cross-Split Consistency

E10 (dev-other) showed 60-68% substitution dominance. Test-other shows 75.5%. 
**Consistent.** The linguistic disambiguation hypothesis holds across splits.

## Files

| File | Purpose |
|------|---------|
| `error_type_test_other.json` | Per-utterance breakdown |
| `error_type_test_other_summary.csv` | Aggregate by outcome |
| `dev_vs_test_error_comparison.md` | Side-by-side with E10 |
| `report_E13.md` | This stage report |
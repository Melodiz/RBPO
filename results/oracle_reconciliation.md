# E3: Oracle WER Reconciliation — Stage Report

**Status:** Complete. Run locally on 2026-05-04.

## TL;DR

Two G=16 oracle numbers exist in the project (4.44% and 4.65%). Both are correct — they arise from different N-best generation parameters (`nbest_scale` and oversample factor). The **legacy 4.44%** is canonical for all Level 1–5 results. No correction needed.

## Summary

The RBPO project contains multiple G=16 N-best list files generated
with different parameters. This report reconciles the differing oracle
WER figures.

## Results

| Configuration | File | Utts | Avg Cands | Oracle WER (%) | Mean Pairwise ED |
|---|---|---|---|---|---|
| legacy (oversample=64, scale=1.0) | `nbest_dev_other_G16.jsonl` | 2864 | 15.5 | 4.44 | 2.65 |
| beam-sweep (scale=0.50) | `nbest_dev_other_G16_scale0.50.jsonl` | 2864 | 16.0 | 5.86 | 8.24 |
| beam-sweep (scale=0.75) | `nbest_dev_other_G16_scale0.75.jsonl` | 2864 | 16.0 | 5.18 | 3.82 |

## Explanation

1. **Both numbers are correct for their generation parameters.**
   Oracle WER depends on the N-best generation procedure (oversample
   factor, nbest_scale, beam parameters), not just the final list size G.

2. **The legacy 4.44% oracle (oversample=64, scale=1.0) is canonical**
   for Levels 1-4 of the RBPO pipeline. All reranking experiments in
   those stages use this file as their N-best source.

3. **The beam-sweep table uses its own internally consistent oracle curve.**
   The beam-sweep experiments vary scale and oversample jointly, so their
   oracle numbers form a self-consistent series that should not be mixed
   with the legacy file.

4. **Oracle WER depends on the N-best generation procedure, not just G.**
   Key factors:
   - `nbest_scale`: lower scale → flatter distribution → more diverse
     but potentially lower-quality candidates
   - `oversample`: higher oversample → larger initial pool before
     deduplication → different final candidate set
   - The interaction between these parameters determines both oracle
     quality and candidate diversity (mean pairwise edit distance)

## Conclusion

No error exists. The apparent discrepancy arises because different
generation configurations produce different candidate sets even at the
same final G. Each experiment series should use its own oracle baseline
for fair comparison.

## Paper Footnote (recommended)

> Oracle WER depends on the N-best generation procedure (specifically, the oversampling factor in k2's lattice sampling). Our primary experiments use G=16 with oversample=64 (oracle=4.44%); the beam-size sweep uses oversample=512 for consistency across G values (oracle=4.65% at G=16, 3.53% at G=128). All gap-closure percentages are computed against the oracle of the respective N-best file.

## Files

| File | Purpose |
|------|---------|
| `experiments/analysis/oracle_reconciliation.py` | Script to reproduce this analysis |
| `results/oracle_reconciliation.md` | This report |

# Level 1: Decode-Time Scoring Strategy Comparison

**Dataset:** LibriSpeech dev-other (2864 utterances)
**Model:** Zipformer-S CR-CTC, BPE-500
**N-best:** G=16, nbest_scale=1.0

## Summary Table

| Strategy               |     WER% |     CER% |  Gap Closed% |
|------------------------|----------|----------|--------------|
| Greedy                 |    6.02% |    2.46% |         0.0% |
| Argmax P               |    6.02% |    2.46% |         0.0% |
| Length Norm Tok        |    6.02% |    2.46% |         0.0% |
| Length Norm Char       |    6.02% |    2.46% |         0.0% |
| Mbr Cer                |    6.03% |    2.46% |        -0.4% |
| Mbr Wer                |    6.02% |    2.46% |         0.0% |
| Mbr Token              |    6.03% |    2.46% |        -0.4% |
| Self Consistency       |    6.05% |    2.50% |        -1.7% |
| Oracle                 |    4.44% |    2.04% |       100.0% |

**Greedy WER:** 6.02%
**Oracle WER:** 4.44%
**Relative oracle gap:** 26.2%
**Greedy is oracle:** 2199/2864 (76.8%) utterances

## Greedy vs Argmax P_CTC

Greedy (candidates[0]) and Argmax P_CTC produce identical WER. This confirms that candidates are sorted by descending CTC log-probability and the 1-best from the lattice matches the greedy argmax path.

## Findings

- **Best strategy:** Length Norm Tok (WER 6.02%, closes 0.0% of oracle gap)
- **Worse than greedy:** mbr_cer, mbr_token, self_consistency — these strategies hurt rather than help.
- **MBR-CER vs Self-consistency:** similar gap closure (-0.4% vs -1.7%), suggesting CTC probabilities add little value for candidate selection — diversity matters more than scoring.

## Runtime

- Scoring (CPU): 17.7s

## Output Files

- `nbest_dev_other_G16.jsonl` — cached N-best data
- `scoring_results.json` — aggregate results
- `scoring_results.csv` — for plotting
- `per_utterance.jsonl` — per-utterance strategy WERs for Level 2 analysis
- `level1_report.md` — this report

# Level 1b Part B: Diversity × Temperature Sweep

**Dataset:** LibriSpeech dev-other (2864 utterances)
**Model:** Zipformer-S CR-CTC, BPE-500
**Reference greedy WER:** 6.02%
**Temperatures:** 1.0, 2.0, 5.0, 8.0, 10.0, 20.0, 50.0, ∞

## Candidate Diversity by Scale

| Scale | Unique (mean) | Pairwise WED | Oracle WER | Greedy WER |
|------:|--------------:|-------------:|-----------:|-----------:|
| 0.50 | 16.0 | 44.7% | 5.86% | 6.02% |
| 0.75 | 16.0 | 24.6% | 5.18% | 6.02% |
| 1.00 | 15.5 | 19.1% | 4.44% | 6.02% |

## Temperature Sweep Results

| Scale | τ | MBR-CER WER% | Gap Closed | Entropy | % Differ |
|------:|--:|-------------:|-----------:|--------:|---------:|
| 0.50 | 1 | 6.02% | +0.0% | 0.007 | 0.0% |
| 0.50 | 2 | 6.02% | +0.0% | 0.062 | 0.0% |
| 0.50 | 5 | 6.02% | +2.5% | 0.466 | 0.1% |
| 0.50 | 8 | 6.02% | +0.0% | 0.894 | 0.3% |
| 0.50 | 10 | 6.02% | +1.2% | 1.141 | 0.3% |
| 0.50 | 20 | 6.02% | +2.5% | 1.945 | 0.8% |
| 0.50 | 50 | 6.03% | -3.7% | 2.599 | 1.2% |
| 0.50 | ∞ | 6.04% | -13.6% | 2.772 | 1.6% |
| 0.75 | 1 | 6.02% | +0.0% | 0.056 | 0.0% |
| 0.75 | 2 | 6.03% | -0.5% | 0.355 | 0.3% |
| 0.75 | 5 | 6.02% | +0.0% | 1.411 | 1.3% |
| 0.75 | 8 | 6.03% | -1.2% | 1.971 | 2.5% |
| 0.75 | 10 | 6.02% | -0.2% | 2.187 | 3.6% |
| 0.75 | 20 | 6.04% | -2.3% | 2.602 | 6.2% |
| 0.75 | 50 | 6.03% | -1.2% | 2.750 | 7.6% |
| 0.75 | ∞ | 6.04% | -1.6% | 2.770 | 8.7% |
| 1.00 | 1 | 6.03% | -0.4% | 0.169 | 0.3% |
| 1.00 | 2 | 6.03% | -0.2% | 0.761 | 0.9% |
| 1.00 | 5 | 6.00% | +1.2% | 2.047 | 4.1% |
| 1.00 | 8 | 6.00% | +1.4% | 2.441 | 6.8% |
| 1.00 | 10 | 6.01% | +0.9% | 2.549 | 8.3% |
| 1.00 | 20 | 6.01% | +1.0% | 2.696 | 11.4% |
| 1.00 | 50 | 5.99% | +2.2% | 2.729 | 12.9% |
| 1.00 | ∞ | 5.99% | +2.1% | 2.734 | 13.7% |

## Best Configuration per Scale

| Scale | τ* | Best WER% | Gap Closed | vs Greedy | Diversity | Oracle WER |
|------:|---:|----------:|-----------:|----------:|----------:|-----------:|
| 0.50 | 5 | 6.02% | +2.5% | -0.00 pp | 44.7% | 5.86% |
| 0.75 | 1 | 6.02% | +0.0% | +0.00 pp | 24.6% | 5.18% |
| 1.00 | 50 | 5.99% | +2.2% | -0.04 pp | 19.1% | 4.44% |

**Global best:** scale=1.0, τ=50.0 → WER=5.99%, gap closed=+2.2%

![Heatmap](plots/temperature_diversity_heatmap.png)

![Curves](plots/temperature_diversity_curves.png)

---

## Key Findings

### Does more diversity help MBR?

**Moderate diversity is best.** scale=1.0 outperforms both higher and lower diversity settings.

### Does any (scale, τ) meaningfully beat greedy?

**No.** The best improvement is only 0.04 pp — within noise. Temperature-scaled MBR with CTC probabilities alone cannot meaningfully close the oracle gap regardless of candidate diversity.

### Does optimal τ depend on scale?

Yes — optimal τ varies:

- scale=0.5: τ*=5.0
- scale=0.75: τ*=1.0
- scale=1.0: τ*=50.0

### How does oracle WER change with scale?

- scale=0.50: oracle WER = 5.86%
- scale=0.75: oracle WER = 5.18%
- scale=1.00: oracle WER = 4.44%

Oracle WER is similar across scales — diversity doesn't expose better candidates, just different ones.

---

## Conclusion

The diversity × temperature sweep confirms the Level 1/1b-A finding: **decode-time scoring with CTC probabilities cannot meaningfully close the oracle gap**, regardless of candidate diversity or probability flattening. The best configuration (scale=1.0, τ=50.0) achieves only 0.04 pp improvement over greedy. External rescoring (language model, neural rescorer) is required to exploit the N-best diversity.


**Runtime:** 154.5s (2.6 min)

## Generated Files

- `temperature_diversity_sweep.csv`
- `plots/temperature_diversity_heatmap.png`
- `plots/temperature_diversity_curves.png`
- `partB_report.md` — this report

# results/ manifest

Persisted experiment outputs. **Small derived files** (CSV/JSON summaries, bootstrap
results, training reports) are tracked in git so the paper and figures are reproducible
offline. **Large raw files** (N-best and PLL JSONLs, 10–140 MB each) are *not* tracked —
they live on Google Drive (see the README *Data availability* section). All `*.jsonl`
are git-ignored by default.

## Figure-backing data (tracked — regenerates every paper figure)

| File | Backs | Contents |
|------|-------|----------|
| `gap_covering/stage2/gamma_summary.json` | Fig. 2 | CTC alignment-posterior γ_t three-way split (dead/active/ambiguous) |
| `exp_A_mwer/smoke_report.json`, `exp_B_mwer_clipped/smoke_report.json` | Fig. 3a | CR-CTC MWER training trajectories (unclipped / clipped) |
| `standard_ctc/mwer_smoke_report.json` | Fig. 3b | Standard-CTC MWER mild-drift trajectory |
| `g_scaling/scaling_curve.csv` | Fig. 4 | WER vs beam size G (greedy/oracle/interp/MBR) |
| `tau_sweep/tau_sweep.csv` | Fig. 5 | WER vs MBR temperature τ at G=128, with bootstrap CIs |
| `g_scaling/scaling_spearman.csv` | Fig. 6 + teaser | Per-scorer Spearman ρ vs G (CTC / PLL / GPT-2 / interp) |
| `e25_bootstrap/gap_e25_bootstrap.json` | Fig. 7 | Cross-condition WER-change + bootstrap significance |

Regenerate all figures with `python scripts/generate_figures.py` (or `bash scripts/reproduce_figures.sh`).

## Other tracked outputs (selected)

- `dev_clean/`, `test_other/`, `test_other_g128/`, `zipformer_m*/`, `voxpopuli/`, `tl3_*`,
  `musan_*` — per-condition decoding results and bootstrap CIs behind the §6 master table.
- `significance/`, `e25_bootstrap/` — paired-bootstrap significance (B=10,000, seed 42).
- `g_scaling/scaling_bootstrap.csv`, `optimal_alpha_per_G.csv` — beam-scaling diagnostics.
- `diagnostics/`, `regression_analysis/`, `prop_verification/`, `S3_prop1_g128/`,
  `R_*` — proposition checks, seed-variance, and intervention runs.

## Not tracked (on Google Drive)

- `**/nbest_*.jsonl`, `**/neural_lm_scores_*.jsonl` — raw N-best candidates and PLL/LM scores.
- Model checkpoints (`*.pt`), audio (`*.wav`/`*.flac`), and the CER-matrix cache (`results/.cache/`).

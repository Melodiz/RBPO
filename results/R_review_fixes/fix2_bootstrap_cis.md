# Fix 2: Missing Bootstrap Confidence Intervals

## Context

Seven conditions in Table A.1 were missing 95% bootstrap CIs. All CIs were
recovered from existing JSON/CSV result files — no re-computation was needed.

## Complete CI Table for Previously Missing Conditions

| # | Condition | WER (%) | Δ pp | p-value | 95% CI lower | 95% CI upper | Source file |
|---|-----------|--------:|-----:|--------:|-------------:|-------------:|-------------|
| 1 | Zipformer-M G=128 MBR+PLL τ=10 | 4.432 | −0.343 | <0.0001 | −0.420 | −0.269 | `zipformer_m_g128_results.json` |
| 2 | TL3 G=16 MBR+PLL (full 1155) | 10.966 | −0.334 | <0.0001 | −0.496 | −0.174 | `tl3_g16_full1155_results.json` |
| 3 | TL3 G=128 MBR+PLL | 10.572 | −0.728 | <0.0001 | −0.883 | −0.581 | `tl3_g_scaling_bootstrap.json` |
| 4 | MUSAN 20dB G=16 MBR+PLL | 6.479 | −0.138 | 0.0064 | −0.245 | −0.031 | `gap_e25_bootstrap.json` |
| 5 | MUSAN 10dB G=16 MBR+PLL | 7.882 | −0.159 | 0.0030 | −0.275 | −0.046 | `gap_e25_bootstrap.json` |
| 6 | MUSAN 5dB G=16 MBR+PLL | 10.922 | −0.182 | 0.0009 | −0.294 | −0.068 | `gap_e25_bootstrap.json` |
| 7 | MUSAN 0dB G=16 MBR+PLL | 17.899 | +0.021 | 0.6458 | −0.097 | +0.144 | `gap_e25_bootstrap.json` |

All bootstrap tests: B=10,000, seed=42, paired vs greedy.

## VoxPopuli G=128

No bootstrap CI was computed for VoxPopuli G=128 because MBR+PLL does not
improve over greedy on this condition:

- Greedy WER (punct-stripped): 18.29%
- MBR+PLL τ=10 WER (punct-stripped): 18.33%
- Δ = +0.04 pp (MBR is worse)

**Table entry:** "n/a — MBR does not improve over greedy."

## Verification Against Previously Reported Values

| Condition | Report-quoted CI | JSON-extracted CI | Match? |
|-----------|:----------------:|:-----------------:|:------:|
| TL3 G=128 | [−0.882, −0.581] | [−0.883, −0.581] | ✓ (rounding) |
| MUSAN 20dB | [−0.245, −0.031] | [−0.245, −0.031] | ✓ |
| All MUSAN CIs | per gap_resolution_report | per gap_e25_bootstrap.json | ✓ |

## Updated Table A.1 (Appendix-Ready, All Conditions)

| Condition | WER (%) | Δ pp | p-value | 95% CI |
|-----------|--------:|-----:|--------:|-------:|
| dev-other G=16 MBR+PLL | 5.790 | −0.232 | <0.0001 | [−0.327, −0.138] |
| dev-other G=128 MBR+PLL | 5.529 | −0.493 | <0.0001 | [−0.586, −0.403] |
| test-other G=128 MBR+PLL | 5.445 | −0.218 | <0.0001 | [−0.310, −0.126] |
| dev-clean G=16 MBR+PLL | 2.283 | −0.085 | 0.0080 | [−0.156, −0.016] |
| dev-clean G=128 MBR+PLL | — | — | — | (not run at G=128) |
| Zipformer-M G=16 MBR+PLL | 4.520 | −0.212 | <0.0001 | [−0.282, −0.143] |
| Zipformer-M G=128 MBR+PLL | 4.432 | −0.343 | <0.0001 | [−0.420, −0.269] |
| TL3 G=16 MBR+PLL (full 1155) | 10.966 | −0.334 | <0.0001 | [−0.496, −0.174] |
| TL3 G=128 MBR+PLL | 10.572 | −0.728 | <0.0001 | [−0.883, −0.581] |
| MUSAN 20dB G=16 | 6.479 | −0.138 | 0.0064 | [−0.245, −0.031] |
| MUSAN 10dB G=16 | 7.882 | −0.159 | 0.0030 | [−0.275, −0.046] |
| MUSAN 5dB G=16 | 10.922 | −0.182 | 0.0009 | [−0.294, −0.068] |
| MUSAN 0dB G=16 | 17.899 | +0.021 | 0.6458 | [−0.097, +0.144] |
| VoxPopuli G=128 | 18.326 | +0.038 | n/a | n/a (MBR hurts) |

**Note on dev-clean G=128:** This condition was run with the Zipformer-M model
(row 7 above), not with the base Zipformer-S at G=128 on dev-clean. The dev-clean
G=16 entry uses Zipformer-S.

## Extraction Notes

- All CIs extracted from existing result files; zero re-computation required.
- The `tl3_g16_full1155_results.json` CI uses the full 1155-utterance TL3
  test set (not the 700-utterance subset from the initial bootstrap run).
  The 700-utterance run gave p=0.0093; the full run gives p<0.0001.
- VoxPopuli has no bootstrap because the effect goes in the wrong direction.

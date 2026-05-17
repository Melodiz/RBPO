# Error Budget Comparison Across Methods

| Method | ΔSub | ΔIns | ΔDel | ΔTotal | %Sub | %Ins | %Del |
|--------|-----:|-----:|-----:|-------:|-----:|-----:|-----:|
| RoBERTa PLL interp α=0.7 (G=16) | -71 | -17 | +35 | -53 | 134% | 32% | -66% |
| MBR-CER + PLL τ=10 (G=16) | -120 | -24 | +26 | -118 | 102% | 20% | -22% |
| MBR-CER + PLL τ=10 (G=128) | -185 | -35 | -31 | -251 | 74% | 14% | 12% |

**%Sub/Ins/Del** = fraction of net error change from each type. Values >60% for substitutions confirm the linguistic disambiguation hypothesis.

# Dev-Other vs Test-Other Error Type Comparison

## MBR-CER + PLL τ=10, G=128

| Metric | Dev-Other (E10) | Test-Other (E13) |
|--------|----------------:|-----------------:|
| Utterances | 2864 | 2939 |
| Switched | — | 656 |
| Improve | — | 314 |
| Regress | — | 84 |
| %Sub (improving) | ~60-68% (E10) | 75.5% |
| %Ins (improving) | ~15-20% (E10) | 10.8% |
| %Del (improving) | ~15-20% (E10) | 13.7% |

## Interpretation

**Confirmed:** Substitution dominance generalizes to test-other. The information bottleneck is specifically a linguistic disambiguation bottleneck.

## Examples (Improvements)

### Example 1
- **Ref:** "o eternal everywhere o eternal nowhere o eternal in vain"
- **Greedy:** "ternal everywhereternal nowhereternal in vain" (3S 0I 5D)
- **Method:** "eternal everywhere eternal nowhere eternal in vain" (0S 0I 3D)
- Δ: -3S +0I -2D

### Example 2
- **Ref:** "and sheriff glispin's order not to shoot was the beginning of the protectorate that minnesota people established over us"
- **Greedy:** "in sheriff glispon's order not to shoot was the beginning of the protector of that minnesota people established over us" (3S 1I 0D)
- **Method:** "in sheriff glispin's order not to shoot was the beginning of the protectorate that minnesota people established over us" (1S 0I 0D)
- Δ: -2S -1I +0D

### Example 3
- **Ref:** "what's all that there dickie asked pointing to the odd knobbly bundles of all sorts and shapes tied on to the perambulator's front"
- **Greedy:** "what's on that there dickie asked pointing to the odd k noobbly bundles of all sorts and shapes tied on to the perm reator's front" (3S 2I 0D)
- **Method:** "what's on that there dickie asked pointing to the odd knobbly bundles of all sorts and shapes tied on to the permbulator's front" (2S 0I 0D)
- Δ: -1S -2I +0D

## Examples (Regressions)

### Regression 1
- **Ref:** "very good but we must first catch our house mother"
- **Greedy:** "very good but we must first catch our house mother" (0S 0I 0D)
- **Method:** "very good but we must first catch our housemother" (1S 0I 1D)
- Δ: +1S +0I +1D

### Regression 2
- **Ref:** "one pound of lobster meat one teaspoonful of butter one half pint of cream yolks of four eggs one wine glass of sherry lobster fat"
- **Greedy:** "one pound of lobster meat one teaspoonful of butter one half pint of cream yolks of four eggs one wine glass of sherry lobster fat" (0S 0I 0D)
- **Method:** "one pound of lobster meat one teaspoonful of butter one half pint of cream yolks of four eggs one wineglass of sherry lobster fat" (1S 0I 1D)
- Δ: +1S +0I +1D

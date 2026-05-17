# 5 Worst Regressions (Largest WER Increase)

## #1: `1686-142278-0068-805`
- **Failure mode:** greedy_was_perfect
- **Ref (1 words):** "no"
- **Greedy:** "no"
  - Errors: 0 (0S 0I 0D) → WER=0.0%
- **Method:** ""
  - Errors: 1 (0S 0I 1D) → WER=100.0%
- **Delta:** +1 errors (+100.0pp WER)
- **PLL:** greedy=-16.17, method=0.00 (gap=+16.17)
- **CTC:** greedy=-0.17, method=-4.60

## #2: `1255-138279-0008-354`
- **Failure mode:** greedy_was_perfect
- **Ref (2 words):** "two three"
- **Greedy:** "two three"
  - Errors: 0 (0S 0I 0D) → WER=0.0%
- **Method:** "two two three"
  - Errors: 1 (0S 1I 0D) → WER=50.0%
- **Delta:** +1 errors (+50.0pp WER)
- **PLL:** greedy=-18.72, method=-4.92 (gap=+13.81)
- **CTC:** greedy=-0.95, method=-7.00

## #3: `1255-90407-0006-321`
- **Failure mode:** greedy_was_perfect
- **Ref (4 words):** "he's my father indeed"
- **Greedy:** "he's my father indeed"
  - Errors: 0 (0S 0I 0D) → WER=0.0%
- **Method:** "he is my father indeed"
  - Errors: 2 (1S 1I 0D) → WER=50.0%
- **Delta:** +2 errors (+50.0pp WER)
- **PLL:** greedy=-26.18, method=-20.46 (gap=+5.72)
- **CTC:** greedy=-4.06, method=-9.06

## #4: `1686-142278-0087-824`
- **Failure mode:** other
- **Ref (4 words):** "margaret was almost stunned"
- **Greedy:** "margaret was all the stunned"
  - Errors: 2 (1S 1I 0D) → WER=50.0%
- **Method:** "margaret was all the stuned"
  - Errors: 3 (2S 1I 0D) → WER=75.0%
- **Delta:** +1 errors (+25.0pp WER)
- **PLL:** greedy=-59.92, method=-69.31 (gap=-9.39)
- **CTC:** greedy=-6.59, method=-11.54

## #5: `1651-136854-0013-1314`
- **Failure mode:** greedy_was_perfect
- **Ref (4 words):** "scholar's friendship like ladies"
- **Greedy:** "scholar's friendship like ladies"
  - Errors: 0 (0S 0I 0D) → WER=0.0%
- **Method:** "scholars friendship like ladies"
  - Errors: 1 (1S 0I 0D) → WER=25.0%
- **Delta:** +1 errors (+25.0pp WER)
- **PLL:** greedy=-50.16, method=-42.44 (gap=+7.73)
- **CTC:** greedy=-4.63, method=-8.46

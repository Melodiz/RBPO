# Best Improvement Examples

## RoBERTa PLL interp α=0.7 (G=16)

### Example 1

**Utterance:** `8254-84205-0051-1729`
- Reference: "they seem to me to be hatching up some dodge or another replied griggs"
- Greedy:    "it see seemed to me to be hatching up some dodg or another replied griggs"  (3S 1I 0D)
- Method:    "they seem to me to be hatching up some dodge or another replied griggs"  (0S 0I 0D)
- Delta: -3S -1I +0D = -4 total

### Example 2

**Utterance:** `1585-157660-0011-1400`
- Reference: "the polite jack replied all right sir take your word for it"
- Greedy:    "thelite jack replied all right sir take a word for it"  (2S 0I 1D)
- Method:    "the polite jack replied all right sir take your word for it"  (0S 0I 0D)
- Delta: -2S +0I -1D = -3 total

### Example 3

**Utterance:** `6455-67804-0020-1151`
- Reference: "she's prettier than ever to day and is enjoying herself"
- Greedy:    "she is prettier than ever to day in it enjoying herself"  (3S 1I 0D)
- Method:    "tis prettier than ever to day and is enjoying herself"  (1S 0I 0D)
- Delta: -2S -1I +0D = -3 total

### Example 4

**Utterance:** `8288-274162-0000-2359`
- Reference: "the captain sitting buried in his leathern armchair his spurs fixed in the floor his sword between his legs was reading a number of letters as he twisted his mustache"
- Greedy:    "the captain sitting buried in his leather arm chaair his spurs fixed in the floor his sword between his legs was reading a number of letters as he twisted his motache"  (3S 1I 0D)
- Method:    "the captain sitting buried in his leather armchair his spurs fixed in the floor his sword between his legs was reading a number of letters as he twisted his mustache"  (1S 0I 0D)
- Delta: -2S -1I +0D = -3 total

### Example 5

**Utterance:** `116-288046-0010-2597`
- Reference: "but first let us be impersonal the epithets irreverent blasphemer atheist and infidel are flung at a man not from pity but from envy"
- Greedy:    "but firstless be impersonal the epithet's irreverent blasphemer atheist and infidel are flung at a man not from pity but from envy"  (2S 0I 2D)
- Method:    "but first let be impersonal the epithets irreverent blasphemer atheist and infidel are flung at a man not from pity but from envy"  (0S 0I 1D)
- Delta: -2S +0I -1D = -3 total

## MBR-CER + PLL τ=10 (G=16)

### Example 1

**Utterance:** `5849-50963-0009-2082`
- Reference: "well i'm a rich man now and so is my mate mc laughlin for that wood was contracted for by the largest and richest piano firm in this country and now it is all but delivered to them and the money in our hands"
- Greedy:    "while i am a rich man now and so is my mate mac glaun for that wood was contracted for by the largest and richest pianofrm in this country and now it is all but delivered to them and the money in our hands"  (5S 1I 1D)
- Method:    "while i'm a rich man now and so is my mate macglaklan for that wood was contracted for by the largest and richest piano firm in this country and now it is all but delivered to them and the money in our hands"  (2S 0I 1D)
- Delta: -3S -1I +0D = -4 total

### Example 2

**Utterance:** `8254-84205-0051-1729`
- Reference: "they seem to me to be hatching up some dodge or another replied griggs"
- Greedy:    "it see seemed to me to be hatching up some dodg or another replied griggs"  (3S 1I 0D)
- Method:    "they seem to me to be hatching up some dodge or another replied griggs"  (0S 0I 0D)
- Delta: -3S -1I +0D = -4 total

### Example 3

**Utterance:** `1585-157660-0011-1400`
- Reference: "the polite jack replied all right sir take your word for it"
- Greedy:    "thelite jack replied all right sir take a word for it"  (2S 0I 1D)
- Method:    "the polite jack replied all right sir take your word for it"  (0S 0I 0D)
- Delta: -2S +0I -1D = -3 total

### Example 4

**Utterance:** `1630-73710-0002-1826`
- Reference: "it makes the heart ache but to picture such vicissitudes to the imagination"
- Greedy:    "it makes the heartache but to picture such vicissitude to the imagination"  (2S 0I 1D)
- Method:    "it makes the heart ache but to picture such vicissitudes to the imagination"  (0S 0I 0D)
- Delta: -2S +0I -1D = -3 total

### Example 5

**Utterance:** `6455-67804-0020-1151`
- Reference: "she's prettier than ever to day and is enjoying herself"
- Greedy:    "she is prettier than ever to day in it enjoying herself"  (3S 1I 0D)
- Method:    "tis prettier than ever to day and is enjoying herself"  (1S 0I 0D)
- Delta: -2S -1I +0D = -3 total

## MBR-CER + PLL τ=10 (G=128)

### Example 1

**Utterance:** `6455-67804-0020-1151`
- Reference: "she's prettier than ever to day and is enjoying herself"
- Greedy:    "she is prettier than ever to day in it enjoying herself"  (3S 1I 0D)
- Method:    "she's prettier than ever to day and is enjoying herself"  (0S 0I 0D)
- Delta: -3S -1I +0D = -4 total

### Example 2

**Utterance:** `2506-11278-0005-1234`
- Reference: "it was her snobbish sentiment that misled her and made her vanities a prey to the swindling fortune teller"
- Greedy:    "it was her snobbish sentiment that misled her and made her vanities a prey did this swindling fortune tell her"  (3S 1I 0D)
- Method:    "it was her snobbish sentiment that misled her and made her vanities a prey to this swindling fortune teller"  (1S 0I 0D)
- Delta: -2S -1I +0D = -3 total

### Example 3

**Utterance:** `1630-73710-0002-1826`
- Reference: "it makes the heart ache but to picture such vicissitudes to the imagination"
- Greedy:    "it makes the heartache but to picture such vicissitude to the imagination"  (2S 0I 1D)
- Method:    "it makes the heart ache but to picture such vicissitudes to the imagination"  (0S 0I 0D)
- Delta: -2S +0I -1D = -3 total

### Example 4

**Utterance:** `5849-50963-0009-2082`
- Reference: "well i'm a rich man now and so is my mate mc laughlin for that wood was contracted for by the largest and richest piano firm in this country and now it is all but delivered to them and the money in our hands"
- Greedy:    "while i am a rich man now and so is my mate mac glaun for that wood was contracted for by the largest and richest pianofrm in this country and now it is all but delivered to them and the money in our hands"  (5S 1I 1D)
- Method:    "well i am a rich man now and so is my mate maclauchlan for that wood was contracted for by the largest and richest piano firm in this country and now it is all but delivered to them and the money in our hands"  (2S 1I 1D)
- Delta: -3S +0I +0D = -3 total

### Example 5

**Utterance:** `7697-105815-0034-2318`
- Reference: "with what a light step did he now climb the hill"
- Greedy:    "with what a li step did he now clamb the hiel"  (3S 0I 0D)
- Method:    "with what a light step did he now climb the hill"  (0S 0I 0D)
- Delta: -3S +0I +0D = -3 total

---

# Regression Examples

## RoBERTa PLL interp α=0.7 (G=16)

### Regression 1

**Utterance:** `6455-67803-0025-1217`
- Reference: "i am on my way to my to lady theobald"
- Greedy:    "i am on my way to my to lady theobald"  (0S 0I 0D)
- Method:    "i'm on my way to my lady theobald"  (1S 0I 2D)
- Delta: +1S +0I +2D = +3 total

### Regression 2

**Utterance:** `8254-84205-0057-1735`
- Reference: "no i'm going to put it in force at once we start to night"
- Greedy:    "no i'm going to put it in force at once we start to night"  (0S 0I 0D)
- Method:    "no i'm going to put it in force out once we start ton"  (2S 0I 1D)
- Delta: +2S +0I +1D = +3 total

### Regression 3

**Utterance:** `4831-25894-0012-31`
- Reference: "see now have no fear give them bella monica that is merry and will make the laugh whispered tommo tuning his harp"
- Greedy:    "see now have no fear give them bellamonica that is marian and will make the laugh whispered tomma tuning his harp"  (3S 0I 1D)
- Method:    "say now have no fear give them bellamonica that is marian and will make them laugh whispered tomma tuning his harp"  (5S 0I 1D)
- Delta: +2S +0I +0D = +2 total

## MBR-CER + PLL τ=10 (G=16)

### Regression 1

**Utterance:** `6455-67803-0025-1217`
- Reference: "i am on my way to my to lady theobald"
- Greedy:    "i am on my way to my to lady theobald"  (0S 0I 0D)
- Method:    "i'm on my way to my lady theobald"  (1S 0I 2D)
- Delta: +1S +0I +2D = +3 total

### Regression 2

**Utterance:** `6123-59186-0040-174`
- Reference: "nothing of tiniest worth have i wrought pondered planned no one thing asking blame or praise since the pale corpse like birth of this diurnal unit bearing blanks in all its rays dullest of dull hued days"
- Greedy:    "nothing of tiniest worth have i wrought pondered planned no one thing asking blame or praise since the pale corpse like birth of this diurnal unit bearing blanks in all its rays dullest of dull hued days"  (0S 0I 0D)
- Method:    "nothing of tiniest worth have i wrought pondered planned no one thing asking blame or praise since the pale corpselike birth of this diurnal unit bearing blanks in all its rays dullest of dull hued days"  (1S 0I 1D)
- Delta: +1S +0I +1D = +2 total

### Regression 3

**Utterance:** `3660-172182-0007-604`
- Reference: "lady said he at the gate there is a knight and i saw never a man of so pitiful an aspect to look upon as he"
- Greedy:    "lady said he at the gate there is a knight and i saw never a man of so pitiful an aspect to look upon as he"  (0S 0I 0D)
- Method:    "lady said at the gate there was a knight and i saw never a man of so pitiful an aspect to look upon as he"  (1S 0I 1D)
- Delta: +1S +0I +1D = +2 total

## MBR-CER + PLL τ=10 (G=128)

### Regression 1

**Utterance:** `8254-84205-0004-1682`
- Reference: "every one was busy for the keeping watch regularly took up a good deal of time"
- Greedy:    "every one was busy for the keeping watch regularly took up a good deal of time"  (0S 0I 0D)
- Method:    "everyone was busy for the keeping watch regularly took up a good deal of time"  (1S 0I 1D)
- Delta: +1S +0I +1D = +2 total

### Regression 2

**Utterance:** `6467-94831-0000-2144`
- Reference: "peleg snuggers the general utility man of the hall had just brought the boys up from cedarville to which place they had journeyed from ithaca on the regular afternoon boat running up cayuga lake"
- Greedy:    "peleck snuckers the generalleting man of the hall had just brought the boys up from cedarville to which place they had journeyed from etaka on the regular afternoon boat running up caugalech"  (5S 0I 2D)
- Method:    "pele snuckers the general lelleting men of the hall had just brought the boys up from cedarville to his place they had journeyed from eta on the regular afternoon boat running upeugie"  (7S 0I 2D)
- Delta: +2S +0I +0D = +2 total

### Regression 3

**Utterance:** `6467-62797-0002-2192`
- Reference: "when the umbilical cord of a kondh baby sloughs off a spider is burnt in the fire and its ashes are placed in a cocoanut shell mixed with castor oil and applied by means of a fowl's feather to the navel"
- Greedy:    "when the umbilical cord of a coned baby lost off a spider is burnt in the fire and its ashes are placed in a cocoanut shell mixed with castra oil and applied by means of a full's feather to the navel"  (4S 0I 0D)
- Method:    "when the umbilical cord of ac con babyse lost off a spider is burned in the fire its ashes are placed in a cocoanut shell mixed with castor oil and applied by means of a fowl's feather to the navel"  (5S 0I 1D)
- Delta: +1S +0I +1D = +2 total

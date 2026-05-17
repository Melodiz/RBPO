# Spearman ρ Bootstrap Analysis

Per-utterance Spearman ρ(score, WER) with 95% bootstrap CIs.

## Corpus-Level

| Scorer | ρ | 95% CI | N |
|--------|---:|--------|---:|
| CTC log-prob | -0.3474 | [-0.3556, -0.3389] | 2858 |
| RoBERTa PLL | -0.4844 | [-0.4935, -0.4754] | 2858 |
| Interpolated (α=0.6 CTC + 0.4 PLL) | -0.5270 | [-0.5351, -0.5189] | 2858 |
| GPT-2 LL | -0.4005 | [-0.4102, -0.3909] | 2858 |

## By Utterance Length (Terciles)

| Stratum | Scorer | ρ | 95% CI | N |
|---------|--------|---:|--------|---:|
| short | CTC log-prob | -0.3845 | [-0.3983, -0.3704] | 1006 |
| short | RoBERTa PLL | -0.4253 | [-0.4416, -0.4095] | 1006 |
| short | Interpolated (α=0.6 CTC + 0.4 PLL) | -0.4852 | [-0.4995, -0.4707] | 1006 |
| short | GPT-2 LL | -0.3243 | [-0.3405, -0.3079] | 1006 |
| medium | CTC log-prob | -0.3376 | [-0.3515, -0.3236] | 961 |
| medium | RoBERTa PLL | -0.4748 | [-0.4895, -0.4596] | 961 |
| medium | Interpolated (α=0.6 CTC + 0.4 PLL) | -0.5142 | [-0.5277, -0.5000] | 961 |
| medium | GPT-2 LL | -0.3963 | [-0.4125, -0.3799] | 961 |
| long | CTC log-prob | -0.3159 | [-0.3308, -0.3014] | 891 |
| long | RoBERTa PLL | -0.5614 | [-0.5753, -0.5472] | 891 |
| long | Interpolated (α=0.6 CTC + 0.4 PLL) | -0.5879 | [-0.6011, -0.5747] | 891 |
| long | GPT-2 LL | -0.4912 | [-0.5063, -0.4757] | 891 |

## By Error Regime

| Regime | Scorer | ρ | 95% CI | N |
|--------|--------|---:|--------|---:|
| greedy_optimal | CTC log-prob | -0.3795 | [-0.3882, -0.3707] | 2193 |
| greedy_optimal | RoBERTa PLL | -0.4866 | [-0.4967, -0.4763] | 2193 |
| greedy_optimal | Interpolated (α=0.6 CTC + 0.4 PLL) | -0.5356 | [-0.5447, -0.5266] | 2193 |
| greedy_optimal | GPT-2 LL | -0.4008 | [-0.4118, -0.3898] | 2193 |
| recoverable | CTC log-prob | -0.2415 | [-0.2603, -0.2228] | 665 |
| recoverable | RoBERTa PLL | -0.4772 | [-0.4974, -0.4567] | 665 |
| recoverable | Interpolated (α=0.6 CTC + 0.4 PLL) | -0.4984 | [-0.5176, -0.4790] | 665 |
| recoverable | GPT-2 LL | -0.3995 | [-0.4201, -0.3783] | 665 |

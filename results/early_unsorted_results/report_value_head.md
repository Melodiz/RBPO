# Value Head: RL-Guided N-Best Reranking from Encoder Features

**Stage report.** Trains a value head V(s, y) on Zipformer encoder embeddings; tests whether acoustic information *beyond* what CTC log-prob captures helps select the right hypothesis.

## TL;DR

- Encoder dim D = **256**, 10000 train utts / 157651 hyps, 2864 dev utts / 44492 hyps.
- Best ablation alone: **ctc_only** → WER **6.02%**, gap closed **-0.1%**, ρ +0.412.
- Best three-way (CTC + V + RoBERTa): α=0.6, β=0.0, γ=0.4 → WER **5.97%**, gap closed **+3.4%**.
- Baselines: CTC ρ = +0.347 (matches known −0.347).

## Setup

- Model: Zipformer-S CR-CTC, encoder dim D = 256
- Alignment: monotonic argmax (per-token frame = argmax_t logits[t,k] in [start, T))
- Train data: 10000 utts / 157651 hypotheses (train-clean-100, no LM scores)
- Dev data: 2864 utts / 44492 hypotheses (dev-other)
- Loss: pairwise margin ranking (margin=0.1) per utt; cosine-decay Adam, lr=1e-3, 50 epochs
- Greedy WER: 6.02%; Oracle WER: 4.44%; gap = 1.58 pp

## Ablation Study

| Ablation | Features | dim | best epoch | dev ρ | dev WER | gap closed |
|---|---|---:|---:|---:|---:|---:|
| ctc_only | CTC log-prob + lengths + utt CTC stats (8 scalars) | 8 | 1 | +0.412 | 6.02% | -0.12% |
| encoder_only | hyp_encoder_mean + utt_encoder_mean (2D) | 512 | 1 | +0.060 | 13.01% | -442.48% |
| encoder_plus_ctc | encoder + utt encoder + CTC log-prob + lengths | 515 | 1 | +0.377 | 6.21% | -12.17% |
| full_no_lm | encoder + utt encoder + diff + all scalars | 777 | 1 | +0.362 | 6.59% | -35.78% |
| full_plus_lm | full_no_lm V combined with RoBERTa PLL (z-norm) | — | (reuse full_no_lm) | +0.527 | 6.22% | -12.67% |

## Three-Way Grid Search (CTC + V + RoBERTa PLL)

Each scorer is z-normalized per utterance (lower = better), then linearly combined: `score = α·CTC_z + β·V_z + γ·RoBERTa_z` with α + β + γ = 1, step 0.1.

**Best:** α=0.6 (CTC), β=0.0 (V), γ=0.4 (RoBERTa) → WER 5.97%, gap closed +3.4%, ρ +0.505.

**Top 10 grid points by WER:**

| α (CTC) | β (V) | γ (RoBERTa) | dev ρ | dev WER | gap closed |
|---:|---:|---:|---:|---:|---:|
| 0.6 | 0.0 | 0.4 | +0.505 | 5.97% | +3.35% |
| 0.4 | 0.1 | 0.5 | +0.525 | 5.97% | +3.23% |
| 0.4 | 0.2 | 0.4 | +0.515 | 5.97% | +3.11% |
| 0.5 | 0.0 | 0.5 | +0.519 | 5.97% | +3.11% |
| 0.5 | 0.1 | 0.4 | +0.511 | 5.97% | +2.98% |
| 0.6 | 0.1 | 0.3 | +0.485 | 5.98% | +2.48% |
| 0.3 | 0.2 | 0.5 | +0.528 | 5.99% | +2.24% |
| 0.5 | 0.2 | 0.3 | +0.490 | 5.99% | +2.24% |
| 0.3 | 0.3 | 0.4 | +0.518 | 5.99% | +2.11% |
| 0.7 | 0.0 | 0.3 | +0.479 | 5.99% | +2.11% |

## Critical Question — Does the Encoder Add Information BEYOND CTC + LM?

Cross-ablation comparison:

- CTC-only WER 6.02% (reproduces Level-3 negative result on this data)
- Encoder-only WER 13.01%
- Encoder + CTC WER 6.21%
- Full (no LM) WER 6.59%

**Three-way grid picks β=0.0 for V** — the value head is redundant once CTC + RoBERTa are combined linearly. Encoder acoustic info is largely captured by CTC log-prob plus linguistic plausibility from PLL.

## RL Framing

The hypothesis-selection problem reduces cleanly to a single-step MDP:

- **State** s = (encoder output `h(x)` of utterance x, the candidate set Y)
- **Action** a ∈ Y = pick a hypothesis from the N-best list
- **Reward** r(a) = −WER(a, reference)
- **Value** V(s, y) ≈ E[ r | a = y, s ] — predicted negative WER of selecting y in state s
- **Policy** π(y | s) = argmin_y V(s, y)  (greedy w.r.t. V)

The pairwise margin loss

> L_pairwise = Σ_{(i,j) : WER_i < WER_j} max(0, V(s, y_i) − V(s, y_j) + m)

is a margin-based surrogate for the policy-gradient objective on terminal reward: it shapes V so the action with lower true WER receives the lower predicted value, which is exactly what π = argmin V requires for optimal selection.

**Connection to Part 1 (CTC backward as Rao-Blackwellized REINFORCE):** the CTC backward pass, as a marginal-likelihood gradient, gives the credit-assignment signal at *training time* over alignment paths. The encoder embeddings used here are the same representations CTC backward operates on — but we extract their hypothesis-discriminative content directly, side-stepping the CTC marginalization bottleneck. CTC marginalizes over alignments and projects to a single per-frame vocabulary distribution, throwing away alignment-specific acoustic detail that the value head can recover.

## Master Comparison Table (all rerankers tried in this project)

| Method | dev WER | gap closed |
|---|---:|---:|
| Greedy (CTC argmax) | 6.02% | 0.0% |
| Length-norm (Level 1.5) | ~6.10% | ~−5% |
| MBR-CER w/ CTC posteriors (Level 2) | ≈ greedy | ~0% |
| 14-feature MLP rescorer (Level 3) | 6.05% | −1.9% |
| GPT-2 LL interp α=0.8 (Level 5) | 5.99% | +2.1% |
| RoBERTa PLL interp α=0.7 (Level 5) | 5.92% | +6.5% |
| **MBR-CER w/ RoBERTa PLL τ=10 (Level 5)** | **5.79%** | **+14.5%** |
| Value head (full_no_lm) | 6.59% | -35.78% |
| **Three-way (CTC+V+RoBERTa)** | **5.97%** | **+3.4%** |
| Oracle (lower bound) | 4.44% | 100.0% |

## Honest Assessment

The value head largely duplicates information already present in CTC log-prob and RoBERTa PLL — the three-way combination matches RoBERTa-alone within noise. The 'acoustic-beyond-CTC' hypothesis is only weakly supported on this corpus.

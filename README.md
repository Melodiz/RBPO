# RBPO

Reward-based preference optimization for CTC-based ASR N-best reranking, built on icefall/k2.

## Setup

Python >= 3.10. Key deps: torch, k2, lhotse, icefall, sentencepiece, editdistance, jiwer.

    pip install -e . && pip install -r rbpo/requirements.txt

icefall is not pip-installable; clone it and add to `PYTHONPATH`.

## Repo Structure

- `scripts/` -- pipeline scripts: N-best generation, PLL scoring, MBR reranking, interpolation
- `experiments/` -- per-experiment drivers, organized by stage:
  - `training/` -- MWER, RAFT, reranker training, N-best generation, value head training
  - `decoding/` -- HLG, MBR, shallow fusion, neural LM scoring, beam sweep, contrastive decoding, MC-dropout, temperature sweep
  - `evaluation/` -- dev/test set evaluation scripts
  - `analysis/` -- gamma analysis, gradient variance (flat/RB/Viterbi), oracle WER, level-2 decomposition, feature extraction, plotting
  - `robustness/` -- cross-domain verification
  - `data_prep/` -- corpus preparation (TED-LIUM 3, train-clean-100, train data helpers)
- `rbpo/` -- installable package with shared utilities, training logic, and tests
- `results/` -- persisted outputs (CSV, JSON, reports); large JSONLs live on Google Drive
- `report/` -- thesis LaTeX sources and figures
- `notebooks/` -- Colab session notes and troubleshooting

## Usage

    python scripts/generate_nbest.py \
        --cuts cuts.jsonl.gz --checkpoint pretrained.pt --bpe bpe.model \
        --G 16 --output nbest.jsonl

    python scripts/score_pll.py \
        --nbest nbest.jsonl --output nbest_pll.jsonl --model roberta-base

    python scripts/rerank_mbr.py \
        --nbest nbest_pll.jsonl --output mbr_results.json \
        --utility cer --tau 10.0 --pll-weight 0.5

## Results

Zipformer-S CR-CTC (BPE-500, 22.1M params) on LibriSpeech, G=16.
See `experiments/` scripts for reproduction.

| Method                        | dev-other | test-other |
|-------------------------------|-----------|------------|
| Greedy                        |   6.02    |    5.96    |
| RoBERTa PLL interp (a=0.7)   |   5.92    |    5.85    |
| MBR-CER + RoBERTa PLL (t=10) |   5.79    |    5.77    |
| Oracle                        |   4.44    |    4.41    |

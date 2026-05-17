import editdistance


def compute_wer(hypothesis: str, reference: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    return editdistance.eval(hyp_words, ref_words) / len(ref_words)


def compute_wer_from_tokens(hyp_tokens: list[int], ref_tokens: list[int]) -> float:
    if len(ref_tokens) == 0:
        return 0.0 if len(hyp_tokens) == 0 else 1.0

    return editdistance.eval(hyp_tokens, ref_tokens) / len(ref_tokens)


def compute_batch_wer(hypotheses: list[str], references: list[str]) -> list[float]:
    assert len(hypotheses) == len(references)
    return [compute_wer(h, r) for h, r in zip(hypotheses, references)]

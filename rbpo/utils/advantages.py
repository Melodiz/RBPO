import torch


def group_relative_advantages(rewards: list[float] | torch.Tensor) -> torch.Tensor:
    if not isinstance(rewards, torch.Tensor):
        rewards = torch.tensor(rewards, dtype=torch.float32)
    return rewards - rewards.mean()


def group_relative_advantages_per_utterance(
    rewards: torch.Tensor, num_per_utt: list[int]
) -> torch.Tensor:
    advantages = torch.empty_like(rewards)
    offset = 0
    for n in num_per_utt:
        group = rewards[offset : offset + n]
        advantages[offset : offset + n] = group - group.mean()
        offset += n
    return advantages

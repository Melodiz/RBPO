import torch


def clip_surrogate(
    rho: torch.Tensor,
    advantage: torch.Tensor,
    eps_low: float = 0.2,
    eps_high: float = 0.28,
) -> torch.Tensor:
    clipped_rho = torch.clamp(rho, 1.0 - eps_low, 1.0 + eps_high)
    return torch.min(rho * advantage, clipped_rho * advantage)


def length_normalize_ratio(
    log_rho: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    return torch.exp(log_rho / lengths)

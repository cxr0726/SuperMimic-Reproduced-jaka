"""Return targets for reset sampling, separate from actor/critic learning targets."""

from __future__ import annotations

import torch


@torch.no_grad()
def mean_reward_continuation(discounted_return, reward_sum, steps, terminated, gamma):
    """Extrapolate mean reward after a normal cutoff; physical failures get no tail."""
    if not 0 < gamma < 1:
        raise ValueError("Sampler discount must lie strictly between zero and one")
    if (steps <= 0).any():
        raise ValueError("A sampler target needs at least one real transition")
    tail = gamma ** steps * (reward_sum / steps) / (1 - gamma)
    return discounted_return + torch.where(terminated, 0.0, tail)


@torch.no_grad()
def update_bin_returns(bin_return, bins, returns, alpha):
    """Average duplicate start-bin targets before one EMA update per touched bin."""
    if not 0 < alpha <= 1:
        raise ValueError("Sampler EMA new-data weight must lie in (0, 1]")
    if bins.numel() == 0:
        return
    unique, inverse = bins.unique(return_inverse=True)
    targets = returns.to(bin_return.dtype)
    sums = torch.zeros_like(unique, dtype=targets.dtype).scatter_add_(0, inverse, targets)
    counts = torch.bincount(inverse, minlength=unique.numel())
    bin_return[unique] = torch.lerp(bin_return[unique], sums / counts, alpha)

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


class ResetEpisodeDiagnostics:
    """GPU sums for completed episodes; never changes rewards or sampler targets.

    The first *real* control transition is kept separate from all subsequent
    transitions. This is a boundary diagnostic, not a claim that all first-step
    reward loss is an artificial reset cost. Sums are consumed only at logging
    intervals, and can be reduced across ranks before converting to CPU values.
    """

    FIELDS = (
        "episodes", "post_first_episodes", "steps", "first_reward", "first_action_rate",
        "post_first_reward_mean", "post_first_action_rate_mean",
        "post_first_reward_sum", "post_first_steps",
        "first_target_contribution", "first_action_rate_target_contribution",
        "post_first_target_contribution", "sampler_target",
    )

    def __init__(self, num_envs: int, device, gamma: float):
        self.gamma = gamma
        self.first_reward = torch.zeros(num_envs, device=device, dtype=torch.float64)
        self.first_action_rate = torch.zeros_like(self.first_reward)
        self.action_rate_sum = torch.zeros_like(self.first_reward)
        self.totals = torch.zeros(len(self.FIELDS), device=device, dtype=torch.float64)

    @torch.no_grad()
    def record(self, ids, reward, action_rate_reward, episode_steps):
        first = episode_steps == 0
        # Keep the existing ids shape: boolean compression would add dynamic
        # CUDA indexing/synchronization solely for these optional diagnostics.
        self.first_reward[ids] = torch.where(first, reward.double(), self.first_reward[ids])
        self.first_action_rate[ids] = torch.where(
            first, action_rate_reward.double(), self.first_action_rate[ids],
        )
        self.action_rate_sum[ids] += action_rate_reward.double()

    @torch.no_grad()
    def finalize(self, ids, reward_sum, steps, terminated, targets):
        first, first_rate = self.first_reward[ids], self.first_action_rate[ids]
        has_post_first = steps > 1
        post_steps = (steps - 1).clamp_min(1)
        post_reward = reward_sum - first
        post_rate = self.action_rate_sum[ids] - first_rate
        # Linear decomposition of the *unchanged* sampler target. At a normal
        # cutoff, first reward is also included in the extrapolated mean tail.
        tail_multiplier = torch.where(
            terminated, 0.0, self.gamma ** steps / (steps * (1 - self.gamma)),
        )
        multiplier = 1 + tail_multiplier
        first_target = first * multiplier
        first_rate_target = first_rate * multiplier
        values = (
            torch.ones_like(first), has_post_first.double(), steps.double(), first, first_rate,
            torch.where(has_post_first, post_reward / post_steps, 0.0),
            torch.where(has_post_first, post_rate / post_steps, 0.0),
            post_reward, (steps - 1).double(), first_target, first_rate_target,
            targets - first_target, targets,
        )
        self.totals += torch.stack(values).sum(dim=1)
        self.first_reward[ids] = 0
        self.first_action_rate[ids] = 0
        self.action_rate_sum[ids] = 0

    @torch.no_grad()
    def consume(self):
        totals = self.totals.clone()
        self.totals.zero_()
        return totals


def reset_diagnostic_metrics(totals: torch.Tensor, max_step_reward: float) -> dict[str, float]:
    """CPU conversion only at a logging interval, after an optional rank sum.

    Most means are episode-balanced, like train/episode_norm_return. The
    explicitly named step-weighted mean has a different denominator. One-step
    episodes contribute to first-step metrics, never to post-first means.
    """
    sums = dict(zip(ResetEpisodeDiagnostics.FIELDS, totals.tolist(), strict=True))
    count, post_count = sums["episodes"], sums["post_first_episodes"]
    result = {
        "reset/completed_episodes": count,
        "tracking/post_first_step_episodes": post_count,
    }
    if count:
        result.update({
            "reset/episode_steps": sums["steps"] / count,
            "reset/first_step_reward": sums["first_reward"] / count,
            "reset/first_step_action_rate_reward": sums["first_action_rate"] / count,
            "reset/first_step_target_contribution": sums["first_target_contribution"] / count,
            "reset/first_step_action_rate_target_contribution": sums["first_action_rate_target_contribution"] / count,
            "tracking/post_first_step_target_contribution": sums["post_first_target_contribution"] / count,
            "sampler/completed_episode_target": sums["sampler_target"] / count,
        })
    if post_count:
        post_mean = sums["post_first_reward_mean"] / post_count
        result.update({
            "tracking/post_first_step_reward": post_mean,
            "tracking/post_first_step_norm_return": post_mean / max_step_reward,
            "tracking/post_first_step_action_rate_reward": sums["post_first_action_rate_mean"] / post_count,
            "tracking/post_first_step_norm_return_step_weighted": (
                sums["post_first_reward_sum"] / (max_step_reward * sums["post_first_steps"])
            ),
        })
    return result

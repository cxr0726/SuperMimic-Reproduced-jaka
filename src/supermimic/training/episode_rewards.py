"""Episode-balanced reward metrics for the physical training environment."""

import math

import torch


class PhysicalEpisodeRewards:
    def __init__(self, num_envs: int, term_names: list[str], reward_cap: float, device: str):
        self.term_names = tuple(term_names)
        self.reward_cap = reward_cap
        self.steps = torch.zeros(num_envs, device=device)
        self.reward_sum = torch.zeros_like(self.steps)
        shape = (num_envs, len(self.term_names))
        self.first_terms = torch.zeros(shape, device=device)
        self.term_sum = torch.zeros(shape, dtype=torch.float64, device=device)
        # Overall return, length, completed count, and count with post-first steps.
        # The remaining rows hold the three vectors of normalized term sums.
        self.totals = torch.zeros(4 + 3 * len(self.term_names), dtype=torch.float64, device=device)
        self.term_totals = self.totals[4:].view(3, len(self.term_names))

    @torch.no_grad()
    def record(self, reward: torch.Tensor, terms: torch.Tensor, done: torch.Tensor) -> None:
        self.first_terms.copy_(torch.where(self.steps[:, None] == 0, terms, self.first_terms))
        self.term_sum.add_(terms)
        self.reward_sum.add_(reward)
        self.steps.add_(1)

        ids = done.nonzero().flatten()
        if ids.numel() == 0:
            return
        lengths = self.steps[ids]
        full_scale = lengths * self.reward_cap
        post_scale = (lengths - 1).clamp_min(1) * self.reward_cap
        first = self.first_terms[ids].double()
        full = self.term_sum[ids]
        self.totals[0] += (self.reward_sum[ids] / full_scale).sum(dtype=torch.float64)
        self.totals[1] += lengths.sum(dtype=torch.float64)
        self.totals[2] += ids.numel()
        self.totals[3] += (lengths > 1).sum()
        self.term_totals[0] += (full / full_scale[:, None]).sum(dim=0)
        self.term_totals[1] += first.sum(dim=0) / self.reward_cap
        # One-step episodes have an exactly zero numerator here. The logging
        # denominator excludes them entirely, including across distributed ranks.
        self.term_totals[2] += ((full - first) / post_scale[:, None]).sum(dim=0)

        self.steps[ids] = 0
        self.reward_sum[ids] = 0
        self.term_sum[ids] = 0

    def consume(self) -> torch.Tensor:
        """Return completed-episode totals for an optional rank sum, keeping live episodes."""
        totals = self.totals.clone()
        self.totals.zero_()
        return totals

    def metrics(self, totals: torch.Tensor) -> tuple[int, dict[str, float]]:
        total, steps, count, post_count = totals[:4].tolist()
        if count == 0:
            return 0, {}
        avg_return = total / count
        metrics = {
            "train/episode_norm_return": avg_return,
            "train/episode_len": steps / count,
            "train/reward_terms/post_first_episodes": post_count,
        }
        whole, first, post = totals[4:].view(3, len(self.term_names)).tolist()
        metrics["train/reward_terms/episode_sum_error"] = math.fsum(whole) / count - avg_return
        for phase, terms, denominator in (
            ("episode", whole, count),
            ("first_step", first, count),
            ("post_first_step", post, post_count),
        ):
            if denominator:
                for name, value in zip(self.term_names, terms, strict=True):
                    metrics[f"train/reward_terms/{phase}/{name}"] = value / denominator
        return int(count), metrics

"""Return targets for reset sampling, separate from actor/critic learning targets."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class WindowValueSamplerCfg:
    """Official ReturnSampler statistics (HybridRobotics/supermimic, Apache-2.0).

    Statistical semantics follow upstream commit 5efbf114. Motion-bin geometry
    and reset eligibility remain the responsibility of this framework.
    """

    gamma: float = 0.99
    horizon: int = 20
    memory: int = 256
    steepness: float = 25.0
    explore: float = 0.5
    keep: float = 0.1


class WindowValueSampler:
    """n-step table-value RSI sampler updated on fixed physical-step windows.

    One sample is the mean return of an env/episode/bin within this window.
    Long unfinished episodes therefore contribute every ``horizon`` steps.
    Only real physical transitions belong in ``record``; imaginary rollouts and
    reset callbacks must never update this state.
    """

    _TENSORS = (
        "value",
        "count",
        "effort",
        "probabilities",
        "_frames",
        "_rewards",
        "_terminated",
        "_clip_end",
        "_timed_out",
    )

    def __init__(
        self,
        cfg: WindowValueSamplerCfg,
        bin_count: int,
        frame_count: int,
        num_envs: int,
        max_step_reward: float,
        device="cpu",
        *,
        bin_of_frame: Callable[[torch.Tensor], torch.Tensor] | None = None,
        sampleable_bins: torch.Tensor | None = None,
        last_frame_of: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        if not 0 < cfg.gamma < 1:
            raise ValueError("Window sampler gamma must lie in (0, 1)")
        for name in ("horizon", "memory"):
            value = getattr(cfg, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Window sampler {name} must be a positive integer")
        if not all(
            math.isfinite(x) for x in (cfg.steepness, cfg.explore, cfg.keep, max_step_reward)
        ):
            raise ValueError("Window sampler parameters must be finite")
        if cfg.steepness < 0 or cfg.explore < 0 or cfg.keep < 0 or cfg.explore + cfg.keep >= 1:
            raise ValueError("Window sampler mixture must leave a positive focus share")
        if min(bin_count, frame_count, num_envs) < 1 or max_step_reward <= 0:
            raise ValueError("Window sampler dimensions and reward ceiling must be positive")
        self.cfg, self.bin_count, self.frame_count = cfg, bin_count, frame_count
        self.num_envs = num_envs
        self.v_max = max_step_reward / (1 - cfg.gamma)
        self.bin_of_frame = bin_of_frame
        if last_frame_of is not None and not callable(last_frame_of):
            raise ValueError("Window sampler last_frame_of must be callable")
        self.last_frame_of = last_frame_of
        self.bootstrap_next_frame = (
            "clip_clamped" if last_frame_of is not None else "global_clamped"
        )
        if sampleable_bins is None:
            sampleable_bins = torch.ones(bin_count, dtype=torch.bool, device=device)
        if sampleable_bins.shape != (bin_count,) or sampleable_bins.dtype != torch.bool:
            raise ValueError("Window sampler eligibility must be a bin-sized boolean vector")
        self.sampleable_bins = sampleable_bins.to(device).clone()
        if not bool(self.sampleable_bins.any()):
            raise ValueError("Window sampler needs at least one eligible reset bin")
        self.num_sampleable_bins = int(self.sampleable_bins.sum())
        self.value = torch.full((bin_count,), self.v_max, device=device)
        self.count = torch.zeros_like(self.value)
        self.effort = torch.zeros(bin_count, dtype=torch.float64, device=device)
        self.probabilities = self.sampleable_bins.float() / self.num_sampleable_bins
        shape = (num_envs, cfg.horizon)
        self._frames = torch.zeros(shape, dtype=torch.long, device=device)
        self._rewards = torch.zeros(shape, device=device)
        self._terminated = torch.zeros(shape, dtype=torch.bool, device=device)
        self._clip_end = torch.zeros_like(self._terminated)
        self._timed_out = torch.zeros_like(self._terminated)
        self._step = 0
        self.num_updates = 0

    def _bins(self, frames):
        if self.bin_of_frame is None:
            return frames * self.bin_count // self.frame_count
        return self.bin_of_frame(frames)

    @torch.no_grad()
    def record(self, frames, rewards, terminated, clip_end, timed_out):
        """Record pre-step frames and their transition outcomes, exactly once."""
        for name, value in (
            ("frames", frames),
            ("rewards", rewards),
            ("terminated", terminated),
            ("clip_end", clip_end),
            ("timed_out", timed_out),
        ):
            if value.shape != (self.num_envs,):
                raise ValueError(f"Window sampler {name} must have shape ({self.num_envs},)")
        col = self._step
        self._frames[:, col] = frames
        self._rewards[:, col] = rewards
        self._terminated[:, col] = terminated
        self._clip_end[:, col] = clip_end
        self._timed_out[:, col] = timed_out
        self._step += 1
        if self._step < self.cfg.horizon:
            return None
        self._step = 0
        self._update()
        self.probabilities.copy_(self._distribution())
        self.num_updates += 1
        return self.probabilities

    def _dones(self):
        return self._terminated | self._clip_end | self._timed_out

    def _returns(self):
        """Official lambda=1 backward recursion with three boundary tail rules."""
        next_frames = (self._frames + 1).clamp(max=self.frame_count - 1)
        if self.last_frame_of is not None:
            # Use the recorded pre-step frame's source clip. Auto-reset may
            # already have installed an unrelated clip in the live command.
            # This only fixes the successor value lookup, not physical done
            # timing, episode grouping, or the existing terminal/tail rules.
            next_frames = torch.minimum(next_frames, self.last_frame_of(self._frames))
        next_value = torch.where(
            self._terminated,
            0.0,
            torch.where(self._clip_end, self.value.mean(), self.value[self._bins(next_frames)]),
        )
        dones = self._dones()
        returns = torch.empty_like(self._rewards)
        following = next_value[:, -1]
        for t in reversed(range(self.cfg.horizon)):
            following = torch.where(dones[:, t], next_value[:, t], following)
            returns[:, t] = self._rewards[:, t] + self.cfg.gamma * following
            following = returns[:, t]
        return returns

    def _update(self):
        returns, dones = self._returns(), self._dones()
        episode = dones.long().cumsum(1) - dones.long()
        envs = torch.arange(self.num_envs, device=dones.device).unsqueeze(1)
        key = (envs * self.cfg.horizon + episode) * self.bin_count + self._bins(self._frames)
        keys, inverse, visits = torch.unique(key.flatten(), return_inverse=True, return_counts=True)
        samples = torch.zeros(keys.numel(), device=returns.device)
        samples.scatter_add_(0, inverse, returns.flatten())
        samples /= visits
        bins = keys % self.bin_count
        new = torch.zeros_like(self.count).scatter_add_(0, bins, torch.ones_like(samples))
        total = torch.zeros_like(self.value).scatter_add_(0, bins, samples)
        # Keep these storages stable: the command exposes value as bin_return.
        self.count.copy_(self.count.clamp(max=self.cfg.memory) + new)
        self.value.copy_(
            torch.where(
                new > 0,
                self.value + (total - new * self.value) / self.count.clamp_min(1),
                self.value,
            )
        )

    def _distribution(self):
        # The upstream statistical algorithm is unchanged. Our H32 reset policy
        # additionally excludes bins with no admissible reset frame; their
        # visited-state value estimates still participate in bootstrap/clip tails.
        logits = (-self.cfg.steepness * self.value / self.v_max).masked_fill(
            ~self.sampleable_bins,
            -torch.inf,
        )
        focus = torch.softmax(logits, 0)
        self.effort.add_(focus.double().square())
        keep = (self.effort / self.effort.sum()).float()
        uniform = self.sampleable_bins.float() / self.num_sampleable_bins
        return (
            self.cfg.explore * uniform
            + (1 - self.cfg.explore - self.cfg.keep) * focus
            + self.cfg.keep * keep
        )

    def keep_distribution(self):
        total = self.effort.sum()
        return torch.where(
            total > 0,
            (self.effort / total.clamp_min(1e-300)).float(),
            self.sampleable_bins.float() / self.num_sampleable_bins,
        )

    @torch.no_grad()
    def discard_pending_window(self):
        """Use on simulator restart; do not stitch old episodes to a fresh reset."""
        self._step = 0
        for name in ("_frames", "_rewards", "_terminated", "_clip_end", "_timed_out"):
            getattr(self, name).zero_()

    @torch.no_grad()
    def reset(self):
        self.value.fill_(self.v_max)
        self.count.zero_()
        self.effort.zero_()
        self.probabilities.copy_(self.sampleable_bins.float() / self.num_sampleable_bins)
        self.num_updates = 0
        self.discard_pending_window()

    def state_dict(self):
        return {
            "version": 2,
            "bootstrap_next_frame": self.bootstrap_next_frame,
            "cfg": asdict(self.cfg),
            "v_max": self.v_max,
            "bin_count": self.bin_count,
            "frame_count": self.frame_count,
            "num_envs": self.num_envs,
            "step": self._step,
            "num_updates": self.num_updates,
            "sampleable_bins": self.sampleable_bins.clone(),
            **{name: getattr(self, name).clone() for name in self._TENSORS},
        }

    def validate_state_dict(self, state):
        """Validate the complete snapshot before any live buffer is changed."""
        if not isinstance(state, Mapping):
            raise ValueError("Window sampler checkpoint must be a mapping")  # noqa: TRY004
        version = state.get("version")
        if isinstance(version, bool) or not isinstance(version, int) or version not in (1, 2):
            raise ValueError("Window sampler checkpoint version differs from runtime")
        if version == 1 and self.last_frame_of is not None:
            raise ValueError("Window sampler checkpoint bootstrap_next_frame differs from runtime")
        # Old standalone global-bin snapshots had no successor-policy field.
        # They are safe only in the unchanged global-clamp implementation.
        saved_successor = state.get("bootstrap_next_frame", "global_clamped" if version == 1 else None)
        if saved_successor != self.bootstrap_next_frame:
            raise ValueError("Window sampler checkpoint bootstrap_next_frame differs from runtime")
        expected = {
            "cfg": asdict(self.cfg),
            "v_max": self.v_max,
            "bin_count": self.bin_count,
            "frame_count": self.frame_count,
            "num_envs": self.num_envs,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"Window sampler checkpoint {key} differs from runtime")
        for name, lower, upper in (("step", 0, self.cfg.horizon), ("num_updates", 0, None)):
            value = state.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < lower
                or (upper is not None and value >= upper)
            ):
                raise ValueError(f"Window sampler checkpoint {name} is invalid")
        eligible = state.get("sampleable_bins")
        if (
            not isinstance(eligible, torch.Tensor)
            or eligible.dtype != torch.bool
            or eligible.shape != self.sampleable_bins.shape
            or not torch.equal(eligible.to(self.sampleable_bins.device), self.sampleable_bins)
        ):
            raise ValueError("Window sampler checkpoint eligibility differs from runtime")
        converted = {}
        for name in self._TENSORS:
            src, target = state.get(name), getattr(self, name)
            if (
                not isinstance(src, torch.Tensor)
                or src.shape != target.shape
                or src.dtype != target.dtype
            ):
                raise ValueError(f"Window sampler checkpoint {name} has incompatible shape/dtype")
            value = src.to(target)
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"Window sampler checkpoint {name} contains nonfinite values")
            converted[name] = value
        for name in ("count", "effort", "probabilities"):
            if bool((converted[name] < 0).any()):
                raise ValueError(f"Window sampler checkpoint {name} must be nonnegative")
        probs = converted["probabilities"]
        if (
            not bool(torch.isclose(probs.sum(), probs.new_tensor(1.0), rtol=1e-5, atol=1e-5))
            or bool((probs[~self.sampleable_bins] != 0).any())
            or bool((converted["effort"][~self.sampleable_bins] != 0).any())
        ):
            raise ValueError("Window sampler checkpoint probability/effort support is invalid")
        frames = converted["_frames"][:, : state["step"]]
        if bool(((frames < 0) | (frames >= self.frame_count)).any()):
            raise ValueError("Window sampler checkpoint pending frame is outside the library")
        return converted

    @torch.no_grad()
    def load_state_dict(self, state):
        converted = self.validate_state_dict(state)
        for name, value in converted.items():
            getattr(self, name).copy_(value)
        self._step, self.num_updates = state["step"], state["num_updates"]


@torch.no_grad()
def mean_reward_continuation(discounted_return, reward_sum, steps, terminated, gamma):
    """Extrapolate mean reward after a normal cutoff; physical failures get no tail."""
    if not 0 < gamma < 1:
        raise ValueError("Sampler discount must lie strictly between zero and one")
    if (steps <= 0).any():
        raise ValueError("A sampler target needs at least one real transition")
    tail = gamma**steps * (reward_sum / steps) / (1 - gamma)
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
        "episodes",
        "post_first_episodes",
        "steps",
        "first_reward",
        "first_action_rate",
        "post_first_reward_mean",
        "post_first_action_rate_mean",
        "post_first_reward_sum",
        "post_first_steps",
        "first_target_contribution",
        "first_action_rate_target_contribution",
        "post_first_target_contribution",
        "sampler_target",
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
            first,
            action_rate_reward.double(),
            self.first_action_rate[ids],
        )
        self.action_rate_sum[ids] += action_rate_reward.double()

    @torch.no_grad()
    def finalize(self, ids, reward_sum, steps, terminated, targets, *, extrapolate: bool = True):
        first, first_rate = self.first_reward[ids], self.first_action_rate[ids]
        has_post_first = steps > 1
        post_steps = (steps - 1).clamp_min(1)
        post_reward = reward_sum - first
        post_rate = self.action_rate_sum[ids] - first_rate
        # Linear decomposition of the *unchanged* sampler target. At a normal
        # cutoff, first reward is also included in the extrapolated mean tail.
        tail_multiplier = (
            torch.where(
                terminated,
                0.0,
                self.gamma**steps / (steps * (1 - self.gamma)),
            )
            if extrapolate
            else torch.zeros_like(first)
        )
        multiplier = 1 + tail_multiplier
        first_target = first * multiplier
        first_rate_target = first_rate * multiplier
        values = (
            torch.ones_like(first),
            has_post_first.double(),
            steps.double(),
            first,
            first_rate,
            torch.where(has_post_first, post_reward / post_steps, 0.0),
            torch.where(has_post_first, post_rate / post_steps, 0.0),
            post_reward,
            (steps - 1).double(),
            first_target,
            first_rate_target,
            targets - first_target,
            targets,
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


def reset_diagnostic_metrics(
    totals: torch.Tensor, max_step_reward: float, *, sampler_kind="legacy"
) -> dict[str, float]:
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
        result.update(
            {
                "reset/episode_steps": sums["steps"] / count,
                "reset/first_step_reward": sums["first_reward"] / count,
                "reset/first_step_action_rate_reward": sums["first_action_rate"] / count,
                "reset/first_step_target_contribution": sums["first_target_contribution"] / count,
                "reset/first_step_action_rate_target_contribution": sums[
                    "first_action_rate_target_contribution"
                ]
                / count,
                "tracking/post_first_step_target_contribution": sums[
                    "post_first_target_contribution"
                ]
                / count,
                "sampler/completed_episode_target": sums["sampler_target"] / count,
            }
        )
    if post_count:
        post_mean = sums["post_first_reward_mean"] / post_count
        result.update(
            {
                "tracking/post_first_step_reward": post_mean,
                "tracking/post_first_step_norm_return": post_mean / max_step_reward,
                "tracking/post_first_step_action_rate_reward": sums["post_first_action_rate_mean"]
                / post_count,
                "tracking/post_first_step_norm_return_step_weighted": (
                    sums["post_first_reward_sum"] / (max_step_reward * sums["post_first_steps"])
                ),
            }
        )
    if sampler_kind == "window_value":
        # These episode diagnostics contain observed D, not the sampler's
        # per-window/bin bootstrap targets. Do not present them as V estimates.
        result = {
            key.replace("target_contribution", "discounted_return_contribution").replace(
                "sampler/completed_episode_target",
                "diagnostics/completed_episode_discounted_return",
            ): value
            for key, value in result.items()
        }
    return result

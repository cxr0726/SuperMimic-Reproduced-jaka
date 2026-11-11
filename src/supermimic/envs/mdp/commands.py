"""Multi-clip motion command with future-frame goals and return-based reset sampling.

Exposes the same attribute API as mjlab's ``MotionCommand`` (``anchor_pos_w``,
``robot_body_pos_w``, ``body_pos_relative_w`` ...) so mjlab's tracking reward
functions work unchanged, plus:

* ``goal``        (N, K*68) future-frame goal features (plan D5)
* ``root_*``      reference pelvis quantities for the paper's root terminations
* ``clip_end``    (N,) True once the clip's last frame has been reached (time-out)
* return-based reset sampling over 1 s bins of the whole library (plan D14)
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.envs.mdp.reset_sampling import (
    ResetEpisodeDiagnostics,
    WindowValueSampler,
    WindowValueSamplerCfg,
    mean_reward_continuation,
    update_bin_returns,
)

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


class MultiClipMotionCommand(CommandTerm):
    cfg: MultiClipMotionCommandCfg
    _env: ManagerBasedRlEnv

    def __init__(self, cfg: MultiClipMotionCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]
        self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
        self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        self.robot_root_body_index = self.robot.body_names.index(cfg.root_body_name)

        default_joint_pos = self.robot.data.default_joint_pos[0].detach()
        self.library = MotionLibrary(
            cfg.npz_files, cfg.body_names, default_joint_pos, device=self.device, bin_frames=cfg.bin_frames,
            compact=cfg.compact_library, store_dtype=torch.float16 if cfg.library_fp16 else torch.float32,
            reference_fk_compile=cfg.reference_fk_compile,
            robot_name=cfg.robot_name, root_body_name=cfg.root_body_name,
            joint_names=self.robot.joint_names,
            merge_short_tail_bins=cfg.merge_short_tail_bins,
            packed_clip_names=cfg.packed_clip_names,
            tail_bin_merge_fraction=cfg.tail_bin_merge_fraction,
        )
        print(f"[INFO] {self.library.summary()}")
        lib = self.library

        n = self.num_envs
        self.frame = torch.zeros(n, dtype=torch.long, device=self.device)
        self.clip_end = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._refresh_ref()
        self.body_pos_relative_w = torch.zeros(n, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(n, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        # Return-based reset sampling state.
        self._configure_reset_sampling()
        self.bin_return = torch.zeros(lib.num_bins, device=self.device)
        self.bin_memory = torch.zeros_like(self.bin_return)
        if cfg.reset_sampler not in ("legacy", "window_value"):
            raise ValueError(f"Unknown reset sampler: {cfg.reset_sampler}")
        self.window_sampler = None
        if cfg.reset_sampler == "window_value":
            if not math.isclose(cfg.hard_ratio, 1 - cfg.uniform_ratio - cfg.memory_ratio):
                raise ValueError("Window sampler mixture weights must sum to one")
            self.window_sampler = WindowValueSampler(
                WindowValueSamplerCfg(
                    gamma=cfg.gamma, horizon=cfg.return_horizon, memory=cfg.return_memory,
                    steepness=cfg.softmax_temp, explore=cfg.uniform_ratio, keep=cfg.memory_ratio,
                ), lib.num_bins, lib.num_frames, n, cfg.g_max * (1 - cfg.gamma), self.device,
                bin_of_frame=lib.bin_of_frame, sampleable_bins=self.sampleable_bins,
                last_frame_of=lib.last_frame_of,
            )
            self.bin_return = self.window_sampler.value
        self.reset_sampler_statistics()
        self.sampler_contract = self._make_sampler_contract()
        # Evaluation hook: when >= 0, the next reset of that env starts exactly at this global frame.
        self.forced_frame = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.ep_reward_sum = torch.zeros(n, dtype=torch.float64, device=self.device)
        self.ep_discounted_return = torch.zeros_like(self.ep_reward_sum)
        self.episode_start_bin = torch.zeros(n, dtype=torch.long, device=self.device)
        self.ep_len = torch.zeros(n, dtype=torch.long, device=self.device)
        self.last_logged_step = torch.full_like(self.ep_len, env.common_step_counter)
        self.reset_diagnostics = ResetEpisodeDiagnostics(n, self.device, cfg.gamma)
        self._compute_dt: float | torch.Tensor = 0.0

        for k in ("error_anchor_pos", "error_anchor_rot", "error_body_pos", "error_body_rot",
                  "error_joint_pos", "error_joint_vel", "sampling_entropy", "episode_return_to_go"):
            self.metrics[k] = torch.zeros(n, device=self.device)

    def _configure_reset_sampling(self) -> None:
        """Precompute admissible starts; runtime sampling stays entirely on-device."""
        lib, horizon = self.library, self.cfg.reset_min_remaining_steps
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("reset_min_remaining_steps must be a positive integer")
        # clip_last is inclusive: frame + H <= clip_last, so the exclusive
        # bound is clip_last - H + 1. Retain global bin IDs for diagnostics.
        self.sample_bin_end = torch.minimum(lib.bin_end, lib.clip_last[lib.bin_clip] - horizon + 1)
        self.sampleable_bins = lib.bin_start < self.sample_bin_end
        self.num_sampleable_bins = int(self.sampleable_bins.sum().item())
        if self.num_sampleable_bins == 0:
            raise ValueError(f"Motion library has no reset state with {horizon} subsequent reference steps")
        if horizon > 1:
            counts = (lib.clip_last - lib.clip_start + 1 - horizon).clamp_min(0)
            self.sampleable_clip_ids = (counts > 0).nonzero().flatten()
            self.sampleable_frame_cumsum = counts.cumsum(0)
            self.sampleable_frame_offsets = self.sampleable_frame_cumsum - counts
            self.num_sampleable_frames = int(self.sampleable_frame_cumsum[-1].item())

    def _make_sampler_contract(self) -> dict:
        cfg = self.cfg
        layout = [(clip.name, clip.length) for clip in self.library.clips]
        contract = {
            "version": 2, "return_target": "start_bin_mean_reward_continuation",
            "duplicate_bins": "batch_mean", "initial_return": 0.0,
            "bin_frames": cfg.bin_frames, "gamma": cfg.gamma, "g_max": cfg.g_max,
            "bin_ema": cfg.ema_alpha, "history_method": "ema_per_sample_call",
            "history_ema": cfg.memory_alpha, "softmax_temp": cfg.softmax_temp,
            "mixture": [cfg.uniform_ratio, cfg.hard_ratio, cfg.memory_ratio],
            "reset_boundary": "frame_before_inclusive_last",
            "clip_layout_sha256": hashlib.sha256(
                json.dumps(layout, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        if cfg.merge_short_tail_bins:
            contract.update(
                version=3, tail_bin_policy="merge_remainder_strictly_less_than_half_reference_bin",
            )
        if cfg.reset_min_remaining_steps > 1 or (cfg.merge_short_tail_bins and cfg.tail_bin_merge_fraction != 0.5):
            contract.update(
                version=4,
                merge_short_tail_bins=cfg.merge_short_tail_bins,
                tail_bin_merge_fraction=cfg.tail_bin_merge_fraction,
                tail_bin_policy="merge_remainder_strictly_less_than_fraction_of_reference_bin"
                if cfg.merge_short_tail_bins else "keep_all_remainders",
                reset_boundary="frame_plus_min_remaining_steps_le_inclusive_last",
                reset_min_remaining_steps=cfg.reset_min_remaining_steps,
            )
        if cfg.reset_sampler == "window_value":
            for name in ("bin_ema", "history_ema"):
                contract.pop(name)
            contract.update(
                version=6, algorithm="official_window_value",
                return_target="window_nstep_bin_value_bootstrap",
                duplicate_bins="mean_per_env_episode_bin_within_window",
                initial_return=cfg.g_max, history_method="cumulative_focus_squared_per_window",
                update_horizon=cfg.return_horizon, count_memory=cfg.return_memory,
                failure_timeout_precedence="timeout",
                clip_end_bootstrap="all_bin_value_mean",
                frame_bin_mapping="clip_local_with_merged_tail",
                bootstrap_next_frame="clip_clamped",
            )
        return contract

    # ------------------------------------------------------------------ reference (mjlab MotionCommand API)

    def _refresh_ref(self) -> None:
        """Cache the reference state / tracked-body states of the current target frames."""
        self._ref = self.library.ref(self.frame)
        self._bodies = self.library.bodies(self.frame)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def goal(self) -> torch.Tensor:
        return self.library.goal_window(self.frame, self.cfg.future_frames)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._ref.joint_pos

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._ref.joint_vel

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._bodies.pos_w + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._bodies.quat_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._bodies.lin_vel_w

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._bodies.ang_vel_w

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self._bodies.pos_w[:, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self._bodies.quat_w[:, self.motion_anchor_body_index]

    @property
    def root_pos_w(self) -> torch.Tensor:
        return self._ref.root_pos_w + self._env.scene.env_origins

    @property
    def root_quat_w(self) -> torch.Tensor:
        return self._ref.root_quat_w

    # ------------------------------------------------------------------ robot

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_root_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.robot_root_body_index]

    @property
    def robot_root_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.robot_root_body_index]

    # ------------------------------------------------------------------ metrics

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(-1)
        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    # ------------------------------------------------------------------ reward bookkeeping for the sampler

    def reset_sampler_statistics(self) -> None:
        """Discard scores from a different sampler algorithm without changing networks."""
        window = getattr(self, "window_sampler", None)
        if window is not None:
            window.reset()
        else:
            self.bin_return.zero_()
        self.bin_memory.copy_(self.sampleable_bins / self.num_sampleable_bins)

    def record_window_transition(self, pre_frames, rewards, terminated, timeouts, clip_end):
        """Called once after a vectorised physical step, never from reset callbacks.

        The caller must snapshot frames before env.step: auto-reset has already
        installed a new reference by the time env.step returns. As upstream,
        simultaneous timeout/failure is treated as truncation, not a fall.
        """
        window = getattr(self, "window_sampler", None)
        if window is None or self.cfg.sampling_mode != "return":
            return None
        truncated = timeouts.bool() | clip_end.bool()
        probabilities = window.record(
            pre_frames, rewards, terminated.bool() & ~truncated,
            clip_end.bool(), truncated & ~clip_end.bool(),
        )
        if probabilities is not None:
            self.bin_memory.copy_(window.keep_distribution())
            entropy = -(probabilities * (probabilities + 1e-12).log()).sum()
            self.metrics["sampling_entropy"][:] = entropy / math.log(max(self.num_sampleable_bins, 2))
        return probabilities

    def window_sampler_state_dict(self):
        window = getattr(self, "window_sampler", None)
        return window.state_dict() if window is not None else None

    def load_window_sampler_state_dict(self, state, *, discard_pending=False):
        window = getattr(self, "window_sampler", None)
        if window is None:
            raise ValueError("A window-value sampler checkpoint needs the window_value algorithm")
        window.load_state_dict(state)
        if discard_pending:
            window.discard_pending_window()
        self.bin_memory.copy_(window.keep_distribution())
        p = window.probabilities
        self.metrics["sampling_entropy"][:] = -(p * (p + 1e-12).log()).sum() / math.log(max(self.num_sampleable_bins, 2))

    def _log_step_rewards(self, env_mask: torch.Tensor | None = None) -> None:
        """Count each real simulator reward once, including the terminal transition."""
        reward = getattr(self._env, "reward_buf", None)
        if reward is None:
            return
        pending = self.last_logged_step < self._env.common_step_counter
        if env_mask is not None:
            pending = pending & env_mask
        ids = pending.nonzero().flatten()
        if ids.numel() == 0:
            return
        first = ids[self.ep_len[ids] == 0]
        self.episode_start_bin[first] = self.library.bin_of_frame(self.frame[first])
        r = reward[ids].double()
        reward_manager = getattr(self._env, "reward_manager", None)
        if reward_manager is not None and "action_rate_l2" in reward_manager.active_terms:
            # Public API includes the same term weight and dt as reward_buf.
            action_rate = reward_manager.get_term("action_rate_l2")[ids]
        else:
            action_rate = torch.zeros_like(r)
        self.reset_diagnostics.record(ids, r, action_rate, self.ep_len[ids])
        self.ep_discounted_return[ids] += self.cfg.gamma ** self.ep_len[ids].double() * r
        self.ep_reward_sum[ids] += r
        self.ep_len[ids] += 1
        self.last_logged_step[ids] = self._env.common_step_counter

    def _finalize_episodes(self, env_ids: torch.Tensor) -> None:
        """Finalize diagnostics; only legacy mode updates episode-start returns."""
        ids = env_ids[self.ep_len[env_ids] > 0]
        if ids.numel() == 0:
            return
        terminated = self._env.termination_manager.terminated[ids]
        is_window = self.cfg.reset_sampler == "window_value"
        targets = self.ep_discounted_return[ids] if is_window else mean_reward_continuation(
            self.ep_discounted_return[ids], self.ep_reward_sum[ids],
            self.ep_len[ids], terminated, self.cfg.gamma,
        )
        self.reset_diagnostics.finalize(
            ids, self.ep_reward_sum[ids], self.ep_len[ids], terminated, targets,
            extrapolate=not is_window,
        )
        if not is_window:
            update_bin_returns(self.bin_return, self.episode_start_bin[ids], targets, self.cfg.ema_alpha)
        self.metrics["episode_return_to_go"][ids] = targets.float()
        self.ep_len[ids] = 0
        self.ep_reward_sum[ids] = 0
        self.ep_discounted_return[ids] = 0

    # ------------------------------------------------------------------ sampling

    def _sample_frames(self, n: int) -> torch.Tensor:
        lib = self.library
        cfg = self.cfg
        if cfg.sampling_mode == "start":
            if cfg.reset_min_remaining_steps > 1:
                indices = torch.randint(0, self.sampleable_clip_ids.numel(), (n,), device=self.device)
                return lib.clip_start[self.sampleable_clip_ids[indices]]
            clips = torch.randint(0, lib.num_clips, (n,), device=self.device)
            return lib.clip_start[clips]
        if cfg.sampling_mode == "uniform":
            # Uniform over frames (long clips get proportionally more starts).
            if cfg.reset_min_remaining_steps > 1:
                rank = torch.randint(0, self.num_sampleable_frames, (n,), device=self.device)
                clips = torch.searchsorted(self.sampleable_frame_cumsum, rank, right=True)
                return lib.clip_start[clips] + rank - self.sampleable_frame_offsets[clips]
            f = torch.randint(0, lib.num_frames, (n,), device=self.device)
            return torch.minimum(f, lib.last_frame_of(f) - 1).clamp_min(lib.clip_start[lib.frame_clip[f]])
        assert cfg.sampling_mode == "return"
        if cfg.reset_sampler == "window_value":
            # Distribution/keep are refreshed only every return_horizon real
            # control steps. Reset frequency does not set their time scale.
            p = self.window_sampler.probabilities
            b = torch.multinomial(p, n, replacement=True)
            u = torch.rand(n, device=self.device)
            return lib.bin_start[b] + (u * (self.sample_bin_end[b] - lib.bin_start[b]).float()).long()
        logits = (-cfg.softmax_temp * self.bin_return / cfg.g_max).masked_fill(~self.sampleable_bins, -torch.inf)
        hard = torch.softmax(logits, dim=0)
        self.bin_memory.lerp_(hard, cfg.memory_alpha)
        p = (
            cfg.uniform_ratio * self.sampleable_bins / self.num_sampleable_bins
            + cfg.hard_ratio * hard
            + cfg.memory_ratio * self.bin_memory / self.bin_memory.sum()
        )
        p = p / p.sum()
        b = torch.multinomial(p, n, replacement=True)
        u = torch.rand(n, device=self.device)
        f = lib.bin_start[b] + (u * (self.sample_bin_end[b] - lib.bin_start[b]).float()).long()
        ent = -(p * (p + 1e-12).log()).sum() / math.log(max(self.num_sampleable_bins, 2))
        self.metrics["sampling_entropy"][:] = ent
        return f

    def _write_reference_state_to_sim(self, env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel):
        # Spawn clearance changes the actual initial state, never the library,
        # reference reward/goal height, or subsequent commanded trajectory.
        root_pos = root_pos.clone()
        root_pos[:, 2] += self.cfg.reset_root_z_offset
        soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.robot.reset(env_ids=env_ids)

    def _resample_command(self, env_ids: torch.Tensor):
        # Episode bookkeeping for the envs being reset (terminal reward is already in reward_buf).
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[env_ids] = True
        self._log_step_rewards(mask)
        self._finalize_episodes(env_ids)
        self.ep_len[env_ids] = 0
        self.ep_reward_sum[env_ids] = 0
        self.ep_discounted_return[env_ids] = 0

        sampled = self._sample_frames(len(env_ids))
        forced = self.forced_frame[env_ids]
        selected = torch.where(forced >= 0, forced, sampled)
        if self.cfg.reset_min_remaining_steps > 1:
            # forced_frame is a public evaluation hook. Assert on-device so
            # regular GPU resets do not add a CPU synchronization point.
            torch._assert_async(self._valid_reset_frames(selected).all(), self._reset_frame_error())
        self.frame[env_ids] = selected
        self.clip_end[env_ids] = False

        r = self.library.ref(self.frame[env_ids])
        root_pos = r.root_pos_w + self._env.scene.env_origins[env_ids]
        root_ori = r.root_quat_w.clone()
        root_lin_vel = r.root_lin_vel_w.clone()
        root_ang_vel = r.root_ang_vel_w.clone()
        joint_pos = r.joint_pos.clone()
        joint_vel = r.joint_vel

        keys = ["x", "y", "z", "roll", "pitch", "yaw"]
        ranges = torch.tensor([self.cfg.pose_range.get(k, (0.0, 0.0)) for k in keys], device=self.device)
        rs = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos = root_pos + rs[:, 0:3]
        root_ori = quat_mul(quat_from_euler_xyz(rs[:, 3], rs[:, 4], rs[:, 5]), root_ori)
        ranges = torch.tensor([self.cfg.velocity_range.get(k, (0.0, 0.0)) for k in keys], device=self.device)
        rs = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel = root_lin_vel + rs[:, :3]
        root_ang_vel = root_ang_vel + rs[:, 3:]
        lo, hi = self.cfg.joint_position_range
        if hi > lo:
            joint_pos = joint_pos + sample_uniform(lo, hi, joint_pos.shape, device=self.device)

        self._write_reference_state_to_sim(env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel)

    def update_relative_body_poses(self) -> None:
        nb = len(self.cfg.body_names)
        anchor_pos = self.anchor_pos_w[:, None, :].repeat(1, nb, 1)
        anchor_quat = self.anchor_quat_w[:, None, :].repeat(1, nb, 1)
        robot_anchor_pos = self.robot_anchor_pos_w[:, None, :].repeat(1, nb, 1)
        robot_anchor_quat = self.robot_anchor_quat_w[:, None, :].repeat(1, nb, 1)
        delta_pos_w = robot_anchor_pos
        delta_pos_w[..., 2] = anchor_pos[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(anchor_quat)))
        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos)

    def compute(self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        # mjlab uses dt=0 after explicit resets and for freshly auto-reset envs.
        # The base class scopes timers but does not pass dt to _update_command.
        self._compute_dt = dt
        super().compute(dt, env_ids)

    def _update_command(self, env_ids: torch.Tensor | None = None):
        # Clip end must be judged against the clip of the *current* frame (frame+1 may already be
        # the first frame of the next clip in the concatenated library).
        last = self.library.last_frame_of(self.frame)
        dt = self._compute_dt
        if isinstance(dt, torch.Tensor):
            advance = dt > 0
        else:
            advance = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            if dt > 0:
                advance[:] = env_ids is None
                if env_ids is not None:
                    advance[env_ids] = True
        self._log_step_rewards(advance)
        self.frame[advance] += 1
        over = advance & (self.frame > last)
        if over.any():
            self.clip_end |= over
            self.frame = torch.where(over, last, self.frame)
        self._refresh_ref()
        self.update_relative_body_poses()

    def reset_to_frame(self, env_ids: torch.Tensor, frame: torch.Tensor) -> None:
        """Deterministic reference-frame reset plus configured actual-root clearance."""
        if self.cfg.reset_min_remaining_steps > 1 and not bool(self._valid_reset_frames(frame).all()):
            raise ValueError(self._reset_frame_error())
        self.ep_len[env_ids] = 0
        self.ep_reward_sum[env_ids] = 0
        self.ep_discounted_return[env_ids] = 0
        self.last_logged_step[env_ids] = self._env.common_step_counter
        self.frame[env_ids] = frame
        self.clip_end[env_ids] = False
        r = self.library.ref(frame)
        self._write_reference_state_to_sim(
            env_ids, r.root_pos_w + self._env.scene.env_origins[env_ids], r.root_quat_w, r.root_lin_vel_w, r.root_ang_vel_w, r.joint_pos, r.joint_vel
        )

    def _valid_reset_frames(self, frame: torch.Tensor) -> torch.Tensor:
        lib = self.library
        last = lib.last_frame_of(frame.clamp(0, lib.num_frames - 1))
        return (frame >= 0) & (frame < lib.num_frames) & (frame + self.cfg.reset_min_remaining_steps <= last)

    def _reset_frame_error(self) -> str:
        return ("Reset frame must be inside its clip with at least "
                f"{self.cfg.reset_min_remaining_steps} subsequent reference steps; "
                "set reset_min_remaining_steps=1 explicitly for unrestricted evaluation starts")

    # ------------------------------------------------------------------ debug visualization (viewer only)

    _ghost_model = None

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        """Draw the reference pose as a translucent ghost robot (same approach as mjlab's MotionCommand)."""
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        if self._ghost_model is None:
            import copy

            m = copy.deepcopy(self._env.sim.mj_model)
            for gi in range(m.ngeom):
                if m.geom_contype[gi] != 0 or m.geom_conaffinity[gi] != 0:
                    m.geom_rgba[gi, 3] = 0  # hide collision geoms
                else:
                    m.geom_rgba[gi] = (0.2, 0.6, 1.0, 0.45)
            self._ghost_model = m
        indexing = self.robot.indexing
        free_q = indexing.free_joint_q_adr.cpu().numpy()
        joint_q = indexing.joint_q_adr.cpu().numpy()
        root_pos = self.root_pos_w.cpu().numpy()
        root_quat = self.root_quat_w.cpu().numpy()
        joint_pos = self.joint_pos.cpu().numpy()
        import numpy as np

        for e in env_indices:
            qpos = np.zeros(self._ghost_model.nq)
            qpos[free_q[0:3]] = root_pos[e]
            qpos[free_q[3:7]] = root_quat[e]
            qpos[joint_q] = joint_pos[e]
            visualizer.add_ghost_mesh(qpos, model=self._ghost_model, label=f"ghost_{e}")


@dataclass(kw_only=True)
class MultiClipMotionCommandCfg(CommandTermCfg):
    npz_files: list[str]
    anchor_body_name: str
    root_body_name: str = "pelvis"
    body_names: tuple[str, ...]
    entity_name: str = "robot"
    robot_name: str = "g1"
    future_frames: int = 10
    bin_frames: int = 50
    merge_short_tail_bins: bool = False
    tail_bin_merge_fraction: float = 0.5
    reset_min_remaining_steps: int = 1
    pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    joint_position_range: tuple[float, float] = (0.0, 0.0)
    reset_root_z_offset: float = 0.0
    sampling_mode: Literal["return", "uniform", "start"] = "return"
    reset_sampler: Literal["legacy", "window_value"] = "legacy"
    return_horizon: int = 20
    return_memory: int = 256
    # Return-based sampling: start-bin targets, mean-reward continuation at cutoffs.
    uniform_ratio: float = 0.5
    hard_ratio: float = 0.4
    memory_ratio: float = 0.1
    softmax_temp: float = 25.0
    g_max: float = 10.0
    gamma: float = 0.99
    ema_alpha: float = 0.2
    memory_alpha: float = 0.001
    compact_library: bool = False
    packed_clip_names: tuple[str, ...] | None = None
    reference_fk_compile: bool = False
    library_fp16: bool = True
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    def build(self, env: ManagerBasedRlEnv) -> MultiClipMotionCommand:
        return MultiClipMotionCommand(self, env)

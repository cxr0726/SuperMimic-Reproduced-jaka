"""Policy (goal encoder + shared decoder) and critic (paper Table A.4)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from supermimic.normalization import RunningNormalizer
from supermimic.observation_layout import (
    goal_noise_half,
    policy_observation_dimension,
    reference_joint_positions,
)
from supermimic.world_model.dynamics import MLP
from supermimic.world_model.heads import observation_terms

GOAL_DIM = 680
LATENT_DIM = 32
ACT_DIM = 29


@dataclass
class PolicyCfg:
    encoder_hidden: tuple[int, ...] = (1024, 512, 256)
    decoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    latent_dim: int = LATENT_DIM
    history: int | None = 10  # None is resolved by the training configuration
    observation_layout: Literal["supermimic", "isaac_khan"] = "supermimic"
    future_frames: int = 10
    # SuperMimic scales (divide): q, qd, ang, gravity, action. Khan inputs are pre-scaled.
    obs_term_scale: tuple[float, ...] = (1.0, 5.0, 2.0, 1.0, 1.0)
    # SuperMimic goal scales: q, qd, gravity, ang, lin, height; Isaac uses identity.
    goal_term_scale: tuple[float, ...] = (1.0, 5.0, 1.0, 2.0, 1.0, 1.0)
    decoder_last_gain: float = 0.01
    goal_dim: int = GOAL_DIM  # G1:680 (+9 anchor); Khan Isaac:155
    num_joints: int = ACT_DIM
    num_tracked_bodies: int | None = None  # Training resolves this from RobotSpec.
    # Missing action/noise fields retain legacy semantics; checkpoint loaders
    # resolve historical hidden widths explicitly. TrainConfig resolves fresh runs.
    action_mode: Literal["auto", "default", "reference_residual"] = "default"
    reference_residual_beta: float = 1.0
    reference_frame_offset: int = 1
    goal_noise_z: float | None = 0.0  # Uniform half-width in meters, encoder only.
    goal_noise_joint_pos: float | None = 0.0  # Uniform half-width in radians.
    running_normalization: bool = False  # Historical checkpoints keep their fixed input scales.
    hidden_layer_norm: bool = False  # Missing checkpoint fields retain the original SiLU MLP.
    hidden_activation: Literal["silu", "gelu"] = "silu"


@dataclass
class CriticCfg:
    hidden: tuple[int, ...] = (256, 128)
    hidden_layer_norm: bool = False
    hidden_activation: Literal["silu", "gelu"] = "silu"


def _hidden_activation(name: str) -> type[nn.Module]:
    activations = {"silu": nn.SiLU, "gelu": nn.GELU}
    if name not in activations:
        raise ValueError(f"Unknown policy/critic hidden activation: {name!r}")
    return activations[name]


def critic_goal_dimension(cfg: PolicyCfg) -> int:
    """Clean command, future orientation and reference velocity features for Khan."""
    if cfg.observation_layout == "isaac_khan":
        # Per reference frame: orientation6 + qd(J) + gravity3 + ang_b3 + lin_h3.
        return cfg.goal_dim + cfg.future_frames * (6 + cfg.num_joints + 9)
    return cfg.goal_dim


def critic_error_dimension(cfg: PolicyCfg) -> int:
    """Khan's current ref-minus-robot joint/body errors and relative root rotation."""
    if cfg.observation_layout == "isaac_khan":
        num_bodies = 15 if cfg.num_tracked_bodies is None else cfg.num_tracked_bodies
        if num_bodies < 1:
            raise ValueError("Khan critic needs at least one tracked body")
        return cfg.num_joints + 3 * num_bodies + 6
    return 0


def _obs_scale_vector(cfg: PolicyCfg) -> torch.Tensor:
    if cfg.history is None:
        raise ValueError("Resolve the policy observation history before constructing a network")
    if cfg.observation_layout == "isaac_khan":
        return torch.ones(policy_observation_dimension(
            cfg.observation_layout, cfg.num_joints, cfg.history, cfg.future_frames,
        ))
    parts = []
    for (_, d), s in zip(observation_terms(cfg.num_joints), cfg.obs_term_scale):
        parts.append(torch.full((cfg.history * d,), 1.0 / s))
    return torch.cat(parts)  # term-major layout matches the flattened history


def _goal_scale_vector(cfg: PolicyCfg, goal_dim: int) -> torch.Tensor:
    if cfg.observation_layout == "isaac_khan":
        return torch.ones(goal_dim)
    sizes = (cfg.num_joints, cfg.num_joints, 3, 3, 3, 1)
    per_frame = torch.cat([torch.full((n,), 1.0 / s) for n, s in zip(sizes, cfg.goal_term_scale)])
    k, extra = divmod(goal_dim, 2 * cfg.num_joints + 10)
    return torch.cat([per_frame.repeat(k), torch.ones(extra)])


def _critic_goal_scale_vector(cfg: PolicyCfg, goal_dim: int) -> torch.Tensor:
    """Scale critic-only reference velocities without changing the actor goal.

    Khan stores [command, orientation, extra_frames]. Each extra frame contains
    [joint velocity, projected gravity, angular velocity in root, linear velocity
    in heading], with the original SuperMimic goal divisors [5, 1, 2, 1].
    """
    if cfg.observation_layout != "isaac_khan":
        return _goal_scale_vector(cfg, goal_dim)
    expected = critic_goal_dimension(cfg)
    if goal_dim != expected:
        raise ValueError(f"Khan critic reference needs {expected} values, received {goal_dim}")
    prefix_width = cfg.goal_dim + 6 * cfg.future_frames
    extra_frame = torch.cat([
        torch.full((cfg.num_joints,), 1.0 / 5.0), torch.ones(3),
        torch.full((3,), 1.0 / 2.0), torch.ones(3),
    ])
    return torch.cat([torch.ones(prefix_width), extra_frame.repeat(cfg.future_frames)])


class MotionEncoder(nn.Module):
    def __init__(self, cfg: PolicyCfg, goal_dim: int | None = None):
        super().__init__()
        goal_dim = goal_dim or cfg.goal_dim
        self.mlp = MLP(goal_dim, cfg.encoder_hidden, cfg.latent_dim,
                       layer_norm=cfg.hidden_layer_norm, act=_hidden_activation(cfg.hidden_activation))
        self.register_buffer("scale", _goal_scale_vector(cfg, goal_dim))
        self.normalizer = RunningNormalizer(goal_dim) if cfg.running_normalization else None

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        inputs = self.normalizer(goal) if self.normalizer is not None else goal * self.scale
        return torch.tanh(self.mlp(inputs))


class ActorObservationNormalizer(nn.Module):
    """Share raw sensor moments over history lags; keep future orientation separate."""

    def __init__(self, cfg: PolicyCfg):
        super().__init__()
        if cfg.history is None:
            raise ValueError("Resolve actor history before constructing normalization")
        self.history = cfg.history
        self.terms = observation_terms(cfg.num_joints, cfg.observation_layout)
        self.orientation_width = 6 * cfg.future_frames if cfg.observation_layout == "isaac_khan" else 0
        self.sensor = RunningNormalizer(sum(width for _, width in self.terms))
        self.orientation = RunningNormalizer(self.orientation_width) if self.orientation_width else None
        # The Khan environment supplies already scaled sensor terms. Restore
        # physical units before empirical normalization, including eps units.
        self.raw_scales = tuple(
            2.0 if cfg.observation_layout == "isaac_khan" and name == "base_ang_vel" else
            5.0 if cfg.observation_layout == "isaac_khan" and name == "joint_vel" else 1.0
            for name, _ in self.terms
        )

    def _terms(self, obs_hist: torch.Tensor):
        offset = self.orientation_width
        for (_, width), raw_scale in zip(self.terms, self.raw_scales, strict=True):
            term = obs_hist[..., offset:offset + self.history * width]
            yield term.unflatten(-1, (self.history, width)) * raw_scale
            offset += self.history * width

    def forward(self, obs_hist: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.orientation is not None:
            parts.append(self.orientation(obs_hist[..., :self.orientation_width]))
        offset = 0
        for (_, width), term in zip(self.terms, self._terms(obs_hist), strict=True):
            normalized = (term - self.sensor._mean[offset:offset + width]) / (
                self.sensor._std[offset:offset + width] + self.sensor.eps
            )
            parts.append(normalized.flatten(-2, -1))
            offset += width
        return torch.cat(parts, dim=-1)

    @torch.no_grad()
    def update(self, obs_hist: torch.Tensor) -> None:
        # Only the newest sensor is fresh. Updating all history slots would
        # repeatedly count older observations and weight reset padding H times.
        self.sensor.update(torch.cat([term[..., -1, :] for term in self._terms(obs_hist)], dim=-1))
        if self.orientation is not None:
            self.orientation.update(obs_hist[..., :self.orientation_width])


class Decoder(nn.Module):
    def __init__(self, cfg: PolicyCfg, obs_dim: int | None = None):
        super().__init__()
        if cfg.history is None:
            raise ValueError("Resolve the policy observation history before constructing a network")
        obs_dim = policy_observation_dimension(
            cfg.observation_layout, cfg.num_joints, cfg.history, cfg.future_frames,
        ) if obs_dim is None else obs_dim
        self.mlp = MLP(cfg.latent_dim + obs_dim, cfg.decoder_hidden, cfg.num_joints,
                       last_gain=cfg.decoder_last_gain, layer_norm=cfg.hidden_layer_norm,
                       act=_hidden_activation(cfg.hidden_activation))
        self.register_buffer("scale", _obs_scale_vector(cfg))
        self.observation_normalizer = ActorObservationNormalizer(cfg) if cfg.running_normalization else None

    def forward(self, z: torch.Tensor, obs_hist: torch.Tensor) -> torch.Tensor:
        obs = (self.observation_normalizer(obs_hist) if self.observation_normalizer is not None
               else obs_hist * self.scale)
        return self.mlp(torch.cat([z, obs], dim=-1))


class Policy(nn.Module):
    """Return the complete default-relative action, including an optional reference skip.

    ``goal`` is always clean for reference composition. An explicitly supplied
    ``encoder_goal`` can be noisy without injecting that noise directly into
    the PD target. Forward itself is deterministic, including in train mode.
    """

    def __init__(self, cfg: PolicyCfg, *, default_joint_pos: torch.Tensor | None = None,
                 action_scale: torch.Tensor | None = None):
        super().__init__()
        if cfg.action_mode not in ("default", "reference_residual"):
            raise ValueError("Resolve policy action_mode before constructing a network")
        if cfg.goal_noise_z is None or cfg.goal_noise_joint_pos is None:
            raise ValueError("Resolve policy goal noise widths before constructing a network")
        if not math.isfinite(cfg.reference_residual_beta) or not 0 <= cfg.reference_residual_beta <= 1:
            raise ValueError("reference_residual_beta must be finite and in [0, 1]")
        self.cfg = cfg
        self.encoders = nn.ModuleDict({"motion": MotionEncoder(cfg)})
        self.decoder = Decoder(cfg)
        self.register_buffer("encoder_goal_noise_half", goal_noise_half(
            layout=cfg.observation_layout, num_joints=cfg.num_joints,
            future_frames=cfg.future_frames, goal_dim=cfg.goal_dim,
            z_half=cfg.goal_noise_z, joint_pos_half=cfg.goal_noise_joint_pos,
        ), persistent=False)
        self.has_encoder_goal_noise = bool(cfg.goal_noise_z or cfg.goal_noise_joint_pos)
        if cfg.action_mode == "reference_residual":
            # Validate the layout/frame once at construction, before a CUDA or
            # ONNX graph can be built with the wrong reference timing.
            if not isinstance(cfg.reference_frame_offset, int) or isinstance(cfg.reference_frame_offset, bool):
                raise ValueError("reference_frame_offset must be an integer")
            reference_joint_positions(
                torch.empty(cfg.goal_dim), layout=cfg.observation_layout,
                num_joints=cfg.num_joints, future_frames=cfg.future_frames,
                frame_offset=cfg.reference_frame_offset,
            )
            if default_joint_pos is None or action_scale is None:
                raise ValueError("Reference-residual policy requires robot default_joint_pos and action_scale")
            q0 = torch.as_tensor(default_joint_pos, dtype=torch.float32).detach().clone()
            alpha = torch.as_tensor(action_scale, dtype=torch.float32).detach().clone()
            if q0.shape != (cfg.num_joints,) or alpha.shape != (cfg.num_joints,):
                raise ValueError("Reference action defaults/scales must follow the policy joint order and width")
            if not torch.isfinite(q0).all() or not torch.isfinite(alpha).all() or (alpha <= 0).any():
                raise ValueError("Reference action defaults/scales must be finite, with positive scales")
            self.register_buffer("reference_q_default", q0)
            self.register_buffer("reference_action_scale", alpha)

    def make_encoder_goal(self, goal: torch.Tensor, enabled: bool = True) -> torch.Tensor:
        """Draw independent per-element augmentation without changing clean reference.

        Collection and imagination call this outside activation recomputation.
        Disabled/zero augmentation consumes no RNG and retains the legacy path.
        """
        if not enabled or not self.has_encoder_goal_noise:
            return goal
        return goal + (2 * torch.rand_like(goal) - 1) * self.encoder_goal_noise_half.to(goal)

    @torch.no_grad()
    def update_normalization(self, obs_hist: torch.Tensor, goal: torch.Tensor, *,
                             encoder_goal: torch.Tensor | None = None) -> None:
        """Update once from fresh real observations, reusing the actual noisy goal."""
        if not self.cfg.running_normalization:
            return
        self.decoder.observation_normalizer.update(obs_hist)
        self.encoders["motion"].normalizer.update(goal if encoder_goal is None else encoder_goal)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor, interface: str = "motion",
                *, encoder_goal: torch.Tensor | None = None) -> torch.Tensor:
        action = self.decoder(self.encoders[interface](goal if encoder_goal is None else encoder_goal), obs_hist)
        if self.cfg.action_mode == "reference_residual":
            q_ref = reference_joint_positions(
                goal, layout=self.cfg.observation_layout, num_joints=self.cfg.num_joints,
                future_frames=self.cfg.future_frames, frame_offset=self.cfg.reference_frame_offset,
            )
            action = action + self.cfg.reference_residual_beta * (
                q_ref - self.reference_q_default
            ) / self.reference_action_scale
        return action


class Critic(nn.Module):
    """V(state, reference, xi, root drift, tracking errors); G1 input788, Khan DR input605."""

    def __init__(self, cfg: CriticCfg, feat_dim: int | None = None, goal_dim: int | None = None, xi_dim: int | None = None, pcfg: PolicyCfg | None = None):
        super().__init__()
        pcfg = pcfg or PolicyCfg()
        feat_dim = 13 + 2 * pcfg.num_joints if feat_dim is None else feat_dim
        xi_dim = pcfg.num_joints + 4 if xi_dim is None else xi_dim
        goal_dim = critic_goal_dimension(pcfg) if goal_dim is None else goal_dim
        self.error_dim = critic_error_dimension(pcfg)
        self.mlp = MLP(feat_dim + goal_dim + xi_dim + 4 + self.error_dim, cfg.hidden, 1,
                       layer_norm=cfg.hidden_layer_norm, act=_hidden_activation(cfg.hidden_activation))
        self.register_buffer("goal_scale", _critic_goal_scale_vector(pcfg, goal_dim))

    def forward(self, feat: torch.Tensor, goal: torch.Tensor, xi_norm: torch.Tensor,
                e_root: torch.Tensor, tracking_errors: torch.Tensor | None = None) -> torch.Tensor:
        parts = [feat, goal * self.goal_scale, xi_norm, e_root]
        if self.error_dim:
            if tracking_errors is None or tracking_errors.shape[-1] != self.error_dim:
                raise ValueError(f"Khan critic requires {self.error_dim} current tracking error values")
            parts.append(tracking_errors)
        elif tracking_errors is not None:
            raise ValueError("SuperMimic critic does not use extra tracking errors")
        return self.mlp(torch.cat(parts, dim=-1)).squeeze(-1)

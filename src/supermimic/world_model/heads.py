"""Analytic heads of the world model (plan D4): observation, reward, termination.

Everything here is a pure differentiable function of the simulator state
``s = [qpos | qvel]`` (71-d for G1, 67-d for Khan; root relative to the env origin,
see ``StateIO``), the previous clean action, the sim context ``xi`` and the
reference frame, so the same code serves the simulator-side tests and the
imagined rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.envs.mdp.rewards import NECK_JOINT_NAMES, TrackingRewardCfg, tracking_reward_terms
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.kinematics.rotations import projected_gravity, quat_apply_inverse
from supermimic.world_model.state_spec import StateSpec

OBS_DIM = 93
# Term layout of the ``policy`` observation group (order matters: term-major history).
OBS_TERMS = (("joint_pos", 29), ("joint_vel", 29), ("base_ang_vel", 3), ("projected_gravity", 3), ("actions", 29))


def observation_terms(num_joints: int = 29) -> tuple[tuple[str, int], ...]:
    """Paper Table A.1 term layout, parameterised only by robot joint count."""
    return (("joint_pos", num_joints), ("joint_vel", num_joints), ("base_ang_vel", 3),
            ("projected_gravity", 3), ("actions", num_joints))


def split_state(s: torch.Tensor):
    """-> root pose, joint positions, root velocities, joint velocities."""
    return StateSpec.split(s)


def analytic_observation(
    s: torch.Tensor, prev_action: torch.Tensor, xi: torch.Tensor, default_joint_pos: torch.Tensor
) -> torch.Tensor:
    """Noise-free proprioception (paper Table A.1, width 3*num_joints+6).

    The joint-sized slice of xi after friction is the encoder bias:
    observed joint positions are ``q + bias - q0``.
    """
    _, root_quat, q, _, ang_b, qd = split_state(s)
    bias = xi[..., 1:1 + default_joint_pos.shape[-1]]
    return torch.cat([q + bias - default_joint_pos, qd, ang_b, projected_gravity(root_quat), prev_action], dim=-1)


def anchor_relative_obs(robot_anchor_pos: torch.Tensor, robot_anchor_quat: torch.Tensor, ref_anchor_pos: torch.Tensor, ref_anchor_quat: torch.Tensor) -> torch.Tensor:
    """Reference anchor pose expressed in the robot anchor frame: [pos (3), rot6d (6)] (= mjlab's
    motion_anchor_pos_b / motion_anchor_ori_b, plan D5 alternative)."""
    from supermimic.kinematics.rotations import matrix_from_quat, quat_inv, quat_mul

    q10 = quat_inv(robot_anchor_quat)
    pos = quat_apply_inverse(robot_anchor_quat, ref_anchor_pos - robot_anchor_pos)
    q12 = quat_mul(q10, ref_anchor_quat)
    mat = matrix_from_quat(q12)
    return torch.cat([pos, mat[..., :2].reshape(*q12.shape[:-1], 6)], dim=-1)


ANCHOR_OBS_DIM = 9


def flatten_history_term_major(hist: torch.Tensor, num_joints: int | None = None) -> torch.Tensor:
    """Flatten oldest→newest history in mjlab's term-major layout.

    G1 uses (H, N, 93) -> (N, H*93); Khan uses (H, N, 87) -> (N, H*87).
    """
    H, N, width = hist.shape
    if num_joints is None:
        if width < 9 or (width - 6) % 3:
            raise ValueError(f"Invalid proprioceptive observation width: {width}")
        num_joints = (width - 6) // 3
    if width != 3 * num_joints + 6:
        raise ValueError(f"Observation width {width} does not match {num_joints} joints")
    chunks = []
    start = 0
    for _, d in observation_terms(num_joints):
        chunks.append(hist[:, :, start : start + d].permute(1, 0, 2).reshape(N, H * d))
        start += d
    return torch.cat(chunks, dim=-1)


@dataclass
class ReferenceFrame:
    """Reference quantities of one target frame for a batch (N, ·)."""

    body_pos_w: torch.Tensor  # (N, B, 3)
    body_quat_w: torch.Tensor  # (N, B, 4)
    body_lin_vel_w: torch.Tensor  # (N, B, 3)
    body_ang_vel_w: torch.Tensor  # (N, B, 3)
    root_pos_w: torch.Tensor  # (N, 3)
    root_quat_w: torch.Tensor  # (N, 4)
    joint_pos: torch.Tensor | None = None  # (N, J), required for enabled neck tracking
    joint_vel: torch.Tensor | None = None  # (N, J)


class AnalyticHeads:
    """Reward / termination features computed from the predicted state through differentiable FK."""

    def __init__(
        self,
        fk: G1Kinematics,
        tracked_body_names: tuple[str, ...],
        anchor_body_name: str,
        root_body_name: str,
        soft_limits: torch.Tensor,
        default_joint_pos: torch.Tensor,
        reward_cfg: TrackingRewardCfg | None = None,
    ):
        self.fk = fk
        self.num_joints = fk.nq - 7
        if default_joint_pos.shape != (self.num_joints,) or soft_limits.shape != (self.num_joints, 2):
            raise ValueError("Joint defaults and soft limits must match the FK model")
        self.tracked_idx = fk.body_indices(tracked_body_names)
        self.anchor_index = tracked_body_names.index(anchor_body_name)
        self.root_fk_index = fk.body_index(root_body_name)
        self.soft_limits = soft_limits
        self.default_joint_pos = default_joint_pos
        joint_names = fk.joint_names[1:]  # The first joint is the free root, not an action.
        self.neck_joint_ids = [i for i, name in enumerate(joint_names) if name in NECK_JOINT_NAMES]
        self.reward_cfg = reward_cfg or TrackingRewardCfg.for_joint_names(joint_names)
        if (self.reward_cfg.w_neck_pos or self.reward_cfg.w_neck_vel) and not self.neck_joint_ids:
            raise ValueError("Neck tracking reward requires neck joints in the FK model")

    def robot_bodies(self, s: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.fk(s[..., :self.fk.nq], s[..., self.fk.nq:])
        idx = self.tracked_idx
        return {
            "pos": out["pos"][..., idx, :],
            "quat": out["quat"][..., idx, :],
            "lin_vel": out["lin_vel"][..., idx, :],
            "ang_vel": out["ang_vel"][..., idx, :],
            "root_pos": out["pos"][..., self.root_fk_index, :],
            "root_quat": out["quat"][..., self.root_fk_index, :],
        }

    def reward(self, s_next: torch.Tensor, action: torch.Tensor, prev_action: torch.Tensor, ref: ReferenceFrame, bodies=None):
        """Reward r_{t+1} of reaching s_next (= s_{t+1}) under target frame g_t. Returns dict with 'total'."""
        b = bodies if bodies is not None else self.robot_bodies(s_next)
        return tracking_reward_terms(
            robot_body_pos_w=b["pos"],
            robot_body_quat_w=b["quat"],
            robot_body_lin_vel_w=b["lin_vel"],
            robot_body_ang_vel_w=b["ang_vel"],
            ref_body_pos_w=ref.body_pos_w,
            ref_body_quat_w=ref.body_quat_w,
            ref_body_lin_vel_w=ref.body_lin_vel_w,
            ref_body_ang_vel_w=ref.body_ang_vel_w,
            anchor_index=self.anchor_index,
            action=action,
            prev_action=prev_action,
            joint_pos=s_next[..., 7:self.fk.nq],
            soft_limits=self.soft_limits,
            cfg=self.reward_cfg,
            joint_vel=s_next[..., self.fk.nq + 6:],
            ref_joint_pos=ref.joint_pos,
            ref_joint_vel=ref.joint_vel,
            neck_joint_ids=self.neck_joint_ids,
        )

    @staticmethod
    def termination_features(root_pos: torch.Tensor, root_quat: torch.Tensor, ref: ReferenceFrame) -> torch.Tensor:
        """[height error, tilt error] (N, 2) — the quantities the paper thresholds (0.25 m, 0.8)."""
        g = torch.tensor([0.0, 0.0, -1.0], dtype=root_quat.dtype, device=root_quat.device).expand(root_quat.shape[:-1] + (3,))
        tilt = quat_apply_inverse(ref.root_quat_w, g)[..., 2] - quat_apply_inverse(root_quat, g)[..., 2]
        height = ref.root_pos_w[..., 2] - root_pos[..., 2]
        return torch.stack([height, tilt], dim=-1)

    def terminated(self, s: torch.Tensor, ref: ReferenceFrame, height_thr: float = 0.25, tilt_thr: float = 0.8) -> torch.Tensor:
        f = self.termination_features(s[..., 0:3], s[..., 3:7], ref)  # pelvis == free-joint root
        return (f[..., 0].abs() > height_thr) | (f[..., 1].abs() > tilt_thr)

    @staticmethod
    def root_error_features(s: torch.Tensor, ref: ReferenceFrame) -> torch.Tensor:
        """Critic-only privileged drift features (plan D10): root position error in the reference
        heading frame (3) and yaw error (1). (N, 4)"""
        from supermimic.kinematics.rotations import (
            heading_yaw,
            quat_apply_inverse,
            wrap_to_pi,
            yaw_quat,
        )

        dp = quat_apply_inverse(yaw_quat(ref.root_quat_w), s[..., 0:3] - ref.root_pos_w)
        dyaw = wrap_to_pi(heading_yaw(s[..., 3:7]) - heading_yaw(ref.root_quat_w))
        return torch.cat([dp, dyaw[..., None]], dim=-1)

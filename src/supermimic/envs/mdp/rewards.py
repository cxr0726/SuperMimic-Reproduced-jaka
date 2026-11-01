"""Analytic BeyondMimic tracking reward in torch (plan D4/D7).

One differentiable implementation shared by the simulator side (tests / logging)
and the imagination side (world-model reward head). Mirrors mjlab's
``tasks/tracking/mdp/rewards.py`` term by term, including the re-anchoring of the
reference bodies to the robot anchor in xy + yaw only
(``MotionCommand.update_relative_body_poses``).

Inputs are world-frame link quantities of the *tracked* bodies (same ordering as
``TRACKED_BODIES``), the anchor being one of them (``anchor_index``).
Rewards are returned already weighted and multiplied by ``dt`` like mjlab's reward
manager (``scale_rewards_by_dt=True``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_box_minus, quat_inv, quat_mul, yaw_quat

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.scene_entity_config import SceneEntityCfg

NECK_JOINT_NAMES = ("Neck_yaw_joint", "Neck_pitch_joint")


@dataclass
class TrackingRewardCfg:
    std_body_pos: float = 0.3
    std_body_ori: float = 0.4
    std_body_lin_vel: float = 1.0
    std_body_ang_vel: float = 3.14
    std_root_pos: float = 0.3
    std_root_ori: float = 0.4
    w_body_pos: float = 1.0
    w_body_ori: float = 1.0
    w_body_lin_vel: float = 1.0
    w_body_ang_vel: float = 1.0
    w_root_pos: float = 0.5
    w_root_ori: float = 0.5
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    std_neck_pos: float = 0.3
    std_neck_vel: float = 1.0
    w_neck_pos: float = 0.0
    w_neck_vel: float = 0.0
    dt: float = 0.02

    @classmethod
    def for_joint_names(cls, joint_names: Sequence[str]) -> TrackingRewardCfg:
        """Enable reference neck tracking for Khan while retaining the G1 reward."""
        if set(NECK_JOINT_NAMES).issubset(joint_names):
            return cls(w_neck_pos=0.5, w_neck_vel=0.1)
        return cls()

    @property
    def max_step_reward(self) -> float:
        return self.dt * (self.w_body_pos + self.w_body_ori + self.w_body_lin_vel + self.w_body_ang_vel + self.w_root_pos + self.w_root_ori + self.w_neck_pos + self.w_neck_vel)


def joint_tracking_error_exp(
    actual: torch.Tensor, reference: torch.Tensor, std: float,
) -> torch.Tensor:
    """Unweighted tracking reward, averaging squared error over selected joints."""
    if std <= 0 or actual.shape[-1] == 0:
        raise ValueError("Joint tracking requires positive std and at least one joint")
    return torch.exp(-(actual - reference).square().mean(-1) / std**2)


def motion_neck_position_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Track reference neck angles using the resolved entity joint selection."""
    command = env.command_manager.get_term(command_name)
    robot = env.scene[asset_cfg.name]
    return joint_tracking_error_exp(
        robot.data.joint_pos[:, asset_cfg.joint_ids],
        command.joint_pos[:, asset_cfg.joint_ids], std,
    )


def motion_neck_velocity_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Track reference neck velocities using the same core as imagined rewards."""
    command = env.command_manager.get_term(command_name)
    robot = env.scene[asset_cfg.name]
    return joint_tracking_error_exp(
        robot.data.joint_vel[:, asset_cfg.joint_ids],
        command.joint_vel[:, asset_cfg.joint_ids], std,
    )


def reanchor_reference(
    ref_body_pos_w: torch.Tensor,
    ref_body_quat_w: torch.Tensor,
    ref_anchor_pos_w: torch.Tensor,
    ref_anchor_quat_w: torch.Tensor,
    robot_anchor_pos_w: torch.Tensor,
    robot_anchor_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Express the reference bodies in a frame that shares the robot anchor's xy position and yaw.

    Exactly mjlab's ``update_relative_body_poses``: z, roll and pitch stay those of the reference.
    Shapes: bodies (N, B, ·), anchors (N, ·).
    """
    nb = ref_body_pos_w.shape[-2]
    delta_pos = robot_anchor_pos_w.clone()
    delta_pos = torch.cat([delta_pos[..., :2], ref_anchor_pos_w[..., 2:3]], dim=-1)  # (N,3)
    delta_ori = yaw_quat(quat_mul(robot_anchor_quat_w, quat_inv(ref_anchor_quat_w)))  # (N,4)
    delta_ori_b = delta_ori[..., None, :].expand(*delta_ori.shape[:-1], nb, 4)
    rel_quat = quat_mul(delta_ori_b, ref_body_quat_w)
    rel_pos = delta_pos[..., None, :] + quat_apply(delta_ori_b, ref_body_pos_w - ref_anchor_pos_w[..., None, :])
    return rel_pos, rel_quat


def _sq_quat_err(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Squared geodesic angle between quaternions, smooth at zero (== quat_error_magnitude**2)."""
    v = quat_box_minus(q1, q2)
    return (v * v).sum(-1)


def tracking_reward_terms(
    *,
    robot_body_pos_w: torch.Tensor,
    robot_body_quat_w: torch.Tensor,
    robot_body_lin_vel_w: torch.Tensor,
    robot_body_ang_vel_w: torch.Tensor,
    ref_body_pos_w: torch.Tensor,
    ref_body_quat_w: torch.Tensor,
    ref_body_lin_vel_w: torch.Tensor,
    ref_body_ang_vel_w: torch.Tensor,
    anchor_index: int,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    joint_pos: torch.Tensor,
    soft_limits: torch.Tensor,  # (29, 2) or (N, 29, 2)
    cfg: TrackingRewardCfg,
    joint_vel: torch.Tensor | None = None,
    ref_joint_pos: torch.Tensor | None = None,
    ref_joint_vel: torch.Tensor | None = None,
    neck_joint_ids: Sequence[int] = (),
) -> dict[str, torch.Tensor]:
    """Per-term weighted, dt-scaled rewards (N,) plus 'total'."""
    ra_pos = robot_body_pos_w[..., anchor_index, :]
    ra_quat = robot_body_quat_w[..., anchor_index, :]
    ma_pos = ref_body_pos_w[..., anchor_index, :]
    ma_quat = ref_body_quat_w[..., anchor_index, :]
    rel_pos, rel_quat = reanchor_reference(ref_body_pos_w, ref_body_quat_w, ma_pos, ma_quat, ra_pos, ra_quat)

    e_root_pos = ((ma_pos - ra_pos) ** 2).sum(-1)
    e_root_ori = _sq_quat_err(ma_quat, ra_quat)
    e_body_pos = ((rel_pos - robot_body_pos_w) ** 2).sum(-1).mean(-1)
    e_body_ori = _sq_quat_err(rel_quat, robot_body_quat_w).mean(-1)
    e_body_lin = ((ref_body_lin_vel_w - robot_body_lin_vel_w) ** 2).sum(-1).mean(-1)
    e_body_ang = ((ref_body_ang_vel_w - robot_body_ang_vel_w) ** 2).sum(-1).mean(-1)

    lo = soft_limits[..., 0]
    hi = soft_limits[..., 1]
    out = (-(joint_pos - lo)).clamp(min=0.0) + (joint_pos - hi).clamp(min=0.0)
    joint_limit = out.sum(-1)
    action_rate = ((action - prev_action) ** 2).sum(-1)

    dt = cfg.dt
    terms = {
        "motion_global_root_pos": dt * cfg.w_root_pos * torch.exp(-e_root_pos / cfg.std_root_pos**2),
        "motion_global_root_ori": dt * cfg.w_root_ori * torch.exp(-e_root_ori / cfg.std_root_ori**2),
        "motion_body_pos": dt * cfg.w_body_pos * torch.exp(-e_body_pos / cfg.std_body_pos**2),
        "motion_body_ori": dt * cfg.w_body_ori * torch.exp(-e_body_ori / cfg.std_body_ori**2),
        "motion_body_lin_vel": dt * cfg.w_body_lin_vel * torch.exp(-e_body_lin / cfg.std_body_lin_vel**2),
        "motion_body_ang_vel": dt * cfg.w_body_ang_vel * torch.exp(-e_body_ang / cfg.std_body_ang_vel**2),
        "action_rate_l2": dt * cfg.w_action_rate * action_rate,
        "joint_limit": dt * cfg.w_joint_limit * joint_limit,
    }
    if cfg.w_neck_pos:
        if not neck_joint_ids or ref_joint_pos is None:
            raise ValueError("Neck position reward requires neck joint IDs and reference positions")
        terms["motion_neck_pos"] = dt * cfg.w_neck_pos * joint_tracking_error_exp(
            joint_pos[..., neck_joint_ids], ref_joint_pos[..., neck_joint_ids], cfg.std_neck_pos,
        )
    if cfg.w_neck_vel:
        if not neck_joint_ids or joint_vel is None or ref_joint_vel is None:
            raise ValueError("Neck velocity reward requires joint velocities and reference velocities")
        terms["motion_neck_vel"] = dt * cfg.w_neck_vel * joint_tracking_error_exp(
            joint_vel[..., neck_joint_ids], ref_joint_vel[..., neck_joint_ids], cfg.std_neck_vel,
        )
    total = terms["motion_global_root_pos"]
    for k, v in terms.items():
        if k != "motion_global_root_pos":
            total = total + v
    terms["total"] = total
    return terms

"""Observation layouts shared by simulation, imagination and policy construction.

The Khan layout retains ProjGravObservationsCfg features and five-frame windows.
Actor velocity scales and the critic's reference velocity features follow
SuperMimic. Reference windows include the current reference frame.
"""

from __future__ import annotations

import math
from typing import Literal

import torch

from supermimic.kinematics.rotations import (
    heading_quat,
    matrix_from_quat,
    projected_gravity,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
)

ObservationLayout = Literal["supermimic", "isaac_khan"]
ObservationLayoutOption = Literal["auto", "supermimic", "isaac_khan"]


def resolve_observation_layout(robot: str, layout: str = "auto") -> ObservationLayout:
    if layout == "auto":
        return "isaac_khan" if robot == "khan_mini" else "supermimic"
    if layout not in ("supermimic", "isaac_khan"):
        raise ValueError(f"Unknown observation layout: {layout}")
    if layout == "isaac_khan" and robot != "khan_mini":
        raise ValueError("The isaac_khan observation layout requires Khan_mini")
    return layout


def resolve_observation_sizes(
    layout: str, future_frames: int | None = None, history: int | None = None,
    anchor_obs: bool = False,
) -> tuple[int, int]:
    default = 5 if layout == "isaac_khan" else 10
    future_frames = default if future_frames is None else future_frames
    history = default if history is None else history
    if future_frames < 1 or history < 1:
        raise ValueError("Observation reference window and history must be positive")
    if layout == "isaac_khan":
        if future_frames != 5 or history != 5:
            raise ValueError("isaac_khan requires exactly 5 reference frames and 5 history frames")
        if anchor_obs:
            raise ValueError("isaac_khan already includes future anchor orientation in proprioception; "
                             "the legacy anchor_obs goal extension must be disabled")
    elif layout != "supermimic":
        raise ValueError(f"Unknown observation layout: {layout}")
    return future_frames, history


def goal_dimension(
    layout: str, num_joints: int, future_frames: int, anchor_obs: bool = False,
) -> int:
    if layout == "isaac_khan":
        if anchor_obs:
            raise ValueError("isaac_khan has no legacy anchor_obs goal extension")
        return future_frames * (num_joints + 4)
    return future_frames * (2 * num_joints + 10) + (9 if anchor_obs else 0)


def policy_observation_dimension(
    layout: str, num_joints: int, history: int, future_frames: int,
) -> int:
    return history * (3 * num_joints + 6) + (6 * future_frames if layout == "isaac_khan" else 0)


def isaac_goal_command(
    root_pos: torch.Tensor, root_quat: torch.Tensor, joint_pos: torch.Tensor,
) -> torch.Tensor:
    """[all reference-frame displacements, all world heights, all absolute q].

    Inputs have shape (N,K,3/4/J). The origin and full orientation belong to
    reference frame zero, independently of the real robot's tracking error.
    """
    origin = root_pos[:, :1]
    orientation = root_quat[:, :1].expand_as(root_quat)
    delta = quat_apply_inverse(orientation, root_pos - origin)
    return torch.cat([delta.flatten(1), root_pos[..., 2].flatten(1), joint_pos.flatten(1)], -1)


def reference_joint_positions(
    goal: torch.Tensor, *, layout: str, num_joints: int, future_frames: int,
    frame_offset: int,
) -> torch.Tensor:
    """Extract absolute reference radians from a clean term-major Khan goal.

    Khan's window starts at the current frame, unlike InstantTrack's f+1
    window. Offset 1 therefore selects the second q block (47:74 for J=27,
    K=5). Tail padding remains the motion library's responsibility.
    """
    if layout != "isaac_khan":
        raise ValueError("Reference-residual actions require the isaac_khan goal layout")
    if not 0 <= frame_offset < future_frames:
        raise ValueError("Reference action frame_offset must lie inside the goal window")
    if goal.shape[-1] != future_frames * (num_joints + 4):
        raise ValueError("Reference action goal width differs from its configured layout")
    start = 4 * future_frames + frame_offset * num_joints
    return goal[..., start:start + num_joints]


def goal_noise_half(
    *, layout: str, num_joints: int, future_frames: int, goal_dim: int,
    z_half: float, joint_pos_half: float,
) -> torch.Tensor:
    """Actor-encoder uniform half-widths in raw goal units, never PD-target noise.

    Isaac Khan: [displacement 3K (clean), world height K, absolute joints KJ].
    Other layouts retain the legacy clean-goal behavior.
    """
    if any(not math.isfinite(x) or x < 0 for x in (z_half, joint_pos_half)):
        raise ValueError("Goal noise half-widths must be finite and nonnegative")
    if layout != "isaac_khan":
        if z_half or joint_pos_half:
            raise ValueError("Goal encoder noise is only supported for isaac_khan")
        return torch.zeros(goal_dim)
    if goal_dim != future_frames * (num_joints + 4):
        raise ValueError("Goal noise width differs from the isaac_khan goal layout")
    return torch.cat([
        torch.zeros(3 * future_frames), torch.full((future_frames,), float(z_half)),
        torch.full((future_frames * num_joints,), float(joint_pos_half)),
    ])


def isaac_anchor_orientation(robot_quat: torch.Tensor, reference_quat: torch.Tensor) -> torch.Tensor:
    """Future reference rotations in the current robot base frame, (N,6*K).

    Each 3x2 matrix is flattened by rows: [R00,R01,R10,R11,R20,R21], exactly as
    Isaac's ``matrix_from_quat(ori_b)[..., :2].reshape(N,-1)``.
    """
    robot_quat = robot_quat[:, None, :].expand_as(reference_quat)
    relative = quat_mul(quat_inv(robot_quat), reference_quat)
    return matrix_from_quat(relative)[..., :2].flatten(1)


def critic_reference_extras(
    joint_vel: torch.Tensor, root_quat: torch.Tensor,
    root_ang_vel_w: torch.Tensor, root_lin_vel_w: torch.Tensor,
) -> torch.Tensor:
    """Clean reference velocities/gravity, flattened as K successive frames.

    Each frame contains [joint velocity (J), projected gravity (3), root angular
    velocity in that reference root's full frame (3), root linear velocity in
    that reference root's heading frame (3)]. No actor noise or robot-state
    dependence is introduced. The critic applies the original SuperMimic fixed
    scales (divide joint velocity by 5 and angular velocity by 2).
    """
    gravity = projected_gravity(root_quat)
    angular = quat_apply_inverse(root_quat, root_ang_vel_w)
    linear = quat_apply_inverse(heading_quat(root_quat), root_lin_vel_w)
    return torch.cat([joint_vel, gravity, angular, linear], dim=-1).flatten(1)

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from supermimic.envs.mdp.commands import MultiClipMotionCommand
from supermimic.observation_layout import isaac_anchor_orientation
from supermimic.observation_layout import isaac_goal_command as make_isaac_goal

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def goal_future_frames(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """(N, K*68) future reference-frame features (plan D5)."""
    command = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return command.goal


def isaac_goal_command(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Isaac command: reference-relative displacement, absolute z and absolute q."""
    command = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    reference = command.library.reference_window(command.frame, command.cfg.future_frames)
    world_pos = reference.root_pos_w + env.scene.env_origins[:, None, :]
    return make_isaac_goal(world_pos, reference.root_quat_w, reference.joint_pos)


def isaac_motion_anchor_orientation(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Five reference orientations relative to the robot's current base orientation."""
    command = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    reference = command.library.reference_window(command.frame, command.cfg.future_frames)
    return isaac_anchor_orientation(command.robot_anchor_quat_w, reference.root_quat_w)

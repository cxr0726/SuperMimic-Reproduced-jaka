"""Checkpoint compatibility for observation, reward and robot-physics semantics."""

from __future__ import annotations

import hashlib
from dataclasses import asdict

from supermimic.envs.mdp.rewards import TrackingRewardCfg
from supermimic.robots.registry import RobotName, get_robot_spec


def robot_contract(robot: RobotName) -> dict:
    spec = get_robot_spec(robot)
    contract = {
        "version": 1,
        "robot": robot,
        "root_body": spec.root_body,
        "anchor_body": spec.anchor_body,
        "com_body": spec.com_body,
        "tracked_bodies": list(spec.tracked_bodies),
        "joint_names": list(spec.joint_names),
        "reward_config": asdict(TrackingRewardCfg.for_joint_names(spec.joint_names)),
        "action_scale": spec.action_scale,
    }
    if robot == "khan_mini":
        from supermimic.robots.khan_mini import DEFAULT_ASSET_DIR

        # Neither equal tensor widths nor matching body names prove equal physics.
        contract["model_sha256"] = hashlib.sha256((DEFAULT_ASSET_DIR / "robot.xml").read_bytes()).hexdigest()
        contract["profile_sha256"] = hashlib.sha256((DEFAULT_ASSET_DIR / "profile.json").read_bytes()).hexdigest()
    return contract


def validate_robot_contract(checkpoint: dict, robot: RobotName) -> None:
    saved = checkpoint.get("robot_contract")
    if saved is None:
        if robot == "g1":
            return  # Historical G1 checkpoints retain the original training semantics.
        raise ValueError(
            "Legacy Khan checkpoint has no robot_contract: its 14-body/waist-anchor "
            "training semantics cannot be assumed compatible with the current "
            "15-body/base-anchor/neck-reward model. Start a new run; evaluate old "
            "policies with scripts/analysis/eval_legacy_khan_benchmark.py, which exports "
            "the matching legacy source and assets without changing this checkout."
        )
    expected = robot_contract(robot)
    differences = sorted(key for key in set(saved) | set(expected) if saved.get(key) != expected.get(key))
    if differences:
        raise ValueError("Checkpoint robot_contract differs from current model: " + ", ".join(differences))

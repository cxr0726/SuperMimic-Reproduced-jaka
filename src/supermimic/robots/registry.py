"""Robot-specific assets and names shared by simulation and learned training.

All algorithms use the same observation terms, rewards and randomization; only
these asset bindings and their resulting tensor dimensions depend on the robot.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

RobotName = Literal["g1", "khan_mini"]

G1_TRACKED_BODIES = (
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)


@dataclass(frozen=True)
class RobotSpec:
    name: RobotName
    joint_names: tuple[str, ...]
    root_body: str
    anchor_body: str
    tracked_bodies: tuple[str, ...]
    foot_geom_regex: str
    default_npz_dir: Path
    nconmax: int = 35
    njmax: int = 250

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def state_dim(self) -> int:
        return 13 + 2 * self.num_joints

    @property
    def goal_feature_dim(self) -> int:
        return 2 * self.num_joints + 10

    def load_model(self):
        if self.name == "g1":
            from supermimic.data.motion_convert import load_g1_model

            return load_g1_model()
        from supermimic.robots.khan_mini import get_khan_mini_spec

        return get_khan_mini_spec().compile()

    def entity_cfg(self):
        if self.name == "g1":
            from mjlab.asset_zoo.robots import get_g1_robot_cfg

            return get_g1_robot_cfg()
        from supermimic.robots.khan_mini import get_khan_mini_cfg

        return get_khan_mini_cfg()

    @property
    def action_scale(self) -> dict[str, float]:
        if self.name == "g1":
            from mjlab.asset_zoo.robots import G1_ACTION_SCALE

            return dict(G1_ACTION_SCALE)
        from supermimic.robots.khan_mini import load_profile

        return dict(load_profile()["action_scale"])


@lru_cache(maxsize=2)
def get_robot_spec(name: RobotName = "g1") -> RobotSpec:
    if name == "g1":
        from supermimic.data.motion_convert import G1_JOINT_NAMES

        return RobotSpec(
            name=name, joint_names=tuple(G1_JOINT_NAMES), root_body="pelvis",
            anchor_body="torso_link", tracked_bodies=G1_TRACKED_BODIES,
            foot_geom_regex=r"^(left|right)_foot[1-7]_collision$",
            default_npz_dir=Path("data/npz/lafan1"),
        )
    if name == "khan_mini":
        from supermimic.robots.khan_mini import load_profile

        profile = load_profile()
        return RobotSpec(
            name=name, joint_names=tuple(profile["joint_names"]),
            root_body=profile["root_body"], anchor_body=profile["anchor_body"],
            tracked_bodies=tuple(profile["tracked_bodies"]),
            foot_geom_regex=profile["foot_geom_regex"],
            default_npz_dir=Path("data/npz/khan_mini/lafan1"),
            # Conservative allocation headroom, not a measured minimum.
            # Retain the original solver and timestep.
            nconmax=128, njmax=512,
        )
    raise ValueError(f"Unknown robot {name!r}; expected g1 or khan_mini")

"""Robot-specific assets and names shared by simulation and learned training.

Optimization is shared. Assets, tracked bodies, COM randomization and optional
neck rewards depend on the robot; observation_layout resolves the input schema.
"""

from __future__ import annotations

import math
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
    com_body: str
    tracked_bodies: tuple[str, ...]
    foot_geom_regex: str
    default_npz_dir: Path
    neck_joint_names: tuple[str, ...] = ()
    nconmax: int = 35
    njmax: int = 250
    default_actuator_delay: bool = False
    default_reset_root_z_offset: float = 0.0

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def state_dim(self) -> int:
        return 13 + 2 * self.num_joints

    @property
    def goal_feature_dim(self) -> int:
        """Legacy SuperMimic frame width; Isaac dimensions use goal_dimension()."""
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
            anchor_body="torso_link", com_body="torso_link", tracked_bodies=G1_TRACKED_BODIES,
            foot_geom_regex=r"^(left|right)_foot[1-7]_collision$",
            default_npz_dir=Path("data/npz/lafan1"),
        )
    if name == "khan_mini":
        from supermimic.robots.khan_mini import load_profile

        profile = load_profile()
        return RobotSpec(
            name=name, joint_names=tuple(profile["joint_names"]),
            root_body=profile["root_body"], anchor_body=profile["anchor_body"],
            com_body=profile["com_body"],
            tracked_bodies=tuple(profile["tracked_bodies"]),
            foot_geom_regex=profile["foot_geom_regex"],
            default_npz_dir=Path("data/npz/khan_mini/lafan1"),
            neck_joint_names=tuple(profile["neck_joint_names"]),
            # Conservative allocation headroom, not a measured minimum.
            # Retain the original solver and timestep.
            nconmax=128, njmax=512,
            default_actuator_delay=True,
            default_reset_root_z_offset=0.05,
        )
    raise ValueError(f"Unknown robot {name!r}; expected g1 or khan_mini")


def resolve_robot_environment(
    robot: RobotName, *, actuator_delay: bool | None = None,
    actuator_delay_max_ms: float = 20.0, reset_root_z_offset: float | None = None,
) -> dict:
    """Resolve saved simulation settings without changing G1's original defaults."""
    spec = get_robot_spec(robot)
    delay = spec.default_actuator_delay if actuator_delay is None else actuator_delay
    offset = spec.default_reset_root_z_offset if reset_root_z_offset is None else reset_root_z_offset
    if (not math.isfinite(actuator_delay_max_ms) or actuator_delay_max_ms < 0
            or not math.isclose(actuator_delay_max_ms / 5, round(actuator_delay_max_ms / 5))):
        raise ValueError("Actuator delay must be a nonnegative multiple of the 5 ms physics step")
    if not math.isfinite(offset) or offset < 0:
        raise ValueError("reset_root_z_offset must be finite and nonnegative")
    return {"actuator_delay": bool(delay), "actuator_delay_max_ms": float(actuator_delay_max_ms),
            "reset_root_z_offset": float(offset)}

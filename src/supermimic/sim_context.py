"""Shared, versioned layouts for simulator, replay and learned-model context.

The legacy prefix stays unchanged so analytic encoder observations can continue
to read xi[..., 1:1+num_joints]. Delay is a parameter, not a delay-queue state.
"""

import math
from dataclasses import dataclass
from typing import Literal

PHYSICS_STEP_MS = 5
CONTROL_SUBSTEPS = 4

ContextKind = Literal["base", "khan_dr_v1", "khan_dr_v2", "khan_dr_v3", "khan_dr_v4"]
KHAN_ANKLE_JOINTS = (
    "Left_ankle_pitch_joint", "Left_ankle_roll_joint",
    "Right_ankle_pitch_joint", "Right_ankle_roll_joint",
)
KHAN_LOWER_BODY_JOINTS = tuple(
    f"{side}_{joint}_joint"
    for side in ("Left", "Right")
    for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
)
KHAN_FOOT_BODIES = ("Left_ankle_roll_Link", "Right_ankle_roll_Link")


def validate_ankle_encoder_bias_half_range(robot: str, value: float) -> float:
    """Radians, independent of the fixed xi feature divisor (0.01 rad)."""
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("ankle_encoder_bias_half_range must be finite and nonnegative")
    if robot != "khan_mini" and value != 0.01:
        raise ValueError("Custom ankle_encoder_bias_half_range is only supported for Khan Mini")
    return value


@dataclass(frozen=True)
class SimContextLayout:
    num_joints: int
    kind: ContextKind = "base"

    def __post_init__(self):
        if self.num_joints < 1:
            raise ValueError("num_joints must be positive")
        if self.kind not in ("base", "khan_dr_v1", "khan_dr_v2", "khan_dr_v3", "khan_dr_v4"):
            raise ValueError(f"Unknown context layout: {self.kind}")
        if self.kind != "base" and self.num_joints != 27:
            raise ValueError(f"{self.kind} requires the 27-joint Khan model")

    @property
    def sizes(self) -> dict[str, int]:
        groups = {"friction": 1, "encoder_bias": self.num_joints, "com_offset": 3}
        if self.kind == "khan_dr_v1":
            groups.update(kp_factor=self.num_joints, kd_factor=self.num_joints,
                          ankle_armature_factor=4, torso_mass_delta=1, delay_steps=1)
        elif self.kind in ("khan_dr_v2", "khan_dr_v3", "khan_dr_v4"):
            groups.update(kp_factor=self.num_joints, kd_factor=self.num_joints,
                          lower_body_armature_factor=len(KHAN_LOWER_BODY_JOINTS),
                          torso_mass_delta=1, delay_steps=1)
            if self.kind in ("khan_dr_v3", "khan_dr_v4"):
                groups["foot_mass_delta"] = len(KHAN_FOOT_BODIES)
            if self.kind == "khan_dr_v4":
                groups["ankle_frictionloss"] = len(KHAN_ANKLE_JOINTS)
        return groups

    @property
    def slices(self) -> dict[str, slice]:
        result, start = {}, 0
        for name, size in self.sizes.items():
            result[name] = slice(start, start + size)
            start += size
        return result

    @property
    def dim(self) -> int:
        return sum(self.sizes.values())

    @property
    def offset(self) -> tuple[float, ...]:
        values = (0.75,) + (0.0,) * (self.num_joints + 3)
        if self.kind != "base":
            count = 4 if self.kind == "khan_dr_v1" else len(KHAN_LOWER_BODY_JOINTS)
            values += (1.0,) * (2 * self.num_joints + count) + (0.5, 2.0)
        if self.kind == "khan_dr_v3":
            values += (0.1,) * len(KHAN_FOOT_BODIES)
        elif self.kind == "khan_dr_v4":
            values += (0.075,) * len(KHAN_FOOT_BODIES) + (0.175,) * len(KHAN_ANKLE_JOINTS)
        return values

    @property
    def scale(self) -> tuple[float, ...]:
        com_scale = (0.035, 0.07, 0.07) if self.kind in ("khan_dr_v2", "khan_dr_v3", "khan_dr_v4") else (0.05,) * 3
        values = (0.45,) + (0.01,) * self.num_joints + com_scale
        if self.kind != "base":
            # Gains/armature .8..1.2; mass delta -2..3 kg; lag 0..4 x 5 ms.
            count = 4 if self.kind == "khan_dr_v1" else len(KHAN_LOWER_BODY_JOINTS)
            values += (0.2,) * (2 * self.num_joints + count) + (2.5, 2.0)
        if self.kind == "khan_dr_v3":
            values += (0.1,) * len(KHAN_FOOT_BODIES)
        elif self.kind == "khan_dr_v4":
            values += (0.075,) * (len(KHAN_FOOT_BODIES) + len(KHAN_ANKLE_JOINTS))
        return values


def context_kind_for_robot(robot: str) -> ContextKind:
    """Fresh-training default; historical loaders must pass their saved layout."""
    return "khan_dr_v4" if robot == "khan_mini" else "base"


def resolve_context_layout(robot: str, requested: ContextKind | None = None) -> ContextKind:
    layout = context_kind_for_robot(robot) if requested is None else requested
    allowed = ("khan_dr_v1", "khan_dr_v2", "khan_dr_v3", "khan_dr_v4") if robot == "khan_mini" else ("base",)
    if layout not in allowed:
        raise ValueError(f"Robot {robot} requires context_layout in {allowed}, received {layout!r}")
    return layout

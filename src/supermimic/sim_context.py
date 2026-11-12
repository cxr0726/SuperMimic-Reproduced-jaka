"""Shared, versioned layouts for simulator, replay and learned-model context.

The legacy prefix stays unchanged so analytic encoder observations can continue
to read xi[..., 1:1+num_joints]. Delay is a parameter, not a delay-queue state.
"""

from dataclasses import dataclass
from typing import Literal

ContextKind = Literal["base", "khan_dr_v1", "khan_dr_v2", "khan_dr_v3"]
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


@dataclass(frozen=True)
class SimContextLayout:
    num_joints: int
    kind: ContextKind = "base"

    def __post_init__(self):
        if self.num_joints < 1:
            raise ValueError("num_joints must be positive")
        if self.kind not in ("base", "khan_dr_v1", "khan_dr_v2", "khan_dr_v3"):
            raise ValueError(f"Unknown context layout: {self.kind}")
        if self.kind in ("khan_dr_v1", "khan_dr_v2", "khan_dr_v3") and self.num_joints != 27:
            raise ValueError(f"{self.kind} requires the 27-joint Khan model")

    @property
    def sizes(self) -> dict[str, int]:
        groups = {"friction": 1, "encoder_bias": self.num_joints, "com_offset": 3}
        if self.kind == "khan_dr_v1":
            groups.update(kp_factor=self.num_joints, kd_factor=self.num_joints,
                          ankle_armature_factor=4, torso_mass_delta=1, delay_steps=1)
        elif self.kind in ("khan_dr_v2", "khan_dr_v3"):
            groups.update(kp_factor=self.num_joints, kd_factor=self.num_joints,
                          lower_body_armature_factor=len(KHAN_LOWER_BODY_JOINTS),
                          torso_mass_delta=1, delay_steps=1)
            if self.kind == "khan_dr_v3":
                groups["foot_mass_delta"] = len(KHAN_FOOT_BODIES)
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
        if self.kind in ("khan_dr_v1", "khan_dr_v2", "khan_dr_v3"):
            count = 4 if self.kind == "khan_dr_v1" else len(KHAN_LOWER_BODY_JOINTS)
            values += (1.0,) * (2 * self.num_joints + count) + (0.5, 2.0)
        if self.kind == "khan_dr_v3":
            values += (0.1,) * len(KHAN_FOOT_BODIES)
        return values

    @property
    def scale(self) -> tuple[float, ...]:
        com_scale = (0.035, 0.07, 0.07) if self.kind in ("khan_dr_v2", "khan_dr_v3") else (0.05,) * 3
        values = (0.45,) + (0.01,) * self.num_joints + com_scale
        if self.kind in ("khan_dr_v1", "khan_dr_v2", "khan_dr_v3"):
            # Gains/armature .8..1.2; mass delta -2..3 kg; lag 0..4 x 5 ms.
            count = 4 if self.kind == "khan_dr_v1" else len(KHAN_LOWER_BODY_JOINTS)
            values += (0.2,) * (2 * self.num_joints + count) + (2.5, 2.0)
        if self.kind == "khan_dr_v3":
            values += (0.1,) * len(KHAN_FOOT_BODIES)
        return values


def context_kind_for_robot(robot: str) -> ContextKind:
    """Fresh-training default; historical loaders must pass their saved layout."""
    return "khan_dr_v3" if robot == "khan_mini" else "base"


def resolve_context_layout(robot: str, requested: ContextKind | None = None) -> ContextKind:
    layout = context_kind_for_robot(robot) if requested is None else requested
    allowed = ("khan_dr_v1", "khan_dr_v2", "khan_dr_v3") if robot == "khan_mini" else ("base",)
    if layout not in allowed:
        raise ValueError(f"Robot {robot} requires context_layout in {allowed}, received {layout!r}")
    return layout

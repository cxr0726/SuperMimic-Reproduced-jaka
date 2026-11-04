"""Shared, versioned layouts for simulator, replay and learned-model context.

The legacy prefix stays unchanged so analytic encoder observations can continue
to read xi[..., 1:1+num_joints]. Delay is a parameter, not a delay-queue state.
"""

from dataclasses import dataclass
from typing import Literal

ContextKind = Literal["base", "khan_dr_v1"]
KHAN_ANKLE_JOINTS = (
    "Left_ankle_pitch_joint", "Left_ankle_roll_joint",
    "Right_ankle_pitch_joint", "Right_ankle_roll_joint",
)


@dataclass(frozen=True)
class SimContextLayout:
    num_joints: int
    kind: ContextKind = "base"

    def __post_init__(self):
        if self.num_joints < 1:
            raise ValueError("num_joints must be positive")
        if self.kind not in ("base", "khan_dr_v1"):
            raise ValueError(f"Unknown context layout: {self.kind}")
        if self.kind == "khan_dr_v1" and self.num_joints != 27:
            raise ValueError("khan_dr_v1 requires the 27-joint Khan model")

    @property
    def sizes(self) -> dict[str, int]:
        groups = {"friction": 1, "encoder_bias": self.num_joints, "com_offset": 3}
        if self.kind == "khan_dr_v1":
            groups.update(kp_factor=self.num_joints, kd_factor=self.num_joints,
                          ankle_armature_factor=4, torso_mass_delta=1, delay_steps=1)
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
        if self.kind == "khan_dr_v1":
            values += (1.0,) * (2 * self.num_joints + 4) + (0.5, 2.0)
        return values

    @property
    def scale(self) -> tuple[float, ...]:
        values = (0.45,) + (0.01,) * self.num_joints + (0.05,) * 3
        if self.kind == "khan_dr_v1":
            # Gains/armature .8..1.2; mass delta -2..3 kg; lag 0..4 x 5 ms.
            values += (0.2,) * (2 * self.num_joints + 4) + (2.5, 2.0)
        return values


def context_kind_for_robot(robot: str) -> ContextKind:
    return "khan_dr_v1" if robot == "khan_mini" else "base"

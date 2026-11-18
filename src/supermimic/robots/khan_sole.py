"""Build the two convex foot colliders in the Khan ankle-roll link frame."""

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial import ConvexHull


@dataclass(frozen=True)
class SoleDimensions:
    """Dimensions in mm. X points toward the toes, Z points up."""

    length: float = 202
    heel_x: float = -75
    rear_width: float = 87
    forefoot_width: float = 105
    toe_to_widest: float = 56
    flare_length: float = 46
    heel_cap_length: float = 34
    edge_radius: float = 2
    bottom_z: float = -48
    sole_top_z: float = -38

    @property
    def widest_x(self) -> float:
        return self.heel_x + self.length - self.toe_to_widest

    @property
    def split_x(self) -> float:
        return self.widest_x - self.flare_length


SOLE_MM = SoleDimensions()


def _half_outline(part: str, inset: float) -> tuple[np.ndarray, np.ndarray]:
    """Longitudinal stations and half-widths of one horizontal ring."""
    s = SOLE_MM
    arc = np.linspace(0, np.pi / 2, 17)
    rear = s.rear_width / 2 - inset
    front = s.forefoot_width / 2 - inset
    if part == "rear":
        cap_end = s.heel_x + s.heel_cap_length
        x = cap_end - (s.heel_cap_length - inset) * np.cos(arc)
        return np.r_[x, s.split_x], np.r_[rear * np.sin(arc), rear]
    ramp = np.linspace(0, 1, 9)
    x = np.r_[s.split_x + s.flare_length * ramp,
              s.widest_x + (s.toe_to_widest - inset) * np.sin(arc[1:])]
    width = np.r_[rear + (front - rear) * np.sin(ramp * np.pi / 2),
                  front * np.cos(arc[1:])]
    return x, width


def _cad_roof(model: mujoco.MjModel, visual_name: str) -> np.ndarray:
    """Original CAD hull above the sole plate, including its section at the joint between pieces."""
    geom = model.geom(visual_name).id
    mesh = model.geom_dataid[geom]
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
    vertices = model.mesh_vert[start:start + count] @ rotation.reshape(3, 3).T
    vertices = (vertices + model.geom_pos[geom]) * 1000
    hull = ConvexHull(vertices)
    edges = hull.simplices[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    p, q = vertices[edges[:, 0]], vertices[edges[:, 1]]
    cut = (p[:, 0] < SOLE_MM.split_x) != (q[:, 0] < SOLE_MM.split_x)
    p, q = p[cut], q[cut]
    section = p + ((SOLE_MM.split_x - p[:, 0]) / (q[:, 0] - p[:, 0]))[:, None] * (q - p)
    section[:, 0] = SOLE_MM.split_x
    roof = np.vstack((vertices[hull.vertices], section))
    return roof[roof[:, 2] > SOLE_MM.sole_top_z]


def sole_meshes(model: mujoco.MjModel, visual_name: str) -> dict[str, np.ndarray]:
    """Return hull vertices in metres for inline MJCF meshes, with a shared flush seam."""
    s = SOLE_MM
    roof = _cad_roof(model, visual_name)
    angle = np.linspace(0, np.pi / 2, 5)
    insets = np.r_[s.edge_radius * (1 - np.sin(angle)), 0]
    heights = np.r_[s.bottom_z + s.edge_radius * (1 - np.cos(angle)), s.sole_top_z]
    meshes = {}
    for part in ("rear", "forefoot"):
        rings = []
        for inset, height in zip(insets, heights, strict=True):
            x, y = _half_outline(part, inset)
            z = np.full_like(x, height)
            rings.extend((np.column_stack((x, y, z)), np.column_stack((x, -y, z))))
        # Preserve the ankle/instep roof inside the new outline. The split itself
        # has no fillet: the external bottom edges meet without a groove or overlap.
        x, y = _half_outline(part, 0)
        upper = roof[(roof[:, 0] >= x[0]) & (roof[:, 0] <= x[-1])].copy()
        limit = np.interp(upper[:, 0], x, y)
        upper[:, 1] = np.clip(upper[:, 1], -limit, limit)
        vertices = np.unique(np.round(np.vstack([*rings, upper]), 5), axis=0)
        meshes[part] = vertices[ConvexHull(vertices).vertices] / 1000
    return meshes

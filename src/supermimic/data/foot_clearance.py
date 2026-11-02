"""CPU kinematic foot-ground clearance for Khan motion NPZs.

Ground is the world plane z=0. Collision clearance uses the actual geometry,
including capsule endpoints/radius; visual clearance uses the foot's welded
visual meshes. No simulation step, height correction or data write is applied.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from supermimic.robots.khan_mini import (
    DEFAULT_ASSET_DIR,
    get_khan_mini_spec,
    load_profile,
)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def mesh_support_vertices(model: mujoco.MjModel, geom_id: int) -> np.ndarray:
    """Compiled mesh support vertices; no second application of mesh transforms.

    A linear support query has the same minimum over the full vertex set and
    its convex hull. Reducing to hull vertices speeds the all-frame audit while
    preserving the exact visual bottom, even for a nonconvex source mesh.
    """
    mesh_id = int(model.geom_dataid[geom_id])
    if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH or mesh_id < 0:
        raise ValueError("mesh_support_vertices requires a mesh geom")
    start = int(model.mesh_vertadr[mesh_id])
    count = int(model.mesh_vertnum[mesh_id])
    vertices = model.mesh_vert[start:start + count].astype(np.float64)
    if count > 4:
        try:
            vertices = vertices[ConvexHull(vertices).vertices]
        except QhullError:
            pass  # Flat/degenerate meshes still have a well-defined Z minimum.
    return vertices


def geom_bottom_z(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
    *,
    mesh_vertices: np.ndarray | None = None,
) -> float:
    """Exact lowest world Z of a primitive or compiled mesh's vertices."""
    kind = model.geom_type[geom_id]
    size = model.geom_size[geom_id]
    position = data.geom_xpos[geom_id]
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    z_axis = rotation[2]
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        extent = size[0]
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        extent = size[0] + size[1] * abs(z_axis[2])
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        extent = size[1] * abs(z_axis[2]) + size[0] * np.sqrt(max(0.0, 1 - z_axis[2] ** 2))
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        extent = np.dot(np.abs(z_axis), size)
    elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        extent = np.linalg.norm(z_axis * size)
    elif kind == mujoco.mjtGeom.mjGEOM_MESH:
        vertices = mesh_support_vertices(model, geom_id) if mesh_vertices is None else mesh_vertices
        return float(position[2] + np.min(vertices @ z_axis))
    else:
        raise ValueError(f"Unsupported foot geometry type: {mujoco.mjtGeom(int(kind)).name}")
    return float(position[2] - extent)


def foot_geom_groups(model: mujoco.MjModel, profile: dict | None = None) -> dict:
    """Find collision soles and the visuals welded to each ankle-roll foot."""
    profile = profile if profile is not None else load_profile()
    groups = {}
    collision = (model.geom_contype | model.geom_conaffinity) != 0
    for foot in profile["foot_bodies"]:
        candidates = [i for i in range(1, model.nbody)
                      if model.body(i).name.rsplit("/", 1)[-1] == foot]
        if len(candidates) != 1:
            raise ValueError(f"Expected one body named {foot}, found {len(candidates)}")
        body_id = candidates[0]
        weld_id = model.body_weldid[body_id]
        assembly = model.body_weldid[model.geom_bodyid] == weld_id
        collision_ids = np.flatnonzero(assembly & collision).tolist()
        visual_ids = np.flatnonzero(assembly & (model.geom_group == 2)).tolist()
        if not collision_ids or not visual_ids:
            raise ValueError(f"Foot {foot} lacks collision or visual geometry")
        groups[foot] = {"collision": collision_ids, "visual": visual_ids}
    return groups


def load_npz_motion(path: str | Path, asset_dir: str | Path | None = None) -> dict:
    """Read a named, finite Khan motion; reject mismatched body/joint layouts."""
    profile = load_profile(asset_dir)
    with np.load(path, allow_pickle=False) as archive:
        motion = {key: archive[key] for key in archive.files}
    required = ("fps", "joint_pos", "body_pos_w", "body_quat_w", "joint_names", "body_names")
    if any(key not in motion for key in required):
        raise ValueError("NPZ is missing required motion fields")
    if motion["fps"].size != 1:
        raise ValueError("fps must have one value")
    fps = float(motion["fps"].reshape(-1)[0])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if motion["joint_names"].tolist() != profile["joint_names"]:
        raise ValueError("NPZ joint_names differ from the Khan model")
    if motion["body_names"].tolist() != profile["body_names"]:
        raise ValueError("NPZ body_names differ from the Khan model")
    n = len(motion["joint_pos"])
    if n < 1:
        raise ValueError("Motion is empty")
    for key, shape in (("joint_pos", (n, 27)), ("body_pos_w", (n, 45, 3)),
                       ("body_quat_w", (n, 45, 4))):
        if motion[key].shape != shape or not np.isfinite(motion[key]).all():
            raise ValueError(f"Invalid {key}: expected finite {shape}")
    norm = np.linalg.norm(motion["body_quat_w"], axis=-1)
    if not np.allclose(norm, 1, atol=1e-5, rtol=0):
        raise ValueError("body_quat_w must contain unit WXYZ quaternions")
    return motion


def _height_summary(z: np.ndarray, frames: np.ndarray, fps: float, tolerance: float) -> dict:
    worst = int(np.argmin(z))
    below = z <= -tolerance
    return {
        "min_z_m": float(z[worst]),
        "min_frame": int(frames[worst]), "min_time_s": float(frames[worst] / fps),
        "max_penetration_m": max(0.0, float(-z[worst])),
        "negative_height_frames": int((z < 0).sum()),
        "below_tolerance_frames": int(below.sum()),
        "below_tolerance_fraction": float(below.mean()),
        "z_percentiles_m": dict(zip(("p0", "p1", "p50", "p99"),
                                    np.quantile(z, [0, 0.01, 0.5, 0.99]).tolist(), strict=True)),
    }


def audit_npz(
    npz_path: str | Path,
    asset_dir: str | Path | None = None,
    *,
    tolerance_m: float = 0.001,
    stride: int = 1,
) -> dict:
    """Audit all requested frames; return JSON summary plus ``_per_frame`` arrays.

    ``_per_frame`` is intended for CSV output and is omitted from JSON by the CLI.
    A below-tolerance frame means bottom Z <= -tolerance_m. The last frame is
    always included, including when a nonunit stride is requested.
    """
    if not np.isfinite(tolerance_m) or tolerance_m < 0 or stride < 1:
        raise ValueError("tolerance_m must be nonnegative and stride must be positive")
    path = Path(npz_path)
    asset_root = Path(asset_dir) if asset_dir is not None else DEFAULT_ASSET_DIR
    profile = load_profile(asset_root)
    model = get_khan_mini_spec(asset_root).compile()
    groups = foot_geom_groups(model, profile)
    motion = load_npz_motion(path, asset_root)
    fps = float(motion["fps"].reshape(-1)[0])
    n = len(motion["joint_pos"])
    frames = np.unique(np.r_[np.arange(0, n, stride), n - 1]).astype(np.int64)
    root_id = motion["body_names"].tolist().index(profile["root_body"])
    data = mujoco.MjData(model)
    # Collision meshes need the same support cache as visual meshes; otherwise
    # geom_bottom_z would reconstruct a convex hull for every motion frame.
    support = {g: mesh_support_vertices(model, g) for group in groups.values()
               for kind in ("collision", "visual") for g in group[kind]
               if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH}
    heights = {foot: {kind: np.empty(len(frames)) for kind in ("collision", "visual")}
               for foot in groups}
    collision_ids = np.array([g for group in groups.values() for g in group["collision"]])
    all_capsules = np.all(model.geom_type[collision_ids] == mujoco.mjtGeom.mjGEOM_CAPSULE)
    for i, frame in enumerate(frames):
        data.qpos[:3] = motion["body_pos_w"][frame, root_id]
        data.qpos[3:7] = motion["body_quat_w"][frame, root_id]
        data.qpos[7:] = motion["joint_pos"][frame]
        mujoco.mj_kinematics(model, data)
        capsule_bottom = None
        if all_capsules:
            capsule_bottom = (data.geom_xpos[collision_ids, 2]
                              - np.abs(data.geom_xmat[collision_ids, 8]) * model.geom_size[collision_ids, 1]
                              - model.geom_size[collision_ids, 0])
        offset = 0
        for foot, group in groups.items():
            if capsule_bottom is not None:
                count = len(group["collision"])
                heights[foot]["collision"][i] = capsule_bottom[offset:offset + count].min()
                offset += count
            else:
                heights[foot]["collision"][i] = min(
                    geom_bottom_z(model, data, g, mesh_vertices=support.get(g)) for g in group["collision"]
                )
            heights[foot]["visual"][i] = min(
                geom_bottom_z(model, data, g, mesh_vertices=support.get(g)) for g in group["visual"]
            )
    per_frame = {"frame": frames, "time_s": frames / fps}
    feet = {}
    for foot, group in groups.items():
        feet[foot] = {}
        for kind in ("collision", "visual"):
            values = heights[foot][kind]
            per_frame[f"{foot}_{kind}_min_z_m"] = values
            feet[foot][kind] = {
                **_height_summary(values, frames, fps, tolerance_m),
                "geometry_names": [model.geom(g).name for g in group[kind]],
            }
    combined = {}
    for kind in ("collision", "visual"):
        values = np.minimum.reduce([height[kind] for height in heights.values()])
        per_frame[f"both_feet_{kind}_min_z_m"] = values
        combined[kind] = _height_summary(values, frames, fps, tolerance_m)
    return {
        "status": "audited", "npz": str(path.resolve()), "clip": path.stem,
        "npz_sha256": _sha256(path), "model_sha256": _sha256(asset_root / "robot.xml"),
        "profile_sha256": _sha256(asset_root / "profile.json"),
        "fps": fps, "total_frames": n, "evaluated_frames": len(frames), "stride": stride,
        "tolerance_m": tolerance_m, "ground_z_m": 0.0, "feet": feet, "combined": combined,
        "criterion": "min geometry Z <= -tolerance_m",
        "visual_scope": "visual geoms on each ankle_roll body and its welded fixed bodies",
        "notes": ["Kinematic geometry audit only; no dynamics, height lift, clipping or contact repair.",
                  "Collision bottoms use exact primitives or compiled mesh vertex support; visual bottoms use compiled mesh vertex support."],
        "_per_frame": per_frame,
    }

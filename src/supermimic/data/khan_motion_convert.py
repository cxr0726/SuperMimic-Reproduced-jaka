"""Convert retargeted JAKA/Khan joblib motion packs to the SuperMimic NPZ schema.

The legacy JAKA pack has 25 columns: the Khan joint order with both wrist-yaw
joints omitted. Those two targets are filled from the model's home pose. Root
trajectories, all source frames, and the remaining joint angles are preserved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

import joblib
import mujoco
import numpy as np

from supermimic.data.motion_convert import (
    ResampledMotion,
    forward_kinematics,
    resample_motion,
)
from supermimic.robots.khan_mini import (
    DEFAULT_ASSET_DIR,
    ROOT,
    get_khan_mini_spec,
    load_profile,
)

MISSING_WRISTS = ("Left_wrist_yaw_joint", "Right_wrist_yaw_joint")
DEFAULT_INPUT = ROOT / (
    "ref_thirdparty/jaka_code/dataset_pkl_jaka/"
    "lafan_pkl_data_new_wrist_merged_motion_new2_jaka_lafan_new_wrist.pkl"
)
DEFAULT_OUTPUT = ROOT / "data/npz/khan_mini/lafan1"


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _array(data: dict, name: str, width: int | None = None) -> np.ndarray:
    value = np.asarray(data[name], dtype=np.float64)
    if value.ndim != 2 or (width is not None and value.shape[1] != width):
        raise ValueError(f"{name}: expected (T,{width or 'joints'}), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return value


def source_joint_names(data: dict, profile: dict) -> list[str]:
    """Resolve the pack's documented layout or its explicit column names."""
    dof = _array(data, "dof_pos")
    target = profile["joint_names"]
    if data.get("joint_names") is not None:
        names = [str(name) for name in data["joint_names"]]
    elif dof.shape[1] == 25:
        names = [name for name in target if name not in MISSING_WRISTS]
    elif dof.shape[1] == 27:
        names = list(target)
    else:
        raise ValueError(f"Unsupported source joint count {dof.shape[1]}; expected JAKA 25 or Khan 27")
    if len(names) != dof.shape[1] or len(set(names)) != len(names):
        raise ValueError("joint_names must uniquely name every dof_pos column")
    unknown = set(names) - set(target)
    missing = set(target) - set(names)
    if unknown or not missing.issubset(MISSING_WRISTS):
        raise ValueError(f"Unsupported joint mapping: unknown={sorted(unknown)}, missing={sorted(missing)}")
    return names


def prepare_khan_motion(
    data: dict, profile: dict, *, output_fps: float = 50.0, quat_order: str = "wxyz"
) -> ResampledMotion:
    """Map source joints and resample without trimming, lifting or clipping."""
    fps_value = np.asarray(data["fps"])
    if fps_value.size != 1:
        raise ValueError("fps must be a scalar")
    input_fps = float(fps_value.reshape(-1)[0])
    if not np.isfinite(input_fps) or input_fps <= 0:
        raise ValueError("fps must be finite and positive")
    if not np.isfinite(output_fps) or output_fps <= 0:
        raise ValueError("output_fps must be finite and positive")
    pos = _array(data, "root_pos", 3)
    quat = _array(data, "root_rot", 4)
    dof = _array(data, "dof_pos")
    n = len(pos)
    if n < 3 or len(quat) != n or len(dof) != n:
        raise ValueError("root_pos, root_rot and dof_pos need the same T >= 3")
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    if (norm < 1e-8).any():
        raise ValueError("root_rot contains a zero quaternion")
    quat = quat / norm
    if quat_order == "wxyz":
        quat_xyzw = quat[:, [1, 2, 3, 0]]
    elif quat_order == "xyzw":
        quat_xyzw = quat
    else:
        raise ValueError("quat_order must be wxyz or xyzw")
    names = source_joint_names(data, profile)
    lookup = {name: i for i, name in enumerate(names)}
    target = np.empty((n, len(profile["joint_names"])), dtype=np.float64)
    for i, name in enumerate(profile["joint_names"]):
        target[:, i] = dof[:, lookup[name]] if name in lookup else profile["default_joint_pos"][name]
    raw = np.concatenate((pos, quat_xyzw, target), axis=1)
    return resample_motion(raw, input_fps, float(output_fps))


def _check_model(model: mujoco.MjModel, profile: dict) -> None:
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
             for i in range(1, model.njnt)]
    if (model.nq, model.nv) != (34, 33) or names != profile["joint_names"]:
        raise ValueError("Expected the 27-joint Khan_mini model and profile order")
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Khan_mini requires a free root joint")


def convert_clip(
    data: dict,
    out_path: str | Path,
    *,
    model: mujoco.MjModel | None = None,
    asset_dir: str | Path | None = None,
    source: str = "",
    clip_name: str = "",
    quat_order: str = "wxyz",
    output_fps: float = 50.0,
) -> dict:
    """Write one pickle-free NPZ, retaining the model's exact body/joint order."""
    profile = load_profile(asset_dir)
    model = model if model is not None else get_khan_mini_spec(asset_dir).compile()
    _check_model(model, profile)
    motion = prepare_khan_motion(data, profile, output_fps=output_fps, quat_order=quat_order)
    bodies = forward_kinematics(model, motion)
    names = source_joint_names(data, profile)
    filled = [name for name in profile["joint_names"] if name not in names]
    arrays = {
        "fps": np.array([motion.fps], dtype=np.float32),
        "joint_pos": motion.joint_pos.astype(np.float32),
        "joint_vel": motion.joint_vel.astype(np.float32),
        "body_pos_w": bodies.pos_w.astype(np.float32),
        "body_quat_w": bodies.quat_w.astype(np.float32),
        "body_lin_vel_w": bodies.lin_vel_w.astype(np.float32),
        "body_ang_vel_w": bodies.ang_vel_w.astype(np.float32),
        "joint_names": np.asarray(profile["joint_names"], dtype=np.str_),
        "body_names": np.asarray(bodies.body_names, dtype=np.str_),
        "source": np.array(str(source)),
        "input_fps": np.array([float(np.asarray(data["fps"]).reshape(-1)[0])], dtype=np.float32),
        "robot": np.array("khan_mini"),
        "source_clip": np.array(clip_name),
        "source_quat_order": np.array(quat_order),
        "source_joint_names": np.asarray(names, dtype=np.str_),
        "filled_joint_names": np.asarray(filled, dtype=np.str_),
        "filled_joint_values": np.asarray([profile["default_joint_pos"][name] for name in filled],
                                          dtype=np.float32),
    }
    if any(not np.isfinite(value).all() for value in arrays.values() if value.dtype.kind == "f"):
        raise ValueError("Conversion produced nonfinite float32 output")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)
    limits = model.jnt_range[1:]
    violation = np.maximum(limits[:, 0] - motion.joint_pos, motion.joint_pos - limits[:, 1]).clip(0)
    velocity_limits = np.array([profile["velocity_limit"][name] for name in profile["joint_names"]])
    return {
        "name": out_path.stem, "source_clip": clip_name,
        "input_frames": len(data["root_pos"]), "frames": motion.num_frames,
        "input_fps": float(arrays["input_fps"][0]), "fps": motion.fps,
        "duration_s": (motion.num_frames - 1) / motion.fps,
        "source_dofs": len(names),
        "filled_joints": {name: profile["default_joint_pos"][name] for name in filled},
        "max_joint_limit_violation_rad": float(violation.max()),
        "joint_samples_beyond_hard_limits_1e_6": int((violation > 1e-6).sum()),
        "joint_samples_beyond_velocity_metadata": int((np.abs(motion.joint_vel) > velocity_limits).sum()),
        "max_joint_speed_rad_s": float(np.abs(motion.joint_vel).max()),
        "output_bytes": out_path.stat().st_size, "sha256": _sha256(out_path),
    }


def convert_khan_pkl(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    asset_dir: str | Path | None = None,
    output_fps: float = 50.0,
    quat_order: str = "wxyz",
    limit: int | None = None,
    force: bool = False,
) -> dict:
    """Convert a trusted local joblib pack; publish only a complete output set."""
    source = Path(input_path).resolve()
    target = Path(output_dir).resolve()
    if source.is_relative_to(target):
        raise ValueError("Output directory must not contain the source pack")
    if target.exists() and not force:
        raise FileExistsError(f"{target} already exists; choose another output directory or --force")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    motions = joblib.load(source)
    if not isinstance(motions, dict) or not motions:
        raise ValueError("Expected a nonempty mapping of clip names to motion dictionaries")
    items = list(motions.items())[:limit]
    safe_names = [re.sub(r"[^a-zA-Z0-9_.-]", "_", str(name)) for name, _ in items]
    if any(name in ("", ".", "..") for name in safe_names) or len(set(safe_names)) != len(safe_names):
        raise ValueError("Clip names produce empty, unsafe or duplicate output filenames")
    profile = load_profile(asset_dir)
    model = get_khan_mini_spec(asset_dir).compile()
    _check_model(model, profile)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix=".khan-motion-", dir=target.parent) as temporary:
        stage = Path(temporary) / "clips"
        stage.mkdir()
        for i, ((name, data), safe) in enumerate(zip(items, safe_names, strict=True), 1):
            info = convert_clip(data, stage / f"{safe}.npz", model=model, asset_dir=asset_dir,
                                source=str(source), clip_name=str(name), quat_order=quat_order,
                                output_fps=output_fps)
            rows.append(info)
            print(f"[{i}/{len(items)}] {safe}: {info['frames']} frames @ {info['fps']:g} Hz", flush=True)
        fields = ["name", "frames", "fps", "duration_s", "input_fps", "source_dofs"]
        with (stage / "manifest.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        report = {
            "status": "converted", "robot": "khan_mini", "source": str(source),
            "source_sha256": _sha256(source), "source_serialization": "joblib",
            "source_clip_count": len(motions), "converted_clips": len(rows),
            "output_dir": str(target), "output_fps": output_fps, "source_quat_order": quat_order,
            "joint_names": profile["joint_names"], "body_names": profile["body_names"],
            "robot_assets_sha256": {
                name: _sha256(Path(asset_dir or DEFAULT_ASSET_DIR) / name)
                for name in ("robot.xml", "profile.json")
            },
            "processing": {
                "trimmed_frames": 0, "root_height_offset_m": 0.0,
                "joint_clipping": False, "contact_repair": False,
                "missing_wrist_yaw": "fill from model home (0.3 rad each), retain source neck columns",
                "quaternions": "normalize and shortest-path SLERP; output wxyz",
                "velocities": "gradient at output dt; SO(3) central difference; MuJoCo link-origin FK",
            },
            "total_frames": sum(row["frames"] for row in rows),
            "total_duration_s": sum(row["duration_s"] for row in rows),
            "total_output_bytes": sum(row["output_bytes"] for row in rows),
            "hard_limit_violation_samples_1e_6": sum(row["joint_samples_beyond_hard_limits_1e_6"] for row in rows),
            "velocity_metadata_exceedance_samples": sum(row["joint_samples_beyond_velocity_metadata"] for row in rows),
            "clips": rows,
        }
        (stage / "conversion_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if target.exists():
            shutil.rmtree(target)
        stage.rename(target)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--asset-dir", type=Path)
    parser.add_argument("--output-fps", type=float, default=50.0)
    parser.add_argument("--quat-order", choices=("wxyz", "xyzw"), default="wxyz")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = convert_khan_pkl(args.input, args.output_dir, asset_dir=args.asset_dir,
                              output_fps=args.output_fps, quat_order=args.quat_order,
                              limit=args.limit, force=args.force)
    print(json.dumps({key: report[key] for key in
                      ("output_dir", "converted_clips", "total_frames", "total_duration_s")}, indent=2))


if __name__ == "__main__":
    main()

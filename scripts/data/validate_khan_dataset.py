"""Independently validate 25/27-DOF JAKA packs against 50 Hz Khan NPZs.

Reads trusted local joblib input and pickle-free NPZ output. Uses NumPy/SciPy
derivatives and MuJoCo mj_forward/mj_objectVelocity, never the converter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / (
    "ref_thirdparty/jaka_code/dataset_pkl_jaka/"
    "lafan_pkl_data_new_wrist_merged_motion_new2_jaka_lafan_new_wrist.pkl"
)
WRISTS = ("Left_wrist_yaw_joint", "Right_wrist_yaw_joint")
TRAJECTORIES = {
    "joint_pos": (27,), "joint_vel": (27,), "body_pos_w": (45, 3),
    "body_quat_w": (45, 4), "body_lin_vel_w": (45, 3), "body_ang_vel_w": (45, 3),
}
REQUIRED = set(TRAJECTORIES) | {
    "fps", "input_fps", "joint_names", "body_names", "source", "robot",
    "source_clip", "source_quat_order", "source_joint_names",
    "filled_joint_names", "filled_joint_values",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rotations(wxyz: np.ndarray) -> Rotation:
    return Rotation.from_quat(wxyz[:, [1, 2, 3, 0]])


def max_error(actual, expected, label: str, tolerance: float) -> float:
    error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    if error > tolerance:
        raise ValueError(f"{label}: maximum error {error:g} exceeds {tolerance:g}")
    return error


def independent_reference(source: dict, names: list[str], home: np.ndarray) -> dict:
    """Reconstruct the output grid with NumPy interpolation and SciPy rotations.

    Close quaternion pairs use normalized linear interpolation, matching the
    converter's documented small-angle approximation, without calling it.
    """
    position = np.asarray(source["root_pos"], dtype=np.float64)
    quaternion = np.array(source["root_rot"], dtype=np.float64, copy=True)
    dof = np.asarray(source["dof_pos"], dtype=np.float64)
    count = len(position)
    fps_value = np.asarray(source["fps"])
    if fps_value.size != 1:
        raise ValueError("Source fps must be scalar")
    fps = float(fps_value.reshape(-1)[0])
    if not np.isfinite(fps) or fps <= 0 or count < 3:
        raise ValueError("Source must have positive fps and at least three frames")
    if position.shape != (count, 3) or quaternion.shape != (count, 4):
        raise ValueError("Invalid source root shapes")
    if dof.ndim != 2 or len(dof) != count or dof.shape[1] not in (25, 27):
        raise ValueError("Source dof_pos must have 25 or 27 columns")
    if not all(np.isfinite(a).all() for a in (position, quaternion, dof)):
        raise ValueError("Nonfinite source arrays")
    source_names = source.get("joint_names")
    if source_names is None:
        source_names = [n for n in names if dof.shape[1] == 27 or n not in WRISTS]
    source_names = [str(n) for n in source_names]
    missing = [n for n in names if n not in source_names]
    if (len(source_names) != dof.shape[1] or len(set(source_names)) != len(source_names)
            or set(source_names)-set(names) or not set(missing).issubset(WRISTS)):
        raise ValueError("Invalid source joint names")
    source_joints = np.broadcast_to(home, (count, 27)).copy()
    source_joints[:, [names.index(n) for n in source_names]] = dof
    norms = np.linalg.norm(quaternion, axis=1, keepdims=True)
    if (norms < 1e-8).any():
        raise ValueError("Zero source quaternion")
    quaternion /= norms
    source_time = np.arange(count, dtype=np.float64) / fps
    output_count = int(np.floor(source_time[-1]*50 + 1e-6)) + 1
    time = np.arange(output_count, dtype=np.float64) / 50
    root_pos = np.column_stack([np.interp(time, source_time, a) for a in position.T])
    joints = np.column_stack([np.interp(time, source_time, a) for a in source_joints.T])
    # A grid point within the converter's 1e-6-frame endpoint tolerance is its
    # endpoint, rather than an extrapolation beyond the source's support.
    evaluation_time = np.minimum(time, source_time[-1])
    interpolated = Slerp(source_time, rotations(quaternion))(evaluation_time)
    root_quat = interpolated.as_quat()[:, [3, 0, 1, 2]]
    left = np.clip(np.searchsorted(source_time, evaluation_time, side="right")-1, 0, count-2)
    fraction = ((evaluation_time-source_time[left])/(source_time[left+1]-source_time[left]))[:, None]
    a, b = quaternion[left], quaternion[left+1]
    dot = np.sum(a*b, axis=1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    close = np.abs(dot[:, 0]) > .9995
    linear_quat = (1-fraction)*a + fraction*b
    linear_quat /= np.linalg.norm(linear_quat, axis=1, keepdims=True)
    root_quat[close] = linear_quat[close]
    return {"input_fps": fps, "input_frames": count, "source_names": source_names,
            "filled_names": missing, "root_pos": root_pos, "root_quat": root_quat,
            "joint_pos": joints}


def validate_dataset(input_path: Path, npz_dir: Path, asset_dir: Path, samples: int = 8) -> dict:
    pack = joblib.load(input_path)
    if not isinstance(pack, dict) or not pack:
        raise ValueError("Expected a nonempty source clip dictionary")
    model = mujoco.MjModel.from_xml_path(str(asset_dir / "robot.xml"))
    profile = json.loads((asset_dir / "profile.json").read_text())
    names = [model.joint(i).name for i in range(1, model.njnt)]
    bodies = [model.body(i).name for i in range(1, model.nbody)]
    assert (model.nq, model.nv, len(bodies)) == (34, 33, 45)
    assert names == profile["joint_names"]
    home = np.array([profile["default_joint_pos"][name] for name in names])
    velocity_limits = np.array([profile["velocity_limit"][name] for name in names])
    if {path.stem for path in npz_dir.glob("*.npz")} != set(pack):
        raise ValueError("NPZ clip set differs from the source pack")
    report = {
        "status": "running", "input": str(input_path.resolve()),
        "source_sha256": sha256(input_path), "npz_dir": str(npz_dir.resolve()),
        "model_sha256": sha256(asset_dir / "robot.xml"),
        "profile_sha256": sha256(asset_dir / "profile.json"),
        "validation": "Independent NumPy/SciPy derivatives and CPU MuJoCo FK/objectVelocity",
        "clip_count": len(pack), "frame_count": 0, "duration_s": 0.0,
        "fk_frames_checked": 0, "body_count": len(bodies), "clips": [],
        "max_errors": {}, "position_limit_samples_gt_1e_6": 0,
        "velocity_metadata_exceedance_samples": 0,
    }
    data = mujoco.MjData(model)
    for clip, source in pack.items():
        path = npz_dir / f"{clip}.npz"
        source_ref = independent_reference(source, names, home)
        root_pos, root_quat, joints = (source_ref[k] for k in ("root_pos", "root_quat", "joint_pos"))
        n = len(root_pos)
        linear = np.gradient(root_pos, 0.02, axis=0)
        joint_velocity = np.gradient(joints, 0.02, axis=0)
        rotation = rotations(root_quat)
        middle = (rotation[2:] * rotation[:-2].inv()).as_rotvec() / 0.04
        angular = np.concatenate([middle[:1], middle, middle[-1:]], axis=0)
        body_angular = rotation.inv().apply(angular)
        errors = {}
        with np.load(path, allow_pickle=False) as archive:
            motion = {key: archive[key] for key in archive.files}
            assert REQUIRED.issubset(motion), clip
            for key, tail in TRAJECTORIES.items():
                assert motion[key].shape == (n, *tail), (clip, key, motion[key].shape)
                assert motion[key].dtype == np.float32, (clip, key, motion[key].dtype)
            for key, value in motion.items():
                if value.dtype.kind in "fiu":
                    assert np.isfinite(value).all(), (clip, key)
            for key, expected_fps in (("fps", 50.0), ("input_fps", source_ref["input_fps"])):
                assert motion[key].shape == (1,) and np.isclose(float(motion[key][0]), expected_fps)
                assert motion[key].dtype == np.float32, (clip, key)
            assert motion["joint_names"].tolist() == names, clip
            assert motion["body_names"].tolist() == bodies, clip
            assert motion["source_joint_names"].tolist() == source_ref["source_names"], clip
            assert str(motion["source_clip"]) == clip
            assert str(motion["robot"]) == "khan_mini"
            assert str(motion["source_quat_order"]) == "wxyz"
            assert Path(str(motion["source"])).resolve() == input_path.resolve()
            assert motion["filled_joint_names"].tolist() == source_ref["filled_names"]
            assert motion["filled_joint_values"].shape == (len(source_ref["filled_names"]),)
            assert motion["filled_joint_values"].dtype == np.float32
            if source_ref["filled_names"]:
                max_error(motion["filled_joint_values"], home[[names.index(n) for n in source_ref["filled_names"]]],
                          clip + "/wrist-fill", 1e-7)
            errors["joint_pos_rad"] = max_error(motion["joint_pos"], joints, clip + "/joints", 1e-6)
            root_id = bodies.index("base_link")
            errors["root_pos_m"] = max_error(
                motion["body_pos_w"][:, root_id], root_pos, clip + "/root-pos", 1e-6
            )
            errors["root_rotation_rad"] = float(np.max((
                rotations(motion["body_quat_w"][:, root_id]) * rotation.inv()
            ).magnitude()))
            assert errors["root_rotation_rad"] < 1e-6, clip
            errors["joint_vel_rad_s"] = max_error(
                motion["joint_vel"], joint_velocity, clip + "/joint-velocity", 1e-4
            )
            errors["root_lin_vel_m_s"] = max_error(
                motion["body_lin_vel_w"][:, root_id], linear, clip + "/root-linear", 1e-4
            )
            errors["root_ang_vel_rad_s"] = max_error(
                motion["body_ang_vel_w"][:, root_id], angular, clip + "/root-angular", 1e-4
            )
            errors["quaternion_norm"] = float(np.max(np.abs(
                np.linalg.norm(motion["body_quat_w"], axis=-1) - 1
            )))
            assert errors["quaternion_norm"] < 1e-6, clip
            frames = np.unique(np.linspace(0, n - 1, min(samples, n)).astype(int))
            for t in frames:
                data.qpos[:] = np.concatenate([root_pos[t], root_quat[t], joints[t]])
                data.qvel[:] = np.concatenate([linear[t], body_angular[t], joint_velocity[t]])
                mujoco.mj_forward(model, data)
                actual_q = motion["body_quat_w"][t]
                sign = np.where((actual_q * data.xquat[1:]).sum(-1, keepdims=True) < 0, -1, 1)
                checks = {
                    "fk_body_pos_m": (motion["body_pos_w"][t], data.xpos[1:], 2e-6),
                    "fk_body_quat": (actual_q * sign, data.xquat[1:], 2e-6),
                }
                expected_velocity = np.empty((45, 6))
                for body in range(1, model.nbody):
                    mujoco.mj_objectVelocity(
                        model, data, mujoco.mjtObj.mjOBJ_XBODY, body, expected_velocity[body - 1], 0
                    )
                checks.update(
                    fk_body_lin_vel_m_s=(motion["body_lin_vel_w"][t], expected_velocity[:, 3:], 2e-4),
                    fk_body_ang_vel_rad_s=(motion["body_ang_vel_w"][t], expected_velocity[:, :3], 2e-4),
                )
                for key, (actual, expected, tolerance) in checks.items():
                    errors[key] = max(errors.get(key, 0), max_error(actual, expected, clip + "/" + key, tolerance))
        violation = np.maximum(model.jnt_range[1:, 0] - joints, joints - model.jnt_range[1:, 1]).clip(0)
        position_count = int((violation > 1e-6).sum())
        speed_count = int((np.abs(joint_velocity) > velocity_limits).sum())
        report["clips"].append({
            "name": clip, "frames": n, "duration_s": (n - 1) / 50,
            "input_frames": source_ref["input_frames"], "input_fps": source_ref["input_fps"],
            "source_dofs": len(source_ref["source_names"]), "filled_joints": source_ref["filled_names"],
            "output_sha256": sha256(path), "max_errors": errors,
            "fk_sample_frames": frames.tolist(), "position_limit_samples_gt_1e_6": position_count,
            "max_position_limit_excess_rad": float(violation.max()),
            "velocity_metadata_exceedance_samples": speed_count,
        })
        for key, value in errors.items():
            report["max_errors"][key] = max(report["max_errors"].get(key, 0), value)
        report["frame_count"] += n
        report["duration_s"] += (n - 1) / 50
        report["fk_frames_checked"] += len(frames)
        report["position_limit_samples_gt_1e_6"] += position_count
        report["velocity_metadata_exceedance_samples"] += speed_count
        print(f"[PASS] {clip}: {n} frames, {len(frames)} FK samples", flush=True)
    report["status"] = "passed"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--npz-dir", type=Path, default=ROOT / "data/npz/khan_mini/lafan1")
    parser.add_argument("--asset-dir", type=Path, default=ROOT / "assets/robots/khan_mini")
    parser.add_argument("--output", type=Path, default=ROOT / "logs/khan_lafan_data/validation.json")
    parser.add_argument("--frames-per-clip", type=int, default=8)
    args = parser.parse_args()
    if args.frames_per_clip < 1:
        parser.error("--frames-per-clip must be positive")
    report = {"status": "interrupted"}
    try:
        report = validate_dataset(args.input, args.npz_dir, args.asset_dir, args.frames_per_clip)
    except Exception as exc:
        report = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"[PASS] validation report: {args.output}")


if __name__ == "__main__":
    main()

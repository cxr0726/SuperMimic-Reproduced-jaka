"""Packed reference export preserves training semantics, not just NPZ shapes."""

from __future__ import annotations

import hashlib
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from supermimic.data.library_to_npz import export_library
from supermimic.envs.mdp.motion_library import MotionLibrary, read_npz_reference
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.observation_layout import critic_reference_extras, isaac_goal_command
from supermimic.robots.registry import get_robot_spec


def _pack(path: Path, robot: str = "khan_mini"):
    spec = get_robot_spec(robot)
    model = spec.load_model()
    rng = np.random.default_rng(991)
    n = 16
    time = np.arange(n) / 50
    position = np.column_stack((time*.2, time*-.1, .65 + .03*np.sin(time)))
    position[0, 2] = .02  # Export must not silently add reset clearance / floor lift.
    axis = np.array([.2, -.3, .4])
    axis /= np.linalg.norm(axis)
    half_angle = .13 + time*1.3
    quaternion = np.column_stack((np.cos(half_angle), np.sin(half_angle)[:, None]*axis))
    quaternion[::3] *= -1  # Equivalent signs must not change reference semantics.
    joints = rng.uniform(model.jnt_range[1:, 0], model.jnt_range[1:, 1],
                         size=(n, spec.num_joints))
    velocity = rng.normal(size=(n, 6 + spec.num_joints)) * 1.7
    ref = torch.tensor(np.concatenate((position, quaternion, joints, velocity), axis=1),
                       dtype=torch.float16)
    assert (ref[:, 3:7].float().norm(dim=-1)-1).abs().max() > 1e-5
    pack = {"ref": ref, "names": ["alpha", "beta"], "starts": [0, 9],
            "lengths": [9, 7], "robot": robot, "joint_names": spec.joint_names,
            "root_body_name": spec.root_body, "fps": 50.}
    torch.save(pack, path)
    return path, pack


def _library(paths, robot, *, compact=False):
    spec = get_robot_spec(robot)
    return MotionLibrary(paths, spec.tracked_bodies, torch.zeros(spec.num_joints),
                         device="cpu", compact=compact, robot_name=robot,
                         root_body_name=spec.root_body, reference_fk_compile=False)


def _arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


@pytest.mark.parametrize("robot", ["g1", "khan_mini"])
def test_packed_export_round_trip_is_lossless_for_training(tmp_path, robot):
    path, pack = _pack(tmp_path / "source.pt", robot)
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    output = tmp_path / "exported"
    export_library(path, output, robot=robot, chunk_size=3)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_hash
    assert sorted(p.name for p in output.glob("*.npz")) == ["alpha.npz", "beta.npz"]
    compact = _library([path], robot, compact=True)
    full = _library([output / "alpha.npz", output / "beta.npz"], robot)
    frames = torch.tensor([0, 2, 3, 8, 9, 11, 12, 15, 0])
    torch.testing.assert_close(full.ref71(frames), compact.ref71(frames), rtol=0, atol=0)
    # This would fail if full NPZ reconstructed normalized root quaternion or
    # transported root angular velocity instead of preserving the packed state.
    # Goal transforms are precomputed at a different batch size in full mode;
    # shared float32 quaternion reductions can differ by one ulp.
    torch.testing.assert_close(full.goal_window(frames, 5), compact.goal_window(frames, 5),
                               rtol=1e-6, atol=1e-7)
    if robot == "khan_mini":
        reference_windows = [library.reference_window(frames, 5) for library in (full, compact)]
        goals = [isaac_goal_command(r.root_pos_w, r.root_quat_w, r.joint_pos)
                 for r in reference_windows]
        extras = [critic_reference_extras(r.joint_vel, r.root_quat_w,
                                         r.root_ang_vel_w, r.root_lin_vel_w)
                  for r in reference_windows]
        torch.testing.assert_close(goals[0], goals[1], rtol=0, atol=0)
        torch.testing.assert_close(extras[0], extras[1], rtol=0, atol=0)
    actual, expected = full.bodies(frames), compact.bodies(frames)
    for field in ("pos_w", "quat_w", "lin_vel_w", "ang_vel_w"):
        torch.testing.assert_close(getattr(actual, field), getattr(expected, field),
                                   rtol=2e-5, atol=2e-6)
    assert full._ref[0, 2] == compact._ref[0, 2]
    spec = get_robot_spec(robot)
    for name, start, length in zip(pack["names"], pack["starts"], pack["lengths"], strict=True):
        data = _arrays(output / f"{name}.npz")
        assert data["reference_state_format"].item() == "supermimic.compact_reference.v1"
        assert data["reference_state"].dtype == np.float32
        np.testing.assert_array_equal(data["reference_state"], pack["ref"][start:start+length].float())
        np.testing.assert_array_equal(data["joint_pos"], pack["ref"][start:start+length, 7:7+spec.num_joints].float())
        np.testing.assert_array_equal(data["joint_vel"], pack["ref"][start:start+length, 13+spec.num_joints:].float())
        assert tuple(data["joint_names"]) == spec.joint_names
        assert tuple(data["body_names"]) == spec.tracked_bodies
        assert float(data["fps"].reshape(-1)[0]) == 50.
        assert data["body_pos_w"].shape == (length, len(spec.tracked_bodies), 3)
        for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                    "body_lin_vel_w", "body_ang_vel_w"):
            assert data[key].dtype == np.float32 and np.isfinite(data[key]).all()
        np.testing.assert_allclose(np.linalg.norm(data["body_quat_w"], axis=-1), 1,
                                   atol=1e-5, rtol=0)


@pytest.mark.parametrize("robot", ["g1", "khan_mini"])
def test_export_velocity_matches_independent_mujoco_oracle(tmp_path, robot):
    path, _ = _pack(tmp_path / "source.pt", robot)
    output = tmp_path / "npz"
    export_library(path, output, robot=robot, chunk_size=2, all_bodies=True)
    compact = _library([path], robot, compact=True)
    spec = get_robot_spec(robot)
    model = spec.load_model()
    data = mujoco.MjData(model)
    arrays = _arrays(output / "alpha.npz")
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(name))
           for name in arrays["body_names"]]
    assert len(ids) == model.nbody-1
    qpos, qvel = compact.ref_qpos_qvel(torch.tensor([0, 4, 8]))
    velocity = np.empty(6)
    for local, qp, qv in zip((0, 4, 8), qpos.double().numpy(), qvel.double().numpy(), strict=True):
        qp[3:7] /= np.linalg.norm(qp[3:7])
        data.qpos[:] = qp
        data.qvel[:] = qv
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(arrays["body_pos_w"][local], data.xpos[ids], atol=2e-5, rtol=2e-5)
        actual_quat, expected_quat = arrays["body_quat_w"][local], data.xquat[ids]
        sign = np.where((actual_quat*expected_quat).sum(-1, keepdims=True) < 0, -1, 1)
        np.testing.assert_allclose(actual_quat*sign, expected_quat, atol=2e-5, rtol=2e-5)
        for column, body in enumerate(ids):
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, body, velocity, 0)
            np.testing.assert_allclose(arrays["body_ang_vel_w"][local, column], velocity[:3], atol=2e-4, rtol=2e-5)
            np.testing.assert_allclose(arrays["body_lin_vel_w"][local, column], velocity[3:], atol=2e-4, rtol=2e-5)


def test_selection_and_preflight_no_overwrite(tmp_path):
    path, _ = _pack(tmp_path / "source.pt")
    first = tmp_path / "first"
    export_library(path, first, clips=["beta"])
    assert sorted(p.name for p in first.glob("*.npz")) == ["beta.npz"]
    limited = tmp_path / "limited"
    export_library(path, limited, limit=1)
    assert sorted(p.name for p in limited.glob("*.npz")) == ["alpha.npz"]
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "beta.npz").write_bytes(b"preserve existing bytes")
    with pytest.raises(FileExistsError):
        export_library(path, existing)
    assert not (existing / "alpha.npz").exists()
    assert (existing / "beta.npz").read_bytes() == b"preserve existing bytes"
    with pytest.raises(ValueError):
        export_library(path, tmp_path / "unknown", clips=["not_in_pack"])
    assert not list((tmp_path / "unknown").glob("*.npz"))


def test_full_export_query_has_no_online_fk_even_with_compile_flag(tmp_path, monkeypatch):
    path, _ = _pack(tmp_path / "source.pt")
    output = tmp_path / "full"
    export_library(path, output, limit=1)

    def forbidden(*args, **kwargs):
        raise AssertionError("Full NPZ mode must use stored body arrays")

    monkeypatch.setattr(torch, "compile", forbidden)
    monkeypatch.setattr(G1Kinematics, "forward", forbidden)
    monkeypatch.setattr(G1Kinematics, "__call__", forbidden)
    spec = get_robot_spec("khan_mini")
    library = MotionLibrary([output / "alpha.npz"], spec.tracked_bodies, torch.zeros(27),
                            device="cpu", compact=False, robot_name=spec.name,
                            root_body_name=spec.root_body, reference_fk_compile=True)
    assert not library.reference_fk_compiled and not hasattr(library, "_fk")
    assert library.bodies(torch.tensor([0, 3, 8])).pos_w.shape == (3, 15, 3)


def test_legacy_full_npz_without_reference_extension_still_uses_body_root(tmp_path):
    path, _ = _pack(tmp_path / "source.pt")
    output = tmp_path / "export"
    export_library(path, output, limit=1)
    fields = _arrays(output / "alpha.npz")
    fields.pop("reference_state")
    fields.pop("reference_state_format")
    legacy = tmp_path / "legacy.npz"
    np.savez(legacy, **fields)
    spec = get_robot_spec("khan_mini")
    ref, _ = read_npz_reference(legacy, robot_name=spec.name, root_body_name=spec.root_body,
                               joint_names=spec.joint_names, body_names=spec.tracked_bodies)
    root = list(fields["body_names"]).index(spec.root_body)
    np.testing.assert_array_equal(ref[:, :3], fields["body_pos_w"][:, root])
    np.testing.assert_array_equal(ref[:, 3:7], fields["body_quat_w"][:, root])
    np.testing.assert_array_equal(ref[:, 37:40], fields["body_ang_vel_w"][:, root])


@pytest.mark.parametrize("problem", ["robot", "joint_names", "root", "fps", "starts",
                                     "lengths", "width", "nan", "zero_quaternion",
                                     "duplicate_name", "traversal", "backslash", "non_numeric"])
def test_invalid_pack_fails_before_output(tmp_path, problem):
    path, pack = _pack(tmp_path / "bad.pt")
    if problem == "robot":
        pack["robot"] = "g1"
    elif problem == "joint_names":
        pack["joint_names"] = pack["joint_names"][::-1]
    elif problem == "root":
        pack["root_body_name"] = "waist_yaw_Link"
    elif problem == "fps":
        pack["fps"] = 30.
    elif problem == "starts":
        pack["starts"] = [0, 8]
    elif problem == "lengths":
        pack["lengths"] = [9, 0]
    elif problem == "width":
        pack["ref"] = pack["ref"][:, :-1]
    elif problem == "nan":
        pack["ref"][0, 7] = float("nan")
    elif problem == "zero_quaternion":
        pack["ref"][0, 3:7] = 0
    elif problem == "duplicate_name":
        pack["names"] = ["alpha", "alpha"]
    elif problem == "traversal":
        pack["names"] = ["../outside", "beta"]
    elif problem == "backslash":
        pack["names"] = ["folder\\outside", "beta"]
    else:
        pack["ref"] = [["not numeric"]]
    torch.save(pack, path)
    output = tmp_path / "output"
    with pytest.raises((ValueError, TypeError)):
        export_library(path, output)
    assert not list(output.glob("*.npz"))
    assert not (tmp_path / "outside.npz").exists()


@pytest.mark.parametrize("problem", ["marker", "missing_marker", "missing_reference",
                                     "width", "joint_position", "joint_velocity",
                                     "root_position", "root_orientation", "root_velocity",
                                     "root_angular_velocity"])
def test_full_loader_rejects_inconsistent_reference_extension(tmp_path, problem):
    path, _ = _pack(tmp_path / "source.pt")
    output = tmp_path / "export"
    export_library(path, output, limit=1)
    arrays = _arrays(output / "alpha.npz")
    if problem == "marker":
        arrays["reference_state_format"] = np.array("future.unsupported")
    elif problem == "missing_marker":
        arrays.pop("reference_state_format")
    elif problem == "missing_reference":
        arrays.pop("reference_state")
    elif problem == "width":
        arrays["reference_state"] = arrays["reference_state"][:, :-1]
    else:
        column = {"joint_position": 7, "joint_velocity": 40, "root_position": 0,
                  "root_orientation": 3, "root_velocity": 34,
                  "root_angular_velocity": 37}[problem]
        arrays["reference_state"][0, column] += .5
    corrupt = tmp_path / "bad.npz"
    np.savez(corrupt, **arrays)
    spec = get_robot_spec("khan_mini")
    with pytest.raises(ValueError):
        read_npz_reference(corrupt, robot_name=spec.name, root_body_name=spec.root_body,
                           joint_names=spec.joint_names, body_names=spec.tracked_bodies)

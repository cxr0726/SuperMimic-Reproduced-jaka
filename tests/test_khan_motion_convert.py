"""CPU checks for Khan PKL mapping, time conventions and exported link velocities."""

from copy import deepcopy

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from supermimic.data.khan_motion_convert import convert_clip, prepare_khan_motion
from supermimic.robots.khan_mini import DEFAULT_ASSET_DIR, get_khan_mini_spec, load_profile

pytestmark = pytest.mark.skipif(
    not (DEFAULT_ASSET_DIR / "robot.xml").is_file(), reason="Khan_mini assets not built"
)
MISSING_WRISTS = {"Left_wrist_yaw_joint", "Right_wrist_yaw_joint"}


@pytest.fixture(scope="module")
def profile():
    return load_profile()


def synthetic_clip(profile, *, fps=20.0, duration=0.5, dofs=27):
    frames = round(duration * fps) + 1
    times = np.arange(frames, dtype=np.float64) / fps
    angles = np.column_stack((np.full(frames, 0.2), np.full(frames, -0.1), 0.3 + 0.4 * times))
    xyzw = Rotation.from_euler("xyz", angles).as_quat()
    names = [name for name in profile["joint_names"]
             if dofs == 27 or name not in MISSING_WRISTS]
    home = np.array([profile["default_joint_pos"][name] for name in names])
    rates = 0.02 * (np.arange(len(names)) + 1) / len(names)
    return {
        "fps": fps,
        "root_pos": np.column_stack((0.1 + 0.2 * times, -0.3 + 0.1 * times,
                                     0.4 + 0.05 * times)),
        "root_rot": xyzw[:, [3, 0, 1, 2]],
        "dof_pos": home + times[:, None] * rates,
    }


def assert_quaternion_equivalent(actual, expected, atol=1e-8):
    sign = np.where(np.sum(actual * expected, axis=-1, keepdims=True) < 0, -1.0, 1.0)
    np.testing.assert_allclose(actual * sign, expected, atol=atol, rtol=0)


def test_25_dof_mapping_preserves_neck_and_fills_only_missing_wrists(profile):
    clip = synthetic_clip(profile, dofs=25)
    # Distinct columns expose positional insertion bugs, including the last two neck columns.
    clip["dof_pos"][:] = 0.01 * (np.arange(25) + 1)
    original = deepcopy(clip)
    motion = prepare_khan_motion(clip, profile, output_fps=20.0)
    source_names = [name for name in profile["joint_names"] if name not in MISSING_WRISTS]
    assert motion.joint_pos.shape == (11, 27)
    for index, name in enumerate(profile["joint_names"]):
        if name in MISSING_WRISTS:
            np.testing.assert_allclose(motion.joint_pos[:, index], 0.3, atol=1e-15)
            np.testing.assert_allclose(motion.joint_vel[:, index], 0.0, atol=1e-15)
        else:
            expected = original["dof_pos"][:, source_names.index(name)]
            np.testing.assert_allclose(motion.joint_pos[:, index], expected, atol=1e-15)
    assert motion.joint_pos[0, -2] == pytest.approx(0.24)
    assert motion.joint_pos[0, -1] == pytest.approx(0.25)
    for key in ("root_pos", "root_rot", "dof_pos"):
        np.testing.assert_array_equal(clip[key], original[key])


def test_named_columns_can_be_reordered_without_losing_joint_identity(profile):
    clip = synthetic_clip(profile)
    expected = prepare_khan_motion(clip, profile, output_fps=20.0)
    shuffled = deepcopy(clip)
    permutation = np.random.default_rng(2026).permutation(27)
    shuffled["dof_pos"] = clip["dof_pos"][:, permutation]
    shuffled["joint_names"] = [profile["joint_names"][index] for index in permutation]
    actual = prepare_khan_motion(shuffled, profile, output_fps=20.0)
    np.testing.assert_allclose(actual.joint_pos, expected.joint_pos, atol=1e-14)
    np.testing.assert_allclose(actual.joint_vel, expected.joint_vel, atol=1e-13)


def test_explicit_wxyz_and_xyzw_source_orders_produce_same_motion(profile):
    wxyz = synthetic_clip(profile)
    xyzw = deepcopy(wxyz)
    xyzw["root_rot"] = wxyz["root_rot"][:, [1, 2, 3, 0]]
    a = prepare_khan_motion(wxyz, profile, quat_order="wxyz")
    b = prepare_khan_motion(xyzw, profile, quat_order="xyzw")
    assert_quaternion_equivalent(a.root_quat, b.root_quat, atol=1e-14)
    np.testing.assert_allclose(a.root_ang_vel_w, b.root_ang_vel_w, atol=1e-12)


@pytest.mark.parametrize(("fps", "duration"), [(30.0, 2.0), (50.0, 3.0)])
def test_timebase_uses_sample_intervals_without_trim_or_height_shift(profile, fps, duration):
    clip = synthetic_clip(profile, fps=fps, duration=duration)
    motion = prepare_khan_motion(clip, profile)
    assert motion.num_frames == round(duration * 50.0) + 1
    np.testing.assert_allclose(motion.root_pos[0], clip["root_pos"][0], atol=1e-14)
    np.testing.assert_allclose(motion.root_pos[-1], clip["root_pos"][-1], atol=1e-14)
    np.testing.assert_allclose(motion.joint_pos[0], clip["dof_pos"][0], atol=1e-14)
    np.testing.assert_allclose(motion.joint_pos[-1], clip["dof_pos"][-1], atol=1e-14)
    np.testing.assert_allclose(
        motion.root_lin_vel_w, np.broadcast_to([0.2, 0.1, 0.05], motion.root_pos.shape),
        atol=1e-12,
    )


def test_shortest_path_slerp_sign_flips_keep_constant_world_angular_velocity(profile):
    clip = synthetic_clip(profile, fps=5.0, duration=2.0)
    times = np.arange(11) / 5.0
    xyzw = Rotation.from_rotvec(np.column_stack((times * 0, times * 0, times * 0.4))).as_quat()
    clip["root_rot"] = xyzw[:, [3, 0, 1, 2]]
    clip["root_rot"][1::2] *= -1
    motion = prepare_khan_motion(clip, profile)
    output_times = np.arange(101) / 50.0
    expected = Rotation.from_rotvec(
        np.column_stack((output_times * 0, output_times * 0, output_times * 0.4))
    ).as_quat()[:, [3, 0, 1, 2]]
    assert_quaternion_equivalent(motion.root_quat, expected, atol=1e-10)
    np.testing.assert_allclose(
        motion.root_ang_vel_w, np.broadcast_to([0, 0, 0.4], (101, 3)), atol=1e-9,
    )


def test_npz_schema_and_independent_mujoco_body_pose_velocity_parity(profile, tmp_path):
    clip = synthetic_clip(profile)
    model = get_khan_mini_spec().compile()
    output = tmp_path / "synthetic_motion.npz"
    source = "synthetic_source.pkl"
    convert_clip(clip, output, model=model, source=source, clip_name="synthetic", output_fps=20.0)
    motion = prepare_khan_motion(clip, profile, output_fps=20.0)
    with np.load(output, allow_pickle=False) as saved:
        required = {"fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                    "body_lin_vel_w", "body_ang_vel_w", "joint_names", "body_names",
                    "source", "input_fps"}
        assert required <= set(saved.files)
        assert saved["joint_names"].tolist() == profile["joint_names"]
        assert saved["body_names"].tolist() == [model.body(i).name for i in range(1, 46)]
        assert str(saved["source"].item()) == source
        assert float(saved["fps"][0]) == 20.0 and float(saved["input_fps"][0]) == 20.0
        assert saved["joint_pos"].shape == (11, 27)
        assert saved["body_pos_w"].shape == (11, 45, 3)
        assert saved["body_quat_w"].shape == (11, 45, 4)
        for name in required - {"joint_names", "body_names", "source"}:
            assert saved[name].dtype == np.float32
            assert np.isfinite(saved[name]).all()

        data = mujoco.MjData(model)
        velocity = np.empty(6)
        for frame in (0, 5, 10):
            data.qpos[:3] = motion.root_pos[frame]
            data.qpos[3:7] = motion.root_quat[frame]
            data.qpos[7:] = motion.joint_pos[frame]
            data.qvel[:3] = [0.2, 0.1, 0.05]
            rotation = Rotation.from_quat(motion.root_quat[frame, [1, 2, 3, 0]])
            data.qvel[3:6] = rotation.inv().apply([0, 0, 0.4])
            data.qvel[6:] = 0.02 * (np.arange(27) + 1) / 27
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(saved["body_pos_w"][frame], data.xpos[1:], atol=2e-7)
            assert_quaternion_equivalent(saved["body_quat_w"][frame], data.xquat[1:], atol=1e-7)
            for body_id in range(1, 46):
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY,
                                        body_id, velocity, 0)
                np.testing.assert_allclose(saved["body_lin_vel_w"][frame, body_id - 1],
                                           velocity[3:], atol=2e-6)
                np.testing.assert_allclose(saved["body_ang_vel_w"][frame, body_id - 1],
                                           velocity[:3], atol=2e-6)


@pytest.mark.parametrize("case", ["zero_fps", "nan_fps", "nan_dof", "zero_quaternion",
                                 "short_root", "bad_position_shape"])
def test_invalid_time_or_numerical_data_is_rejected(profile, case):
    clip = synthetic_clip(profile)
    if case == "zero_fps":
        clip["fps"] = 0.0
    elif case == "nan_fps":
        clip["fps"] = np.nan
    elif case == "nan_dof":
        clip["dof_pos"][0, 4] = np.nan
    elif case == "zero_quaternion":
        clip["root_rot"][0] = 0.0
    elif case == "short_root":
        clip["root_rot"] = clip["root_rot"][:-1]
    elif case == "bad_position_shape":
        clip["root_pos"] = clip["root_pos"][:, :2]
    with pytest.raises(ValueError):
        prepare_khan_motion(clip, profile)


def test_unsupported_or_ambiguous_joint_layouts_are_rejected(profile):
    clip = synthetic_clip(profile)
    clip["dof_pos"] = clip["dof_pos"][:, :26]
    with pytest.raises(ValueError):
        prepare_khan_motion(clip, profile)
    clip = synthetic_clip(profile)
    clip["joint_names"] = profile["joint_names"].copy()
    clip["joint_names"][-1] = clip["joint_names"][-2]
    with pytest.raises(ValueError):
        prepare_khan_motion(clip, profile)
    clip = synthetic_clip(profile)
    # Naming makes clear that a non-wrist DOF is absent; it must not be padded.
    clip["dof_pos"] = clip["dof_pos"][:, :-1]
    clip["joint_names"] = profile["joint_names"][:-1]
    with pytest.raises(ValueError):
        prepare_khan_motion(clip, profile)

"""CPU checks of robot-dependent reference layouts and transition recording."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from supermimic.data.khan_motion_convert import convert_clip
from supermimic.data.recorder import ReplayRecorder, ReplayRecorderCfg
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.envs.sim_context import SimContext
from supermimic.envs.state_io import StateIO
from supermimic.robots.khan_mini import load_profile
from supermimic.robots.registry import get_robot_spec


@pytest.fixture
def khan_motion(tmp_path):
    profile = load_profile()
    n = 7
    times = np.arange(n) / 50
    home = np.array([profile["default_joint_pos"][name] for name in profile["joint_names"]])
    clip = {
        "fps": 50,
        "root_pos": np.column_stack((times * 0.2, times * 0.1, 0.65 + times * 0.01)),
        "root_rot": np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        "dof_pos": home + times[:, None] * np.linspace(0.01, 0.05, 27),
    }
    path = tmp_path / "synthetic.npz"
    convert_clip(clip, path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return path, arrays, home


def khan_library(paths, home, **kwargs):
    spec = get_robot_spec("khan_mini")
    return MotionLibrary(paths, spec.tracked_bodies, home, device="cpu",
                         robot_name=spec.name, root_body_name=spec.root_body,
                         joint_names=spec.joint_names, **kwargs)


def test_khan_compact_matches_full_and_clip_end(khan_motion):
    path, arrays, home = khan_motion
    full = khan_library([path], home)
    compact = khan_library([path], home, compact=True, store_dtype=torch.float32)
    frame = torch.tensor([0, 2, 6])
    assert full.ref_dim == 67 and full.goal_feat_dim == 64
    assert full.ref71(frame).shape == (3, 67)
    assert full.goal_window(frame, 10).shape == (3, 640)
    qp, qv = compact.ref_qpos_qvel(frame)
    assert qp.shape == (3, 34) and qv.shape == (3, 33)
    torch.testing.assert_close(full.ref(frame).joint_pos, torch.tensor(arrays["joint_pos"])[frame])
    torch.testing.assert_close(full.goal_window(frame, 10), compact.goal_window(frame, 10))
    for name in ("pos_w", "lin_vel_w", "ang_vel_w"):
        torch.testing.assert_close(getattr(full.bodies(frame), name), getattr(compact.bodies(frame), name),
                                   rtol=2e-5, atol=2e-6)
    expected = full.goal_feat[6].repeat(10)
    torch.testing.assert_close(full.goal_window(torch.tensor([6]), 10)[0], expected)
    torch.testing.assert_close(full.joint_vel, torch.tensor(arrays["joint_vel"]))
    torch.testing.assert_close(full.root_lin_vel_w, full.ref(torch.arange(7)).root_lin_vel_w)
    torch.testing.assert_close(full.root_ang_vel_w, full.ref(torch.arange(7)).root_ang_vel_w)


def test_each_clip_resolves_its_own_body_order(khan_motion, tmp_path):
    path, arrays, home = khan_motion
    reordered = {name: value.copy() for name, value in arrays.items()}
    permutation = np.arange(len(arrays["body_names"]))[::-1]
    reordered["body_names"] = arrays["body_names"][permutation]
    for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        reordered[key] = arrays[key][:, permutation]
    second = tmp_path / "reordered.npz"
    np.savez(second, **reordered)
    lib = khan_library([path, second], home)
    torch.testing.assert_close(lib.ref71(torch.arange(7)), lib.ref71(torch.arange(7, 14)))
    torch.testing.assert_close(lib.bodies(torch.arange(7)).pos_w, lib.bodies(torch.arange(7, 14)).pos_w)


@pytest.mark.parametrize("issue", ["fps", "joint_names", "joint_width", "quaternion", "robot"])
def test_reject_incompatible_npz(khan_motion, tmp_path, issue):
    _, arrays, home = khan_motion
    if issue == "fps":
        arrays["fps"] = np.array([60.0])
    elif issue == "joint_names":
        arrays["joint_names"] = arrays["joint_names"][::-1]
    elif issue == "joint_width":
        arrays["joint_pos"] = arrays["joint_pos"][:, :-1]
    elif issue == "quaternion":
        arrays["body_quat_w"][:, 0] = 0
    else:
        arrays["robot"] = np.array("g1")
    path = tmp_path / "incompatible.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError):
        khan_library([path], home)


def test_packed_khan_metadata_and_legacy_g1(khan_motion, tmp_path):
    path, _, home = khan_motion
    full = khan_library([path], home)
    packed = {"ref": full.ref71(torch.arange(7)), "names": ["synthetic"],
              "starts": [0], "lengths": [7]}
    cache = tmp_path / "khan.pt"
    torch.save(packed, cache)
    with pytest.raises(ValueError, match="metadata"):
        khan_library([cache], home, compact=True)
    spec = get_robot_spec("khan_mini")
    packed.update(robot=spec.name, joint_names=spec.joint_names,
                  root_body_name=spec.root_body, fps=50)
    torch.save(packed, cache)
    compact = khan_library([cache], home, compact=True, store_dtype=torch.float32)
    torch.testing.assert_close(compact.goal_window(torch.tensor([1]), 10), full.goal_window(torch.tensor([1]), 10))
    g1ref = torch.zeros(3, 71)
    g1ref[:, 3] = 1
    g1cache = tmp_path / "legacy_g1.pt"
    torch.save({"ref": g1ref, "names": ["legacy"], "starts": [0], "lengths": [3]}, g1cache)
    g1 = MotionLibrary([g1cache], get_robot_spec("g1").tracked_bodies,
                       np.zeros(29), device="cpu", compact=True)
    assert g1.goal_window(torch.tensor([0]), 10).shape == (1, 680)
    with pytest.raises(ValueError, match="columns"):
        khan_library([g1cache], home, compact=True)


def test_compact_builder_round_trip(khan_motion, tmp_path):
    path, _, home = khan_motion
    script_path = Path(__file__).resolve().parents[1] / "scripts/data/build_library.py"
    module_spec = importlib.util.spec_from_file_location("khan_test_build_library", script_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    out = tmp_path / "packed" / "khan.pt"
    module.main(path.parent, out, workers=1, robot="khan_mini")
    pack = torch.load(out, map_location="cpu", weights_only=False)
    full = khan_library([path], home)
    torch.testing.assert_close(pack["ref"], full.ref71(torch.arange(7)).half())
    compact = khan_library([out], home, compact=True)
    assert compact.num_frames == 7 and compact.num_clips == 1
    assert compact.goal_window(torch.tensor([0]), 10).shape == (1, 640)
    torch.testing.assert_close(compact.goal_window(torch.tensor([0]), 10),
                               full.goal_window(torch.tensor([0]), 10), rtol=0, atol=7e-4)
    assert pack["robot"] == "khan_mini" and pack["fps"] == 50


def fake_env(num_joints):
    n = 2
    origins = torch.tensor([[0., 0., 0.], [3., 0., 0.]])
    qpos = torch.zeros(n, num_joints + 7)
    qpos[:, :3] = origins + torch.tensor([0.1, 0.2, 0.65])
    qpos[:, 3] = 1
    sim_data = SimpleNamespace(qpos=qpos, qvel=torch.zeros(n, num_joints + 6),
                               qacc_warmstart=torch.ones(n, num_joints + 6))
    nominal_com = np.array([[0., 0., 0.], [0.01, 0.02, 0.03]])
    model = SimpleNamespace(geom_friction=torch.tensor([[[0.6, 0., 0.]], [[0.9, 0., 0.]]]),
                            body_ipos=torch.tensor(nominal_com, dtype=torch.float32))
    robot = SimpleNamespace(
        data=SimpleNamespace(encoder_bias=torch.full((n, num_joints), 0.005)),
        body_names=["anchor"],
        indexing=SimpleNamespace(geom_ids=torch.tensor([0]), body_ids=torch.tensor([1])),
        find_geoms=lambda _regex: ([0], ["foot"]),
    )

    class Scene(dict):
        pass

    scene = Scene(robot=robot)
    scene.env_origins = origins
    sim = SimpleNamespace(data=sim_data, model=model, mj_model=SimpleNamespace(body_ipos=nominal_com),
                          forward=lambda: None)
    return SimpleNamespace(sim=sim, scene=scene, num_envs=n, device="cpu")


@pytest.mark.parametrize("num_joints", [27, 29])
def test_dynamic_state_context_and_recorder_terminal_rows(num_joints):
    env = fake_env(num_joints)
    io = StateIO(env)
    original = io.read()
    assert original.shape == (2, num_joints * 2 + 13)
    modified = original[1:].clone()
    modified[:, 0] += 0.4
    io.write(torch.tensor([1]), modified)
    torch.testing.assert_close(io.read()[1:], modified)
    assert env.sim.data.qacc_warmstart[1].count_nonzero() == 0
    context = SimContext(env, "foot", "anchor")
    xi = context.read()
    assert xi.shape == (2, num_joints + 4)
    torch.testing.assert_close(xi[:, 0], torch.tensor([0.6, 0.9]))
    torch.testing.assert_close(xi[:, 1:1 + num_joints], env.scene["robot"].data.encoder_bias)
    torch.testing.assert_close(xi[:, -3:], torch.zeros(2, 3))
    command = SimpleNamespace(frame=torch.tensor([0, 1]), library=SimpleNamespace(frame_clip=torch.tensor([0, 0])))
    env.command_manager = SimpleNamespace(get_term=lambda _name: command)
    action = torch.zeros(2, num_joints)
    env.action_manager = SimpleNamespace(action=action, get_term=lambda _name: SimpleNamespace(executed_raw=action))
    env.reward_buf = torch.ones(2)
    env.reset_terminated = torch.tensor([True, False])
    recorder = ReplayRecorder(ReplayRecorderCfg(params={"capacity": 8, "foot_geom_regex": "foot", "com_body_name": "anchor"}), env)
    recorder.record_post_reset(torch.tensor([0, 1]))
    recorder.record_pre_reset(torch.tensor([0]))
    assert recorder.buffer.s.shape == (2, 8, 13 + 2 * num_joints)
    assert recorder.buffer.a.shape == (2, 8, num_joints)
    assert recorder.buffer.xi.shape == (2, 8, num_joints + 4)
    assert bool(recorder.buffer.terminal[0, 1])
    assert recorder.buffer.size == 2

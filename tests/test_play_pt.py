"""Named PT playback keeps exact references and the established actor interface."""

from __future__ import annotations

import hashlib
import importlib.util
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from supermimic.data import packed_subset
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.robots.registry import get_robot_spec


def _pack(path, dtype=torch.float16):
    robot = get_robot_spec("khan_mini")
    lengths = [3, 7, 5]
    reference = torch.randn(15, robot.state_dim, generator=torch.Generator().manual_seed(961))
    reference[:, 3:7] = torch.tensor([1., 0., 0., 0.])
    reference = reference.to(dtype)
    payload = {
        "ref": reference,
        "names": ["unselected", "mixed__pose", "lafan1__walk"],
        "starts": [0, 3, 10], "lengths": lengths,
        "robot": robot.name, "joint_names": robot.joint_names,
        "root_body_name": robot.root_body, "fps": 50.,
        "source_inventory_sha256": "source-inventory",
    }
    torch.save(payload, path)
    return payload


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_named_selection_preserves_source_order_values_dtype_and_file(tmp_path, dtype):
    path = tmp_path / "library.pt"
    original = _pack(path, dtype)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = packed_subset.load_packed_selection(path, ["lafan1__walk", "mixed__pose"])

    assert result["names"] == ["mixed__pose", "lafan1__walk"]
    assert result["starts"] == [0, 7] and result["lengths"] == [7, 5]
    assert result["ref"].dtype == dtype
    assert torch.equal(result["ref"], original["ref"][3:])
    assert result["joint_names"] == original["joint_names"]
    assert result["root_body_name"] == "base_link"
    # A selected playback buffer is not a writable view into the source file.
    result["ref"][0, 0] += 1
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert torch.equal(torch.load(path, weights_only=False)["ref"], original["ref"])


@pytest.mark.parametrize("names", [[], ["mixed__pose", "mixed__pose"], ["not_in_library"]])
def test_invalid_selection_never_silently_plays_another_motion(tmp_path, names):
    path = tmp_path / "library.pt"
    _pack(path)
    with pytest.raises(ValueError):
        packed_subset.load_packed_selection(path, names)


@pytest.mark.parametrize("damage", ["fps", "joint_names", "nan", "quaternion"])
def test_selection_rejects_bad_layout_or_selected_reference_state(tmp_path, damage):
    path = tmp_path / "library.pt"
    payload = _pack(path)
    if damage == "fps":
        payload["fps"] = 30.
    elif damage == "joint_names":
        payload["joint_names"] = tuple(reversed(payload["joint_names"]))
    elif damage == "nan":
        payload["ref"][3, 0] = float("nan")
    else:
        payload["ref"][3, 3:7] = 0
    torch.save(payload, path)
    with pytest.raises(ValueError):
        packed_subset.load_packed_selection(path, ["mixed__pose"])


def test_runtime_contains_only_requested_motion_and_fk_matches_full_pt(tmp_path):
    path = tmp_path / "library.pt"
    original = _pack(path)
    robot = get_robot_spec("khan_mini")
    kwargs = {
        "body_names": robot.tracked_bodies, "default_joint_pos": torch.zeros(robot.num_joints),
        "device": "cpu", "compact": True, "robot_name": robot.name, "root_body_name": robot.root_body,
    }
    full = MotionLibrary([path], **kwargs)
    selected = MotionLibrary([path], packed_clip_names=("mixed__pose",), **kwargs)

    assert [clip.name for clip in selected.clips] == ["mixed__pose"]
    assert selected.num_clips == 1 and selected.num_frames == 7
    assert selected.clip_start.tolist() == [0] and selected.clip_last.tolist() == [6]
    assert selected._ref.numel() == 7 * robot.state_dim
    torch.testing.assert_close(selected.ref71(torch.arange(7)), original["ref"][3:10].float(), rtol=0, atol=0)
    actual, expected = selected.bodies(torch.arange(7)), full.bodies(torch.arange(3, 10))
    for key in ("pos_w", "quat_w", "lin_vel_w", "ang_vel_w"):
        torch.testing.assert_close(getattr(actual, key), getattr(expected, key), rtol=0, atol=0)


def _play_module(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("play_pt_test_entry", scripts / "play_pt.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("viewer", ["native", "viser"])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("steps", [None, 3])
def test_viewer_receives_compact_selection_and_actor_contract_and_closes(tmp_path, monkeypatch, viewer, fail, steps):
    module = _play_module(monkeypatch)
    library = tmp_path / "library.pt"
    _pack(library)
    captured = {}

    class FakePolicy:
        cfg = SimpleNamespace(history=5, future_frames=5, observation_layout="isaac_khan", goal_dim=155)

        def __call__(self, policy_obs, goal):
            assert policy_obs.shape == (2, 465) and goal.shape == (2, 155)
            return torch.ones(2, 27)

    cfg = {"robot": "khan_mini", "world_model": {"height_threshold": .19, "tilt_threshold": .7}, "env": {
        "future_frames": 5, "anchor_obs": False, "reset_root_z_offset": .05,
        "library_fp16": True,
    }}
    monkeypatch.setattr(module, "load_policy_from_checkpoint", lambda path, device: (FakePolicy(), cfg))

    class FakeEnv:
        def close(self):
            captured["closed"] = True

        def reset(self):
            captured["resets"] = captured.get("resets", 0) + 1
            return {"policy": torch.zeros(2, 465), "goal": torch.zeros(2, 155)}, {}

    def make_env(args, device):
        captured.update(args=args, device=device)
        return FakeEnv()

    class FakeViewer:
        def __init__(self, env, policy):
            self.env, self.policy = env, policy

        def run(self, **kwargs):
            captured["run_kwargs"] = kwargs
            for _ in range(2):
                obs, _ = self.env.reset()
                assert torch.equal(self.policy(obs), torch.ones(2, 27))
            if fail:
                raise RuntimeError("viewer failed")

    monkeypatch.setattr(module, "make_env", make_env)
    monkeypatch.setattr(module, "RslRlVecEnvWrapper", lambda env: env)
    monkeypatch.setattr(module, "NativeMujocoViewer", FakeViewer)
    monkeypatch.setattr(module, "ViserPlayViewer", FakeViewer)
    arguments = {"ckpt": tmp_path / "actor.pt", "library": library, "clips": ["mixed__pose"],
                 "num_envs": 2, "device": "cpu", "viewer": viewer, "ghost": False,
                 "reference_fk_compile": True, "steps": steps}
    if fail:
        with pytest.raises(RuntimeError, match="viewer failed"):
            module.main(**arguments)
    else:
        module.main(**arguments)

    args = captured["args"]
    assert args.npz_files == [str(library)]
    assert args.packed_clip_names == ("mixed__pose",)
    assert args.compact_library and args.reference_fk_compile
    assert args.num_envs == 2 and args.history_length == 5 and args.future_frames == 5
    assert args.observation_layout == "isaac_khan" and args.reset_root_z_offset == .05
    assert args.root_height_threshold == .19 and args.root_tilt_threshold == .7
    assert args.sampling_mode == "start" and args.episode_length_s == 1e5
    assert not args.debug_vis and not args.record_transitions
    assert not args.domain_randomization and not args.obs_noise and not args.reset_noise
    assert args.action_noise_std == 0 and args.actuator_delay is False
    assert captured["run_kwargs"] == ({} if steps is None else {"num_steps": steps})
    assert captured["resets"] == 2 and captured["closed"]


def test_nonpositive_environment_count_fails_before_loading_actor(tmp_path, monkeypatch):
    module = _play_module(monkeypatch)

    def unexpected(*args):
        raise AssertionError("actor should not be loaded for invalid num_envs")

    monkeypatch.setattr(module, "load_policy_from_checkpoint", unexpected)
    with pytest.raises(ValueError, match="positive"):
        module.main(ckpt=tmp_path / "actor.pt", library=tmp_path / "motions.pt",
                    clips=["clip"], num_envs=0)


def test_native_renderer_stops_before_environment_is_freed(tmp_path, monkeypatch):
    """The passive viewer returns from close before its daemon frees GL resources."""
    module = _play_module(monkeypatch)
    stop = threading.Event()
    worker = None
    cfg = {"robot": "khan_mini", "world_model": {}, "env": {"future_frames": 5}}
    policy = SimpleNamespace(cfg=SimpleNamespace(history=5, observation_layout="isaac_khan"))
    monkeypatch.setattr(module, "load_policy_from_checkpoint", lambda *a: (policy, cfg))

    class Env:
        def close(self):
            assert worker is not None and not worker.is_alive()

    class Viewer:
        def __init__(self, *args):
            pass

        def run(self):
            nonlocal worker

            def render():
                stop.wait(2)
                time.sleep(.02)

            worker = threading.Thread(target=render, name="test (_launch_internal)", daemon=True)
            worker.start()

        def close(self):
            stop.set()

    monkeypatch.setattr(module, "make_env", lambda *a, **k: Env())
    monkeypatch.setattr(module, "RslRlVecEnvWrapper", lambda env: env)
    monkeypatch.setattr(module, "NativeMujocoViewer", Viewer)
    module.main(tmp_path / "actor.pt", tmp_path / "motions.pt", ["clip"], viewer="native")

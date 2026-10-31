"""A bounded GPU replay check that forbids every dynamics-step entry point."""

import hashlib

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from supermimic.data.khan_motion_convert import convert_clip
from supermimic.eval.khan_replay import KinematicReplayEnv
from supermimic.robots.khan_mini import DEFAULT_ASSET_DIR, load_profile

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Kinematic mjlab replay needs CUDA"),
    pytest.mark.skipif(not (DEFAULT_ASSET_DIR / "robot.xml").is_file(), reason="Khan assets missing"),
]


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def test_replay_writes_source_frames_without_dynamics_or_source_mutation(tmp_path, monkeypatch):
    profile = load_profile()
    times = np.arange(5, dtype=np.float64) / 50.0
    home = np.array([profile["default_joint_pos"][name] for name in profile["joint_names"]])
    xyzw = Rotation.from_euler("xyz", np.column_stack((
        0.01 + 0.05 * times, -0.02 + 0.04 * times, 0.3 + 0.2 * times,
    ))).as_quat()
    clip = {
        "fps": 50.0,
        "root_pos": np.column_stack((0.1 + 0.2 * times, -0.2 + 0.1 * times,
                                     0.65 + 0.05 * times)),
        "root_rot": xyzw[:, [3, 0, 1, 2]],
        "dof_pos": home + times[:, None] * np.linspace(0.01, 0.05, 27),
    }
    path = tmp_path / "synthetic_replay.npz"
    convert_clip(clip, path, source="synthetic", clip_name="synthetic_replay")
    before = file_hash(path)
    with np.load(path, allow_pickle=False) as archive:
        source = {name: archive[name] for name in archive.files}
    root_index = source["body_names"].tolist().index("base_link")
    replay = None
    try:
        replay = KinematicReplayEnv(path, start_frame=1, end_frame=4, loop=False)

        def forbidden_step(*_args, **_kwargs):
            raise AssertionError("Kinematic replay called a dynamics step")

        monkeypatch.setattr(replay.unwrapped, "step", forbidden_step)
        monkeypatch.setattr(replay.unwrapped.sim, "step", forbidden_step)
        original_time = replay.unwrapped.sim.data.time[:].detach().cpu().clone()

        def assert_frame(frame):
            assert replay.frame == frame
            position = source["body_pos_w"][frame, root_index].copy()
            position += replay.unwrapped.scene.env_origins[0].cpu().numpy()
            expected = np.concatenate((position, source["body_quat_w"][frame, root_index],
                                       source["joint_pos"][frame]))
            actual = replay.unwrapped.sim.data.qpos[0].cpu().numpy()
            np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=0)
            np.testing.assert_allclose(replay.robot.data.root_link_pos_w[0].cpu().numpy(),
                                       position, atol=2e-7, rtol=0)
            torch.testing.assert_close(
                replay.unwrapped.sim.data.time[:].cpu(), original_time, rtol=0, atol=0
            )

        assert_frame(1)
        replay.seek(3)
        assert_frame(3)
        assert replay.finished
        replay.seek(2)
        assert_frame(2)
        assert not replay.finished
        replay.seek(-999)
        assert_frame(1)
        replay.seek(999)
        assert_frame(3)

        _, _, terminated, _, extras = replay.step(None)
        assert_frame(3)
        assert bool(terminated[0]) and extras["frame"] == 3

        replay.loop = True
        replay.seek(3)
        assert not replay.finished
        _, _, terminated, _, extras = replay.step(None)
        assert_frame(1)
        assert not bool(terminated[0]) and extras["frame"] == 1
        replay.step(None)
        assert_frame(2)

        observations, _ = replay.reset()
        assert_frame(1)
        assert observations["policy"].shape == (1, 27)
        assert replay.forward_count >= 9
    finally:
        if replay is not None:
            replay.close()
    assert file_hash(path) == before

"""Bin merging changes reset groups, never reference frames or clip boundaries."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.robots.registry import get_robot_spec


def _inputs(tmp_path, lengths):
    spec = get_robot_spec("g1")
    files, references, starts = [], [], []
    offset = 0
    for i, length in enumerate(lengths):
        root = np.zeros((length, 3), dtype=np.float32)
        root[:, 0] = np.arange(length, dtype=np.float32) / 100
        root[:, 1], root[:, 2] = i, 1
        quat = np.zeros((length, 4), dtype=np.float32)
        quat[:, 0] = 1
        joints = np.zeros((length, spec.num_joints), dtype=np.float32)
        joints[:, 0] = np.sin(np.arange(length) / 20).astype(np.float32) * .1
        linear = np.zeros((length, 3), dtype=np.float32)
        linear[:, 0] = .5
        angular = np.zeros_like(linear)
        joint_vel = np.zeros_like(joints)
        ref = np.concatenate((root, quat, joints, linear, angular, joint_vel), axis=-1)
        path = tmp_path / f"clip_{i}.npz"
        np.savez(path, fps=np.asarray(50), robot=np.asarray("g1"),
                 joint_names=np.asarray(spec.joint_names), body_names=np.asarray(["pelvis"]),
                 joint_pos=joints, joint_vel=joint_vel, body_pos_w=root[:, None],
                 body_quat_w=quat[:, None], body_lin_vel_w=linear[:, None],
                 body_ang_vel_w=angular[:, None])
        files.append(path)
        references.append(torch.from_numpy(ref))
        starts.append(offset)
        offset += length
    packed = tmp_path / "library.pt"
    torch.save({"names": [path.stem for path in files], "starts": starts,
                "lengths": lengths, "ref": torch.cat(references),
                "robot": "g1", "fps": 50, "root_body_name": "pelvis",
                "joint_names": spec.joint_names}, packed)
    return files, packed


def _load(inputs, storage, *, merge, bin_frames=50, fraction=0.5):
    files, packed = inputs
    return MotionLibrary([packed] if storage == "compact_pt" else files,
                         ("pelvis",), torch.zeros(29), device="cpu",
                         compact=storage != "full_npz", store_dtype=torch.float32,
                         merge_short_tail_bins=merge, bin_frames=bin_frames,
                         tail_bin_merge_fraction=fraction)


@pytest.mark.parametrize("storage", ["full_npz", "compact_npz", "compact_pt"])
def test_half_tail_layout_and_all_frame_mappings(tmp_path, storage):
    lengths = [75, 76, 101, 124, 125, 24, 1]
    inputs = _inputs(tmp_path, lengths)
    library = _load(inputs, storage, merge=True)
    expected_lengths = [[50, 25], [50, 26], [50, 51], [50, 74], [50, 50, 25], [24], [1]]
    assert library.bins_per_clip.tolist() == [len(values) for values in expected_lengths]
    assert library.num_bins == 13
    assert library.num_merged_tail_bins == 2
    for clip, expected in enumerate(expected_lengths):
        bins = library.bin_clip == clip
        assert (library.bin_end-library.bin_start)[bins].tolist() == expected
        start, end = int(library.clip_start[clip]), int(library.clip_last[clip])+1
        frame = torch.arange(start, end)
        ids = library.bin_of_frame(frame)
        assert (library.bin_clip[ids] == clip).all()
        assert (library.bin_start[ids] <= frame).all()
        assert (frame < library.bin_end[ids]).all()
        # Every bin's first and final reference state map back to that bin,
        # including merged-tail states beyond the old final 50-frame boundary.
        wanted = torch.arange(library.num_bins)[bins]
        torch.testing.assert_close(library.bin_of_frame(library.bin_start[bins]), wanted)
        torch.testing.assert_close(library.bin_of_frame(library.bin_end[bins]-1), wanted)
    sample_end = torch.minimum(library.bin_end, library.clip_last[library.bin_clip])
    sampleable = library.bin_start < sample_end
    assert (~sampleable).sum() == 1  # The unavoidable one-state clip has no transition.
    assert "2 short tail bins merged" in library.summary()


@pytest.mark.parametrize("storage", ["full_npz", "compact_npz", "compact_pt"])
@pytest.mark.parametrize("fraction", [0.5, 1.0])
def test_merging_preserves_reference_goals_and_body_queries(tmp_path, storage, fraction):
    inputs = _inputs(tmp_path, [101, 125, 51])
    legacy = _load(inputs, storage, merge=False)
    merged = _load(inputs, storage, merge=True, fraction=fraction)
    frames = torch.tensor([0, 49, 50, 99, 100, 101, 225, 226, 276, 100])
    for query in ("ref71", "ref_qpos_qvel"):
        expected, actual = getattr(legacy, query)(frames), getattr(merged, query)(frames)
        if isinstance(actual, tuple):
            for a, b in zip(actual, expected, strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        else:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(legacy.goal_window(frames, 5), merged.goal_window(frames, 5), rtol=0, atol=0)
    expected, actual = legacy.bodies(frames), merged.bodies(frames)
    for name in ("pos_w", "quat_w", "lin_vel_w", "ang_vel_w"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)


def test_legacy_default_keeps_original_layout(tmp_path):
    inputs = _inputs(tmp_path, [101, 51, 125])
    library = MotionLibrary(inputs[0], ("pelvis",), torch.zeros(29), device="cpu")
    assert not library.merge_short_tail_bins
    assert library.bins_per_clip.tolist() == [3, 2, 3]
    assert library.num_merged_tail_bins == 0
    assert (library.bin_end-library.bin_start).tolist() == [50, 50, 1, 50, 1, 50, 50, 25]
    for frame in range(library.num_frames):
        clip = int(library.frame_clip[frame])
        expected = int(library.clip_bin_offset[clip])+(frame-int(library.clip_start[clip]))//50
        assert library.bin_of_frame(torch.tensor(frame)) == expected


@pytest.mark.parametrize("bin_frames,lengths,expected_lengths", [
    (1, [1, 2, 3], [[1], [1, 1], [1, 1, 1]]),
    (2, [3, 5], [[2, 1], [2, 2, 1]]),  # Exactly half is not merged.
    (3, [4, 5, 7], [[4], [3, 2], [3, 4]]),
    (50, [50, 51, 74, 75, 100, 101], [[50], [51], [74], [50, 25], [50, 50], [50, 51]]),
])
def test_merge_threshold_and_small_bin_widths(tmp_path, bin_frames, lengths, expected_lengths):
    library = _load(_inputs(tmp_path, lengths), "full_npz", merge=True, bin_frames=bin_frames)
    for clip, expected in enumerate(expected_lengths):
        bins = library.bin_clip == clip
        assert (library.bin_end-library.bin_start)[bins].tolist() == expected
    frames = torch.arange(library.num_frames).reshape(1, -1)
    ids = library.bin_of_frame(frames)
    assert ids.shape == frames.shape
    assert (library.bin_start[ids] <= frames).all()
    assert (frames < library.bin_end[ids]).all()


@pytest.mark.parametrize("storage", ["full_npz", "compact_npz", "compact_pt"])
def test_full_tail_merges_all_nonempty_remainders_within_each_clip(tmp_path, storage):
    lengths = [1, 24, 49, 50, 51, 74, 75, 99, 100, 101, 125, 149, 150]
    library = _load(_inputs(tmp_path, lengths), storage, merge=True, fraction=1.0)
    expected = [[1], [24], [49], [50], [51], [74], [75], [99], [50, 50],
                [50, 51], [50, 75], [50, 99], [50, 50, 50]]
    assert library.num_merged_tail_bins == 7
    assert library.bins_per_clip.tolist() == [len(bins) for bins in expected]
    for clip, bins in enumerate(expected):
        mask = library.bin_clip == clip
        assert (library.bin_end - library.bin_start)[mask].tolist() == bins
    frames = torch.arange(library.num_frames)
    ids = library.bin_of_frame(frames)
    torch.testing.assert_close(library.bin_clip[ids], library.frame_clip)
    assert (library.bin_start[ids] <= frames).all()
    assert (frames < library.bin_end[ids]).all()


@pytest.mark.parametrize("fraction", [0, -0.5, 1.1, float("nan"), float("inf")])
def test_invalid_merge_fraction_rejected(tmp_path, fraction):
    with pytest.raises(ValueError, match="tail_bin_merge_fraction"):
        _load(_inputs(tmp_path, [51]), "full_npz", merge=True, fraction=fraction)

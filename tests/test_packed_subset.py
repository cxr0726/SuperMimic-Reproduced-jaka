"""Subset extraction preserves stored states and rejects unsafe selection/reuse."""

from __future__ import annotations

import hashlib

import pytest
import torch

from supermimic.data.packed_subset import mixed480_names, subset_library
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.robots.registry import get_robot_spec


def _pack(path, *, names=None):
    spec = get_robot_spec("khan_mini")
    names = names or ["unselected", "mixed__pose", "lafan1__walk"]
    lengths = [3, 7, 5][:len(names)]
    starts = [sum(lengths[:i]) for i in range(len(lengths))]
    generator = torch.Generator().manual_seed(672)
    ref = torch.randn(sum(lengths), spec.state_dim, generator=generator).half()
    ref[:, 3:7] = torch.tensor([1, 0, 0, 0])
    pack = {"ref": ref, "names": names, "starts": starts, "lengths": lengths,
            "robot": spec.name, "joint_names": spec.joint_names, "fps": 50.,
            "root_body_name": spec.root_body, "source_inventory_sha256": "original-inventory"}
    torch.save(pack, path)
    return pack


def test_exact_subset_state_roundtrip_and_source_unchanged(tmp_path):
    source = tmp_path/"all.pt"
    original = _pack(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    target = tmp_path/"selected.pt"
    report = subset_library(source, target, names=["lafan1__walk", "mixed__pose"], chunk_size=2)
    result = torch.load(target, weights_only=False)
    assert report["status"] == "created"
    assert report["selected_clips"] == 2 and report["selected_frames"] == 12
    assert result["names"] == ["mixed__pose", "lafan1__walk"]  # source order
    assert result["starts"] == [0, 7] and result["lengths"] == [7, 5]
    assert result["ref"].dtype == original["ref"].dtype == torch.float16
    assert torch.equal(result["ref"], original["ref"][3:])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    assert result["subset_provenance"]["parent_sha256"] == digest
    assert result["subset_provenance"]["parent_source_inventory_sha256"] == "original-inventory"
    assert "source_inventory_sha256" not in result  # no claim that the old inventory is the subset
    spec = get_robot_spec("khan_mini")
    lib = MotionLibrary([target], spec.tracked_bodies, torch.zeros(spec.num_joints),
                        device="cpu", compact=True, robot_name=spec.name,
                        root_body_name=spec.root_body)
    assert lib.num_clips == 2
    torch.testing.assert_close(lib.ref71(torch.arange(12)), original["ref"][3:].float(), rtol=0, atol=0)


def test_motion_list_names_map_prefixes_and_do_not_load_npz(tmp_path):
    source = tmp_path/"all.pt"
    _pack(source)
    directory = tmp_path/"filenames"
    directory.mkdir()
    for name in ("pose", "walk"):
        (directory/f"{name}.npz").write_bytes(b"contents deliberately not NPZ")
    target = tmp_path/"subset.pt"
    report = subset_library(source, target, motion_list_dir=directory)
    assert report["names"] == ["mixed__pose", "lafan1__walk"]


def test_missing_ambiguous_and_empty_selections_fail_without_output(tmp_path):
    source = tmp_path/"all.pt"
    _pack(source, names=["pose", "mixed__pose", "other"])
    directory = tmp_path/"filenames"
    directory.mkdir()
    (directory/"pose.npz").touch()
    target = tmp_path/"subset.pt"
    with pytest.raises(ValueError, match="ambiguous"):
        subset_library(source, target, motion_list_dir=directory)
    with pytest.raises(ValueError, match="missing"):
        subset_library(source, target, names=["missing"])
    with pytest.raises(ValueError, match="nonempty and unique"):
        subset_library(source, target, names=[])
    with pytest.raises(ValueError, match="nonempty and unique"):
        subset_library(source, target, names=["pose", "pose"])
    with pytest.raises(ValueError, match="exactly one"):
        subset_library(source, target, names=["pose"], subset="mixed480")
    assert not target.exists()


def test_output_protection_and_verified_idempotent_reuse(tmp_path):
    source, target = tmp_path/"all.pt", tmp_path/"subset.pt"
    _pack(source)
    subset_library(source, target, names=["mixed__pose"])
    original_bytes = target.read_bytes()
    with pytest.raises(FileExistsError):
        subset_library(source, target, names=["mixed__pose"])
    report = subset_library(source, target, names=["mixed__pose"], reuse_existing=True)
    assert report["status"] == "reused_verified"
    assert target.read_bytes() == original_bytes
    with pytest.raises(ValueError, match="layout"):
        subset_library(source, target, names=["lafan1__walk"], reuse_existing=True)
    damaged = torch.load(target, weights_only=False)
    damaged["ref"][0, 0] += 1
    torch.save(damaged, target)
    with pytest.raises(ValueError, match="frames differ"):
        subset_library(source, target, names=["mixed__pose"], reuse_existing=True)
    with pytest.raises(ValueError, match="differ from the source"):
        subset_library(source, source, names=["mixed__pose"], reuse_existing=True)


@pytest.mark.parametrize("damage,match", [
    ("gap", "contiguous"), ("width", "67 columns"), ("joint_names", "joint_names"),
    ("nonfinite", "NaN/Inf"), ("quaternion", "quaternions"), ("fps", "50 Hz"),
])
def test_bad_source_rejected_before_publication(tmp_path, damage, match):
    source, target = tmp_path/"all.pt", tmp_path/"subset.pt"
    pack = _pack(source)
    if damage == "gap":
        pack["starts"][1] += 1
    elif damage == "width":
        pack["ref"] = pack["ref"][:, :-1]
    elif damage == "joint_names":
        pack["joint_names"] = tuple(reversed(pack["joint_names"]))
    elif damage == "nonfinite":
        pack["ref"][3, 0] = float("nan")
    elif damage == "quaternion":
        pack["ref"][3, 3:7] = 0
    else:
        pack["fps"] = 30.
    torch.save(pack, source)
    with pytest.raises(ValueError, match=match):
        subset_library(source, target, names=["mixed__pose"])
    assert not target.exists()


def test_builtin_manifest_is_480_not_484_or_first480():
    names = mixed480_names()
    assert len(names) == len(set(names)) == 480
    assert sum(n.startswith("mixed__") for n in names) == 444
    assert sum(n.startswith("lafan1__") for n in names) == 36
    assert "lafan1__fightAndSports1_subject4" in names
    assert "mixed__newkneel" in names
    assert not any(n.startswith("kneeling_start_") for n in names)

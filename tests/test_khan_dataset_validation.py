"""Independent validation accepts legacy and recorded Khan time/joint layouts."""

import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from supermimic.data.khan_motion_convert import convert_khan_pkl
from supermimic.robots.khan_mini import DEFAULT_ASSET_DIR, load_profile


@pytest.fixture(scope="module")
def validator():
    path = Path(__file__).resolve().parents[1] / "scripts/data/validate_khan_dataset.py"
    spec = importlib.util.spec_from_file_location("validate_khan_dataset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_clip(dofs, fps):
    profile = load_profile()
    names = [n for n in profile["joint_names"] if dofs == 27 or "wrist_yaw" not in n]
    t = np.arange(300, dtype=np.float64)/fps
    quaternion = Rotation.from_euler("xyz", np.column_stack([
        .04*np.sin(t), .03*np.cos(t), .15*t,
    ])).as_quat()[:, [3, 0, 1, 2]]
    quaternion[1::2] *= -1
    return {"fps": fps,
            "root_pos": np.column_stack([.1*t, -.05*t, .65+.01*np.sin(t)]),
            "root_rot": quaternion,
            "dof_pos": np.array([profile["default_joint_pos"][n] for n in names])
                       + .02*np.sin(t[:, None]+np.arange(dofs)[None, :])}


@pytest.mark.parametrize("dofs,fps", [(25, 50.), (27, 49.8), (27, 50.)])
def test_validator_handles_source_width_and_resampling(validator, tmp_path, dofs, fps):
    source = tmp_path / "pack.pkl"
    clip = source_clip(dofs, fps)
    original_quat = clip["root_rot"].copy()
    expected = validator.independent_reference(
        clip, load_profile()["joint_names"],
        np.array([load_profile()["default_joint_pos"][n]
                  for n in load_profile()["joint_names"]]),
    )
    np.testing.assert_array_equal(clip["root_rot"], original_quat)
    joblib.dump({"synthetic": clip}, source)
    output = tmp_path / "npz"
    converted = convert_khan_pkl(source, output)
    report = validator.validate_dataset(source, output, DEFAULT_ASSET_DIR, samples=3)
    assert report["status"] == "passed"
    assert report["frame_count"] == converted["total_frames"] == len(expected["joint_pos"])
    assert report["clips"][0]["source_dofs"] == dofs
    assert len(report["clips"][0]["filled_joints"]) == (2 if dofs == 25 else 0)
    assert report["max_errors"]["fk_body_pos_m"] < 2e-6
    if fps != 50:
        assert report["frame_count"] > len(clip["dof_pos"])


def test_validator_catches_a_changed_recorded_wrist(validator, tmp_path):
    source = tmp_path / "pack.pkl"
    joblib.dump({"synthetic": source_clip(27, 49.8)}, source)
    output = tmp_path / "npz"
    convert_khan_pkl(source, output)
    path = output / "synthetic.npz"
    with np.load(path, allow_pickle=False) as saved:
        arrays = {k: saved[k] for k in saved.files}
    index = arrays["joint_names"].tolist().index("Left_wrist_yaw_joint")
    arrays["joint_pos"][10, index] += .1
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="joints"):
        validator.validate_dataset(source, output, DEFAULT_ASSET_DIR)

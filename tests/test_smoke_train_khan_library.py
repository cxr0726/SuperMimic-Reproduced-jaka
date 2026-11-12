"""CPU checks for the real training smoke's data-source integration."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def smoke():
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        spec = importlib.util.spec_from_file_location("test_khan_library_smoke", ROOT / "scripts/smoke_train_khan.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(ROOT / "scripts"))


def library():
    names = json.loads((ROOT / "assets/robots/khan_mini/profile.json").read_text())["joint_names"]
    ref = torch.zeros(8, 67, dtype=torch.float16)
    ref[:, 3] = 1
    for name in ("Left_wrist_yaw_joint", "Right_wrist_yaw_joint"):
        ref[:, 7 + names.index(name)] = .3
    return SimpleNamespace(
        clips=[SimpleNamespace(name="first", start=0, length=3), SimpleNamespace(name="second", start=3, length=5)],
        num_clips=2, num_frames=8, compact=True, fps=50., _ref=ref,
        joint_names=tuple(names), num_joints=27,
    )


def test_default_npz_behavior_preserved(smoke):
    args = smoke.parse_args([])
    cfg = smoke.TrainConfig(robot="khan_mini")
    smoke.configure_data_source(cfg, args)
    assert cfg.data.npz_dir == (ROOT / "data/npz/khan_mini/lafan1").resolve()
    assert cfg.data.library is None and not cfg.env.compact_library
    assert args.num_envs == 32 and args.steps == 50 and not args.activation_checkpoint


@pytest.mark.parametrize("context_layout", [None, "khan_dr_v1", "khan_dr_v2", "khan_dr_v3"])
def test_library_config_preserves_production_optimization(smoke, tmp_path, context_layout):
    packed = tmp_path / "motions.pt"
    packed.write_bytes(b"CLI placeholder")
    args = smoke.parse_args(["--library", str(packed), "--num-envs", "8", "--steps", "50", "--activation-checkpoint"])
    cfg = smoke.TrainConfig(robot="khan_mini")
    cfg.env.context_layout = context_layout
    smoke.configure_data_source(cfg, args)
    cfg.env.num_envs = args.num_envs
    cfg.env.observation_layout = args.observation_layout
    cfg.schedule.stop_after_iterations = args.steps
    cfg.policy_trainer.activation_checkpoint = args.activation_checkpoint
    result = smoke.verify_production_config(cfg)
    assert cfg.data.library == packed.resolve() and cfg.data.npz_dir is None
    assert cfg.env.compact_library
    assert result["matches_train_config_optimization_defaults"]
    assert result["context_layout"] == cfg.env.context_layout == (context_layout or "khan_dr_v3")
    assert result["warmup_iterations"] == 40 and result["lr_schedule_total_iterations"] == 320000


def test_sources_mutually_exclusive(smoke, tmp_path):
    with pytest.raises(SystemExit) as error:
        smoke.parse_args(["--data-dir", str(tmp_path), "--library", str(tmp_path / "x.pt")])
    assert error.value.code == 2


@pytest.mark.parametrize("name", ["missing.pt", "wrong.npz"])
def test_invalid_library_rejected(smoke, tmp_path, name):
    if name.endswith(".npz"):
        (tmp_path / name).write_bytes(b"wrong extension")
    with pytest.raises(SystemExit):
        smoke.parse_args(["--library", str(tmp_path / name)])


def test_packed_summary_reports_actual_loaded_clips(smoke, tmp_path):
    packed = tmp_path / "motions.pt"
    packed.write_bytes(b"exact selected pack")
    cfg = smoke.TrainConfig(robot="khan_mini")
    cfg.data.library = packed
    cfg.data.npz_dir = tmp_path / "does-not-exist"
    result = smoke.dataset_summary(SimpleNamespace(lib=library()), cfg)
    assert result["source_type"] == "packed_library" and result["paths"] == [str(packed)]
    assert result["library_sha256"] == hashlib.sha256(packed.read_bytes()).hexdigest()
    assert result["clips"] == 2 and result["frames"] == 8
    assert result["clip_names"] == ["first", "second"] and result["clip_lengths"] == [3, 5]
    assert result["reference_shape"] == [8, 67] and result["storage_dtype"] == "torch.float16"
    wrist = result["fixed_joint_reference_ranges"]["Left_wrist_yaw_joint"]
    assert wrist["minimum_rad"] == wrist["maximum_rad"] == wrist["quantized_target_rad"]


def test_npz_summary_reports_loaded_subset(smoke, tmp_path):
    (tmp_path / "unloaded.npz").touch()
    cfg = smoke.TrainConfig(robot="khan_mini")
    cfg.data.npz_dir = tmp_path
    lib = library()
    lib.compact = False
    result = smoke.dataset_summary(SimpleNamespace(lib=lib), cfg)
    assert result["paths"] == [str(tmp_path / "first.npz"), str(tmp_path / "second.npz")]
    assert not result["compact"] and result["library_path"] is None


def test_storage_cast_nonfinite_rejected(smoke, tmp_path):
    cfg = smoke.TrainConfig(robot="khan_mini")
    cfg.data.npz_dir = tmp_path
    lib = library()
    lib._ref[0, 0] = torch.tensor(65520., dtype=torch.float32).half()
    with pytest.raises(ValueError, match="runtime storage cast"):
        smoke.dataset_summary(SimpleNamespace(lib=lib), cfg)


def test_duplicate_clip_identity_rejected(smoke, tmp_path):
    cfg = smoke.TrainConfig(robot="khan_mini")
    cfg.data.npz_dir = tmp_path
    lib = library()
    lib.clips[1].name = "first"
    with pytest.raises(ValueError, match="clip names/frame metadata"):
        smoke.dataset_summary(SimpleNamespace(lib=lib), cfg)


def test_cuda_guard_reports_clean_error(smoke, monkeypatch):
    args = smoke.parse_args([])
    monkeypatch.setattr(smoke, "parse_args", lambda: args)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="CUDA is required"):
        smoke.main()


def test_existing_run_guard_precedes_gpu_initialization(smoke, tmp_path, monkeypatch):
    args = smoke.parse_args(["--log-root", str(tmp_path), "--run-name", "already"])
    run = tmp_path / "already"
    run.mkdir()
    (run / "summary.json").write_text("{}")
    monkeypatch.setattr(smoke, "parse_args", lambda: args)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(SystemExit, match="existing run"):
        smoke.main()

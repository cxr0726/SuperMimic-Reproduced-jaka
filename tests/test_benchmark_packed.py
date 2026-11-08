"""A packed motion file must schedule all its clips, not one file-sized episode."""

import importlib.util
from pathlib import Path

import pytest
import torch

from supermimic.eval import benchmark
from supermimic.policy.nets import Policy, PolicyCfg


def packed(tmp_path, names):
    path = tmp_path / "motions.pt"
    torch.save({"names": names, "ref": torch.zeros(2, 67, dtype=torch.float16)}, path)
    return str(path)


@pytest.mark.parametrize("requested,expected", [(None, 1024), (2048, 2048), (9000, 8452)])
def test_packed_parallelism_uses_clip_count_not_file_count(tmp_path, requested, expected):
    path = packed(tmp_path, [f"clip_{i}" for i in range(8452)])
    assert benchmark.benchmark_source([path], requested) == (expected, True)


def test_full_npz_parallelism_and_invalid_sources(tmp_path):
    assert benchmark.benchmark_source(["a.npz", "b.npz"], 4096) == (2, False)
    with pytest.raises(ValueError, match="positive"):
        benchmark.benchmark_source(["a.npz"], 0)
    with pytest.raises(ValueError, match="at least one"):
        benchmark.benchmark_source([], None)
    with pytest.raises(ValueError, match="one packed"):
        benchmark.benchmark_source(["a.pt", "b.npz"], 2)
    with pytest.raises(ValueError, match="nonempty, unique"):
        benchmark.benchmark_source([packed(tmp_path, ["a", "a"])], 2)


def test_packed_scene_enables_compact_without_changing_eval_protocol(tmp_path, monkeypatch):
    policy = Policy(PolicyCfg(
        encoder_hidden=(8,), decoder_hidden=(8,), latent_dim=4,
        num_joints=27, num_tracked_bodies=15, goal_dim=155,
        history=5, future_frames=5, observation_layout="isaac_khan",
    ))
    path = packed(tmp_path, ["a", "b", "c"])
    captured = {}

    def inspect(args, device):
        captured.update(args=args, device=device)
        raise RuntimeError("inspected scene")

    monkeypatch.setattr(benchmark, "make_env", inspect)
    with pytest.raises(RuntimeError, match="inspected scene"):
        benchmark.run_benchmark(
            policy, [path], device="cpu", robot="khan_mini", num_envs=2048,
            reference_fk_compile=True, reset_root_z_offset=.05,
        )
    args = captured["args"]
    assert args.num_envs == 3 and args.compact_library and args.reference_fk_compile
    assert args.npz_files == [path] and args.sampling_mode == "start"
    assert not args.record_transitions and not args.domain_randomization
    assert not args.obs_noise and not args.reset_noise and args.action_noise_std == 0
    assert args.actuator_delay is False and args.reset_root_z_offset == .05


def test_packed_cli_rejects_ambiguous_npz_filters_before_loading_checkpoint(tmp_path):
    source = Path(__file__).resolve().parents[1] / "scripts/eval_benchmark.py"
    spec = importlib.util.spec_from_file_location("eval_packed_cli", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="do not combine"):
        module.main(ckpt=tmp_path / "not_loaded.pt", library=tmp_path / "motions.pt",
                    npz_dir=tmp_path / "npz")

"""Historical and residual checkpoints keep their semantics in FK benchmarks."""

import importlib.util
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from supermimic.config import TrainConfig, resolve_training_observations
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.robots.contracts import robot_contract
from supermimic.robots.khan_mini import load_profile


@pytest.fixture
def benchmark_module():
    path = Path(__file__).resolve().parents[1] / "scripts/benchmarks/benchmark_reference_fk_imagination.py"
    spec = importlib.util.spec_from_file_location("reference_fk_restore_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_restore_policy_missing_fields_uses_legacy_not_fresh_training_defaults(benchmark_module):
    cfg = TrainConfig(robot="khan_mini")
    assert cfg.policy.action_mode == "auto"
    assert cfg.policy.goal_noise_z is None
    saved = {
        "robot": "khan_mini",
        "env": {"observation_layout": "isaac_khan", "future_frames": 5},
        "policy": {
            "history": 5, "future_frames": 5, "observation_layout": "isaac_khan",
            "num_joints": 27, "goal_dim": 155,
            "encoder_hidden": [16, 8], "decoder_hidden": [16, 8],
        },
    }
    benchmark_module.restore_config(cfg, saved)
    resolve_training_observations(cfg)
    assert cfg.policy.action_mode == "default"
    assert cfg.policy.goal_noise_z == cfg.policy.goal_noise_joint_pos == 0.0
    assert cfg.policy.encoder_hidden == (16, 8)
    assert cfg.policy.decoder_hidden == (16, 8)
    assert cfg.policy.history == cfg.policy.future_frames == 5


@pytest.mark.parametrize("mode", ["legacy", "reference_residual"])
def test_benchmark_checkpoint_loader_roundtrips_policy_on_cpu(benchmark_module, tmp_path, mode):
    torch.manual_seed(17)
    cfg = TrainConfig(robot="khan_mini", device="cpu")
    cfg.policy = PolicyCfg(
        encoder_hidden=(8,), decoder_hidden=(8,), latent_dim=4, history=5,
        action_mode="default" if mode == "legacy" else mode,
        goal_noise_z=0.0 if mode == "legacy" else .01,
        goal_noise_joint_pos=0.0 if mode == "legacy" else .05,
    )
    resolve_training_observations(cfg)
    kwargs = {}
    if mode == "reference_residual":
        profile = load_profile()
        kwargs = {
            "default_joint_pos": torch.tensor([profile["default_joint_pos"][j] for j in profile["joint_names"]]),
            "action_scale": torch.tensor([profile["action_scale"][j] for j in profile["joint_names"]]),
        }
    original = Policy(cfg.policy, **kwargs)
    config = asdict(cfg)
    if mode == "legacy":
        for name in ("action_mode", "reference_residual_beta", "reference_frame_offset",
                     "goal_noise_z", "goal_noise_joint_pos"):
            config["policy"].pop(name)
    saved = {
        "config": config,
        "policy": original.state_dict(),
        "wm": {"dummy": torch.tensor([3.])},
        "critic": {"dummy": torch.tensor([5.])},
        "robot_contract": robot_contract("khan_mini", "isaac_khan", env_config=config["env"]),
    }
    path = tmp_path / "benchmark.pt"
    torch.save(saved, path)
    restored_cfg, restored, checkpoint_view = benchmark_module.load_benchmark_checkpoint(path, "cpu")
    assert restored_cfg.policy is restored.cfg
    assert restored.cfg.action_mode == ("default" if mode == "legacy" else mode)
    assert restored.cfg.goal_noise_z == (0.0 if mode == "legacy" else .01)
    assert restored.cfg.goal_noise_joint_pos == (0.0 if mode == "legacy" else .05)
    assert restored_cfg.env.observation_layout == "isaac_khan"
    assert restored_cfg.policy.history == restored_cfg.env.future_frames == 5
    obs, goal = torch.randn(3, 465), torch.randn(3, 155)
    torch.testing.assert_close(restored(obs, goal), original(obs, goal), rtol=0, atol=0)
    assert checkpoint_view["wm"]["dummy"].item() == 3.
    assert checkpoint_view["critic"]["dummy"].item() == 5.
    assert restored(obs, goal).requires_grad, "benchmark policy must retain actor BPTT gradients"

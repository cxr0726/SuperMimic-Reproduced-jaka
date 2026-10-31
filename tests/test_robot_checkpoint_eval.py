"""CPU checkpoint loading and display-scene compatibility for G1 and Khan."""

import importlib.util
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from supermimic.eval import benchmark
from supermimic.policy.nets import Critic, CriticCfg, Policy, PolicyCfg
from supermimic.robots.registry import get_robot_spec


def _checkpoint(tmp_path, robot, anchor=False, legacy=False):
    spec = get_robot_spec(robot)
    cfg = PolicyCfg(encoder_hidden=(8,), decoder_hidden=(8,), latent_dim=4,
                    num_joints=spec.num_joints, history=3,
                    goal_dim=2 * spec.goal_feature_dim + (9 if anchor else 0))
    policy = Policy(cfg)
    config = {"policy": asdict(cfg), "env": {"future_frames": 2, "anchor_obs": anchor}}
    if legacy:
        config["policy"].pop("num_joints")
    else:
        config["robot"] = robot
    checkpoint = {"policy": policy.state_dict(), "config": config}
    path = tmp_path / "policy.pt"
    torch.save(checkpoint, path)
    return path, policy, checkpoint


@pytest.mark.parametrize("robot,legacy", [("g1", True), ("g1", False), ("khan_mini", False)])
@pytest.mark.parametrize("anchor", [False, True])
def test_checkpoint_policy_roundtrip_uses_robot_layout(tmp_path, robot, legacy, anchor):
    path, original, _ = _checkpoint(tmp_path, robot, anchor, legacy)
    loaded, config = benchmark.load_policy_from_checkpoint(path, "cpu")
    assert loaded.cfg.num_joints == get_robot_spec(robot).num_joints
    assert not loaded.training
    obs = torch.randn(2, loaded.cfg.history * (3 * loaded.cfg.num_joints + 6))
    goal = torch.randn(2, loaded.cfg.goal_dim)
    torch.testing.assert_close(loaded(obs, goal), original(obs, goal), rtol=0, atol=0)
    assert config.get("robot", "g1") == robot


@pytest.mark.parametrize("bad_field,bad_value", [("num_joints", 29), ("goal_dim", 680)])
def test_checkpoint_inconsistent_robot_layout_is_rejected(tmp_path, bad_field, bad_value):
    path, _, checkpoint = _checkpoint(tmp_path, "khan_mini")
    checkpoint["config"]["policy"][bad_field] = bad_value
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="Checkpoint"):
        benchmark.load_policy_from_checkpoint(path, "cpu")


def test_benchmark_forwards_robot_and_policy_history_before_creating_simulation(tmp_path, monkeypatch):
    _, policy, _ = _checkpoint(tmp_path, "khan_mini")
    captured = {}

    def inspect_args(args, device):
        captured["args"], captured["device"] = args, device
        raise RuntimeError("inspected scene configuration")

    monkeypatch.setattr(benchmark, "make_env", inspect_args)
    with pytest.raises(RuntimeError, match="inspected scene"):
        benchmark.run_benchmark(policy, ["motion.npz"], device="cpu", robot="khan_mini", future_frames=2)
    assert captured["args"].robot == "khan_mini"
    assert captured["args"].history_length == 3
    assert captured["args"].future_frames == 2
    assert not captured["args"].domain_randomization
    with pytest.raises(ValueError, match="joint count"):
        benchmark.run_benchmark(policy, ["motion.npz"], device="cpu", robot="g1")


@pytest.mark.parametrize("robot", ["g1", "khan_mini"])
def test_rollout_display_model_compiles_with_matching_visual_only_ghost(robot):
    source = Path(__file__).resolve().parents[1] / "scripts/media/render_rollout.py"
    module_spec = importlib.util.spec_from_file_location("render_robot_rollout", source)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    original = get_robot_spec(robot).load_model()
    model = module.load_rollout_model(robot)
    assert model.nq == 2 * original.nq and model.nv == 2 * original.nv
    ghost_bodies = [i for i in range(model.nbody) if model.body(i).name.startswith("ref_")]
    ghost_geoms = np.isin(model.geom_bodyid, ghost_bodies)
    assert ghost_geoms.sum() == (original.geom_group == 2).sum()
    assert np.all(model.geom_contype[ghost_geoms] == 0)
    assert np.all(model.geom_conaffinity[ghost_geoms] == 0)
    assert np.all(model.geom_group[ghost_geoms] == 1)
    data = mujoco.MjData(model)
    data.qpos[:] = np.tile(original.qpos0, 2)
    mujoco.mj_kinematics(model, data)
    assert np.isfinite(data.xpos).all()


def test_khan_anchor_policy_and_critic_keep_production_hidden_layers():
    cfg = PolicyCfg(num_joints=27, goal_dim=649)
    policy = Policy(cfg)
    critic = Critic(CriticCfg(), pcfg=cfg)
    encoder_layers = [layer for layer in policy.encoders["motion"].mlp.net
                      if isinstance(layer, torch.nn.Linear)]
    decoder_layers = [layer for layer in policy.decoder.mlp.net
                      if isinstance(layer, torch.nn.Linear)]
    critic_layers = [layer for layer in critic.mlp.net if isinstance(layer, torch.nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in encoder_layers] == [
        (649, 2048), (2048, 1024), (1024, 512), (512, 256), (256, 32)
    ]
    assert [(layer.in_features, layer.out_features) for layer in decoder_layers] == [
        (902, 2048), (2048, 1024), (1024, 512), (512, 256), (256, 27)
    ]
    assert [(layer.in_features, layer.out_features) for layer in critic_layers] == [
        (751, 256), (256, 128), (128, 1)
    ]
    observations, goals = torch.randn(2, 870), torch.randn(2, 649)
    action = policy(observations, goals)
    value = critic(torch.randn(2, 67), goals, torch.randn(2, 31), torch.randn(2, 4))
    assert action.shape == (2, 27) and value.shape == (2,)
    loss = action.square().mean() + value.square().mean()
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for model in (policy, critic) for parameter in model.parameters())


@pytest.mark.parametrize("saved_anchor,current_anchor", [(False, True), (True, False)])
def test_training_resume_rejects_anchor_switch_before_loading_models(tmp_path, saved_anchor, current_anchor):
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer

    trainer = Trainer.__new__(Trainer)
    trainer.cfg = TrainConfig(robot="khan_mini", device="cpu")
    trainer.cfg.env.anchor_obs = current_anchor
    trainer.cfg.policy.num_joints = 27
    trainer.cfg.policy.goal_dim = 640 + (9 if current_anchor else 0)
    trainer.device = "cpu"
    path = tmp_path / "wrong_anchor.pt"
    torch.save({"config": {"robot": "khan_mini",
                           "env": {"anchor_obs": saved_anchor, "future_frames": 10},
                           "policy": {"num_joints": 27, "history": 10,
                                      "goal_dim": 640 + (9 if saved_anchor else 0)}}}, path)
    # Deliberately no model or optimizer attributes: reject metadata before loading any state.
    with pytest.raises(ValueError, match="Checkpoint anchor_obs"):
        trainer.load(path)


@pytest.mark.parametrize("field,value", [("future_frames", 9), ("history", 3), ("goal_dim", 640)])
def test_training_resume_rejects_goal_or_history_layout_before_loading_models(tmp_path, field, value):
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer

    trainer = Trainer.__new__(Trainer)
    trainer.cfg = TrainConfig(robot="khan_mini", device="cpu")
    trainer.cfg.env.anchor_obs = True
    trainer.cfg.policy.num_joints = 27
    trainer.cfg.policy.goal_dim = 649
    trainer.device = "cpu"
    saved = {"robot": "khan_mini", "env": {"anchor_obs": True, "future_frames": 10},
             "policy": {"num_joints": 27, "history": 10, "goal_dim": 649}}
    saved["env" if field == "future_frames" else "policy"][field] = value
    path = tmp_path / "wrong_goal_layout.pt"
    torch.save({"config": saved}, path)
    with pytest.raises(ValueError, match="Checkpoint goal/observation layout differs"):
        trainer.load(path)

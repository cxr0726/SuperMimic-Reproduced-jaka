"""Official hidden blocks preserve actor/critic IO, statistics and old checkpoints."""

import copy
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch
from torch import nn

from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.policy.ema import ActorEMA
from supermimic.policy.nets import Critic, CriticCfg, Policy, PolicyCfg
from supermimic.robots.contracts import robot_contract


def _cfg(layout="isaac_khan", *, normalize=True, **kwargs):
    khan = layout == "isaac_khan"
    return PolicyCfg(
        encoder_hidden=(16, 8), decoder_hidden=(16, 8), latent_dim=4,
        num_joints=27 if khan else 29, num_tracked_bodies=15 if khan else 14,
        history=5 if khan else 3, future_frames=5 if khan else 2,
        goal_dim=155 if khan else 136, observation_layout=layout,
        running_normalization=normalize, hidden_layer_norm=True,
        hidden_activation="gelu", **kwargs,
    )


def _assert_blocks(network, hidden_count):
    layers = list(network.net)
    assert len(layers) == hidden_count * 3 + 1
    for index in range(hidden_count):
        assert [type(layer) for layer in layers[index * 3:index * 3 + 3]] == [
            nn.Linear, nn.LayerNorm, nn.GELU,
        ]
        assert layers[index * 3 + 1].eps == 1e-5
        assert layers[index * 3 + 1].elementwise_affine
    assert type(layers[-1]) is nn.Linear


@pytest.mark.parametrize("layout", ["isaac_khan", "supermimic"])
@pytest.mark.parametrize("normalize", [False, True])
def test_all_actor_critic_hidden_blocks_and_gradients(layout, normalize):
    cfg = _cfg(layout, normalize=normalize)
    policy = Policy(cfg)
    critic = Critic(CriticCfg(hidden=(12, 8), hidden_layer_norm=True,
                              hidden_activation="gelu"), pcfg=cfg,
                    xi_dim=91 if layout == "isaac_khan" else 33)
    for model in (policy.encoders["motion"], policy.decoder, critic):
        _assert_blocks(model.mlp, 2)
    obs_width = cfg.history * (3 * cfg.num_joints + 6) + (30 if layout == "isaac_khan" else 0)
    obs, goal = torch.randn(7, obs_width), torch.randn(7, cfg.goal_dim)
    if normalize:
        policy.update_normalization(obs, goal)
    before = {name: value.clone() for name, value in policy.named_buffers()}
    obs.requires_grad_()
    goal.requires_grad_()
    action = policy(obs, goal)
    assert action.shape == (7, cfg.num_joints)
    action.square().sum().backward()
    assert torch.isfinite(obs.grad).all() and obs.grad.abs().sum() > 0
    assert torch.isfinite(goal.grad).all() and goal.grad.abs().sum() > 0
    for module in (policy.encoders["motion"], policy.decoder):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert all(value is not None and torch.isfinite(value).all() for value in gradients)
    for name, value in policy.named_buffers():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    # The frozen critic still provides the terminal-state gradient used by actor BPTT.
    critic.requires_grad_(False)
    feat = torch.randn(7, 13 + 2 * cfg.num_joints, requires_grad=True)
    critic_goal = torch.randn(7, 365 if layout == "isaac_khan" else cfg.goal_dim,
                              requires_grad=True)
    xi = torch.randn(7, 91 if layout == "isaac_khan" else 33)
    root = torch.randn(7, 4, requires_grad=True)
    errors = torch.randn(7, 78, requires_grad=True) if layout == "isaac_khan" else None
    values = critic(feat, critic_goal, xi, root, errors)
    assert values.shape == (7,)
    values.sum().backward()
    for inputs in (feat, critic_goal, root, errors):
        if inputs is not None:
            assert torch.isfinite(inputs.grad).all() and inputs.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in critic.parameters())


def test_encoder_tanh_and_unbounded_final_action_value_are_preserved():
    cfg = _cfg()
    policy = Policy(cfg)
    critic = Critic(CriticCfg(hidden=(8,), hidden_layer_norm=True,
                              hidden_activation="gelu"), pcfg=cfg)
    goal, obs = torch.randn(3, 155), torch.randn(3, 465)
    encoder = policy.encoders["motion"]
    torch.testing.assert_close(encoder(goal), encoder.mlp(encoder.normalizer(goal)).tanh(),
                               atol=0, rtol=0)
    with torch.no_grad():
        policy.decoder.mlp.net[-1].weight.zero_()
        policy.decoder.mlp.net[-1].bias.fill_(2.)
        critic.mlp.net[-1].weight.zero_()
        critic.mlp.net[-1].bias.fill_(-3.)
    torch.testing.assert_close(policy(obs, goal), torch.full((3, 27), 2.), atol=0, rtol=0)
    torch.testing.assert_close(critic(torch.zeros(3, 67), torch.zeros(3, 365),
                                     torch.zeros(3, 31), torch.zeros(3, 4),
                                     torch.zeros(3, 78)), torch.full((3,), -3.), atol=0, rtol=0)


def test_decoder_initialization_gain_and_parameter_delta():
    cfg = _cfg()
    policy = Policy(cfg)
    last = policy.decoder.mlp.net[-1]
    torch.testing.assert_close(last.bias, torch.zeros_like(last.bias), atol=0, rtol=0)
    # Orthogonal_ on a tall output matrix gives orthogonal columns.
    gram = last.weight.T @ last.weight
    torch.testing.assert_close(gram, torch.eye(8) * cfg.decoder_last_gain**2,
                               atol=1e-10, rtol=1e-5)
    production = replace(cfg, encoder_hidden=(1024, 512, 256),
                         decoder_hidden=(2048, 1024, 512, 256), latent_dim=32)
    new = Policy(production)
    old = Policy(replace(production, hidden_layer_norm=False, hidden_activation="silu"))
    count = lambda model: sum(parameter.numel() for parameter in model.parameters())
    assert count(new) - count(old) == 11264
    new_critic = Critic(CriticCfg(hidden_layer_norm=True, hidden_activation="gelu"), pcfg=cfg)
    old_critic = Critic(CriticCfg(), pcfg=cfg)
    assert count(new_critic) - count(old_critic) == 768


def test_ema_includes_layer_norm_parameters_and_copies_running_statistics():
    policy = Policy(_cfg())
    ema = ActorEMA(policy, decay=.995)
    assert any("net.1.weight" in name for name, _ in ema.module.named_parameters())
    obs, goal = torch.randn(5, 465), torch.randn(5, 155)
    policy.update_normalization(obs, goal)
    ema.sync_buffers(policy)
    assert ema.num_updates == 0
    optimizer = torch.optim.Adam(policy.parameters(), lr=2e-4)
    optimizer.zero_grad()
    policy(obs, goal).square().sum().backward()
    optimizer.step()
    ema.update(policy)
    assert ema.num_updates == 1
    torch.testing.assert_close(ema.module(obs, goal), policy(obs, goal), atol=0, rtol=0)
    for name, value in ema.module.state_dict().items():
        torch.testing.assert_close(value, policy.state_dict()[name], atol=0, rtol=0)


@pytest.mark.parametrize("which", ["actor", "critic"])
def test_unknown_hidden_activation_is_rejected(which):
    with pytest.raises(ValueError, match="Unknown policy/critic hidden activation"):
        if which == "actor":
            Policy(replace(_cfg(), hidden_activation="relu"))
        else:
            Critic(CriticCfg(hidden_activation="relu"), pcfg=_cfg())


def _checkpoint(tmp_path, *, modern=True, normalize=True):
    cfg = _cfg(normalize=normalize)
    if not modern:
        cfg.hidden_layer_norm, cfg.hidden_activation = False, "silu"
    policy = Policy(cfg)
    obs, goal = torch.randn(11, 465), torch.randn(11, 155)
    if normalize:
        policy.update_normalization(obs, goal)
    config = asdict(cfg)
    if not modern:
        config.pop("hidden_layer_norm")
        config.pop("hidden_activation")
        if not normalize:
            config.pop("running_normalization")
    checkpoint = {
        "iteration": 50, "policy": copy.deepcopy(policy.state_dict()),
        "config": {"robot": "khan_mini", "policy": config,
                   "env": {"future_frames": 5, "observation_layout": "isaac_khan", "anchor_obs": False},
                   "world_model": {"state_spec": {"dt": .02}}},
        "robot_contract": robot_contract("khan_mini", "isaac_khan"),
    }
    path = tmp_path / "actor.pt"
    torch.save(checkpoint, path)
    return path, policy.eval(), obs, goal


@pytest.mark.parametrize("modern,normalize", [(True, True), (False, True), (False, False)])
def test_checkpoint_preserves_saved_hidden_structure_or_legacy_fallback(tmp_path, modern, normalize):
    path, original, obs, goal = _checkpoint(tmp_path, modern=modern, normalize=normalize)
    restored, _ = load_policy_from_checkpoint(path, "cpu")
    assert restored.cfg.hidden_layer_norm is modern
    assert restored.cfg.hidden_activation == ("gelu" if modern else "silu")
    assert restored.cfg.running_normalization is normalize
    torch.testing.assert_close(restored(obs, goal), original(obs, goal), atol=0, rtol=0)


@pytest.mark.parametrize("input_layout", ["packed", "split"])
def test_layer_norm_gelu_actor_onnx_dynamic_batch_parity(tmp_path, input_layout):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from supermimic.deployment.onnx_export import export_policy

    path, policy, obs, goal = _checkpoint(tmp_path)
    output = tmp_path / f"actor_ln_{input_layout}.onnx"
    result = export_policy(path, output, input_layout=input_layout, validation_samples=8)
    assert result["validation"]["all_checks_passed"]
    exported = onnx.load(output)
    onnx.checker.check_model(exported, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(output), options, providers=["CPUExecutionProvider"])
    for batch in (1, 3):
        sensor, command = obs[:batch], goal[:batch]
        feeds = ({"obs": torch.cat([command, sensor], -1).numpy()} if input_layout == "packed"
                 else {"policy_obs": sensor.numpy(), "goal": command.numpy()})
        with torch.no_grad():
            expected = policy(sensor, command).numpy()
        np.testing.assert_allclose(session.run(["actions"], feeds)[0], expected,
                                   rtol=1e-4, atol=1e-6)

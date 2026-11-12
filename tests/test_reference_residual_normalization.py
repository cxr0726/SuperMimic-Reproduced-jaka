"""Physical reference bypass under normalized, noisy LN/GELU actor inputs."""

import copy
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from supermimic.observation_layout import reference_joint_positions
from supermimic.policy.ema import ActorEMA
from supermimic.policy.nets import Policy, PolicyCfg


def _cfg(**overrides):
    return replace(PolicyCfg(
        encoder_hidden=(16, 8), decoder_hidden=(16, 8), latent_dim=4,
        history=5, future_frames=5, num_joints=27, num_tracked_bodies=15,
        goal_dim=155, observation_layout="isaac_khan",
        action_mode="reference_residual", running_normalization=True,
        hidden_layer_norm=True, hidden_activation="gelu",
        goal_noise_z=.01, goal_noise_joint_pos=.05,
        obs_term_scale=(1.,) * 6, goal_term_scale=(1.,) * 3,
    ), **overrides)


def _policy(cfg=None, *, physical_profile=False):
    if physical_profile:
        from supermimic.robots.khan_mini import load_profile

        profile = load_profile()
        names = profile["joint_names"]
        q0 = torch.tensor([profile["default_joint_pos"][name] for name in names])
        scale = torch.tensor([profile["action_scale"][name] for name in names])
    else:
        q0, scale = torch.linspace(-.4, .3, 27), torch.linspace(.2, .8, 27)
    policy = Policy(cfg or _cfg(), default_joint_pos=q0, action_scale=scale)
    rng = torch.Generator().manual_seed(651)
    # Mean/std differ by coordinate and future offset, so raw and standardized
    # radians cannot accidentally give the same bypass or the same gradient.
    goals = (torch.randn(23, 155, generator=rng) * torch.linspace(.2, 1.7, 155)
             + torch.linspace(-3., 4., 155))
    observations = torch.randn(23, 465, generator=rng) * 1.3 + 2.
    policy.update_normalization(observations, goals)
    return policy


def _inputs(batch=3):
    rng = torch.Generator().manual_seed(921)
    obs = torch.randn(batch, 465, generator=rng)
    joints = (.1 * torch.arange(batch)[:, None, None]
              + .3 * torch.arange(5)[None, :, None]
              + .02 * torch.arange(27)[None, None, :])
    goal = torch.cat([torch.full((batch, 15), .2), torch.full((batch, 5), .65),
                      joints.flatten(1)], -1)
    return obs, goal, joints


def _zero_residual(policy):
    with torch.no_grad():
        policy.decoder.mlp.net[-1].weight.zero_()
        policy.decoder.mlp.net[-1].bias.zero_()


@pytest.mark.parametrize("offset", [0, 1, 4])
def test_normalized_noisy_encoder_zero_residual_retains_raw_reference_radians(offset):
    policy = _policy(_cfg(reference_frame_offset=offset))
    _zero_residual(policy)
    obs, goal, joints = _inputs()
    noisy = policy.make_encoder_goal(goal)
    before = {name: value.clone() for name, value in policy.named_buffers()}
    clean_before, noisy_before = goal.clone(), noisy.clone()
    encoder_inputs = []
    hook = policy.encoders["motion"].mlp.register_forward_pre_hook(
        lambda module, args: encoder_inputs.append(args[0].detach().clone()))
    action = policy(obs, goal, encoder_goal=noisy)
    hook.remove()
    expected_normalized = policy.encoders["motion"].normalizer(noisy)
    torch.testing.assert_close(encoder_inputs[0], expected_normalized, rtol=0, atol=0)
    assert not torch.allclose(expected_normalized, goal)
    target = policy.reference_q_default + policy.reference_action_scale * action
    torch.testing.assert_close(target, joints[:, offset], rtol=1e-6, atol=2e-7)
    torch.testing.assert_close(goal, clean_before, rtol=0, atol=0)
    torch.testing.assert_close(noisy, noisy_before, rtol=0, atol=0)
    for name, value in policy.named_buffers():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_reference_bypass_is_invariant_to_encoder_and_sensor_normalizer_moments():
    policy = _policy()
    _zero_residual(policy)
    obs, goal, _ = _inputs()
    first = policy(obs, goal, encoder_goal=goal + 10.)
    with torch.no_grad():
        for module in (policy.encoders["motion"].normalizer,
                       policy.decoder.observation_normalizer.sensor,
                       policy.decoder.observation_normalizer.orientation):
            module._mean.add_(50.)
            module._var.mul_(.3)
            module._std.copy_(module._var.sqrt())
    second = policy(obs, goal, encoder_goal=goal - 10.)
    torch.testing.assert_close(second, first, rtol=0, atol=0)


def test_reference_gradient_uses_physical_units_without_detach_or_normalizer_scale():
    beta = .7
    policy = _policy(_cfg(reference_residual_beta=beta))
    _zero_residual(policy)
    obs, goal, _ = _inputs()
    goal.requires_grad_()
    encoder_goal = (goal.detach() + 3.).requires_grad_()
    action = policy(obs, goal, encoder_goal=encoder_goal)
    gradient = torch.autograd.grad(action.sum(), goal, retain_graph=True)[0]
    expected = torch.zeros_like(goal)
    expected[:, 47:74] = beta / policy.reference_action_scale
    torch.testing.assert_close(gradient, expected, rtol=0, atol=0)
    target = policy.reference_q_default + policy.reference_action_scale * action
    target_gradient = torch.autograd.grad(target.sum(), goal)[0]
    expected[:, 47:74] = beta
    torch.testing.assert_close(target_gradient, expected, rtol=1e-6, atol=1e-7)


def test_nonzero_decoder_normalized_residual_and_raw_reference_compose_once():
    policy = _policy(_cfg(reference_residual_beta=.6))
    obs, goal, _ = _inputs()
    noisy = policy.make_encoder_goal(goal)
    residual = policy.decoder(policy.encoders["motion"](noisy), obs)
    q_ref = reference_joint_positions(goal, layout="isaac_khan", num_joints=27,
                                      future_frames=5, frame_offset=1)
    skip = .6 * (q_ref - policy.reference_q_default) / policy.reference_action_scale
    torch.testing.assert_close(policy(obs, goal, encoder_goal=noisy), residual + skip,
                               rtol=0, atol=0)
    assert residual.abs().max() > 0


def test_compiled_normalized_residual_values_and_gradients_match_eager():
    policy = _policy()
    other = copy.deepcopy(policy)
    compiled = torch.compile(other, backend="eager", fullgraph=True)
    obs, goal, _ = _inputs()
    noisy = policy.make_encoder_goal(goal)
    eager_action = policy(obs, goal, encoder_goal=noisy)
    compiled_action = compiled(obs, goal, encoder_goal=noisy)
    torch.testing.assert_close(compiled_action, eager_action, rtol=0, atol=0)
    eager_action.square().mean().backward()
    compiled_action.square().mean().backward()
    for original, traced in zip(policy.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(traced.grad, original.grad, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["packed", "split"])
def test_normalized_ln_residual_ema_checkpoint_and_onnx_preserve_complete_action(tmp_path, layout):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from supermimic.deployment.onnx_export import export_policy
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    from supermimic.robots.contracts import robot_contract

    policy = _policy(physical_profile=True)
    ema = ActorEMA(policy)
    ema.update(policy)
    with torch.no_grad():
        policy.decoder.mlp.net[-1].bias.add_(.4)
    ema.update(policy)
    obs, goal, _ = _inputs()
    with torch.no_grad():
        expected = ema.module(obs, goal)
        assert not torch.equal(policy(obs, goal), expected)
    checkpoint = {
        "iteration": 50, "policy": policy.state_dict(), "policy_ema": ema.state_dict(),
        "policy_ema_decay": ema.decay, "policy_ema_updates": ema.num_updates,
        "config": {"robot": "khan_mini", "policy": asdict(policy.cfg),
                   "env": {"future_frames": 5, "observation_layout": "isaac_khan", "anchor_obs": False},
                   "world_model": {"state_spec": {"dt": .02}}},
        "robot_contract": robot_contract("khan_mini", "isaac_khan"),
    }
    path = tmp_path / "normalized_residual.pt"
    torch.save(checkpoint, path)
    restored, _ = load_policy_from_checkpoint(path, "cpu", policy_weights="ema")
    assert restored.cfg.action_mode == "reference_residual"
    assert restored.cfg.running_normalization and restored.cfg.hidden_layer_norm
    assert restored.cfg.hidden_activation == "gelu"
    with torch.no_grad():
        torch.testing.assert_close(restored(obs, goal), expected, rtol=0, atol=0)
    output = tmp_path / f"normalized_residual_{layout}.onnx"
    report = export_policy(path, output, input_layout=layout, validation_samples=3,
                           policy_weights="ema")
    assert report["validation"]["all_checks_passed"]
    assert report["schema"]["action"]["reference_skip_embedded"]
    assert report["schema"]["action"]["reference_skip_units"] == "raw joint radians"
    assert report["metadata"]["running_normalization"] == "true"
    onnx.checker.check_model(onnx.load(output), full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(output), options, providers=["CPUExecutionProvider"])
    for batch in (1, 3):
        feed = ({"obs": torch.cat([goal[:batch], obs[:batch]], -1).numpy()} if layout == "packed"
                else {"policy_obs": obs[:batch].numpy(), "goal": goal[:batch].numpy()})
        np.testing.assert_allclose(session.run(["actions"], feed)[0], expected[:batch].numpy(),
                                   rtol=1e-5, atol=2e-6)
    ema_buffers = dict(ema.module.named_buffers())
    for name, value in restored.named_buffers():
        torch.testing.assert_close(value, ema_buffers[name], rtol=0, atol=0)

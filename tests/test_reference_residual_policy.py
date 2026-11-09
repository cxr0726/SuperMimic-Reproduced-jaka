"""The reference skip is a PD prior, not a second route for encoder noise."""

import copy
from dataclasses import replace

import pytest
import torch

from supermimic.observation_layout import reference_joint_positions
from supermimic.policy.nets import Policy, PolicyCfg


def _cfg(**kwargs):
    cfg = PolicyCfg(
        encoder_hidden=(16,), decoder_hidden=(16,), latent_dim=4,
        history=5, future_frames=5, num_joints=27, goal_dim=155,
        observation_layout="isaac_khan", action_mode="reference_residual",
        goal_noise_z=0.01, goal_noise_joint_pos=0.05,
    )
    return replace(cfg, **kwargs)


def _policy(cfg=None):
    # Nonuniform defaults/scales also catch missed alpha cancellation.
    return Policy(
        cfg or _cfg(), default_joint_pos=torch.linspace(-0.4, 0.3, 27),
        action_scale=torch.linspace(0.2, 0.7, 27),
    )


def _zero_decoder(policy):
    with torch.no_grad():
        for parameter in policy.decoder.parameters():
            parameter.zero_()


def _inputs(batch=3):
    # Distinct joint, time and environment values detect incorrect block/phase.
    joints = (torch.arange(batch)[:, None, None] * 0.1
              + torch.arange(5)[None, :, None] * 0.2
              + torch.arange(27)[None, None, :] * 0.01)
    goal = torch.cat([torch.zeros(batch, 15), torch.full((batch, 5), 0.6),
                      joints.flatten(1)], -1)
    return torch.randn(batch, 465), goal, joints


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("offset", [0, 1, 4])
def test_zero_decoder_pd_target_matches_selected_reference_and_beta(beta, offset):
    policy = _policy(_cfg(reference_residual_beta=beta, reference_frame_offset=offset))
    _zero_decoder(policy)
    obs, goal, joints = _inputs()
    total = policy(obs, goal)
    target = policy.reference_q_default + policy.reference_action_scale * total
    expected = (1 - beta) * policy.reference_q_default + beta * joints[:, offset]
    torch.testing.assert_close(target, expected, rtol=1e-6, atol=1e-7)


def test_noisy_encoder_does_not_change_reference_skip_or_clean_goal():
    policy = _policy()
    _zero_decoder(policy)
    obs, goal, joints = _inputs(8)
    clean = goal.clone()
    encoder_goal = policy.make_encoder_goal(goal)
    assert not torch.equal(encoder_goal[:, 20:], goal[:, 20:])
    action = policy(obs, goal, encoder_goal=encoder_goal)
    target = policy.reference_q_default + policy.reference_action_scale * action
    torch.testing.assert_close(target, joints[:, 1], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(goal, clean, rtol=0, atol=0)
    # Forward is deterministic even with policy.train(); augmentation is explicit.
    torch.testing.assert_close(policy(obs, goal), action, rtol=0, atol=0)


def test_noise_reaches_encoder_and_policy_gradient_without_mutating_reference():
    policy = _policy()
    obs, goal, _ = _inputs()
    encoder_goal = policy.make_encoder_goal(goal)
    clean_encoded = policy.encoders["motion"](goal)
    noisy_encoded = policy.encoders["motion"](encoder_goal)
    assert not torch.equal(clean_encoded, noisy_encoded)
    action = policy(obs, goal, encoder_goal=encoder_goal)
    residual = policy.decoder(noisy_encoded, obs)
    expected = residual + (goal[:, 47:74] - policy.reference_q_default) / policy.reference_action_scale
    torch.testing.assert_close(action, expected, rtol=0, atol=0)
    action.square().mean().backward()
    for parameter in policy.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_noise_is_independent_uniform_in_physical_goal_units():
    policy = _policy()
    torch.manual_seed(391)
    goal = torch.zeros(8192, 155)
    noisy = policy.make_encoder_goal(goal)
    assert torch.equal(noisy[:, :15], goal[:, :15])
    for samples, half in ((noisy[:, 15:20], 0.01), (noisy[:, 20:], 0.05)):
        assert samples.abs().max() <= half
        assert abs(samples.mean()) < half * 0.02
        torch.testing.assert_close(samples.square().mean(), torch.tensor(half**2 / 3),
                                   rtol=0.02, atol=0)
    q_noise = noisy[:, 20:].reshape(-1, 5, 27)
    # The reference window is neither one common bias nor a broadcast joint draw.
    assert (q_noise[:, 0] != q_noise[:, 1]).float().mean() > 0.99
    assert (q_noise[..., 0] != q_noise[..., 1]).float().mean() > 0.99
    correlation = (q_noise[:, 0, 0] * q_noise[:, 1, 0]).mean() / (0.05**2 / 3)
    assert abs(correlation) < 0.04
    assert not torch.equal(noisy, policy.make_encoder_goal(goal))
    assert torch.count_nonzero(goal) == 0


@pytest.mark.parametrize("enabled,zero_widths", [(False, False), (True, True)])
def test_disabled_noise_keeps_identity_and_rng(enabled, zero_widths):
    cfg = _cfg(goal_noise_z=0, goal_noise_joint_pos=0) if zero_widths else _cfg()
    policy = _policy(cfg)
    _, goal, _ = _inputs()
    rng = torch.random.get_rng_state().clone()
    assert policy.make_encoder_goal(goal, enabled=enabled) is goal
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_legacy_policy_has_identical_state_dict_and_forward_contract():
    cfg = _cfg(action_mode="default", goal_noise_z=0, goal_noise_joint_pos=0)
    policy = Policy(cfg)
    state = policy.state_dict()
    assert not any(key.startswith(("reference_", "encoder_goal_noise")) for key in state)
    clone = Policy(cfg)
    clone.load_state_dict(state, strict=True)
    obs, goal, _ = _inputs()
    expected = policy.decoder(policy.encoders["motion"](goal), obs)
    torch.testing.assert_close(policy(obs, goal), expected, rtol=0, atol=0)
    torch.testing.assert_close(clone(obs, goal), expected, rtol=0, atol=0)


def test_saved_reference_constants_are_independent_and_restored():
    defaults = torch.arange(27, dtype=torch.float32) / 100
    scales = torch.full((27,), 0.5)
    policy = Policy(_cfg(), default_joint_pos=defaults, action_scale=scales)
    defaults.zero_()
    scales.fill_(9)
    assert policy.reference_q_default[-1] == 0.26
    assert policy.reference_action_scale[0] == 0.5
    clone = _policy()
    clone.load_state_dict(policy.state_dict(), strict=True)
    obs, goal, _ = _inputs()
    torch.testing.assert_close(policy(obs, goal), clone(obs, goal), rtol=0, atol=0)


@pytest.mark.parametrize("overrides", [
    {"action_mode": "auto"}, {"action_mode": "unknown"},
    {"goal_noise_z": None}, {"goal_noise_joint_pos": None},
    {"goal_noise_z": -0.01}, {"goal_noise_joint_pos": float("inf")},
    {"reference_residual_beta": -0.1}, {"reference_residual_beta": 1.1},
    {"reference_residual_beta": float("nan")},
    {"reference_frame_offset": -1}, {"reference_frame_offset": 5},
    {"reference_frame_offset": 1.5},
    {"observation_layout": "supermimic"},
])
def test_invalid_policy_settings_fail_before_training(overrides):
    with pytest.raises(ValueError):
        _policy(_cfg(**overrides))


@pytest.mark.parametrize("defaults,scales", [
    (None, None), (torch.zeros(26), torch.ones(27)),
    (torch.zeros(27), torch.zeros(27)), (torch.zeros(27), -torch.ones(27)),
    (torch.full((27,), float("nan")), torch.ones(27)),
])
def test_reference_constants_must_match_joint_shape_and_be_finite(defaults, scales):
    with pytest.raises(ValueError):
        Policy(_cfg(), default_joint_pos=defaults, action_scale=scales)


def test_reference_extractor_rejects_wrong_goal_width():
    with pytest.raises(ValueError, match="width"):
        reference_joint_positions(torch.zeros(2, 154), layout="isaac_khan", num_joints=27,
                                  future_frames=5, frame_offset=1)


def test_compiled_forward_matches_eager_values_and_parameter_gradients():
    policy = _policy()
    compiled_policy = copy.deepcopy(policy)
    compiled = torch.compile(compiled_policy, backend="eager", fullgraph=True)
    obs, goal, _ = _inputs()
    encoder_goal = policy.make_encoder_goal(goal)
    eager_action = policy(obs, goal, encoder_goal=encoder_goal)
    compiled_action = compiled(obs, goal, encoder_goal=encoder_goal)
    torch.testing.assert_close(compiled_action, eager_action, rtol=0, atol=0)
    eager_action.square().sum().backward()
    compiled_action.square().sum().backward()
    for eager, compiled_parameter in zip(policy.parameters(), compiled_policy.parameters()):
        torch.testing.assert_close(compiled_parameter.grad, eager.grad, rtol=0, atol=0)

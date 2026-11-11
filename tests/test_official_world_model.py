"""Official WM input moments, hidden blocks, and analytic termination contracts."""

import copy
import io
import math
from dataclasses import replace

import pytest
import torch

from supermimic.data.replay_buffer import StartBatch
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.imagination.rollout import imagine
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.robots.registry import get_robot_spec
from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


@pytest.fixture(params=["g1", "khan_mini"])
def heads(request):
    robot = get_robot_spec(request.param)
    model = robot.load_model()
    return AnalyticHeads(
        G1Kinematics(model), robot.tracked_bodies, robot.anchor_body, robot.root_body,
        torch.tensor(model.jnt_range[1:], dtype=torch.float32),
        torch.tensor(model.qpos0[7:], dtype=torch.float32),
    )


def _model(heads, **kwargs):
    cfg = WorldModelCfg(
        hidden=(12, 8), term_hidden=(8,), input_normalization=True,
        hidden_layer_norm=True, hidden_activation="gelu", termination_mode="analytic",
        **kwargs,
    )
    count = heads.num_joints
    scale = torch.linspace(.2, .6, count)
    offset = torch.linspace(-.15, .25, count)
    return WorldModel(cfg, heads, "cpu", action_scale=scale, action_offset=offset)


def _states(wm, count=6):
    generator = torch.Generator().manual_seed(427)
    state = .1 * torch.randn(count, wm.spec.state_dim, generator=generator)
    state[:, 2] += .65
    quat = torch.randn(count, 4, generator=generator)
    state[:, 3:7] = quat / quat.norm(dim=-1, keepdim=True)
    state[:, 7:wm.spec.nq] += wm.heads.default_joint_pos
    return state


def _reference(wm, state):
    bodies = wm.heads.robot_bodies(state)
    return ReferenceFrame(
        *(bodies[name] for name in
          ("pos", "quat", "lin_vel", "ang_vel", "root_pos", "root_quat")),
        state[:, 7:wm.spec.nq], state[:, wm.spec.nq + 6:],
    )


def _normalizer_buffers(wm):
    return {name: buffer.detach().clone() for name, buffer in wm.named_buffers()
            if name.startswith(("state_normalizer.", "action_normalizer."))}


def test_fresh_moments_are_raw_features_and_physical_pd_targets_with_stable_storage(heads):
    wm = _model(heads)
    state = _states(wm)
    action = torch.linspace(-1, 1, 6 * heads.num_joints).reshape(6, heads.num_joints)
    addresses = {name: value.data_ptr() for name, value in wm.named_buffers()}
    wm.update_normalization(state[:2], action[:2])
    wm.update_normalization(state[2:].reshape(2, 2, -1), action[2:].reshape(2, 2, -1))
    raw_features = wm.spec.canonical(state) * wm.spec.in_scale
    physical_targets = action * wm.action_scale + wm.action_offset
    for normalizer, samples in ((wm.state_normalizer, raw_features),
                                (wm.action_normalizer, physical_targets)):
        assert normalizer.count.item() == 6
        torch.testing.assert_close(normalizer.mean, samples.mean(0), rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(normalizer._var, samples.var(0, unbiased=False),
                                   rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(normalizer.std, samples.std(0, unbiased=False),
                                   rtol=1e-5, atol=1e-7)
        assert not normalizer.training
    assert addresses == {name: value.data_ptr() for name, value in wm.named_buffers()}
    # These are physical raw angular/joint velocities, not divided by old scales.
    torch.testing.assert_close(wm.state_normalizer.mean[10:13],
                               state[:, wm.spec.nq + 3:wm.spec.nq + 6].mean(0))
    torch.testing.assert_close(wm.state_normalizer.mean[-heads.num_joints:],
                               state[:, -heads.num_joints:].mean(0))


def test_running_input_transform_replaces_fixed_scale_instead_of_applying_both(heads):
    wm = _model(heads, state_spec=StateSpecCfg(in_scale=(3., 4., 5., 6., 7., 8.)))
    state = _states(wm, count=2)
    action = torch.linspace(-.7, .9, 2 * heads.num_joints).reshape(2, heads.num_joints)
    captured = []
    hook = wm.dynamics.register_forward_pre_hook(lambda module, args: captured.append(args))
    before = _normalizer_buffers(wm)
    wm.step(state, action, wm.spec.xi_offset.repeat(2, 1))
    hook.remove()
    # Initial empirical moments are mean0/std1; official eps is outside sqrt.
    expected = wm.spec.canonical(state) * wm.spec.in_scale / 1.01
    torch.testing.assert_close(captured[0][0], expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(captured[0][0][:, 10:13],
                               state[:, wm.spec.nq + 3:wm.spec.nq + 6] / 1.01)
    torch.testing.assert_close(captured[0][0][:, -heads.num_joints:],
                               state[:, -heads.num_joints:] / 1.01)
    torch.testing.assert_close(captured[0][1],
                               (action * wm.action_scale + wm.action_offset) / 1.01)
    torch.testing.assert_close(captured[0][2], torch.zeros(2, wm.spec.xi_dim))
    torch.testing.assert_close(_normalizer_buffers(wm), before, rtol=0, atol=0)


def test_hidden_layer_norm_gelu_and_legacy_architecture_are_separate(heads):
    wm = _model(heads)
    assert [type(layer) for layer in wm.dynamics.mlp.net] == [
        torch.nn.Linear, torch.nn.LayerNorm, torch.nn.GELU,
        torch.nn.Linear, torch.nn.LayerNorm, torch.nn.GELU, torch.nn.Linear,
    ]
    legacy = WorldModel(WorldModelCfg(hidden=(12, 8), term_hidden=(8,)), heads, "cpu")
    assert [type(layer) for layer in legacy.dynamics.mlp.net] == [
        torch.nn.Linear, torch.nn.SiLU, torch.nn.Linear, torch.nn.SiLU, torch.nn.Linear,
    ]
    assert legacy.term_head is not None and wm.term_head is None
    assert not any(name.startswith("term_head.") for name, _ in wm.named_parameters())
    assert any(name.startswith("term_head.") for name, _ in legacy.named_parameters())
    assert not any("normalizer" in name for name in legacy.state_dict())
    # Official options do not silently remove our zero-output integration prior.
    for model in (wm, legacy):
        assert torch.count_nonzero(model.dynamics.mlp.net[-1].weight) == 0
        assert torch.count_nonzero(model.dynamics.mlp.net[-1].bias) == 0


def test_analytic_termination_probability_matches_both_physical_threshold_rules(heads):
    wm = _model(heads)
    state = torch.zeros(5, wm.spec.state_dim)
    state[:, 3] = 1
    ref = ReferenceFrame(None, None, None, None, state[:, :3].clone(), state[:, 3:7].clone())
    state[:4, 2] = torch.tensor([0., .125, .25, .375])
    roll = math.acos(1. - wm.cfg.tilt_threshold)
    state[4, 3:7] = torch.tensor([math.cos(roll / 2), math.sin(roll / 2), 0., 0.])
    probability = wm.termination_logits(state, ref).sigmoid()
    torch.testing.assert_close(probability[:4], torch.tensor([-20., -10., 0., 10.]).sigmoid(),
                               rtol=1e-6, atol=1e-7)
    assert probability[4].item() == pytest.approx(.5, abs=2e-6)
    assert probability[3] > .9999
    assert probability[0] < 1e-8


def test_analytic_termination_keeps_gradient_to_height_and_tilt(heads):
    wm = _model(heads)
    state = torch.zeros(2, wm.spec.state_dim)
    state[:, 3] = 1
    ref = ReferenceFrame(None, None, None, None, state[:, :3].clone(), state[:, 3:7].clone())
    state[0, 2] = .2
    roll = 1.25
    state[1, 3:7] = torch.tensor([math.cos(roll / 2), math.sin(roll / 2), 0., 0.])
    state.requires_grad_()
    p = wm.termination_logits(state, ref).sigmoid()
    p.sum().backward()
    assert torch.isfinite(state.grad).all()
    expected_height_grad = p[0].detach() * (1 - p[0].detach()) / (.25 * .05)
    torch.testing.assert_close(state.grad[0, 2], expected_height_grad)
    assert state.grad[1, 3:7].abs().sum() > 0
    assert state.grad[0, 7:].count_nonzero() == 0


def test_analytic_wm_loss_has_no_termination_label_fit_and_legacy_bce_stays_active(heads):
    wm = _model(heads, term_loss_weight=100.)
    state = _states(wm, count=2)
    action = torch.zeros(2, heads.num_joints)
    xi = wm.spec.xi_offset.repeat(2, 1)
    ref = _reference(wm, state)
    truth = wm.step(state, action, xi).detach()
    truth[:, 2] += .01
    states = torch.stack([state, truth])
    metrics = []
    for label in (0., 1.):
        result = wm.window_loss(states, action[None], xi,
                                torch.full((1, 2), label), [ref])
        metrics.append(result)
        assert result[1]["term_bce"].item() == 0
        torch.testing.assert_close(result[0].detach(), result[1]["state_loss"])
    torch.testing.assert_close(metrics[0][0], metrics[1][0], rtol=0, atol=0)
    legacy = WorldModel(WorldModelCfg(hidden=(12,), term_hidden=(8,)), heads, "cpu")
    loss, diagnostic = legacy.window_loss(states, action[None], xi, torch.ones(1, 2), [ref])
    assert diagnostic["term_bce"].item() > 0
    loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in legacy.term_head.parameters())


def test_checkpoint_restores_moments_and_the_next_prediction_exactly(heads):
    torch.manual_seed(57)
    wm = _model(heads)
    state = _states(wm)
    action = torch.randn(6, heads.num_joints)
    wm.update_normalization(state, action)
    with torch.no_grad():
        wm.dynamics.mlp.net[-1].weight.normal_(std=.003)
    stream = io.BytesIO()
    torch.save(wm.state_dict(), stream)
    stream.seek(0)
    restored = WorldModel(copy.deepcopy(wm.cfg), heads, "cpu", action_scale=wm.action_scale,
                          action_offset=wm.action_offset)
    addresses = {name: buffer.data_ptr() for name, buffer in restored.named_buffers()}
    restored.load_state_dict(torch.load(stream, weights_only=True))
    assert addresses == {name: buffer.data_ptr() for name, buffer in restored.named_buffers()}
    xi = wm.spec.xi_offset.repeat(6, 1)
    torch.testing.assert_close(wm.step(state, action, xi), restored.step(state, action, xi),
                               rtol=0, atol=0)
    next_state, next_action = state[:3] + .001, action[:3] + .01
    for model in (wm, restored):
        model.update_normalization(next_state, next_action)
    torch.testing.assert_close(wm.state_dict(), restored.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(wm.step(next_state, next_action, xi[:3]),
                               restored.step(next_state, next_action, xi[:3]), rtol=0, atol=0)


def test_forward_loss_and_imagination_do_not_update_empirical_statistics(heads):
    wm = _model(heads)
    state = _states(wm, count=12)
    # Keep a consistent, nearly stationary reference so the real FK rollout is finite.
    state[:] = state[:1]
    wm.update_normalization(state, torch.zeros(12, heads.num_joints))
    library = MotionLibrary.__new__(MotionLibrary)
    library.device, library.dtype, library.compact = torch.device("cpu"), torch.float32, False
    library.default_joint_pos = heads.default_joint_pos
    library._ref = state
    bodies = heads.robot_bodies(state)
    library.body_pos_w, library.body_quat_w = bodies["pos"], bodies["quat"]
    library.body_lin_vel_w, library.body_ang_vel_w = bodies["lin_vel"], bodies["ang_vel"]
    library.frame_clip, library.clip_last = torch.zeros(12, dtype=torch.long), torch.tensor([11])
    library.goal_feat_dim = 2 * heads.num_joints + 10
    library.goal_feat = torch.zeros(12, library.goal_feat_dim)
    policy = Policy(PolicyCfg(
        encoder_hidden=(8,), decoder_hidden=(8,), latent_dim=4, history=2,
        num_joints=heads.num_joints, goal_dim=2 * library.goal_feat_dim,
        future_frames=2,
    ))
    initial = state[:2].clone()
    action = torch.zeros(2, heads.num_joints)
    xi = wm.spec.xi_offset.repeat(2, 1)
    start = StartBatch(initial, xi, torch.zeros(2, dtype=torch.long), action,
                       initial[None].repeat(2, 1, 1), action[None].repeat(2, 1, 1))
    before = _normalizer_buffers(wm)
    for training in (True, False):
        wm.train(training)
        next_state = wm.step(initial, action, xi)
        wm.window_loss(torch.stack([initial, next_state.detach()]), action[None], xi,
                       torch.zeros(1, 2), [_reference(wm, initial)])
        rollout = imagine(wm, policy, library, start, horizon=3, future_frames=2,
                          obs_noise=False)
        assert torch.isfinite(rollout.states).all() and torch.isfinite(rollout.rewards).all()
        torch.testing.assert_close(_normalizer_buffers(wm), before, rtol=0, atol=0)


def test_legacy_forward_remains_the_original_scaled_state_raw_action_path(heads):
    wm = WorldModel(WorldModelCfg(hidden=(8,), term_hidden=(8,)), heads, "cpu")
    state = _states(wm, 2)
    action = torch.randn(2, heads.num_joints)
    xi = wm.spec.xi_offset.repeat(2, 1)
    captured = []
    hook = wm.dynamics.register_forward_pre_hook(lambda module, args: captured.append(args))
    wm.update_normalization(state, action)  # Disabled path is a no-op.
    result = wm.step(state, action, xi)
    hook.remove()
    torch.testing.assert_close(captured[0][0], wm.spec.canonical(state), rtol=0, atol=0)
    torch.testing.assert_close(captured[0][1], action, rtol=0, atol=0)
    torch.testing.assert_close(result, wm.spec.integrate(state, torch.zeros(2, wm.spec.delta_dim)))


@pytest.mark.parametrize("kwargs", [
    {"termination_mode": "invalid"}, {"termination_softness": 0.},
    {"height_threshold": float("nan")}, {"tilt_threshold": -1.},
])
def test_invalid_analytic_termination_configuration_is_rejected(heads, kwargs):
    cfg = replace(WorldModelCfg(hidden=(8,)), **kwargs)
    with pytest.raises(ValueError):
        WorldModel(cfg, heads, "cpu")

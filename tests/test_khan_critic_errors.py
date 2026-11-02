"""Independent critic error geometry and CPU actor/critic training checks."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from supermimic.data.replay_buffer import ReplayBuffer, StartBatch
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.imagination.policy_trainer import PolicyTrainer, PolicyTrainerCfg
from supermimic.imagination.rollout import imagine
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import (
    Critic,
    CriticCfg,
    Policy,
    PolicyCfg,
    critic_error_dimension,
)
from supermimic.robots.khan_mini import load_profile
from supermimic.robots.registry import get_robot_spec
from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


def _wxyz(rotation):
    return np.roll(rotation.as_quat(), 1, axis=-1)


def test_critic_error_geometry_matches_independent_reference_and_is_differentiable():
    rng = np.random.default_rng(873)
    count, bodies, joints = 2, 15, 27
    robot_rotation = Rotation.from_euler("xyz", [[0.4, -0.3, 0.8], [-0.2, 0.5, -0.6]])
    ref_rotation = Rotation.from_euler("xyz", [[-0.2, 0.6, -0.3], [0.5, -0.4, 0.9]])
    state = np.zeros((count, 67))
    state[:, :3], state[:, 3:7] = rng.normal(size=(count, 3)), _wxyz(robot_rotation)
    state[:, 7:34] = rng.normal(size=(count, joints))
    actual_pos = rng.normal(size=(count, bodies, 3))
    actual_pos[:, 0] = state[:, :3]
    ref_pos = rng.normal(size=(count, bodies, 3))
    ref_joint = rng.normal(size=(count, joints))
    actual_quat = np.repeat(state[:, None, 3:7], bodies, axis=1)
    ref_quat = np.repeat(_wxyz(ref_rotation)[:, None], bodies, axis=1)
    reference = ReferenceFrame(
        torch.tensor(ref_pos), torch.tensor(ref_quat), torch.zeros(count, bodies, 3),
        torch.zeros(count, bodies, 3), torch.tensor(ref_pos[:, 0]),
        torch.tensor(_wxyz(ref_rotation)), torch.tensor(ref_joint),
    )
    # Independently reproduce InstantTrack's XY/yaw reanchoring and actual
    # heading frame. Nonzero roll/pitch distinguishes heading from full root.
    expected_body = []
    for i in range(count):
        delta_yaw = (robot_rotation[i] * ref_rotation[i].inv()).as_euler("xyz")[2]
        delta = Rotation.from_euler("z", delta_yaw)
        anchor = np.r_[state[i, :2], ref_pos[i, 0, 2]]
        reanchored = anchor + delta.apply(ref_pos[i] - ref_pos[i, 0])
        heading = Rotation.from_euler("z", robot_rotation[i].as_euler("xyz")[2])
        expected_body.append(heading.inv().apply(reanchored - actual_pos[i]).reshape(-1))
    expected_rotation = (robot_rotation.inv() * ref_rotation).as_matrix()[..., :2].reshape(count, 6)
    expected = np.concatenate((ref_joint - state[:, 7:34], expected_body, expected_rotation), -1)
    heads = AnalyticHeads.__new__(AnalyticHeads)
    heads.fk, heads.anchor_index = SimpleNamespace(nq=34), 0
    state_t = torch.tensor(state, requires_grad=True)
    actual_t = torch.tensor(actual_pos, requires_grad=True)
    errors = heads.critic_error_features(
        state_t, reference, {"pos": actual_t, "quat": torch.tensor(actual_quat)},
    )
    assert errors.shape == (count, 78)
    np.testing.assert_allclose(errors.detach(), expected, rtol=0, atol=1e-12)
    (errors * torch.arange(78)).sum().backward()
    for gradient in (state_t.grad, actual_t.grad):
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().max() > 0
    np.testing.assert_allclose(state_t.grad[:, 7:34], -np.arange(27)[None].repeat(count, 0))


@pytest.mark.parametrize("body_count", [1, 15, 18])
def test_critic_appends_only_khan_errors_and_preserves_existing_input_order(body_count):
    cfg = PolicyCfg(observation_layout="isaac_khan", num_joints=27, goal_dim=155,
                    future_frames=5, num_tracked_bodies=body_count)
    width = 27 + 3 * body_count + 6
    assert critic_error_dimension(cfg) == width
    critic = Critic(CriticCfg(hidden=()), pcfg=cfg)
    assert critic.mlp.net[0].in_features == 467 + width
    parts = [torch.randn(2, size) for size in (67, 365, 31, 4, width)]
    # Inspect the exact input before the unchanged value MLP.
    captured = []
    hook = critic.mlp.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    assert critic(*parts).shape == (2,)
    hook.remove()
    expected = torch.cat((parts[0], parts[1] * critic.goal_scale, *parts[2:]), -1)
    torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="current tracking error"):
        critic(*parts[:4])
    with pytest.raises(ValueError, match="current tracking error"):
        critic(*parts[:4], torch.zeros(2, width - 1))
    original = Critic(CriticCfg(hidden=()))
    assert original.mlp.net[0].in_features == 788
    assert critic_error_dimension(PolicyCfg()) == 0
    assert original(*(torch.zeros(2, size) for size in (71, 680, 33, 4))).shape == (2,)
    assert critic_error_dimension(PolicyCfg(observation_layout="isaac_khan", num_joints=27)) == 78


def _training_components():
    robot = get_robot_spec("khan_mini")
    model = robot.load_model()
    profile = load_profile()
    q0 = torch.tensor([profile["default_joint_pos"][name] for name in robot.joint_names])
    limits = torch.tensor(np.asarray(model.jnt_range[1:]), dtype=torch.float32)
    heads = AnalyticHeads(G1Kinematics(model), robot.tracked_bodies, robot.anchor_body,
                          robot.root_body, limits, q0)
    wm = WorldModel(WorldModelCfg(hidden=(12,), term_hidden=(8,)), heads, "cpu")
    with torch.no_grad():
        wm.dynamics.mlp.net[-1].weight.normal_(std=0.002)
    library = MotionLibrary.__new__(MotionLibrary)
    library.device, library.dtype, library.compact = torch.device("cpu"), torch.float32, False
    library.default_joint_pos = q0
    library._ref = torch.zeros(12, 67)
    library._ref[:, :3] = torch.tensor([0.1, -0.2, 0.65])
    library._ref[:, 0] += torch.arange(12) * 0.01
    library._ref[:, 3] = 1
    library._ref[:, 7:34] = q0 + torch.arange(12)[:, None] * torch.linspace(-0.002, 0.002, 27)
    body = heads.robot_bodies(library._ref)
    library.body_pos_w, library.body_quat_w = body["pos"], body["quat"]
    library.body_lin_vel_w, library.body_ang_vel_w = body["lin_vel"], body["ang_vel"]
    library.frame_clip, library.clip_last = torch.zeros(12, dtype=torch.long), torch.tensor([11])
    cfg = PolicyCfg(encoder_hidden=(12,), decoder_hidden=(12,), latent_dim=4, history=5,
                    future_frames=5, num_joints=27, goal_dim=155, observation_layout="isaac_khan",
                    num_tracked_bodies=len(robot.tracked_bodies))
    return wm, Policy(cfg), Critic(CriticCfg(hidden=(8,)), pcfg=cfg), library


def test_real_khan_fk_errors_at_tracking_and_explicit_neck_error_is_observable():
    wm, _, _, library = _training_components()
    state = library._ref[:2].clone().requires_grad_()
    body = wm.heads.robot_bodies(state)
    reference = ReferenceFrame(*(body[name].detach() for name in
                                 ("pos", "quat", "lin_vel", "ang_vel", "root_pos", "root_quat")),
                               state[:, 7:34].detach(), state[:, 40:].detach())
    errors = wm.heads.critic_error_features(state, reference)
    torch.testing.assert_close(errors[:, :72], torch.zeros(2, 72), rtol=0, atol=1e-7)
    torch.testing.assert_close(errors[:, 72:], torch.tensor([[1., 0., 0., 1., 0., 0.]]).repeat(2, 1))
    # The head link origin coincides with the neck axes, so head position alone
    # need not expose neck rotations. Explicit q errors expose both neck DOFs.
    neck_ids = wm.heads.neck_joint_ids
    derivative, = torch.autograd.grad(errors[:, neck_ids].sum(), state)
    assert torch.isfinite(derivative).all()
    torch.testing.assert_close(derivative[:, [7 + i for i in neck_ids]],
                               -torch.ones(2, len(neck_ids)))


def test_khan_critic_bootstrap_and_update_use_all_current_reference_frames():
    torch.manual_seed(846)
    wm, policy, critic, library = _training_components()
    trainer = PolicyTrainer(wm, policy, critic, library, PolicyTrainerCfg(
        horizon=3, batch_size=4, history=5, future_frames=5, observation_layout="isaac_khan",
        critic_epochs=1, critic_minibatches=2, compile=None, obs_noise=True,
    ))
    frame = torch.tensor([0, 8])  # Second sample bootstraps exactly at the last frame.
    state = library._ref[frame].clone()
    start = StartBatch(state, wm.spec.xi_offset.repeat(2, 1), frame, torch.zeros(2, 27),
                       state.repeat(5, 1, 1), torch.zeros(5, 2, 27))
    rollout = imagine(wm, policy, library, start, 3, 5, step_fn=trainer.step_fn)
    feat, goals, root_error, errors = trainer._critic_inputs(rollout)
    assert errors.shape == (4, 2, 78)
    assert goals.shape == (4, 2, 365)
    assert rollout.frames[:, 1].tolist() == [8, 9, 10, 11]
    torch.testing.assert_close(rollout.critic_reference.joint_pos,
                               library._ref[rollout.frames.reshape(-1), 7:34])
    with torch.no_grad():
        expected = wm.heads.critic_error_features(rollout.states.flatten(0, 1),
                                                 rollout.critic_reference).reshape(4, 2, 78)
    torch.testing.assert_close(errors, expected)
    # Retain the terminal-state BPTT path but avoid retaining the unused FK
    # graphs for earlier value-fit-only frames.
    gradient, = torch.autograd.grad((errors * torch.arange(78)).sum(), rollout.states,
                                   retain_graph=True)
    torch.testing.assert_close(gradient[:-1], torch.zeros_like(gradient[:-1]))
    assert gradient[-1].abs().sum() > 0 and torch.isfinite(gradient).all()
    bootstrap = critic(feat[-1], goals[-1], wm.spec.normalize_xi(start.xi),
                       root_error[-1], errors[-1])
    actor_gradients = torch.autograd.grad(bootstrap.sum(), tuple(policy.parameters()),
                                         allow_unused=True)
    assert any(g is not None and g.abs().sum() > 0 for g in actor_gradients)
    assert all(g is None or torch.isfinite(g).all() for g in actor_gradients)

    buffer = ReplayBuffer(2, 24, 67, 27, 31, "cpu")
    for t in range(10):
        buffer.add(torch.arange(2), library._ref[t].repeat(2, 1), torch.zeros(2, 27),
                   torch.zeros(2, 27), start.xi, torch.zeros(2), torch.zeros(2),
                   torch.full((2,), t), torch.zeros(2, dtype=torch.long),
                   torch.zeros(2, dtype=torch.bool), torch.full((2,), t == 0))
    metrics = trainer.update(buffer)
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["policy_grad_norm"] > 0
    assert metrics["critic_loss"] >= 0
    assert critic.mlp.net[0].in_features == 545

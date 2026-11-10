"""CPU checks for joint-dependent policy normalization and differentiable imagination."""

from types import SimpleNamespace

import pytest
import torch

from supermimic.data.replay_buffer import StartBatch
from supermimic.envs.mdp.motion_library import RefBodies, RefState
from supermimic.imagination.compiled import REWARD_TERM_NAMES
from supermimic.imagination.rollout import OBS_NOISE_HALF, imagine, observation_noise_half
from supermimic.policy.nets import Critic, CriticCfg, Policy, PolicyCfg
from supermimic.world_model.heads import analytic_observation


@pytest.mark.parametrize("num_joints", [27, 29])
@pytest.mark.parametrize("anchor_obs", [False, True])
def test_policy_critic_shapes_scales_and_gradients(num_joints, anchor_obs):
    history, future_frames, batch = 3, 2, 4
    goal_frame_dim = 2 * num_joints + 10
    goal_dim = future_frames * goal_frame_dim + (9 if anchor_obs else 0)
    cfg = PolicyCfg(
        encoder_hidden=(16, 8), decoder_hidden=(16, 8), latent_dim=4,
        history=history, goal_dim=goal_dim, num_joints=num_joints,
    )
    policy = Policy(cfg)
    critic = Critic(CriticCfg(hidden=(8,)), pcfg=cfg)
    obs = torch.randn(batch, history * (3 * num_joints + 6))
    goal = torch.randn(batch, goal_dim)
    action = policy(obs, goal)
    value = critic(
        torch.randn(batch, 13 + 2 * num_joints), goal,
        torch.randn(batch, num_joints + 4), torch.randn(batch, 4),
    )
    assert action.shape == (batch, num_joints)
    assert value.shape == (batch,)
    expected_observation_scale = torch.cat([
        torch.full((history * size,), 1 / scale)
        for size, scale in zip((num_joints, num_joints, 3, 3, num_joints), cfg.obs_term_scale)
    ])
    expected_goal_scale = torch.cat([
        torch.full((size,), 1 / scale)
        for size, scale in zip((num_joints, num_joints, 3, 3, 3, 1), cfg.goal_term_scale)
    ]).repeat(future_frames)
    if anchor_obs:
        expected_goal_scale = torch.cat((expected_goal_scale, torch.ones(9)))
    torch.testing.assert_close(policy.decoder.scale, expected_observation_scale, rtol=0, atol=0)
    torch.testing.assert_close(policy.encoders["motion"].scale, expected_goal_scale, rtol=0, atol=0)
    torch.testing.assert_close(critic.goal_scale, expected_goal_scale, rtol=0, atol=0)
    (action.square().mean() + value.square().mean()).backward()
    for module in (policy, critic):
        gradients = [p.grad for p in module.parameters()]
        assert all(grad is not None and torch.isfinite(grad).all() for grad in gradients)
        assert any(grad.abs().sum() > 0 for grad in gradients)


def test_policy_formal_hidden_defaults_and_explicit_critic_dimensions_remain_available():
    cfg = PolicyCfg()
    assert cfg.num_joints == 29 and cfg.history == 10 and cfg.goal_dim == 680
    assert cfg.encoder_hidden == (1024, 512, 256)
    assert cfg.decoder_hidden == (2048, 1024, 512, 256)
    assert cfg.latent_dim == 32 and cfg.decoder_last_gain == 0.01
    critic = Critic(CriticCfg(hidden=(8,)), feat_dim=5, goal_dim=68, xi_dim=7)
    assert critic(torch.zeros(2, 5), torch.zeros(2, 68), torch.zeros(2, 7), torch.zeros(2, 4)).shape == (2,)
    torch.testing.assert_close(observation_noise_half(29), OBS_NOISE_HALF, rtol=0, atol=0)
    expected = torch.tensor([0.01] * 27 + [0.5] * 27 + [0.2] * 3 + [0.05] * 3 + [0.0] * 27)
    torch.testing.assert_close(observation_noise_half(27), expected, rtol=0, atol=0)


class _ReferenceLibrary:
    def __init__(self, num_joints):
        self.default_joint_pos = torch.zeros(num_joints)

    def last_frame_of(self, frame):
        return torch.full_like(frame, 100)

    def goal_window(self, frame, future_frames):
        return torch.zeros(len(frame), future_frames * (2 * len(self.default_joint_pos) + 10))

    def ref(self, frame):
        n, joints = len(frame), len(self.default_joint_pos)
        quat = torch.zeros(n, 4)
        quat[:, 0] = 1
        return RefState(torch.zeros(n, 3), quat, torch.zeros(n, joints),
                        torch.zeros(n, 3), torch.zeros(n, 3), torch.zeros(n, joints))

    def bodies(self, frame):
        ref = self.ref(frame)
        return RefBodies(ref.root_pos_w[:, None], ref.root_quat_w[:, None],
                         ref.root_lin_vel_w[:, None], ref.root_ang_vel_w[:, None])


@pytest.mark.parametrize("num_joints", [27, 29])
def test_imagination_uses_robot_noise_layout_and_preserves_action_gradients(num_joints):
    """Exercise the real rollout with a differentiable transition, independently of FK."""
    batch, history, future_frames, horizon = 3, 2, 2, 2
    cfg = PolicyCfg(encoder_hidden=(8,), decoder_hidden=(8,), latent_dim=4,
                    history=history, goal_dim=future_frames * (2 * num_joints + 10),
                    num_joints=num_joints)
    policy = Policy(cfg)
    library = _ReferenceLibrary(num_joints)
    state = torch.zeros(batch, 13 + 2 * num_joints)
    state[:, 3] = 1
    xi = torch.zeros(batch, num_joints + 4)
    start = StartBatch(state, xi, torch.zeros(batch, dtype=torch.long),
                       torch.zeros(batch, num_joints), state.repeat(history, 1, 1),
                       torch.zeros(history, batch, num_joints))
    heads = SimpleNamespace(anchor_index=0,
                            root_error_features=lambda state, ref: state[:, :4] * 0)
    wm = SimpleNamespace(heads=heads)

    def transition(state, action, previous, context, ref):
        del previous, ref
        next_state = torch.cat((state[:, :7], state[:, 7:7 + num_joints] + action,
                                state[:, 7 + num_joints:]), dim=-1)
        reward = -(action - 0.5).square().mean(-1)
        obs = analytic_observation(next_state, action, context, library.default_joint_pos)
        terms = reward[None].expand(len(REWARD_TERM_NAMES), -1)
        return next_state, reward, terms, reward * 0 - 2, obs, next_state[:, :7]

    rollout = imagine(wm, policy, library, start, horizon, future_frames,
                      obs_noise=True, step_fn=transition)
    assert rollout.states.shape == (horizon + 1, batch, 13 + 2 * num_joints)
    assert rollout.actions.shape == (horizon, batch, num_joints)
    assert rollout.goals.shape == (horizon + 1, batch, cfg.goal_dim)
    assert torch.isfinite(rollout.rewards).all()
    (-rollout.rewards.mean()).backward()
    gradients = [p.grad for p in policy.parameters()]
    assert all(grad is not None and torch.isfinite(grad).all() for grad in gradients)
    assert any(grad.abs().sum() > 0 for grad in gradients)

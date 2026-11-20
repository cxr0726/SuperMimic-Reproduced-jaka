"""Delayed commands must agree across real replay, WM fitting and imagination."""

from types import SimpleNamespace

import torch

from supermimic.data.replay_buffer import ReplayBuffer, StartBatch
from supermimic.envs.mdp.motion_library import RefBodies, RefState
from supermimic.imagination.compiled import REWARD_TERM_NAMES
from supermimic.imagination.rollout import imagine
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame, analytic_observation
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


def _delayed_wm():
    cfg = WorldModelCfg(
        hidden=(16,), state_spec=StateSpecCfg(context_layout="khan_dr_v4"),
        input_normalization=True, termination_mode="analytic", delay_history=True,
    )
    heads = SimpleNamespace(
        num_joints=27, default_joint_pos=torch.zeros(27),
        termination_features=AnalyticHeads.termination_features,
    )
    wm = WorldModel(cfg, heads, "cpu", action_scale=torch.full((27,), .5))
    with torch.no_grad():
        # The standard zero-last initializer makes all action gradients zero.
        wm.dynamics.mlp.net[-1].weight.normal_(std=.025)
    return wm


def _states(batch, num_joints=27):
    state = torch.zeros(batch, 13 + 2 * num_joints)
    state[:, 2] = .7
    state[:, 3] = 1
    return state


def test_only_physically_active_commands_affect_state_and_receive_gradients():
    torch.manual_seed(17)
    wm = _delayed_wm()
    state = _states(6)
    xi = wm.spec.xi_offset.repeat(6, 1)
    xi[:, wm.delay_xi_index] = torch.tensor([0., 1., 2., 3., 4., 4.])
    current = torch.randn(6, 27, requires_grad=True)
    previous = torch.randn(6, 27, requires_grad=True)
    reset = torch.tensor([False, False, False, False, False, True])

    result = wm.step(state, current, xi, previous, reset)
    current_grad, previous_grad = torch.autograd.grad(result.sum(), (current, previous))
    assert (current_grad.abs().sum(-1) > 0).tolist() == [True, True, True, True, False, True]
    assert (previous_grad.abs().sum(-1) > 0).tolist() == [False, True, True, True, True, False]

    # A full-period delay applies the previous command throughout, while a reset
    # bypasses it. Both are equivalent to constant zero-lag commands.
    indices = torch.tensor([4, 5])
    constant = current.clone()
    constant[4] = previous[4]
    zero_lag = xi.clone()
    zero_lag[indices, wm.delay_xi_index] = 0
    expected = wm.step(state, constant, zero_lag, previous, reset)
    torch.testing.assert_close(result[indices], expected[indices], rtol=0, atol=0)


def test_stepwise_loss_and_online_error_advance_delay_history_after_reset():
    torch.manual_seed(19)
    wm = _delayed_wm()
    initial = _states(2)
    xi = wm.spec.xi_offset.repeat(2, 1)
    xi[:, wm.delay_xi_index] = 4
    reset = torch.tensor([False, True])
    continuing = torch.zeros_like(reset)
    previous = torch.randn(2, 27)
    actions = torch.randn(3, 2, 27)
    with torch.no_grad():
        first = wm.step(initial, actions[0], xi, previous, reset)
        second = wm.step(first, actions[1], xi, actions[0], continuing)
        third = wm.step(second, actions[2], xi, actions[1], continuing)
    states = torch.stack((initial, first, second, third))
    refs = [ReferenceFrame(None, None, None, None, s[:, :3], s[:, 3:7]) for s in states[1:]]
    done = torch.zeros(3, 2)
    loss, _ = wm.window_loss(states, actions, xi, done, refs, previous, reset)
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-10, rtol=0)
    errors = wm.rollout_error(states, actions, xi, previous, reset)
    torch.testing.assert_close(errors, torch.zeros_like(errors), atol=1e-10, rtol=0)

    wrong_previous = previous + 1
    wrong_loss, _ = wm.window_loss(states, actions, xi, done, refs, wrong_previous, reset)
    assert wrong_loss > 1e-8
    assert (wm.rollout_error(states, actions, xi, wrong_previous, reset) > 0).all()


def test_delayed_replay_cannot_wrap_to_future_or_cross_reset_for_previous_command():
    torch.manual_seed(11)
    replay = ReplayBuffer(1, 5, 1, 1, 1, "cpu", delay_history=True)
    env = torch.tensor([0])

    def add(t, first_step):
        value = torch.tensor([[float(t)]])
        replay.add(env, value, -value, value, value, value[0], torch.zeros(1),
                   torch.tensor([t]), env, torch.tensor([False]), torch.tensor([first_step]))

    for t in range(5):
        add(t, first_step=t == 0)
    assert replay.window_mask(1)[0, 0]  # Reset can start a window without a predecessor.
    add(5, first_step=False)
    assert not replay.window_mask(1)[0, 1]  # Oldest live row cannot use the newest predecessor.
    windows = replay.sample_windows(64, 1)
    starts = replay.sample_starts(64, 1)
    assert not windows.first_step.any() and not starts.first_step.any()
    torch.testing.assert_close(windows.prev_exec[:, 0], windows.states[0, :, 0] - 1)
    torch.testing.assert_close(starts.prev_exec[:, 0], starts.state[:, 0] - 1)
    torch.testing.assert_close(starts.prev_action, -starts.prev_exec)

    replay.end_episode(env)
    add(6, first_step=True)
    add(7, first_step=False)
    windows = replay.sample_windows(64, 1)
    first = windows.frame[0] == 6
    assert first.any()
    assert torch.equal(windows.first_step, first)
    assert not windows.prev_exec[first].count_nonzero()


def test_imagination_passes_replayed_and_new_commands_without_detaching_them():
    torch.manual_seed(21)
    q0 = torch.zeros(1)
    policy = Policy(PolicyCfg(
        encoder_hidden=(8,), decoder_hidden=(8,), latent_dim=4,
        history=1, future_frames=1, num_joints=1, goal_dim=12,
    ))

    def ref(frame):
        s = _states(frame.numel(), num_joints=1)
        return RefState(s[:, :3], s[:, 3:7], s[:, 7:8], s[:, 8:11], s[:, 11:14], s[:, 14:15])

    def bodies(frame):
        r = ref(frame)
        return RefBodies(r.root_pos_w[:, None], r.root_quat_w[:, None],
                         r.root_lin_vel_w[:, None], r.root_ang_vel_w[:, None])

    library = SimpleNamespace(
        default_joint_pos=q0, last_frame_of=lambda frame: torch.full_like(frame, 8),
        goal_window=lambda frame, future: torch.zeros(frame.numel(), 12 * future),
        ref=ref, bodies=bodies,
    )
    wm = SimpleNamespace(heads=SimpleNamespace(
        anchor_index=0, root_error_features=AnalyticHeads.root_error_features,
    ))
    s = _states(2, num_joints=1)
    z = torch.zeros(2, 1)
    xi = torch.zeros(2, 5)
    previous = torch.tensor([[.25], [-.25]])
    first = torch.tensor([False, True])
    start = StartBatch(s, xi, torch.zeros(2, dtype=torch.long), z, s[None], z[None],
                       prev_exec=previous, first_step=first)
    seen = []

    def step(state, action, a_prev, context, target, prev_exec, first_step):
        seen.append((action, a_prev, prev_exec, first_step))
        # Emulate a full held control period, with the first post-reset step immediate.
        applied = torch.where(first_step[:, None], action, prev_exec)
        next_state = state + torch.nn.functional.pad(applied, (7, 7))
        reward = next_state[:, 7]
        obs = analytic_observation(next_state, action, context, q0)
        terms = reward[None].expand(len(REWARD_TERM_NAMES), -1)
        return next_state, reward, terms, reward * 0 - 2, obs, next_state[:, :7]

    rollout = imagine(wm, policy, library, start, horizon=2, obs_noise=False, step_fn=step)
    assert seen[0][2] is previous
    assert seen[0][3] is first
    assert seen[1][2] is seen[0][0]
    assert not seen[1][3].any()
    assert seen[0][1] is z
    gradient, = torch.autograd.grad(rollout.rewards[1, 0], seen[0][0])
    torch.testing.assert_close(gradient, torch.tensor([[1.], [0.]]))

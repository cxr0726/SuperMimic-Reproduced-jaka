"""Reference-horizon eligibility is independent of collection episode length."""

from types import SimpleNamespace

import pytest
import torch

from supermimic.data.replay_buffer import ReplayBuffer, StartBatch
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.imagination.compiled import REWARD_TERM_NAMES
from supermimic.imagination.rollout import imagine
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.world_model.heads import analytic_observation


def _buffer(capacity=16):
    return ReplayBuffer(1, capacity, 1, 1, 1, "cpu")


def _add(buffer, frame, *, clip=0, terminal=False, ep_start=False):
    buffer.add(
        torch.tensor([0]), torch.tensor([[float(frame)]]),
        torch.tensor([[100.0 + frame]]), torch.tensor([[100.0 + frame]]),
        torch.zeros(1, 1), torch.zeros(1), torch.zeros(1),
        torch.tensor([frame]), torch.tensor([clip]),
        torch.tensor([terminal]), torch.tensor([ep_start]),
    )


def test_full_horizon_includes_equality_and_never_draws_short_tails():
    buffer = _buffer()
    for frame in (6, 7, 8):
        _add(buffer, frame, ep_start=frame == 6)
    ends = torch.tensor([10])
    starts = buffer.sample_starts(1024, 2, horizon=3, clip_last=ends)
    assert set(starts.frame.tolist()) == {6, 7}
    assert bool((starts.frame + 3 <= 10).all())
    # Existing consumers can still request starts without a reference horizon.
    assert 8 in buffer.sample_starts(1024, 2).frame.tolist()


def test_each_clip_uses_its_own_inclusive_end_in_global_frame_coordinates():
    buffer = _buffer()
    for frame in (7, 8, 17, 18):
        _add(buffer, frame, clip=frame // 10, ep_start=frame in (7, 17))
        if frame == 8:
            buffer.end_episode(torch.tensor([0]))
    ends = torch.tensor([9, 19])
    starts = buffer.sample_starts(1024, 2, horizon=2, clip_last=ends)
    assert set(starts.frame.tolist()) == {7, 17}
    assert bool((starts.frame + 2 <= ends[starts.frame // 10]).all())


def test_horizon_filter_preserves_history_across_the_ring_storage_boundary():
    buffer = _buffer(capacity=8)
    for frame in range(12):
        _add(buffer, frame, ep_start=frame == 0)
    # Global frame 8 is stored at ring slot zero, and its history is in slots
    # 6, 7, 0. Older reference-eligible slots lack the retained ring history.
    starts = buffer.sample_starts(
        8, 3, horizon=3, clip_last=torch.tensor([11]),
    )
    assert starts.frame.tolist() == [8] * 8
    assert starts.hist_states[:, 0, 0].tolist() == [6.0, 7.0, 8.0]
    assert starts.hist_actions[:, 0, 0].tolist() == [105.0, 106.0, 107.0]
    assert bool(starts.history_valid.all())


def test_no_eligible_reference_start_fails_instead_of_rejection_loop_or_padding():
    buffer = _buffer()
    for frame in (8, 9):
        _add(buffer, frame, ep_start=frame == 8)
    with pytest.raises(RuntimeError, match="full reference horizon"):
        buffer.sample_starts(4, 2, horizon=3, clip_last=torch.tensor([10]))


def test_collection_timeout_does_not_require_future_logged_transitions():
    buffer = _buffer()
    _add(buffer, 0, ep_start=True)
    # Terminal replay rows also represent normal collection timeouts; their
    # failure label is zero. The reference itself still has nine transitions.
    _add(buffer, 1, terminal=True)
    buffer.end_episode(torch.tensor([0]))
    assert not bool(buffer.window_mask(5).any())
    starts = buffer.sample_starts(
        4, 3, horizon=5, clip_last=torch.tensor([9]),
    )
    assert starts.frame.tolist() == [0] * 4
    assert starts.hist_states[:, 0, 0].tolist() == [0.0, 0.0, 0.0]
    assert starts.hist_actions[:, 0, 0].tolist() == [0.0, 0.0, 0.0]
    assert starts.history_valid[:, 0].tolist() == [False, False, True]


@pytest.mark.parametrize("kwargs", [
    {"horizon": 3},
    {"clip_last": torch.tensor([5])},
    {"horizon": 0, "clip_last": torch.tensor([5])},
    {"horizon": 3, "clip_last": torch.empty(0, dtype=torch.long)},
    {"horizon": 3, "clip_last": torch.ones(1, 1, dtype=torch.long)},
])
def test_invalid_reference_horizon_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        _buffer().sample_starts(1, 2, **kwargs)


def _tiny_library():
    # Use production library accessors so the independent future-goal clamp is
    # tested even when the rollout ends exactly on the real last frame.
    library = MotionLibrary.__new__(MotionLibrary)
    library.device, library.dtype, library.compact = torch.device("cpu"), torch.float32, False
    library.num_joints, library.goal_feat_dim = 1, 12
    library.default_joint_pos = torch.zeros(1)
    library._ref = torch.zeros(4, 15)
    library._ref[:, 3] = 1
    library.frame_clip = torch.zeros(4, dtype=torch.long)
    library.clip_last = torch.tensor([3])
    library.body_pos_w = library._ref[:, None, :3]
    library.body_quat_w = library._ref[:, None, 3:7]
    library.body_lin_vel_w = torch.zeros(4, 1, 3)
    library.body_ang_vel_w = torch.zeros(4, 1, 3)
    library.goal_feat = torch.arange(4, dtype=torch.float32)[:, None].expand(4, 12)
    return library


def _start(frame):
    state = torch.zeros(1, 15)
    state[:, 3] = 1
    return StartBatch(
        state, torch.zeros(1, 5), torch.tensor([frame]), torch.zeros(1, 1),
        state.repeat(2, 1, 1), torch.zeros(2, 1, 1),
    )


def test_imagination_rechecks_horizon_before_running_policy_or_world_model():
    with pytest.raises(ValueError, match="full reference horizon"):
        imagine(None, None, _tiny_library(), _start(2), horizon=2)
    with pytest.raises(ValueError, match="positive"):
        imagine(None, None, _tiny_library(), _start(0), horizon=0)


def test_imagination_reaches_last_frame_and_keeps_future_goal_padding():
    library = _tiny_library()
    policy = Policy(PolicyCfg(
        encoder_hidden=(8,), decoder_hidden=(8,), latent_dim=4,
        num_joints=1, history=2, future_frames=4, goal_dim=48,
    ))
    heads = SimpleNamespace(
        anchor_index=0,
        root_error_features=lambda state, ref: torch.zeros(len(state), 4),
    )

    def transition(state, action, previous, context, ref):
        reward = -action.square().sum(-1)
        obs = analytic_observation(state, action, context, library.default_joint_pos)
        terms = reward[None].expand(len(REWARD_TERM_NAMES), -1)
        return state, reward, terms, reward * 0 - 20, obs, state[:, :7]

    rollout = imagine(
        SimpleNamespace(heads=heads), policy, library, _start(1), horizon=2,
        future_frames=4, obs_noise=False, step_fn=transition,
    )
    assert rollout.frames[:, 0].tolist() == [1, 2, 3]
    assert rollout.goals[:, 0].reshape(3, 4, 12)[:, :, 0].tolist() == [
        [1.0, 2.0, 3.0, 3.0], [2.0, 3.0, 3.0, 3.0], [3.0, 3.0, 3.0, 3.0],
    ]

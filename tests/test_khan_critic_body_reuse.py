"""Reusing reward FK for the critic preserves frame alignment and actor BPTT."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from supermimic.data.replay_buffer import StartBatch
from supermimic.imagination.compiled import ImagStep
from supermimic.imagination.objective import policy_objective
from supermimic.imagination.policy_trainer import PolicyTrainer, PolicyTrainerCfg
from supermimic.imagination.rollout import imagine
from supermimic.world_model.wm_trainer import reference_at

# Share the existing small, real Khan FK fixture without assuming that tests
# is an importable package (some Python installations contain another tests).
_fixture_spec = importlib.util.spec_from_file_location(
    "khan_critic_reuse_fixture", Path(__file__).with_name("test_khan_critic_errors.py"),
)
_fixture_module = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(_fixture_module)
_training_components = _fixture_module._training_components


class _CountFK:
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.calls = 0
        self.samples = 0

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def __call__(self, qpos, qvel):
        self.calls += 1
        self.samples += qpos.numel() // qpos.shape[-1]
        return self.wrapped(qpos, qvel)


def _start(wm, library):
    frame = torch.tensor([0, 7])
    state = library._ref[frame].clone()
    # Nonzero tracking errors exercise reference reanchoring and value gradients.
    state[:, 2] += 0.02
    state[:, 7:34] += 0.03
    return StartBatch(
        state, wm.spec.xi_offset.repeat(2, 1), frame, torch.zeros(2, 27),
        state.repeat(5, 1, 1), torch.zeros(5, 2, 27),
    )


def _gradients(value, policy, *, retain_graph=False):
    return torch.autograd.grad(
        value, tuple(policy.parameters()), allow_unused=True, retain_graph=retain_graph,
    )


def _evaluate(*, cache, checkpoint):
    torch.manual_seed(333)
    wm, policy, critic, library = _training_components()
    wm.requires_grad_(False)
    critic.requires_grad_(False)
    trainer = PolicyTrainer(wm, policy, critic, library, PolicyTrainerCfg(
        horizon=3, batch_size=2, history=5, future_frames=5,
        observation_layout="isaac_khan", compile=None, activation_checkpoint=checkpoint,
    ))
    step = ImagStep(wm, library.default_joint_pos, observation_layout="isaac_khan",
                    return_body_poses=cache)
    counter = _CountFK(wm.heads.fk)
    wm.heads.fk = counter
    start = _start(wm, library)
    torch.manual_seed(415)
    rollout = imagine(wm, policy, library, start, horizon=3, future_frames=5,
                      step_fn=step, obs_noise=True, activation_checkpoint=checkpoint)
    # Both paths need just the actor's initial FK and one reward FK per step.
    assert (counter.calls, counter.samples) == (4, 8)
    feat, goals, root_error, errors = trainer._critic_inputs(rollout)
    assert errors.shape == (4, 2, 78)
    # The old path reruns FK for the 3-frame prefix and the bootstrap frame.
    assert (counter.calls, counter.samples) == ((4, 8) if cache else (6, 16))
    if cache:
        assert rollout.critic_body_pos.shape == (4, 2, 15, 3)
        assert rollout.critic_body_quat.shape == (4, 2, 15, 4)
        prefix_geometry = (rollout.critic_body_pos[:-1].sum()
                           + rollout.critic_body_quat[:-1].sum())
        prefix_gradients = _gradients(prefix_geometry, policy, retain_graph=True)
        assert all(g is None or torch.count_nonzero(g) == 0 for g in prefix_gradients)
    else:
        assert rollout.critic_body_pos is rollout.critic_body_quat is None

    value = critic(feat[-1], goals[-1], wm.spec.normalize_xi(start.xi),
                   root_error[-1], errors[-1])
    bootstrap_gradients = _gradients(value.sum(), policy, retain_graph=True)
    assert any(g is not None and g.abs().sum() > 0 for g in bootstrap_gradients)
    objective = policy_objective(rollout, value, 0.99)
    gradients = _gradients(objective, policy)
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    return {
        "states": rollout.states.detach(), "actions": rollout.actions.detach(),
        "rewards": rollout.rewards.detach(), "survival": rollout.survival.detach(),
        "goals": goals.detach(), "errors": errors.detach(), "value": value.detach(),
        "objective": objective.detach(), "bootstrap_gradients": bootstrap_gradients,
        "gradients": gradients, "rng": torch.get_rng_state(),
    }


@pytest.mark.parametrize("checkpoint", [False, True])
def test_reused_body_poses_match_original_fk_values_and_policy_gradients(checkpoint):
    original = _evaluate(cache=False, checkpoint=checkpoint)
    reused = _evaluate(cache=True, checkpoint=checkpoint)
    for name in original:
        torch.testing.assert_close(original[name], reused[name], rtol=2e-6, atol=1e-8, msg=name)


def test_reused_body_poses_preserve_activation_checkpoint_results_and_rng():
    eager = _evaluate(cache=True, checkpoint=False)
    checkpointed = _evaluate(cache=True, checkpoint=True)
    for name in eager:
        torch.testing.assert_close(eager[name], checkpointed[name], rtol=2e-6, atol=1e-8,
                                   msg=name)


def test_reward_uses_previous_reference_and_cached_critic_uses_current_reference():
    torch.manual_seed(334)
    wm, policy, critic, library = _training_components()
    trainer = PolicyTrainer(wm, policy, critic, library, PolicyTrainerCfg(
        horizon=3, history=5, future_frames=5, observation_layout="isaac_khan", compile=None,
    ))
    start = _start(wm, library)
    # Omitting step_fn also opts the native Khan rollout into FK reuse.
    rollout = imagine(wm, policy, library, start, horizon=3, future_frames=5, obs_noise=False)
    _, _, _, errors = trainer._critic_inputs(rollout)
    assert rollout.critic_body_pos is not None
    previous_action = start.prev_action
    wrong_reward_differences = []
    for time in range(3):
        old_ref = reference_at(library, rollout.frames[time])
        current_ref = reference_at(library, rollout.frames[time + 1])
        next_state, action = rollout.states[time + 1], rollout.actions[time]
        with torch.no_grad():
            expected_reward = wm.heads.reward(next_state, action, previous_action, old_ref)["total"]
            wrong_reward = wm.heads.reward(next_state, action, previous_action, current_ref)["total"]
            expected_errors = wm.heads.critic_error_features(next_state, current_ref)
            wrong_errors = wm.heads.critic_error_features(next_state, old_ref)
        torch.testing.assert_close(rollout.rewards[time], expected_reward, rtol=0, atol=1e-7)
        torch.testing.assert_close(errors[time + 1], expected_errors, rtol=0, atol=1e-7)
        assert (expected_errors - wrong_errors).abs().max() > 1e-4
        wrong_reward_differences.append((expected_reward - wrong_reward).abs().max())
        previous_action = action
    assert torch.stack(wrong_reward_differences).max() > 1e-7


def test_body_pose_outputs_are_opt_in_and_legacy_custom_steps_keep_fk_fallback():
    torch.manual_seed(335)
    wm, policy, critic, library = _training_components()
    start = _start(wm, library)
    ref = reference_at(library, start.frame)
    step = ImagStep(wm, library.default_joint_pos, observation_layout="isaac_khan")
    cached_step = ImagStep(wm, library.default_joint_pos, observation_layout="isaac_khan",
                           return_body_poses=True)
    original = step(start.state, start.prev_action, start.prev_action, start.xi, ref)
    cached = cached_step(start.state, start.prev_action, start.prev_action, start.xi, ref)
    assert len(original) == 6 and len(cached) == 8
    for before, after in zip(original, cached[:6], strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    expected_bodies = wm.heads.robot_bodies(cached[0])
    torch.testing.assert_close(cached[6], expected_bodies["pos"], rtol=0, atol=0)
    torch.testing.assert_close(cached[7], expected_bodies["quat"], rtol=0, atol=0)

    # A callable without ImagStep's optional-output flag remains a six-output
    # custom transition. Its two-argument synthetic error head must still work.
    def custom_step(*args):
        return step(*args)

    custom_step.term_names = step.term_names
    original_head = wm.heads.critic_error_features
    wm.heads.critic_error_features = lambda state, reference: original_head(state, reference)
    rollout = imagine(wm, policy, library, start, horizon=2, future_frames=5,
                      obs_noise=False, step_fn=custom_step)
    trainer = PolicyTrainer(wm, policy, critic, library, PolicyTrainerCfg(
        observation_layout="isaac_khan", compile=None,
    ))
    assert rollout.critic_body_pos is rollout.critic_body_quat is None
    assert trainer._critic_inputs(rollout)[-1].shape == (3, 2, 78)


def test_supermimic_rollout_can_ignore_explicitly_requested_body_pose_outputs():
    wm, _, _, library = _training_components()
    start = _start(wm, library)

    def policy(obs, goal):
        return torch.zeros(len(obs), 27)

    policy.cfg = SimpleNamespace(observation_layout="supermimic", future_frames=5)
    library.goal_window = lambda frame, k: torch.zeros(len(frame), k * 64)
    step = ImagStep(wm, library.default_joint_pos, return_body_poses=True)
    rollout = imagine(wm, policy, library, start, horizon=2, future_frames=5,
                      obs_noise=False, step_fn=step)
    assert rollout.states.shape == (3, 2, 67)
    assert rollout.critic_body_pos is rollout.critic_body_quat is None


@pytest.mark.parametrize("layout,use_critic,expected", [
    ("isaac_khan", True, True), ("isaac_khan", False, False),
    ("supermimic", True, False), ("supermimic", False, False),
])
def test_trainer_enables_body_reuse_only_for_khan_critic(layout, use_critic, expected):
    wm, policy, critic, library = _training_components()
    trainer = PolicyTrainer(wm, policy, critic, library, PolicyTrainerCfg(
        observation_layout=layout, use_critic=use_critic, compile=None,
    ))
    assert trainer.step_fn.return_body_poses is expected

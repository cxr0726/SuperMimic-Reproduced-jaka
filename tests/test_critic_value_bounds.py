"""Official critic upper-bound semantics through the real optimizer update."""

from types import SimpleNamespace

import pytest
import torch

from supermimic.imagination import compiled
from supermimic.imagination import policy_trainer as trainer_module
from supermimic.imagination.objective import gae_targets, policy_objective
from supermimic.imagination.policy_trainer import (
    PolicyTrainer,
    PolicyTrainerCfg,
    critic_bound_diagnostics,
    critic_value_ceiling,
)
from supermimic.imagination.rollout import ImagRollout


@pytest.mark.parametrize("step_reward, expected", [(.1, 10.), (.112, 11.2)])
def test_ceiling_uses_the_actual_robot_reward_ceiling(step_reward, expected):
    assert critic_value_ceiling(step_reward, .99) == pytest.approx(expected)


@pytest.mark.parametrize("reward, gamma", [(0, .99), (-1, .99), (float("inf"), .99),
                                           (.112, 1), (.112, -1), (.112, float("nan"))])
def test_invalid_infinite_horizon_bound_is_rejected(reward, gamma):
    with pytest.raises(ValueError):
        critic_value_ceiling(reward, gamma)


def test_upper_clamp_preserves_negative_values_and_only_saturates_excess_gradient():
    raw = torch.tensor([-100., 0., 10., 12.], requires_grad=True)
    bounded = raw.clamp(max=11.2)
    torch.testing.assert_close(bounded, torch.tensor([-100., 0., 10., 11.2]))
    bounded.sum().backward()
    torch.testing.assert_close(raw.grad, torch.tensor([1., 1., 1., 0.]))


def test_bootstrap_cap_is_in_the_actor_graph_and_does_not_clip_reward():
    raw = torch.tensor([20., -20.], requires_grad=True)
    rewards = torch.tensor([[-3., .3], [-4., .4]], requires_grad=True)
    ro = SimpleNamespace(rewards=rewards, survival=torch.ones(3, 2))
    objective = policy_objective(ro, raw.clamp(max=11.2), gamma=.99)
    expected = ((rewards[0] + .99 * rewards[1]) + .99**2 * torch.tensor([11.2, -20.])).mean()
    torch.testing.assert_close(objective, expected)
    objective.backward()
    torch.testing.assert_close(raw.grad, torch.tensor([0., .99**2 / 2]))
    torch.testing.assert_close(rewards.grad, torch.tensor([[.5, .5], [.495, .495]]))


def test_gae_targets_are_detached_and_only_bounded_after_the_recursion():
    rewards = torch.tensor([[.112, -2.], [.112, -3.]], requires_grad=True)
    values = torch.tensor([[20., -20.], [20., -20.], [20., -20.]], requires_grad=True)
    raw = gae_targets(rewards, torch.zeros(2, 2), values, .99, .95)
    target = raw.clamp(max=11.2)
    assert not raw.requires_grad and not target.requires_grad
    assert (target[:, 0] == 11.2).all()
    torch.testing.assert_close(target[:, 1], raw[:, 1])
    assert (target[:, 1] < -20).all()
    assert values.grad is None and rewards.grad is None


def test_diagnostics_measure_the_unbounded_distribution_without_gradients():
    values = torch.arange(100., requires_grad=True)
    result = critic_bound_diagnostics(50., value=values, target=-values)
    assert result == {
        "critic_value_raw_exceed_fraction": pytest.approx(.49),
        "critic_value_raw_max": 99., "critic_value_raw_p99": 98.,
        "critic_target_raw_exceed_fraction": 0.,
        "critic_target_raw_max": 0., "critic_target_raw_p99": -1.,
    }
    assert values.grad is None
    assert critic_bound_diagnostics(50., value=torch.empty(0)) == {}


class _Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.action = torch.nn.Parameter(torch.tensor(0.))

    def forward(self, observation):
        return self.action.expand(len(observation), 1)


class _Critic(torch.nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(float(offset)))

    def forward(self, state, goal, xi, drift, tracking_errors=None):
        return self.offset + state[..., 0]


def _trainer(monkeypatch, enabled, offset=20.):
    monkeypatch.setattr(compiled, "ImagStep", lambda *args, **kwargs: object())
    wm = torch.nn.Linear(2, 2)
    wm.spec = SimpleNamespace(normalize_xi=lambda xi: xi, canonical=lambda state: state)
    wm.heads = SimpleNamespace(reward_cfg=SimpleNamespace(max_step_reward=.112))
    lib = SimpleNamespace(default_joint_pos=torch.zeros(1), clip_last=torch.tensor([100]))
    cfg = PolicyTrainerCfg(
        horizon=2, batch_size=4, history=1, future_frames=1, compile=None,
        obs_noise=False, ema_decay=.75, critic_value_bound=enabled,
        critic_epochs=1, critic_minibatches=2, lr=.001,
    )
    trainer = PolicyTrainer(wm, _Actor(), _Critic(offset), lib, cfg)

    def imagine(wm, policy, library, start, horizon, future_frames, **kwargs):
        action = policy(torch.ones(4, 2))
        states = torch.stack([torch.cat([action, action * 0], -1)] * 3)
        return ImagRollout(
            states=states, actions=action[None].expand(2, -1, -1),
            rewards=(.112 - action[:, 0].square())[None].expand(2, -1),
            dhat=torch.zeros(3, 4), survival=torch.ones(3, 4),
            goals=torch.zeros(3, 4, 1), e_root=torch.zeros(3, 4, 4),
            frames=torch.zeros(3, 4, dtype=torch.long), reward_terms={},
        )

    monkeypatch.setattr(trainer_module, "imagine", imagine)
    buffer = SimpleNamespace(sample_starts=lambda *args, **kwargs:
                             SimpleNamespace(xi=torch.zeros(4, 1)))
    return trainer, buffer


@pytest.mark.parametrize("enabled", [False, True])
def test_real_actor_and_critic_steps_apply_both_bounds_and_record_raw_diagnostics(monkeypatch, enabled):
    trainer, buffer = _trainer(monkeypatch, enabled)
    fitted_targets = []
    mse_loss = torch.nn.functional.mse_loss

    def inspect_mse(pred, target, **kwargs):
        fitted_targets.append(target.clone())
        assert not target.requires_grad
        return mse_loss(pred, target, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "mse_loss", inspect_mse)
    metrics = trainer.update(buffer)
    assert trainer.value_ceiling == pytest.approx(11.2)
    assert metrics["critic_value_bound_enabled"] == float(enabled)
    for group in ("bootstrap", "value", "target"):
        assert metrics[f"critic_{group}_raw_exceed_fraction"] == 1.
        assert metrics[f"critic_{group}_raw_max"] > 11.2
        assert metrics[f"critic_{group}_raw_p99"] > 11.2
    assert len(fitted_targets) == 2
    if enabled:
        assert trainer.policy.action.item() == 0.0  # Saturated bootstrap has zero actor gradient.
        assert all((target <= 11.2).all() for target in fitted_targets)
        assert metrics["critic_target_mean"] == pytest.approx(11.2)
    else:
        assert trainer.policy.action.item() > 0.0
        assert all((target > 11.2).all() for target in fitted_targets)
        assert metrics["critic_target_mean"] > 11.2
    assert trainer.critic.offset.item() < 20.0  # Raw critic remains trainable above the ceiling.
    assert trainer.ema.num_updates == 1
    assert all(torch.isfinite(parameter).all() for parameter in trainer.policy.parameters())


def test_negative_critic_values_remain_unclipped_in_real_update(monkeypatch):
    trainer, buffer = _trainer(monkeypatch, enabled=True, offset=-20.)
    metrics = trainer.update(buffer)
    assert metrics["critic_bootstrap_raw_exceed_fraction"] == 0.
    assert metrics["critic_value_raw_exceed_fraction"] == 0.
    assert metrics["critic_target_raw_exceed_fraction"] == 0.
    assert metrics["critic_target_mean"] < -10.
    assert trainer.policy.action.item() > 0.


def test_enabled_bounds_reject_missing_reward_metadata(monkeypatch):
    monkeypatch.setattr(compiled, "ImagStep", lambda *args, **kwargs: object())
    with pytest.raises(ValueError, match="reward ceiling"):
        PolicyTrainer(torch.nn.Linear(2, 2), _Actor(), _Critic(20.),
                      SimpleNamespace(default_joint_pos=torch.zeros(1)),
                      PolicyTrainerCfg(critic_value_bound=True))

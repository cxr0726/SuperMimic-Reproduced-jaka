"""CPU contracts for the averaged actor and its optimizer/checkpoint lifecycle."""

import copy
import io
from types import SimpleNamespace

import pytest
import torch

from supermimic.config import TrainConfig, resolve_training_observations
from supermimic.imagination import policy_trainer as trainer_module
from supermimic.imagination.policy_trainer import PolicyTrainer, PolicyTrainerCfg
from supermimic.imagination.rollout import ImagRollout
from supermimic.policy.ema import ActorEMA
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.robots.registry import get_robot_spec
from supermimic.training.loop import Trainer
from supermimic.world_model.world_model import WorldModelCfg


def _policy():
    cfg = PolicyCfg(
        encoder_hidden=(8,), decoder_hidden=(8,), latent_dim=4,
        history=5, future_frames=5, num_joints=27, goal_dim=155,
        observation_layout="isaac_khan", action_mode="reference_residual",
        goal_noise_z=.01, goal_noise_joint_pos=.05,
    )
    actor = Policy(
        cfg, default_joint_pos=torch.linspace(-.2, .3, 27),
        action_scale=torch.linspace(.2, .6, 27),
    )
    actor.register_buffer("update_counter", torch.tensor(0, dtype=torch.int64))
    return actor


@torch.no_grad()
def _set_parameters(actor, value):
    for parameter in actor.parameters():
        parameter.fill_(value)


def _checkpoint(ema):
    # Exercise a real serialization boundary: state_dict values otherwise alias
    # the shadow tensors and could disguise an incomplete checkpoint restore.
    stream = io.BytesIO()
    torch.save({
        "policy_ema": ema.state_dict(),
        "policy_ema_updates": ema.num_updates,
        "policy_ema_decay": ema.decay,
    }, stream)
    stream.seek(0)
    return torch.load(stream, weights_only=False)


def test_initial_shadow_is_independent_frozen_and_preserves_rng():
    actor = _policy().train()
    rng = torch.random.get_rng_state().clone()
    ema = ActorEMA(actor, decay=.75)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert actor.training and not ema.module.training
    assert ema.num_updates == 0
    assert all(parameter.requires_grad for parameter in actor.parameters())
    assert all(not parameter.requires_grad for parameter in ema.module.parameters())
    for name, value in actor.state_dict().items():
        shadow = ema.state_dict()[name]
        torch.testing.assert_close(shadow, value, rtol=0, atol=0)
        assert shadow.data_ptr() != value.data_ptr()
    observation, goal = torch.randn(2, 465), torch.randn(2, 155)
    torch.testing.assert_close(
        ema.module(observation, goal), actor(observation, goal), rtol=0, atol=0,
    )
    assert not ema.module(observation, goal).requires_grad


def test_first_actor_update_copies_then_parameters_smooth_and_all_buffers_copy():
    actor = _policy()
    _set_parameters(actor, 0)
    ema = ActorEMA(actor, decay=.75)
    _set_parameters(actor, 2)
    ema.update(actor)
    assert ema.num_updates == 1
    for parameter in ema.module.parameters():
        torch.testing.assert_close(parameter, torch.full_like(parameter, 2), rtol=0, atol=0)

    _set_parameters(actor, 10)
    with torch.no_grad():
        # Include physical constants, scaling, integer and nonpersistent noise
        # buffers: none are a learned quantity that may be averaged.
        for index, buffer in enumerate(actor.buffers()):
            buffer.fill_(index + 3)
    ema.update(actor)
    assert ema.num_updates == 2
    for parameter in ema.module.parameters():
        torch.testing.assert_close(parameter, torch.full_like(parameter, 4), rtol=0, atol=0)
        assert parameter.grad is None and not parameter.requires_grad
    for name, buffer in actor.named_buffers():
        shadow = dict(ema.module.named_buffers())[name]
        torch.testing.assert_close(shadow, buffer, rtol=0, atol=0)
        assert shadow.data_ptr() != buffer.data_ptr()
    assert not ema.module.training


def test_checkpoint_roundtrip_preserves_next_averaging_update():
    actor = _policy()
    ema = ActorEMA(actor, decay=.75)
    for value in (2, 10):
        _set_parameters(actor, value)
        ema.update(actor)
    saved = _checkpoint(ema)
    restored = ActorEMA(copy.deepcopy(actor), decay=.75)
    restored.restore(actor, saved)
    assert restored.num_updates == ema.num_updates == 2
    torch.testing.assert_close(restored.state_dict(), ema.state_dict(), rtol=0, atol=0)

    _set_parameters(actor, -4)
    actor.update_counter.fill_(9)
    ema.update(actor)
    restored.update(actor)
    assert restored.num_updates == ema.num_updates == 3
    torch.testing.assert_close(restored.state_dict(), ema.state_dict(), rtol=0, atol=0)
    for parameter in restored.module.parameters():
        torch.testing.assert_close(parameter, torch.full_like(parameter, 2), rtol=0, atol=0)


@pytest.mark.parametrize("checkpoint", [{}, {"policy_ema": None}])
def test_legacy_restore_initializes_from_the_already_restored_live_actor(checkpoint):
    actor = _policy()
    ema = ActorEMA(actor, decay=.75)
    _set_parameters(actor, 2)
    ema.update(actor)
    # This represents the training loader restoring the live actor after the
    # PolicyTrainer (and its initial shadow) have already been constructed.
    _set_parameters(actor, 7)
    actor.update_counter.fill_(13)
    ema.restore(actor, checkpoint)
    assert ema.num_updates == 0
    torch.testing.assert_close(ema.state_dict(), actor.state_dict(), rtol=0, atol=0)
    assert not ema.module.training
    _set_parameters(actor, 11)
    ema.update(actor)
    assert ema.num_updates == 1
    torch.testing.assert_close(ema.state_dict(), actor.state_dict(), rtol=0, atol=0)


def test_restore_rejects_a_changed_decay_instead_of_reinterpreting_history():
    actor = _policy()
    original = ActorEMA(actor, decay=.75)
    original.update(actor)
    saved = _checkpoint(original)
    changed = ActorEMA(actor, decay=.5)
    before = copy.deepcopy(changed.state_dict())
    with pytest.raises(ValueError, match="(?i)decay"):
        changed.restore(actor, saved)
    assert changed.num_updates == 0
    torch.testing.assert_close(changed.state_dict(), before, rtol=0, atol=0)


@pytest.mark.parametrize("count", [-1, True, 1.5, None])
def test_restore_requires_a_valid_actor_update_count(count):
    actor = _policy()
    ema = ActorEMA(actor, decay=.75)
    ema.update(actor)
    saved = _checkpoint(ema)
    saved["policy_ema_updates"] = count
    before = copy.deepcopy(ema.state_dict())
    with pytest.raises(ValueError, match="policy_ema_updates"):
        ema.restore(actor, saved)
    assert ema.num_updates == 1
    torch.testing.assert_close(ema.state_dict(), before, rtol=0, atol=0)


@pytest.mark.parametrize("decay", [True, float("nan"), float("inf"), None])
def test_restore_rejects_invalid_decay_metadata(decay):
    actor = _policy()
    ema = ActorEMA(actor, decay=.75)
    saved = _checkpoint(ema)
    saved["policy_ema_decay"] = decay
    with pytest.raises(ValueError, match="(?i)decay"):
        ema.restore(actor, saved)


@pytest.mark.parametrize("corruption", [
    "missing_key", "shape", "dtype", "nonfinite", "reference_q_default", "reference_action_scale",
])
def test_restore_rejects_corrupt_weights_or_physical_mapping_before_mutating_shadow(corruption):
    actor = _policy()
    ema = ActorEMA(actor, decay=.75)
    ema.update(actor)
    saved = _checkpoint(ema)
    state = saved["policy_ema"]
    parameter_name = next(iter(dict(actor.named_parameters())))
    if corruption == "missing_key":
        del state[parameter_name]
    elif corruption == "shape":
        state[parameter_name] = state[parameter_name].flatten()
    elif corruption == "dtype":
        state[parameter_name] = state[parameter_name].double()
    elif corruption == "nonfinite":
        state[parameter_name].flatten()[0] = float("nan")
    else:
        state[corruption][0] += .01
    before = copy.deepcopy(ema.state_dict())
    with pytest.raises(ValueError, match="policy_ema"):
        ema.restore(actor, saved)
    assert ema.num_updates == 1
    torch.testing.assert_close(ema.state_dict(), before, rtol=0, atol=0)


class _TinyCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Linear(2, 1)

    def forward(self, features, goal, xi, root_error, tracking_errors=None):
        return self.value(features).squeeze(-1)


def _tiny_trainer(monkeypatch, decay):
    # Keep the actual PolicyTrainer and Adam steps; the synthetic rollout only
    # removes simulator/FK work irrelevant to the EMA integration contract.
    from supermimic.imagination import compiled

    monkeypatch.setattr(compiled, "ImagStep", lambda *args, **kwargs: object())
    actor = torch.nn.Linear(2, 1)
    _set_parameters(actor, 0)
    wm = torch.nn.Linear(2, 2)
    wm.spec = SimpleNamespace(normalize_xi=lambda xi: xi, canonical=lambda state: state)
    lib = SimpleNamespace(default_joint_pos=torch.zeros(2), clip_last=torch.tensor([100]))
    cfg = PolicyTrainerCfg(
        horizon=1, batch_size=2, history=1, future_frames=1,
        lr=.1, critic_epochs=3, critic_minibatches=2, compile=None,
        obs_noise=False, ema_decay=decay,
    )
    trainer = PolicyTrainer(wm, actor, _TinyCritic(), lib, cfg)

    def imagine(wm, policy, library, start, horizon, future_frames, **kwargs):
        assert policy is actor, "actor optimization must use live parameters"
        action = policy(torch.ones(2, 2))[None]
        return ImagRollout(
            states=torch.ones(2, 2, 2), actions=action,
            rewards=-(action.squeeze(-1) - 1).square(),
            dhat=torch.zeros(2, 2), survival=torch.ones(2, 2),
            goals=torch.zeros(2, 2, 1), e_root=torch.zeros(2, 2, 4),
            frames=torch.zeros(2, 2, dtype=torch.long), reward_terms={},
        )

    monkeypatch.setattr(trainer_module, "imagine", imagine)
    buffer = SimpleNamespace(sample_starts=lambda *args, **kwargs:
                             SimpleNamespace(xi=torch.zeros(2, 1)))
    return trainer, buffer


def test_ema_updates_once_immediately_after_each_actor_step_and_ignores_critic_steps(monkeypatch):
    trainer, buffer = _tiny_trainer(monkeypatch, decay=.75)
    ema = trainer.ema
    assert ema is not None
    optimizer_ids = {id(parameter) for group in trainer.opt.param_groups
                     for parameter in group["params"]}
    assert optimizer_ids == {id(parameter) for parameter in trainer.policy.parameters()}
    assert optimizer_ids.isdisjoint(id(parameter) for parameter in ema.module.parameters())
    events, after_steps = [], []
    actor_step, critic_step, ema_update = trainer.opt.step, trainer.critic_opt.step, ema.update

    def step_actor(*args, **kwargs):
        result = actor_step(*args, **kwargs)
        after_steps.append([parameter.detach().clone() for parameter in trainer.policy.parameters()])
        events.append("actor")
        return result

    def update_ema(policy):
        assert events[-1] == "actor"
        events.append("ema")
        return ema_update(policy)

    def step_critic(*args, **kwargs):
        events.append("critic")
        return critic_step(*args, **kwargs)

    monkeypatch.setattr(trainer.opt, "step", step_actor)
    monkeypatch.setattr(ema, "update", update_ema)
    monkeypatch.setattr(trainer.critic_opt, "step", step_critic)
    assert ema.num_updates == 0
    for expected_updates in (1, 2):
        trainer.update(buffer)
        assert ema.num_updates == expected_updates
    assert events == (["actor", "ema"] + ["critic"] * 6) * 2
    assert all(parameter.grad is None for parameter in ema.module.parameters())
    for first, second, averaged in zip(*after_steps, ema.module.parameters()):
        assert not torch.equal(first, second), "the test must perform real optimizer updates"
        torch.testing.assert_close(averaged, .75 * first + .25 * second, rtol=1e-6, atol=1e-8)


def test_zero_decay_disables_the_shadow_without_disabling_actor_updates(monkeypatch):
    trainer, buffer = _tiny_trainer(monkeypatch, decay=0)
    assert trainer.ema is None
    before = [parameter.detach().clone() for parameter in trainer.policy.parameters()]
    trainer.update(buffer)
    assert all(not torch.equal(old, new) for old, new in zip(before, trainer.policy.parameters()))


def _checkpoint_trainer(tmp_path):
    trainer = Trainer.__new__(Trainer)
    trainer.is_main, trainer.device, trainer.log_dir = True, "cpu", tmp_path
    trainer.iteration, trainer.start_iteration, trainer.num_updates = 400, 0, 11
    trainer.policy = _policy()
    trainer.cfg = TrainConfig(robot="khan_mini", device="cpu", policy=trainer.policy.cfg)
    # This checkpoint fixture uses toy Linear modules rather than the new
    # normalized WM. Describe their historical training semantics explicitly.
    trainer.cfg.world_model = WorldModelCfg()
    trainer.cfg.policy.running_normalization = False
    trainer.cfg.policy_trainer.critic_value_bound = False
    trainer.cfg.env.reset_sampler = "legacy"
    trainer.cfg.policy_trainer.ema_decay = .75
    resolve_training_observations(trainer.cfg)
    trainer.robot_spec = get_robot_spec(trainer.cfg.robot)
    trainer.wm, trainer.critic = [torch.nn.Linear(2, 2) for _ in range(2)]
    trainer.wm_trainer = SimpleNamespace(opt=torch.optim.Adam(trainer.wm.parameters(), lr=.003))
    trainer.policy_trainer = SimpleNamespace(
        opt=torch.optim.Adam(trainer.policy.parameters(), lr=.005, betas=(.9, .95)),
        critic_opt=torch.optim.Adam(trainer.critic.parameters(), lr=.007),
        ema=ActorEMA(trainer.policy, decay=.75),
    )
    trainer.command = SimpleNamespace(
        sampler_contract={"version": 1, "purpose": "actor-ema-checkpoint-test"},
        bin_return=torch.tensor([1., 2.]), bin_memory=torch.tensor([.4, .6]),
        sampleable_bins=torch.ones(2, dtype=torch.bool),
    )
    return trainer


def _optimizers(trainer):
    return (trainer.wm_trainer.opt, trainer.policy_trainer.opt, trainer.policy_trainer.critic_opt)


def _checkpoint_training_step(trainer):
    sensor = torch.linspace(-.1, .1, 465).repeat(2, 1)
    goal = torch.full((2, 155), .1)
    small_input = torch.tensor([[.1, .2], [.3, -.4]])
    for model, optimizer in zip(
        (trainer.wm, trainer.policy, trainer.critic), _optimizers(trainer), strict=True,
    ):
        optimizer.zero_grad(set_to_none=True)
        output = model(sensor, goal) if model is trainer.policy else model(small_input)
        output.square().mean().backward()
        optimizer.step()
        if model is trainer.policy:
            trainer.policy_trainer.ema.update(trainer.policy)
    trainer.iteration += 1
    trainer.num_updates += 1


@pytest.mark.parametrize("legacy", [False, True])
def test_trainer_checkpoint_restores_live_ema_optimizers_and_next_update(tmp_path, legacy):
    original = _checkpoint_trainer(tmp_path)
    for _ in range(2):
        _checkpoint_training_step(original)
    original.save()
    path = tmp_path / "ckpt_latest.pt"
    saved = torch.load(path, weights_only=False)
    assert saved["policy_ema_decay"] == original.cfg.policy_trainer.ema_decay == .75
    assert saved["policy_ema_updates"] == 2
    assert saved["num_updates"] == 13, "EMA counts actor updates independently of WM updates"
    torch.testing.assert_close(saved["policy"], original.policy.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(saved["policy_ema"], original.policy_trainer.ema.state_dict(),
                               rtol=0, atol=0)
    parameter_name = next(iter(dict(original.policy.named_parameters())))
    assert not torch.equal(saved["policy"][parameter_name], saved["policy_ema"][parameter_name])
    if legacy:
        for key in ("policy_ema", "policy_ema_decay", "policy_ema_updates"):
            del saved[key]
        torch.save(saved, path)

    resumed = _checkpoint_trainer(tmp_path)
    for _ in range(3):
        _checkpoint_training_step(resumed)
    for index, model in enumerate((resumed.wm, resumed.policy, resumed.critic)):
        _set_parameters(model, index + 5)
    _set_parameters(resumed.policy_trainer.ema.module, -7)
    resumed.policy_trainer.ema.num_updates = 99
    for optimizer in _optimizers(resumed):
        optimizer.param_groups[0]["lr"] = .5
    resumed.command.bin_return.fill_(-9)
    resumed.load(path)
    assert (resumed.iteration, resumed.start_iteration, resumed.num_updates) == (402, 402, 13)
    assert resumed.policy_trainer.ema.num_updates == (0 if legacy else 2)
    if legacy:
        # A historical checkpoint intentionally starts a new averaging window
        # using the restored live actor, not the constructor's random shadow.
        original.policy_trainer.ema.restore(original.policy, {})
    for old, restored in zip(
        (original.wm, original.policy, original.critic, original.policy_trainer.ema.module),
        (resumed.wm, resumed.policy, resumed.critic, resumed.policy_trainer.ema.module), strict=True,
    ):
        torch.testing.assert_close(old.state_dict(), restored.state_dict(), rtol=0, atol=0)
    for old, restored in zip(_optimizers(original), _optimizers(resumed), strict=True):
        torch.testing.assert_close(old.state_dict(), restored.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(resumed.command.bin_return, original.command.bin_return, rtol=0, atol=0)

    _checkpoint_training_step(original)
    _checkpoint_training_step(resumed)
    assert resumed.policy_trainer.ema.num_updates == original.policy_trainer.ema.num_updates
    for old, restored in zip(
        (original.wm, original.policy, original.critic, original.policy_trainer.ema.module),
        (resumed.wm, resumed.policy, resumed.critic, resumed.policy_trainer.ema.module), strict=True,
    ):
        torch.testing.assert_close(old.state_dict(), restored.state_dict(), rtol=0, atol=0)
    for old, restored in zip(_optimizers(original), _optimizers(resumed), strict=True):
        torch.testing.assert_close(old.state_dict(), restored.state_dict(), rtol=0, atol=0)


@pytest.mark.parametrize("field,value", [
    ("encoder_hidden", [16]), ("decoder_hidden", [16]), ("latent_dim", 5),
])
def test_trainer_rejects_incompatible_architecture_before_modifying_any_state(tmp_path, field, value):
    original = _checkpoint_trainer(tmp_path)
    _checkpoint_training_step(original)
    original.save()
    path = tmp_path / "ckpt_latest.pt"
    saved = torch.load(path, weights_only=False)
    saved["config"]["policy"][field] = value
    torch.save(saved, path)

    resumed = _checkpoint_trainer(tmp_path)
    _checkpoint_training_step(resumed)
    for index, model in enumerate((resumed.wm, resumed.policy, resumed.critic)):
        _set_parameters(model, index + 5)
    models = (resumed.wm, resumed.policy, resumed.critic, resumed.policy_trainer.ema.module)
    before_models = [copy.deepcopy(model.state_dict()) for model in models]
    before_optimizers = [copy.deepcopy(optimizer.state_dict()) for optimizer in _optimizers(resumed)]
    before_counts = (resumed.iteration, resumed.start_iteration, resumed.num_updates,
                     resumed.policy_trainer.ema.num_updates)
    with pytest.raises(ValueError, match="Checkpoint policy architecture differs.*start a new run"):
        resumed.load(path)
    for model, before in zip(models, before_models, strict=True):
        torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
    for optimizer, before in zip(_optimizers(resumed), before_optimizers, strict=True):
        torch.testing.assert_close(optimizer.state_dict(), before, rtol=0, atol=0)
    assert before_counts == (resumed.iteration, resumed.start_iteration, resumed.num_updates,
                             resumed.policy_trainer.ema.num_updates)

"""CPU checks for fresh defaults, collection augmentation and resume semantics."""

from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from supermimic.config import TrainConfig, resolve_training_observations
from supermimic.policy.nets import Policy
from supermimic.robots.registry import get_robot_spec
from supermimic.training.loop import Trainer, validate_policy_training_semantics


def _small_policy_config(**overrides):
    cfg = TrainConfig(robot="khan_mini")
    # Keep fixed-scale legacy coverage; individual tests also enable the fresh
    # running-normalization path without changing the checkpoint fallback.
    cfg.policy.action_mode = "reference_residual"
    cfg.policy.running_normalization = False
    cfg.policy.encoder_hidden = cfg.policy.decoder_hidden = (16,)
    cfg.policy.latent_dim = 4
    resolve_training_observations(cfg)
    return replace(cfg.policy, **overrides)


def _policy(cfg):
    robot_args = ({"default_joint_pos": torch.linspace(-0.2, 0.2, 27),
                   "action_scale": torch.full((27,), 0.5)}
                  if cfg.action_mode == "reference_residual" else {})
    return Policy(cfg, **robot_args)


@pytest.mark.parametrize("robot,mode,z_half,q_half,goal_dim", [
    ("khan_mini", "reference_residual", 0.01, 0.05, 155),
    ("g1", "default", 0.0, 0.0, 680),
])
def test_fresh_defaults_resolve_by_layout_without_changing_optimization(
    robot, mode, z_half, q_half, goal_dim,
):
    cfg = TrainConfig(robot=robot)
    original_networks = (cfg.policy.encoder_hidden, cfg.policy.decoder_hidden,
                         cfg.policy.latent_dim, cfg.policy.decoder_last_gain)
    original_optimizer = (cfg.wm_trainer.lr, cfg.wm_trainer.batch_size,
                          cfg.policy_trainer.lr, cfg.policy_trainer.batch_size,
                          cfg.policy_trainer.horizon, cfg.schedule.updates_per_env_step)
    assert cfg.policy.action_mode == "auto"
    assert cfg.policy.goal_noise_z is None and cfg.policy.goal_noise_joint_pos is None
    resolve_training_observations(cfg)
    assert cfg.policy.action_mode == mode
    assert cfg.policy.goal_noise_z == z_half
    assert cfg.policy.goal_noise_joint_pos == q_half
    assert cfg.policy.goal_dim == goal_dim
    assert (cfg.policy.encoder_hidden, cfg.policy.decoder_hidden,
            cfg.policy.latent_dim, cfg.policy.decoder_last_gain) == original_networks
    assert (cfg.wm_trainer.lr, cfg.wm_trainer.batch_size,
            cfg.policy_trainer.lr, cfg.policy_trainer.batch_size,
            cfg.policy_trainer.horizon, cfg.schedule.updates_per_env_step) == original_optimizer
    resolved = asdict(cfg)
    resolve_training_observations(cfg)
    assert asdict(cfg) == resolved


def test_explicit_legacy_and_noise_ablation_survive_default_resolution():
    cfg = TrainConfig(robot="khan_mini")
    cfg.policy.action_mode = "default"
    cfg.policy.goal_noise_z = cfg.policy.goal_noise_joint_pos = 0.0
    resolve_training_observations(cfg)
    assert cfg.policy.action_mode == "default"
    assert cfg.policy.goal_noise_z == cfg.policy.goal_noise_joint_pos == 0.0
    # A legacy observation layout explicitly selected for Khan remains legacy.
    cfg = TrainConfig(robot="khan_mini")
    cfg.env.observation_layout = "supermimic"
    resolve_training_observations(cfg)
    assert cfg.policy.action_mode == "default"
    assert cfg.policy.goal_noise_z == cfg.policy.goal_noise_joint_pos == 0.0


@pytest.mark.parametrize("obs_noise", [False, True])
@pytest.mark.parametrize("goal_noise", [False, True])
@pytest.mark.parametrize("running_normalization", [False, True])
def test_collection_passes_clean_reference_and_separate_encoder_copy(
    obs_noise, goal_noise, running_normalization,
):
    cfg = _small_policy_config(running_normalization=running_normalization,
                              **({} if goal_noise else
                                 {"goal_noise_z": 0.0, "goal_noise_joint_pos": 0.0}))
    policy = _policy(cfg)
    trainer = SimpleNamespace(policy=policy, cfg=SimpleNamespace(env=SimpleNamespace(obs_noise=obs_noise)))
    obs = {"policy": torch.randn(4, 465), "goal": torch.randn(4, 155)}
    original = {key: value.clone() for key, value in obs.items()}
    captures = []
    handle = policy.register_forward_pre_hook(
        lambda module, args, kwargs: captures.append((args, kwargs)), with_kwargs=True,
    )
    rng_before = torch.random.get_rng_state().clone()
    action = Trainer.act(trainer, obs)
    rng_after = torch.random.get_rng_state().clone()
    handle.remove()
    assert len(captures) == 1
    args, kwargs = captures[0]
    assert args[0] is obs["policy"] and args[1] is obs["goal"]
    encoder_goal = kwargs["encoder_goal"]
    if obs_noise and goal_noise:
        assert encoder_goal is not obs["goal"]
        torch.testing.assert_close(encoder_goal[:, :15], obs["goal"][:, :15], rtol=0, atol=0)
        assert not torch.equal(encoder_goal[:, 20:], obs["goal"][:, 20:])
        assert (encoder_goal[:, 15:20] - obs["goal"][:, 15:20]).abs().max() <= 0.010001
        assert (encoder_goal[:, 20:] - obs["goal"][:, 20:]).abs().max() <= 0.050001
        assert not torch.equal(rng_before, rng_after)
    else:
        assert encoder_goal is obs["goal"]
        assert torch.equal(rng_before, rng_after)
    for key, value in obs.items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)
    assert not action.requires_grad
    with torch.no_grad():
        expected = policy(obs["policy"], obs["goal"], encoder_goal=encoder_goal)
    torch.testing.assert_close(action, expected, rtol=0, atol=0)


def test_legacy_collection_matches_original_output_and_rng():
    cfg = _small_policy_config(action_mode="default", goal_noise_z=0, goal_noise_joint_pos=0)
    policy = _policy(cfg)
    trainer = SimpleNamespace(policy=policy, cfg=SimpleNamespace(env=SimpleNamespace(obs_noise=True)))
    obs = {"policy": torch.randn(2, 465), "goal": torch.randn(2, 155)}
    rng = torch.random.get_rng_state().clone()
    actual = Trainer.act(trainer, obs)
    expected = policy.decoder(policy.encoders["motion"](obs["goal"]), obs["policy"])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_matching_resume_and_legacy_missing_fields_are_accepted():
    fresh = _small_policy_config()
    validate_policy_training_semantics(asdict(fresh), fresh)
    legacy = replace(fresh, action_mode="default", goal_noise_z=0, goal_noise_joint_pos=0)
    validate_policy_training_semantics({}, legacy)
    # Residual beta/frame have no meaning for a legacy decoder; old weights can
    # still load when only those inactive options differ.
    validate_policy_training_semantics(
        {"reference_residual_beta": 0.0, "reference_frame_offset": 4}, legacy,
    )


@pytest.mark.parametrize("changed", [
    {"action_mode": "default"}, {"reference_residual_beta": 0.5},
    {"reference_frame_offset": 0}, {"goal_noise_z": 0.0},
    {"goal_noise_joint_pos": 0.0},
])
def test_resume_rejects_changed_action_or_noise_semantics(changed):
    current = _small_policy_config()
    saved = {**asdict(current), **changed}
    with pytest.raises(ValueError, match="action/goal-noise semantics"):
        validate_policy_training_semantics(saved, current)


def test_zero_beta_still_preserves_explicit_residual_checkpoint_contract():
    requested = _small_policy_config(reference_residual_beta=0.0)
    validate_policy_training_semantics(asdict(requested), requested)
    # No implicit conversion between representations, even if the zero-beta
    # function happens to be equivalent: buffer and resume contracts differ.
    with pytest.raises(ValueError, match="action/goal-noise semantics"):
        validate_policy_training_semantics({**asdict(requested), "action_mode": "default"}, requested)


def test_trainer_rejects_legacy_resume_before_touching_model_state(tmp_path):
    cfg = TrainConfig(robot="khan_mini")
    resolve_training_observations(cfg)
    path = tmp_path / "legacy.pt"
    # Deliberately no model/optimizer state: refusal must precede any load.
    torch.save({"config": {"robot": "khan_mini", "env": {}, "policy": {}}}, path)
    trainer = SimpleNamespace(cfg=cfg, device="cpu")
    with pytest.raises(ValueError, match="Start a new run"):
        Trainer.load(trainer, path)


@pytest.mark.parametrize("changed_key", ["reference_q_default", "reference_action_scale"])
def test_trainer_rejects_mismatched_physical_reference_buffers_before_loading(tmp_path, changed_key):
    from supermimic.robots.contracts import robot_contract

    cfg = TrainConfig(robot="khan_mini")
    resolve_training_observations(cfg)
    cfg.policy = _small_policy_config()
    policy = _policy(cfg.policy)
    original = {key: value.clone() for key, value in policy.state_dict().items()}
    saved = {key: value.clone() for key, value in original.items()}
    saved[changed_key][0] += 0.125
    path = tmp_path / "wrong_reference_mapping.pt"
    torch.save({"config": asdict(cfg), "policy": saved,
                "robot_contract": robot_contract("khan_mini", "isaac_khan", env_config=asdict(cfg.env))}, path)
    trainer = SimpleNamespace(cfg=cfg, policy=policy, device="cpu", robot_spec=get_robot_spec("khan_mini"))
    with pytest.raises(ValueError, match=changed_key):
        Trainer.load(trainer, path)
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)

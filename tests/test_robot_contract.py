"""Reject same-width checkpoints whose anchor/reward/physics meanings changed."""

from copy import deepcopy

import pytest

from supermimic.robots.contracts import robot_contract, validate_robot_contract


def test_legacy_g1_stays_compatible_but_legacy_khan_is_not_silently_reinterpreted():
    validate_robot_contract({}, "g1")
    with pytest.raises(ValueError, match="Legacy Khan"):
        validate_robot_contract({}, "khan_mini")


@pytest.mark.parametrize("robot", ["g1", "khan_mini"])
def test_unchanged_contract_is_accepted(robot):
    validate_robot_contract({"robot_contract": robot_contract(robot)}, robot)


def test_legacy_31_dim_khan_context_cannot_resume_as_extended_dynamics():
    contract = deepcopy(robot_contract("khan_mini"))
    assert contract["simulation_schema"]["xi_dim"] == 91
    assert contract["observation_schema"]["critic_input_dim"] == 605
    contract["version"] = 5
    contract["simulation_schema"]["xi_dim"] = 31
    contract["simulation_schema"]["world_model_input_dim"] = 125
    contract["simulation_schema"]["delay_in_xi"] = False
    with pytest.raises(ValueError, match="simulation_schema"):
        validate_robot_contract({"robot_contract": contract}, "khan_mini")


@pytest.mark.parametrize("field,value", [
    ("anchor_body", "waist_yaw_Link"), ("com_body", "base_link"),
    ("tracked_bodies", ["base_link"]), ("model_sha256", "different"),
])
def test_changed_semantics_are_rejected_even_with_equal_policy_dimensions(field, value):
    contract = deepcopy(robot_contract("khan_mini"))
    contract[field] = value
    with pytest.raises(ValueError, match=field):
        validate_robot_contract({"robot_contract": contract}, "khan_mini")


def test_changed_actor_scale_is_rejected_even_when_observation_widths_match():
    contract = deepcopy(robot_contract("khan_mini"))
    contract["observation_schema"]["sensor_scales"] = {"base_ang_vel": 0.25, "joint_vel": 0.05}
    with pytest.raises(ValueError, match="observation_schema"):
        validate_robot_contract({"robot_contract": contract}, "khan_mini")


def test_noisy_reference_goal_contract_is_rejected_even_when_dimensions_match():
    contract = deepcopy(robot_contract("khan_mini"))
    assert contract["version"] == 6
    assert contract["observation_schema"]["actor_goal_noise"] is False
    assert contract["observation_schema"]["noise_half_width_before_scale"]["goal_root_z"] == 0
    assert contract["observation_schema"]["noise_half_width_before_scale"]["goal_joint_pos"] == 0
    contract["observation_schema"]["actor_goal_noise"] = True
    contract["observation_schema"]["noise_half_width_before_scale"]["goal_root_z"] = 0.01
    contract["observation_schema"]["noise_half_width_before_scale"]["goal_joint_pos"] = 0.05
    with pytest.raises(ValueError, match="observation_schema"):
        validate_robot_contract({"robot_contract": contract}, "khan_mini")


def test_saved_nondefault_delay_and_reset_semantics_are_validated():
    settings = {"actuator_delay": False, "actuator_delay_max_ms": 10.0,
                "reset_root_z_offset": 0.0, "domain_randomization": True}
    checkpoint = {"config": {"env": settings},
                  "robot_contract": robot_contract("khan_mini", env_config=settings)}
    validate_robot_contract(checkpoint, "khan_mini")
    checkpoint["config"]["env"]["reset_root_z_offset"] = 0.05
    with pytest.raises(ValueError, match="simulation_schema"):
        validate_robot_contract(checkpoint, "khan_mini")


@pytest.mark.parametrize("field,value", [
    ("actuator_delay", False), ("actuator_delay_max_ms", 10.0),
    ("reset_root_z_offset", 0.0), ("domain_randomization", False),
])
def test_resume_rejects_changed_delay_or_reset_before_loading_weights(tmp_path, field, value):
    from dataclasses import asdict

    import torch

    from supermimic.config import TrainConfig, resolve_training_observations
    from supermimic.robots.registry import get_robot_spec
    from supermimic.training.loop import Trainer

    trainer = Trainer.__new__(Trainer)
    trainer.cfg = TrainConfig(robot="khan_mini", device="cpu")
    resolve_training_observations(trainer.cfg)
    trainer.robot_spec = get_robot_spec("khan_mini")
    trainer.device = "cpu"
    saved = asdict(trainer.cfg)
    saved["env"][field] = value
    checkpoint = {"config": saved,
                  "robot_contract": robot_contract("khan_mini", env_config=saved["env"])}
    path = tmp_path / "different_physics.pt"
    torch.save(checkpoint, path)
    # No networks are constructed: reject incompatible semantics before loading.
    with pytest.raises(ValueError, match="Checkpoint environment semantics differ"):
        trainer.load(path)

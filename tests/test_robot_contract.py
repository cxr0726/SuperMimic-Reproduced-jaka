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
    assert contract["version"] == 4
    assert contract["observation_schema"]["actor_goal_noise"] is False
    assert contract["observation_schema"]["noise_half_width_before_scale"]["goal_root_z"] == 0
    assert contract["observation_schema"]["noise_half_width_before_scale"]["goal_joint_pos"] == 0
    contract["observation_schema"]["actor_goal_noise"] = True
    contract["observation_schema"]["noise_half_width_before_scale"]["goal_root_z"] = 0.01
    contract["observation_schema"]["noise_half_width_before_scale"]["goal_joint_pos"] = 0.05
    with pytest.raises(ValueError, match="observation_schema"):
        validate_robot_contract({"robot_contract": contract}, "khan_mini")

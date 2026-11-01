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

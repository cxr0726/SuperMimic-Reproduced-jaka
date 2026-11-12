"""The expanded plant context must not reinterpret historical checkpoints."""

import hashlib
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from supermimic.config import TrainConfig, resolve_training_observations
from supermimic.robots.contracts import (
    checkpoint_context_layout,
    robot_contract,
    validate_robot_contract,
)
from supermimic.world_model.state_spec import StateSpec


@pytest.mark.parametrize("fresh_queue", [False, True])
def test_v1_robot_contract_matches_previous_revision_exactly(fresh_queue):
    env = {"actuator_delay": False, "delay_reset_first_step_no_delay": fresh_queue}
    expected = robot_contract("khan_mini", "isaac_khan", env_config=env)
    # Golden simulation schemas from 0c94c78, excluding model file hashes so
    # legitimate asset changes still use the separate robot-model contract.
    digest = hashlib.sha256(json.dumps(expected["simulation_schema"], sort_keys=True).encode()).hexdigest()
    assert digest == (
        "26b331531ef3b1050c41c1a849e612c2bcc32ec2fc9971525893e9cfc0dc1893" if fresh_queue else
        "97d66a146ecf76952462606b477d867c55c7028459f11c03d2c0e28c877caa2a"
    )
    assert robot_contract("khan_mini", "isaac_khan", env_config={
        **env, "context_layout": "khan_dr_v1",
    }) == expected


@pytest.mark.parametrize("kind,version,xi,wm,critic", [
    ("khan_dr_v2", 8, 99, 193, 613), ("khan_dr_v3", 9, 101, 195, 615),
])
def test_new_context_is_explicit_in_checkpoint_contract(kind, version, xi, wm, critic):
    cfg = TrainConfig(robot="khan_mini")
    cfg.env.context_layout = kind
    resolve_training_observations(cfg)
    saved = asdict(cfg)
    contract = robot_contract("khan_mini", "isaac_khan", env_config=saved["env"])
    assert contract["version"] == version
    schema = contract["simulation_schema"]
    assert schema["xi_dim"] == xi and schema["world_model_input_dim"] == wm
    assert contract["observation_schema"]["critic_input_dim"] == critic
    assert schema["xi_fields"]["lower_body_armature_factor"] == 12
    assert "ankle_armature_factor" not in schema["xi_fields"]
    assert schema["dynamics_randomization"]["lower_body_armature_factor"]["mode"] == "startup"
    if kind == "khan_dr_v3":
        assert contract["reward_config"]["w_foot_pos"] == contract["reward_config"]["w_foot_ori"] == .25
        assert schema["xi_fields"]["foot_mass_delta"] == 2
        assert contract["foot_tracking_schema"]["max_step_reward"] == pytest.approx(.122)
    else:
        assert "w_foot_pos" not in contract["reward_config"]
    validate_robot_contract({"config": saved, "robot_contract": contract}, "khan_mini", "isaac_khan")
    saved["env"]["context_layout"] = "khan_dr_v1"
    with pytest.raises(ValueError, match="robot_contract differs"):
        validate_robot_contract({"config": saved, "robot_contract": contract}, "khan_mini", "isaac_khan")


@pytest.mark.parametrize("saved_layout", [None, "khan_dr_v1", "khan_dr_v2", "khan_dr_v3"])
def test_restore_context_does_not_apply_fresh_defaults(saved_layout):
    path = Path(__file__).resolve().parents[1] / "scripts/benchmarks/benchmark_reference_fk_imagination.py"
    spec = importlib.util.spec_from_file_location("restore_context_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    saved = {"robot": "khan_mini", "env": {}, "world_model": {"state_spec": {}}}
    if saved_layout:
        saved["world_model"]["state_spec"]["context_layout"] = saved_layout
    cfg = TrainConfig()
    module.restore_config(cfg, saved)
    resolve_training_observations(cfg)
    expected = saved_layout or "khan_dr_v1"
    assert checkpoint_context_layout(saved) == cfg.env.context_layout == expected
    assert cfg.world_model.state_spec.context_layout == expected
    runtime_spec = StateSpec(cfg.world_model.state_spec, device="cpu", num_joints=27)
    assert runtime_spec.xi_dim == {"khan_dr_v1": 91, "khan_dr_v2": 99, "khan_dr_v3": 101}[expected]
    assert len(runtime_spec.xi_scale) == runtime_spec.xi_dim


def test_explicit_checkpoint_context_wins_over_wm_default():
    assert checkpoint_context_layout({"robot": "g1"}) == "base"
    assert checkpoint_context_layout({
        "robot": "khan_mini", "env": {"context_layout": "khan_dr_v1"},
        "world_model": {"state_spec": {"context_layout": "khan_dr_v2"}},
    }) == "khan_dr_v1"

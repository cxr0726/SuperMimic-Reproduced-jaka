"""The calibrated foot plant exposes startup friction to the learned models."""

from dataclasses import asdict
import importlib.util
from pathlib import Path

import pytest
import torch
from mjlab.envs.mdp import dr

from supermimic.config import TrainConfig, resolve_training_observations
from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg
from supermimic.robots.contracts import robot_contract, validate_robot_contract
from supermimic.sim_context import KHAN_ANKLE_JOINTS, SimContextLayout
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg


def test_v4_records_actual_friction_and_normalizes_new_ranges_without_moving_old_fields():
    old, new = SimContextLayout(27, "khan_dr_v3"), SimContextLayout(27, "khan_dr_v4")
    assert new.dim == 105
    assert {name: new.slices[name] for name in old.slices} == old.slices
    assert new.slices["ankle_frictionloss"] == slice(101, 105)
    assert new.offset[:99] == old.offset[:99] and new.scale[:99] == old.scale[:99]
    spec = StateSpec(StateSpecCfg(context_layout="khan_dr_v4"), "cpu", torch.float64, num_joints=27)
    raw = spec.xi_offset.repeat(3, 1)
    raw[:, 99:101] = torch.tensor([[0., .15], [.075, .075], [.15, 0.]], dtype=torch.float64)
    raw[:, 101:] = torch.tensor([[.1], [.175], [.25]], dtype=torch.float64)
    normalized = spec.normalize_xi(raw)
    torch.testing.assert_close(normalized[:, 99:101], torch.tensor([[-1., 1.], [0., 0.], [1., -1.]], dtype=torch.float64))
    torch.testing.assert_close(normalized[:, 101:], torch.tensor([[-1.], [0.], [1.]], dtype=torch.float64).expand(-1, 4))


def test_fresh_v4_shares_foot_payload_and_randomizes_ankle_friction_independently():
    fresh = TrainConfig(robot="khan_mini")
    resolve_training_observations(fresh)
    assert fresh.env.context_layout == fresh.world_model.state_spec.context_layout == "khan_dr_v4"
    assert fresh.env.ankle_encoder_bias_half_range == .02
    assert fresh.env.foot_mass_shared
    cfg = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=["unused"], robot="khan_mini"))
    for name, func, shared in (
        ("foot_mass", dr.body_mass, True), ("ankle_frictionloss", dr.joint_friction, False),
    ):
        event = cfg.events[name]
        assert event.mode == "startup" and event.func is func
        assert event.params["ranges"] == (0., .15)
        assert event.params["operation"] == "add" and event.params["shared_random"] is shared
    assert cfg.events["ankle_frictionloss"].params["asset_cfg"].joint_names == KHAN_ANKLE_JOINTS
    assert "dof_frictionloss" in cfg.events["ankle_frictionloss"].func.model_fields
    nominal = make_env_cfg(SuperMimicEnvCfgArgs(
        npz_files=["unused"], robot="khan_mini", domain_randomization=False,
    ))
    assert set(nominal.events) == {"foot_friction"}


def test_v4_contract_and_restore_preserve_physics_and_expanded_network_inputs():
    cfg = TrainConfig(robot="khan_mini")
    cfg.env.actuator_delay = False
    cfg.env.disable_neck_action_noise = True
    resolve_training_observations(cfg)
    saved = asdict(cfg)
    contract = robot_contract("khan_mini", "isaac_khan", env_config=saved["env"])
    assert contract["version"] == 15
    schema = contract["simulation_schema"]
    assert schema["xi_dim"] == 105 and schema["world_model_input_dim"] == 199
    assert contract["observation_schema"]["critic_input_dim"] == 619
    assert schema["ankle_frictionloss_joint_order"] == list(KHAN_ANKLE_JOINTS)
    assert schema["dynamics_randomization"]["foot_mass_delta_kg"]["range"] == [0., .15]
    assert schema["dynamics_randomization"]["foot_mass_delta_kg"]["independent_per_body"] is False
    assert schema["dynamics_randomization"]["ankle_frictionloss_nm"]["total_range"] == [.1, .25]
    assert contract["nominal_joint_physics"]["joint_frictionloss_randomized"] is True
    for foot in contract["nominal_foot_inertials"]["bodies"].values():
        assert foot["effective_mass_kg"] == .58502
        assert foot["inertia_scale"] == pytest.approx(.58502/.48502)
    validate_robot_contract({"config": saved, "robot_contract": contract}, "khan_mini", "isaac_khan")
    path = Path(__file__).resolve().parents[1] / "scripts/benchmarks/benchmark_reference_fk_imagination.py"
    spec = importlib.util.spec_from_file_location("restore_v4_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    restored = TrainConfig()
    module.restore_config(restored, saved)
    resolve_training_observations(restored)
    assert asdict(restored.env) == saved["env"]
    saved["env"]["context_layout"] = "khan_dr_v3"
    with pytest.raises(ValueError, match="robot_contract differs"):
        validate_robot_contract({"config": saved, "robot_contract": contract}, "khan_mini", "isaac_khan")

    saved["env"]["context_layout"] = "khan_dr_v4"
    del saved["env"]["foot_mass_shared"]
    historical = robot_contract("khan_mini", "isaac_khan", env_config=saved["env"])
    assert historical["version"] == 14
    assert historical["simulation_schema"]["dynamics_randomization"]["foot_mass_delta_kg"][
        "independent_per_body"
    ] is True
    validate_robot_contract({"config": saved, "robot_contract": historical}, "khan_mini", "isaac_khan")
    module.restore_config(restored, saved)
    resolve_training_observations(restored)
    assert restored.env.foot_mass_shared is False

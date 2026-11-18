"""Checks for startup ankle dry friction and its replay context."""

from dataclasses import asdict
from pathlib import Path

import mujoco
import pytest
import torch
import warp as wp
from mjlab.envs.mdp import dr

from supermimic.config import TrainConfig, resolve_training_observations
from supermimic.envs import SuperMimicEnvCfgArgs, make_env, make_env_cfg
from supermimic.envs.sim_context import SimContext
from supermimic.robots.contracts import robot_contract, validate_robot_contract
from supermimic.robots.registry import get_robot_spec
from supermimic.sim_context import KHAN_ANKLE_JOINTS, SimContextLayout
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg

NPZ = Path(__file__).resolve().parents[1] / "data/npz/khan_mini/lafan1/walk1_subject1.npz"
ANKLES = (
    "Left_ankle_pitch_joint", "Left_ankle_roll_joint",
    "Right_ankle_pitch_joint", "Right_ankle_roll_joint",
)
FIELD = "ankle_frictionloss"


def test_frictionloss_context_appends_four_axes_without_reinterpreting_v3():
    v3 = SimContextLayout(27, "khan_dr_v3")
    v4 = SimContextLayout(27, "khan_dr_v4")
    assert KHAN_ANKLE_JOINTS == ANKLES
    assert v3.dim == 101 and v4.dim == 105
    assert v4.slices[FIELD] == slice(101, 105)
    assert {name: v4.slices[name] for name in v3.slices} == v3.slices
    assert v4.offset[:99] == v3.offset[:99] and v4.scale[:99] == v3.scale[:99]
    assert v4.offset[99:] == (0.075,) * 2 + (0.175,) * 4
    assert v4.scale[99:] == (0.075,) * 6

    state = StateSpec(StateSpecCfg(context_layout=v4.kind), "cpu", num_joints=27)
    raw = state.xi_offset.repeat(1, 1)
    raw[:, v4.slices[FIELD]] = torch.tensor([[0.1, 0.175, 0.25, 0.1]])
    normalized = state.normalize_xi(raw)
    torch.testing.assert_close(normalized[:, 101:], torch.tensor([[-1.0, 0.0, 1.0, -1.0]]))


def test_frictionloss_is_independent_additive_startup_sampling_for_v4():
    active = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=["unused.npz"], robot="khan_mini"))
    term = active.events[FIELD]
    assert term.mode == "startup" and term.func is dr.joint_friction
    assert "dof_frictionloss" in term.func.model_fields
    assert term.params["asset_cfg"].joint_names == ANKLES
    assert term.params["operation"] == "add" and term.params["ranges"] == (0.0, 0.15)
    assert term.params.get("distribution", "uniform") == "uniform"
    assert term.params.get("shared_random", False) is False
    assert active.recorders["replay"].params["context_layout"] == "khan_dr_v4"

    v3 = make_env_cfg(SuperMimicEnvCfgArgs(
        npz_files=["unused.npz"], robot="khan_mini", context_layout="khan_dr_v3",
    ))
    assert set(active.events) == {*v3.events, FIELD}
    assert active.events["foot_mass"].mode == v3.events["foot_mass"].mode == "startup"
    assert active.rewards == v3.rewards and len(active.rewards) == 10
    nominal = make_env_cfg(SuperMimicEnvCfgArgs(
        npz_files=["unused.npz"], robot="khan_mini", domain_randomization=False,
    ))
    assert set(nominal.events) == {"foot_friction"}
    g1 = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=["unused.npz"], robot="g1"))
    assert FIELD not in g1.events


def test_frictionloss_layout_is_explicit_in_checkpoint_contract():
    cfg = TrainConfig(robot="khan_mini")
    resolve_training_observations(cfg)
    assert cfg.env.context_layout == cfg.world_model.state_spec.context_layout == "khan_dr_v4"
    assert cfg.policy.action_mode == "default" and not cfg.env.foot_tracking
    saved = asdict(cfg)
    contract = robot_contract("khan_mini", "isaac_khan", env_config=saved["env"])
    schema = contract["simulation_schema"]
    assert schema["xi_fields"][FIELD] == 4
    assert schema["xi_dim"] == 105 and schema["world_model_input_dim"] == 199
    assert contract["observation_schema"]["critic_input_dim"] == 619
    validate_robot_contract({"config": saved, "robot_contract": contract}, "khan_mini", "isaac_khan")

    v3_env = {**saved["env"], "context_layout": "khan_dr_v3"}
    legacy = robot_contract("khan_mini", "isaac_khan", env_config=v3_env)
    assert legacy["version"] != contract["version"]
    assert legacy["simulation_schema"]["xi_dim"] == 101
    assert FIELD not in legacy["simulation_schema"]["xi_fields"]
    assert legacy["reward_config"] == contract["reward_config"]
    with pytest.raises(ValueError, match="robot_contract differs"):
        validate_robot_contract({"config": saved, "robot_contract": legacy}, "khan_mini", "isaac_khan")


def _check_frictionloss_readback(env, randomized):
    robot = env.scene["robot"]
    rec = env.recorder_manager.get_term("replay")
    columns = rec.ctx.layout.slices[FIELD]
    dofs = robot.indexing.joint_v_adr.reshape(-1)[[robot.joint_names.index(name) for name in ANKLES]]
    nominal = env.sim.get_default_field("dof_frictionloss")
    applied = env.sim.model.dof_frictionloss[...]
    if applied.ndim == nominal.ndim:
        applied = applied.unsqueeze(0).expand(env.num_envs, -1)
    assert applied.shape == (env.num_envs, nominal.numel())
    actual = applied[:, dofs]
    torch.testing.assert_close(rec.ctx.read()[:, columns], actual, rtol=0, atol=0)

    other_dofs = torch.ones_like(nominal, dtype=torch.bool)
    other_dofs[dofs] = False
    torch.testing.assert_close(applied[:, other_dofs], nominal[other_dofs].expand(env.num_envs, -1),
                               rtol=0, atol=0)
    if randomized:
        assert ((actual >= 0.1) & (actual <= 0.25)).all()
        assert (actual.std(dim=0) > 0).all()
        assert all(row.unique().numel() > 1 for row in actual)
    else:
        torch.testing.assert_close(actual, nominal[dofs].expand_as(actual), rtol=0, atol=0)
    return actual


@pytest.mark.slow
@pytest.mark.parametrize("randomized", [False, True])
def test_applied_ankle_frictionloss_reaches_replay_and_survives_reset(randomized):
    if not torch.cuda.is_available() or not NPZ.is_file():
        pytest.skip("needs CUDA and converted Khan LAFAN1")
    env = make_env(SuperMimicEnvCfgArgs(
        npz_files=[str(NPZ)], robot="khan_mini", context_layout="khan_dr_v4",
        num_envs=8, replay_capacity=64, action_noise_std=0.0, obs_noise=False, reset_noise=False,
        domain_randomization=randomized, actuator_delay=False, sampling_mode="start", seed=317,
    ), device="cuda:0")
    try:
        startup = _check_frictionloss_readback(env, randomized).clone()
        env.reset(seed=317)
        rec = env.recorder_manager.get_term("replay")
        spec = get_robot_spec("khan_mini")
        v3 = SimContext(env, spec.foot_geom_regex, spec.com_body, context_layout="khan_dr_v3")
        columns = rec.ctx.layout.slices[FIELD]
        ids = torch.tensor([1, 3, 5, 7], device=env.device)
        all_ids = torch.arange(env.num_envs, device=env.device)
        env.reset(env_ids=ids)
        torch.testing.assert_close(rec.ctx.read()[:, :101], v3.read(), rtol=0, atol=0)
        torch.testing.assert_close(rec.prev_xi[:, columns], startup, rtol=0, atol=0)

        actions = torch.zeros(env.num_envs, spec.num_joints, device=env.device)
        for step in range(10):
            if step == 6:
                env.episode_length_buf[ids] = env.max_episode_length - 1
            pointer = rec.buffer.ptr.clone()
            env.step(actions)
            data = env.sim.wp_data
            expected_rows = spec.num_joints
            constraint_count = wp.to_torch(data.nefc)
            assert data.njmax == 512 and (constraint_count < data.njmax).all()
            assert (wp.to_torch(data.nf) == expected_rows).all()
            # Dynamic model fields must actually produce solver constraint rows.
            types = wp.to_torch(data.efc.type)
            row = torch.arange(data.njmax, device=env.device)
            valid = row < constraint_count[:, None]
            friction_rows = valid & (types == int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF))
            assert (friction_rows.sum(-1) == expected_rows).all()
            torch.testing.assert_close(_check_frictionloss_readback(env, randomized), startup,
                                       rtol=0, atol=0)
            torch.testing.assert_close(rec.prev_xi[:, columns], startup, rtol=0, atol=0)
            torch.testing.assert_close(rec.buffer.xi[all_ids, pointer, columns], startup,
                                       rtol=0, atol=0)
            if step == 6:
                assert env.reset_buf[ids].all()
        for batch in (rec.buffer.sample_windows(16, 3), rec.buffer.sample_starts(16, hist=5)):
            assert batch.xi.shape == (16, 105)
            # Replaying a row must return all four values from the same world.
            assert ((batch.xi[:, None, columns] == startup[None, :, :]).all(-1).any(-1)).all()
    finally:
        env.close()

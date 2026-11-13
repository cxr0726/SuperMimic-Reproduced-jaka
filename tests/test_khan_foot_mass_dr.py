"""Startup foot payloads, physical model constants, and context v3 stay aligned."""

from copy import copy
from dataclasses import fields
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
import warp as wp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import RecomputeLevel

from supermimic.envs import SuperMimicEnvCfgArgs, make_env, make_env_cfg
from supermimic.envs.mdp.rewards import TrackingRewardCfg
from supermimic.envs.sim_context import SimContext
from supermimic.envs.state_io import StateIO
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.robots.registry import get_robot_spec
from supermimic.sim_context import KHAN_FOOT_BODIES, SimContextLayout
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg
from supermimic.world_model.wm_trainer import reference_at

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data/npz/khan_mini/lafan1/walk1_subject1.npz"
FEET = ("Left_ankle_roll_Link", "Right_ankle_roll_Link")
DERIVED = ("dof_invweight0", "body_invweight0", "actuator_acc0")


def test_v3_appends_physical_foot_mass_deltas_without_changing_v2_prefix():
    old, new = SimContextLayout(27, "khan_dr_v2"), SimContextLayout(27, "khan_dr_v3")
    assert KHAN_FOOT_BODIES == FEET
    assert old.dim == 99 and new.dim == 101
    assert new.slices["foot_mass_delta"] == slice(99, 101)
    assert {name: new.slices[name] for name in old.slices} == old.slices
    assert new.offset[:99] == old.offset and new.scale[:99] == old.scale
    assert new.offset[99:] == new.scale[99:] == (0.1, 0.1)
    spec = StateSpec(StateSpecCfg(context_layout="khan_dr_v3"), "cpu", torch.float64,
                     num_joints=27)
    assert spec.xi_dim == 101
    raw = spec.xi_offset.repeat(3, 1)
    raw[:, 99:] = torch.tensor([[0., 0.2], [0.1, 0.1], [0.2, 0.]], dtype=torch.float64)
    torch.testing.assert_close(spec.normalize_xi(raw)[:, 99:], torch.tensor(
        [[-1., 1.], [0., 0.], [1., -1.]], dtype=torch.float64,
    ), rtol=0, atol=0)


def test_foot_payload_is_independent_startup_only_and_opt_in_by_context_version():
    cfg = make_env_cfg(SuperMimicEnvCfgArgs(
        npz_files=["unused.npz"], robot="khan_mini", context_layout="khan_dr_v3",
    ))
    term = cfg.events["foot_mass"]
    assert term.mode == "startup" and term.func is dr.body_mass
    assert term.func.recompute == RecomputeLevel.set_const
    assert term.params["asset_cfg"].body_names == FEET
    assert term.params["asset_cfg"].preserve_order
    assert term.params["operation"] == "add" and term.params["ranges"] == (0., 0.2)
    assert term.params.get("shared_random", False) is False
    assert cfg.recorders["replay"].params["context_layout"] == "khan_dr_v3"
    assert cfg.events["lower_body_armature"].mode == cfg.events["waist_mass"].mode == "startup"
    assert cfg.events["actuator_gains"].mode == "reset"
    assert cfg.events["base_com"].params["ranges"] == {
        0: (-0.035, 0.035), 1: (-0.07, 0.07), 2: (-0.07, 0.07),
    }
    for kind in ("khan_dr_v1", "khan_dr_v2"):
        legacy = make_env_cfg(SuperMimicEnvCfgArgs(
            npz_files=["unused.npz"], robot="khan_mini", context_layout=kind,
        ))
        assert "foot_mass" not in legacy.events
    nominal = make_env_cfg(SuperMimicEnvCfgArgs(
        npz_files=["unused.npz"], robot="khan_mini", context_layout="khan_dr_v3",
        domain_randomization=False,
    ))
    assert set(nominal.events) == {"foot_friction"}


def _make_env(*, randomized, foot_tracking=False):
    if not torch.cuda.is_available() or not NPZ.is_file():
        pytest.skip("needs CUDA and converted Khan LAFAN1")
    return make_env(SuperMimicEnvCfgArgs(
        npz_files=[str(NPZ)], robot="khan_mini", context_layout="khan_dr_v3",
        foot_tracking=foot_tracking, num_envs=4, replay_capacity=64, action_noise_std=0., obs_noise=False,
        reset_noise=False, domain_randomization=randomized, actuator_delay=False,
        sampling_mode="start", seed=317,
    ), device="cuda:0")


def _world_field(env, name):
    value = getattr(env.sim.model, name)[...]
    nominal = env.sim.get_default_field(name)
    if value.ndim == nominal.ndim:
        value = value.unsqueeze(0).expand(env.num_envs, *value.shape)
    return value, nominal


def _readback(env, *, randomized):
    robot = env.scene["robot"]
    ctx = env.recorder_manager.get_term("replay").ctx
    xi = ctx.read()
    assert xi.shape == (4, 101) and torch.isfinite(xi).all()
    foot_ids = torch.tensor([
        int(robot.indexing.body_ids[robot.body_names.index(name)]) for name in FEET
    ], device=env.device)
    torch.testing.assert_close(ctx.foot_body_global_ids.long(), foot_ids, rtol=0, atol=0)
    mass, nominal = _world_field(env, "body_mass")
    delta = mass[:, foot_ids] - nominal[foot_ids]
    torch.testing.assert_close(xi[:, 99:101], delta, rtol=0, atol=0)
    waist = int(robot.indexing.body_ids[robot.body_names.index("waist_yaw_Link")])
    torch.testing.assert_close(xi[:, 97], mass[:, waist] - nominal[waist], rtol=0, atol=0)
    unselected = torch.ones(nominal.shape, dtype=torch.bool, device=env.device)
    unselected[foot_ids] = False
    unselected[waist] = False
    torch.testing.assert_close(mass[:, unselected], nominal[unselected].expand(4, -1),
                               rtol=0, atol=0)
    inertia, default_inertia = _world_field(env, "body_inertia")
    torch.testing.assert_close(inertia, default_inertia.expand_as(inertia), rtol=0, atol=0)
    # The unchanged v2 reader on the same plant must give exactly the v3 prefix.
    robot_spec = get_robot_spec("khan_mini")
    v2 = SimContext(env, robot_spec.foot_geom_regex, robot_spec.com_body,
                    context_layout="khan_dr_v2")
    torch.testing.assert_close(xi[:, :99], v2.read(), rtol=0, atol=0)
    spec = StateSpec(StateSpecCfg(context_layout="khan_dr_v3"), env.device, num_joints=27)
    assert (spec.normalize_xi(xi).abs() <= 1 + 1e-5).all()
    if randomized:
        assert ((delta >= 0.) & (delta <= 0.2 + 1e-7)).all()
        assert not torch.equal(delta[:, 0], delta[:, 1])
        assert (delta.std(dim=0) > 0.005).all()
        assert ((xi[:, 97] >= -2.) & (xi[:, 97] <= 3.)).all()
    else:
        torch.testing.assert_close(delta, torch.zeros_like(delta), rtol=0, atol=0)
        torch.testing.assert_close(mass, nominal.expand_as(mass), rtol=0, atol=0)
        # A zero physical payload is the lower endpoint, not the centered mean.
        torch.testing.assert_close(spec.normalize_xi(xi)[:, 99:],
                                   -torch.ones_like(delta), rtol=0, atol=0)
    return xi


def _cpu_constant_reference(env):
    model_inputs = []
    for field in fields(env.sim.wp_model):
        value = getattr(env.sim.wp_model, field.name)
        if not isinstance(value, wp.array) or not hasattr(env.sim.mj_model, field.name):
            continue
        target = getattr(env.sim.mj_model, field.name)
        if not isinstance(target, np.ndarray):
            continue
        shape = getattr(field.type, "shape", ())
        batched = bool(shape and shape[0] in ("*", "nworld"))
        raw = wp.to_torch(value).cpu().numpy().copy()
        if (raw.shape[1:] if batched else raw.shape) == target.shape:
            model_inputs.append((field.name, raw, batched))
    result = {name: [] for name in DERIVED}
    for world in range(env.num_envs):
        model = copy(env.sim.mj_model)
        for name, raw, batched in model_inputs:
            getattr(model, name)[:] = raw[world % raw.shape[0]] if batched else raw
        mujoco.mj_setConst(model, mujoco.MjData(model))
        for name in DERIVED:
            result[name].append(np.array(getattr(model, name), copy=True))
    return {name: np.stack(values) for name, values in result.items()}


def _assert_constants(env):
    expected = _cpu_constant_reference(env)
    for name in DERIVED:
        actual, _ = _world_field(env, name)
        torch.testing.assert_close(actual.double(), torch.as_tensor(
            expected[name], device=actual.device, dtype=torch.float64,
        ), rtol=1e-5, atol=1e-6, msg=name)


def _assert_reward_parity(env, *, foot_tracking):
    """Compare all physical terms on identical, nontrivial instantaneous states."""
    robot, spec = env.scene["robot"], get_robot_spec("khan_mini")
    command = env.command_manager.get_term("motion")
    cfg = TrackingRewardCfg.for_joint_names(spec.joint_names, foot_tracking=foot_tracking)
    cap = 0.122 if foot_tracking else 0.112
    assert cfg.max_step_reward == pytest.approx(cap)
    assert env.step_dt * sum(max(0., t.weight) for t in env.cfg.rewards.values()) == pytest.approx(cap)
    heads = AnalyticHeads(
        G1Kinematics(spec.load_model(), device=env.device), spec.tracked_bodies,
        spec.anchor_body, spec.root_body, robot.data.soft_joint_pos_limits[0].clone(),
        robot.data.default_joint_pos[0].clone(), reward_cfg=cfg,
    )
    io = StateIO(env)
    base = io.read()
    ids = torch.arange(env.num_envs, device=env.device)
    worst = {name: 0. for name in env.cfg.rewards}
    for _ in range(4):
        state = base.clone()
        state[:, :3] += .03 * torch.randn_like(state[:, :3])
        state[:, 3:7] += .06 * torch.randn_like(state[:, 3:7])
        state[:, 3:7] /= torch.linalg.vector_norm(state[:, 3:7], dim=-1, keepdim=True)
        state[:, 7:io.nq] += .12 * torch.randn_like(state[:, 7:io.nq])
        state[:, io.nq:] = .7 * torch.randn_like(state[:, io.nq:])
        io.write(ids, state)
        env.scene.update(env.step_dt)
        command._refresh_ref()
        command.update_relative_body_poses()
        env.action_manager.action.copy_(.2 * torch.randn_like(env.action_manager.action))
        env.action_manager.prev_action.copy_(.2 * torch.randn_like(env.action_manager.prev_action))
        # Supply the existing tracked-body result: reward, including the feet,
        # must not require a second forward-kinematics evaluation.
        bodies = heads.robot_bodies(state)
        reference = reference_at(command.library, command.frame)
        terms = heads.reward(state, env.action_manager.action, env.action_manager.prev_action,
                             reference, bodies=bodies)
        assert set(terms) == {*env.cfg.rewards, "total"}
        for name in env.cfg.rewards:
            term = env.reward_manager.get_term_cfg(name)
            physical = env.step_dt * term.weight * term.func(env, **term.params)
            error = (terms[name] - physical).abs().max().item()
            worst[name] = max(worst[name], error)
            torch.testing.assert_close(terms[name], physical, rtol=1e-5, atol=5e-7, msg=name)
    assert max(worst.values()) <= 5e-7


@pytest.mark.slow
def test_startup_foot_payload_survives_partial_and_automatic_resets_with_matching_constants():
    env = _make_env(randomized=True)
    try:
        startup_mass = _world_field(env, "body_mass")[0].clone()
        _assert_constants(env)
        env.reset(seed=317)
        rec = env.recorder_manager.get_term("replay")
        xi = _readback(env, randomized=True)
        frozen_ids = [*range(28, 31), *range(85, 98), 99, 100]
        frozen = xi[:, frozen_ids].clone()
        constants = {name: _world_field(env, name)[0].clone() for name in DERIVED}
        ids = torch.tensor([1, 3], device=env.device)
        keep = torch.tensor([0, 2], device=env.device)
        for _ in range(4):
            env.reset(env_ids=ids)
            current = _readback(env, randomized=True)
            torch.testing.assert_close(current[:, frozen_ids], frozen, rtol=0, atol=0)
            torch.testing.assert_close(current[keep], xi[keep], rtol=0, atol=0)
            assert not torch.equal(current[ids, 31:85], xi[ids, 31:85])
            torch.testing.assert_close(_world_field(env, "body_mass")[0], startup_mass, rtol=0, atol=0)
            torch.testing.assert_close(rec.prev_xi, current, rtol=0, atol=0)
            xi = current
        for step in range(8):
            if step == 3:
                env.episode_length_buf[ids] = env.max_episode_length - 1
            before, ptr = rec.ctx.read(), rec.buffer.ptr.clone()
            env.step(torch.zeros(4, 27, device=env.device))
            after = _readback(env, randomized=True)
            torch.testing.assert_close(after[:, frozen_ids], frozen, rtol=0, atol=0)
            torch.testing.assert_close(rec.buffer.xi[torch.arange(4, device=env.device), ptr],
                                       before, rtol=0, atol=0)
            torch.testing.assert_close(rec.prev_xi, after, rtol=0, atol=0)
            if step == 3:
                assert env.reset_buf[ids].all()
        for name in DERIVED:
            torch.testing.assert_close(_world_field(env, name)[0], constants[name], rtol=0, atol=0)
        _assert_constants(env)
        assert rec.buffer.sample_starts(8, hist=5).xi.shape == (8, 101)
    finally:
        env.close()


@pytest.mark.slow
@pytest.mark.parametrize("foot_tracking", [False, True])
def test_nominal_v3_evaluation_has_no_foot_payload_and_matches_all_analytic_rewards(foot_tracking):
    env = _make_env(randomized=False, foot_tracking=foot_tracking)
    try:
        env.reset(seed=317)
        xi = _readback(env, randomized=False)
        _assert_constants(env)
        env.reset(env_ids=torch.tensor([1, 3], device=env.device))
        torch.testing.assert_close(_readback(env, randomized=False), xi, rtol=0, atol=0)
        _assert_reward_parity(env, foot_tracking=foot_tracking)
    finally:
        env.close()

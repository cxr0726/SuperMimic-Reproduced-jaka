"""Khan v2 startup inertial DR and its actual 99-d simulator context agree.

The CUDA checks use four worlds without a policy/optimizer. Inertial constants
are compared to a separate CPU64 MuJoCo model with the exact GPU model inputs.
"""

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
from supermimic.envs.partial_constants import khan_joint_armature
from supermimic.sim_context import (
    KHAN_LOWER_BODY_JOINTS,
    SimContextLayout,
    resolve_context_layout,
)
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data/npz/khan_mini/lafan1/walk1_subject1.npz"
LOWER_BODY = (
    "Left_hip_pitch_joint", "Left_hip_roll_joint", "Left_hip_yaw_joint",
    "Left_knee_joint", "Left_ankle_pitch_joint", "Left_ankle_roll_joint",
    "Right_hip_pitch_joint", "Right_hip_roll_joint", "Right_hip_yaw_joint",
    "Right_knee_joint", "Right_ankle_pitch_joint", "Right_ankle_roll_joint",
)
SLICES = {
    "friction": slice(0, 1), "encoder_bias": slice(1, 28),
    "com_offset": slice(28, 31), "kp_factor": slice(31, 58),
    "kd_factor": slice(58, 85), "lower_body_armature_factor": slice(85, 97),
    "torso_mass_delta": slice(97, 98), "delay_steps": slice(98, 99),
}
COM_RANGE = {0: (-0.035, 0.035), 1: (-0.07, 0.07), 2: (-0.07, 0.07)}
OLD_COM_RANGE = {0: (-0.025, 0.025), 1: (-0.05, 0.05), 2: (-0.05, 0.05)}
DERIVED = ("dof_invweight0", "body_invweight0", "actuator_acc0")


def _cfg(**kwargs):
    kwargs.setdefault("context_layout", "khan_dr_v2")
    return make_env_cfg(SuperMimicEnvCfgArgs(
        npz_files=["unused.npz"], robot="khan_mini", **kwargs,
    ))


def test_v2_layout_joint_order_and_normalization_are_explicit():
    assert KHAN_LOWER_BODY_JOINTS == LOWER_BODY
    layout = SimContextLayout(27, "khan_dr_v2")
    assert layout.dim == 99 and layout.slices == SLICES
    spec = StateSpec(StateSpecCfg(context_layout="khan_dr_v2"), "cpu", torch.float64,
                     num_joints=27)
    assert spec.xi_dim == 99 and spec.state_dim == 67 and spec.delta_dim == 66
    torch.testing.assert_close(spec.xi_scale[28:31], torch.tensor(
        [0.035, 0.07, 0.07], dtype=torch.float64,
    ), rtol=0, atol=0)
    torch.testing.assert_close(spec.normalize_xi(spec.xi_offset), torch.zeros(99, dtype=torch.float64))
    # Independently fill physical endpoints rather than deriving them from the
    # layout, so the test catches meters, factors, kg and lag-tick mixups.
    raw = spec.xi_offset.repeat(2, 1)
    raw[:, 28:31] = torch.tensor([[-0.035, -0.07, -0.07], [0.035, 0.07, 0.07]])
    raw[0, 31:97], raw[1, 31:97] = 0.8, 1.2
    raw[:, 97] = torch.tensor([-2.0, 3.0])
    raw[:, 98] = torch.tensor([0.0, 4.0])
    normalized = spec.normalize_xi(raw)
    torch.testing.assert_close(normalized[0, 28:], -torch.ones(71, dtype=torch.float64))
    torch.testing.assert_close(normalized[1, 28:], torch.ones(71, dtype=torch.float64))


def test_v2_uses_stock_startup_armature_but_gains_stay_per_reset():
    cfg = _cfg()
    assert resolve_context_layout("khan_mini") == "khan_dr_v3"
    assert cfg.recorders["replay"].params["context_layout"] == "khan_dr_v2"
    term = cfg.events["lower_body_armature"]
    assert "ankle_armature" not in cfg.events
    assert term.mode == "startup" and term.func is dr.joint_armature
    assert term.func.recompute == RecomputeLevel.set_const_0
    assert term.params["asset_cfg"].joint_names == LOWER_BODY
    assert term.params["asset_cfg"].preserve_order
    assert term.params["operation"] == "scale" and term.params["ranges"] == (0.8, 1.2)
    assert term.params["shared_random"] is False
    assert cfg.events["actuator_gains"].mode == "reset"
    assert cfg.events["actuator_gains"].params["kp_range"] == (0.8, 1.2)
    assert cfg.events["actuator_gains"].params["kd_range"] == (0.8, 1.2)
    assert cfg.events["base_com"].mode == cfg.events["waist_mass"].mode == "startup"
    assert cfg.events["base_com"].params["ranges"] == COM_RANGE
    assert cfg.events["base_com"].params["asset_cfg"].body_names == ("waist_yaw_Link",)
    assert cfg.events["waist_mass"].params["ranges"] == (-2.0, 3.0)


def test_explicit_v1_keeps_historical_reset_armature_and_com_while_eval_is_nominal():
    legacy = _cfg(context_layout="khan_dr_v1")
    assert SimContextLayout(27, "khan_dr_v1").dim == 91
    assert legacy.recorders["replay"].params["context_layout"] == "khan_dr_v1"
    term = legacy.events["ankle_armature"]
    assert term.mode == "reset" and term.func is khan_joint_armature
    assert term.params["asset_cfg"].joint_names == tuple(n for n in LOWER_BODY if "ankle" in n)
    assert legacy.events["base_com"].params["ranges"] == OLD_COM_RANGE
    assert "lower_body_armature" not in legacy.events
    for kind in ("khan_dr_v1", "khan_dr_v2"):
        nominal = _cfg(context_layout=kind, domain_randomization=False)
        assert set(nominal.events) == {"foot_friction"}
        assert nominal.events["foot_friction"].params["ranges"] == (1.0, 1.0)
        assert nominal.recorders["replay"].params["context_layout"] == kind
    g1 = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=["unused.npz"], robot="g1"))
    assert resolve_context_layout("g1") == "base"
    assert g1.events["base_com"].params["ranges"] == OLD_COM_RANGE
    assert g1.recorders["replay"].params["context_layout"] == "base"
    assert "lower_body_armature" not in g1.events
    with pytest.raises(ValueError):
        resolve_context_layout("g1", "khan_dr_v2")


def _make_env(*, randomized):
    if not torch.cuda.is_available() or not NPZ.is_file():
        pytest.skip("needs CUDA and converted Khan LAFAN1")
    return make_env(SuperMimicEnvCfgArgs(
        npz_files=[str(NPZ)], robot="khan_mini", context_layout="khan_dr_v2",
        num_envs=4, replay_capacity=64, action_noise_std=0.0,
        obs_noise=False, reset_noise=False, domain_randomization=randomized,
        actuator_delay=False, sampling_mode="start", seed=173,
    ), device="cuda:0")


def _world_field(env, name):
    value = getattr(env.sim.model, name)[...]
    nominal = env.sim.get_default_field(name)
    if value.ndim == nominal.ndim:
        value = value.unsqueeze(0).expand(env.num_envs, *value.shape)
    assert value.ndim == nominal.ndim + 1
    return value, nominal


def _readback(env, *, randomized):
    robot = env.scene["robot"]
    ctx = env.recorder_manager.get_term("replay").ctx
    xi = ctx.read()
    assert xi.shape == (4, 99) and torch.isfinite(xi).all()
    joint_dofs = robot.indexing.joint_v_adr.long()
    lower_ids = torch.tensor([robot.joint_names.index(n) for n in LOWER_BODY], device=env.device)
    lower_dofs = joint_dofs[lower_ids]
    torch.testing.assert_close(ctx.armature_dof_ids.long(), lower_dofs, rtol=0, atol=0)
    armature, default_armature = _world_field(env, "dof_armature")
    factors = armature[:, lower_dofs] / default_armature[lower_dofs]
    torch.testing.assert_close(xi[:, 85:97], factors, rtol=0, atol=0)
    other = torch.ones(27, dtype=torch.bool, device=env.device)
    other[lower_ids] = False
    torch.testing.assert_close(armature[:, joint_dofs[other]],
                               default_armature[joint_dofs[other]].expand(4, -1), rtol=0, atol=0)
    gains, default_gains = _world_field(env, "actuator_gainprm")
    bias, default_bias = _world_field(env, "actuator_biasprm")
    kp, kd = torch.empty(4, 27, device=env.device), torch.empty(4, 27, device=env.device)
    for actuator in robot.actuators:
        ids = actuator.global_ctrl_ids
        kp[:, actuator.target_ids] = gains[:, ids, 0] / default_gains[ids, 0]
        kd[:, actuator.target_ids] = bias[:, ids, 2] / default_bias[ids, 2]
        torch.testing.assert_close(bias[:, ids, 1], -gains[:, ids, 0], rtol=0, atol=0)
    torch.testing.assert_close(xi[:, 31:58], kp, rtol=0, atol=0)
    torch.testing.assert_close(xi[:, 58:85], kd, rtol=0, atol=0)
    com, default_com = _world_field(env, "body_ipos")
    masses, default_masses = _world_field(env, "body_mass")
    waist = int(robot.indexing.body_ids[robot.body_names.index("waist_yaw_Link")])
    torch.testing.assert_close(xi[:, 28:31], com[:, waist] - default_com[waist], rtol=0, atol=0)
    torch.testing.assert_close(xi[:, 97], masses[:, waist] - default_masses[waist], rtol=0, atol=0)
    other_bodies = torch.arange(default_masses.numel(), device=env.device) != waist
    torch.testing.assert_close(masses[:, other_bodies],
                               default_masses[other_bodies].expand(4, -1), rtol=0, atol=0)
    inertia, default_inertia = _world_field(env, "body_inertia")
    torch.testing.assert_close(inertia, default_inertia.expand_as(inertia), rtol=0, atol=0)
    # Disabling delay for the new training must leave its explicit context slot
    # present and nominal, independent of whether physical DR is enabled.
    torch.testing.assert_close(xi[:, 98], torch.zeros(4, device=env.device), rtol=0, atol=0)
    if randomized:
        for value in (factors, kp, kd):
            assert ((value >= 0.8 - 1e-6) & (value <= 1.2 + 1e-6)).all()
        assert ((xi[:, 97] >= -2.0) & (xi[:, 97] <= 3.0)).all()
        bound = torch.tensor([0.035, 0.07, 0.07], device=env.device)
        assert (xi[:, 28:31].abs() <= bound + 1e-7).all()
        assert (factors != 1.0).any(dim=0).all()
        assert factors.std() > 0.03 and not torch.equal(factors[:, 0], factors[:, 1])
    else:
        torch.testing.assert_close(xi[:, 0], torch.ones(4, device=env.device), rtol=0, atol=0)
        torch.testing.assert_close(xi[:, 1:31], torch.zeros(4, 30, device=env.device), rtol=0, atol=0)
        torch.testing.assert_close(xi[:, 31:97], torch.ones(4, 66, device=env.device), rtol=0, atol=0)
        torch.testing.assert_close(xi[:, 97:], torch.zeros(4, 2, device=env.device), rtol=0, atol=0)
    spec = StateSpec(StateSpecCfg(context_layout="khan_dr_v2"), env.device, num_joints=27)
    assert (spec.normalize_xi(xi).abs() <= 1 + 1e-5).all()
    assert not hasattr(env, "_khan_armature_constants")
    assert not hasattr(env.sim, "_khan_armature_constants")
    return xi


def _cpu_constant_reference(env):
    """Match every available GPU input before independently calling mj_setConst."""
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
        # Separate float32 CRB reductions may differ in the last few ppm;
        # compare to CPU64 with the established backend 10-ppm bound.
        torch.testing.assert_close(actual.double(), torch.as_tensor(
            expected[name], device=actual.device, dtype=torch.float64,
        ), rtol=1e-5, atol=1e-6, msg=name)


@pytest.mark.slow
def test_v2_startup_randomization_constants_and_replay_survive_resets():
    env = _make_env(randomized=True)
    try:
        # __init__ has already fired startup events, prior to any public reset.
        startup_armature = _world_field(env, "dof_armature")[0].clone()
        _assert_constants(env)
        env.reset(seed=173)
        xi = _readback(env, randomized=True)
        rec = env.recorder_manager.get_term("replay")
        fixed_ids = [*range(28, 31), *range(85, 98)]
        fixed_xi = xi[:, fixed_ids].clone()
        fixed_constants = {name: _world_field(env, name)[0].clone() for name in DERIVED}
        torch.testing.assert_close(rec.prev_xi, xi, rtol=0, atol=0)
        chosen = torch.tensor([1, 3], device=env.device)
        kept = torch.tensor([0, 2], device=env.device)
        for _ in range(5):
            env.reset(env_ids=chosen)
            current = _readback(env, randomized=True)
            torch.testing.assert_close(current[:, fixed_ids], fixed_xi, rtol=0, atol=0)
            torch.testing.assert_close(current[kept], xi[kept], rtol=0, atol=0)
            assert not torch.equal(current[chosen, 31:85], xi[chosen, 31:85])
            torch.testing.assert_close(_world_field(env, "dof_armature")[0],
                                       startup_armature, rtol=0, atol=0)
            for name in DERIVED:
                torch.testing.assert_close(_world_field(env, name)[0],
                                           fixed_constants[name], rtol=0, atol=0)
            torch.testing.assert_close(rec.prev_xi, current, rtol=0, atol=0)
            xi = current
        # Exercise the production auto-reset path and context snapshot timing.
        for step in range(8):
            if step == 3:
                env.episode_length_buf[chosen] = env.max_episode_length - 1
            before, ptr = rec.ctx.read(), rec.buffer.ptr.clone()
            env.step(torch.zeros(4, 27, device=env.device))
            rows = torch.arange(4, device=env.device)
            torch.testing.assert_close(rec.buffer.xi[rows, ptr], before, rtol=0, atol=0)
            after = _readback(env, randomized=True)
            torch.testing.assert_close(after[:, fixed_ids], fixed_xi, rtol=0, atol=0)
            torch.testing.assert_close(rec.prev_xi, after, rtol=0, atol=0)
            if step == 3:
                assert env.reset_buf[chosen].all()
            torch.testing.assert_close(after[~env.reset_buf], before[~env.reset_buf], rtol=0, atol=0)
        _assert_constants(env)
        starts = rec.buffer.sample_starts(8, hist=5)
        assert starts.xi.shape == (8, 99)
    finally:
        env.close()


@pytest.mark.slow
def test_v2_nominal_evaluation_has_unit_factors_and_zero_offsets():
    env = _make_env(randomized=False)
    try:
        env.reset(seed=173)
        xi = _readback(env, randomized=False)
        _assert_constants(env)
        env.reset(env_ids=torch.tensor([1, 3], device=env.device))
        torch.testing.assert_close(_readback(env, randomized=False), xi, rtol=0, atol=0)
    finally:
        env.close()

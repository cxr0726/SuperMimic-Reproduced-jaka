"""Physical DR values, packed context and replay agree for the real Khan plant.

The CUDA tests use eight mjlab environments and no optimizer. They check the
simulator fields that actually drive dynamics, rather than sampled-event caches.
"""

from pathlib import Path

import pytest
import torch

from supermimic.envs import SuperMimicEnvCfgArgs, make_env, make_env_cfg
from supermimic.sim_context import SimContextLayout
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data/npz/khan_mini/lafan1/walk1_subject1.npz"
ANKLES = (
    "Left_ankle_pitch_joint", "Left_ankle_roll_joint",
    "Right_ankle_pitch_joint", "Right_ankle_roll_joint",
)
SLICES = {
    "friction": slice(0, 1), "encoder_bias": slice(1, 28),
    "com_offset": slice(28, 31), "kp_factor": slice(31, 58),
    "kd_factor": slice(58, 85), "ankle_armature_factor": slice(85, 89),
    "torso_mass_delta": slice(89, 90), "delay_steps": slice(90, 91),
}


def test_extended_context_has_explicit_robot_order_and_normalized_bounds():
    layout = SimContextLayout(27, kind="khan_dr_v1")
    assert layout.dim == 91
    assert layout.slices == SLICES
    spec = StateSpec(StateSpecCfg(context_layout="khan_dr_v1"), "cpu", torch.float64,
                     num_joints=27)
    assert spec.xi_dim == 91
    assert spec.state_dim == 67 and spec.delta_dim == 66
    torch.testing.assert_close(spec.normalize_xi(spec.xi_offset), torch.zeros(91, dtype=torch.float64))
    # Independent physical endpoints exercise gain factors vs raw gains, kg vs
    # mass factors, and 5 ms ticks vs milliseconds in the delay slot.
    raw = spec.xi_offset.repeat(2, 1)
    for name in ("kp_factor", "kd_factor", "ankle_armature_factor"):
        raw[0, SLICES[name]] = 0.8
        raw[1, SLICES[name]] = 1.2
    raw[:, SLICES["torso_mass_delta"]] = torch.tensor([[-2.0], [3.0]])
    raw[:, SLICES["delay_steps"]] = torch.tensor([[0.0], [4.0]])
    normalized = spec.normalize_xi(raw)
    torch.testing.assert_close(normalized[0, 31:], -torch.ones(60, dtype=torch.float64))
    torch.testing.assert_close(normalized[1, 31:], torch.ones(60, dtype=torch.float64))
    # The old layout remains selectable for legacy networks and G1.
    assert SimContextLayout(27, kind="base").dim == 31
    assert StateSpec(StateSpecCfg(), "cpu", num_joints=29).xi_dim == 33


def test_dr_is_khan_only_and_nominal_evaluation_removes_physical_events():
    khan = make_env_cfg(SuperMimicEnvCfgArgs(
        npz_files=["unused.npz"], robot="khan_mini", context_layout="khan_dr_v1",
    ))
    nominal = make_env_cfg(SuperMimicEnvCfgArgs(
        npz_files=["unused.npz"], robot="khan_mini", context_layout="khan_dr_v1",
        domain_randomization=False,
    ))
    g1 = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=["unused.npz"], robot="g1"))
    added = set(khan.events) - set(g1.events)
    assert len(added) == 3
    assert {khan.events[name].mode for name in added} == {"reset", "startup"}
    assert set(nominal.events) == {"foot_friction"}
    assert khan.recorders["replay"].params["context_layout"] == "khan_dr_v1"
    assert nominal.recorders["replay"].params["context_layout"] == "khan_dr_v1"
    assert g1.recorders["replay"].params.get("context_layout", "base") == "base"
    # COM stays on the waist with its existing range, not the base-link anchor.
    com = khan.events["base_com"].params
    assert com["asset_cfg"].body_names == ("waist_yaw_Link",)
    assert com["ranges"] == {0: (-0.025, 0.025), 1: (-0.05, 0.05), 2: (-0.05, 0.05)}


def _make_env(dr):
    if not torch.cuda.is_available() or not NPZ.is_file():
        pytest.skip("needs CUDA and converted Khan LAFAN1")
    return make_env(SuperMimicEnvCfgArgs(
        npz_files=[str(NPZ)], robot="khan_mini", context_layout="khan_dr_v1",
        num_envs=8, replay_capacity=64,
        action_noise_std=0.0, obs_noise=False, reset_noise=False,
        domain_randomization=dr, sampling_mode="start", seed=73,
    ), device="cuda:0")


def _world_field(env, name):
    # Slice the mjlab TorchArray proxy to expose an ordinary tensor for assertions.
    value = getattr(env.sim.model, name)[...]
    nominal = env.sim.get_default_field(name)
    if value.ndim == nominal.ndim:
        value = value.unsqueeze(0).expand(env.num_envs, *value.shape)
    assert value.ndim == nominal.ndim + 1
    return value, nominal


def _assert_readback(env, *, randomized):
    robot = env.scene["robot"]
    ctx = env.recorder_manager.get_term("replay").ctx
    xi = ctx.read()
    assert xi.shape == (8, 91) and torch.isfinite(xi).all()
    gains, default_gains = _world_field(env, "actuator_gainprm")
    bias, default_bias = _world_field(env, "actuator_biasprm")
    kp = torch.zeros(8, 27, device=env.device)
    kd = torch.zeros_like(kp)
    for actuator in robot.actuators:
        ids = actuator.global_ctrl_ids
        kp[:, actuator.target_ids] = gains[:, ids, 0] / default_gains[ids, 0]
        kd[:, actuator.target_ids] = bias[:, ids, 2] / default_bias[ids, 2]
        # MuJoCo's position feedback gain must match its target gain, while Kd
        # is independently sampled into the velocity feedback coefficient.
        torch.testing.assert_close(bias[:, ids, 1], -gains[:, ids, 0], rtol=0, atol=0)
    torch.testing.assert_close(xi[:, SLICES["kp_factor"]], kp, rtol=0, atol=0)
    torch.testing.assert_close(xi[:, SLICES["kd_factor"]], kd, rtol=0, atol=0)

    armature, default_armature = _world_field(env, "dof_armature")
    joint_dofs = robot.indexing.joint_v_adr.long()
    ankle_ids = torch.tensor([robot.joint_names.index(n) for n in ANKLES], device=env.device)
    ankle_dofs = joint_dofs[ankle_ids]
    torch.testing.assert_close(default_armature[ankle_dofs], torch.tensor(
        [0.02, 0.01, 0.02, 0.01], device=env.device,
    ), rtol=0, atol=1e-8)
    ratios = armature[:, ankle_dofs] / default_armature[ankle_dofs]
    torch.testing.assert_close(xi[:, SLICES["ankle_armature_factor"]], ratios, rtol=0, atol=0)
    other = torch.ones(27, dtype=torch.bool, device=env.device)
    other[ankle_ids] = False
    torch.testing.assert_close(armature[:, joint_dofs[other]],
                               default_armature[joint_dofs[other]].expand(8, -1), rtol=0, atol=0)

    masses, default_masses = _world_field(env, "body_mass")
    waist = int(robot.indexing.body_ids[robot.body_names.index("waist_yaw_Link")])
    delta = masses[:, waist] - default_masses[waist]
    torch.testing.assert_close(xi[:, SLICES["torso_mass_delta"]], delta[:, None], rtol=0, atol=0)
    assert (masses[:, waist] > 0).all()
    body_other = torch.arange(default_masses.numel(), device=env.device) != waist
    torch.testing.assert_close(masses[:, body_other], default_masses[body_other].expand(8, -1),
                               rtol=0, atol=0)
    # This event represents a point-mass addition at the COM; it does not
    # independently randomize the authored body's rotational inertia tensor.
    inertia, default_inertia = _world_field(env, "body_inertia")
    torch.testing.assert_close(inertia, default_inertia.expand_as(inertia), rtol=0, atol=0)

    groups = robot._builtin_group._delayed_groups
    if randomized:
        assert len(groups) == 1
        lag = groups[0].delay_buffer.current_lags
        assert ((lag >= 0) & (lag <= 4)).all()
        torch.testing.assert_close(xi[:, SLICES["delay_steps"]], lag.float()[:, None], rtol=0, atol=0)
        for values in (kp, kd, ratios):
            assert ((values >= 0.8 - 1e-6) & (values <= 1.2 + 1e-6)).all()
        assert ((delta >= -2 - 1e-6) & (delta <= 3 + 1e-6)).all()
    else:
        torch.testing.assert_close(xi[:, 31:89], torch.ones_like(xi[:, 31:89]), rtol=0, atol=0)
        torch.testing.assert_close(xi[:, 89:], torch.zeros_like(xi[:, 89:]), rtol=0, atol=0)
    return xi


@pytest.mark.slow
def test_actual_dr_readback_partial_resets_and_replay_keep_same_context():
    env = _make_env(True)
    try:
        env.reset(seed=73)
        rec = env.recorder_manager.get_term("replay")
        xi = _assert_readback(env, randomized=True)
        torch.testing.assert_close(rec.prev_xi, xi, rtol=0, atol=0)
        assert xi[:, 31:58].std() > 0.02
        assert xi[:, 58:85].std() > 0.02
        assert not torch.equal(xi[:, 31:58], xi[:, 58:85])
        assert not torch.equal(xi[:, 85], xi[:, 86])
        ids = torch.tensor([1, 3, 6], device=env.device)
        keep = torch.ones(8, dtype=torch.bool, device=env.device)
        keep[ids] = False
        fixed_mass_com = torch.cat((xi[:, 28:31], xi[:, 89:90]), dim=-1).clone()
        changed = torch.zeros(3, dtype=torch.bool, device=env.device)
        for _ in range(12):
            env.reset(env_ids=ids)
            current = _assert_readback(env, randomized=True)
            torch.testing.assert_close(current[keep], xi[keep], rtol=0, atol=0)
            torch.testing.assert_close(torch.cat((current[:, 28:31], current[:, 89:90]), dim=-1),
                                       fixed_mass_com, rtol=0, atol=0)
            torch.testing.assert_close(rec.prev_xi, current, rtol=0, atol=0)
            changed |= (current[ids, 31:89] != xi[ids, 31:89]).any(dim=-1)
            # Bounds relative to original gains/armature catch reset compounding.
            xi = current
        assert changed.all()

        # Every saved transition uses the context before its action, including
        # transitions that terminate and trigger an episode/context reset.
        for step in range(8):
            # Force one genuine auto-reset through time_out, rather than only
            # testing the public explicit reset path.
            if step == 3:
                env.episode_length_buf[ids] = env.max_episode_length - 1
            before = rec.ctx.read()
            ptr = rec.buffer.ptr.clone()
            env.step(torch.zeros(8, 27, device=env.device))
            rows = torch.arange(8, device=env.device)
            torch.testing.assert_close(rec.buffer.xi[rows, ptr], before, rtol=0, atol=0)
            after = _assert_readback(env, randomized=True)
            torch.testing.assert_close(rec.prev_xi, after, rtol=0, atol=0)
            if step == 3:
                assert env.reset_buf[ids].all()
            held = ~env.reset_buf
            torch.testing.assert_close(after[held], before[held], rtol=0, atol=0)
        starts = rec.buffer.sample_starts(16, hist=5)
        assert starts.xi.shape == (16, 91)
        spec = StateSpec(StateSpecCfg(context_layout="khan_dr_v1"), env.device, num_joints=27)
        assert torch.isfinite(spec.normalize_xi(starts.xi)).all()
    finally:
        env.close()


@pytest.mark.slow
def test_nominal_evaluation_has_extended_context_but_no_randomized_dynamics():
    env = _make_env(False)
    try:
        env.reset()
        xi = _assert_readback(env, randomized=False)
        torch.testing.assert_close(xi[:, 1:31], torch.zeros_like(xi[:, 1:31]), rtol=0, atol=0)
        torch.testing.assert_close(xi[:, 0], torch.ones_like(xi[:, 0]), rtol=0, atol=0)
    finally:
        env.close()

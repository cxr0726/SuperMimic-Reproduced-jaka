"""Khan's episode-held actuator delay and actual-root reset clearance.

The bounded CUDA checks exercise mjlab's real fused actuator controls without
training a network.  Run the configuration checks alone with ``-m 'not slow'``.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from supermimic.envs import SuperMimicEnvCfgArgs, make_env, make_env_cfg
from supermimic.envs.actions import NoisyJointPositionActionCfg
from supermimic.envs.state_io import StateIO
from supermimic.robots.contracts import robot_contract
from supermimic.robots.registry import get_robot_spec
from supermimic.world_model.dynamics import DynamicsMLP
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data/npz/khan_mini/lafan1/walk1_subject1.npz"
NEEDS_CUDA_DATA = pytest.mark.skipif(
    not torch.cuda.is_available() or not NPZ.is_file(),
    reason="needs CUDA and converted Khan LAFAN1",
)
DELAY_FIELDS = (
    "delay_min_lag", "delay_max_lag", "delay_hold_prob",
    "delay_update_period", "delay_per_env_phase",
)


def _cfg(**kwargs):
    args = SuperMimicEnvCfgArgs(npz_files=["unused.npz"], robot="khan_mini", **kwargs)
    return make_env_cfg(args)


def _actuators(cfg):
    return cfg.scene.entities["robot"].articulation.actuators


def _three_channel_action(owned=(0, 1, 2)):
    """Small entity adapter exercising real mjlab three-field delay queues."""
    from mjlab.actuator.actuator import TransmissionType
    from mjlab.utils.buffers import DelayBuffer

    data = SimpleNamespace(
        joint_pos_target=torch.zeros(2, 3), joint_vel_target=torch.zeros(2, 3),
        joint_effort_target=torch.zeros(2, 3), encoder_bias=torch.full((2, 3), .01),
    )
    groups = [SimpleNamespace(
        target_ids=torch.tensor(ids), transmission_type=TransmissionType.JOINT,
        delay_buffer=DelayBuffer(min_lag=0, max_lag=4, batch_size=2, device="cpu", update_period=2**40),
    ) for ids in ([2, 0], [1])]

    def set_target(target, joint_ids):
        data.joint_pos_target[:, joint_ids] = target

    # The custom entry deliberately aliases a fused buffer: it must not prime a
    # second time using a one-joint layout instead of the group's [2,0] layout.
    duplicate = SimpleNamespace(cfg=SimpleNamespace(transmission_type=TransmissionType.JOINT),
                                _delay_buffer=groups[0].delay_buffer, target_ids=torch.tensor([2]))
    entity = SimpleNamespace(
        data=data, set_joint_position_target=set_target,
        find_joints_by_actuator_names=lambda names: (list(owned), [f"j{i}" for i in owned]),
        _builtin_group=SimpleNamespace(_delayed_groups=[]),
        _fused_actuator_group=SimpleNamespace(_groups=groups), _custom_actuators=[duplicate],
    )
    env = SimpleNamespace(num_envs=2, device="cpu", scene={"robot": entity})
    term = NoisyJointPositionActionCfg(
        entity_name="robot", actuator_names=(".*",), use_default_offset=False,
        scale=.5, noise_std=0., delay_reset_first_step_no_delay=True,
    ).build(env)
    return term, data, groups


@pytest.mark.parametrize("warm_buffer", [False, True])
def test_first_step_handles_reordered_three_channel_groups_and_uninitialized_queues(warm_buffer):
    from mjlab.actuator.actuator import delay_command

    term, data, groups = _three_channel_action()
    assert len(term._delay_reset_groups) == 2
    if warm_buffer:
        for group in groups:
            group.delay_buffer.append(torch.zeros(2, group.target_ids.numel(), 3))
            group.delay_buffer.compute()
            group.delay_buffer.set_lags(torch.full((2,), 4))
    term.reset()
    # Nonzero velocity/effort channels must retain the physical command too.
    data.joint_vel_target[:] = torch.tensor([1., 2., 3.])
    data.joint_effort_target[:] = torch.tensor([4., 5., 6.])
    action0 = torch.tensor([[.2, .4, .6], [-.2, -.4, -.6]])
    term.process_actions(action0)
    clocks = [g.delay_buffer._step_count.clone() for g in groups]
    term.apply_actions()
    for group, clock in zip(groups, clocks, strict=True):
        buffer = group.delay_buffer
        torch.testing.assert_close(buffer._step_count, clock, rtol=0, atol=0)
        fields = [getattr(data, attr)[:, group.target_ids] for attr in
                  ("joint_pos_target", "joint_vel_target", "joint_effort_target")]
        expected = torch.stack(fields, dim=-1)
        if warm_buffer:
            torch.testing.assert_close(buffer._buffer.buffer, expected[:, None].expand(-1, 5, -1, -1),
                                       rtol=0, atol=0)
        else:
            assert not buffer.is_initialized
        for _ in range(4):
            output = torch.stack(delay_command(buffer, *fields), dim=-1)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
        buffer.set_lags(torch.full((2,), 4))
        # The four first-step appends populate a complete a0 history, so even a
        # previously uninitialized queue delays every substep of a1 at lag4.
        fields1 = [field + 10 for field in fields]
        for _ in range(4):
            output = torch.stack(delay_command(buffer, *fields1), dim=-1)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_first_step_rejects_fused_queue_partly_owned_by_another_action():
    with pytest.raises(ValueError, match="own every joint"):
        _three_channel_action(owned=(0, 1))


@pytest.mark.slow
@NEEDS_CUDA_DATA
def test_explicitly_disabled_delay_has_no_queue_and_preserves_first_action_rate():
    env = make_env(SuperMimicEnvCfgArgs(
        npz_files=[str(NPZ)], robot="khan_mini", num_envs=4,
        actuator_delay=False, delay_reset_first_step_no_delay=True,
        merge_short_tail_bins=True,
        reset_noise=False, obs_noise=False, action_noise_std=0., seed=37,
    ), device="cuda:0")
    try:
        command = env.command_manager.get_term("motion")
        command.forced_frame[:] = command.library.clip_start[0] + 20
        env.reset()
        robot = env.scene["robot"]
        context = env.recorder_manager.get_term("replay").ctx
        assert context.delay_buffer is None
        assert not env.action_manager.get_term("joint_pos")._delay_reset_groups
        assert all(not actuator.has_delay and actuator._delay_buffer is None
                   for actuator in robot.actuators)
        assert context.read().shape == (4, 91)
        assert torch.count_nonzero(context.read()[:, -1]) == 0
        assert command.sampler_contract["version"] == 3
        state = StateIO(env).read()
        torch.testing.assert_close(state[:, 2]-command.root_pos_w[:, 2],
                                   torch.full((4,), .05, device=env.device), atol=2e-6, rtol=0)

        action = torch.full((4, 27), .2, device=env.device)
        env.action_manager.process_action(action)
        env.action_manager.apply_action()
        env.scene.write_data_to_sim()
        torch.testing.assert_close(env.sim.data.ctrl[:, context.ctrl_ids],
                                   robot.data.joint_pos_target, rtol=0, atol=0)
        # Reset again: the following step is the true episode boundary, with
        # raw previous action zero, rather than an arbitrary imagination t0.
        env.reset()
        env.step(action)
        expected_rate = -.002 * action.square().sum(-1)
        torch.testing.assert_close(env.reward_manager.get_term("action_rate_l2"),
                                   expected_rate, atol=1e-7, rtol=1e-6)
        assert context.delay_buffer is None
        assert torch.count_nonzero(context.read()[:, -1]) == 0
    finally:
        env.close()


@pytest.mark.slow
@NEEDS_CUDA_DATA
def test_fresh_reset_targets_bypass_first_control_step_then_resume_lags():
    from mjlab.envs import ManagerBasedRlEnv

    cfg = make_env_cfg(SuperMimicEnvCfgArgs(
        npz_files=[str(NPZ)], robot="khan_mini", num_envs=8, replay_capacity=32,
        actuator_delay=True, delay_reset_first_step_no_delay=True,
        action_noise_std=.11, obs_noise=False, reset_noise=False,
        domain_randomization=True, sampling_mode="start", seed=37,
    ))
    cfg.actions["joint_pos"].clip = {".*": (-.13, .13)}
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        command = env.command_manager.get_term("motion")
        command.forced_frame[:] = command.library.clip_start[0] + 20
        env.reset()
        robot = env.scene["robot"]
        term = env.action_manager.get_term("joint_pos")
        group, = robot._builtin_group._delayed_groups
        buffer = group.delay_buffer
        assert len(term._delay_reset_groups) == 1  # 27 aliased actuators, one queue.
        assert torch.count_nonzero(robot.data.joint_pos) > 0
        lag = torch.tensor([0, 1, 2, 3, 4, 0, 2, 4], device=env.device)
        buffer.set_lags(lag)
        bias = torch.linspace(-.02, .03, 27, device=env.device).expand(8, -1)
        robot.data.encoder_bias[:] = bias
        action0 = torch.linspace(-.6, .6, 27, device=env.device).expand(8, -1)
        env.action_manager.process_action(action0)
        torch.testing.assert_close(term.raw_action, action0, rtol=0, atol=0)
        assert torch.count_nonzero(term.executed_raw - action0) > 0
        processed0 = (term.executed_raw * term._scale + term._offset).clamp(-.13, .13)
        expected0 = processed0 - bias[:, term._target_ids]
        expected0 = expected0[:, group.target_ids]
        pointer = buffer._buffer._pointer
        clock = buffer._step_count.clone()
        xi_before = env.recorder_manager.get_term("replay").ctx.read().clone()

        # Priming alone must not advance any queue or delay sampling clock.
        env.action_manager.apply_action()
        assert buffer._buffer._pointer == pointer
        torch.testing.assert_close(buffer._step_count, clock, rtol=0, atol=0)
        torch.testing.assert_close(buffer.current_lags, lag, rtol=0, atol=0)
        torch.testing.assert_close(buffer._buffer.buffer,
                                   expected0[:, None].expand(-1, 5, -1), rtol=0, atol=0)
        torch.testing.assert_close(env.recorder_manager.get_term("replay").ctx.read(),
                                   xi_before, rtol=0, atol=0)
        assert torch.count_nonzero(env.action_manager.prev_action) == 0
        for _ in range(4):
            env.action_manager.apply_action()
            env.scene.write_data_to_sim()
            torch.testing.assert_close(env.sim.data.ctrl[:, group.ctrl_ids], expected0, rtol=0, atol=0)
            env.sim.step()
            env.scene.update(dt=env.physics_dt)

        # At a1, exactly lag physics substeps still execute a0 (0..4 substeps).
        env.action_manager.process_action(-action0)
        expected1 = ((term.executed_raw * term._scale + term._offset).clamp(-.13, .13)
                     - bias[:, term._target_ids])[:, group.target_ids]
        for substep in range(4):
            env.action_manager.apply_action()
            env.scene.write_data_to_sim()
            expected = torch.where((lag > substep)[:, None], expected0, expected1)
            torch.testing.assert_close(env.sim.data.ctrl[:, group.ctrl_ids], expected, rtol=0, atol=0)
        torch.testing.assert_close(env.action_manager.prev_action, action0, rtol=0, atol=0)

        # Partial reset: queue rows of the continuing episodes must survive
        # priming byte-for-byte, with no clock/lag mutation in any environment.
        reset_ids = torch.tensor([0, 2, 4], device=env.device)
        env.reset(env_ids=reset_ids)
        buffer.set_lags(lag)
        robot.data.encoder_bias[:] = bias
        keep = torch.ones(8, dtype=torch.bool, device=env.device)
        keep[reset_ids] = False
        history = buffer._buffer.buffer.clone()
        lengths = buffer._buffer.current_length.clone()
        clock = buffer._step_count.clone()
        pointer = buffer._buffer._pointer
        env.action_manager.process_action(action0 * .3)
        env.action_manager.apply_action()
        expected2 = robot.data.joint_pos_target[:, group.target_ids].clone()
        torch.testing.assert_close(buffer._buffer.buffer[keep], history[keep], rtol=0, atol=0)
        torch.testing.assert_close(buffer._buffer.current_length[keep], lengths[keep], rtol=0, atol=0)
        torch.testing.assert_close(buffer._buffer.buffer[reset_ids],
                                   expected2[reset_ids, None].expand(-1, 5, -1), rtol=0, atol=0)
        torch.testing.assert_close(buffer._step_count, clock, rtol=0, atol=0)
        torch.testing.assert_close(buffer.current_lags, lag, rtol=0, atol=0)
        assert buffer._buffer._pointer == pointer
        env.scene.write_data_to_sim()
        # Continuing rows still execute their historical commands at lag>0.
        expected_keep = history[keep, (5-lag[keep]).clamp_max(4)]
        expected_keep = torch.where((lag[keep] == 0)[:, None], expected2[keep], expected_keep)
        torch.testing.assert_close(env.sim.data.ctrl[:, group.ctrl_ids][keep], expected_keep, rtol=0, atol=0)
        torch.testing.assert_close(env.sim.data.ctrl[:, group.ctrl_ids][reset_ids], expected2[reset_ids],
                                   rtol=0, atol=0)

        # A real env.step still charges exactly the existing raw 0→a0 rate.
        env.reset()
        action = torch.full_like(action0, .2)
        env.step(action)
        torch.testing.assert_close(env.reward_manager.get_term("action_rate_l2"),
                                   -.002 * action.square().sum(-1), rtol=1e-6, atol=1e-7)
    finally:
        env.close()


@pytest.mark.parametrize("max_ms,max_lag", [(0.0, 0), (5.0, 1), (10.0, 2), (20.0, 4)])
def test_delay_configuration_matches_morphmimic_physics_steps(max_ms, max_lag):
    cfg = _cfg(actuator_delay_max_ms=max_ms)
    assert cfg.sim.mujoco.timestep == 0.005
    assert cfg.decimation == 4
    assert len(_actuators(cfg)) == 27
    expected = (0, max_lag, 0.0, 2**40, False)
    for actuator in _actuators(cfg):
        assert tuple(getattr(actuator, name) for name in DELAY_FIELDS) == expected
    assert cfg.commands["motion"].reset_root_z_offset == 0.05


@pytest.mark.parametrize("settings", [
    {"domain_randomization": False},
    {"actuator_delay": False},
    {"actuator_delay": True, "domain_randomization": False},
])
def test_nominal_or_explicitly_disabled_delay_keeps_reset_clearance(settings):
    cfg = _cfg(**settings)
    for actuator in _actuators(cfg):
        assert actuator.delay_min_lag == actuator.delay_max_lag == 0
    # Nominal evaluation shares the same spawn clearance as training.
    assert cfg.commands["motion"].reset_root_z_offset == 0.05


def test_g1_delay_and_reset_defaults_remain_unchanged():
    spec = get_robot_spec("g1")
    cfg = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=["unused.npz"], robot="g1"))
    baseline = spec.entity_cfg().articulation.actuators
    assert not spec.default_actuator_delay
    assert spec.default_reset_root_z_offset == 0.0
    assert cfg.commands["motion"].reset_root_z_offset == 0.0
    assert len(_actuators(cfg)) == len(baseline)
    for actual, original in zip(_actuators(cfg), baseline, strict=True):
        assert actual.delay_min_lag == actual.delay_max_lag == 0
        assert tuple(getattr(actual, name) for name in DELAY_FIELDS) == tuple(
            getattr(original, name) for name in DELAY_FIELDS
        )


@pytest.mark.parametrize("max_ms", [-5.0, 2.5, 21.0, float("nan"), float("inf")])
def test_invalid_delay_configuration_is_rejected(max_ms):
    with pytest.raises(ValueError, match="5 ms physics step"):
        _cfg(actuator_delay_max_ms=max_ms)


@pytest.mark.parametrize("offset", [-0.001, float("nan"), float("inf")])
def test_invalid_root_offset_configuration_is_rejected(offset):
    with pytest.raises(ValueError, match="reset_root_z_offset"):
        _cfg(reset_root_z_offset=offset)


def test_explicit_reset_offset_and_implicit_world_model_contract():
    cfg = _cfg(reset_root_z_offset=0.0)
    assert cfg.commands["motion"].reset_root_z_offset == 0.0
    contract = robot_contract("khan_mini")
    assert contract["version"] == 6
    schema = contract["simulation_schema"]
    assert schema["actuator_delay"] and schema["actuator_delay_enabled_in_training"]
    assert schema["actuator_delay_max_ms"] == 20.0
    assert schema["reset_root_z_offset"] == 0.05
    assert schema["reference_height_offset"] == 0.0
    assert schema["xi_dim"] == 91 and schema["world_model_input_dim"] == 185
    assert not schema["world_model_delay_queue"] and schema["delay_in_xi"]
    nominal = robot_contract("khan_mini", env_config={"domain_randomization": False})
    assert not nominal["simulation_schema"]["actuator_delay_enabled_in_training"]
    assert nominal["simulation_schema"]["reset_root_z_offset"] == 0.05


@pytest.fixture(scope="module")
def delayed_khan():
    if not torch.cuda.is_available() or not NPZ.is_file():
        pytest.skip("needs CUDA and converted Khan LAFAN1")
    env = make_env(SuperMimicEnvCfgArgs(
        npz_files=[str(NPZ)], robot="khan_mini", num_envs=8, replay_capacity=32,
        action_noise_std=0.0, obs_noise=False, reset_noise=False,
        domain_randomization=True, sampling_mode="start", seed=37,
    ), device="cuda:0")
    yield env
    env.close()


@pytest.mark.slow
@NEEDS_CUDA_DATA
def test_real_fused_delay_is_held_and_partial_reset_resamples(delayed_khan):
    env = delayed_khan
    env.reset(seed=37)
    robot = env.scene["robot"]
    groups = robot._builtin_group._delayed_groups
    assert len(groups) == 1
    group = groups[0]
    buffer = group.delay_buffer
    assert buffer is not None and buffer.per_env
    assert buffer.min_lag == 0 and buffer.max_lag == 4
    assert buffer.update_period == 2**40 and not buffer.per_env_phase
    assert group.ctrl_ids.numel() == env.action_manager.total_action_dim == 27
    assert all(actuator._delay_buffer is buffer for actuator in robot.actuators)
    lags = buffer.current_lags.clone()
    assert lags.shape == (8,) and ((lags >= 0) & (lags <= 4)).all()
    # Match MorphMimic's initial pending target: robot.reset clears physical
    # targets to zero, rather than priming the queue with the reference posture.
    torch.testing.assert_close(buffer.peek(), torch.zeros_like(buffer.peek()), rtol=0, atol=0)

    # Write through the entity's actual fused control path at physics cadence.
    # Constant values across joints make the expected delayed ctrl independent
    # of actuator/joint ordering, and avoid integrating an artificial ramp.
    for target in range(1, 9):
        robot.data.joint_pos_target[:] = float(target)
        env.scene.write_data_to_sim()
        torch.testing.assert_close(buffer.current_lags, lags, rtol=0, atol=0)
        expected = (target - lags).clamp_min(0).float()[:, None].expand(-1, 27)
        torch.testing.assert_close(env.sim.data.ctrl[:, group.ctrl_ids], expected, rtol=0, atol=0)

    reset_ids = torch.tensor([1, 3, 6], device=env.device)
    keep = torch.ones(8, dtype=torch.bool, device=env.device)
    keep[reset_ids] = False
    ever_changed = torch.zeros(len(reset_ids), dtype=torch.bool, device=env.device)
    for _ in range(6):
        env.reset(env_ids=reset_ids)
        now = buffer.current_lags.clone()
        assert ((now >= 0) & (now <= 4)).all()
        torch.testing.assert_close(now[keep], lags[keep], rtol=0, atol=0)
        torch.testing.assert_close(
            env.sim.data.ctrl[reset_ids[:, None], group.ctrl_ids[None, :]],
            torch.zeros(len(reset_ids), 27, device=env.device), rtol=0, atol=0,
        )
        ever_changed |= now[reset_ids] != lags[reset_ids]
    # A fresh draw may equal its old value; check repeated resets rather than
    # demanding a different lag on every individual episode boundary.
    assert ever_changed.all()


def _snapshot_reference(library, frames):
    ref = library.ref(frames)
    return tuple(getattr(ref, name).clone() for name in (
        "root_pos_w", "root_quat_w", "joint_pos", "joint_vel",
    ))


@pytest.mark.slow
@NEEDS_CUDA_DATA
def test_both_reset_paths_lift_only_actual_root_and_keep_xi_unchanged(delayed_khan):
    env = delayed_khan
    cmd = env.command_manager.get_term("motion")
    library = cmd.library
    frames = library.clip_start[0].expand(8) + 20
    before = _snapshot_reference(library, frames)
    cmd.forced_frame[:] = frames
    try:
        env.reset()
        state = StateIO(env).read()
        expected = before[0].clone()
        expected[:, 2] += 0.05
        torch.testing.assert_close(state[:, :3], expected, rtol=0, atol=2e-6)
        torch.testing.assert_close(state[:, 3:7], before[1], rtol=0, atol=2e-6)
        torch.testing.assert_close(cmd.frame, frames, rtol=0, atol=0)
        for actual, original in zip(_snapshot_reference(library, frames), before, strict=True):
            torch.testing.assert_close(actual, original, rtol=0, atol=0)

        reset_ids = torch.tensor([0, 2, 5], device=env.device)
        new_frames = frames[reset_ids] + torch.tensor([3, 7, 11], device=env.device)
        before_direct = _snapshot_reference(library, new_frames)
        state_before = state.clone()
        cmd.reset_to_frame(reset_ids, new_frames)
        state_after = StateIO(env).read()
        expected_direct = before_direct[0].clone()
        expected_direct[:, 2] += 0.05
        torch.testing.assert_close(state_after[reset_ids, :3], expected_direct, rtol=0, atol=2e-6)
        keep = torch.ones(8, dtype=torch.bool, device=env.device)
        keep[reset_ids] = False
        torch.testing.assert_close(state_after[keep], state_before[keep], rtol=0, atol=0)
        for actual, original in zip(
            _snapshot_reference(library, new_frames), before_direct, strict=True,
        ):
            torch.testing.assert_close(actual, original, rtol=0, atol=0)

        # The queue remains implicit, but xi now includes its held lag parameter.
        xi = env.recorder_manager.get_term("replay").ctx.read()
        assert xi.shape == (8, 91)
        spec = StateSpec(StateSpecCfg(context_layout="khan_dr_v1"), device=env.device, num_joints=27)
        assert state_after.shape == (8, spec.state_dim) == (8, 67)
        dynamics = DynamicsMLP(
            feat_dim=spec.feat_dim, act_dim=27, xi_dim=spec.xi_dim,
            hidden=(8,), out_dim=spec.delta_dim,
        ).to(env.device)
        assert dynamics.mlp.net[0].in_features == 185
        assert dynamics(spec.canonical(state_after), torch.zeros(8, 27, device=env.device),
                        spec.normalize_xi(xi)).shape == (8, 66)
    finally:
        cmd.forced_frame.fill_(-1)

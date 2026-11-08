"""Khan's episode-held actuator delay and actual-root reset clearance.

The bounded CUDA checks exercise mjlab's real fused actuator controls without
training a network.  Run the configuration checks alone with ``-m 'not slow'``.
"""

from pathlib import Path

import pytest
import torch

from supermimic.envs import SuperMimicEnvCfgArgs, make_env, make_env_cfg
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


@pytest.mark.slow
@NEEDS_CUDA_DATA
def test_new_training_has_no_delay_queue_and_preserves_first_action_rate():
    from supermimic.config import TrainConfig, resolve_training_observations

    training = TrainConfig(robot="khan_mini")
    resolve_training_observations(training)
    env = make_env(SuperMimicEnvCfgArgs(
        npz_files=[str(NPZ)], robot="khan_mini", num_envs=4,
        actuator_delay=training.env.actuator_delay,
        merge_short_tail_bins=training.env.merge_short_tail_bins,
        reset_noise=False, obs_noise=False, action_noise_std=0., seed=37,
    ), device="cuda:0")
    try:
        command = env.command_manager.get_term("motion")
        command.forced_frame[:] = command.library.clip_start[0] + 20
        env.reset()
        robot = env.scene["robot"]
        context = env.recorder_manager.get_term("replay").ctx
        assert context.delay_buffer is None
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

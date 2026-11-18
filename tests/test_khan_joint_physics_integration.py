"""Bounded physical checks of fixed joint friction, knee limits and action labels.

Two simulated robots, eight control steps per condition; no learner or optimizer.
"""

from pathlib import Path

import mujoco
import pytest
import torch
from mujoco_warp import OverflowType

from supermimic.config import TrainConfig, resolve_training_observations
from supermimic.envs import SuperMimicEnvCfgArgs, make_env, make_env_cfg
from supermimic.envs.state_io import StateIO
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.robots.registry import get_robot_spec
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.wm_trainer import reference_at
from supermimic.world_model.world_model import WorldModel, WorldModelCfg

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data/npz/khan_mini/lafan1/walk1_subject1.npz"
NEEDS_CUDA_DATA = pytest.mark.skipif(
    not torch.cuda.is_available() or not NPZ.is_file(),
    reason="needs CUDA and converted Khan LAFAN1",
)
KNEES = ("Left_knee_joint", "Right_knee_joint")


def _assert_fixed_friction(env):
    """Read the live Warp model, including broadcast and per-world layouts."""
    model = env.sim.mj_model
    actual = env.sim.model.dof_frictionloss[...]
    if actual.ndim == 1:
        actual = actual.unsqueeze(0)
    assert actual.shape[-1] == model.nv
    hinge_dofs = []
    free_dofs = []
    for j in range(model.njnt):
        start = int(model.jnt_dofadr[j])
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            hinge_dofs.append(start)
        elif model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            free_dofs.extend(range(start, start + 6))
    assert len(hinge_dofs) == 27 and len(free_dofs) == 6
    torch.testing.assert_close(actual[:, hinge_dofs], torch.full_like(actual[:, hinge_dofs], .1),
                               rtol=0, atol=0)
    torch.testing.assert_close(actual[:, free_dofs], torch.zeros_like(actual[:, free_dofs]),
                               rtol=0, atol=0)
    # Iteration-budget flags are distinct from structural buffer overflows.
    capacity_mask = int(OverflowType.ALL & ~(OverflowType.ITERATIONS | OverflowType.LS_ITERATIONS))
    bits = env.sim.data.overflow[...]
    assert not (bits & capacity_mask).any(), f"MuJoCo Warp capacity overflow: {bits.tolist()}"


@pytest.mark.parametrize(("robot", "expected_std"), [("khan_mini", 0.), ("g1", .1)])
def test_collectors_agree_on_default_action_noise(robot, expected_std):
    training = TrainConfig(robot=robot)
    resolve_training_observations(training)
    env_cfg = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=["unused.npz"], robot=robot))
    assert training.env.action_noise_std == env_cfg.actions["joint_pos"].noise_std == expected_std


@pytest.mark.slow
@NEEDS_CUDA_DATA
@pytest.mark.parametrize(("randomized", "noise_std"), [
    pytest.param(True, None, id="clean_training_default"),
    pytest.param(True, .1, id="legacy_training_noise"),
    pytest.param(False, 0., id="nominal_eval"),
])
def test_live_friction_knee_penalty_and_executed_action_labels(randomized, noise_std, record_property):
    noise_overrides = ({} if noise_std is None else
                       {"action_noise_std": noise_std, "disable_neck_action_noise": True})
    env = make_env(SuperMimicEnvCfgArgs(
        npz_files=[str(NPZ)], robot="khan_mini", context_layout="khan_dr_v3",
        num_envs=2, replay_capacity=32, sampling_mode="start", seed=631,
        actuator_delay=False, domain_randomization=randomized,
        reset_noise=randomized, obs_noise=randomized,
        **noise_overrides,
    ), device="cuda:0")
    try:
        _assert_fixed_friction(env)  # Startup DR must not overwrite dry friction.
        env.reset()
        robot = env.scene["robot"]
        spec = get_robot_spec("khan_mini")
        command = env.command_manager.get_term("motion")
        recorder = env.recorder_manager.get_term("replay")
        action = env.action_manager.get_term("joint_pos")
        io = StateIO(env)
        assert action.noise_std == (noise_std or 0.)
        assert bool(robot.data.encoder_bias.any()) is randomized
        assert tuple(robot.joint_names) == tuple(action.target_names) == spec.joint_names
        assert recorder.ctx.read().shape == (2, 101)
        knee = [spec.joint_names.index(name) for name in KNEES]
        neck = [spec.joint_names.index(name) for name in spec.neck_joint_names]
        other = [i for i in range(27) if i not in neck]
        torch.testing.assert_close(robot.data.soft_joint_pos_limits[:, knee, 0],
                                   torch.full_like(robot.data.soft_joint_pos_limits[:, knee, 0], -.03),
                                   rtol=0, atol=2e-7)
        heads = AnalyticHeads(
            G1Kinematics(spec.load_model(), device=env.device), spec.tracked_bodies,
            spec.anchor_body, spec.root_body, robot.data.soft_joint_pos_limits[0].clone(),
            robot.data.default_joint_pos[0].clone(),
        )
        # Same affine WM input mapping as Trainer, but a tiny untrained network.
        wm = WorldModel(WorldModelCfg(
            hidden=(8,), input_normalization=True, termination_mode="analytic",
            state_spec=StateSpecCfg(context_layout="khan_dr_v3"),
        ), heads, env.device,
            action_scale=torch.as_tensor(action._scale, device=env.device).expand_as(
                robot.data.default_joint_pos)[0],
            action_offset=torch.as_tensor(action._offset, device=env.device).expand_as(
                robot.data.default_joint_pos)[0],
        )
        assert wm.spec.xi_dim == 101
        rows = torch.arange(env.num_envs, device=env.device)
        clean = torch.linspace(-.04, .04, 54, device=env.device).reshape(2, 27)
        observed_bits, max_nefc = 0, 0
        for step in range(8):
            if step == 3:
                env.reset(env_ids=rows[:1])
                _assert_fixed_friction(env)
            ptr = recorder.buffer.ptr.clone()
            obs, reward, _, _, _ = env.step(clean)
            _assert_fixed_friction(env)
            for bits in env.sim.data.overflow[...].tolist():
                observed_bits |= bits
            max_nefc = max(max_nefc, int(env.sim.data.nefc[...].max()))
            assert torch.isfinite(reward).all() and torch.isfinite(obs["policy"]).all()
            logged = recorder.buffer.a_exec[rows, ptr]
            torch.testing.assert_close(recorder.buffer.a[rows, ptr], clean, rtol=0, atol=0)
            torch.testing.assert_close(recorder.last_step_actions, logged, rtol=0, atol=0)
            torch.testing.assert_close(logged[:, neck], clean[:, neck], rtol=0, atol=0)
            if noise_std:
                assert (logged[:, other] != clean[:, other]).all()
            else:
                torch.testing.assert_close(logged, clean, rtol=0, atol=0)
            held = ~env.reset_buf
            assert held.any(), "bounded action check needs a non-reset physical row"
            torch.testing.assert_close(action.executed_raw[held], logged[held], rtol=0, atol=0)
            torch.testing.assert_close(action.raw_action[held], clean[held], rtol=0, atol=0)
            torch.testing.assert_close(
                wm.proc_action(recorder.last_step_actions)[held],
                robot.data.joint_pos_target[held] + robot.data.encoder_bias[held],
                rtol=0, atol=2e-7,
            )
            assert recorder.ctx.read().shape == (2, 101)

        record_property("physical_steps", 8)
        record_property("max_nefc", max_nefc)
        record_property("overflow_bits_or", observed_bits)

        # Compare actual and imagined rewards at both sides of the new boundary.
        # Keep all other joints inside their soft ranges to isolate the knees.
        state = io.read()
        state[:, 7:34] = robot.data.soft_joint_pos_limits.mean(-1)
        state[:, 7 + torch.tensor(knee, device=env.device)] = torch.tensor(
            [[-.02, -.02], [-.04, -.04]], device=env.device,
        )
        state[:, 34:] = 0.
        io.write(rows, state)
        env.scene.update(env.step_dt)
        command._refresh_ref()
        command.update_relative_body_poses()
        actual_state = io.read()
        ref = reference_at(command.library, command.frame)
        imagined = heads.reward(actual_state, clean, clean, ref)["joint_limit"]
        term = env.reward_manager.get_term_cfg("joint_limit")
        physical = env.step_dt * term.weight * term.func(env, **term.params)
        expected = torch.tensor([0., -.0008], device=env.device)
        torch.testing.assert_close(physical, expected, rtol=0, atol=3e-8)
        torch.testing.assert_close(imagined, physical, rtol=0, atol=3e-8)
        _assert_fixed_friction(env)
    finally:
        env.close()

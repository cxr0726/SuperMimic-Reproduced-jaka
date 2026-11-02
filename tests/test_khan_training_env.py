"""Khan training-environment contract and bounded CUDA integration checks.

These tests exercise simulation, reference commands, analytic heads and replay;
they do not construct optimizers or train a policy/world model.
"""

from pathlib import Path

import pytest
import torch

from supermimic.envs import SuperMimicEnvCfgArgs, make_env, make_env_cfg
from supermimic.envs.mdp import terminations as sm_term
from supermimic.envs.state_io import StateIO
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.robots.registry import get_robot_spec
from supermimic.world_model.heads import (
    AnalyticHeads,
    analytic_observation,
    anchor_relative_obs,
    flatten_history_term_major,
    observation_terms,
)
from supermimic.world_model.wm_trainer import reference_at

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data/npz/khan_mini/lafan1/walk1_subject1.npz"
NEEDS_CUDA_DATA = pytest.mark.skipif(
    not torch.cuda.is_available() or not NPZ.is_file(), reason="needs CUDA and converted Khan LAFAN1"
)


def test_khan_config_uses_isaac_observation_and_extends_neck_tracking_reward():
    g1 = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=["unused.npz"]))
    khan = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=["unused.npz"], robot="khan_mini"))
    spec = get_robot_spec("khan_mini")
    assert len(spec.tracked_bodies) == 15
    assert spec.tracked_bodies[0] == spec.root_body == spec.anchor_body == "base_link"
    assert spec.tracked_bodies[7] == spec.com_body == "waist_yaw_Link"
    assert spec.tracked_bodies[-1] == "Neck_pitch_Link"
    assert spec.neck_joint_names == ("Neck_yaw_joint", "Neck_pitch_joint")
    assert khan.commands["motion"].body_names == spec.tracked_bodies
    assert khan.commands["motion"].root_body_name == spec.root_body
    assert khan.commands["motion"].anchor_body_name == spec.anchor_body
    assert khan.commands["motion"].reset_root_z_offset == 0.05
    assert g1.commands["motion"].reset_root_z_offset == 0.0
    assert khan.viewer.body_name == spec.anchor_body
    assert khan.recorders["replay"].params["com_body_name"] == spec.com_body
    assert set(khan.actions["joint_pos"].scale.values()) == {0.5}
    assert len(khan.actions["joint_pos"].scale) == 27
    assert tuple(khan.observations["policy"].terms) == (
        "motion_anchor_ori_b", "projected_gravity", "base_ang_vel", "joint_pos", "joint_vel", "actions"
    )
    assert tuple(khan.observations["goal"].terms) == ("command",)
    assert khan.commands["motion"].future_frames == 5
    assert g1.commands["motion"].future_frames == 10
    assert len(khan.rewards) == 10 and len(g1.rewards) == 8
    for name, original in g1.rewards.items():
        term = khan.rewards[name]
        assert term.func is original.func
        assert term.weight == original.weight
        assert term.params == original.params
    for name, weight, std in (("motion_neck_pos", 0.5, 0.3),
                              ("motion_neck_vel", 0.1, 1.0)):
        term = khan.rewards[name]
        assert term.weight == weight and term.params["std"] == std
        assert term.params["command_name"] == "motion"
        assert term.params["asset_cfg"].joint_names == spec.neck_joint_names
    for name, term in khan.terminations.items():
        original = g1.terminations[name]
        assert term.func is original.func
        assert term.time_out == original.time_out
        assert term.params == original.params
    assert khan.events["base_com"].params["asset_cfg"].body_names == (spec.com_body,)
    assert khan.events["base_com"].params["ranges"] == g1.events["base_com"].params["ranges"]
    for name in ("encoder_bias", "foot_friction"):
        assert khan.events[name].func is g1.events[name].func
        assert khan.events[name].mode == g1.events[name].mode == "reset"
    assert khan.events["encoder_bias"].params["bias_range"] == (-0.01, 0.01)
    assert khan.events["foot_friction"].params["ranges"] == (0.3, 1.2)
    assert khan.events["foot_friction"].params["asset_cfg"].geom_names == spec.foot_geom_regex
    assert khan.events["foot_friction"].params["shared_random"]
    assert khan.sim.mujoco == g1.sim.mujoco
    assert khan.decimation == g1.decimation == 4
    assert khan.commands["motion"].pose_range == g1.commands["motion"].pose_range
    assert khan.commands["motion"].velocity_range == g1.commands["motion"].velocity_range
    assert khan.commands["motion"].joint_position_range == (-0.1, 0.1)
    for actuator in khan.scene.entities["robot"].articulation.actuators:
        assert actuator.delay_min_lag == 0 and actuator.delay_max_lag == 4
        assert actuator.delay_update_period == 2**40
        assert actuator.delay_hold_prob == 0.0 and not actuator.delay_per_env_phase


@pytest.fixture(scope="module")
def setup_khan():
    if not torch.cuda.is_available() or not NPZ.is_file():
        pytest.skip("needs CUDA and converted Khan LAFAN1")
    env = make_env(SuperMimicEnvCfgArgs(
        npz_files=[str(NPZ)], robot="khan_mini", num_envs=8, replay_capacity=64,
        obs_noise=False, seed=19,
    ), device="cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    robot = env.scene["robot"]
    spec = get_robot_spec("khan_mini")
    heads = AnalyticHeads(
        G1Kinematics(spec.load_model(), device=env.device), spec.tracked_bodies,
        spec.anchor_body, spec.root_body, robot.data.soft_joint_pos_limits[0].clone(),
        robot.data.default_joint_pos[0].clone(),
    )
    yield env, robot, spec, heads, StateIO(env)
    env.close()


def _refresh(env, command):
    # Compare all implementations on the same post-integration instantaneous state.
    env.sim.forward()
    env.scene.update(env.step_dt)
    command._refresh_ref()
    command.update_relative_body_poses()


@NEEDS_CUDA_DATA
def test_khan_spaces_observation_reward_history_and_replay(setup_khan):
    env, robot, spec, heads, io = setup_khan
    obs, _ = env.reset()
    cmd = env.command_manager.get_term("motion")
    rec = env.recorder_manager.get_term("replay")
    assert obs["policy"].shape == (8, 465)
    assert obs["goal"].shape == (8, 155)
    assert env.action_manager.total_action_dim == 27
    assert io.read().shape == (8, 67) and rec.ctx.read().shape == (8, 31)
    assert tuple(robot.joint_names) == spec.joint_names
    assert cmd.cfg.body_names == spec.tracked_bodies
    assert tuple(robot.body_names[index] for index in cmd.body_indexes.tolist()) == spec.tracked_bodies
    assert robot.body_names[cmd.robot_anchor_body_index] == spec.anchor_body
    assert robot.body_names[cmd.robot_root_body_index] == spec.root_body
    hist = []
    age = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    full_history_checks = 0
    worst_rewards = {name: 0.0 for name in env.cfg.rewards}
    for _ in range(40):
        actions = 0.1 * torch.randn(env.num_envs, 27, device=env.device)
        obs, reward, _, _, _ = env.step(actions)
        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(obs["goal"]).all()
        assert torch.isfinite(reward).all() and reward.max() <= 0.112001
        _refresh(env, cmd)
        state, xi = io.read(), rec.ctx.read()
        current = analytic_observation(state, env.action_manager.action, xi,
                                        robot.data.default_joint_pos[0], observation_layout="isaac_khan")
        hist.append(current.clone())
        age = torch.where(env.reset_buf, torch.zeros_like(age), age + 1)
        start = 0
        for name, width in observation_terms(27, observation_layout="isaac_khan"):
            block = obs["policy"][:, 30 + start * 5:30 + (start + width) * 5].reshape(8, 5, width)
            torch.testing.assert_close(block[:, -1], current[:, start:start + width],
                                       rtol=0, atol=2e-5, msg=name)
            start += width
        if len(hist) >= 5 and (age >= 5).any():
            stable = age >= 5
            full = flatten_history_term_major(torch.stack(hist[-5:]), observation_layout="isaac_khan")
            torch.testing.assert_close(full[stable], obs["policy"][stable, 30:], rtol=0, atol=2e-5)
            full_history_checks += int(stable.sum())
        from supermimic.observation_layout import isaac_anchor_orientation, isaac_goal_command

        future = cmd.library.reference_window(cmd.frame, 5)
        torch.testing.assert_close(obs["goal"], isaac_goal_command(
            future.root_pos_w, future.root_quat_w, future.joint_pos), rtol=0, atol=1e-6)
        torch.testing.assert_close(obs["policy"][:, :30], isaac_anchor_orientation(
            cmd.robot_anchor_quat_w, future.root_quat_w), rtol=0, atol=2e-5)
        ref = reference_at(cmd.library, cmd.frame)
        ours = heads.reward(state, env.action_manager.action, env.action_manager.prev_action, ref)
        for name in env.cfg.rewards:
            # The manager owns resolved joint IDs; env.cfg retains unresolved
            # SceneEntityCfg selectors (slice(None) before resolution).
            term = env.reward_manager.get_term_cfg(name)
            if name in ("motion_neck_pos", "motion_neck_vel"):
                selected = term.params["asset_cfg"].joint_ids
                assert tuple(robot.joint_names[i] for i in selected) == spec.neck_joint_names
            theirs = term.weight * term.func(env, **term.params)
            error = (ours[name] / env.step_dt - theirs).abs().max().item()
            worst_rewards[name] = max(worst_rewards[name], error)
        features = heads.termination_features(state[:, :3], state[:, 3:7], ref)
        torch.testing.assert_close(features[:, 0], cmd.root_pos_w[:, 2] - cmd.robot_root_pos_w[:, 2],
                                   rtol=0, atol=2e-5)
        assert torch.equal(features[:, 0].abs() > 0.25, sm_term.bad_root_height(env, "motion"))
        assert torch.equal(features[:, 1].abs() > 0.8, sm_term.bad_root_tilt(env, "motion"))
    assert full_history_checks > 0, "No reset-free 5-frame observation segment was exercised"
    assert max(worst_rewards.values()) < 2e-5, worst_rewards
    buf = rec.buffer
    assert buf.size >= env.num_envs * 20
    window = buf.sample_windows(32, 8)
    assert window.states.shape == (9, 32, 67)
    assert window.a_exec.shape == window.a_clean.shape == (8, 32, 27)
    assert window.xi.shape == (32, 31)
    for value in (window.states, window.a_exec, window.a_clean, window.xi, window.reward):
        assert torch.isfinite(value).all()
    assert 0.05 < (window.a_exec - window.a_clean).std() < 0.2
    starts = buf.sample_starts(32, hist=5)
    assert starts.hist_states.shape == (5, 32, 67)
    assert starts.hist_actions.shape == (5, 32, 27)
    assert torch.equal(starts.hist_states[-1], starts.state)
    assert torch.isfinite(starts.hist_states).all()


@NEEDS_CUDA_DATA
def test_khan_domain_randomization_and_reset_noise(setup_khan):
    from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_apply_inverse, quat_inv, quat_mul

    env, robot, spec, _, io = setup_khan
    cmd = env.command_manager.get_term("motion")
    rec = env.recorder_manager.get_term("replay")
    frame = cmd.library.clip_start[0].expand(env.num_envs) + 100
    cmd.forced_frame[:] = frame
    try:
        env.reset()
        _refresh(env, cmd)
        before = rec.ctx.read()
        assert before.shape == (8, 31)
        assert ((before[:, 0] >= 0.3) & (before[:, 0] <= 1.2)).all()
        assert before[:, 0].std() > 0.01
        assert before[:, 1:28].abs().max() <= 0.010001
        limits = torch.tensor([0.025, 0.05, 0.05], device=env.device)
        assert (before[:, 28:].abs() <= limits + 1e-6).all()
        expected_com_id = robot.indexing.body_ids[robot.body_names.index(spec.com_body)]
        assert rec.ctx.com_body_global == int(expected_com_id)
        anchor_id = robot.indexing.body_ids[robot.body_names.index(spec.anchor_body)]
        assert int(expected_com_id) != int(anchor_id)
        local_ids, _ = robot.find_geoms(spec.foot_geom_regex)
        assert len(local_ids) == 2
        geom_ids = robot.indexing.geom_ids[local_ids]
        actual_friction = env.sim.model.geom_friction[:, geom_ids, 0]
        torch.testing.assert_close(actual_friction, before[:, :1].expand(-1, 2), rtol=0, atol=1e-6)

        # Reset writes the sampled/forced frame without advancing its target.
        # Remove the deterministic actual-root clearance before checking noise.
        source = cmd.library.ref(frame)
        state = io.read()
        delta_pos = state[:, :3] - source.root_pos_w
        delta_pos[:, 2] -= cmd.cfg.reset_root_z_offset
        assert (delta_pos.abs() <= torch.tensor([0.05, 0.05, 0.01], device=env.device) + 1e-6).all()
        delta_rot = quat_mul(state[:, 3:7], quat_inv(source.root_quat_w))
        rpy = torch.stack(euler_xyz_from_quat(delta_rot), -1)
        assert (rpy.abs() <= torch.tensor([0.1, 0.1, 0.2], device=env.device) + 1e-5).all()
        assert (state[:, 34:37] - source.root_lin_vel_w).abs().max() <= 0.500001
        assert (state[:, 36] - source.root_lin_vel_w[:, 2]).abs().max() <= 0.200001
        angular_world = robot.data.root_link_ang_vel_w
        assert ((angular_world - source.root_ang_vel_w).abs()
                <= torch.tensor([0.52, 0.52, 0.78], device=env.device) + 1e-5).all()
        soft = robot.data.soft_joint_pos_limits
        nominal_joint_pos = source.joint_pos.clamp(soft[..., 0], soft[..., 1])
        assert (state[:, 7:34] - nominal_joint_pos).abs().max() <= 0.100001
        torch.testing.assert_close(state[:, 40:67], source.joint_vel, rtol=0, atol=1e-6)
        # Body-frame state angular velocity and world-frame simulator API agree.
        torch.testing.assert_close(state[:, 37:40], quat_apply_inverse(state[:, 3:7], angular_world),
                                   rtol=0, atol=2e-5)
        env.reset()
        after = rec.ctx.read()
        assert not torch.equal(before[:, :28], after[:, :28])
        torch.testing.assert_close(before[:, 28:], after[:, 28:], rtol=0, atol=0)
    finally:
        cmd.forced_frame.fill_(-1)


@NEEDS_CUDA_DATA
def test_khan_root_terminations_and_state_io(setup_khan):
    from supermimic.kinematics.rotations import projected_gravity

    env, _, _, heads, io = setup_khan
    env.reset()
    cmd = env.command_manager.get_term("motion")
    ref = reference_at(cmd.library, cmd.frame)
    base = io.read()
    base[:, :3] = ref.root_pos_w
    base[:, 3:7] = ref.root_quat_w
    ids = torch.arange(env.num_envs, device=env.device)
    try:
        modified = base.clone()
        modified[0, 2] += 0.3
        ref_gravity_z = projected_gravity(ref.root_quat_w)[1, 2]
        modified[1, 3:7] = torch.tensor(
            [0.0, 1.0, 0.0, 0.0] if ref_gravity_z < 0 else [1.0, 0.0, 0.0, 0.0],
            device=env.device,
        )
        io.write(ids, modified)
        _refresh(env, cmd)
        torch.testing.assert_close(io.read(), modified, rtol=0, atol=2e-6)
        bad_height = sm_term.bad_root_height(env, "motion")
        bad_tilt = sm_term.bad_root_tilt(env, "motion")
        assert bad_height.tolist() == [True, False, False, False, False, False, False, False]
        assert bad_tilt.tolist() == [False, True, False, False, False, False, False, False]
        assert torch.equal(heads.terminated(io.read(), ref), bad_height | bad_tilt)
    finally:
        io.write(ids, base)
        _refresh(env, cmd)


@NEEDS_CUDA_DATA
@pytest.mark.parametrize("obs_noise", [False, True])
def test_khan_anchor_goal_matches_reference_pose_in_robot_base_frame(obs_noise):
    """The actual mjlab goal appends one 3+6 anchor pose, with matching noise."""
    env = make_env(SuperMimicEnvCfgArgs(
        npz_files=[str(NPZ)], robot="khan_mini", num_envs=4,
        observation_layout="supermimic",
        anchor_obs=True, obs_noise=obs_noise, reset_noise=False,
        domain_randomization=False, action_noise_std=0.0,
        sampling_mode="start", record_transitions=False, seed=29,
    ), device="cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        robot, spec = env.scene["robot"], get_robot_spec("khan_mini")
        heads = AnalyticHeads(
            G1Kinematics(spec.load_model(), device=env.device), spec.tracked_bodies,
            spec.anchor_body, spec.root_body, robot.data.soft_joint_pos_limits[0].clone(),
            robot.data.default_joint_pos[0].clone(),
        )
        io = StateIO(env)
        cmd = env.command_manager.get_term("motion")
        obs, _ = env.reset()
        assert len(spec.tracked_bodies) == 15
        assert spec.anchor_body == "base_link" and heads.anchor_index == 0
        assert spec.com_body == "waist_yaw_Link"
        assert spec.tracked_bodies[-1] == "Neck_pitch_Link"
        assert robot.body_names[cmd.robot_anchor_body_index] == spec.anchor_body
        assert tuple(env.cfg.observations["goal"].terms) == (
            "future_frames", "anchor_pos", "anchor_ori"
        )
        residuals = []
        for _ in range(8):
            assert obs["policy"].shape == (4, 870)
            assert obs["goal"].shape == (4, 649)
            assert torch.isfinite(obs["policy"]).all() and torch.isfinite(obs["goal"]).all()
            _refresh(env, cmd)
            bodies = heads.robot_bodies(io.read())
            ref = reference_at(cmd.library, cmd.frame)
            index = heads.anchor_index
            expected_anchor = anchor_relative_obs(
                bodies["pos"][:, index], bodies["quat"][:, index],
                ref.body_pos_w[:, index], ref.body_quat_w[:, index],
            )
            torch.testing.assert_close(obs["goal"][:, :640], cmd.library.goal_window(cmd.frame, 10),
                                       rtol=0, atol=1e-6)
            residual = obs["goal"][:, 640:] - expected_anchor
            if obs_noise:
                assert residual[:, :3].abs().max() <= 0.25003
                assert residual[:, 3:].abs().max() <= 0.05003
                residuals.append(residual.clone())
            else:
                torch.testing.assert_close(obs["goal"][:, 640:], expected_anchor,
                                           rtol=0, atol=3e-5)
            obs, reward, *_ = env.step(torch.zeros(4, 27, device=env.device))
            assert torch.isfinite(reward).all()
        if obs_noise:
            residual = torch.cat(residuals)
            assert residual[:, :3].std() > 0.02
            assert residual[:, 3:].std() > 0.01
    finally:
        env.close()

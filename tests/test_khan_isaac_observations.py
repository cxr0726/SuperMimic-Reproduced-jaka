"""Independent numerical checks of the Isaac Khan actor observation contract."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from supermimic.data.replay_buffer import ReplayBuffer, StartBatch
from supermimic.envs import SuperMimicEnvCfgArgs, make_env, make_env_cfg
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.envs.state_io import StateIO
from supermimic.imagination.compiled import REWARD_TERM_NAMES
from supermimic.imagination.policy_trainer import PolicyTrainer
from supermimic.imagination.rollout import imagine
from supermimic.observation_layout import (
    critic_reference_extras,
    goal_dimension,
    isaac_anchor_orientation,
    isaac_goal_command,
    policy_observation_dimension,
    resolve_observation_layout,
    resolve_observation_sizes,
)
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.world_model.heads import (
    AnalyticHeads,
    analytic_observation,
    flatten_history_term_major,
)

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data/npz/khan_mini/lafan1/walk1_subject1.npz"
NEEDS_CUDA_DATA = pytest.mark.skipif(
    not torch.cuda.is_available() or not NPZ.is_file(), reason="needs CUDA and Khan NPZ"
)


def wxyz(rotation):
    return np.roll(rotation.as_quat(), 1, axis=-1)


def test_layout_defaults_and_conflicting_legacy_flags():
    assert resolve_observation_layout("khan_mini") == "isaac_khan"
    assert resolve_observation_layout("g1") == "supermimic"
    assert resolve_observation_sizes("isaac_khan") == (5, 5)
    assert resolve_observation_sizes("supermimic") == (10, 10)
    assert goal_dimension("isaac_khan", 27, 5) == 155
    assert policy_observation_dimension("isaac_khan", 27, 5, 5) == 465
    assert goal_dimension("supermimic", 29, 10) == 680
    assert policy_observation_dimension("supermimic", 29, 10, 10) == 930
    for future, history, anchor in ((10, 5, False), (5, 10, False), (5, 5, True)):
        with pytest.raises(ValueError):
            resolve_observation_sizes("isaac_khan", future, history, anchor)
    with pytest.raises(ValueError):
        resolve_observation_layout("g1", "isaac_khan")


def test_goal_uses_full_current_reference_rotation_and_term_major_absolute_values():
    rng = np.random.default_rng(284)
    positions = rng.normal(size=(2, 5, 3)) + np.array([4.0, -3.0, 0.7])
    rotations = Rotation.from_euler("xyz", rng.normal(size=(10, 3)) * 0.4)
    quaternions = wxyz(rotations).reshape(2, 5, 4)
    joints = rng.normal(size=(2, 5, 27)) + 0.4  # Deliberately nonzero absolute angles.
    expected_diff = np.stack([
        rotations[5 * sample].inv().apply(positions[sample] - positions[sample, 0])
        for sample in range(2)
    ])
    expected = np.concatenate((expected_diff.reshape(2, 15), positions[..., 2],
                               joints.reshape(2, 135)), axis=-1)
    actual = isaac_goal_command(*(torch.tensor(value) for value in (positions, quaternions, joints)))
    assert actual.shape == (2, 155)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(actual[:, :3], 0)
    np.testing.assert_array_equal(actual[:, 20:], joints.reshape(2, 135))
    # Common XY translation cancels; absolute world height is retained.
    translated = positions + np.array([22.0, -31.0, 0.2])
    moved = isaac_goal_command(torch.tensor(translated), torch.tensor(quaternions), torch.tensor(joints))
    np.testing.assert_allclose(moved[:, :15], actual[:, :15], atol=1e-12)
    np.testing.assert_allclose(moved[:, 15:20], actual[:, 15:20] + 0.2, atol=1e-12)


def test_future_orientation_uses_robot_frame_and_row_flattened_two_columns_with_gradients():
    robot = Rotation.from_euler("xyz", [[0.3, -0.4, 0.8], [-0.2, 0.5, -0.6]])
    reference = Rotation.from_euler("xyz", np.arange(30).reshape(10, 3) * 0.027)
    expected = np.stack([
        (robot[sample].inv() * reference[sample * 5:(sample + 1) * 5]).as_matrix()[..., :2]
        .reshape(30) for sample in range(2)
    ])
    robot_q = torch.tensor(wxyz(robot), requires_grad=True)
    ref_q = torch.tensor(wxyz(reference).reshape(2, 5, 4))
    actual = isaac_anchor_orientation(robot_q, ref_q)
    np.testing.assert_allclose(actual.detach(), expected, rtol=0, atol=1e-12)
    (actual * torch.arange(30)).sum().backward()
    assert robot_q.grad is not None and torch.isfinite(robot_q.grad).all()
    assert robot_q.grad.abs().max() > 1
    # Rotating the actual robot changes prop orientation while goal stays reference-only.
    changed = isaac_anchor_orientation(torch.tensor(wxyz(Rotation.identity(2))), ref_q)
    assert not torch.allclose(actual, changed)


def test_critic_reference_extras_use_future_body_and_heading_frames_with_frame_major_order():
    rng = np.random.default_rng(736)
    angles = rng.uniform(-0.6, 0.6, (10, 3))
    angles[:, 2] += np.linspace(-1.0, 1.0, 10)
    rotation = Rotation.from_euler("xyz", angles)
    velocity = rng.normal(size=(10, 3))
    angular = rng.normal(size=(10, 3))
    joint_velocity = np.arange(270, dtype=np.float64).reshape(10, 27) * 0.013 - 1.2
    gravity = rotation.inv().apply(np.tile([0.0, 0.0, -1.0], (10, 1)))
    angular_body = rotation.inv().apply(angular)
    heading = Rotation.from_rotvec(np.column_stack((np.zeros((10, 2)), angles[:, 2])))
    velocity_heading = heading.inv().apply(velocity)
    expected = np.concatenate((joint_velocity, gravity, angular_body, velocity_heading), -1)
    inputs = (joint_velocity.reshape(2, 5, 27), wxyz(rotation).reshape(2, 5, 4),
              angular.reshape(2, 5, 3), velocity.reshape(2, 5, 3))
    actual = critic_reference_extras(*(torch.tensor(value) for value in inputs))
    assert actual.shape == (2, 180)
    np.testing.assert_allclose(actual, expected.reshape(2, 180), rtol=0, atol=1e-12)
    # Heading-frame linear velocity excludes root roll/pitch; angular velocity
    # and projected gravity must use the complete future root rotation instead.
    assert not np.allclose(velocity_heading, rotation.inv().apply(velocity))
    assert not np.allclose(angular_body, heading.inv().apply(angular))
    # A common rotation about world Z must not change any of these local inputs.
    yaw = Rotation.from_rotvec([0.0, 0.0, 1.1])
    moved = critic_reference_extras(
        torch.tensor(inputs[0]), torch.tensor(wxyz(yaw * rotation).reshape(2, 5, 4)),
        torch.tensor(yaw.apply(angular).reshape(2, 5, 3)),
        torch.tensor(yaw.apply(velocity).reshape(2, 5, 3)),
    )
    torch.testing.assert_close(actual, moved, rtol=0, atol=1e-12)


@pytest.mark.parametrize("compact", [False, True])
def test_reference_window_clamps_each_clip_and_includes_current_frame(compact):
    # Use distinct consecutive clip trajectories so crossing a boundary is observable.
    lib = MotionLibrary.__new__(MotionLibrary)
    lib.device, lib.dtype, lib.compact = torch.device("cpu"), torch.float64, compact
    lib._ref = torch.arange(10 * 67, dtype=torch.float64).reshape(10, 67)
    lib.frame_clip = torch.tensor([0] * 4 + [1] * 6)
    lib.clip_last = torch.tensor([3, 9])
    frames = torch.tensor([0, 2, 3, 7, 9])
    ref = lib.reference_window(frames, 5)
    indices = torch.tensor([[0, 1, 2, 3, 3], [2, 3, 3, 3, 3], [3] * 5,
                            [7, 8, 9, 9, 9], [9] * 5])
    torch.testing.assert_close(ref.root_pos_w, lib._ref[indices, :3])
    torch.testing.assert_close(ref.joint_pos, lib._ref[indices, 7:34])
    assert ref.root_quat_w.shape == (5, 5, 4)


def test_analytic_proprio_order_static_scales_and_history_order():
    rotation = Rotation.from_euler("xyz", [[0.23, -0.41, 0.62], [0.1, 0.3, -0.2]])
    state = torch.arange(2 * 67, dtype=torch.float64).reshape(2, 67) * 0.01
    state[:, 3:7] = torch.tensor(wxyz(rotation))
    action = torch.arange(54, dtype=torch.float64).reshape(2, 27) * 0.02
    xi = torch.zeros(2, 31, dtype=torch.float64)
    xi[:, 1:28] = torch.linspace(-0.01, 0.01, 27)
    default = torch.linspace(-0.3, 0.4, 27, dtype=torch.float64)
    obs = analytic_observation(state, action, xi, default, observation_layout="isaac_khan")
    gravity = torch.tensor(rotation.inv().apply(np.tile([0, 0, -1], (2, 1))))
    expected = torch.cat((gravity, state[:, 37:40] * 0.5,
                          state[:, 7:34] + xi[:, 1:28] - default,
                          state[:, 40:] * 0.2, action), dim=-1)
    torch.testing.assert_close(obs, expected, rtol=0, atol=1e-12)
    history = torch.stack([obs + time for time in range(5)])
    actual = flatten_history_term_major(history, observation_layout="isaac_khan")
    expected_flat = torch.cat([history[:, :, sl].transpose(0, 1).flatten(1) for sl in
                               (slice(0, 3), slice(3, 6), slice(6, 33), slice(33, 60), slice(60, 87))], -1)
    torch.testing.assert_close(actual, expected_flat, rtol=0, atol=0)


def test_noise_configuration_scales_after_corruption_and_excludes_orientation_history():
    cfg = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=["unused.npz"], robot="khan_mini"))
    policy = cfg.observations["policy"]
    assert tuple(policy.terms) == ("motion_anchor_ori_b", "projected_gravity", "base_ang_vel",
                                   "joint_pos", "joint_vel", "actions")
    assert policy.history_length in (None, 0)
    assert policy.terms["motion_anchor_ori_b"].history_length == 0
    for name in ("projected_gravity", "base_ang_vel", "joint_pos", "joint_vel", "actions"):
        assert policy.terms[name].history_length == 5
    expected = {"motion_anchor_ori_b": (0.05, None), "projected_gravity": (0.07, None),
                "base_ang_vel": (0.2, 0.5), "joint_pos": (0.01, None), "joint_vel": (0.5, 0.2)}
    for name, (noise, scale) in expected.items():
        term = policy.terms[name]
        assert term.noise.n_min == -noise and term.noise.n_max == noise
        assert term.scale in ((None, 1.0) if scale is None else (scale,))
    assert policy.terms["actions"].noise is None
    goal = cfg.observations["goal"]
    assert tuple(goal.terms) == ("command",)
    assert goal.terms["command"].noise is None
    assert not goal.enable_corruption, "Reference goals must stay clean even when actor sensors are noisy"


def test_replay_marks_backfilled_rows_so_reset_noise_can_be_shared():
    buffer = ReplayBuffer(1, 16, 67, 27, 31, "cpu")
    for i in range(3):
        buffer.add(torch.tensor([0]), torch.full((1, 67), float(i)), torch.zeros(1, 27),
                   torch.zeros(1, 27), torch.zeros(1, 31), torch.zeros(1), torch.zeros(1),
                   torch.tensor([i]), torch.zeros(1, dtype=torch.long),
                   torch.zeros(1, dtype=torch.bool), torch.tensor([i == 0]))
    buffer._sample_from_mask = lambda mask, n: (torch.zeros(n, dtype=torch.long),
                                               torch.full((n,), 2, dtype=torch.long))
    start = buffer.sample_starts(2, 5)
    assert start.history_valid.tolist() == [[False, False], [False, False], [True, True],
                                           [True, True], [True, True]]
    assert start.hist_states[:, 0, 0].tolist() == [0.0, 0.0, 0.0, 1.0, 2.0]


def _tiny_library():
    lib = MotionLibrary.__new__(MotionLibrary)
    lib.device, lib.dtype, lib.compact = torch.device("cpu"), torch.float32, False
    lib.default_joint_pos = torch.linspace(-0.1, 0.1, 27)
    lib._ref = torch.zeros(12, 67)
    lib._ref[:, :3] = torch.stack((torch.arange(12) * 0.01, torch.zeros(12), torch.full((12,), 0.65)), -1)
    lib._ref[:, 3:7] = torch.tensor(wxyz(Rotation.from_euler("xyz", np.arange(36).reshape(12, 3) * 0.01)))
    lib._ref[:, 7:34] = lib.default_joint_pos + torch.arange(12)[:, None] * 0.01
    lib._ref[:, 34:37] = torch.tensor([0.5, -0.3, 0.1]) + torch.arange(12)[:, None] * 0.02
    lib._ref[:, 37:40] = torch.tensor([0.2, -0.4, 0.3]) + torch.arange(12)[:, None] * 0.01
    lib._ref[:, 40:] = torch.linspace(-0.8, 0.9, 27) + torch.arange(12)[:, None] * 0.03
    lib.frame_clip, lib.clip_last = torch.zeros(12, dtype=torch.long), torch.tensor([11])
    lib.body_pos_w, lib.body_quat_w = lib._ref[:, None, :3], lib._ref[:, None, 3:7]
    lib.body_lin_vel_w = lib.body_ang_vel_w = torch.zeros(12, 1, 3)
    return lib


@pytest.mark.parametrize("noise", [False, True])
@pytest.mark.parametrize("action_mode", ["default", "reference_residual"])
@pytest.mark.parametrize("running_normalization", [False, True])
def test_imagination_builds_current_future_orientation_and_five_term_histories(
    noise, action_mode, running_normalization,
):
    torch.manual_seed(101)
    lib = _tiny_library()
    policy_args = ({"default_joint_pos": lib.default_joint_pos,
                    "action_scale": torch.full((27,), .5)}
                   if action_mode == "reference_residual" else {})
    policy = Policy(PolicyCfg(encoder_hidden=(12,), decoder_hidden=(12,), latent_dim=4,
                              history=5, future_frames=5, num_joints=27, goal_dim=155,
                              observation_layout="isaac_khan", num_tracked_bodies=1,
                              action_mode=action_mode, goal_noise_z=.01, goal_noise_joint_pos=.05,
                              running_normalization=running_normalization,
                              hidden_layer_norm=running_normalization,
                              hidden_activation="gelu" if running_normalization else "silu"),
                    **policy_args)
    if running_normalization:
        policy.update_normalization(torch.randn(7, 465) * 2 + 3,
                                    torch.randn(7, 155) * .3 + 2)
    buffers_before = {name: value.clone() for name, value in policy.named_buffers()}
    state = torch.zeros(2, 67)
    state[:, :3], state[:, 3:7] = lib._ref[:2, :3], lib._ref[:2, 3:7]
    state[:, 7:34] = lib.default_joint_pos
    history = state.repeat(5, 1, 1)
    # H=2 can end exactly on frame 11; the K=5 goal still exercises padding.
    start = StartBatch(state, torch.zeros(2, 31), torch.tensor([0, 9]), torch.zeros(2, 27),
                       history, torch.zeros(5, 2, 27),
                       history_valid=torch.tensor([[False, False], [False, False],
                                                   [True, True], [True, True], [True, True]]))
    heads = SimpleNamespace(
        fk=SimpleNamespace(nq=34),
        anchor_index=0, root_error_features=lambda state, ref: state[:, :4] * 0,
        robot_bodies=lambda state: {"pos": state[:, None, :3], "quat": state[:, None, 3:7]},
    )
    heads.critic_error_features = lambda state, ref: AnalyticHeads.critic_error_features(heads, state, ref)
    captured = []
    hook = policy.register_forward_pre_hook(lambda module, args: captured.append(args))
    encoder_inputs = []
    encoder_hook = policy.encoders["motion"].register_forward_pre_hook(
        lambda module, args: encoder_inputs.append(args[0].detach().clone()))
    decoder_outputs = []
    decoder_hook = policy.decoder.register_forward_hook(
        lambda module, args, output: decoder_outputs.append(output.detach().clone()))

    def transition(current, action, previous, context, ref):
        next_joint = current[:, 7:34] + action * 0.1
        # Make robot orientation depend on actions to exercise the future-orientation
        # gradient path on the next imagined step.
        quat = current[:, 3:7] + torch.cat((torch.zeros_like(action[:, :1]), action[:, :3] * 0.02), -1)
        quat = quat / quat.norm(dim=-1, keepdim=True)
        nxt = torch.cat((current[:, :3], quat, next_joint, current[:, 34:]), -1)
        reward = -(action - 0.2).square().mean(-1)
        prop = analytic_observation(nxt, action, context, lib.default_joint_pos,
                                    observation_layout="isaac_khan")
        return nxt, reward, reward[None].expand(len(REWARD_TERM_NAMES), -1), reward * 0 - 2, prop, nxt[:, :7]

    rollout = imagine(SimpleNamespace(heads=heads), policy, lib, start, 2, 5,
                      obs_noise=noise, step_fn=transition)
    hook.remove()
    encoder_hook.remove()
    decoder_hook.remove()
    assert len(captured) == 2 and captured[0][0].shape == (2, 465)
    assert rollout.goals.shape == (3, 2, 155)
    assert rollout.critic_goals.shape == (3, 2, 365)
    for time, frame in enumerate(rollout.frames):
        reference = lib.reference_window(frame, 5)
        expected_critic = torch.cat((
            isaac_goal_command(reference.root_pos_w, reference.root_quat_w, reference.joint_pos),
            isaac_anchor_orientation(rollout.states[time, :, 3:7], reference.root_quat_w),
            critic_reference_extras(reference.joint_vel, reference.root_quat_w,
                                    reference.root_ang_vel_w, reference.root_lin_vel_w),
        ), dim=-1)
        # Actor corruption must not enter privileged critic targets, including
        # its appended future-orientation information at every imagined step.
        torch.testing.assert_close(rollout.critic_goals[time], expected_critic, rtol=0, atol=1e-6)
        torch.testing.assert_close(rollout.goals[time], expected_critic[:, :155], rtol=0, atol=0)
        if time < 2:
            if action_mode == "reference_residual":
                expected_skip = (reference.joint_pos[:, 1] - lib.default_joint_pos) / .5
                torch.testing.assert_close(rollout.actions[time] - decoder_outputs[time],
                                           expected_skip, rtol=1e-6, atol=1e-6)
            else:
                torch.testing.assert_close(rollout.actions[time], decoder_outputs[time], rtol=0, atol=0)
            delta = encoder_inputs[time] - rollout.goals[time]
            assert torch.equal(delta[:, :15], torch.zeros_like(delta[:, :15]))
            assert delta[:, 15:20].abs().max() <= (.010001 if noise else 0.)
            assert delta[:, 20:].abs().max() <= (.050001 if noise else 0.)
            if noise:
                assert delta[:, 15:20].abs().max() > 0
                assert delta[:, 20:].abs().max() > .025
    if noise:
        assert not torch.equal(captured[0][0][:, :30], rollout.critic_goals[0, :, 155:185])
    trainer = SimpleNamespace(wm=SimpleNamespace(
        spec=SimpleNamespace(canonical=lambda state: state), heads=heads,
    ))
    _, value_goals, _, value_errors = PolicyTrainer._critic_inputs(trainer, rollout)
    assert value_goals is rollout.critic_goals
    assert value_errors.shape == (3, 2, 36)  # This synthetic fixture tracks only its root.
    torch.testing.assert_close(
        value_errors[..., :27], lib._ref[rollout.frames, 7:34] - rollout.states[..., 7:34],
    )
    torch.testing.assert_close(rollout.critic_reference.root_pos_w,
                               lib._ref[rollout.frames.reshape(-1), :3])
    orientation_gradients = torch.autograd.grad(
        (rollout.critic_goals[-1, :, 155:185] * torch.arange(30)).sum(),
        tuple(policy.parameters()), retain_graph=True, allow_unused=True,
    )
    assert any(gradient is not None and gradient.abs().sum() > 0 for gradient in orientation_gradients)
    assert all(gradient is None or torch.isfinite(gradient).all() for gradient in orientation_gradients)
    initial_obs, initial_goal = captured[0]
    future = lib.reference_window(start.frame, 5)
    ori = isaac_anchor_orientation(state[:, 3:7], future.root_quat_w)
    goal = isaac_goal_command(future.root_pos_w, future.root_quat_w, future.joint_pos)
    assert (initial_obs[:, :30] - ori).abs().max() <= (0.050001 if noise else 1e-6)
    torch.testing.assert_close(initial_goal, goal, rtol=0, atol=0)
    clean_proprio = analytic_observation(state, torch.zeros(2, 27), start.xi,
                                        lib.default_joint_pos, observation_layout="isaac_khan")
    offset, clean_offset = 30, 0
    for index, (width, half) in enumerate(((3, 0.07), (3, 0.1), (27, 0.01), (27, 0.1), (27, 0.0))):
        block = initial_obs[:, offset:offset + 5 * width].reshape(2, 5, width)
        torch.testing.assert_close(block[:, :3], block[:, :1].expand(-1, 3, -1), rtol=0, atol=0)
        residual = block - clean_proprio[:, None, clean_offset:clean_offset + width]
        assert residual.abs().max() <= half * int(noise) + 1e-6
        if noise and index in (1, 3):
            # With the original SuperMimic scales, sensor corruption reaches
            # +/-0.1 rather than the previous +/-0.05 and +/-0.025.
            assert residual.abs().max() > 0.06
        if noise and width == 3:
            assert not torch.equal(block[:, 2], block[:, 3])
        offset += 5 * width
        clean_offset += width
    (-rollout.rewards.mean()).backward()
    gradients = [parameter.grad for parameter in policy.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().sum() > 0 for gradient in gradients)
    for name, value in policy.named_buffers():
        torch.testing.assert_close(value, buffers_before[name], rtol=0, atol=0)


def test_isaac_orientation_compiles_with_autograd():
    reference = torch.tensor(wxyz(Rotation.from_euler("xyz", np.arange(15).reshape(5, 3) * 0.03)),
                             dtype=torch.float64)[None]
    root = torch.tensor(wxyz(Rotation.from_euler("xyz", [[0.2, -0.3, 0.4]])),
                        dtype=torch.float64, requires_grad=True)
    compiled = torch.compile(isaac_anchor_orientation, backend="aot_eager", fullgraph=True)
    outputs, gradients = [], []
    for function in (isaac_anchor_orientation, compiled):
        value = function(root, reference)
        gradient, = torch.autograd.grad((value * torch.arange(30)).sum(), root)
        outputs.append(value)
        gradients.append(gradient)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=1e-12)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=1e-12)


@NEEDS_CUDA_DATA
@pytest.mark.parametrize("noise", [False, True])
def test_actual_environment_isaac_observations_and_reset_history(noise):
    env = make_env(SuperMimicEnvCfgArgs(
        npz_files=[str(NPZ)], robot="khan_mini", num_envs=4, obs_noise=noise,
        reset_noise=False, domain_randomization=False, action_noise_std=0,
        sampling_mode="start", record_transitions=True, replay_capacity=32,
    ), device="cuda:0")
    try:
        obs, _ = env.reset()
        torch.backends.cuda.matmul.allow_tf32 = False
        assert obs["policy"].shape == (4, 465) and obs["goal"].shape == (4, 155)
        cmd, robot = env.command_manager.get_term("motion"), env.scene["robot"]
        assert cmd.cfg.future_frames == 5
        ref = cmd.library.reference_window(cmd.frame, 5)
        clean_goal = isaac_goal_command(
            ref.root_pos_w + env.scene.env_origins[:, None, :], ref.root_quat_w, ref.joint_pos,
        )
        torch.testing.assert_close(obs["goal"], clean_goal, rtol=0, atol=0)
        rotation = isaac_anchor_orientation(cmd.robot_anchor_quat_w, ref.root_quat_w)
        assert (obs["policy"][:, :30] - rotation).abs().max() <= (0.05001 if noise else 3e-6)
        io = StateIO(env)
        rec = env.recorder_manager.get_term("replay")
        clean = analytic_observation(io.read(), env.action_manager.action, rec.ctx.read(),
                                     robot.data.default_joint_pos[0], observation_layout="isaac_khan")
        offset, current_offset = 30, 0
        for width, half in ((3, 0.07), (3, 0.1), (27, 0.01), (27, 0.1), (27, 0.0)):
            history = obs["policy"][:, offset:offset + width * 5].reshape(4, 5, width)
            # A reset repeats the exact first noised observation, including its noise.
            torch.testing.assert_close(history, history[:, :1].expand_as(history), rtol=0, atol=0)
            residual = history[:, -1] - clean[:, current_offset:current_offset + width]
            assert residual.abs().max() <= (half if noise else 0) + 3e-5
            offset += 5 * width
            current_offset += width
        if noise:
            assert (obs["policy"][:, :30] - rotation).abs().max() > 0.005
    finally:
        env.close()

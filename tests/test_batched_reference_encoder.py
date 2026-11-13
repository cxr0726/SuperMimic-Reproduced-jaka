"""Reference batching preserves closed-loop actor gradients and noise semantics."""

import copy

import numpy as np
import pytest
import torch

from supermimic.data.khan_motion_convert import convert_clip
from supermimic.data.replay_buffer import StartBatch
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.envs.mdp.rewards import TrackingRewardCfg
from supermimic.imagination.objective import policy_objective
from supermimic.imagination.rollout import imagine
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.robots.khan_mini import load_profile
from supermimic.robots.registry import get_robot_spec
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


def _components(tmp_path, *, layout="isaac_khan", anchor=False):
    robot = get_robot_spec("khan_mini")
    profile, model = load_profile(), robot.load_model()
    q0 = np.array([profile["default_joint_pos"][name] for name in robot.joint_names])
    t = np.arange(48) / 50
    clip = {
        "fps": 50.0,
        "root_pos": np.column_stack((.1 * t, .02 * t, .65 + .01 * t)),
        "root_rot": np.tile([1., 0., 0., 0.], (len(t), 1)),
        "dof_pos": q0 + .02 * np.sin(t[:, None] * np.linspace(.5, 2., 27)),
    }
    path = tmp_path / "batch_encoder_motion.npz"
    convert_clip(clip, path, model=model, source="synthetic", clip_name=path.stem)
    lib = MotionLibrary([path], robot.tracked_bodies, q0, device="cpu",
                        robot_name=robot.name, root_body_name=robot.root_body)
    low, high = model.jnt_range[1:].T
    mid, half = (low + high) / 2, (high - low) * .9 / 2
    limits = torch.tensor(np.column_stack((mid - half, mid + half)), dtype=torch.float32)
    heads = AnalyticHeads(
        G1Kinematics(model), robot.tracked_bodies, robot.anchor_body, robot.root_body,
        limits, lib.default_joint_pos,
        reward_cfg=TrackingRewardCfg.for_joint_names(robot.joint_names, foot_tracking=True),
    )
    wm = WorldModel(WorldModelCfg(
        hidden=(16,), state_spec=StateSpecCfg(context_layout="khan_dr_v3"),
        input_normalization=True, termination_mode="analytic",
    ), heads, "cpu", action_scale=torch.full((27,), .5), action_offset=lib.default_joint_pos)
    with torch.no_grad():
        wm.dynamics.mlp.net[-1].weight.normal_(std=.002)
    wm.requires_grad_(False)
    history, future = (5, 5) if layout == "isaac_khan" else (2, 2)
    goal_width = 155 if layout == "isaac_khan" else future * robot.goal_feature_dim + (9 if anchor else 0)
    cfg = PolicyCfg(
        encoder_hidden=(16, 8), decoder_hidden=(16, 8), latent_dim=4,
        history=history, future_frames=future, goal_dim=goal_width, num_joints=27,
        num_tracked_bodies=15, observation_layout=layout,
        action_mode="reference_residual" if layout == "isaac_khan" else "default",
        goal_noise_z=.01 if layout == "isaac_khan" else 0.,
        goal_noise_joint_pos=.05 if layout == "isaac_khan" else 0.,
        running_normalization=True, hidden_layer_norm=True, hidden_activation="gelu",
    )
    kwargs = ({"default_joint_pos": lib.default_joint_pos, "action_scale": torch.full((27,), .5)}
              if layout == "isaac_khan" else {})
    policy = Policy(cfg, **kwargs)
    policy.update_normalization(torch.randn(12, policy.decoder.scale.numel()) * .3,
                                torch.randn(12, goal_width) * .2)
    # Last start reaches the final frame exactly under H32; its future goal
    # windows must still clamp independently without crossing into a clip.
    frames = torch.tensor([2, 3, 6, 15])
    history_frames = frames[None] + torch.arange(1 - history, 1)[:, None]
    valid = history_frames >= 0
    history_frames = history_frames.clamp_min(0)
    start = StartBatch(
        state=lib.ref71(frames), xi=wm.spec.xi_offset.expand(4, -1).clone(),
        frame=frames, prev_action=torch.zeros(4, 27),
        hist_states=lib.ref71(history_frames), hist_actions=torch.zeros(history, 4, 27),
        history_valid=valid,
    )
    return wm, policy, lib, start


def _loss(rollout):
    terminal = .03 * rollout.states[-1].square().mean(-1)
    return -policy_objective(rollout, terminal, .99)


@pytest.mark.parametrize("noise", [False, True])
@pytest.mark.parametrize("recompute", [False, True])
def test_batch_matches_stepwise_actions_rng_and_all_policy_gradients(tmp_path, noise, recompute):
    torch.manual_seed(16)
    wm, original, lib, start = _components(tmp_path)
    policies = [copy.deepcopy(original), copy.deepcopy(original)]
    outputs, gradients, rng_states, encoder_shapes = [], [], [], []
    original_buffers = {name: value.clone() for name, value in original.named_buffers()}
    for policy, batching in zip(policies, (False, True), strict=True):
        calls = []
        hook = policy.encoders["motion"].register_forward_pre_hook(
            lambda module, args, calls=calls: calls.append(tuple(args[0].shape)),
        )
        torch.manual_seed(279)
        result = imagine(wm, policy, lib, start, horizon=3, future_frames=5,
                         obs_noise=noise, activation_checkpoint=recompute,
                         batch_reference_encoder=batching, tensor_metrics=True)
        hook.remove()
        encoder_shapes.append(calls)
        _loss(result).backward()
        outputs.append(result)
        gradients.append([p.grad.clone() for p in policy.parameters()])
        rng_states.append(torch.get_rng_state())
        torch.testing.assert_close(dict(policy.named_buffers()), original_buffers, rtol=0, atol=0)
    assert encoder_shapes[0] == [(4, 155)] * 3
    assert encoder_shapes[1] == [(12, 155)]
    torch.testing.assert_close(rng_states[0], rng_states[1], rtol=0, atol=0)
    for field in ("states", "actions", "rewards", "survival", "goals", "critic_goals", "e_root"):
        torch.testing.assert_close(getattr(outputs[0], field), getattr(outputs[1], field),
                                   rtol=3e-5, atol=2e-6, msg=field)
    assert any(g.abs().sum() > 0 for g in gradients[1][:4]), "encoder must remain trainable"
    for eager, batched in zip(gradients[0], gradients[1], strict=True):
        torch.testing.assert_close(eager, batched, rtol=1e-4, atol=3e-6)
    torch.testing.assert_close(outputs[0].reward_terms, outputs[1].reward_terms,
                               rtol=3e-5, atol=2e-6)


def test_h32_uses_one_encoder_call_and_keeps_raw_reference_skip(tmp_path):
    torch.manual_seed(41)
    wm, policy, lib, start = _components(tmp_path)
    with torch.no_grad():
        for parameter in policy.decoder.parameters():
            parameter.zero_()
    shapes = []
    hook = policy.encoders["motion"].register_forward_pre_hook(
        lambda module, args: shapes.append(tuple(args[0].shape)),
    )
    with torch.no_grad():
        rollout = imagine(wm, policy, lib, start, 32, 5, obs_noise=True,
                          batch_reference_encoder=True, tensor_metrics=True)
    hook.remove()
    assert shapes == [(32 * 4, 155)]
    raw_next = lib.ref(rollout.frames[:-1] + 1).joint_pos
    torch.testing.assert_close(rollout.actions, (raw_next - lib.default_joint_pos) / .5,
                               rtol=0, atol=0)
    assert all(isinstance(value, torch.Tensor) and value.ndim == 0
               for value in rollout.reward_terms.values())


@pytest.mark.parametrize("anchor", [False, True])
def test_legacy_goal_batching_and_state_dependent_anchor_fallback(tmp_path, anchor):
    torch.manual_seed(19)
    wm, policy, lib, start = _components(tmp_path, layout="supermimic", anchor=anchor)
    outputs, shapes = [], []
    for batching in (False, True):
        calls = []
        hook = policy.encoders["motion"].register_forward_pre_hook(
            lambda module, args, calls=calls: calls.append(tuple(args[0].shape)),
        )
        torch.manual_seed(982)
        outputs.append(imagine(wm, policy, lib, start, 3, 2, obs_noise=True,
                               anchor_obs=anchor, batch_reference_encoder=batching))
        shapes.append(calls)
        hook.remove()
    assert len(shapes[1]) == (3 if anchor else 1)
    torch.testing.assert_close(outputs[0].actions, outputs[1].actions, rtol=2e-5, atol=1e-6)
    torch.testing.assert_close(outputs[0].states, outputs[1].states, rtol=2e-5, atol=1e-6)


def test_tensor_metrics_and_prevalidated_horizon_avoid_host_reads(tmp_path, monkeypatch):
    wm, policy, lib, start = _components(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("captured imagination must not read tensor values on the host")

    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "__bool__", forbidden)
        context.setattr(torch.Tensor, "tolist", forbidden)
        result = imagine(wm, policy, lib, start, 2, 5, obs_noise=True,
                         batch_reference_encoder=True, tensor_metrics=True,
                         validate_horizon=False)
    assert result.rewards.shape == (2, 4)


def test_decode_from_latent_equals_forward_with_nontrivial_normalization(tmp_path):
    _, policy, _, _ = _components(tmp_path)
    obs = torch.randn(7, 465)
    goal = torch.randn(7, 155)
    noisy = goal + torch.randn_like(goal) * .2
    expected = policy(obs, goal, encoder_goal=noisy)
    actual = policy.decode_from_latent(obs, goal, policy.encode_goal(goal, encoder_goal=noisy))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

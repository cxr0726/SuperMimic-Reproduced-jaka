"""Complete Actor/Critic capture keeps Adam, EMA and restore semantics intact."""

import copy
import io

import numpy as np
import pytest
import torch

from supermimic.data.khan_motion_convert import convert_clip
from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.envs.mdp.rewards import TrackingRewardCfg
from supermimic.imagination.policy_trainer import PolicyTrainer, PolicyTrainerCfg
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Critic, CriticCfg, Policy, PolicyCfg
from supermimic.robots.khan_mini import load_profile
from supermimic.robots.registry import get_robot_spec
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


def _components(tmp_path, device, *, goal_noise=False):
    torch.manual_seed(17)
    robot = get_robot_spec("khan_mini")
    profile, model = load_profile(), robot.load_model()
    q0 = np.array([profile["default_joint_pos"][name] for name in robot.joint_names])
    path = tmp_path / "capture_stationary.npz"
    clip = {
        "fps": 50., "root_pos": np.tile([0., 0., .65], (48, 1)),
        "root_rot": np.tile([1., 0., 0., 0.], (48, 1)),
        "dof_pos": np.tile(q0, (48, 1)),
    }
    convert_clip(clip, path, model=model, source="synthetic", clip_name=path.stem)
    lib = MotionLibrary(
        [path], robot.tracked_bodies, q0, device=device, store_dtype=torch.float32,
        robot_name=robot.name, root_body_name=robot.root_body,
    )
    limits = torch.tensor(np.asarray(model.jnt_range[1:]), dtype=torch.float32, device=device)
    midpoint, half = limits.mean(-1), .45 * (limits[:, 1] - limits[:, 0])
    limits = torch.stack((midpoint - half, midpoint + half), -1)
    heads = AnalyticHeads(
        G1Kinematics(model, device=device), robot.tracked_bodies, robot.anchor_body, robot.root_body,
        limits, lib.default_joint_pos,
        reward_cfg=TrackingRewardCfg.for_joint_names(robot.joint_names, foot_tracking=True),
    )
    wm = WorldModel(WorldModelCfg(
        hidden=(8,), state_spec=StateSpecCfg(context_layout="khan_dr_v3"),
        input_normalization=True, hidden_layer_norm=True, hidden_activation="gelu",
        termination_mode="analytic",
    ), heads, device, action_scale=torch.full((27,), .5, device=device),
       action_offset=lib.default_joint_pos)
    with torch.no_grad():
        # Ensure nonzero actor gradients propagate through learned dynamics.
        wm.dynamics.mlp.net[-1].weight.normal_(std=.003)
    pcfg = PolicyCfg(
        encoder_hidden=(8,), decoder_hidden=(8,), latent_dim=4, history=5,
        future_frames=5, goal_dim=155, num_joints=27, num_tracked_bodies=15,
        observation_layout="isaac_khan", action_mode="reference_residual",
        goal_noise_z=.01 if goal_noise else 0., goal_noise_joint_pos=.05 if goal_noise else 0.,
        running_normalization=True,
        hidden_layer_norm=True, hidden_activation="gelu",
    )
    policy = Policy(pcfg, default_joint_pos=lib.default_joint_pos,
                    action_scale=torch.full((27,), .5, device=device)).to(device)
    policy.update_normalization(torch.randn(12, 465, device=device) * .3,
                                torch.randn(12, 155, device=device) * .2)
    critic = Critic(CriticCfg(hidden=(8,), hidden_layer_norm=True, hidden_activation="gelu"),
                    pcfg=pcfg, xi_dim=wm.spec.xi_dim).to(device)
    buffer = ReplayBuffer(1, 16, 67, 27, 101, device)
    ids = torch.tensor([0], device=device)
    state = lib.ref71(torch.tensor([4], device=device))
    # Every possible state/goal/history is identical. Uniform replay draws and
    # critic shuffles are then irrelevant to the expected mathematical update.
    for _ in range(8):
        buffer.add(
            ids, state, torch.zeros(1, 27, device=device), torch.zeros(1, 27, device=device),
            wm.spec.xi_offset[None].clone(), torch.zeros(1, device=device), torch.zeros(1, device=device),
            torch.tensor([4], device=device), torch.zeros(1, dtype=torch.long, device=device),
            torch.zeros(1, dtype=torch.bool, device=device), torch.ones(1, dtype=torch.bool, device=device),
        )
        buffer.end_episode(ids)
    return wm, policy, critic, lib, buffer


def _trainer(components, *, capture, activation_checkpoint=False, obs_noise=False):
    wm, policy, critic, lib, _buffer = components
    cfg = PolicyTrainerCfg(
        horizon=2, batch_size=2, history=5, future_frames=5, obs_noise=obs_noise,
        compile=None, observation_layout="isaac_khan", critic_epochs=1, critic_minibatches=1,
        critic_value_bound=True, cuda_graph=capture, cuda_graph_warmup=3,
        activation_checkpoint=activation_checkpoint,
    )
    return PolicyTrainer(copy.deepcopy(wm), copy.deepcopy(policy), copy.deepcopy(critic), lib, cfg)


def _steps(optimizer):
    return {int(value["step"]) for value in optimizer.state.values() if "step" in value}


def _assert_models_match(actual, expected):
    for name in ("wm", "policy", "critic"):
        torch.testing.assert_close(getattr(actual, name).state_dict(), getattr(expected, name).state_dict(),
                                   rtol=3e-4, atol=3e-6)
    torch.testing.assert_close(actual.ema.state_dict(), expected.ema.state_dict(), rtol=3e-4, atol=3e-6)


def test_cpu_graph_flag_falls_back_and_preserves_update_and_ema_counts(tmp_path):
    components = _components(tmp_path, "cpu")
    trainer = _trainer(components, capture=True)
    before = copy.deepcopy(trainer.policy.state_dict())
    for _ in range(2):
        metrics = trainer.update(components[-1])
        assert all(np.isfinite(value) for value in metrics.values())
    assert trainer._captured_update.phase == "eager" and not trainer._captured_update.captured
    assert trainer.ema.num_updates == 2
    assert _steps(trainer.opt) == _steps(trainer.critic_opt) == {2}
    assert any(not torch.equal(value, before[name]) for name, value in trainer.policy.state_dict().items())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("activation_checkpoint", [False, True])
def test_actor_critic_graph_matches_eager_and_counts_every_call_once(tmp_path, activation_checkpoint):
    components = _components(tmp_path, "cuda:0")
    eager = _trainer(components, capture=False, activation_checkpoint=activation_checkpoint)
    captured = _trainer(components, capture=True, activation_checkpoint=activation_checkpoint)
    wm_initial = copy.deepcopy(captured.wm.state_dict())
    phases = []
    for index in range(6):
        lr = 0. if index == 4 else captured.cfg.lr * (.5 if index == 5 else 1.)
        eager.set_lr(lr)
        captured.set_lr(lr)
        previous_actor = copy.deepcopy(captured.policy.state_dict())
        previous_critic = copy.deepcopy(captured.critic.state_dict())
        expected = eager.update(components[-1])
        actual = captured.update(components[-1])
        phases.append(captured._captured_update.phase)
        assert all(np.isfinite(value) for value in actual.values())
        for key in expected:
            assert actual[key] == pytest.approx(expected[key], rel=5e-4, abs=4e-6), key
        _assert_models_match(captured, eager)
        assert _steps(captured.opt) == _steps(captured.critic_opt) == {index + 1}
        assert captured.ema.num_updates == index + 1
        if index == 4:
            # LR is a live device tensor. An already-recorded graph must obey
            # a new zero LR, while optimizer and EMA counters still advance.
            torch.testing.assert_close(captured.policy.state_dict(), previous_actor, rtol=0, atol=0)
            torch.testing.assert_close(captured.critic.state_dict(), previous_critic, rtol=0, atol=0)
        if index == 5:
            assert any(not torch.equal(value, previous_actor[name])
                       for name, value in captured.policy.state_dict().items())
    assert phases == ["warmup", "warmup", "warmup", "capture", "replay", "replay"]
    assert captured._captured_update.calls == 6
    torch.testing.assert_close(captured.wm.state_dict(), wm_initial, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("activation_checkpoint", [False, True])
def test_checkpoint_restore_invalidates_graph_and_continues_same_updates(tmp_path, activation_checkpoint):
    components = _components(tmp_path, "cuda:0")
    original = _trainer(components, capture=True, activation_checkpoint=activation_checkpoint)
    for _ in range(4):
        original.update(components[-1])
    assert original._captured_update.captured
    snapshot = {
        "wm": original.wm.state_dict(), "policy": original.policy.state_dict(),
        "critic": original.critic.state_dict(), "policy_opt": original.opt.state_dict(),
        "critic_opt": original.critic_opt.state_dict(), "policy_ema": original.ema.state_dict(),
        "policy_ema_updates": original.ema.num_updates, "policy_ema_decay": original.ema.decay,
    }
    serialized = io.BytesIO()
    torch.save(snapshot, serialized)
    serialized.seek(0)
    snapshot = torch.load(serialized, map_location="cuda:0", weights_only=False)
    restored = _trainer(components, capture=True, activation_checkpoint=activation_checkpoint)
    for name in ("wm", "policy", "critic"):
        getattr(restored, name).load_state_dict(snapshot[name])
    restored.opt.load_state_dict(snapshot["policy_opt"])
    restored.critic_opt.load_state_dict(snapshot["critic_opt"])
    restored.ema.restore(restored.policy, snapshot)
    restored.reset_capture()
    assert not restored._captured_update.captured
    assert restored.ema.num_updates == 4
    for _ in range(6):
        original.set_lr(7e-5)
        restored.set_lr(7e-5)
        expected = original.update(components[-1])
        actual = restored.update(components[-1])
        _assert_models_match(restored, original)
        for key in expected:
            assert actual[key] == pytest.approx(expected[key], rel=8e-4, abs=5e-6), key
    assert restored._captured_update.captured
    assert restored.ema.num_updates == original.ema.num_updates == 10
    assert _steps(restored.opt) == _steps(restored.critic_opt) == {10}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("activation_checkpoint", [False, True])
def test_captured_sensor_and_encoder_noise_stay_fresh_with_frozen_parameters(tmp_path, activation_checkpoint):
    components = _components(tmp_path, "cuda:0", goal_noise=True)
    trainer = _trainer(components, capture=True, activation_checkpoint=activation_checkpoint, obs_noise=True)
    trainer.set_lr(0.)
    parameters = {name: value.detach().clone() for name, value in trainer.policy.named_parameters()}
    buffers = {name: value.clone() for name, value in trainer.policy.named_buffers()}
    wm_buffers = {name: value.clone() for name, value in trainer.wm.named_buffers()}
    observed = {}

    def encoder_input(module, inputs):
        # Keep the graph-owned Tensor address. No Python callback runs during
        # replay, but the RNG kernels must fill this storage with fresh noise.
        observed["encoder"] = inputs[0]

    def decoder_input(module, inputs):
        # Exclude the 30 future-orientation channels: sensor history must be
        # independently fresh, rather than passing due only to anchor noise.
        observed["sensor"] = inputs[1][..., 30:]

    hooks = [
        trainer.policy.encoders["motion"].register_forward_pre_hook(encoder_input),
        trainer.policy.decoder.register_forward_pre_hook(decoder_input),
    ]
    try:
        for _ in range(4):
            trainer.update(components[-1])
        assert trainer._captured_update.captured
        previous = {name: value.detach().clone() for name, value in observed.items()}
        action_means = []
        for _ in range(3):
            metrics = trainer.update(components[-1])
            assert trainer._captured_update.phase == "replay"
            action_means.append(metrics["action_abs_mean"])
            for name in ("encoder", "sensor"):
                assert not torch.equal(observed[name], previous[name]), name
                previous[name] = observed[name].detach().clone()
        assert len(set(action_means)) > 1
    finally:
        for hook in hooks:
            hook.remove()
    # Replay states/frames are constant and LR is zero, so changed actions must
    # come from fresh graph RNG, rather than parameter/statistics updates.
    torch.testing.assert_close(dict(trainer.policy.named_parameters()), parameters, rtol=0, atol=0)
    torch.testing.assert_close(dict(trainer.policy.named_buffers()), buffers, rtol=0, atol=0)
    torch.testing.assert_close(dict(trainer.wm.named_buffers()), wm_buffers, rtol=0, atol=0)
    assert trainer.ema.num_updates == 7

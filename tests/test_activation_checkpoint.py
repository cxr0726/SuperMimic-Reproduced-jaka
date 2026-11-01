"""Activation recomputation preserves a full-batch Khan FK actor/critic update."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from supermimic.data.khan_motion_convert import convert_clip
from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.imagination.policy_trainer import PolicyTrainer, PolicyTrainerCfg
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Critic, CriticCfg, Policy, PolicyCfg
from supermimic.robots.khan_mini import load_profile
from supermimic.robots.registry import get_robot_spec
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


def _khan_training_components(tmp_path, anchor_obs, observation_layout="supermimic"):
    robot = get_robot_spec("khan_mini")
    profile = load_profile()
    model = robot.load_model()
    q0 = np.array([profile["default_joint_pos"][name] for name in robot.joint_names])
    t = np.arange(16) / 50
    clip = {
        "fps": 50.0,
        "root_pos": np.column_stack((0.1 * t, 0.02 * t, 0.65 + 0.01 * t)),
        "root_rot": np.tile([1.0, 0.0, 0.0, 0.0], (len(t), 1)),
        "dof_pos": q0 + 0.02 * t[:, None] * np.linspace(-1, 1, len(q0)),
    }
    npz = tmp_path / "khan_checkpoint_equivalence.npz"
    convert_clip(clip, npz, model=model, source="synthetic", clip_name=npz.stem)
    library = MotionLibrary([npz], robot.tracked_bodies, q0, device="cpu",
                            robot_name=robot.name, root_body_name=robot.root_body)
    lo, hi = model.jnt_range[1:].T
    mid, half = (lo + hi) / 2, (hi - lo) * 0.9 / 2
    limits = torch.tensor(np.column_stack((mid - half, mid + half)), dtype=torch.float32)
    heads = AnalyticHeads(G1Kinematics(model), robot.tracked_bodies, robot.anchor_body,
                          robot.root_body, limits, library.default_joint_pos)
    wm = WorldModel(WorldModelCfg(hidden=(16,), term_hidden=(8,)), heads, "cpu")
    # Exercise actual state/action BPTT rather than only the zero-initialized prior.
    with torch.no_grad():
        wm.dynamics.mlp.net[-1].weight.normal_(std=0.001)
    history, future = (5, 5) if observation_layout == "isaac_khan" else (2, 2)
    goal = 155 if observation_layout == "isaac_khan" else 2 * robot.goal_feature_dim + (9 if anchor_obs else 0)
    policy_cfg = PolicyCfg(encoder_hidden=(16,), decoder_hidden=(16,), latent_dim=4,
                           history=history, future_frames=future, num_joints=robot.num_joints,
                           goal_dim=goal, observation_layout=observation_layout)
    policy = Policy(policy_cfg)
    critic = Critic(CriticCfg(hidden=(8,)), pcfg=policy_cfg)
    buffer = ReplayBuffer(2, 32, robot.state_dim, robot.num_joints, robot.num_joints + 4, "cpu")
    env_ids = torch.arange(2)
    for frame in range(12):
        frames = torch.full((2,), frame, dtype=torch.long)
        qpos, qvel = library.ref_qpos_qvel(frames)
        action = torch.full((2, robot.num_joints), 0.01 * np.sin(frame))
        buffer.add(
            env_ids, s=torch.cat((qpos, qvel), -1), a=action, a_exec=action,
            xi=wm.spec.xi_offset.repeat(2, 1), r=torch.zeros(2), done=torch.zeros(2),
            frame=frames, clip=torch.zeros(2, dtype=torch.long),
            terminal=torch.zeros(2, dtype=torch.bool),
            ep_start=torch.full((2,), frame == 0),
        )
    return wm, policy, critic, library, buffer


@pytest.mark.parametrize("observation_layout,anchor_obs", [
    ("supermimic", False), ("supermimic", True), ("isaac_khan", False),
])
def test_activation_checkpoint_preserves_objective_gradients_optimizer_and_rng(tmp_path, observation_layout, anchor_obs):
    torch.manual_seed(123)
    wm, policy, critic, library, buffer = _khan_training_components(tmp_path, anchor_obs, observation_layout)
    cfg = PolicyTrainerCfg(horizon=3, batch_size=4, history=policy.cfg.history,
                           future_frames=policy.cfg.future_frames, observation_layout=observation_layout,
                           critic_epochs=1, critic_minibatches=2, compile=None,
                           obs_noise=True, anchor_obs=anchor_obs)
    assert not cfg.activation_checkpoint
    trainers = [
        PolicyTrainer(copy.deepcopy(wm), copy.deepcopy(policy), copy.deepcopy(critic),
                      library, replace(cfg, activation_checkpoint=enabled))
        for enabled in (False, True)
    ]
    metrics, rng_states = [], []
    for trainer in trainers:
        torch.manual_seed(527)
        metrics.append(trainer.update(buffer))
        rng_states.append(torch.get_rng_state())
    assert metrics[0].keys() == metrics[1].keys()
    for name in metrics[0]:
        assert np.isfinite(metrics[0][name]) and np.isfinite(metrics[1][name])
        np.testing.assert_allclose(metrics[0][name], metrics[1][name], rtol=2e-6, atol=1e-8,
                                   err_msg=name)
    assert metrics[0]["policy_grad_norm"] > 0
    for name in ("policy", "critic"):
        eager, checkpointed = getattr(trainers[0], name), getattr(trainers[1], name)
        torch.testing.assert_close(eager.state_dict(), checkpointed.state_dict(), rtol=2e-6, atol=1e-8)
        for eager_parameter, checkpointed_parameter in zip(eager.parameters(), checkpointed.parameters()):
            assert eager_parameter.grad is not None and checkpointed_parameter.grad is not None
            torch.testing.assert_close(eager_parameter.grad, checkpointed_parameter.grad,
                                       rtol=2e-6, atol=1e-8)
    for name in ("opt", "critic_opt"):
        torch.testing.assert_close(getattr(trainers[0], name).state_dict(),
                                   getattr(trainers[1], name).state_dict(), rtol=2e-6, atol=1e-8)
    torch.testing.assert_close(rng_states[0], rng_states[1], rtol=0, atol=0)
    # Recomputing the frozen world model must not change its weights or accumulate gradients.
    torch.testing.assert_close(trainers[0].wm.state_dict(), trainers[1].wm.state_dict(), rtol=0, atol=0)
    assert all(parameter.grad is None for trainer in trainers for parameter in trainer.wm.parameters())

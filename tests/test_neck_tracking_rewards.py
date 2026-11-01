"""Reference neck rewards agree in simulator helpers and imagined rollouts."""

from dataclasses import replace
from types import SimpleNamespace

import mujoco
import pytest
import torch

from supermimic.data.replay_buffer import StartBatch
from supermimic.envs.mdp.motion_library import RefBodies, RefState
from supermimic.envs.mdp.rewards import (
    TrackingRewardCfg,
    joint_tracking_error_exp,
    motion_neck_position_error_exp,
    motion_neck_velocity_error_exp,
)
from supermimic.imagination.compiled import REWARD_TERM_NAMES, ImagStep
from supermimic.imagination.rollout import imagine
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.robots.khan_mini import get_khan_mini_spec, load_profile
from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


def _components(khan=True):
    if khan:
        model = get_khan_mini_spec().compile()
        profile = load_profile()
        q0 = torch.tensor([profile["default_joint_pos"][n] for n in profile["joint_names"]])
        root, anchor = profile["root_body"], profile["anchor_body"]
    else:
        model = mujoco.MjModel.from_xml_string("""
            <mujoco><worldbody><body name="pelvis"><freejoint name="root"/>
            <geom type="sphere" size=".1"/><body name="leg" pos="0 0 -.2">
            <joint name="left_hip_pitch_joint"/><geom type="sphere" size=".1"/>
            </body></body></worldbody></mujoco>
        """)
        q0 = torch.zeros(1)
        root = anchor = "pelvis"
    names = tuple(dict.fromkeys((root, anchor)))
    limits = torch.tensor([[-100.0, 100.0]]).expand(len(q0), 2)
    heads = AnalyticHeads(G1Kinematics(model), names, anchor, root, limits, q0)
    state = torch.zeros(2, model.nq + model.nv)
    state[:, :model.nq] = torch.tensor(model.qpos0, dtype=torch.float32)
    state[:, 7:model.nq] = q0
    bodies = heads.robot_bodies(state)
    ref = ReferenceFrame(bodies["pos"], bodies["quat"], bodies["lin_vel"], bodies["ang_vel"],
                         bodies["root_pos"], bodies["root_quat"],
                         state[:, 7:model.nq].clone(), state[:, model.nq + 6:].clone())
    return heads, state, ref, bodies


def test_joint_tracking_averages_errors_before_exponential():
    actual = torch.tensor([[0.3, 0.0], [0.1, -0.2]], requires_grad=True)
    reference = torch.tensor([[0.0, 0.0], [0.1, -0.2]])
    reward = joint_tracking_error_exp(actual, reference, std=0.3)
    torch.testing.assert_close(reward, torch.tensor([torch.exp(torch.tensor(-0.5)), 1.0]))
    reward.sum().backward()
    assert actual.grad[0, 0] < 0
    assert actual.grad[0, 1] == 0
    with pytest.raises(ValueError, match="positive std"):
        joint_tracking_error_exp(actual, reference, std=0.0)


def test_factory_keeps_g1_maximum_and_enables_two_khan_terms():
    assert TrackingRewardCfg().max_step_reward == pytest.approx(0.1)
    assert TrackingRewardCfg.for_joint_names(["left_hip_pitch_joint"]).max_step_reward == pytest.approx(0.1)
    cfg = TrackingRewardCfg.for_joint_names(["Neck_pitch_joint", "Neck_yaw_joint"])
    assert (cfg.w_neck_pos, cfg.w_neck_vel) == (0.5, 0.1)
    assert (cfg.std_neck_pos, cfg.std_neck_vel) == (0.3, 1.0)
    assert cfg.max_step_reward == pytest.approx(0.112)


def test_neck_reference_deviation_reduces_only_its_term_and_has_state_gradients():
    heads, state, ref, bodies = _components()
    action = torch.zeros(2, heads.num_joints)
    matching = heads.reward(state, action, action, ref, bodies)
    torch.testing.assert_close(matching["total"], torch.full((2,), 0.112))
    assert len(matching) == 11  # Ten reward terms plus their total.
    assert heads.neck_joint_ids == [25, 26]
    changed = state.clone().requires_grad_()
    with torch.no_grad():
        changed[:, 7 + torch.tensor(heads.neck_joint_ids)] += 0.15
        changed[:, heads.fk.nq + 6 + torch.tensor(heads.neck_joint_ids)] += 0.5
    terms = heads.reward(changed, action, action, ref, bodies)
    assert (terms["motion_neck_pos"] < matching["motion_neck_pos"]).all()
    assert (terms["motion_neck_vel"] < matching["motion_neck_vel"]).all()
    for name in REWARD_TERM_NAMES:
        torch.testing.assert_close(terms[name], matching[name])
    (terms["motion_neck_pos"].sum() + terms["motion_neck_vel"].sum()).backward()
    assert (changed.grad[:, 7 + torch.tensor(heads.neck_joint_ids)] < 0).all()
    assert (changed.grad[:, heads.fk.nq + 6 + torch.tensor(heads.neck_joint_ids)] < 0).all()
    # A nonzero future reference remains the target: the reward does not lock the neck at zero.
    moved_ref = replace(ref, joint_pos=changed[:, 7:heads.fk.nq].detach(),
                        joint_vel=changed[:, heads.fk.nq + 6:].detach())
    moved = heads.reward(changed, action, action, moved_ref, bodies)
    torch.testing.assert_close(moved["motion_neck_pos"], matching["motion_neck_pos"])
    torch.testing.assert_close(moved["motion_neck_vel"], matching["motion_neck_vel"])


def test_simulator_helpers_match_analytic_weights_and_timestep():
    heads, state, ref, bodies = _components()
    q = state[:, 7:heads.fk.nq].clone()
    qd = state[:, heads.fk.nq + 6:].clone()
    q[:, heads.neck_joint_ids] += 0.2
    qd[:, heads.neck_joint_ids] -= 0.4
    changed = state.clone()
    changed[:, 7:heads.fk.nq] = q
    changed[:, heads.fk.nq + 6:] = qd
    command = SimpleNamespace(joint_pos=ref.joint_pos, joint_vel=ref.joint_vel)
    robot = SimpleNamespace(data=SimpleNamespace(joint_pos=q, joint_vel=qd))
    env = SimpleNamespace(scene={"robot": robot},
                          command_manager=SimpleNamespace(get_term=lambda _: command))
    asset = SimpleNamespace(name="robot", joint_ids=list(reversed(heads.neck_joint_ids)))
    cfg = heads.reward_cfg
    action = torch.zeros(2, heads.num_joints)
    terms = heads.reward(changed, action, action, ref, bodies)
    pos = motion_neck_position_error_exp(env, "motion", cfg.std_neck_pos, asset)
    vel = motion_neck_velocity_error_exp(env, "motion", cfg.std_neck_vel, asset)
    torch.testing.assert_close(terms["motion_neck_pos"], pos * cfg.dt * cfg.w_neck_pos)
    torch.testing.assert_close(terms["motion_neck_vel"], vel * cfg.dt * cfg.w_neck_vel)


def test_old_reference_api_and_eight_terms_remain_available_without_neck():
    heads, state, ref, bodies = _components(khan=False)
    legacy_ref = ReferenceFrame(ref.body_pos_w, ref.body_quat_w, ref.body_lin_vel_w,
                                ref.body_ang_vel_w, ref.root_pos_w, ref.root_quat_w)
    action = torch.zeros(2, heads.num_joints)
    terms = heads.reward(state, action, action, legacy_ref, bodies)
    assert set(terms) == {*REWARD_TERM_NAMES, "total"}
    torch.testing.assert_close(terms["total"], torch.full((2,), 0.1))
    wm = WorldModel(WorldModelCfg(hidden=(8,), term_hidden=(4,)), heads, "cpu")
    assert ImagStep(wm, heads.default_joint_pos).term_names == REWARD_TERM_NAMES


def test_imagined_step_and_rollout_carry_reference_neck_targets():
    torch.manual_seed(7)
    heads, state, ref, _ = _components()
    joints = heads.num_joints
    wm = WorldModel(WorldModelCfg(hidden=(8,), term_hidden=(4,)), heads, "cpu")
    step = ImagStep(wm, heads.default_joint_pos)
    assert step.term_names == REWARD_TERM_NAMES + ("motion_neck_pos", "motion_neck_vel")
    action = torch.zeros(2, joints)
    xi = wm.spec.xi_offset.repeat(2, 1)
    out = step(state, action, action, xi, ref)
    assert out[2].shape == (10, 2)
    changed_pos, changed_vel = ref.joint_pos.clone(), ref.joint_vel.clone()
    changed_pos[:, heads.neck_joint_ids] += 0.3
    changed_vel[:, heads.neck_joint_ids] += 1.0
    moved = step(state, action, action, xi,
                 replace(ref, joint_pos=changed_pos, joint_vel=changed_vel))
    assert (moved[2][-2:] < out[2][-2:]).all()
    torch.testing.assert_close(moved[2][:8], out[2][:8])
    # Exercise state/action BPTT away from the exactly matching orientation,
    # using a fitted-model-like response rather than its zero-output prior.
    with torch.no_grad():
        wm.dynamics.mlp.net[-1].weight.normal_(std=0.001)

    def refs(frame):
        n = len(frame)
        return RefState(ref.root_pos_w[:1].expand(n, -1), ref.root_quat_w[:1].expand(n, -1),
                        heads.default_joint_pos.expand(n, -1), torch.zeros(n, 3),
                        torch.zeros(n, 3), torch.zeros(n, joints))

    def bodies(frame):
        n = len(frame)
        return RefBodies(*(v[:1].expand(n, *v.shape[1:]) for v in (
            ref.body_pos_w, ref.body_quat_w, ref.body_lin_vel_w, ref.body_ang_vel_w)))

    future, history = 2, 2
    lib = SimpleNamespace(default_joint_pos=heads.default_joint_pos, ref=refs, bodies=bodies,
                          last_frame_of=lambda frame: frame + 100,
                          goal_window=lambda frame, k: torch.zeros(len(frame), k * (2 * joints + 10)))
    policy = Policy(PolicyCfg(encoder_hidden=(8,), decoder_hidden=(8,), latent_dim=4,
                              history=history, num_joints=joints,
                              goal_dim=future * (2 * joints + 10)))
    # A nonzero initial action also exercises action-rate regularization.
    with torch.no_grad():
        policy.decoder.mlp.net[-1].bias.fill_(0.1)
    start = StartBatch(state, xi, torch.zeros(2, dtype=torch.long), action,
                       state.repeat(history, 1, 1), action.repeat(history, 1, 1))
    rollout = imagine(wm, policy, lib, start, horizon=2, future_frames=future,
                      obs_noise=False, step_fn=step)
    assert set(rollout.reward_terms) == set(step.term_names)
    assert all(torch.isfinite(v).all() for v in (rollout.states, rollout.rewards))
    (-rollout.rewards.sum()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in policy.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters())

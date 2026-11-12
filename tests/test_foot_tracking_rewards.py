"""Foot-specific rewards preserve clean reference geometry in real and imagined use."""

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from supermimic.envs.mdp.rewards import (
    KHAN_FOOT_BODY_NAMES,
    TrackingRewardCfg,
    foot_orientation_tracking_exp,
    foot_position_tracking_exp,
    motion_foot_orientation_error_exp,
    motion_foot_position_error_exp,
    reanchor_reference,
    tracking_reward_terms,
)
from supermimic.imagination.compiled import ImagStep
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.robots.khan_mini import get_khan_mini_spec, load_profile
from supermimic.sim_context import KHAN_FOOT_BODIES
from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


def _identity(shape):
    q = torch.zeros(*shape, 4)
    q[..., 0] = 1
    return q


def _reward_inputs():
    positions = torch.zeros(1, 15, 3)
    positions[:, 0, 2] = 0.8
    positions[:, 1] = torch.tensor([0.15, 0.1, 0.3])
    positions[:, 2] = torch.tensor([-0.1, -0.1, 0.05])
    quaternions = _identity((1, 15))
    velocities = torch.zeros_like(positions)
    return {
        "robot_body_pos_w": positions.clone(), "robot_body_quat_w": quaternions.clone(),
        "robot_body_lin_vel_w": velocities, "robot_body_ang_vel_w": velocities,
        "ref_body_pos_w": positions, "ref_body_quat_w": quaternions,
        "ref_body_lin_vel_w": velocities, "ref_body_ang_vel_w": velocities,
        "anchor_index": 0, "action": torch.zeros(1, 1), "prev_action": torch.zeros(1, 1),
        "joint_pos": torch.zeros(1, 1), "soft_limits": torch.tensor([[-10., 10.]]),
        "cfg": TrackingRewardCfg(w_foot_pos=.5, w_foot_ori=.5), "foot_body_ids": (1, 2),
    }


def test_legacy_rewards_remain_opt_in_and_reward_ceiling_includes_both_feet_terms():
    names = ("Neck_pitch_joint", "Neck_yaw_joint")
    assert KHAN_FOOT_BODY_NAMES == KHAN_FOOT_BODIES
    assert TrackingRewardCfg().max_step_reward == pytest.approx(.1)
    old = TrackingRewardCfg.for_joint_names(names)
    assert old.max_step_reward == pytest.approx(.112)
    assert old.w_foot_pos == old.w_foot_ori == 0
    new = TrackingRewardCfg.for_joint_names(names, foot_tracking=True)
    assert (new.w_foot_pos, new.w_foot_ori) == (.25, .25)
    assert (new.std_foot_pos, new.std_foot_ori) == (.1, .25)
    assert new.max_step_reward == pytest.approx(.122)


def test_one_foot_error_is_independent_and_not_diluted_by_fifteen_bodies():
    inputs = _reward_inputs()
    inputs["robot_body_pos_w"][:, 1, 2] -= .1
    rewards = tracking_reward_terms(**inputs)
    # One foot is perfect, the other has one kernel-width error. Averaging
    # after exp makes this distinct from an exp of the mean squared error.
    cfg = inputs["cfg"]
    pos_scale, ori_scale = cfg.dt * cfg.w_foot_pos, cfg.dt * cfg.w_foot_ori
    expected = pos_scale * (1 + math.exp(-1)) / 2
    assert rewards["motion_foot_pos"].item() == pytest.approx(expected)
    assert rewards["motion_foot_pos"].item() != pytest.approx(pos_scale * math.exp(-.5))
    assert rewards["motion_body_pos"].item() / .02 > .99
    assert rewards["motion_foot_pos"].item() / pos_scale < .7
    assert rewards["motion_foot_ori"].item() == pytest.approx(ori_scale)
    inputs["cfg"] = replace(inputs["cfg"], w_foot_pos=0, w_foot_ori=0)
    legacy = tracking_reward_terms(**inputs)
    assert "motion_foot_pos" not in legacy and "motion_foot_ori" not in legacy
    torch.testing.assert_close(rewards["total"], legacy["total"] + expected + ori_scale)


def test_xy_and_yaw_reanchor_but_robot_root_height_does_not_cancel_foot_height_error():
    inputs = _reward_inputs()
    ref = inputs["ref_body_pos_w"]
    # Rotate by +90 deg and translate in xy, preserving the reference world z.
    actual = ref.clone()
    actual[..., 0] = -ref[..., 1] + 2
    actual[..., 1] = ref[..., 0] - 3
    inputs["robot_body_pos_w"] = actual
    inputs["robot_body_quat_w"][:] = torch.tensor([2**-.5, 0, 0, 2**-.5])
    matched = tracking_reward_terms(**inputs)
    cfg = inputs["cfg"]
    pos_scale, ori_scale = cfg.dt * cfg.w_foot_pos, cfg.dt * cfg.w_foot_ori
    torch.testing.assert_close(matched["motion_foot_pos"], torch.tensor([pos_scale]))
    torch.testing.assert_close(matched["motion_foot_ori"], torch.tensor([ori_scale]))
    inputs["robot_body_pos_w"][..., 2] += .1
    raised = tracking_reward_terms(**inputs)
    # Subtracting each root height would incorrectly remove this real z error.
    assert raised["motion_foot_pos"].item() == pytest.approx(pos_scale * math.exp(-1))


def test_clean_foot_kernels_have_finite_correct_direction_gradients(monkeypatch):
    def noise_is_not_a_reward_input(*args, **kwargs):
        raise AssertionError("Reward attempted to generate observation/goal noise")

    monkeypatch.setattr(torch, "rand_like", noise_is_not_a_reward_input)
    positions = torch.zeros(1, 2, 3)
    positions[0, 0, 2] = -.05
    positions.requires_grad_()
    pos_reward = foot_position_tracking_exp(positions, torch.zeros_like(positions), .1)
    angle = torch.tensor(.25, requires_grad=True)
    q = torch.stack((torch.cos(angle / 2), torch.sin(angle / 2), angle * 0, angle * 0))
    orientations = torch.stack((q, _identity(()))).unsqueeze(0)
    ori_reward = foot_orientation_tracking_exp(orientations, _identity((1, 2)), .25)
    assert ori_reward.item() == pytest.approx((1 + math.exp(-1)) / 2)
    (pos_reward + ori_reward).backward()
    assert torch.isfinite(positions.grad).all() and torch.isfinite(angle.grad)
    assert positions.grad[0, 0, 2] > 0  # Lift the dropped foot.
    assert positions.grad[0, 1].abs().sum() == 0
    assert angle.grad < 0  # Rotate back toward the reference.
    flipped = -orientations.detach()
    torch.testing.assert_close(foot_orientation_tracking_exp(flipped, _identity((1, 2)), .25), ori_reward.detach())


def _components():
    profile = load_profile()
    model = get_khan_mini_spec().compile()
    q0 = torch.tensor([profile["default_joint_pos"][name] for name in profile["joint_names"]])
    cfg = TrackingRewardCfg.for_joint_names(profile["joint_names"], foot_tracking=True)
    heads = AnalyticHeads(
        G1Kinematics(model), tuple(profile["tracked_bodies"]), profile["anchor_body"],
        profile["root_body"], torch.tensor([[-100., 100.]]).repeat(len(q0), 1), q0, cfg,
    )
    state = torch.zeros(2, model.nq + model.nv)
    state[:, :model.nq] = torch.tensor(model.qpos0, dtype=torch.float32)
    state[:, 7:model.nq] = q0
    bodies = heads.robot_bodies(state)
    reference = ReferenceFrame(
        bodies["pos"].clone(), bodies["quat"].clone(), bodies["lin_vel"], bodies["ang_vel"],
        bodies["root_pos"], bodies["root_quat"], state[:, 7:model.nq].clone(),
        state[:, model.nq + 6:].clone(),
    )
    return heads, state, reference, bodies, profile


def test_simulator_helpers_match_imagined_reward_and_reuse_existing_fk(monkeypatch):
    heads, state, ref, bodies, profile = _components()
    ref.body_pos_w[:, heads.foot_body_ids[0], 2] += .06
    ai = heads.anchor_index
    ref_pos, ref_quat = reanchor_reference(
        ref.body_pos_w, ref.body_quat_w, ref.body_pos_w[:, ai], ref.body_quat_w[:, ai],
        bodies["pos"][:, ai], bodies["quat"][:, ai],
    )
    command = SimpleNamespace(
        cfg=SimpleNamespace(body_names=tuple(profile["tracked_bodies"])),
        robot_body_pos_w=bodies["pos"], robot_body_quat_w=bodies["quat"],
        body_pos_relative_w=ref_pos, body_quat_relative_w=ref_quat,
        noisy_goal=torch.full((2, 155), float("nan")),
    )
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command))

    def no_second_fk(*args):
        raise AssertionError("Reward redundantly recomputed existing FK")

    monkeypatch.setattr(heads, "robot_bodies", no_second_fk)
    action = torch.zeros(2, heads.num_joints)
    imagined = heads.reward(state, action, action, ref, bodies)
    cfg = heads.reward_cfg
    torch.testing.assert_close(
        imagined["motion_foot_pos"],
        cfg.dt * cfg.w_foot_pos * motion_foot_position_error_exp(env, "motion", cfg.std_foot_pos),
    )
    torch.testing.assert_close(
        imagined["motion_foot_ori"],
        cfg.dt * cfg.w_foot_ori * motion_foot_orientation_error_exp(env, "motion", cfg.std_foot_ori),
    )


def test_imagined_step_logs_optional_foot_terms_and_validates_tracked_names():
    heads, state, ref, _, _ = _components()
    wm = WorldModel(WorldModelCfg(hidden=(8,), term_hidden=(4,)), heads, "cpu")
    step = ImagStep(wm, heads.default_joint_pos)
    assert step.term_names[-2:] == ("motion_foot_pos", "motion_foot_ori")
    action = torch.zeros(2, heads.num_joints)
    out = step(state, action, action, wm.spec.xi_offset.repeat(2, 1), ref)
    assert out[2].shape == (len(step.term_names), 2)
    assert torch.isfinite(out[2]).all()
    with pytest.raises(ValueError, match="both feet"):
        AnalyticHeads(heads.fk, ("base_link",), "base_link", "base_link", heads.soft_limits,
                      heads.default_joint_pos, heads.reward_cfg)
    # A legacy model is allowed to omit foot bodies because its weights are 0.
    legacy = AnalyticHeads(heads.fk, ("base_link",), "base_link", "base_link", heads.soft_limits,
                           heads.default_joint_pos)
    assert legacy.foot_body_ids == ()

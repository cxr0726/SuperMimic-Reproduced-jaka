"""Robot-sized state, analytic heads and differentiable Khan FK on CPU."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest
import torch

from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.kinematics.rotations import quat_apply, quat_from_yaw, quat_mul
from supermimic.robots.khan_mini import get_khan_mini_spec, load_profile
from supermimic.world_model.heads import (
    AnalyticHeads,
    ReferenceFrame,
    analytic_observation,
    flatten_history_term_major,
    observation_terms,
    split_state,
)
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


@pytest.mark.parametrize("num_joints", [27, 29])
def test_robot_sized_state_integration_and_normalisation(num_joints):
    generator = torch.Generator().manual_seed(num_joints)
    spec = StateSpec(StateSpecCfg(explicit_integration=False), "cpu", torch.float64,
                     num_joints=num_joints)
    assert spec.state_dim == spec.feat_dim == 13 + 2 * num_joints
    assert spec.delta_dim == 12 + 2 * num_joints
    assert spec.xi_dim == num_joints + 4
    states = torch.randn(7, spec.state_dim, dtype=torch.float64, generator=generator)
    states[:, 3:7] /= states[:, 3:7].norm(dim=-1, keepdim=True)
    zero_delta = torch.zeros(7, spec.delta_dim, dtype=torch.float64)
    torch.testing.assert_close(spec.integrate(states, zero_delta), states, rtol=0, atol=1e-14)
    assert spec.error_vector(states, states).shape == (7, spec.delta_dim)
    assert spec.normalized_sq_error(states, states).max() < 1e-24
    assert spec.canonical(states).shape == states.shape
    pieces = split_state(states)
    assert [p.shape[-1] for p in pieces] == [3, 4, num_joints, 3, 3, num_joints]
    torch.testing.assert_close(torch.cat(pieces, -1), states)
    torch.testing.assert_close(spec.normalize_xi(spec.xi_offset), torch.zeros(spec.xi_dim, dtype=torch.float64))
    torch.testing.assert_close(spec.xi_scale[1:1 + num_joints],
                               torch.full((num_joints,), 0.01, dtype=torch.float64))

    # Canonical inputs retain G1's yaw/XY translation invariance.
    yaw = quat_from_yaw(torch.linspace(-2, 2, len(states), dtype=torch.float64))
    changed = states.clone()
    changed[:, :2] += torch.tensor([0.7, -1.2], dtype=torch.float64)
    changed[:, 3:7] = quat_mul(yaw, states[:, 3:7])
    changed[:, spec.nq:spec.nq + 3] = quat_apply(yaw, states[:, spec.nq:spec.nq + 3])
    torch.testing.assert_close(spec.canonical(changed), spec.canonical(states), rtol=0, atol=1e-12)

    integrating = StateSpec(StateSpecCfg(), "cpu", torch.float64, num_joints=num_joints)
    predicted = integrating.integrate(states, zero_delta)
    torch.testing.assert_close(predicted[:, :3], states[:, :3] + 0.02 * pieces[3], rtol=0, atol=1e-14)
    torch.testing.assert_close(predicted[:, 7:spec.nq], pieces[2] + 0.02 * pieces[5], rtol=0, atol=1e-14)


def test_state_context_rejects_incompatible_custom_vectors():
    cfg = StateSpecCfg(xi_offset=(1.0,) * 33)
    with pytest.raises(ValueError, match="31 values"):
        StateSpec(cfg, "cpu", num_joints=27)
    cfg = StateSpecCfg(xi_offset=(0.0,) * 31, xi_scale=(0.0,) * 31)
    with pytest.raises(ValueError, match="positive scales"):
        StateSpec(cfg, "cpu", num_joints=27)


@pytest.mark.parametrize("num_joints", [27, 29])
def test_dynamic_observation_bias_and_term_major_history(num_joints):
    spec = StateSpec(StateSpecCfg(), "cpu", num_joints=num_joints)
    state = torch.arange(spec.state_dim, dtype=torch.float32).repeat(2, 1) / 100
    state[:, 3:7] = torch.tensor([1.0, 0, 0, 0])
    action = torch.full((2, num_joints), 0.7)
    q0 = torch.linspace(-0.2, 0.2, num_joints)
    xi = spec.xi_offset.repeat(2, 1)
    xi[:, 1:1 + num_joints] = torch.linspace(-0.01, 0.01, num_joints)
    obs = analytic_observation(state, action, xi, q0)
    assert obs.shape == (2, 3 * num_joints + 6)
    torch.testing.assert_close(obs[:, :num_joints], state[:, 7:spec.nq] + xi[:, 1:1 + num_joints] - q0)
    torch.testing.assert_close(obs[:, -num_joints:], action)
    hist = torch.stack([obs + i for i in range(10)])
    flat = flatten_history_term_major(hist)
    expected = []
    start = 0
    for _, width in observation_terms(num_joints):
        expected.append(hist[..., start:start + width].permute(1, 0, 2).flatten(1))
        start += width
    torch.testing.assert_close(flat, torch.cat(expected, -1), rtol=0, atol=0)
    assert flat.shape == (2, 10 * (3 * num_joints + 6))


def _khan_states(model, count=5):
    rng = np.random.default_rng(723)
    qpos = rng.normal(size=(count, model.nq)) * 0.1
    quat = rng.normal(size=(count, 4))
    qpos[:, 3:7] = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    lo, hi = model.jnt_range[1:].T
    qpos[:, 7:] = rng.uniform(lo, hi, size=(count, model.nq - 7))
    qvel = rng.normal(size=(count, model.nv))
    return qpos, qvel


def test_khan_fk_fixed_links_velocities_and_gradients():
    model = get_khan_mini_spec().compile()
    qpos, qvel = _khan_states(model)
    qp = torch.tensor(qpos, dtype=torch.float64, requires_grad=True)
    qv = torch.tensor(qvel, dtype=torch.float64, requires_grad=True)
    out = G1Kinematics(model, dtype=torch.float64)(qp, qv)
    data = mujoco.MjData(model)
    velocity = np.zeros(6)
    for sample in range(len(qpos)):
        data.qpos[:] = qpos[sample]
        data.qvel[:] = qvel[sample]
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(out["pos"][sample].detach(), data.xpos[1:], atol=1e-12)
        expected = data.xquat[1:]
        actual = out["quat"][sample].detach().numpy()
        signs = np.sign((actual * expected).sum(-1, keepdims=True))
        np.testing.assert_allclose(actual * signs, expected, atol=1e-12)
        for body in range(1, model.nbody):
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, body, velocity, 0)
            np.testing.assert_allclose(out["lin_vel"][sample, body - 1].detach(), velocity[3:], atol=1e-11)
            np.testing.assert_allclose(out["ang_vel"][sample, body - 1].detach(), velocity[:3], atol=1e-11)
    (out["pos"].square().sum() + out["lin_vel"].square().sum()).backward()
    assert torch.isfinite(qp.grad).all() and torch.isfinite(qv.grad).all()
    assert qp.grad[:, 7:].abs().sum() > 0 and qv.grad[:, 6:].abs().sum() > 0


def test_khan_analytic_reward_and_world_model_boundaries():
    model = get_khan_mini_spec().compile()
    profile = load_profile()
    fk = G1Kinematics(model)
    names = tuple(profile["tracked_bodies"])
    soft_limits = torch.tensor(model.jnt_range[1:], dtype=torch.float32)
    q0 = torch.tensor([profile["default_joint_pos"][name] for name in profile["joint_names"]])
    heads = AnalyticHeads(fk, names, profile["anchor_body"], profile["root_body"], soft_limits, q0)
    qpos, qvel = _khan_states(model, count=3)
    state = torch.tensor(np.concatenate((qpos, qvel), -1), dtype=torch.float32)
    bodies = heads.robot_bodies(state)
    ref = ReferenceFrame(bodies["pos"], bodies["quat"], bodies["lin_vel"], bodies["ang_vel"],
                         bodies["root_pos"], bodies["root_quat"],
                         state[:, 7:model.nq], state[:, model.nq + 6:])
    actions = torch.zeros(3, 27)
    reward = heads.reward(state, actions, actions, ref)
    torch.testing.assert_close(reward["total"], torch.full((3,), 0.112), rtol=0, atol=2e-7)
    assert len(reward) == 11  # Ten terms plus their total.
    assert heads.anchor_index == 0 and names[-1] == "Neck_pitch_Link"
    assert not heads.terminated(state, ref).any()

    wm = WorldModel(WorldModelCfg(hidden=(16, 16), term_hidden=(8,)), heads, "cpu")
    xi = wm.spec.xi_offset.repeat(3, 1)
    next_state = wm.step(state, actions, xi)
    assert next_state.shape == (3, 67)
    first = wm.dynamics.mlp.net[0]
    last = wm.dynamics.mlp.net[-1]
    assert first.in_features == 67 + 27 + 31
    assert last.out_features == 66
    assert wm.termination_logits(next_state, ref).shape == (3,)
    loss, _ = wm.window_loss(torch.stack([state, next_state.detach()]), actions[None], xi,
                             torch.zeros(1, 3), [ref])
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in wm.parameters())


@pytest.mark.parametrize("velocity_term", ["lin_vel", "ang_vel"])
def test_tracked_head_contributes_to_body_velocity_rewards(velocity_term):
    """Isolate the added head's physical velocity from the other 14 body targets."""
    model = get_khan_mini_spec().compile()
    profile = load_profile()
    q0 = torch.tensor([profile["default_joint_pos"][name] for name in profile["joint_names"]],
                      dtype=torch.float64)
    names = tuple(profile["tracked_bodies"])
    heads = AnalyticHeads(
        G1Kinematics(model, dtype=torch.float64), names, profile["anchor_body"],
        profile["root_body"], torch.tensor([[-100., 100.]], dtype=torch.float64).repeat(27, 1), q0,
    )
    assert len(names) == 15 and names[-1] == "Neck_pitch_Link"
    state = torch.zeros(1, model.nq + model.nv, dtype=torch.float64)
    state[:, :model.nq] = torch.tensor(model.qpos0, dtype=torch.float64)
    state[:, 7:model.nq] = q0
    neck = profile["joint_names"].index("Neck_pitch_joint")
    if velocity_term == "lin_vel":
        # The head lies above the base, so root angular motion gives its origin
        # a real linear velocity through the lever arm.
        state[:, model.nq + 3] = 0.4
    else:
        state[:, model.nq + 6 + neck] = 0.6
    state.requires_grad_()
    bodies = heads.robot_bodies(state)
    assert bodies[velocity_term][:, -1].norm() > 0.1
    if velocity_term == "ang_vel":
        # Both neck hinges have the same origin in this URDF. Neck motion affects
        # the head orientation/angular velocity, not its link-origin linear speed.
        torch.testing.assert_close(bodies["ang_vel"][:, :-1],
                                   torch.zeros_like(bodies["ang_vel"][:, :-1]))
        torch.testing.assert_close(bodies["lin_vel"], torch.zeros_like(bodies["lin_vel"]))
    ref = ReferenceFrame(
        bodies["pos"].detach(), bodies["quat"].detach(),
        bodies["lin_vel"].detach(), bodies["ang_vel"].detach(),
        bodies["root_pos"].detach(), bodies["root_quat"].detach(),
        state[:, 7:model.nq].detach(), state[:, model.nq + 6:].detach(),
    )
    action = torch.zeros(1, 27, dtype=torch.float64)
    matching = heads.reward(state, action, action, ref)
    field = f"body_{velocity_term}_w"
    target = getattr(ref, field).clone()
    target[:, -1] = 0
    altered = heads.reward(state, action, action, replace(ref, **{field: target}))
    term = f"motion_body_{velocity_term}"
    assert (altered[term] < matching[term]).all()
    for name in set(matching) - {term, "total"}:
        torch.testing.assert_close(altered[name], matching[name])
    altered[term].sum().backward()
    assert torch.isfinite(state.grad).all() and state.grad.abs().sum() > 0

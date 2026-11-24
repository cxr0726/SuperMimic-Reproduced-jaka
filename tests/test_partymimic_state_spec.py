"""Independent geometric oracles for the two packed world-model representations."""

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from supermimic.world_model.state_spec import (
    StateSpec,
    StateSpecCfg,
    from_mujoco_state,
    to_mujoco_state,
)


def sample_state(spec, seed):
    rng = np.random.default_rng(seed)
    state = rng.normal(size=(6, spec.state_dim))
    state[:, 3:7] = Rotation.random(len(state), random_state=rng).as_quat(scalar_first=True)
    return torch.from_numpy(state)


def rotations(state):
    body = Rotation.from_quat(state[:, 3:7].numpy(), scalar_first=True)
    mat = body.as_matrix()
    yaw = np.arctan2(mat[:, 1, 0], mat[:, 0, 0])
    return body, Rotation.from_euler("z", yaw[:, None])


def party_heading(body):
    matrix = body.as_matrix()
    x, y = matrix[:, 0, 0], matrix[:, 1, 0]
    norm = np.sqrt(x * x + y * y + 1e-9)
    heading = np.zeros_like(matrix)
    heading[:, 0, 0] = heading[:, 1, 1] = x / norm
    heading[:, 1, 0] = y / norm
    heading[:, 0, 1] = -y / norm
    heading[:, 2, 2] = 1.0
    return heading


@pytest.mark.parametrize("n", [27, 29])
def test_world_velocity_storage_and_body_features(n):
    spec = StateSpec(StateSpecCfg(), "cpu", torch.float64, num_joints=n)
    native = sample_state(spec, n)
    body, heading = rotations(native)
    nq = spec.nq
    # Include leading time and batch dimensions, as used by window/replay paths.
    state = from_mujoco_state(native.reshape(2, 3, -1), "partymimic")
    torch.testing.assert_close(to_mujoco_state(state, "partymimic").reshape_as(native), native)
    state = state.reshape_as(native)
    np.testing.assert_allclose(state[:, : nq + 3], native[:, : nq + 3])
    np.testing.assert_allclose(state[:, nq + 3 : nq + 6], body.apply(native[:, nq + 3 : nq + 6]))
    np.testing.assert_allclose(state[:, nq + 6 :], native[:, nq + 6 :])

    feat = spec.canonical(state) * spec.in_scale
    h = party_heading(body)
    tilt = (h.transpose(0, 2, 1) @ body.as_matrix())[:, :, :2].reshape(len(native), 6)
    np.testing.assert_allclose(feat[:, 1:7], tilt, atol=1e-12)
    np.testing.assert_allclose(feat[:, 7:10], body.inv().apply(native[:, nq : nq + 3]))
    np.testing.assert_allclose(feat[:, 10:13], native[:, nq + 3 : nq + 6])
    assert not np.allclose(
        body.inv().apply(native[:, nq : nq + 3]), heading.inv().apply(native[:, nq : nq + 3])
    )

    legacy = StateSpec(StateSpecCfg(representation="legacy"), "cpu", torch.float64, n)
    assert from_mujoco_state(native, "legacy") is native
    assert to_mujoco_state(native, "legacy") is native
    torch.testing.assert_close(spec.critic_features(state), legacy.canonical(native))


@pytest.mark.parametrize("representation", ["partymimic", "legacy"])
@pytest.mark.parametrize("drift_enabled", [True, False])
def test_integrated_state_matches_endpoint_velocity_oracle(representation, drift_enabled):
    spec = StateSpec(
        StateSpecCfg(representation=representation, explicit_integration=drift_enabled),
        "cpu",
        torch.float64,
        num_joints=27,
    )
    state = sample_state(spec, 111)
    body, heading = rotations(state)
    delta = torch.randn(
        (len(state), spec.delta_dim),
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(997),
    )
    result = spec.integrate(state, delta)
    dv, dw, dqd, dp, dtheta, dq = (delta * spec.out_scale).split([3, 3, 27, 3, 3, 27], -1)
    p, _, j, v, w, qd = (x.numpy() for x in spec.split(state))
    dt = spec.dt if drift_enabled else 0.0
    if representation == "partymimic":
        end_v = v + body.apply(dv)
        end_w = w + body.apply(dw)
        v_pose = 3 / 8 * v + 5 / 8 * end_v
        w_pose = body.inv().apply(3 / 8 * w + 5 / 8 * end_w)
        qd_pose = 3 / 8 * qd + 5 / 8 * (qd + dqd.numpy())
        h = party_heading(body)
        drift = h.transpose(0, 2, 1) @ (dt * v_pose[..., None])
        end_p = p + (h @ (dp.numpy()[..., None] + drift))[..., 0]
    else:
        end_v = v + heading.apply(dv)
        end_w = w + dw.numpy()
        v_pose, w_pose, qd_pose = end_v, end_w, qd + dqd.numpy()
        end_p = p + dt * v_pose + heading.apply(dp)
    expected_body = body * Rotation.from_rotvec(dt * w_pose + dtheta.numpy())
    got_p, _, got_j, got_v, got_w, got_qd = spec.split(result)
    got_body, _ = rotations(result)
    np.testing.assert_allclose(got_p, end_p, rtol=0, atol=1e-12)
    np.testing.assert_allclose(got_body.as_matrix(), expected_body.as_matrix(), atol=1e-12)
    np.testing.assert_allclose(got_j, j + dt * qd_pose + dq.numpy(), atol=1e-12)
    np.testing.assert_allclose(got_v, end_v, atol=1e-12)
    np.testing.assert_allclose(got_w, end_w, atol=1e-12)
    np.testing.assert_allclose(got_qd, qd + dqd.numpy(), atol=1e-12)


def test_loss_uses_true_frames_and_existing_nominal_scales():
    spec = StateSpec(StateSpecCfg(), "cpu", torch.float64, num_joints=27)
    pred, truth = sample_state(spec, 129), sample_state(spec, 409)
    predicted_rot, _ = rotations(pred)
    true_rot, _ = rotations(truth)
    h = party_heading(true_rot)
    dp, _, dj, dv, dw, dqd = (x.numpy() for x in spec.split(pred - truth))
    raw = np.concatenate(
        (
            (h.transpose(0, 2, 1) @ dp[..., None])[..., 0],
            (true_rot.inv() * predicted_rot).as_rotvec(),
            true_rot.inv().apply(dv),
            true_rot.inv().apply(dw),
            dj,
            dqd,
        ),
        axis=-1,
    )
    np.testing.assert_allclose(
        spec.error_vector(pred, truth), raw / spec.err_scale.numpy(), atol=1e-12
    )
    same_w = pred.clone()
    same_w[:, spec.nq + 3 : spec.nq + 6] = truth[:, spec.nq + 3 : spec.nq + 6]
    assert torch.count_nonzero(spec.error_vector(same_w, truth)[:, 9:12]) == 0


def test_vertical_root_projection_and_gradients():
    spec = StateSpec(StateSpecCfg(), "cpu", torch.float64, num_joints=27)
    state = sample_state(spec, 15)[:1]
    # Forward axis points straight down, so PartyMimic's heading is diag(0, 0, 1).
    state[:, 3:7] = torch.tensor([0.5, 0.5, 0.5, -0.5])
    state[:, spec.nq : spec.nq + 3] = torch.tensor([1.0, -2.0, 3.0])
    state.requires_grad_()
    delta = state.new_zeros(1, spec.delta_dim, requires_grad=True)
    pred = spec.integrate(state, delta)
    torch.testing.assert_close(pred[:, :3] - state[:, :3], state.new_tensor([[0.0, 0.0, 0.06]]))
    feat = spec.canonical(state)
    torch.testing.assert_close(feat[:, 1:5], torch.zeros_like(feat[:, 1:5]))
    truth = state.detach().clone()
    truth[:, :3] += truth.new_tensor([4.0, -2.0, 1.0])
    truth.requires_grad_()
    error = spec.error_vector(pred, truth)
    assert torch.count_nonzero(error[:, :2]) == 0
    grads = torch.autograd.grad(feat.sum() + error.square().sum(), (state, delta, truth))
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_two_step_rotation_gradients_match_finite_differences():
    spec = StateSpec(StateSpecCfg(), "cpu", torch.float64, num_joints=2)
    s = sample_state(spec, 17)[:1]
    target = sample_state(spec, 41)[:1]
    gen = torch.Generator().manual_seed(293)
    delta = torch.randn((2, 1, spec.delta_dim), dtype=s.dtype, generator=gen, requires_grad=True)

    def objective(state, changes):
        for change in changes.unbind(0):
            state = spec.integrate(state, change)
        return spec.error_vector(state, target).square().sum()

    assert torch.autograd.gradcheck(objective, (s.requires_grad_(), delta), fast_mode=True)

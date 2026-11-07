"""Reference FK compilation preserves compact-library semantics and precision."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.robots.registry import get_robot_spec
from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame


def _packed_library(path: Path, robot: str, dtype=torch.float16) -> Path:
    """Nontrivial root rotations/velocities expose free-joint frame mistakes."""
    spec = get_robot_spec(robot)
    model = spec.load_model()
    rng = np.random.default_rng(195)
    n = 16
    root_pos = rng.normal(size=(n, 3)) * .4
    root_pos[:, 2] += 1
    quat = rng.normal(size=(n, 4))
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    joint = rng.uniform(model.jnt_range[1:, 0], model.jnt_range[1:, 1],
                        size=(n, spec.num_joints))
    velocities = rng.normal(size=(n, 6 + spec.num_joints))
    ref = torch.as_tensor(np.concatenate((root_pos, quat, joint, velocities), axis=-1), dtype=dtype)
    torch.save({"ref": ref, "names": ["a", "b"], "starts": [0, 9], "lengths": [9, 7],
                    "robot": robot, "joint_names": spec.joint_names, "root_body_name": spec.root_body,
                    "fps": 50}, path)
    return path


def _load(path, robot, *, device="cpu", compiled=True, store_dtype=torch.float16):
    spec = get_robot_spec(robot)
    return MotionLibrary([path], spec.tracked_bodies, torch.zeros(spec.num_joints),
                         device=device, compact=True, store_dtype=store_dtype,
                         robot_name=robot, root_body_name=spec.root_body,
                         reference_fk_compile=compiled)


def _tensors(bodies):
    return (bodies.pos_w, bodies.quat_w, bodies.lin_vel_w, bodies.ang_vel_w)


def _assert_bodies_equal(actual, expected):
    for a, b in zip(_tensors(actual), _tensors(expected), strict=True):
        assert a.dtype == b.dtype == torch.float32
        assert torch.isfinite(a).all()
        assert not a.requires_grad
        # Fused float32 arithmetic can differ by a few ulps; no fp16 FK allowed.
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)


def _assert_downstream_equal(eager, compiled, robot):
    """Reference rounding must not materially alter rewards or actor gradients."""
    frames = torch.tensor([0, 3, 8, 9, 11, 15, 0, 15], device="cuda")
    spec = get_robot_spec(robot)
    model = spec.load_model()
    limits = torch.as_tensor(np.asarray(model.jnt_range[1:]), dtype=torch.float32, device="cuda")
    heads = AnalyticHeads(eager._fk, spec.tracked_bodies, spec.anchor_body,
                          spec.root_body, limits, eager.default_joint_pos)
    qp, qv = eager.ref_qpos_qvel(frames)
    state = torch.cat((qp, qv), -1)
    state[:, :3] += .03
    state[:, 7:7 + spec.num_joints] += .02
    results = []
    for library in (eager, compiled):
        body, ref = library.bodies(frames), library.ref(frames)
        target = ReferenceFrame(body.pos_w, body.quat_w, body.lin_vel_w, body.ang_vel_w,
                                ref.root_pos_w, ref.root_quat_w, ref.joint_pos, ref.joint_vel)
        actual = state.detach().clone().requires_grad_()
        action = torch.linspace(-.2, .3, spec.num_joints, device="cuda").repeat(len(frames), 1).requires_grad_()
        rewards = heads.reward(actual, action, torch.zeros_like(action), target)
        errors = heads.critic_error_features(actual, target)
        objective = rewards["total"].mean() + .01 * errors.square().mean()
        state_gradient, action_gradient = torch.autograd.grad(objective, (actual, action))
        results.append((rewards, errors, state_gradient, action_gradient))
    expected, actual = results
    for key in expected[0]:
        torch.testing.assert_close(actual[0][key], expected[0][key], atol=2e-6, rtol=2e-5)
    for a, b in zip(actual[1:], expected[1:], strict=True):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
    assert actual[2].abs().max() > 0 and actual[3].abs().max() > 0


@pytest.mark.parametrize("robot", ["g1", "khan_mini"])
def test_reference_compile_cpu_keeps_eager_path(tmp_path, monkeypatch, robot):
    def forbidden_compile(*args, **kwargs):
        raise AssertionError("CPU compact references should retain the eager path")

    monkeypatch.setattr(torch, "compile", forbidden_compile)
    path = _packed_library(tmp_path / "tiny.pt", robot)
    library = _load(path, robot)
    frame = torch.tensor([[0, 8, 9], [15, 0, 9]])
    output = library.bodies(frame)
    assert output.pos_w.shape == (*frame.shape, len(library.body_names), 3)
    assert output.quat_w.dtype == torch.float32


@pytest.mark.parametrize("robot", ["g1", "khan_mini"])
def test_compact_reference_matches_mujoco_body_origin_velocity(tmp_path, robot):
    """Oracle uses identical dequantized inputs, isolating FK from fp16 storage."""
    path = _packed_library(tmp_path / "tiny.pt", robot)
    library = _load(path, robot, compiled=False)
    frames = torch.tensor([0, 8, 9, 15, 0])
    bodies = library.bodies(frames)
    qp, qv = library.ref_qpos_qvel(frames)
    model = get_robot_spec(robot).load_model()
    data = mujoco.MjData(model)
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in library.body_names]
    pos, quat, lin, ang = [], [], [], []
    velocity = np.zeros(6)
    for qpos, qvel in zip(qp.double().numpy(), qv.double().numpy(), strict=True):
        # The FK normalizes the fp16-dequantized root quaternion before use.
        qpos[3:7] /= np.linalg.norm(qpos[3:7])
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(model, data)
        pos.append(data.xpos[body_ids].copy())
        quat.append(data.xquat[body_ids].copy())
        frame_lin, frame_ang = [], []
        for body_id in body_ids:
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY,
                                    body_id, velocity, 0)
            frame_ang.append(velocity[:3].copy())
            frame_lin.append(velocity[3:].copy())
        lin.append(frame_lin)
        ang.append(frame_ang)
    np.testing.assert_allclose(bodies.pos_w.numpy(), pos, atol=2e-5, rtol=2e-5)
    target_quat = np.asarray(quat)
    actual_quat = bodies.quat_w.numpy()
    signs = np.where((actual_quat * target_quat).sum(-1, keepdims=True) < 0, -1, 1)
    np.testing.assert_allclose(actual_quat * signs, target_quat, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(bodies.lin_vel_w.numpy(), lin, atol=2e-4, rtol=2e-5)
    np.testing.assert_allclose(bodies.ang_vel_w.numpy(), ang, atol=2e-4, rtol=2e-5)


@pytest.mark.parametrize("robot", ["g1", "khan_mini"])
@pytest.mark.parametrize("entry", ["_compact_bodies", "_compact_bodies_selected"])
def test_dynamic_reference_graph_keeps_hinge_motion(tmp_path, robot, entry):
    """Catch symbolic-int/pybind-enum folding even without a CUDA compiler.

    A backend returning the captured graph isolates Dynamo capture correctness
    from Inductor numerics. Random nonzero hinges and joint rates are essential:
    neutral-pose checks can miss an incorrectly fixed-joint tree.
    """
    path = _packed_library(tmp_path / "tiny.pt", robot)
    library = _load(path, robot, compiled=False)
    graphs = []

    def backend(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(getattr(library, entry), fullgraph=True, dynamic=True, backend=backend)
    for count in (8, 3, 17, 32):
        frame = torch.arange(count) % library.num_frames
        with torch.no_grad():
            actual, expected = compiled(frame), library._compact_bodies(frame)
        _assert_bodies_equal(actual, expected)
    assert len(graphs) == 1, "A batch-size change should reuse the dynamic reference graph"


@pytest.mark.parametrize("robot", ["g1", "khan_mini"])
@pytest.mark.parametrize("with_velocity", [False, True])
def test_static_selected_fk_preserves_order_duplicates_and_values(tmp_path, robot, with_velocity):
    path = _packed_library(tmp_path / "tiny.pt", robot)
    library = _load(path, robot, compiled=False)
    qpos, qvel = library.ref_qpos_qvel(torch.tensor([[0, 8], [9, 15]]))
    velocity = qvel if with_velocity else None
    # Distinct, noncontiguous and repeated output indices must preserve caller
    # order exactly, including a repeated root between distant limbs.
    order = (library._fk.nbody - 1, 0, 3, 0)
    full = library._fk(qpos, velocity)
    selected = library._fk(qpos, velocity, body_indices=order)
    assert selected.keys() == full.keys()
    for name in full:
        torch.testing.assert_close(selected[name], full[name][..., list(order), :], atol=0, rtol=0)


@pytest.mark.parametrize("robot", ["g1", "khan_mini"])
def test_reference_compile_flag_reaches_motion_command(tmp_path, robot):
    from supermimic.envs.tracking_env_cfg import SuperMimicEnvCfgArgs, make_env_cfg

    path = _packed_library(tmp_path / "tiny.pt", robot)
    default = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=[str(path)], robot=robot, compact_library=True))
    enabled = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=[str(path)], robot=robot,
                                              compact_library=True, reference_fk_compile=True))
    disabled = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=[str(path)], robot=robot,
                                               compact_library=True, reference_fk_compile=False))
    assert default.commands["motion"].reference_fk_compile is False
    assert enabled.commands["motion"].reference_fk_compile is True
    assert disabled.commands["motion"].reference_fk_compile is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA compiler")
@pytest.mark.parametrize("robot", ["g1", "khan_mini"])
def test_compiled_reference_shapes_values_and_output_lifetime(tmp_path, robot):
    path = _packed_library(tmp_path / "tiny.pt", robot)
    eager = _load(path, robot, device="cuda", compiled=False)
    compiled = _load(path, robot, device="cuda", compiled=True)
    original_storage = compiled._ref.clone()
    retained = None
    retained_copy = None
    # Scalars, empty queries, duplicate/boundary frames and noncontiguous leading
    # dimensions must all preserve the established eager API.
    frames = [torch.tensor(9, device="cuda"),
              torch.empty(0, dtype=torch.long, device="cuda"),
              torch.tensor([15], device="cuda"),
              torch.tensor([0, 8, 9, 15, 0, 8, 9, 15], device="cuda"),
              torch.arange(17, device="cuda") % 16,
              torch.arange(32, device="cuda") % 16,
              (torch.arange(24, device="cuda").reshape(4, 6) % 16).T,
              torch.tensor([15, 9, 8, 0, 15, 9, 8, 0], device="cuda")]
    for frame in frames:
        actual, expected = compiled.bodies(frame), eager.bodies(frame)
        _assert_bodies_equal(actual, expected)
        if frame.numel() == 8 and retained is None:
            retained = actual
            retained_copy = tuple(t.clone() for t in _tensors(actual))
    assert retained is not None
    # Repeated calls and shape changes must not overwrite old reference tensors
    # which remain live throughout the H+1 imagined rollout.
    for actual, saved in zip(_tensors(retained), retained_copy, strict=True):
        torch.testing.assert_close(actual, saved, atol=0, rtol=0)
    torch.testing.assert_close(compiled._ref, original_storage, atol=0, rtol=0)
    torch.testing.assert_close(compiled.goal_window(torch.tensor([8, 15], device="cuda"), 5),
                               eager.goal_window(torch.tensor([8, 15], device="cuda"), 5),
                               atol=0, rtol=0)
    _assert_downstream_equal(eager, compiled, robot)

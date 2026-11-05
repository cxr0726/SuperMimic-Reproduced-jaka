"""Selected-world armature refresh preserves the stock backend's plant semantics."""

from copy import copy
from dataclasses import fields, is_dataclass
from pathlib import Path

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import pytest
import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.utils.torch import configure_torch_backends

from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg
from supermimic.envs.partial_constants import KhanArmatureConstants, khan_joint_armature

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data/npz/khan_mini/lafan1/walk1_subject1.npz"
DERIVED = {
    "dof_invweight0", "body_invweight0", "tendon_length0",
    "tendon_invweight0", "actuator_acc0",
}


def test_khan_armature_event_retains_reset_sampling_without_global_recompute():
    cfg = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=["unused.npz"], robot="khan_mini"))
    term = cfg.events["ankle_armature"]
    assert term.func is khan_joint_armature and term.mode == "reset"
    assert term.params["ranges"] == (0.8, 1.2)
    assert term.params["operation"] == "scale" and not term.params["shared_random"]
    assert term.func.recompute == RecomputeLevel.none
    assert {"dof_armature", *DERIVED} <= set(term.func.model_fields)
    # Stock mjlab remains available, unchanged, for other robots and the oracle.
    assert dr.joint_armature.recompute == RecomputeLevel.set_const_0


def _make_env(*, stock=False):
    if not torch.cuda.is_available() or not NPZ.is_file():
        pytest.skip("needs CUDA and converted Khan LAFAN1")
    args = SuperMimicEnvCfgArgs(
        npz_files=[str(NPZ)], robot="khan_mini", num_envs=8, replay_capacity=64,
        action_noise_std=0.0, obs_noise=False, reset_noise=False,
        sampling_mode="start", seed=73,
    )
    cfg = make_env_cfg(args)
    if stock:
        cfg.events["ankle_armature"].func = dr.joint_armature
    configure_torch_backends()
    return ManagerBasedRlEnv(cfg, device="cuda:0")


def _arrays(obj, prefix=""):
    """All live backend arrays, including nested statistics/contact buffers."""
    if isinstance(obj, wp.array):
        return {prefix: obj}
    result = {}
    if is_dataclass(obj):
        for field in fields(obj):
            name = f"{prefix}.{field.name}" if prefix else field.name
            result.update(_arrays(getattr(obj, field.name), name))
    elif isinstance(obj, (list, tuple)):
        for index, item in enumerate(obj):
            result.update(_arrays(item, f"{prefix}[{index}]"))
    return result


def _snapshot(arrays):
    return {name: wp.to_torch(array).clone() for name, array in arrays.items()}


def _restore(arrays, snapshot):
    for name, array in arrays.items():
        wp.to_torch(array).copy_(snapshot[name])


def _assert_bytes(actual, expected, name):
    # Unused backend workspaces can contain NaN bit patterns; numerical equal
    # would reject preserved NaNs, while this verifies actual byte preservation.
    assert actual.shape == expected.shape, name
    assert torch.equal(actual.contiguous().view(torch.uint8),
                       expected.contiguous().view(torch.uint8)), name


def _assert_close(actual, expected, name):
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-7,
                               msg=lambda message: f"{name}: {message}")


def _assert_constant_close(actual, expected, name):
    # The backend accumulates COM/CRB with float32 atomics; compact packing and
    # graph replay can change reduction order. For the five-world test below,
    # M has condition numbers 4.9k--7.0k. Twenty repetitions measured both stock
    # and compact constants within 6.35 ppm of the matched-input CPU64 oracle,
    # while two float32 outputs differed by up to 5.45 ppm. Check 10 ppm here,
    # separately from exact preservation and trajectory checks.
    expected = torch.as_tensor(expected, device=actual.device, dtype=torch.float64)
    torch.testing.assert_close(actual.double(), expected, rtol=1e-5, atol=1e-6,
                               msg=lambda message: f"{name}: {message}")


def _cpu_constant_reference(env, env_ids):
    """Double precision mj_setConst using the exact float32 GPU model inputs."""
    model_inputs = []
    for field in fields(env.sim.wp_model):
        value = getattr(env.sim.wp_model, field.name)
        if not isinstance(value, wp.array) or not hasattr(env.sim.mj_model, field.name):
            continue
        target = getattr(env.sim.mj_model, field.name)
        if not isinstance(target, np.ndarray):
            continue
        shape = getattr(field.type, "shape", ())
        batched = bool(shape and shape[0] in ("*", "nworld"))
        raw = wp.to_torch(value).cpu().numpy().copy()
        if (raw.shape[1:] if batched else raw.shape) == target.shape:
            model_inputs.append((field.name, raw, batched))
    names = ("dof_invweight0", "body_invweight0", "actuator_acc0")
    result = {name: [] for name in names}
    for world in env_ids.cpu().tolist():
        model = copy(env.sim.mj_model)
        for name, raw, batched in model_inputs:
            getattr(model, name)[:] = raw[world % raw.shape[0]] if batched else raw
        data = mujoco.MjData(model)
        mujoco.mj_setConst(model, data)
        for name in names:
            result[name].append(np.array(getattr(model, name), copy=True))
    return {name: np.stack(values) for name, values in result.items()}


@pytest.mark.slow
@pytest.mark.parametrize("captured", [False, True])
def test_partial_refresh_matches_full_constants_without_touching_live_data(captured):
    env = _make_env()
    try:
        env.reset(seed=73)
        # Exercise reference constants while live kinematics have a nontrivial
        # post-integration state, rather than relying on the reset pose only.
        for _ in range(3):
            env.step(torch.full((8, 27), 0.03, device=env.device))
        env.sim.use_cuda_graph = captured
        helper = KhanArmatureConstants(env.sim)
        model_arrays, data_arrays = _arrays(env.sim.wp_model), _arrays(env.sim.wp_data)
        model_ptrs = {name: array.ptr for name, array in model_arrays.items()}
        data_ptrs = {name: array.ptr for name, array in data_arrays.items()}
        ctx = env.recorder_manager.get_term("replay").ctx
        armature = wp.to_torch(env.sim.wp_model.dof_armature)
        nominal = env.sim.get_default_field("dof_armature")[ctx.ankle_dof_ids]
        sequence = [[], [6, 1, 3], [0], [7, 2], [7, 0, 5, 2, 1],
                    list(range(8)), [4], [0, 7], []]
        for iteration, chosen in enumerate(sequence):
            ids = torch.tensor(chosen, device=env.device, dtype=torch.long)
            keep = torch.ones(8, device=env.device, dtype=torch.bool)
            keep[ids] = False
            if chosen:
                factors = torch.linspace(0.81, 1.19, len(chosen) * 4,
                                         device=env.device).reshape(-1, 4)
                if iteration % 2:
                    factors = factors.flip(-1)
                armature[ids[:, None], ctx.ankle_dof_ids] = nominal * factors
            before_model, before_data = _snapshot(model_arrays), _snapshot(data_arrays)
            torch.cuda.synchronize()
            helper.refresh(ids)
            torch.cuda.synchronize()
            partial = _snapshot(model_arrays)
            if len(chosen) < env.num_envs:
                for name, array in data_arrays.items():
                    _assert_bytes(wp.to_torch(array), before_data[name], f"live Data {name}")
            else:
                # All worlds really reset: using the stock path avoids a large
                # persistent full-batch scratch allocation at startup.
                _assert_bytes(wp.to_torch(env.sim.wp_data.qpos), before_data["qpos"], "full qpos")
                _assert_bytes(wp.to_torch(env.sim.wp_data.qvel), before_data["qvel"], "full qvel")
            assert model_ptrs == {name: array.ptr for name, array in model_arrays.items()}
            assert data_ptrs == {name: array.ptr for name, array in data_arrays.items()}
            for name, current in partial.items():
                if name in DERIVED and current.shape[0] == 8:
                    _assert_bytes(current[keep], before_model[name][keep], f"unselected {name}")
                elif name == "stat.meaninertia":
                    if current.shape[0] == 8:
                        _assert_bytes(current[keep], before_model[name][keep], name)
                    elif 0 not in chosen:
                        _assert_bytes(current, before_model[name], "shared mean without world0")
                else:
                    _assert_bytes(current, before_model[name], f"unrelated model field {name}")
            if not chosen:
                assert helper.last_capacity == 0
                continue
            assert len(chosen) <= helper.last_capacity < 2 * len(chosen)
            if len(chosen) < env.num_envs:
                assert helper.last_capacity in helper.buckets
                bucket = helper.buckets[helper.last_capacity]
                assert (bucket.graph is not None) == captured

            # The oracle starts with exactly the same inputs and cached values.
            # It may touch live Data; the optimized refresh above must not.
            _restore(model_arrays, before_model)
            _restore(data_arrays, before_data)
            mjwarp.set_const_0(env.sim.wp_model, env.sim.wp_data)
            torch.cuda.synchronize()
            full = _snapshot(model_arrays)
            for name in DERIVED:
                if partial[name].shape[0] == 8:
                    _assert_constant_close(partial[name][ids], full[name][ids], name)
                else:
                    _assert_constant_close(partial[name], full[name], name)
            cpu_reference = _cpu_constant_reference(env, ids)
            for name, expected in cpu_reference.items():
                _assert_constant_close(partial[name][ids], expected, f"partial {name} vs CPU64")
                _assert_constant_close(full[name][ids], expected, f"stock {name} vs CPU64")
            # A compacted batch must not replace original world0 with its first
            # selected world when this backend statistic is shared.
            _assert_close(partial["stat.meaninertia"], full["stat.meaninertia"], "meaninertia")
            _assert_bytes(partial["actuator_gainprm"], full["actuator_gainprm"], "Kp preserved")
            _assert_bytes(partial["actuator_biasprm"], full["actuator_biasprm"], "Kd preserved")
            _assert_bytes(wp.to_torch(env.sim.wp_data.qpos), before_data["qpos"], "qpos restored")
            _assert_bytes(wp.to_torch(env.sim.wp_data.qvel), before_data["qvel"], "qvel restored")
            _restore(model_arrays, partial)
            _restore(data_arrays, before_data)
    finally:
        env.close()


def _compare_envs(partial, full):
    # Independently stepped stock-vs-stock GPU worlds are not bit deterministic:
    # a 9-step control measured qvel <=9.8e-6, acceleration <=2.3e-3,
    # observations <=2.0e-6 and inverse-inertia constants <=2.1e-5 apart.
    # These trajectory-only bounds cover that backend floor. The same-world
    # constant oracle above separately checks CPU64 accuracy and byte preservation.
    atol = {"qpos": 1e-6, "qvel": 5e-5, "qacc_warmstart": 5e-3,
            "ctrl": 0.0, "xpos": 1e-6, "xquat": 1e-6, "M": 1e-5}
    for name in ("qpos", "qvel", "qacc_warmstart", "ctrl", "xpos", "xquat", "M"):
        torch.testing.assert_close(getattr(partial.sim.data, name)[...],
                                   getattr(full.sim.data, name)[...],
                                   rtol=2e-6, atol=atol[name],
                                   msg=lambda message, name=name: f"{name}: {message}")
    for name in DERIVED:
        torch.testing.assert_close(getattr(partial.sim.model, name)[...],
                                   getattr(full.sim.model, name)[...],
                                   rtol=2e-6, atol=5e-5, msg=name)
    _assert_close(partial.sim.model.stat.meaninertia[...],
                  full.sim.model.stat.meaninertia[...], "meaninertia")
    for name in partial.obs_buf:
        torch.testing.assert_close(partial.obs_buf[name], full.obs_buf[name],
                                   rtol=2e-6, atol=1e-5, msg=f"obs {name}")
    a, b = (env.recorder_manager.get_term("replay") for env in (partial, full))
    _assert_close(a.prev_xi, b.prev_xi, "replay current context")
    _assert_bytes(a.buffer.ptr, b.buffer.ptr, "replay pointer")
    _assert_bytes(partial.command_manager.get_term("motion").frame,
                  full.command_manager.get_term("motion").frame, "reference frame")


@pytest.mark.slow
@pytest.mark.parametrize("optimized", [False, True], ids=["stock-control", "partial"])
def test_partial_refresh_real_explicit_and_automatic_reset_trajectory_matches_stock(optimized):
    partial, full = _make_env(stock=not optimized), _make_env(stock=True)
    try:
        partial.reset(seed=73)
        full.reset(seed=73)
        # Both constructors execute the stock full refresh. Align backend sums
        # before stepping so unrelated initialization rounding cannot bias the
        # subsequent stock-control or partial-vs-stock comparison.
        _restore(_arrays(full.sim.wp_model), _snapshot(_arrays(partial.sim.wp_model)))
        _restore(_arrays(full.sim.wp_data), _snapshot(_arrays(partial.sim.wp_data)))
        _compare_envs(partial, full)
        ids = torch.tensor([6, 1, 3], device=partial.device)
        for iteration in range(9):
            if iteration == 2:
                torch.manual_seed(910)
                partial.reset(env_ids=ids)
                torch.manual_seed(910)
                full.reset(env_ids=ids)
                _compare_envs(partial, full)
            if iteration == 5:
                for env in (partial, full):
                    env.episode_length_buf[ids] = env.max_episode_length - 1
            action = (0.04 * torch.sin(torch.arange(27, device=partial.device) + iteration))[None].expand(8, -1)
            torch.manual_seed(1000 + iteration)
            result_partial = partial.step(action)
            torch.manual_seed(1000 + iteration)
            result_full = full.step(action)
            for index, name in ((1, "reward"), (2, "terminated"), (3, "timeout")):
                _assert_close(result_partial[index], result_full[index], name)
            if iteration == 5:
                assert partial.reset_buf[ids].all() and full.reset_buf[ids].all()
            _compare_envs(partial, full)
    finally:
        partial.close()
        full.close()

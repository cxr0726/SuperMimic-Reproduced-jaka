"""Evaluate one explicit ankle-pitch armature intervention on a frozen checkpoint.

The normal checkpoint guard runs against a matching, immutable Git snapshot.
Only after it passes, an isolated entity factory overrides the two pitch joints
and their actuator configurations. The original benchmark loop is unchanged.
Run each armature in a separate process, with identical data and num-envs.
--validate-only performs the contract/config/CPU-model checks without a GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path

from eval_legacy_khan_benchmark import PROJECT_DIR, export_revision, sha256

PITCH = ("Left_ankle_pitch_joint", "Right_ankle_pitch_joint")
ANKLES = ("Left_ankle_pitch_joint", "Left_ankle_roll_joint",
          "Right_ankle_pitch_joint", "Right_ankle_roll_joint")
FEET = ("Left_ankle_roll_Link", "Right_ankle_roll_Link")


@contextmanager
def armature_factory(khan, value):
    """Change construction inputs, never checkpoint validation or asset files."""
    original = khan.get_khan_mini_cfg

    def factory(*args, **kwargs):
        cfg = original(*args, **kwargs)
        spec_fn = cfg.spec_fn

        def overridden_spec():
            spec = spec_fn()
            for name in PITCH:
                spec.joint(name).armature = value
            return spec

        cfg.spec_fn = overridden_spec
        changed = []
        actuators = []
        for actuator in cfg.articulation.actuators:
            if len(actuator.target_names_expr) == 1 and actuator.target_names_expr[0] in PITCH:
                changed.append(actuator.target_names_expr[0])
                actuator = replace(actuator, armature=value)
            actuators.append(actuator)
        if set(changed) != set(PITCH) or len(changed) != 2:
            raise RuntimeError("Expected exactly two individually configured pitch actuators")
        cfg.articulation.actuators = tuple(actuators)
        return cfg

    khan.get_khan_mini_cfg = factory
    try:
        yield
    finally:
        khan.get_khan_mini_cfg = original


def cpu_model_check(model, baseline, value, mujoco, np):
    expected = np.array(baseline.dof_armature, copy=True)
    indices = []
    for name in PITCH:
        jid = mujoco.mj_name2id(baseline, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Missing baseline joint {name}")
        indices.append(int(baseline.jnt_dofadr[jid]))
    expected[indices] = value
    np.testing.assert_allclose(model.dof_armature, expected, rtol=0, atol=1e-12)
    return expected, indices


def verify_environment(env, baseline, value, mujoco, np):
    """Verify all DoFs, including unchanged roll joints, in CPU and Warp models."""
    robot = env.scene["robot"]
    model = env.sim.mj_model
    expected = np.array(baseline.dof_armature, copy=True)
    # Scene compilation prefixes names, so map each scalar joint by its suffix.
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name and name.rsplit("/", 1)[-1] in PITCH:
            expected[int(model.jnt_dofadr[jid])] = value
    if model.nv != baseline.nv:
        raise ValueError("Unexpected additional scene DoFs")
    np.testing.assert_allclose(model.dof_armature, expected, rtol=0, atol=1e-12)
    warp = env.sim.model.dof_armature[:].detach().cpu().numpy()
    worlds = warp.reshape(-1, model.nv)
    np.testing.assert_allclose(worlds, np.broadcast_to(expected, worlds.shape), rtol=1e-6, atol=1e-8)
    actuators = env.cfg.scene.entities["robot"].articulation.actuators
    config_values = {a.target_names_expr[0]: float(a.armature) for a in actuators}
    for name in PITCH:
        if not math.isclose(config_values[name], value, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Environment actuator configuration did not retain {name} override")
    if tuple(robot.joint_names) != tuple(
            mujoco.mj_id2name(baseline, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(1, baseline.njnt)):
        raise ValueError("Robot joint order differs from checkpoint's model")
    return {"all_cpu_dofs_verified": True, "all_warp_dofs_verified": True,
            "warp_world_rows": int(worlds.shape[0]), "warp_armature_shape": list(warp.shape),
            "cpu_dof_armature": model.dof_armature.tolist(),
            "actuator_config_armature": config_values}


class TraceCapture:
    """Read at the normal reward check, before any automatic reset or forward."""

    def __init__(self, env, names, selected, np, torch):
        self.env, self.np, self.torch = env, np, torch
        self.names, self.selected = names, set(selected)
        self.current = np.arange(env.num_envs)
        self.steps = np.zeros(env.num_envs, dtype=np.int64)
        self.records = {name: [] for name in selected}
        robot = env.scene["robot"]
        self.joint_names = list(robot.joint_names)
        self.feet = [robot.body_names.index(name) for name in FEET]
        self.flags = list(env.termination_manager.active_terms)
        original = env.reward_manager.compute

        def compute(dt):
            reward = original(dt=dt)
            self.capture(reward)
            return reward

        env.reward_manager.compute = compute

    def capture(self, reward):
        env, torch = self.env, self.torch
        robot, tm = env.scene["robot"], env.termination_manager
        self.steps += 1
        rows = [e for e, ci in enumerate(self.current)
                if ci < len(self.names) and self.names[ci] in self.selected]
        if rows:
            ids = torch.tensor(rows, device=env.device, dtype=torch.long)
            root_pose = env.sim.data.qpos[ids, :7].clone()
            root_pose[:, :3] -= env.scene.env_origins[ids]
            values = {
                "root_pos": root_pose[:, :3], "root_quat": root_pose[:, 3:7],
                "q": robot.data.joint_pos[ids], "qd": robot.data.joint_vel[ids],
                "raw_action": env.action_manager.action[ids],
                "target": robot.data.joint_pos_target[ids],
                "effort": robot.data.actuator_force[ids],
                "foot_origin_height": (robot.data.body_link_pos_w[ids][:, self.feet, 2]
                                       - env.scene.env_origins[ids, 2:3]),
                "foot_lin_vel_w": robot.data.body_link_lin_vel_w[ids][:, self.feet],
                "reward": reward[ids], "terminated": tm.terminated[ids],
                "timeout": tm.time_outs[ids], "overflow": env.sim.data.overflow[ids],
                **{f"termination_{name}": tm.get_term(name)[ids] for name in self.flags},
            }
            cpu = {key: tensor.detach().cpu().numpy().copy() for key, tensor in values.items()}
            for row, e in enumerate(rows):
                self.records[self.names[self.current[e]]].append(
                    {"step": int(self.steps[e]), **{k: v[row] for k, v in cpu.items()}})
        done = tm.dones.detach().cpu().numpy()
        self.current[done] += env.num_envs
        self.steps[done] = 0

    def save(self, directory, dt):
        np = self.np
        directory.mkdir(parents=True, exist_ok=True)
        summaries = {}
        for name, rows in self.records.items():
            if not rows:
                raise ValueError(f"No trace collected for {name}")
            arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
            if not all(np.isfinite(value).all() for value in arrays.values()):
                raise ValueError(f"Nonfinite trace values for {name}")
            np.testing.assert_array_equal(arrays["step"], np.arange(1, len(rows) + 1))
            path = directory / f"{name}.npz"
            np.savez_compressed(path, **arrays, joint_names=np.asarray(self.joint_names),
                                foot_body_names=np.asarray(FEET), dt=np.asarray(dt))
            joints = {}
            for joint in ANKLES:
                i = self.joint_names.index(joint)
                joints[joint] = {key: signal_metrics(arrays[key][:, i], dt, np)
                                 for key in ("q", "qd", "raw_action", "target", "effort")}
            summaries[name] = {
                "trace": str(path.resolve()), "trace_sha256": sha256(path), "steps": len(rows),
                "terminated": bool(arrays["terminated"][-1]),
                "timeout": bool(arrays["timeout"][-1]),
                "termination_flags": {key: bool(arrays[f"termination_{key}"][-1]) for key in self.flags},
                "overflow_bits_or": int(np.bitwise_or.reduce(arrays["overflow"].astype(np.int64))),
                "ankles": joints,
                "foot_origin_height_min_m": arrays["foot_origin_height"].min(axis=0).tolist(),
                "foot_horizontal_speed_rms_mps": np.sqrt(
                    np.mean(np.sum(arrays["foot_lin_vel_w"][..., :2] ** 2, axis=-1), axis=0)).tolist(),
            }
        return summaries


def signal_metrics(signal, dt, np):
    values = np.asarray(signal, dtype=np.float64)
    out = {"mean": float(values.mean()), "rms": float(np.sqrt(np.mean(values ** 2))),
           "max_abs": float(np.abs(values).max())}
    for order in (1, 2):
        difference = np.diff(values, n=order)
        out[f"difference_{order}_rms"] = (float(np.sqrt(np.mean(difference ** 2)))
                                           if difference.size else None)
    # Remove the mean and Hann-window to reduce finite-window leakage. This is
    # a 50 Hz diagnostic; vibrations above its 25 Hz Nyquist limit are unresolved.
    if len(values) >= 8:
        power = np.abs(np.fft.rfft((values - values.mean()) * np.hanning(len(values)))) ** 2
        frequencies = np.fft.rfftfreq(len(values), d=dt)
        out["power_fraction_at_or_above_10hz"] = (float(power[frequencies >= 10].sum() / power.sum())
                                                   if power.sum() > 0 else 0.0)
    else:
        out["power_fraction_at_or_above_10hz"] = None
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="0c6f13dc47392b6acbe58f0198b9d5cc31d454e4")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--npz-dir", type=Path, default=PROJECT_DIR / "data/npz/khan_mini/mixed")
    parser.add_argument("--armature", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=480)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clips", nargs="+", help="Optional subset, otherwise evaluate the entire directory")
    parser.add_argument("--trace-clips", nargs="*", default=[])
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.armature) or args.armature < 0 or args.num_envs < 1:
        parser.error("armature must be finite/nonnegative and num-envs must be positive")
    args.out = args.out.resolve()
    args.ckpt = args.ckpt.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    snapshot = Path(tempfile.mkdtemp(prefix=f"{args.out.stem}_revision_", dir=args.out.parent)) / "checkout"
    commit, hashes = export_revision(args.revision, snapshot)
    sys.path.insert(0, str(snapshot / "src"))

    import mujoco
    import numpy as np
    import torch

    import supermimic
    from supermimic.envs.mdp.motion_library import discover_npz
    from supermimic.eval import benchmark
    from supermimic.robots import khan_mini as khan
    from supermimic.robots.contracts import validate_robot_contract

    if not Path(supermimic.__file__).resolve().is_relative_to(snapshot / "src"):
        raise RuntimeError("Refusing imports outside the frozen snapshot")
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    if cfg.get("robot") != "khan_mini":
        raise ValueError("This experiment is specific to Khan_mini")
    # Never patch this guard. The snapshot, not the intervention, is compared
    # to the checkpoint's original complete source/asset contract.
    validate_robot_contract(checkpoint, "khan_mini", cfg["env"]["observation_layout"])
    policy, cfg = benchmark.load_policy_from_checkpoint(args.ckpt, "cpu")
    baseline = khan.get_khan_mini_spec().compile()
    files = [str(path.resolve()) for path in discover_npz(args.npz_dir.resolve())]
    if args.clips:
        wanted = set(args.clips)
        missing = wanted - {Path(path).stem for path in files}
        if missing:
            raise ValueError(f"Missing subset clips: {sorted(missing)}")
        files = [path for path in files if Path(path).stem in wanted]
    if not files or len({Path(path).stem for path in files}) != len(files):
        raise ValueError("Need nonempty motions with unique clip names")
    missing = set(args.trace_clips) - {Path(path).stem for path in files}
    if missing:
        raise ValueError(f"Missing trace clips: {sorted(missing)}")
    num_envs = min(args.num_envs, len(files))
    manifest = {
        "revision": commit, "snapshot": str(snapshot), "snapshot_sha256": hashes,
        "script_sha256": sha256(Path(__file__)), "checkpoint": str(args.ckpt),
        "checkpoint_sha256": sha256(args.ckpt), "iteration": int(checkpoint["iteration"]),
        "robot_contract": checkpoint["robot_contract"], "strict_original_contract_passed": True,
        "explicit_intervention": {"pitch_joint_names": list(PITCH), "armature": args.armature,
                                  "scope": "entity spec and actuator cfg only, after original guard"},
        "num_envs": num_envs, "device": args.device, "trace_clips": args.trace_clips,
        "npz_sha256": {path: sha256(Path(path)) for path in files},
        "protocol": {"original_benchmark_loop": True, "seed": 0, "start_frame": 0,
                     "noise": False, "domain_randomization": False, "actuator_delay": False,
                     "foot_friction": 1.0, "reset_root_z_offset": cfg["env"].get("reset_root_z_offset")},
        "trace_semantics": "One sample per 20 ms control step, before automatic reset. q/qd are post-integration; effort and foot link-derived kinematics retain benchmark's usual last-physics-step lag (5 ms). No extra forward or sensor refresh. Foot link origin height/velocity are kinematic diagnostics, not contact/sole-penetration measurements. Terminal states included; reset discontinuities excluded. FFT >=10 Hz is limited by 25 Hz Nyquist.",
    }
    with armature_factory(khan, args.armature):
        entity = khan.get_khan_mini_cfg()
        model = entity.spec_fn().compile()
        expected, indices = cpu_model_check(model, baseline, args.armature, mujoco, np)
        manifest["baseline_pitch_armature"] = baseline.dof_armature[indices].tolist()
        manifest["cpu_construction_check"] = {"all_dofs_verified": True,
                                                "expected_dof_armature": expected.tolist()}
        for a in entity.articulation.actuators:
            if a.target_names_expr[0] in PITCH:
                assert a.armature == args.armature
        if not args.validate_only:
            policy = policy.to(args.device)
            original_make_env = benchmark.make_env
            captures = []

            def make_env_with_capture(env_args, device):
                env = original_make_env(env_args, device=device)
                try:
                    manifest["environment_check"] = verify_environment(env, baseline, args.armature, mujoco, np)
                    manifest["max_step_reward"] = env.step_dt * sum(
                        max(0.0, term.weight) for term in env.cfg.rewards.values())
                    names = [clip.name for clip in env.command_manager.get_term("motion").library.clips]
                    if names != [Path(path).stem for path in files]:
                        raise ValueError("Benchmark library reordered clips")
                    if args.trace_clips:
                        captures.append(TraceCapture(env, names, args.trace_clips, np, torch))
                    return env
                except BaseException:
                    env.close()
                    raise

            benchmark.make_env = make_env_with_capture
            started = time.monotonic()
            try:
                results = benchmark.run_benchmark(
                    policy, files, device=args.device, num_envs=num_envs,
                    future_frames=cfg["env"]["future_frames"], history_length=policy.cfg.history,
                    observation_layout=policy.cfg.observation_layout, robot="khan_mini",
                    anchor_obs=cfg["env"].get("anchor_obs", False),
                    reset_root_z_offset=cfg["env"].get("reset_root_z_offset"))
            finally:
                benchmark.make_env = original_make_env
            if len(captures) != int(bool(args.trace_clips)):
                raise RuntimeError("Unexpected benchmark environment/trace construction count")
            trace_summary = (captures[0].save(args.out.parent / f"{args.out.stem}_traces", .02)
                             if captures else {})
            if len(results) != len(files) or {r.name for r in results} != {Path(p).stem for p in files}:
                raise ValueError("Benchmark output clips differ from the requested set")
            for result in results:
                assert 0 < result.steps <= result.frames + 1
                assert result.fail_step == (result.steps if result.failed else -1)
                assert math.isfinite(result.norm_return)
                assert all(math.isfinite(value) for value in result.terms.values())
                assert math.isclose(sum(result.terms.values()) / manifest["max_step_reward"],
                                    result.norm_return, abs_tol=2e-5)
                if result.name in trace_summary:
                    assert result.steps == trace_summary[result.name]["steps"]
                    assert result.failed == trace_summary[result.name]["terminated"]
            payload = {"summary": benchmark.summarize(results),
                       "clips": [asdict(result) for result in results], "trace_summary": trace_summary}
            args.out.write_text(json.dumps(payload, indent=2) + "\n")
            manifest["wall_time_s"] = time.monotonic() - started
            manifest["result_sha256"] = sha256(args.out)
            manifest["result_validation_passed"] = True
            print(json.dumps(payload["summary"], indent=2), flush=True)
    manifest["snapshot_unchanged"] = all(sha256(snapshot / path) == digest for path, digest in hashes.items())
    if not manifest["snapshot_unchanged"]:
        raise RuntimeError("Frozen snapshot changed during evaluation")
    manifest["checkpoint_unchanged"] = sha256(args.ckpt) == manifest["checkpoint_sha256"]
    if not manifest["checkpoint_unchanged"]:
        raise RuntimeError("Checkpoint changed during evaluation")
    manifest["validate_only"] = args.validate_only
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Validated isolated intervention; manifest:", args.out.with_suffix(".manifest.json"), flush=True)


if __name__ == "__main__":
    main()

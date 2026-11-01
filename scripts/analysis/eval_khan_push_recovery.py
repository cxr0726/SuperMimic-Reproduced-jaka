"""Evaluate a frozen 40k Khan policy on complete clips after velocity impulses.

Uses logs/khan_40k_robustness/baseline_source and baseline_assets, deliberately
independent of later training/reward/anchor changes in the active source tree.
Pushes are instantaneous horizontal root velocity increments (m/s), not forces.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def install_frozen_source(directory: Path) -> dict:
    manifest = json.loads((directory / "baseline_manifest.json").read_text())
    frozen_src = directory / "baseline_source" / "src" / "supermimic"
    frozen_asset = directory / "baseline_assets"
    for relative, expected in manifest["source_sha256"].items():
        if sha(frozen_src / relative) != expected:
            raise RuntimeError(f"Frozen source checksum mismatch: {relative}")
    for relative, expected in manifest["asset_sha256"].items():
        if sha(frozen_asset / relative) != expected:
            raise RuntimeError(f"Frozen asset checksum mismatch: {relative}")
    sys.path.insert(0, str(frozen_src.parent.resolve()))
    return manifest


def cpu(tensor):
    return tensor.detach().cpu().numpy().copy()


def run(args):
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    baseline_dir = (args.baseline_dir or args.out_dir).resolve()
    manifest = install_frozen_source(baseline_dir)
    import torch
    from mjlab.utils.lab_api.math import quat_apply_inverse

    import supermimic
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.motion_library import discover_npz
    from supermimic.envs.state_io import StateIO
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    from supermimic.robots.registry import get_robot_spec

    frozen_src = baseline_dir / "baseline_source" / "src"
    if not Path(supermimic.__file__).resolve().is_relative_to(frozen_src):
        raise RuntimeError("The active training package was imported instead of the frozen baseline")
    policy, config = load_policy_from_checkpoint(args.ckpt, args.device)
    checkpoint_metadata = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    iteration = int(checkpoint_metadata["iteration"])
    del checkpoint_metadata
    if iteration != 40000:
        raise ValueError("This frozen robustness protocol is specifically for the 40000 checkpoint")
    spec = get_robot_spec(config["robot"])
    if spec.anchor_body != "waist_yaw_Link" or len(spec.tracked_bodies) != 14:
        raise RuntimeError("Legacy checkpoint semantics were not preserved")
    files = [str(p) for p in discover_npz(args.npz_dir, names=args.clips)]
    file_names = [Path(p).stem for p in files]
    if set(file_names) != set(args.clips):
        raise ValueError("Missing requested motion clips")
    directions = {"forward": (1.0, 0.0), "backward": (-1.0, 0.0),
                  "left": (0.0, 1.0), "right": (0.0, -1.0)}
    cases = []
    for clip in args.clips:
        cases.append({"name": f"{clip}__baseline", "clip": clip, "direction": "none", "delta_v_mps": 0.0})
        for strength in args.strengths:
            for direction in directions:
                cases.append({"name": f"{clip}__{direction}_{strength:g}mps", "clip": clip,
                              "direction": direction, "delta_v_mps": strength})
    count = len(cases)
    env_args = SuperMimicEnvCfgArgs(
        npz_files=files, robot=spec.name, num_envs=count,
        future_frames=config["env"]["future_frames"], history_length=policy.cfg.history,
        episode_length_s=1e5, action_noise_std=0, obs_noise=False, reset_noise=False,
        domain_randomization=False, sampling_mode="start", record_transitions=False,
        seed=0, anchor_obs=config["env"].get("anchor_obs", False),
    )
    env = make_env(env_args, device=args.device)
    try:
        robot = env.scene["robot"]
        io = StateIO(env)
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        library_names = [clip.name for clip in lib.clips]
        clip_indices = torch.tensor([library_names.index(case["clip"]) for case in cases],
                                    device=args.device, dtype=torch.long)
        start_frames = lib.clip_start[clip_indices]
        cmd.forced_frame[:] = start_frames
        obs, _ = env.reset()
        # Every scenario is evaluated only through its first episode. A later
        # auto-reset remains the same reference clip and never enters scores.
        cmd.forced_frame[:] = start_frames
        dt = float(env.step_dt)
        record_steps = round(args.record_seconds / dt)
        push_step = round(args.push_time / dt) + 1
        hold_steps = math.ceil(args.recovery_hold / dt)
        frames = np.array([lib.clips[index].length for index in cpu(clip_indices)], dtype=np.int64)
        active = np.ones(count, dtype=bool)
        steps = np.zeros(count, dtype=np.int64)
        failed = np.zeros(count, dtype=bool)
        completed = np.zeros(count, dtype=bool)
        push_applied = np.zeros(count, dtype=bool)
        recovery_time = np.full(count, np.nan)
        recovery_absolute = np.full(count, np.nan)
        stable = np.zeros(count, dtype=np.int64)
        reward_sum = np.zeros(count)
        term_names = list(env.reward_manager.active_terms)
        term_sum = np.zeros((count, len(term_names)))
        peak_xy = np.zeros(count)
        peak_height = np.zeros(count)
        peak_tilt = np.zeros(count)
        peak_xy_after_push = np.zeros(count)
        peak_height_after_push = np.zeros(count)
        peak_tilt_after_push = np.zeros(count)
        overflow = np.zeros(count, dtype=np.int64)
        overflow_names = ["NEFC", "NJMAX_NNZ", "BROADPHASE", "NARROWPHASE", "CCD", "HFIELD",
                          "CONTACT_MATCH", "NVMAX", "EPA_HORIZON", "ITERATIONS", "LS_ITERATIONS", "TACTILE"]
        overflow_counts = np.zeros((count, len(overflow_names)), dtype=np.int64)
        overflow_first_time = np.full((count, len(overflow_names)), np.nan)
        failure_steps = np.full(count, -1, dtype=np.int64)
        terminal_steps = np.full(count, -1, dtype=np.int64)
        terminal_flags = [{} for _ in cases]
        actual_dv = np.zeros((count, 2))
        headings = np.full(count, np.nan)
        capture = {}
        current_step = 0
        pre_reset_states = {}
        video_qpos = [cpu(io.read()[:, :io.nq])]
        video_reference = [cpu(lib.ref_qpos_qvel(start_frames)[0])]
        last_qpos = video_qpos[0].copy()
        last_reference = video_reference[0].copy()
        camera_frames = torch.minimum(
            start_frames[None, :] + torch.arange(record_steps + 1, device=args.device)[:, None],
            lib.last_frame_of(start_frames)[None, :],
        )
        camera_roots = cpu(lib.ref_qpos_qvel(camera_frames.reshape(-1))[0][:, :3]).reshape(
            record_steps + 1, count, 3
        ).transpose(1, 0, 2)
        metric_rows = []
        original_compute = env.reward_manager.compute
        original_pre_reset = env.recorder_manager.record_pre_reset

        def compute_with_capture(dt):
            reward = original_compute(dt=dt)
            ref_pos = cmd.root_pos_w
            root_pos = cmd.robot_root_pos_w
            gravity = robot.data.gravity_vec_w
            rob_gz = quat_apply_inverse(cmd.robot_root_quat_w, gravity)[:, 2]
            ref_gz = quat_apply_inverse(cmd.root_quat_w, gravity)[:, 2]
            packed = torch.stack([
                reward, (root_pos[:, :2] - ref_pos[:, :2]).norm(dim=-1),
                (root_pos[:, 2] - ref_pos[:, 2]).abs(), (rob_gz - ref_gz).abs(),
                root_pos[:, 2], ref_pos[:, 2],
            ], dim=-1)
            capture["metrics"] = cpu(packed)
            capture["terms"] = cpu(env.reward_manager.term_rewards)
            capture["overflow"] = cpu(env.sim.data.overflow[:])
            capture["flags"] = {name: cpu(env.termination_manager.get_term(name))
                                for name in env.termination_manager.active_terms}
            if current_step <= record_steps:
                capture["state"] = cpu(io.read())
                capture["reference"] = cpu(lib.ref_qpos_qvel(cmd.frame)[0])
            return reward

        def pre_reset_with_capture(env_ids):
            if current_step <= record_steps:
                final = cpu(io.read())
                for index in env_ids.tolist():
                    pre_reset_states[index] = final[index]
            original_pre_reset(env_ids)

        env.reward_manager.compute = compute_with_capture
        env.recorder_manager.record_pre_reset = pre_reset_with_capture
        mass = float(env.sim.mj_model.body_mass.sum())
        strengths = np.array([case["delta_v_mps"] for case in cases])
        started = time.perf_counter()
        while active.any():
            current_step += 1
            if current_step == push_step:
                push_ids = np.flatnonzero(active & (strengths > 0))
                pushed = torch.tensor(push_ids, device=args.device)
                q = robot.data.root_link_quat_w[pushed]
                w, x, y, z = q.unbind(-1)
                yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                local = torch.tensor([directions[cases[index]["direction"]] for index in push_ids],
                                     device=args.device)
                delta = torch.tensor(strengths[push_ids], dtype=q.dtype, device=args.device)
                world = torch.stack([torch.cos(yaw) * local[:, 0] - torch.sin(yaw) * local[:, 1],
                                     torch.sin(yaw) * local[:, 0] + torch.cos(yaw) * local[:, 1]], -1)
                world *= delta[:, None]
                # Free-root translational qvel is in world coordinates. This
                # instantaneous injection leaves angular and joint velocities intact.
                before = env.sim.data.qvel[pushed, :2].clone()
                env.sim.data.qvel[pushed, :2] = before + world
                actual_dv[push_ids] = cpu(env.sim.data.qvel[pushed, :2] - before)
                headings[push_ids] = cpu(yaw)
                push_applied[push_ids] = True
                print(f"Applied velocity increments at t={args.push_time:g}s to {len(push_ids)} scenarios", flush=True)
            action = policy(obs["policy"], obs["goal"])
            obs, _, term, tout, _ = env.step(action)
            metrics = capture["metrics"]
            term_cpu, timeout_cpu = cpu(term), cpu(tout)
            done = (term_cpu | timeout_cpu) & active
            valid = active.copy()
            reward_sum[valid] += metrics[valid, 0]
            term_sum[valid] += capture["terms"][valid]
            steps[valid] += 1
            peak_xy[valid] = np.maximum(peak_xy[valid], metrics[valid, 1])
            peak_height[valid] = np.maximum(peak_height[valid], metrics[valid, 2])
            peak_tilt[valid] = np.maximum(peak_tilt[valid], metrics[valid, 3])
            overflow[valid] |= capture["overflow"][valid]
            for bit_index in range(len(overflow_names)):
                flagged = valid & ((capture["overflow"] & (1 << bit_index)) != 0)
                overflow_counts[flagged, bit_index] += 1
                first = flagged & np.isnan(overflow_first_time[:, bit_index])
                overflow_first_time[first, bit_index] = current_step * dt
            after = valid & push_applied
            peak_xy_after_push[after] = np.maximum(peak_xy_after_push[after], metrics[after, 1])
            peak_height_after_push[after] = np.maximum(peak_height_after_push[after], metrics[after, 2])
            peak_tilt_after_push[after] = np.maximum(peak_tilt_after_push[after], metrics[after, 3])
            good = after & (metrics[:, 1] < args.recovery_xy) & (metrics[:, 2] < args.recovery_height) & (metrics[:, 3] < args.recovery_tilt)
            stable[after] = np.where(good[after], stable[after] + 1, 0)
            recovered_now = after & np.isnan(recovery_time) & (stable >= hold_steps) & ~term_cpu
            recovery_time[recovered_now] = current_step * dt - args.push_time
            recovery_absolute[recovered_now] = current_step * dt
            for index in np.flatnonzero(valid):
                metric_rows.append((index, current_step, current_step * dt, *metrics[index, 1:4],
                                    bool(term_cpu[index]), bool(timeout_cpu[index]), bool(push_applied[index]),
                                    int(capture["overflow"][index])))
            if current_step <= record_steps:
                for index in np.flatnonzero(valid):
                    if done[index]:
                        np.testing.assert_array_equal(capture["state"][index], pre_reset_states[index])
                    last_qpos[index] = capture["state"][index, :io.nq]
                    last_reference[index] = capture["reference"][index]
                video_qpos.append(last_qpos.copy())
                video_reference.append(last_reference.copy())
            for index in np.flatnonzero(done):
                failed[index] = bool(term_cpu[index])
                completed[index] = not failed[index] and bool(capture["flags"]["clip_end"][index])
                failure_steps[index] = current_step if failed[index] else -1
                terminal_steps[index] = current_step
                terminal_flags[index] = {name: bool(value[index]) for name, value in capture["flags"].items()}
            active[done] = False
            if current_step % 1000 == 0:
                print(f"step {current_step} ({current_step*dt:.1f}s) | active {active.sum()}/{count} | "
                      f"failed {failed.sum()} | completed {completed.sum()} | wall {time.perf_counter()-started:.1f}s", flush=True)
            if current_step > int(frames.max()) + 5:
                raise RuntimeError("A scenario exceeded the full reference duration without ending")
        while len(video_qpos) < record_steps + 1:
            video_qpos.append(last_qpos.copy())
            video_reference.append(last_reference.copy())
        qpos_video = np.stack(video_qpos, axis=1)
        reference_video = np.stack(video_reference, axis=1)
        normalized_return = reward_sum / (.1 * np.maximum(steps, 1))
        rows = []
        for index, case in enumerate(cases):
            rows.append({
                "scenario": index, **case, "frames": int(frames[index]), "executed_steps": int(steps[index]),
                "completed_full_clip": bool(completed[index]), "failed": bool(failed[index]),
                "failure_time_s": float(steps[index] * dt) if failed[index] else None,
                "norm_return": float(normalized_return[index]), "completion_fraction": float(steps[index]/frames[index]),
                "push_applied": bool(push_applied[index]), "push_time_s": args.push_time,
                "push_heading_yaw_rad": float(headings[index]) if push_applied[index] else None,
                "actual_delta_v_world_mps": actual_dv[index].tolist(),
                "equivalent_impulse_magnitude_Ns": float(mass * case["delta_v_mps"]) if push_applied[index] else 0,
                "recovered": bool(np.isfinite(recovery_time[index])),
                "recovery_time_after_push_s": float(recovery_time[index]) if np.isfinite(recovery_time[index]) else None,
                "recovery_absolute_time_s": float(recovery_absolute[index]) if np.isfinite(recovery_absolute[index]) else None,
                "recovered_and_completed": bool(np.isfinite(recovery_time[index]) and completed[index]),
                "peak_xy_error_m": float(peak_xy[index]), "peak_height_error_m": float(peak_height[index]),
                "peak_tilt_feature_error": float(peak_tilt[index]),
                "peak_xy_error_after_push_m": float(peak_xy_after_push[index]) if push_applied[index] else None,
                "peak_height_error_after_push_m": float(peak_height_after_push[index]) if push_applied[index] else None,
                "peak_tilt_error_after_push": float(peak_tilt_after_push[index]) if push_applied[index] else None,
                "terminal_flags": terminal_flags[index], "observed_overflow_bits_or": int(overflow[index]),
                "overflow_control_step_counts": {name: int(overflow_counts[index, j])
                                                 for j, name in enumerate(overflow_names)
                                                 if overflow_counts[index, j]},
                "first_observed_overflow_time_s": {name: float(overflow_first_time[index, j])
                                                   for j, name in enumerate(overflow_names)
                                                   if np.isfinite(overflow_first_time[index, j])},
                "reward_terms_per_step": {name: float(term_sum[index, j] / max(steps[index], 1))
                                          for j, name in enumerate(term_names)},
            })
        baselines = [index for index, case in enumerate(cases) if case["delta_v_mps"] == 0]
        representatives = []
        for clip in args.clips:
            candidates = [row for row in rows if row["clip"] == clip and row["push_applied"]]
            successes = [row for row in candidates if row["recovered_and_completed"]]
            failures = [row for row in candidates if row["failed"]]
            if successes:
                chosen = max(successes, key=lambda row: row["delta_v_mps"])
                representatives.append((chosen["scenario"], "recovered_and_completed"))
            if failures:
                visible = [row for row in failures if row["failure_time_s"] <= args.record_seconds]
                chosen = min(visible or failures, key=lambda row: (row["delta_v_mps"], row["failure_time_s"]))
                representatives.append((chosen["scenario"], "failed"))

        def save_video_npz(path, indices, sample_outcomes):
            np.savez_compressed(
                path, names=np.array([cases[index]["name"] for index in indices]),
                source_clip=np.array([cases[index]["clip"] for index in indices]),
                robot_qpos=qpos_video[indices], ref_qpos=reference_video[indices],
                camera_ref_root=camera_roots[indices], lengths=np.full(len(indices), record_steps+1),
                failure_steps=failure_steps[indices], terminal_steps=terminal_steps[indices],
                failed=failed[indices], norm_return=normalized_return[indices],
                robot=np.array(spec.name), fps=np.array(1/dt), checkpoint_iteration=np.array(iteration),
                direction=np.array([cases[index]["direction"] for index in indices]),
                delta_v_mps=np.array([cases[index]["delta_v_mps"] for index in indices]),
                push_time_s=np.full(len(indices), args.push_time), actual_delta_v_world_mps=actual_dv[indices],
                sample_outcome=np.array(sample_outcomes), freeze_after_terminal=np.array(True),
                baseline_asset_dir=np.array(str(baseline_dir/'baseline_assets')),
            )
        save_video_npz(out/'baseline_rollout.npz', baselines, ["baseline"]*len(baselines))
        if representatives:
            save_video_npz(out/'push_rollout.npz', [index for index, _ in representatives],
                           [outcome for _, outcome in representatives])
        with (out/'trace.csv').open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["scenario", "step", "time_s", "xy_error_m", "height_error_m", "tilt_feature_error", "terminated", "timeout", "push_applied", "observed_overflow_bits"])
            writer.writerows(metric_rows)
        csv_rows = [{key: value for key, value in row.items()
                     if key not in ("reward_terms_per_step", "terminal_flags", "actual_delta_v_world_mps",
                                    "overflow_control_step_counts", "first_observed_overflow_time_s")}
                    for row in rows]
        with (out/'results.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
        grouped = []
        for clip in args.clips:
            for strength in [0, *args.strengths]:
                group = [row for row in rows if row["clip"] == clip and row["delta_v_mps"] == strength]
                recovered_times = [row["recovery_time_after_push_s"] for row in group if row["recovered_and_completed"]]
                grouped.append({"clip": clip, "delta_v_mps": strength, "cases": len(group),
                                "completed": sum(row["completed_full_clip"] for row in group),
                                "recovered_and_completed": sum(row["recovered_and_completed"] for row in group),
                                "mean_norm_return": float(np.mean([row["norm_return"] for row in group])),
                                "mean_recovery_time_s": float(np.mean(recovered_times)) if recovered_times else None})
        payload = {
            "checkpoint": str(args.ckpt.resolve()), "checkpoint_sha256": sha(args.ckpt),
            "evaluation_script_sha256": sha(Path(__file__)),
            "checkpoint_iteration": iteration, "training_config": config,
            "runtime_versions": {name: importlib.metadata.version(name) for name in
                                 ("torch", "mjlab", "mujoco", "mujoco-warp", "warp-lang")},
            "dataset_sha256": {str(Path(path).resolve()): sha(Path(path)) for path in files},
            "frozen_manifest": str(baseline_dir/'baseline_manifest.json'), "actual_imported_package": str(supermimic.__file__),
            "robot_mass_kg": mass, "seed": 0, "num_envs": count, "dt": dt,
            "record_seconds": args.record_seconds, "push_time_s": args.push_time,
            "push_semantics": "Instantaneous world translational free-root qvel increment, oriented in each robot's current root heading at push time; no angular/joint velocity change. Values are m/s; mass*delta_v is equivalent total linear impulse in N*s, not applied force.",
            "recovery_criterion": {"root_xy_error_m_lt": args.recovery_xy, "root_height_error_m_lt": args.recovery_height,
                                   "tilt_projected_gravity_difference_lt": args.recovery_tilt,
                                   "continuous_hold_s": args.recovery_hold,
                                   "recovery_time": "Time after impulse until the end of the first full continuous qualifying window. A recovered case can still fail later; recovered_and_completed requires the whole clip."},
            "protocol": {"first_frame_start": True, "action_noise": 0, "obs_noise": False,
                         "reset_noise": False, "domain_randomization": False, "foot_friction": 1,
                         "episode_length_s": 1e5, "root_height_threshold": .25, "tilt_feature_threshold": .8,
                         "normalization": "sum(reward)/(0.1*executed_steps)",
                         "baseline_semantics": manifest["robot_semantics"], "waist_full_size_m": manifest["waist_full_size_m"]},
            "representatives": [{"scenario": index, "outcome": outcome, "case": rows[index]}
                                for index, outcome in representatives],
            "failure_sample_available": any(row["failed"] for row in rows),
            "overflow_notes": {
                "observed_nonzero_case_count": sum(bool(row["observed_overflow_bits_or"]) for row in rows),
                "bit_definitions": {"1": "NEFC", "2": "NJMAX_NNZ", "4": "BROADPHASE", "8": "NARROWPHASE",
                                    "16": "CCD", "32": "HFIELD", "64": "CONTACT_MATCH", "128": "NVMAX",
                                    "256": "EPA_HORIZON", "512": "ITERATIONS", "1024": "LS_ITERATIONS"},
                "cases": [{"name": row["name"], "observed_bits": row["observed_overflow_bits_or"],
                           "control_steps_with_each_flag": row["overflow_control_step_counts"]}
                          for row in rows if row["observed_overflow_bits_or"]],
                "note": "Solver iteration/linesearch limit flags differ from contact/constraint capacity overflows.",
                "sticky_bit_semantics": "MuJoCo Warp overflow bits are latched until explicitly cleared/reset. Counts report control steps observing a set bit, not distinct occurrences. First observed times identify when each bit first appeared in this episode.",
            },
            "summary_by_clip_and_strength": grouped, "cases": rows,
            "limits": "One noiseless run per direction/strength; direction counts are not statistical success probabilities. Recovery follows the moving reference, not zero absolute speed. Overflow is sampled at control steps; trajectories freeze after first termination for display.",
        }
        (out/'results.json').write_text(json.dumps(payload, indent=2, default=str)+'\n')
        snapshot_path = out/'eval_khan_push_recovery_snapshot.py'
        if Path(__file__).resolve() != snapshot_path.resolve():
            shutil.copy2(Path(__file__), snapshot_path)
        print(json.dumps({"summary": grouped, "representatives": payload["representatives"]}, indent=2), flush=True)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', type=Path, default=Path('logs/Khan_lafan1_4096/ckpt_0040000.pt'))
    parser.add_argument('--npz-dir', type=Path, default=Path('data/npz/khan_mini/lafan1'))
    parser.add_argument('--out-dir', type=Path, default=Path('logs/khan_40k_robustness'))
    parser.add_argument('--baseline-dir', type=Path, default=None,
                        help='Reuse another output directory\'s frozen source/assets for a separate experiment')
    parser.add_argument('--clips', nargs='+', default=['dance2_subject1', 'run2_subject4', 'jumps1_subject2'])
    parser.add_argument('--strengths', nargs='+', type=float, default=[.3,.6,1,1.5])
    parser.add_argument('--push-time', type=float, default=3)
    parser.add_argument('--record-seconds', type=float, default=15)
    parser.add_argument('--recovery-xy', type=float, default=.15)
    parser.add_argument('--recovery-height', type=float, default=.10)
    parser.add_argument('--recovery-tilt', type=float, default=.15)
    parser.add_argument('--recovery-hold', type=float, default=.5)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if any(strength <= 0 for strength in args.strengths):
        parser.error("Push strengths must be positive; the zero-strength baseline is added automatically")
    if args.push_time < 0 or min(args.record_seconds, args.recovery_hold) <= 0:
        parser.error("Push time must be nonnegative; recording and recovery-hold durations must be positive")
    import torch

    with torch.no_grad():
        run(args)


if __name__ == '__main__':
    main()

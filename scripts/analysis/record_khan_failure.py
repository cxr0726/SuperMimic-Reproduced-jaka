"""Record benchmark-layout failure traces without changing policy or physics.

The hooks only read tensors. Rewards/termination retain mjlab's normal 0.005 s
derived-kinematics lag; terminal qpos/qvel are saved before automatic reset.
The displayed robot AND reference freeze after the first termination. A separate
continuous reference root is stored solely for a common comparison camera.

This entry point uses the current training model and checkpoint contract. Legacy
14-body/waist-anchor checkpoints require their matching isolated source/assets;
use eval_legacy_khan_benchmark.py for a legacy full-clip benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.sim_context import SimContext
from supermimic.envs.state_io import StateIO
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.robots.registry import get_robot_spec


def cpu(tensor):
    return tensor.detach().cpu().numpy().copy()


@torch.no_grad()
def record(ckpt: Path, npz_dir: Path, out_dir: Path, clips: list[str], seconds: float, device: str):
    policy, config = load_policy_from_checkpoint(ckpt, device)
    iteration = int(torch.load(ckpt, map_location="cpu", weights_only=False)["iteration"])
    spec = get_robot_spec(config["robot"])
    if spec.name != "khan_mini":
        raise ValueError("This diagnostic expects Khan_mini")
    files = [str(p) for p in discover_npz(npz_dir)]
    names = [Path(p).stem for p in files]
    selected = [names.index(name) for name in clips]
    ids = torch.tensor(selected, dtype=torch.long, device=device)
    args = SuperMimicEnvCfgArgs(
        npz_files=files, num_envs=len(files), robot=spec.name,
        history_length=policy.cfg.history, future_frames=config["env"]["future_frames"],
        anchor_obs=config["env"].get("anchor_obs", False), episode_length_s=1e5,
        action_noise_std=0.0, obs_noise=False, reset_noise=False, domain_randomization=False,
        sampling_mode="start", record_transitions=False, seed=0,
    )
    env = make_env(args, device=device)
    try:
        robot = env.scene["robot"]
        io = StateIO(env)
        context = SimContext(env, spec.foot_geom_regex, spec.com_body)
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        if [c.name for c in lib.clips] != names:
            raise ValueError("The benchmark library order differs from discovered files")
        n = len(files)
        count = len(clips)
        current = torch.arange(n, device=device)
        nxt = current + n
        # Match run_benchmark: after each first episode an exhausted environment
        # automatically restarts the final clip, but is excluded from scoring.
        cmd.forced_frame[:] = lib.clip_start[current]
        obs, _ = env.reset()
        cmd.forced_frame[:] = lib.clip_start[torch.clamp(nxt, max=n - 1)]
        dt = float(env.step_dt)
        max_reward = dt * sum(max(0.0, term.weight) for term in env.cfg.rewards.values())
        steps_limit = round(seconds / dt)
        alive = np.ones(count, dtype=bool)
        failure_steps = np.full(count, -1, dtype=np.int64)
        terminal_steps = np.full(count, -1, dtype=np.int64)
        rewards = np.zeros(count, dtype=np.float64)
        executed_steps = np.zeros(count, dtype=np.int64)
        q0 = cpu(io.read()[ids, :io.nq])
        reference0 = cpu(lib.ref_qpos_qvel(lib.clip_start[ids])[0])
        trajectory = [q0]
        references = [reference0]
        camera_frames = lib.clip_start[ids][None, :] + torch.arange(
            steps_limit + 1, device=device
        )[:, None]
        camera_frames = torch.minimum(camera_frames, lib.last_frame_of(lib.clip_start[ids])[None, :])
        camera_ref = cpu(lib.ref_qpos_qvel(camera_frames.reshape(-1))[0][:, :3]).reshape(
            steps_limit + 1, count, 3
        ).transpose(1, 0, 2)
        soft = cpu(robot.data.soft_joint_pos_limits[ids])
        hard = cpu(robot.data.joint_pos_limits[ids])
        from supermimic.robots.khan_mini import load_profile

        profile = load_profile()
        joint_names = list(spec.joint_names)
        effort_limit = np.array([profile["effort_limit"][name] for name in joint_names])
        trace = []
        transition = {key: [] for key in (
            "s", "s_next", "action", "xi", "ref_root_pos", "ref_root_quat", "clip_name",
            "step", "terminated", "height_error_at_check", "tilt_error_at_check",
            "joint_pos_target", "actual_effort", "qpos", "qvel", "ref_qpos", "reward",
            "overflow", "root_pos_at_check", "root_quat_at_check",
        )}
        capture = {}
        terminal_captures = {}

        def snapshot(reward):
            root_pos = cmd.robot_root_pos_w[ids]
            ref_pos = cmd.root_pos_w[ids]
            gravity = robot.data.gravity_vec_w[ids]
            rob_gz = quat_apply_inverse(cmd.robot_root_quat_w[ids], gravity)[:, 2]
            ref_gz = quat_apply_inverse(cmd.root_quat_w[ids], gravity)[:, 2]
            return {
                "s_next": cpu(io.read()[ids]),
                "ref_qpos": cpu(lib.ref_qpos_qvel(cmd.frame[ids])[0]),
                "ref_root_pos": cpu(ref_pos - env.scene.env_origins[ids]),
                "ref_root_quat": cpu(cmd.root_quat_w[ids]),
                "root_pos_at_check": cpu(root_pos - env.scene.env_origins[ids]),
                "root_quat_at_check": cpu(cmd.robot_root_quat_w[ids]),
                "height_error_at_check": cpu((root_pos[:, 2] - ref_pos[:, 2]).abs()),
                "height_signed_error": cpu(root_pos[:, 2] - ref_pos[:, 2]),
                "tilt_error_at_check": cpu((rob_gz - ref_gz).abs()),
                "xy_error": cpu((root_pos[:, :2] - ref_pos[:, :2]).norm(dim=-1)),
                "root_height": cpu(root_pos[:, 2]), "ref_height": cpu(ref_pos[:, 2]),
                "robot_gravity_z": cpu(rob_gz), "ref_gravity_z": cpu(ref_gz),
                "joint_pos_target": cpu(robot.data.joint_pos_target[ids]),
                "actual_effort": cpu(robot.data.actuator_force[ids]),
                "overflow": cpu(env.sim.data.overflow[ids]),
                "reward": cpu(reward[ids]),
                "flags": {name: cpu(env.termination_manager.get_term(name)[ids])
                          for name in env.termination_manager.active_terms},
            }

        original_compute = env.reward_manager.compute
        original_pre_reset = env.recorder_manager.record_pre_reset

        def compute_with_capture(dt):
            reward = original_compute(dt=dt)
            capture.clear()
            capture.update(snapshot(reward))
            return reward

        def pre_reset_with_capture(env_ids):
            # Capture actual final qpos/qvel before auto-reset overwrites them.
            for e in env_ids.tolist():
                if e in selected:
                    terminal_captures[e] = cpu(io.read()[e])
            original_pre_reset(env_ids)

        env.reward_manager.compute = compute_with_capture
        env.recorder_manager.record_pre_reset = pre_reset_with_capture
        for step in range(1, steps_limit + 1):
            state_before = cpu(io.read()[ids])
            xi = cpu(context.read()[ids])
            action = policy(obs["policy"], obs["goal"])
            action_selected = cpu(action[ids])
            obs, _, term, tout, _ = env.step(action)
            terminated = cpu(term[ids])
            ended = cpu((term | tout)[ids])
            shown_qpos = trajectory[-1].copy()
            shown_ref = references[-1].copy()
            for i, name in enumerate(clips):
                if not alive[i]:
                    continue
                next_state = capture["s_next"][i]
                if ended[i]:
                    final_state = terminal_captures[selected[i]]
                    np.testing.assert_array_equal(next_state, final_state)
                shown_qpos[i] = next_state[:io.nq]
                shown_ref[i] = capture["ref_qpos"][i]
                rewards[i] += capture["reward"][i]
                executed_steps[i] += 1
                q = next_state[7:io.nq]
                soft_excess = np.maximum(soft[i, :, 0] - q, 0) + np.maximum(q - soft[i, :, 1], 0)
                target = capture["joint_pos_target"][i]
                target_excess = np.maximum(hard[i, :, 0] - target, 0) + np.maximum(target - hard[i, :, 1], 0)
                force_ratio = np.abs(capture["actual_effort"][i]) / effort_limit
                row = {
                    "name": name, "step": step, "time_s": step * dt,
                    "terminated": bool(terminated[i]), "timeout": bool(ended[i] and not terminated[i]),
                    **{key: float(capture[key][i]) for key in (
                        "height_error_at_check", "height_signed_error", "tilt_error_at_check", "xy_error",
                        "root_height", "ref_height", "robot_gravity_z", "ref_gravity_z", "reward",
                    )},
                    **{f"term/{key}": bool(value[i]) for key, value in capture["flags"].items()},
                    "soft_limit_count": int((soft_excess > 1e-6).sum()),
                    "soft_limit_max_rad": float(soft_excess.max()),
                    "soft_limit_worst_joint": joint_names[int(soft_excess.argmax())],
                    "target_outside_hard_count": int((target_excess > 1e-6).sum()),
                    "target_outside_hard_max_rad": float(target_excess.max()),
                    "target_outside_hard_worst_joint": joint_names[int(target_excess.argmax())],
                    "effort_saturated_count": int((force_ratio >= 0.99).sum()),
                    "effort_max_fraction": float(force_ratio.max()),
                    "effort_max_joint": joint_names[int(force_ratio.argmax())],
                    "action_abs_mean": float(np.abs(action_selected[i]).mean()),
                    "action_abs_max": float(np.abs(action_selected[i]).max()),
                    "joint_velocity_abs_max": float(np.abs(next_state[io.nq + 6:]).max()),
                    "overflow": int(capture["overflow"][i]),
                }
                trace.append(row)
                for key, values in transition.items():
                    value = {
                        "s": state_before[i], "action": action_selected[i], "xi": xi[i],
                        "clip_name": name, "step": step, "terminated": bool(terminated[i]),
                        "qpos": next_state[:io.nq], "qvel": next_state[io.nq:],
                    }.get(key, capture[key][i] if key in capture else None)
                    values.append(value)
                if ended[i]:
                    terminal_steps[i] = step
                    failure_steps[i] = step if terminated[i] else -1
                    alive[i] = False
            trajectory.append(shown_qpos)
            references.append(shown_ref)
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"{iteration:07d}"
        rollout_path = out_dir / f"rollout_{suffix}.npz"
        np.savez_compressed(
            rollout_path, names=np.array(clips), robot_qpos=np.stack(trajectory, axis=1),
            ref_qpos=np.stack(references, axis=1), camera_ref_root=camera_ref,
            lengths=np.full(count, steps_limit + 1), failure_steps=failure_steps, terminal_steps=terminal_steps,
            failed=failure_steps >= 0, norm_return=rewards / (max_reward * executed_steps),
            robot=np.array(spec.name), fps=np.array(1 / dt), checkpoint_iteration=np.array(iteration),
            freeze_after_terminal=np.array(True), joint_names=np.array(joint_names),
        )
        transitions_path = out_dir / f"transitions_{suffix}.npz"
        np.savez_compressed(transitions_path, **{key: np.asarray(value) for key, value in transition.items()},
                            joint_names=np.array(joint_names), soft_limits=soft, hard_limits=hard,
                            effort_limits=effort_limit, dt=np.array(dt))
        csv_path = out_dir / f"trace_{suffix}.csv"
        with csv_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(trace[0]))
            writer.writeheader()
            writer.writerows(trace)
        summaries = []
        for i, name in enumerate(clips):
            rows = [row for row in trace if row["name"] == name]
            failure = next((row for row in rows if row["terminated"]), None)
            summaries.append({
                "name": name, "recorded_steps": len(rows), "failed_within_window": failure is not None,
                "failure_time_s": None if failure is None else failure["time_s"], "failure": failure,
                "norm_return_over_executed_prefix": float(rewards[i] / (max_reward * executed_steps[i])),
                "max_height_error": max(row["height_error_at_check"] for row in rows),
                "max_tilt_error": max(row["tilt_error_at_check"] for row in rows),
                "fraction_steps_with_soft_limit_violation": float(np.mean([row["soft_limit_count"] > 0 for row in rows])),
                "max_soft_limit_violation_rad": max(row["soft_limit_max_rad"] for row in rows),
                "mean_saturated_actuator_count": float(np.mean([row["effort_saturated_count"] for row in rows])),
                "max_effort_fraction": max(row["effort_max_fraction"] for row in rows),
                "overflow_bits_or": int(np.bitwise_or.reduce([row["overflow"] for row in rows])),
            })
        payload = {
            "checkpoint": str(ckpt.resolve()), "iteration": iteration, "seconds": seconds,
            "num_envs": n, "all_clip_names_in_environment_order": names, "dt": dt,
            "protocol": "Original benchmark layout and reset queue; first-frame start, no noise/DR, foot friction 1.0.",
            "capture_semantics": "qpos/qvel are post-integration; height/tilt at termination check use normal mjlab derived kinematics lagging 0.005 s. No extra forward is invoked. Actual terminal raw state is separately confirmed in pre-reset hook. Video robot and reference freeze after the first termination; camera reference stays continuous.",
            "limits_note": "Soft limits are reward boundaries, not physical hard limits. Effort saturation threshold is 99% of configured cap, sampled once per control step, not every physics substep.",
            "rollout": str(rollout_path.resolve()), "transitions": str(transitions_path.resolve()),
            "trace_csv": str(csv_path.resolve()), "clips": summaries,
        }
        result_path = out_dir / f"diagnostics_{suffix}.json"
        result_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--npz-dir", type=Path, default=Path("data/npz/khan_mini/lafan1"))
    parser.add_argument("--out-dir", type=Path, default=Path("logs/Khan_lafan1_4096/analysis/failure_videos"))
    parser.add_argument("--clips", nargs="+", default=["walk3_subject4", "run2_subject4", "jumps1_subject2"])
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    record(args.ckpt, args.npz_dir, args.out_dir, args.clips, args.seconds, args.device)


if __name__ == "__main__":
    main()

"""Evaluate named Khan NPZs and retain true terminal poses for OOD videos.

Uses one environment per motion and disables automatic reset. If a benchmark
fails, a separate diagnostic removes only height/tilt termination and records
the complete motion without resetting the fallen robot.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.torch import configure_torch_backends

from supermimic.envs import SuperMimicEnvCfgArgs
from supermimic.envs.mdp.motion_library import discover_npz, read_npz_reference
from supermimic.envs.state_io import StateIO
from supermimic.envs.tracking_env_cfg import make_env_cfg
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.robots.registry import get_robot_spec


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@torch.no_grad()
def record(policy, path: Path, out: Path, device: str, continuous: bool = False,
           velocity_reference: Path | None = None, reset_root_z_offset: float | None = None) -> dict:
    mode = "continuous_no_fall_reset" if continuous else "benchmark_stop"
    args = SuperMimicEnvCfgArgs(
        npz_files=[str(path)], robot="khan_mini", num_envs=1, future_frames=5,
        history_length=5, observation_layout="isaac_khan", anchor_obs=False,
        episode_length_s=1e5, action_noise_std=0, obs_noise=False, reset_noise=False,
        domain_randomization=False, sampling_mode="start", record_transitions=False, seed=0,
        actuator_delay=False, reset_root_z_offset=reset_root_z_offset,
    )
    cfg = make_env_cfg(args)
    cfg.auto_reset = False
    if continuous:
        del cfg.terminations["root_height"]
        del cfg.terminations["root_tilt"]
    env = ManagerBasedRlEnv(cfg, device=device)
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        io = StateIO(env)
        cmd.forced_frame[:] = lib.clip_start[0]
        obs, _ = env.reset()
        frame_count = lib.clips[0].length
        initial_frame_offset = int(cmd.frame[0] - lib.clip_start[0])
        # clip_end is latched by the command update and observed by the next
        # termination check. Current reset(dt=0) starts at frame0; older source
        # advanced to frame1 during reset. Retain the actual repository protocol.
        expected_steps = frame_count + 1 - initial_frame_offset
        max_reward = env.step_dt * sum(max(0., term.weight) for term in cfg.rewards.values())
        alternate_velocity = None
        if velocity_reference is not None:
            _, bodies = read_npz_reference(
                velocity_reference, robot_name="khan_mini", root_body_name="base_link",
                body_names=get_robot_spec("khan_mini").tracked_bodies,
            )
            alternate_velocity = bodies["lin"]
            if len(alternate_velocity) != frame_count:
                raise ValueError("Velocity-only comparison needs the same reference frame count")

        def cpu(value):
            return value.detach().cpu().numpy().copy()

        # Capture the exact reference used for this reward/termination, before
        # step() updates the command for the next policy call.
        original_compute = env.reward_manager.compute
        capture = {}

        def capture_reward(dt):
            reward = original_compute(dt=dt)
            gravity = env.scene["robot"].data.gravity_vec_w
            actual_gz = quat_apply_inverse(cmd.robot_root_quat_w, gravity)[:, 2]
            ref_gz = quat_apply_inverse(cmd.root_quat_w, gravity)[:, 2]
            ref_qpos, ref_qvel = lib.ref_qpos_qvel(cmd.frame)
            capture.update(
                state=cpu(io.read())[0], reference=cpu(ref_qpos)[0],
                reference_qvel=cpu(ref_qvel)[0], frame=int(cmd.frame[0]),
                height_error_at_check=float((cmd.robot_root_pos_w[:, 2]-cmd.root_pos_w[:, 2]).abs()[0]),
                tilt_error_at_check=float((actual_gz-ref_gz).abs()[0]),
                terms=cpu(env.reward_manager.term_rewards)[0],
                overflow=int(env.sim.data.overflow[0]),
            )
            if alternate_velocity is not None:
                actual = cpu(cmd.robot_body_lin_vel_w)[0]
                source = cpu(cmd.body_lin_vel_w)[0]
                original = .02*np.exp(-np.mean(np.sum((actual-source)**2, axis=-1)))
                index = list(env.reward_manager.active_terms).index("motion_body_lin_vel")
                np.testing.assert_allclose(original, capture["terms"][index], atol=1e-7)
                target = alternate_velocity[capture["frame"]]
                capture["alternate_body_lin_reward"] = float(
                    .02*np.exp(-np.mean(np.sum((actual-target)**2, axis=-1))))
            return reward

        env.reward_manager.compute = capture_reward
        robot_pose = [cpu(io.read())[0, :io.nq]]
        reference_pose = [cpu(lib.ref_qpos_qvel(lib.clip_start)[0])[0]]
        joint_velocity_errors, actions, trace = [], [], []
        reference_frames = [int(lib.clip_start[0])]
        rewards = []
        first_failure = -1
        normal_terminated = False
        normal_timeout = False
        for step in range(1, expected_steps+1):
            action = policy(obs["policy"], obs["goal"])
            obs, reward, term, timeout, _ = env.step(action)
            state, reference = capture["state"], capture["reference"]
            if not np.isfinite(state).all():
                raise RuntimeError(f"{path.stem}: nonfinite state at step {step}")
            # auto_reset=False retains the raw terminal state; assert that the
            # diagnostic hook and the returned environment contain the same pose.
            np.testing.assert_array_equal(cpu(io.read())[0], state)
            robot_pose.append(state[:io.nq])
            reference_pose.append(reference)
            reference_frames.append(capture["frame"])
            actions.append(cpu(action)[0])
            rewards.append(float(reward[0]))
            joint_velocity_errors.append(state[io.nq+6:]-capture["reference_qvel"][6:])
            crossed = capture["height_error_at_check"] > .25 or capture["tilt_error_at_check"] > .8
            if first_failure < 0 and crossed:
                first_failure = step
            row = {
                "step": step, "time_s": step*env.step_dt,
                "used_reference_frame": capture["frame"],
                "height_error_at_check": capture["height_error_at_check"],
                "tilt_error_at_check": capture["tilt_error_at_check"],
                "overflow": capture["overflow"],
                "reward": float(reward[0]),
                "terminated": bool(term[0]), "timeout": bool(timeout[0]),
                "terms": {name: float(value) for name, value in zip(
                    env.reward_manager.active_terms, capture["terms"], strict=True)},
            }
            if alternate_velocity is not None:
                row["alternate_body_lin_reward"] = capture["alternate_body_lin_reward"]
                row["alternate_total_reward"] = (
                    row["reward"] - row["terms"]["motion_body_lin_vel"]
                    + row["alternate_body_lin_reward"])
            trace.append(row)
            if bool((term | timeout)[0]):
                normal_terminated, normal_timeout = bool(term[0]), bool(timeout[0])
                break
        steps = len(rewards)
        poses, references = np.stack(robot_pose), np.stack(reference_pose)
        errors = poses-references
        xy_error = np.linalg.norm(errors[:, :2], axis=-1)
        joint_errors = errors[:, 7:]
        joint_rmse = np.sqrt(np.mean(joint_errors**2, axis=-1))
        per_joint_rmse = np.sqrt(np.mean(joint_errors[1:]**2, axis=0))
        spec = get_robot_spec("khan_mini")
        mean_terms = {name: float(np.mean([t["terms"][name] for t in trace]))
                      for name in env.reward_manager.active_terms}
        norm_return = float(sum(rewards)/(max_reward*steps))
        assert np.isclose(sum(mean_terms.values())/max_reward, norm_return, atol=2e-5)
        assert steps <= expected_steps and (normal_terminated or normal_timeout)
        complete = steps == expected_steps and not normal_terminated
        result = {
            "name": path.stem, "mode": mode, "frames": frame_count, "steps": steps,
            "fps": 1/env.step_dt, "source_motion_duration_s": (frame_count-1)*env.step_dt,
            "benchmark_expected_steps": expected_steps,
            "executed_duration_s": steps*env.step_dt, "completion_fraction": steps/expected_steps,
            "failed": first_failure >= 0, "first_failure_step": first_failure,
            "first_failure_time_s": first_failure*env.step_dt if first_failure >= 0 else None,
            "benchmark_terminated": normal_terminated, "clip_timeout": normal_timeout,
            "full_clip_executed": complete, "norm_return": norm_return,
            "terms": mean_terms, "mean_xy_error_m": float(xy_error[1:].mean()),
            "final_xy_error_m": float(xy_error[-1]), "max_xy_error_m": float(xy_error.max()),
            "mean_abs_z_error_m": float(np.abs(errors[1:, 2]).mean()),
            "joint_rmse_rad": float(np.sqrt(np.mean(joint_errors[1:]**2))),
            "joint_velocity_rmse_rad_s": float(np.sqrt(np.mean(np.array(joint_velocity_errors)**2))),
            "worst_joint_rmse": sorted(
                [{"joint": name, "rmse_rad": float(value)} for name, value in zip(
                    spec.joint_names, per_joint_rmse, strict=True)],
                key=lambda row: row["rmse_rad"], reverse=True)[:8],
            "action_abs_max": float(np.abs(actions).max()),
            "overflow_bits_or": int(np.bitwise_or.reduce([t["overflow"] for t in trace])),
            "initial_reference_frame_offset": initial_frame_offset,
            "auto_reset": False, "reference_pairing": "Exact reward-time reference; post-integration robot",
            "termination_note": "Checks retain normal mjlab 0.005s derived-kinematics lag; no extra forward",
        }
        if alternate_velocity is not None:
            result["velocity_only_counterfactual"] = {
                "reference": str(velocity_reference.resolve()),
                "reference_sha256": sha256(velocity_reference),
                "same_physical_states_actions_and_other_rewards": True,
                "source_norm_return": norm_return,
                "alternate_norm_return": float(sum(t["alternate_total_reward"] for t in trace)
                                               / (max_reward*steps)),
                "source_body_lin_reward_mean": mean_terms["motion_body_lin_vel"],
                "alternate_body_lin_reward_mean": float(np.mean(
                    [t["alternate_body_lin_reward"] for t in trace])),
                "note": "Exact reward-time states; only reference body_lin_vel changes, no physics change",
            }
        out.mkdir(parents=True, exist_ok=True)
        prefix = out / f"{path.stem}_{mode}"
        np.savez_compressed(
            prefix.with_suffix(".npz"), names=np.array([path.stem]),
            robot_qpos=poses[None], ref_qpos=references[None], lengths=np.array([len(poses)]),
            fps=np.array(1/env.step_dt), robot=np.array("khan_mini"), mode=np.array(mode),
            norm_return=np.array([norm_return]), failed=np.array([first_failure>=0]),
            failure_steps=np.array([first_failure]), trace_xy_error=xy_error[None],
            trace_joint_rmse=joint_rmse[None], reference_frames=np.array(reference_frames)[None],
            actions=np.array(actions)[None], rewards=np.array(rewards)[None],
            joint_names=np.array(spec.joint_names), freeze_after_terminal=np.array(False),
        )
        result["rollout"] = str(prefix.with_suffix(".npz").resolve())
        prefix.with_suffix(".json").write_text(json.dumps({**result, "trace": trace}, indent=2)+"\n")
        print(json.dumps(result, indent=2), flush=True)
        return result
    finally:
        env.close()


def main(ckpt: Path, npz_dir: Path, out_dir: Path, device: str = "cuda:0",
         velocity_reference_dir: Path | None = None) -> None:
    configure_torch_backends()
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    if cfg.get("robot") != "khan_mini" or policy.cfg.observation_layout != "isaac_khan":
        raise ValueError("Expected current Khan isaac_khan checkpoint")
    files = discover_npz(npz_dir)
    if not files:
        raise ValueError("No NPZ motions found")
    rows = []
    for path in files:
        alternate = velocity_reference_dir / path.name if velocity_reference_dir else None
        primary = record(policy, path, out_dir, device, velocity_reference=alternate,
                         reset_root_z_offset=cfg["env"].get("reset_root_z_offset"))
        row = {"benchmark": primary}
        if primary["failed"]:
            row["continuous_diagnostic"] = record(
                policy, path, out_dir, device, continuous=True,
                reset_root_z_offset=cfg["env"].get("reset_root_z_offset"),
            )
        rows.append(row)
    manifest = {
        "checkpoint": str(ckpt.resolve()), "checkpoint_sha256": sha256(ckpt),
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "evaluation_script_sha256": sha256(Path(__file__)),
        "motion_sha256": {str(p.resolve()): sha256(p) for p in files},
        "protocol": {"num_envs_per_motion": 1, "seed": 0, "noise": False,
                     "domain_randomization": False, "foot_friction": 1., "max_step_reward": .112,
                     "auto_reset": False, "root_height_threshold": .25, "root_tilt_threshold": .8},
        "clips": rows,
    }
    (out_dir / "ood_evaluation.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    tyro.cli(main)

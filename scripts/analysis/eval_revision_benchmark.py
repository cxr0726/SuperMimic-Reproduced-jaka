"""Run an unmodified benchmark from a Git revision with its source and assets.

The checkpoint contract must match the exported revision. The active checkout is
never changed. Manifests retain source, asset, dataset and checkpoint hashes.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from eval_legacy_khan_benchmark import PROJECT_DIR, export_revision, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--npz-dir", type=Path, default=PROJECT_DIR / "data/npz/khan_mini/lafan1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=36)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    args.out = args.out.resolve()
    args.ckpt = args.ckpt.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    snapshot = Path(tempfile.mkdtemp(prefix=f"{args.out.stem}_revision_", dir=args.out.parent)) / "checkout"
    commit, hashes = export_revision(args.revision, snapshot)
    sys.path.insert(0, str(snapshot / "src"))

    import mujoco
    import torch

    import supermimic
    from supermimic.envs.mdp.motion_library import discover_npz
    from supermimic.eval.benchmark import (
        load_policy_from_checkpoint,
        run_benchmark,
        save_results,
        summarize,
    )
    from supermimic.robots.contracts import validate_robot_contract
    from supermimic.robots.khan_mini import DEFAULT_ASSET_DIR
    from supermimic.robots.registry import get_robot_spec

    if not Path(supermimic.__file__).resolve().is_relative_to(snapshot / "src"):
        raise RuntimeError("Evaluation imported code outside the revision snapshot")
    if not DEFAULT_ASSET_DIR.resolve().is_relative_to(snapshot):
        raise RuntimeError("Evaluation imported robot assets outside the revision snapshot")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    robot = get_robot_spec(cfg.get("robot", "g1"))
    if robot.name != "khan_mini":
        raise ValueError("This asset export currently supports Khan only")
    layout = cfg["env"].get("observation_layout", "supermimic")
    if "observation_layout" in inspect.signature(validate_robot_contract).parameters:
        validate_robot_contract(ck, robot.name, layout)
    else:
        validate_robot_contract(ck, robot.name)
    finite = {
        name: all(bool(torch.isfinite(value).all()) for value in ck[name].values())
        for name in ("policy", "wm", "critic")
    }
    if not all(finite.values()):
        raise ValueError(f"Nonfinite checkpoint parameters: {finite}")
    from supermimic.envs.mdp.rewards import TrackingRewardCfg

    reward_cfg = asdict(TrackingRewardCfg.for_joint_names(robot.joint_names))
    max_reward = reward_cfg["dt"] * sum(max(0.0, value) for key, value in reward_cfg.items() if key.startswith("w_"))
    model = robot.load_model()
    waist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "waist_yaw_Link_45_collision")
    if waist < 0:
        raise ValueError("Missing expected waist collision geometry")
    files = [str(path.resolve()) for path in discover_npz(args.npz_dir.resolve())]
    if not files:
        raise ValueError("No NPZ motions found")
    manifest = {
        "revision": commit, "checkpoint": str(args.ckpt), "checkpoint_sha256": sha256(args.ckpt),
        "iteration": int(ck["iteration"]), "config": cfg, "robot_contract": ck["robot_contract"],
        "snapshot": str(snapshot), "snapshot_sha256": hashes,
        "num_envs": min(args.num_envs, len(files)), "device": args.device,
        "protocol": {"seed": 0, "noise": False, "domain_randomization": False,
                     "actuator_delay": False,
                     "reset_root_z_offset": cfg["env"].get("reset_root_z_offset"),
                     "foot_friction": 1.0, "start_frame": 0, "one_episode_per_clip": True,
                     "max_step_reward": max_reward},
        "waist_collision_full_size_m": (2 * model.geom_size[waist]).tolist(),
        "npz_sha256": {file: sha256(Path(file)) for file in files},
        "checkpoint_model_parameters_finite": finite,
        "libraries": {"torch": torch.__version__, "mujoco": mujoco.__version__},
    }
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    del ck, model
    gc.collect()
    print(f"Checkpoint {manifest['iteration']} | revision {commit} | anchor {robot.anchor_body} | "
          f"tracked {len(robot.tracked_bodies)} | layout {layout} | max reward {max_reward}", flush=True)
    policy, cfg = load_policy_from_checkpoint(args.ckpt, args.device)
    kwargs = {
        "device": args.device, "num_envs": args.num_envs,
        "future_frames": cfg["env"]["future_frames"], "anchor_obs": cfg["env"].get("anchor_obs", False),
        "robot": robot.name, "history_length": policy.cfg.history,
    }
    if "observation_layout" in inspect.signature(run_benchmark).parameters:
        kwargs["observation_layout"] = layout
    if "reset_root_z_offset" in inspect.signature(run_benchmark).parameters:
        kwargs["reset_root_z_offset"] = cfg["env"].get("reset_root_z_offset")
    started = time.monotonic()
    results = run_benchmark(policy, files, **kwargs)
    save_results(results, args.out)
    summary = summarize(results)
    frames = sum(clip.frames for clip in results)
    steps = sum(clip.steps for clip in results)
    assert len(results) == len(files) == len({clip.name for clip in results})
    assert summary["failed"] == sum(clip.failed for clip in results)
    for clip in results:
        # Current reset starts at frame 0; clip_end is observed by the next
        # termination check, so a complete episode may execute frames+1 steps.
        # Older snapshots started at frame 1 and executed frames steps.
        assert 0 < clip.steps <= clip.frames + 1
        assert clip.fail_step == (clip.steps if clip.failed else -1)
        assert math.isfinite(clip.norm_return)
        assert all(math.isfinite(value) for value in clip.terms.values())
        assert math.isclose(sum(clip.terms.values()) / max_reward, clip.norm_return, abs_tol=2e-5)
    unchanged = all(sha256(snapshot / rel) == value for rel, value in hashes.items())
    assert unchanged
    validation = {
        "all_checks_passed": True, "snapshot_unchanged": unchanged,
        "wall_time_s": time.monotonic() - started,
        "clips": len(results), "failed": summary["failed"],
        "total_frames": frames, "total_executed_steps": steps,
        "completion_fraction": steps / frames,
        "mean_norm_return": sum(clip.norm_return for clip in results) / len(results),
        "step_weighted_norm_return": sum(clip.norm_return * clip.steps for clip in results) / steps,
        "full_clip_weighted_return": sum(clip.norm_return * clip.steps for clip in results) / frames,
        "failed_clips": [{"name": clip.name, "fail_step": clip.fail_step,
                          "fail_time_s": clip.fail_step * reward_cfg["dt"],
                          "completion_fraction": clip.steps / clip.frames}
                         for clip in results if clip.failed],
    }
    args.out.with_suffix(".validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(f"{len(results)} clips | failed {summary['failed']} | mean norm return "
          f"{summary['mean_norm_return']:.6f} | step-weighted {summary['step_weighted_norm_return']:.6f}")
    print(f"Completion fraction: {steps / frames:.6%}")
    print("saved", args.out)


if __name__ == "__main__":
    main()

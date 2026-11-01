"""Evaluate legacy Khan checkpoints with an isolated, versioned source/asset tree.

The active training checkout is never changed. Git history must contain the
requested revision; data, checkpoints and generated snapshots remain external.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

LEGACY_REVISION = "fcad4a5f69b3b3c896b559314d1cff430b81990f"
PROJECT_DIR = Path(__file__).resolve().parents[2]
ARCHIVE_PATHS = ("src/supermimic", "assets/robots/khan_mini", "scripts/eval_benchmark.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_revision(revision: str, snapshot: Path) -> tuple[str, dict[str, str]]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"], cwd=PROJECT_DIR,
        text=True,
    ).strip()
    # The snapshot contains exactly the requested commit, including meshes. Never
    # obtain files from the working tree, whose training semantics may differ.
    archive = subprocess.check_output(
        ["git", "archive", "--format=tar", commit, *ARCHIVE_PATHS], cwd=PROJECT_DIR,
    )
    snapshot.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(snapshot, filter="data")
    hashes = {
        str(path.relative_to(snapshot)): sha256(path)
        for path in sorted(snapshot.rglob("*")) if path.is_file()
    }
    return commit, hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--npz-dir", type=Path, default=PROJECT_DIR / "data/npz/khan_mini/lafan1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--revision", default=LEGACY_REVISION)
    parser.add_argument("--num-envs", type=int, default=36)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    args.out = args.out.resolve()
    args.ckpt = args.ckpt.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # A fresh directory lets repeated evaluations retain their provenance and
    # avoids reusing a partially exported or manually modified snapshot.
    snapshot = Path(tempfile.mkdtemp(
        prefix=f"{args.out.stem}_legacy_", dir=args.out.parent,
    )) / "checkout"
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
    from supermimic.robots.khan_mini import DEFAULT_ASSET_DIR
    from supermimic.robots.registry import get_robot_spec

    if not Path(supermimic.__file__).resolve().is_relative_to(snapshot / "src"):
        raise RuntimeError("Legacy snapshot import failed; refusing to use current training code")
    if not DEFAULT_ASSET_DIR.resolve().is_relative_to(snapshot):
        raise RuntimeError("Legacy assets were not loaded from the isolated snapshot")
    robot = get_robot_spec("khan_mini")
    if len(robot.tracked_bodies) != 14 or robot.anchor_body != "waist_yaw_Link":
        raise ValueError("The requested revision does not have legacy 14-body/waist-anchor semantics")
    model = robot.load_model()
    waist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "waist_yaw_Link_45_collision")
    if waist < 0:
        raise ValueError("Legacy model is missing the expected waist collision geometry")
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    finite = {
        module: all(bool(torch.isfinite(value).all()) for value in checkpoint[module].values())
        for module in ("policy", "wm", "critic")
    }
    if not all(finite.values()):
        raise ValueError(f"Checkpoint contains nonfinite model parameters: {finite}")
    if checkpoint["config"].get("robot") != "khan_mini":
        raise ValueError("This runner only supports legacy Khan checkpoints")
    if checkpoint.get("robot_contract") is not None:
        raise ValueError("A checkpoint with a modern robot contract requires the current benchmark")
    iteration = int(checkpoint["iteration"])
    del checkpoint
    files = [str(path.resolve()) for path in discover_npz(args.npz_dir.resolve())]
    if not files:
        raise ValueError(f"No motion NPZ files found in {args.npz_dir}")
    manifest = {
        "revision": commit,
        "checkpoint": str(args.ckpt), "checkpoint_sha256": sha256(args.ckpt),
        "iteration": iteration, "snapshot": str(snapshot), "snapshot_sha256": hashes,
        "robot": "khan_mini", "tracked_bodies": list(robot.tracked_bodies),
        "root_body": robot.root_body, "anchor_body": robot.anchor_body,
        "waist_collision_full_size_m": (2 * model.geom_size[waist]).tolist(),
        "num_envs": min(args.num_envs, len(files)), "device": args.device,
        "protocol": {"seed": 0, "noise": False, "domain_randomization": False,
                     "foot_friction": 1.0, "start_frame": 0, "one_episode_per_clip": True},
        "npz_sha256": {file: sha256(Path(file)) for file in files},
        "checkpoint_model_parameters_finite": finite,
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Legacy checkpoint iteration {iteration}; source {commit}; anchor {robot.anchor_body}", flush=True)
    print(f"Tracked bodies {len(robot.tracked_bodies)}; waist full size {manifest['waist_collision_full_size_m']}", flush=True)
    policy, cfg = load_policy_from_checkpoint(args.ckpt, args.device)
    results = run_benchmark(
        policy, files, device=args.device, num_envs=args.num_envs,
        future_frames=cfg["env"]["future_frames"],
        anchor_obs=cfg["env"].get("anchor_obs", False), robot="khan_mini",
        history_length=policy.cfg.history,
    )
    save_results(results, args.out)
    summary = summarize(results)
    frames = sum(clip.frames for clip in results)
    steps = sum(clip.steps for clip in results)
    # Validate both the aggregate arithmetic and the per-step reward decomposition.
    assert len(results) == len(files) == len({clip.name for clip in results})
    assert summary["failed"] == sum(clip.failed for clip in results)
    for clip in results:
        assert 0 < clip.steps <= clip.frames
        assert clip.fail_step == (clip.steps if clip.failed else -1)
        assert math.isfinite(clip.norm_return)
        assert math.isclose(sum(clip.terms.values()) / 0.1, clip.norm_return, abs_tol=2e-5)
    manifest_unchanged = all(sha256(snapshot / rel) == value for rel, value in hashes.items())
    assert manifest_unchanged
    validation = {
        "all_checks_passed": True, "snapshot_unchanged": manifest_unchanged,
        "clips": len(results), "failed": summary["failed"],
        "total_frames": frames, "total_executed_steps": steps,
        "completion_fraction": steps / frames,
        "mean_norm_return": sum(clip.norm_return for clip in results) / len(results),
        "step_weighted_norm_return": sum(clip.norm_return * clip.steps for clip in results) / steps,
        "full_clip_weighted_return": sum(clip.norm_return * clip.steps for clip in results) / frames,
        "failed_clips": [{"name": clip.name, "fail_step": clip.fail_step,
                          "fail_time_s": clip.fail_step * 0.02,
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

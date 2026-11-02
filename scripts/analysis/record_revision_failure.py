"""Record terminal-aware trajectories with the recorder from a matching Git revision."""

from __future__ import annotations

import argparse
import json
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path

from eval_legacy_khan_benchmark import PROJECT_DIR, export_revision, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--npz-dir", type=Path, default=PROJECT_DIR / "data/npz/khan_mini/lafan1")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--clips", nargs="+", required=True)
    parser.add_argument("--seconds", type=float, default=250.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt = args.ckpt.resolve()
    snapshot = Path(tempfile.mkdtemp(prefix="record_revision_", dir=args.out_dir)) / "checkout"
    commit, hashes = export_revision(args.revision, snapshot)
    recorder_relative = "scripts/analysis/record_khan_failure.py"
    recorder = snapshot / recorder_relative
    recorder.parent.mkdir(parents=True, exist_ok=True)
    recorder.write_bytes(subprocess.check_output(
        ["git", "show", f"{commit}:{recorder_relative}"], cwd=PROJECT_DIR,
    ))
    hashes[recorder_relative] = sha256(recorder)
    sys.path.insert(0, str(snapshot / "src"))

    import torch

    import supermimic
    from supermimic.envs.mdp.motion_library import discover_npz
    from supermimic.robots.contracts import validate_robot_contract
    from supermimic.robots.khan_mini import DEFAULT_ASSET_DIR

    if not Path(supermimic.__file__).resolve().is_relative_to(snapshot / "src"):
        raise RuntimeError("Recorder imported code outside the frozen revision")
    if not DEFAULT_ASSET_DIR.resolve().is_relative_to(snapshot):
        raise RuntimeError("Recorder imported assets outside the frozen revision")
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    validate_robot_contract(checkpoint, checkpoint["config"]["robot"])
    iteration = int(checkpoint["iteration"])
    files = discover_npz(args.npz_dir.resolve())
    missing = set(args.clips) - {p.stem for p in files}
    if missing:
        raise ValueError(f"Missing clips: {sorted(missing)}")
    manifest = {
        "revision": commit, "snapshot": str(snapshot), "snapshot_sha256": hashes,
        "checkpoint": str(args.ckpt), "checkpoint_sha256": sha256(args.ckpt),
        "iteration": iteration, "robot_contract": checkpoint["robot_contract"],
        "npz_sha256": {str(p.resolve()): sha256(p) for p in files},
        "selected_clips": args.clips, "seconds": args.seconds,
        "protocol": "Original revision recorder: 36 benchmark environments, first-frame rollout; clip windows are cut only during rendering.",
        "asset_dir": str(DEFAULT_ASSET_DIR),
    }
    manifest_path = args.out_dir / f"recording_{iteration:07d}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    del checkpoint
    namespace = runpy.run_path(str(recorder), run_name="revision_failure_recorder")
    namespace["record"](
        args.ckpt, args.npz_dir.resolve(), args.out_dir, args.clips, args.seconds, args.device,
    )
    assert all(sha256(snapshot / name) == digest for name, digest in hashes.items())
    manifest["snapshot_unchanged_after_recording"] = True
    outputs = (
        f"rollout_{iteration:07d}.npz", f"transitions_{iteration:07d}.npz",
        f"trace_{iteration:07d}.csv", f"diagnostics_{iteration:07d}.json",
    )
    manifest["output_sha256"] = {name: sha256(args.out_dir / name) for name in outputs}
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print("Recording manifest:", manifest_path, flush=True)


if __name__ == "__main__":
    main()

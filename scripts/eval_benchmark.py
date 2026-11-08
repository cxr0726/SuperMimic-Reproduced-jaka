"""Evaluate a checkpoint on every clip of a motion set (paper Appendix F protocol).

  uv run python scripts/eval_benchmark.py --ckpt logs/e2/ckpt_latest.pt --npz-dir data/npz/lafan1
  uv run python scripts/eval_benchmark.py --ckpt CHECKPOINT --library LIBRARY.pt --num-envs 2048
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch
import tyro

from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import (
    benchmark_source,
    load_policy_from_checkpoint,
    run_benchmark,
    save_results,
    summarize,
)
from supermimic.robots.contracts import robot_contract
from supermimic.robots.registry import get_robot_spec


def main(ckpt: Path, npz_dir: Path | None = None, clips: list[str] | None = None, clip_list: Path | None = None, manifest: Path | None = None, num_envs: int = 1024, device: str = "cuda:0", out: Path | None = None, library: Path | None = None, reference_fk_compile: bool = False) -> None:
    if library is not None and any(value is not None for value in (npz_dir, clips, clip_list, manifest)):
        raise ValueError("--library evaluates the entire packed library; do not combine NPZ/filter options")
    if clip_list is not None:
        clips = [line.strip() for line in clip_list.read_text().splitlines() if line.strip()]
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    robot = get_robot_spec(cfg.get("robot", "g1"))
    if library is not None:
        if library.suffix != ".pt" or not library.is_file():
            raise ValueError("--library must name an existing packed .pt file")
        files = [str(library.resolve())]
    else:
        npz_dir = npz_dir or robot.default_npz_dir
        files = [str(f) for f in discover_npz(npz_dir, names=clips, manifest=manifest)]
    actual_envs, compact = benchmark_source(files, num_envs)
    print(f"Benchmark device={device}; environments={actual_envs}; compact={compact}; reference FK compile={reference_fk_compile}", flush=True)
    started = time.monotonic()
    results = run_benchmark(
        policy, files, device=device, num_envs=num_envs,
        future_frames=cfg["env"]["future_frames"], anchor_obs=cfg["env"].get("anchor_obs", False),
        robot=robot.name, history_length=policy.cfg.history,
        observation_layout=policy.cfg.observation_layout,
        reset_root_z_offset=cfg["env"].get("reset_root_z_offset"),
        reference_fk_compile=reference_fk_compile,
    )
    s = summarize(results)
    print(f"\n{s['clips']} clips | failed {s['failed']} | mean norm return {s['mean_norm_return']:.4f} | step-weighted {s['step_weighted_norm_return']:.4f}")
    if "terms" in s:
        print("per-step term means (max 0.02 each for the 1.0-weight terms, 0.01 for root terms):")
        for k, v in s["terms"].items():
            print(f"  {k:28s} {v:.4f}")
    worst = sorted(results, key=lambda r: r.norm_return)[:10]
    for r in worst:
        print(f"  {r.name:32s} ret {r.norm_return:.3f} steps {r.steps}/{r.frames} {'FAIL@' + str(r.fail_step) if r.failed else ''}")
    out = out or ckpt.parent / f"benchmark_{ckpt.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, out)
    elapsed = time.monotonic() - started
    def sha256(path):
        with Path(path).open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    provenance = {
        "checkpoint": str(ckpt.resolve()), "checkpoint_sha256": sha256(ckpt),
        "library": str(library.resolve()) if library is not None else None,
        "library_sha256": sha256(library) if library is not None else None,
        "npz_dir": str(npz_dir.resolve()) if npz_dir is not None else None,
        "num_envs": actual_envs, "device": device, "compact": compact,
        "reference_fk_compile": reference_fk_compile, "wall_time_s": elapsed,
        "executed_steps": sum(r.steps for r in results),
        "reference_frames": sum(r.frames for r in results),
        "robot_contract": robot_contract(robot.name, observation_layout=policy.cfg.observation_layout),
        "protocol": {"seed": 0, "start_frame": 0, "episodes_per_clip": 1,
                     "noise": False, "domain_randomization": False, "actuator_delay": False,
                     "deterministic_policy": True, "bitwise_physics_guaranteed": False,
                     "reset_root_z_offset": cfg["env"].get("reset_root_z_offset")},
        "torch_version": torch.__version__,
    }
    if torch.device(device).type == "cuda":
        provenance["gpu"] = torch.cuda.get_device_name(device)
        provenance["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        provenance["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        provenance["memory_note"] = "Torch allocator counters exclude MuJoCo Warp allocations."
        provenance["cuda_matmul_fp32_precision"] = torch.backends.cuda.matmul.fp32_precision
    out.with_suffix(".manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("saved", out)


if __name__ == "__main__":
    tyro.cli(main)

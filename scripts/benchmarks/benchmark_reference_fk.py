"""Measure compact reference FK with identical packed data and frame queries.

Cold compilation is reported separately. Both CUDA-event and synchronized host
latencies are measured because eager FK is often launch-bound. This is a
component benchmark, not proof of end-to-end training throughput or convergence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.robots.registry import get_robot_spec


def tensors(bodies):
    return {"pos_w": bodies.pos_w, "quat_w": bodies.quat_w,
                "lin_vel_w": bodies.lin_vel_w, "ang_vel_w": bodies.ang_vel_w}


def graph_counts():
    from torch._dynamo.utils import counters
    return {key: dict(value) for key, value in counters.items() if key in ("stats", "graph_break", "frames")}


def run_timed(library, frames):
    torch.cuda.synchronize()
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start = time.perf_counter()
    start_event.record()
    result = library.bodies(frames)
    end_event.record()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1000, start_event.elapsed_time(end_event)


def summarize(samples):
    return {"mean_ms": statistics.mean(samples), "median_ms": statistics.median(samples),
                "min_ms": min(samples), "max_ms": max(samples), "samples_ms": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--robot", choices=("g1", "khan_mini"), default="khan_mini")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--policy-batch", type=int, default=4096)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.num_envs, args.policy_batch, args.horizon, args.samples, args.warmup) < 1:
        parser.error("sizes, sample count and warmup must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(153)
    torch.backends.cuda.matmul.allow_tf32 = False
    spec = get_robot_spec(args.robot)
    common = {"npz_files": [args.library], "body_names": spec.tracked_bodies,
                  "default_joint_pos": torch.zeros(spec.num_joints), "device": device,
                  "compact": True, "robot_name": args.robot, "root_body_name": spec.root_body}
    eager = MotionLibrary(**common, reference_fk_compile=False)
    compiled = MotionLibrary(**common, reference_fk_compile=True)
    assert eager.num_frames == compiled.num_frames
    eligible = (eager.clip_len - args.horizon).clamp_min(0)
    if not bool(eligible.sum()):
        parser.error("No clip contains a complete H+1 reference window")
    clips = torch.multinomial(eligible.float(), args.policy_batch, replacement=True)
    starts = eager.clip_start[clips] + (torch.rand(args.policy_batch, device=device) * eligible[clips]).long()
    rollout = (starts[None] + torch.arange(args.horizon + 1, device=device)[:, None]).reshape(-1)
    queries = {"environment": torch.randint(eager.num_frames, (args.num_envs,), device=device),
                   "imagined_reference_window": rollout}
    before = graph_counts()
    with args.library.open("rb") as stream:
        library_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {"timestamp": datetime.now(UTC).isoformat(), "python": platform.python_version(),
                  "torch": torch.__version__, "cuda": torch.version.cuda,
                  "gpu": torch.cuda.get_device_name(device), "library": str(args.library.resolve()),
                  "library_sha256": library_sha256,
                  "clips": eager.num_clips, "frames": eager.num_frames, "storage_dtype": str(eager._ref.dtype),
                  "compute_dtype": str(eager.dtype), "arguments": vars(args), "measurements": {},
                  "first_call_note": "First call in this process; an existing Inductor disk cache may be reused.",
                  "window_query_note": "H+1 contiguous frames from eligible starts with enough remaining frames; no terminal padding. This is not the replay buffer's learned visitation distribution.",
                  "limits": "Component-only, one GPU. Same dequantized references; fp16 quantization error is shared, not introduced by compilation."}
    held_outputs = []
    for label, frame in queries.items():
        cold, cold_wall, cold_cuda = run_timed(compiled, frame)
        reference = eager.bodies(frame)
        errors = {}
        for name, value in tensors(cold).items():
            expected = tensors(reference)[name]
            delta = (value - expected).abs()
            errors[name] = {"max_abs": float(delta.max()),
                                "rms": float(delta.square().mean().sqrt()),
                                "dtype": str(value.dtype)}
            torch.testing.assert_close(value, expected, atol=2e-5, rtol=2e-5)
        held_outputs.append((cold, {k: v.clone() for k, v in tensors(cold).items()}))
        for _ in range(args.warmup):
            eager.bodies(frame)
            compiled.bodies(frame)
        timings = {kind: {"wall": [], "cuda": []} for kind in ("eager", "compiled")}
        # Alternate order to reduce bias from clock/temperature drift.
        for sample in range(args.samples):
            order = [("eager", eager), ("compiled", compiled)]
            if sample % 2:
                order.reverse()
            for name, library in order:
                _, wall_ms, cuda_ms = run_timed(library, frame)
                timings[name]["wall"].append(wall_ms)
                timings[name]["cuda"].append(cuda_ms)
        result = {"query_shape": list(frame.shape), "cold_wall_ms": cold_wall,
                      "cold_cuda_ms": cold_cuda, "errors": errors,
                      "timing": {name: {clock: summarize(samples) for clock, samples in values.items()}
                              for name, values in timings.items()}}
        result["wall_speedup"] = result["timing"]["eager"]["wall"]["median_ms"] / result["timing"]["compiled"]["wall"]["median_ms"]
        report["measurements"][label] = result
        # Query different frames at the same shape: CUDA-graph buffer reuse must
        # never overwrite a previously retained H+1 reference window.
        compiled.bodies((frame + 1) % eager.num_frames)
        print(json.dumps({"query": label, "batch": frame.numel(), "cold_wall_ms": cold_wall,
                          "wall_speedup": result["wall_speedup"], "errors": errors}), flush=True)
    graph_count_warm = graph_counts().get("stats", {}).get("unique_graphs", 0)
    for _ in range(3):
        for frame in queries.values():
            compiled.bodies(frame)
    report["shape_switching_did_not_recompile"] = (
        graph_counts().get("stats", {}).get("unique_graphs", 0) == graph_count_warm)
    assert report["shape_switching_did_not_recompile"]
    for body, clone in held_outputs:
        for name, value in tensors(body).items():
            torch.testing.assert_close(value, clone[name], atol=0, rtol=0)
    report["retained_outputs_unchanged"] = True
    report["dynamo_counters_before"] = before
    report["dynamo_counters_after"] = graph_counts()
    report["peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()

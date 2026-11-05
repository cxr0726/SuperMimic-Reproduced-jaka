"""Compare stock/selected-world Khan reset events, without training a policy.

Run from the repository root with its .venv Python. Timings include CPU dispatch
and GPU synchronization; first graph construction is excluded from warm results.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import RecomputeLevel

from supermimic.envs import SuperMimicEnvCfgArgs, make_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--reset-counts", nargs="+", type=int, default=[24, 96])
    parser.add_argument("--reps", type=int, default=25)
    parser.add_argument("--npz-dir", type=Path, default=Path("data/npz/khan_mini/mixed"))
    parser.add_argument("--out", type=Path, default=Path("logs/khan_partial_constants/reset_benchmark.json"))
    args = parser.parse_args()
    files = sorted(args.npz_dir.glob("*.npz"))
    if (args.num_envs < 2 or args.reps < 1 or not files
            or any(n < 1 or n >= args.num_envs for n in args.reset_counts)):
        parser.error("need NPZ data, num-envs>=2, reps>=1, and 0<reset-count<num-envs")
    env = make_env(SuperMimicEnvCfgArgs(
        npz_files=[str(files[0].resolve())], robot="khan_mini", num_envs=args.num_envs,
        record_transitions=False, seed=73,
    ), device="cuda:0")
    try:
        env.reset()
        helper = env._khan_armature_constants
        manager = env.event_manager
        gains = manager.get_term_cfg("actuator_gains")
        armature = manager.get_term_cfg("ankle_armature")

        def bench(function):
            for _ in range(3):
                function()
            torch.cuda.synchronize()
            samples = []
            for _ in range(args.reps):
                torch.cuda.synchronize()
                started = time.perf_counter()
                function()
                torch.cuda.synchronize()
                samples.append(1000 * (time.perf_counter() - started))
            return {"mean_ms": statistics.mean(samples), "median_ms": statistics.median(samples),
                    "min_ms": min(samples), "max_ms": max(samples)}

        report = {
            "gpu": torch.cuda.get_device_name(), "num_envs": env.num_envs,
            "reps": args.reps, "warmup": 3,
            "boundary": "Event/constant refresh only; same live v6 environment. No policy, physical stepping, replay or full env reset. Warm timings exclude scratch/graph construction. Kp/Kd sampler is unchanged.",
            "results": {},
        }
        for count in args.reset_counts:
            # Exclude original world0 and use non-contiguous, reversed indices.
            ids = torch.linspace(1, env.num_envs - 1, count, device=env.device).long().flip(0)
            torch.cuda.synchronize()
            start = time.perf_counter()
            helper.refresh(ids)
            torch.cuda.synchronize()
            first_call = 1000 * (time.perf_counter() - start)
            result = {"scratch_capacity": helper.last_capacity,
                      "first_call_ms": first_call,
                      "partial_constants": bench(lambda reset_ids=ids: helper.refresh(reset_ids)),
                      "stock_full_constants": bench(lambda: env.sim.recompute_constants(RecomputeLevel.set_const_0)),
                      "gain_event_only": bench(lambda reset_ids=ids: gains.func(env, reset_ids, **gains.params)),
                      "armature_draw_only": bench(lambda reset_ids=ids: dr.joint_armature(env, reset_ids, **armature.params))}
            def events(reset_ids=ids):
                manager.apply("reset", env_ids=reset_ids, global_env_step_count=0)
            result["optimized_reset_events"] = bench(events)
            original = armature.func
            try:
                armature.func = dr.joint_armature
                result["stock_reset_events"] = bench(events)
            finally:
                armature.func = original
            report["results"][str(count)] = result
            print(json.dumps({"reset_count": count, **result}, indent=2), flush=True)
        report["cached_capacities"] = sorted(helper.buckets)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved {args.out}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()

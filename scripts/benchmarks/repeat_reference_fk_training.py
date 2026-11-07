"""Repeat full Khan training with phase, GC and Dynamo diagnostics.

One process runs one mode. Optimizers, network widths, batch sizes, H32, UTD and
the 320k learning-rate schedule remain the production defaults. Only execution
length, environment count, reference-FK execution and optional activation
recomputation are selected here. No GC collection or disabling is performed.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch._dynamo.utils import counters

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from smoke_train import package_versions, verify_optimizer, verify_parameters
from smoke_train_khan import dataset_summary, verify_production_config

from supermimic.config import TrainConfig
from supermimic.training.loop import Trainer


def graph_counts():
    return {
        "unique_graphs": counters["stats"]["unique_graphs"],
        "calls_captured": counters["stats"]["calls_captured"],
        "graph_breaks": sum(counters["graph_break"].values()),
    }


def sample_stats(values):
    if not values:
        return None
    return {
        "count": len(values), "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values), "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)), "max_ms": max(values),
        "sum_ms": sum(values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--log-root", type=Path, default=Path("logs/reference_fk_repeat/training"))
    parser.add_argument("--compiled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--activation-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--steady-from", type=int, default=100,
                        help="Discard iterations <= this boundary from steady summaries")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if (args.num_envs < 1 or args.steps <= args.steady_from or args.steady_from < 50
            or args.steps % 50 or args.steady_from % 50):
        parser.error("positive environments, steps > steady-from >=50, both multiples of50")
    if not args.library.is_file() or args.output.exists():
        parser.error("library must exist and output must be new")
    if Path(args.run_name).name != args.run_name:
        parser.error("run-name must be one directory name")
    run_dir = args.log_root.resolve() / args.run_name
    if (run_dir / "config.json").exists():
        parser.error("run directory already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(robot="khan_mini", seed=args.seed, run_name=args.run_name,
                      log_root=args.log_root.resolve())
    cfg.data.library = args.library.resolve()
    cfg.env.compact_library = True
    cfg.env.reference_fk_compile = args.compiled
    cfg.env.num_envs = args.num_envs
    cfg.schedule.stop_after_iterations = args.steps
    cfg.policy_trainer.activation_checkpoint = args.activation_checkpoint
    production = verify_production_config(cfg)
    phase = {"name": "initialization"}
    gc_open, gc_records, records, iterations, nested_events = {}, [], [], [], []
    origin = time.perf_counter()
    trainer = None
    previous_iteration = None
    current_iteration = 0
    jsonl = args.output.with_suffix(".jsonl").open("x", buffering=1)

    def on_gc(action, info):
        generation = info["generation"]
        now = time.perf_counter()
        if action == "start":
            gc_open[generation] = (now, phase["name"], current_iteration)
        elif generation in gc_open:
            begin, began_phase, iteration = gc_open.pop(generation)
            gc_records.append({
                "generation": generation, "start_s": begin-origin, "end_s": now-origin,
                "duration_ms": (now-begin)*1000, "phase": began_phase,
                "end_phase": phase["name"], "iteration": iteration,
                "collected": info.get("collected"), "uncollectable": info.get("uncollectable"),
            })

    gc.callbacks.append(on_gc)
    report = {
        "status": "running", "started_at_utc": datetime.now(UTC).isoformat(),
        "arguments": vars(args), "production_config": production,
        "gpu": torch.cuda.get_device_name(), "versions": package_versions(),
        "gc_enabled": gc.isenabled(), "gc_thresholds": gc.get_threshold(),
        "instrumentation": "Outer env/WM/policy phase wrappers synchronize after, matching the existing loop boundaries, identically for both modes; no presynchronization is added. Nested FK/critic-feature wrappers record CPU enqueue time and CUDA events without additional fences. GC is observed only, never collected/disabled. Primary throughput uses unchanged Trainer TB time/hours boundaries; all slow samples remain included.",
    }
    try:
        torch.cuda.reset_peak_memory_stats()
        init_begin = time.perf_counter()
        trainer = Trainer(cfg)
        report["initialization_wall_s"] = time.perf_counter()-init_begin
        report["dataset"] = dataset_summary(trainer, cfg)
        models = {"wm": trainer.wm, "policy": trainer.policy, "critic": trainer.critic}
        before = {name: {key: value.detach().cpu().clone() for key, value in model.named_parameters()}
                  for name, model in models.items()}

        def finish_iteration(now):
            nonlocal previous_iteration
            if previous_iteration is not None:
                iteration, begin = previous_iteration
                row = {"iteration": iteration, "wall_ms": (now-begin)*1000,
                       "kind": "outer_iteration"}
                iterations.append(row)
                jsonl.write(json.dumps(row)+"\n")
                previous_iteration = None

        original_act = trainer.act

        def act(obs):
            nonlocal previous_iteration, current_iteration
            now = time.perf_counter()
            finish_iteration(now)
            previous_iteration = (trainer.iteration+1, now)
            current_iteration = trainer.iteration+1
            phase["name"] = "collection"
            return original_act(obs)

        trainer.act = act
        original_save = trainer.save

        def save(name="ckpt_latest.pt"):
            finish_iteration(time.perf_counter())
            phase["name"] = "checkpoint"
            return original_save(name)

        trainer.save = save

        def instrument(obj, method, label, parent=None):
            original = getattr(obj, method)

            def wrapped(*call_args, **call_kwargs):
                outer = phase["name"]
                phase["name"] = label
                events = None
                if parent is not None:
                    events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                    events[0].record()
                begin = time.perf_counter()
                gc_index = len(gc_records)
                graphs = graph_counts() if parent is None else None
                try:
                    result = original(*call_args, **call_kwargs)
                    if events is not None:
                        events[1].record()
                    if parent is None:
                        torch.cuda.synchronize()
                finally:
                    end = time.perf_counter()
                    elapsed = (end-begin)*1000
                    phase["name"] = f"diagnostics/{label}"
                    relevant_gc = [r for r in gc_records[gc_index:] if r["end_s"] <= end-origin]
                    record = {
                        "kind": "phase", "phase": label, "iteration": current_iteration,
                        "timing_kind": "synchronized_wall" if parent is None else "host_enqueue",
                        "wall_ms": elapsed, "start_s": begin-origin,
                        "gc_ms": sum(r["duration_ms"] for r in relevant_gc),
                        "gc_count": len(relevant_gc),
                    }
                    if graphs is not None:
                        after = graph_counts()
                        record["dynamo_delta"] = {key: after[key]-graphs[key] for key in graphs}
                    records.append(record)
                    if events is not None:
                        nested_events.append((record, events))
                    jsonl.write(json.dumps(record)+"\n")
                    phase["name"] = outer
                return result

            setattr(obj, method, wrapped)

        instrument(trainer.env, "step", "env_step")
        instrument(trainer.wm_trainer, "update", "wm_update")
        instrument(trainer.policy_trainer, "update", "policy_update")
        # Nested labels localize GC without changing the learning code.
        instrument(trainer.policy_trainer, "_critic_inputs", "critic_features", parent="policy_update")
        instrument(trainer.lib, "bodies", "reference_fk", parent="caller")
        phase["name"] = "run"
        run_begin = time.perf_counter()
        trainer.run()
        report["run_including_reset_and_final_checkpoint_s"] = time.perf_counter()-run_begin
        phase["name"] = "validation"
        torch.cuda.synchronize()
        for record, events in nested_events:
            record["cuda_ms"] = events[0].elapsed_time(events[1])
        report["model_checks"] = {name: verify_parameters(model, before[name])
                                  for name, model in models.items()}
        optimizers = {"wm": trainer.wm_trainer.opt, "policy": trainer.policy_trainer.opt,
                      "critic": trainer.policy_trainer.critic_opt}
        report["optimizer_checks"] = {name: verify_optimizer(opt) for name, opt in optimizers.items()}
        expected_updates = math.floor((args.steps-cfg.schedule.warmup_iterations+1)
                                      * cfg.schedule.updates_per_env_step+1e-9)
        assert trainer.iteration == args.steps and trainer.num_updates == expected_updates
        expected = {"wm": expected_updates*cfg.wm_trainer.batches_per_update,
                    "policy": expected_updates,
                    "critic": expected_updates*cfg.policy_trainer.critic_epochs*cfg.policy_trainer.critic_minibatches}
        for name, steps in expected.items():
            assert report["optimizer_checks"][name]["min_step"] == steps
            assert report["optimizer_checks"][name]["max_step"] == steps
        trainer.writer.flush()
        events = EventAccumulator(str(run_dir/"tb"), size_guidance={"scalars": 0}).Reload()
        hours = {event.step: event.value for event in events.Scalars("time/hours")}
        elapsed = (hours[args.steps]-hours[args.steady_from])*3600
        report["steady"] = {
            "range": [args.steady_from, args.steps], "elapsed_s": elapsed,
            "iterations": args.steps-args.steady_from,
            "iterations_per_s": (args.steps-args.steady_from)/elapsed,
            "phase_stats": {label: sample_stats([r["wall_ms"] for r in records
                                                   if r["phase"] == label and r["iteration"] > args.steady_from])
                            for label in {r["phase"] for r in records}},
            "outer_iteration_stats": sample_stats([r["wall_ms"] for r in iterations
                                                       if r["iteration"] > args.steady_from]),
            "nested_cuda_stats": {label: sample_stats([r["cuda_ms"] for r in records
                                                       if r["phase"] == label and "cuda_ms" in r
                                                       and r["iteration"] > args.steady_from])
                                  for label in ("reference_fk", "critic_features")},
        }
        report["tb_scalars_finite"] = all(math.isfinite(event.value)
            for tag in events.Tags()["scalars"] for event in events.Scalars(tag))
        assert report["tb_scalars_finite"]
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        phase["name"] = "cleanup"
        report["config"] = asdict(cfg)
        report["gc_events"] = gc_records
        report["phase_records"] = records
        report["outer_iterations"] = iterations
        report["dynamo_final"] = graph_counts()
        report["peak_torch_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["total_wall_s"] = time.perf_counter()-origin
        if trainer is not None:
            trainer.env.close()
            trainer.writer.close()
        gc.callbacks.remove(on_gc)
        jsonl.close()
        args.output.write_text(json.dumps(report, indent=2, default=str, allow_nan=False)+"\n")
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "steady": report["steady"]}, default=str), flush=True)


if __name__ == "__main__":
    main()

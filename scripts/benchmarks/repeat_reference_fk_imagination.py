"""Repeated, instrumented same-process eager/compiled reference-FK A/B.

Full actor BPTT and critic bootstrap use fixed starts, weights and noise RNG.
GC remains enabled and is never explicitly collected. Lightweight callbacks and
Dynamo counters identify whether measured pauses coincide with GC or capture.
Warmup, numerical validation, JSON output and cleanup are outside trial timers.
This is not full training: critic fitting and optimizer steps are excluded.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from benchmark_reference_fk_imagination import (
    difference,
    load_benchmark_checkpoint,
    snapshot,
    weights_hash,
)

from supermimic.data.replay_buffer import StartBatch
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.imagination.objective import policy_objective
from supermimic.imagination.policy_trainer import PolicyTrainer
from supermimic.imagination.rollout import imagine
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Critic
from supermimic.robots.registry import get_robot_spec
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.world_model import WorldModel


def dynamo_counters():
    from torch._dynamo.utils import counters
    return {group: {str(k): int(v) for k, v in counters[group].items()}
            for group in ("stats", "graph_break", "frames")}


def counter_delta(after, before):
    return {group: {key: after.get(group, {}).get(key, 0) - before.get(group, {}).get(key, 0)
                    for key in after.get(group, {}).keys() | before.get(group, {}).keys()
                    if after.get(group, {}).get(key, 0) != before.get(group, {}).get(key, 0)}
            for group in after.keys() | before.keys()}


class GCMonitor:
    """A callback records events without changing collection behavior."""

    def __init__(self):
        self.context = {"stage": "setup", "mode": None, "trial": None, "phase": "setup"}
        self.pending = {}
        self.events = []

    def callback(self, phase, info):
        now = time.perf_counter()
        thread = threading.get_ident()
        generation = info["generation"]
        key = (thread, generation)
        if phase == "start":
            self.pending[key] = (now, self.context.copy())
        else:
            started, context = self.pending.pop(key, (now, self.context.copy()))
            self.events.append({**context, "generation": generation, "thread": thread,
                                "start_perf_s": started, "end_perf_s": now,
                                "wall_ms": (now - started) * 1000,
                                "collected": info.get("collected", 0),
                                "uncollectable": info.get("uncollectable", 0)})


def describe(values):
    return {"count": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "std": statistics.pstdev(values),
            "p90": float(np.percentile(values, 90)), "p95": float(np.percentile(values, 95)),
            "min": min(values), "max": max(values)}


def summarize(rows, n, horizon):
    result = {}
    for mode in ("eager", "compiled"):
        selected = [r for r in rows if r["mode"] == mode]
        keys = ("wall_ms", "wall_imagine_enqueue_ms", "wall_critic_inputs_enqueue_ms",
                "wall_bootstrap_backward_sync_ms", "wall_reference_fk_enqueue_ms",
                "cuda_imagine_ms", "cuda_critic_inputs_ms", "cuda_bootstrap_backward_ms",
                "cuda_reference_fk_ms", "gc_wall_ms")
        stats = {key: describe([r[key] for r in selected]) for key in keys}
        total_ms = sum(r["wall_ms"] for r in selected)
        median_forward = stats["cuda_imagine_ms"]["median"]
        pauses = [{"trial": r["trial"], "wall_ms": r["wall_ms"],
                   "extra_forward_ms": r["cuda_imagine_ms"] - median_forward,
                   "gc_wall_ms": r["gc_wall_ms"], "gc_events": r["gc_events"],
                   "dynamo_delta": r["dynamo_delta"]}
                  for r in selected if r["cuda_imagine_ms"] - median_forward >= 1500]
        result[mode] = {"statistics": stats, "sum_timed_wall_ms": total_ms,
                        "actor_bptt_runs_per_second_including_all_pauses": len(selected)*1000/total_ms,
                        "imagined_transitions_per_second_including_all_pauses": len(selected)*n*horizon*1000/total_ms,
                        "forward_pauses_extra_ge_1500ms": pauses,
                        "forward_pause_count_extra_ge_2000ms": sum(
                            r["cuda_imagine_ms"] - median_forward >= 2000 for r in selected),
                        "trials_with_new_graphs": [r["trial"] for r in selected
                            if r["dynamo_delta"].get("stats", {}).get("unique_graphs", 0)]}
    result["compiled_over_eager_throughput_ratio"] = (
        result["compiled"]["actor_bptt_runs_per_second_including_all_pauses"] /
        result["eager"]["actor_bptt_runs_per_second_including_all_pauses"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=30, help="Trials per mode; order alternates AB / BA")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup trials per mode")
    parser.add_argument("--activation-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if min(args.samples, args.warmup) < 1 or not torch.cuda.is_available():
        parser.error("Positive counts and CUDA are required")
    jsonl = args.output.with_suffix(".jsonl")
    if args.output.exists() or jsonl.exists():
        parser.error("Use a new output path; existing results are preserved")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(730)
    cfg, policy, checkpoint = load_benchmark_checkpoint(args.checkpoint, device)
    cfg.policy_trainer.activation_checkpoint = args.activation_checkpoint
    spec = get_robot_spec(cfg.robot)
    model = spec.load_model()
    if cfg.robot == "khan_mini":
        from supermimic.robots.khan_mini import load_profile
        profile = load_profile()
        q0 = torch.tensor([profile["default_joint_pos"][name] for name in spec.joint_names], device=device)
        factor = profile["soft_joint_pos_limit_factor"]
    else:
        q0, factor = torch.zeros(spec.num_joints, device=device), .9
    limits = torch.tensor(np.asarray(model.jnt_range[1:]), device=device, dtype=torch.float32)
    middle, half_range = limits.mean(-1), .5 * (limits[:, 1] - limits[:, 0])
    limits = torch.stack((middle - factor * half_range, middle + factor * half_range), -1)
    heads = AnalyticHeads(G1Kinematics(model, device), spec.tracked_bodies,
                          spec.anchor_body, spec.root_body, limits, q0)
    mapping = ({name: checkpoint["wm"][name] for name in ("action_scale", "action_offset")}
               if cfg.world_model.input_normalization else {})
    wm = WorldModel(cfg.world_model, heads, device, **mapping)
    critic = Critic(cfg.critic, pcfg=cfg.policy, xi_dim=wm.spec.xi_dim).to(device)
    for name, network in (("wm", wm), ("critic", critic)):
        network.load_state_dict(checkpoint[name])
    del checkpoint
    wm.requires_grad_(False)
    critic.requires_grad_(False)
    lib = MotionLibrary([args.library], spec.tracked_bodies, q0, device=device,
                        compact=True, robot_name=cfg.robot, root_body_name=spec.root_body,
                        reference_fk_compile=True)
    compiled_fn, eager_fn = lib._reference_bodies_fn, lib._compact_bodies
    trainer = PolicyTrainer(wm, policy, critic, lib, cfg.policy_trainer)
    n, horizon, history = cfg.policy_trainer.batch_size, cfg.policy_trainer.horizon, cfg.policy.history
    valid = (lib.clip_len - horizon - history + 1).clamp_min(0)
    if not bool(valid.sum()):
        raise ValueError("No clip provides the requested history and horizon")
    clips = torch.multinomial(valid.float(), n, replacement=True)
    frame = lib.clip_start[clips] + history - 1 + (torch.rand(n, device=device) * valid[clips]).long()
    history_frames = frame[None] + torch.arange(1 - history, 1, device=device)[:, None]
    qpos, qvel = lib.ref_qpos_qvel(history_frames)
    hist_states = torch.cat((qpos, qvel), -1)
    hist_states[..., 2] += .05
    hist_actions = (lib.ref(history_frames).joint_pos - q0) / .5
    start = StartBatch(hist_states[-1], wm.spec.xi_offset.repeat(n, 1), frame,
                       hist_actions[-1], hist_states, hist_actions)
    original_bodies = lib.bodies
    monitor = GCMonitor()
    reference_events, reference_wall = [], []
    reference_strides = {}

    def timed_bodies(query):
        phase = monitor.context["phase"]
        monitor.context["phase"] = "reference_fk"
        before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        before.record()
        begin = time.perf_counter()
        result = original_bodies(query)
        reference_wall.append((time.perf_counter() - begin) * 1000)
        after.record()
        reference_events.append((before, after))
        mode = monitor.context["mode"]
        if mode not in reference_strides:
            reference_strides[mode] = {
                key: {"shape": list(value.shape), "stride": list(value.stride()),
                      "contiguous": value.is_contiguous()}
                for key, value in vars(result).items()}
        monitor.context["phase"] = phase
        return result

    lib.bodies = timed_bodies
    rng_cpu, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
    initial_weights = weights_hash((wm, policy, critic))
    original_gc_enabled, original_gc_threshold = gc.isenabled(), gc.get_threshold()

    def run(mode, stage, trial, capture=False):
        monitor.context = {"stage": stage, "mode": mode, "trial": trial, "phase": "setup"}
        lib._reference_bodies_fn = eager_fn if mode == "eager" else compiled_fn
        policy.zero_grad(set_to_none=True)
        torch.set_rng_state(rng_cpu)
        torch.cuda.set_rng_state(rng_cuda)
        reference_events.clear()
        reference_wall.clear()
        before_counters = dynamo_counters()
        before_gc = len(monitor.events)
        torch.cuda.synchronize()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        monitor.context["phase"] = "imagine"
        begin = time.perf_counter()
        events[0].record()
        ro = imagine(wm, policy, lib, start, horizon, cfg.policy_trainer.future_frames,
                     obs_noise=cfg.policy_trainer.obs_noise, step_fn=trainer.step_fn,
                     anchor_obs=cfg.policy_trainer.anchor_obs,
                     activation_checkpoint=cfg.policy_trainer.activation_checkpoint)
        after_imagine = time.perf_counter()
        events[1].record()
        monitor.context["phase"] = "critic_inputs"
        feature, goals, root_error, errors = trainer._critic_inputs(ro)
        after_inputs = time.perf_counter()
        events[2].record()
        monitor.context["phase"] = "bootstrap_backward"
        value = critic(feature[-1], goals[-1], wm.spec.normalize_xi(start.xi),
                       root_error[-1], errors[-1])
        objective = policy_objective(ro, value, cfg.policy_trainer.gamma)
        (-objective / horizon).backward()
        events[3].record()
        torch.cuda.synchronize()
        finished = time.perf_counter()
        monitor.context["phase"] = "postprocess"
        gc_events = [event for event in monitor.events[before_gc:]
                     if event["end_perf_s"] >= begin and event["start_perf_s"] <= finished]
        result = {"stage": stage, "trial": trial, "mode": mode,
                  "start_perf_s": begin, "end_perf_s": finished,
                  "wall_ms": (finished - begin) * 1000,
                  "wall_imagine_enqueue_ms": (after_imagine - begin) * 1000,
                  "wall_critic_inputs_enqueue_ms": (after_inputs - after_imagine) * 1000,
                  "wall_bootstrap_backward_sync_ms": (finished - after_inputs) * 1000,
                  "wall_reference_fk_enqueue_ms": sum(reference_wall),
                  "cuda_imagine_ms": events[0].elapsed_time(events[1]),
                  "cuda_critic_inputs_ms": events[1].elapsed_time(events[2]),
                  "cuda_bootstrap_backward_ms": events[2].elapsed_time(events[3]),
                  "cuda_total_ms": events[0].elapsed_time(events[3]),
                  "cuda_reference_fk_ms": sum(a.elapsed_time(b) for a, b in reference_events),
                  "gc_wall_ms": sum(max(0., min(e["end_perf_s"], finished) -
                                                   max(e["start_perf_s"], begin))*1000
                                    for e in gc_events),
                  "gc_events": gc_events,
                  "dynamo_delta": counter_delta(dynamo_counters(), before_counters)}
        saved = snapshot(ro, errors, policy, objective) if capture else None
        del ro, feature, goals, root_error, errors, value, objective
        monitor.context["phase"] = "between_trials"
        return result, saved

    report = {"status": "running", "timestamp": datetime.now(UTC).isoformat(),
              "checkpoint": str(args.checkpoint.resolve()), "library": str(args.library.resolve()),
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "batch": n, "horizon": horizon, "history": history,
              "samples_per_mode": args.samples, "warmup_per_mode": args.warmup,
              "activation_checkpoint": cfg.policy_trainer.activation_checkpoint,
              "models_unchanged_sha256": initial_weights, "rng_restored_each_trial": True,
              "gc_enabled": original_gc_enabled, "gc_threshold": original_gc_threshold,
              "gc_policy_changed": False,
              "timing_scope": "Actor BPTT and critic bootstrap, excludes critic fit and optimizer step. CPU phase times measure enqueue/synchronization; CUDA event spans include stream idle gaps.",
              "limits": "One process/GPU, fixed reference-derived starts and neutral xi; not replay visitation or complete training. Slow samples are retained. A pause not seen in a finite run is not proof it can never recur."}
    warm, measurements, validations = [], [], []
    gc.callbacks.append(monitor.callback)
    try:
        with jsonl.open("w") as stream:
            stream.write(json.dumps({"type": "header", **report}) + "\n")
            stream.flush()

            def emit(result):
                stream.write(json.dumps({"type": "trial", **result}) + "\n")
                stream.flush()
                print(json.dumps({k: v for k, v in result.items() if k != "gc_events"}), flush=True)

            for repeat in range(args.warmup):
                for mode in (("eager", "compiled") if repeat % 2 == 0 else ("compiled", "eager")):
                    result, _ = run(mode, "warmup", repeat)
                    warm.append(result)
                    emit(result)
            report["counters_after_warmup"] = dynamo_counters()
            measured_loop_start = time.perf_counter()
            for repeat in range(args.samples):
                for order, mode in enumerate(("eager", "compiled") if repeat % 2 == 0 else ("compiled", "eager")):
                    result, _ = run(mode, "measurement", repeat)
                    result["pair"] = repeat
                    result["order_in_pair"] = order
                    measurements.append(result)
                    emit(result)
            report["measured_loop_elapsed_s_including_setup_logging_cleanup"] = time.perf_counter() - measured_loop_start
            # Snapshot copies and comparisons are deliberately outside timed samples.
            snapshots = {}
            for mode in ("eager", "compiled"):
                result, saved = run(mode, "validation", 0, capture=True)
                snapshots[mode] = saved
                validations.append(result)
                emit(result)
            report["differences"] = difference(snapshots["compiled"], snapshots["eager"])
        assert weights_hash((wm, policy, critic)) == initial_weights
        assert gc.isenabled() == original_gc_enabled and gc.get_threshold() == original_gc_threshold
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        gc.callbacks.remove(monitor.callback)
        report.update(warmup=warm, measurements=measurements, validation_trials=validations,
                      gc_events=monitor.events, counters_final=dynamo_counters(),
                      reference_strides=reference_strides)
        if measurements and all(any(r["mode"] == mode for r in measurements) for mode in ("eager", "compiled")):
            report["summary"] = summarize(measurements, n, horizon)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("saved", args.output, flush=True)


if __name__ == "__main__":
    main()

"""Matched learner timing with one real simulation/replay prefill.

Example (bounded local shapes, actual full PT library)::

  .venv/bin/python scripts/benchmarks/benchmark_learner_updates.py \
    --config logs/smoke_learning_capture_20261113/config.json \
    --output logs/learner_update_benchmark.json

All modes use the current prepared GPU sampler. This is an execution-path
comparison, not a reconstruction of the entire historical b024710 baseline.
``eager`` means no whole-update CUDA Graph; the config's torch.compile settings
are retained. Collection/reset, networks and learning budgets do not vary.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch
from benchmark_reference_fk_imagination import restore_config

from supermimic.config import TrainConfig, resolve_training_observations
from supermimic.imagination.policy_trainer import PolicyTrainer
from supermimic.robots.contracts import validate_robot_contract
from supermimic.training.loop import Trainer
from supermimic.world_model.wm_trainer import WorldModelTrainer

MODES = {
    "stepwise_eager": (False, False),
    "batched_eager": (True, False),
    "batched_captured": (True, True),
}


def _cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _state_hash(states):
    digest = hashlib.sha256()
    for name, state in states.items():
        for key, value in state.items():
            digest.update((name + "/" + key).encode())
            digest.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _source_hash():
    """Identify the measured source even when the implementation is uncommitted."""
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    paths = sorted((root / "src/supermimic").rglob("*.py"))
    paths.append(Path(__file__).resolve())
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _summary(values):
    return {"mean": statistics.mean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values), "samples": values}


def _steps(optimizer):
    counts = [int(value["step"]) for value in optimizer.state.values() if "step" in value]
    return {"min": min(counts, default=0), "max": max(counts, default=0), "parameters": len(counts)}


def _dimensions(trainer):
    wm_layers = [layer for layer in trainer.wm.dynamics.modules() if isinstance(layer, torch.nn.Linear)]
    critic_layers = [layer for layer in trainer.critic.modules() if isinstance(layer, torch.nn.Linear)]
    obs = trainer.policy.decoder.scale.numel()
    return {"state": trainer.wm.spec.state_dim, "context": trainer.wm.spec.xi_dim,
            "wm_input": wm_layers[0].in_features, "wm_output": wm_layers[-1].out_features,
            "actor_proprioception": obs, "actor_goal": trainer.cfg.policy.goal_dim,
            "actor_external_input": obs + trainer.cfg.policy.goal_dim,
            "actor_action": trainer.cfg.policy.num_joints, "critic_input": critic_layers[0].in_features}


def _release_learners(trainer):
    """Release graph pools/optimizer/gradient references before the next mode."""
    wm_trainer, policy_trainer = trainer.wm_trainer, trainer.policy_trainer
    if wm_trainer is not None:
        wm_trainer.reset_capture()
        wm_trainer.opt.zero_grad(set_to_none=True)
        wm_trainer.opt.state.clear()
    if policy_trainer is not None:
        policy_trainer.reset_capture()
        for optimizer in (policy_trainer.opt, policy_trainer.critic_opt):
            optimizer.zero_grad(set_to_none=True)
            optimizer.state.clear()
    trainer.wm_trainer, trainer.policy_trainer = None, None
    trainer.wm._loss_step_fn = None
    for module in (trainer.wm, trainer.policy, trainer.critic):
        module.zero_grad(set_to_none=True)
        module.requires_grad_(True)
    del wm_trainer, policy_trainer
    # Compiled callables may retain graph-pool activations or old step objects.
    # The on-disk Inductor kernel cache is intentionally retained and disclosed.
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


@torch.no_grad()
def _prefill(trainer, steps):
    obs, _ = trainer.env.reset()
    begin = time.perf_counter()
    for _ in range(steps):
        frame = trainer.command.frame.clone()
        action = trainer.act(obs)
        obs, reward, terminated, timeout, _ = trainer.env.step(action)
        trainer.command.record_window_transition(
            frame, reward, terminated, timeout, trainer.env.termination_manager.get_term("clip_end"),
        )
        trainer.update_collection_normalization()
    torch.cuda.synchronize()
    # Fail before any timed learning if the real prefill has no legal window.
    windows = trainer.buffer.prepare_windows(trainer.cfg.wm_trainer.window)
    starts = trainer.buffer.prepare_starts(
        trainer.cfg.policy_trainer.history, horizon=trainer.cfg.policy_trainer.horizon,
        clip_last=trainer.lib.clip_last,
    )
    return {"steps": steps, "wall_ms": (time.perf_counter() - begin) * 1000,
            "replay_rows": trainer.buffer.size, "valid_windows": int(windows.count),
            "valid_starts": int(starts.count)}


def _timed_block(trainer):
    torch.cuda.synchronize()
    begin = time.perf_counter()
    wm_metrics = trainer.wm_trainer.update(trainer.buffer)
    torch.cuda.synchronize()
    after_wm = time.perf_counter()
    pi_metrics = trainer.policy_trainer.update(trainer.buffer)
    torch.cuda.synchronize()
    after_pi = time.perf_counter()
    for prefix, metrics in (("wm", wm_metrics), ("policy", pi_metrics)):
        if not all(math.isfinite(value) for value in metrics.values()):
            raise RuntimeError(f"Nonfinite {prefix} metric during benchmark: {metrics}")
    return {"wm_ms": (after_wm - begin) * 1000,
            "policy_ms": (after_pi - after_wm) * 1000,
            "total_ms": (after_pi - begin) * 1000}, {"wm": wm_metrics, "policy": pi_metrics}


def _run_mode(trainer, mode, snapshot, seed, warmup, samples):
    _release_learners(trainer)
    for name, module in (("wm", trainer.wm), ("policy", trainer.policy), ("critic", trainer.critic)):
        module.load_state_dict(snapshot[name])
    if _state_hash({name: _cpu_state(module) for name, module in
                    (("wm", trainer.wm), ("policy", trainer.policy), ("critic", trainer.critic))}) != _state_hash(snapshot):
        raise AssertionError("Mode did not restore identical initial parameters/statistics")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    batch_encoder, capture = MODES[mode]
    wm_cfg, pi_cfg = copy.deepcopy(trainer.cfg.wm_trainer), copy.deepcopy(trainer.cfg.policy_trainer)
    wm_cfg.cuda_graph = pi_cfg.cuda_graph = capture
    pi_cfg.batch_reference_encoder = batch_encoder
    init_begin = time.perf_counter()
    trainer.wm_trainer = WorldModelTrainer(trainer.wm, trainer.lib, wm_cfg)
    trainer.policy_trainer = PolicyTrainer(trainer.wm, trainer.policy, trainer.critic, trainer.lib, pi_cfg)
    torch.cuda.synchronize()
    init_ms = (time.perf_counter() - init_begin) * 1000
    # The same counted learning warm-up budget is used in all three modes.
    warm_rows = []
    for _ in range(warmup):
        row, _ = _timed_block(trainer)
        warm_rows.append(row)
    if capture and not (trainer.wm_trainer.update_graph.captured
                        and trainer.policy_trainer._captured_update.captured):
        raise RuntimeError("Warm-up ended before both complete learner graphs were recorded")
    torch.cuda.reset_peak_memory_stats()
    rows, last = [], None
    for _ in range(samples):
        row, last = _timed_block(trainer)
        rows.append(row)
    optimizer_steps = {"wm": _steps(trainer.wm_trainer.opt), "policy": _steps(trainer.policy_trainer.opt),
                       "critic": _steps(trainer.policy_trainer.critic_opt)}
    blocks = warmup + samples
    expected = {"wm": blocks * wm_cfg.batches_per_update, "policy": blocks,
                "critic": blocks * pi_cfg.critic_epochs * pi_cfg.critic_minibatches if pi_cfg.use_critic else 0}
    for name, count in expected.items():
        actual = optimizer_steps[name]
        if actual["min"] != count or actual["max"] != count:
            raise AssertionError(f"{mode}: unexpected {name} optimizer count {actual}, expected {count}")
    result = {
        "mode": mode, "batch_reference_encoder": batch_encoder, "cuda_graph": capture,
        "learner_init_ms": init_ms, "warmup_including_compile_capture_ms": sum(r["total_ms"] for r in warm_rows),
        "warmup_blocks": warm_rows, "measured_blocks": samples,
        "timing": {key: _summary([row[key] for row in rows]) for key in rows[0]},
        "optimizer_steps": optimizer_steps, "expected_optimizer_steps": expected,
        "wm_graph_captured": trainer.wm_trainer.update_graph.captured,
        "policy_graph_captured": trainer.policy_trainer._captured_update.captured,
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(), "last_metrics": last,
    }
    print(f"{mode}: WM {result['timing']['wm_ms']['median']:.2f} ms, "
          f"Actor+Critic {result['timing']['policy_ms']['median']:.2f} ms, "
          f"total {result['timing']['total_ms']['median']:.2f} ms", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path, help="Training config.json; network/batch/H defaults are retained")
    source.add_argument("--checkpoint", type=Path, help="Use its config and live WM/Actor/Critic weights, fresh optimizers")
    parser.add_argument("--library", type=Path, help="Optional compact PT path override")
    parser.add_argument("--num-envs", type=int, default=8, help="Physical prefill worlds; learner batches remain unchanged")
    parser.add_argument("--prefill-steps", type=int, default=96)
    parser.add_argument("--warmup", type=int, default=4, help="Actual learning blocks per mode before timing")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        parser.error("This benchmark requires a CUDA device")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("This matched CUDA Graph benchmark is single-process/single-GPU")
    if min(args.num_envs, args.prefill_steps, args.warmup, args.samples) < 1:
        parser.error("Environment, prefill, warm-up and sample counts must be positive")
    checkpoint = None
    cfg = TrainConfig(robot="khan_mini")
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
        restore_config(cfg, checkpoint["config"])
    elif args.config:
        restore_config(cfg, json.loads(args.config.read_text()))
    if args.library:
        cfg.data.library = args.library
    if cfg.data.library is not None and not Path(cfg.data.library).is_file():
        parser.error(f"Compact library does not exist: {cfg.data.library}")
    cfg.device, cfg.env.num_envs, cfg.resume = args.device, args.num_envs, None
    if args.seed is not None:
        cfg.seed = args.seed
    cfg.log_root, cfg.run_name = args.output.parent, args.output.stem + "_scene"
    resolve_training_observations(cfg)
    # Prefill never needs a captured learner; each mode constructs fresh ones.
    original_wm_cfg, original_pi_cfg = copy.deepcopy(cfg.wm_trainer), copy.deepcopy(cfg.policy_trainer)
    cfg.wm_trainer.cuda_graph = cfg.policy_trainer.cuda_graph = False
    trainer = Trainer(cfg)
    try:
        if checkpoint is not None:
            validate_robot_contract(checkpoint, cfg.robot, cfg.env.observation_layout)
            for name, module in (("wm", trainer.wm), ("policy", trainer.policy), ("critic", trainer.critic)):
                module.load_state_dict(checkpoint[name])
            if trainer.policy_trainer.ema is not None:
                trainer.policy_trainer.ema.module.load_state_dict(trainer.policy.state_dict())
        prefill = _prefill(trainer, args.prefill_steps)
        snapshot = {name: _cpu_state(module) for name, module in
                    (("wm", trainer.wm), ("policy", trainer.policy), ("critic", trainer.critic))}
        cfg.wm_trainer, cfg.policy_trainer = original_wm_cfg, original_pi_cfg
        warmup = max(args.warmup, cfg.policy_trainer.cuda_graph_warmup + 1,
                     cfg.wm_trainer.cuda_graph_warmup + 1)
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        report = {
            "git_commit": revision, "torch_version": torch.__version__,
            "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
            "python_source_sha256": _source_hash(),
            "gpu": torch.cuda.get_device_name(torch.device(args.device)),
            "device": args.device, "source_config": str(args.config) if args.config else None,
            "source_checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "seed": cfg.seed, "config": asdict(cfg), "prefill": prefill,
            "dimensions": _dimensions(trainer),
            "initial_model_state_sha256": _state_hash(snapshot),
            "protocol": {"same_replay_and_initial_parameters": True, "fresh_optimizers_per_mode": True,
                         "same_prepared_gpu_sampler": True, "physical_collection_during_timing": False,
                         "normalizer_updates_during_timing": False,
                         "same_seed_not_identical_rng_draw_order": True,
                         "wm_capture_scope": "one graph for all WM minibatches in a learning block",
                         "timing": "synchronized wall time, includes sampling and metric transfer",
                         "kernel_cache": "same process; Dynamo reset per mode, disk Inductor cache retained",
                         "torch_memory_counters_exclude_warp_and_cuda_context": True,
                         "not_formal_4096_environment_throughput": True},
            "budget_per_learning_block": {"wm_optimizer_steps": cfg.wm_trainer.batches_per_update,
                                          "wm_batch": cfg.wm_trainer.batch_size, "wm_window": cfg.wm_trainer.window,
                                          "policy_optimizer_steps": 1, "policy_batch": cfg.policy_trainer.batch_size,
                                          "horizon": cfg.policy_trainer.horizon,
                                          "critic_optimizer_steps": cfg.policy_trainer.critic_epochs * cfg.policy_trainer.critic_minibatches},
            "warmup_requested": args.warmup, "warmup_actual": warmup, "modes": [],
        }
        for mode in MODES:
            report["modes"].append(_run_mode(trainer, mode, snapshot, cfg.seed, warmup, args.samples))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    finally:
        _release_learners(trainer)
        if trainer.writer is not None:
            trainer.writer.close()
        trainer.env.close()


if __name__ == "__main__":
    main()

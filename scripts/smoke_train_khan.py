"""Verify Khan training with the repository's full production hyperparameters.

Robot/data selection, Khan physics/observations, world count and execution limit
differ from the original G1 defaults. Optimization settings remain unchanged. Optional activation recomputation
is disabled by default, matching the original execution. The 320k LR schedule, network
widths, batch sizes, compilation, warmup, horizon and UTD are preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch
from smoke_train import package_versions, read_metrics, verify_optimizer, verify_parameters

from supermimic.config import TrainConfig, resolve_training_observations
from supermimic.observation_layout import policy_observation_dimension
from supermimic.training.loop import Trainer

ROOT = Path(__file__).resolve().parents[1]


def _copy_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _copy_to_cpu(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_copy_to_cpu(item) for item in value)
    return value


def _assert_tree_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual.detach().cpu(), expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            _assert_tree_equal(actual[key], value)
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_tree_equal(left, right)
    else:
        assert actual == expected


def verify_checkpoint_restore(trainer, models, checkpoint: Path) -> dict:
    """Actually erase optimizer state and perturb models/sampler before resume."""
    probes = {name: next(model.parameters()).detach().clone() for name, model in models.items()}
    optimizers = {"wm": trainer.wm_trainer.opt, "policy": trainer.policy_trainer.opt,
                  "critic": trainer.policy_trainer.critic_opt}
    states = {name: _copy_to_cpu(opt.state_dict()) for name, opt in optimizers.items()}
    sampler = {key: getattr(trainer.command, key).clone() for key in ("bin_return", "bin_memory")}
    iteration, updates = trainer.iteration, trainer.num_updates
    with torch.no_grad():
        for model in models.values():
            next(model.parameters()).add_(1.0)
        for value in sampler:
            getattr(trainer.command, value).zero_()
    for opt in optimizers.values():
        opt.state.clear()
        for group in opt.param_groups:
            group["lr"] = 123.0
    trainer.iteration = trainer.num_updates = -1
    trainer.load(checkpoint)
    assert (trainer.iteration, trainer.num_updates) == (iteration, updates)
    for name, model in models.items():
        torch.testing.assert_close(next(model.parameters()), probes[name], rtol=0, atol=0)
    for key, value in sampler.items():
        torch.testing.assert_close(getattr(trainer.command, key), value, rtol=0, atol=0)
    for name, opt in optimizers.items():
        _assert_tree_equal(opt.state_dict(), states[name])
    return {"passed": True, "optimizer_states_erased_and_restored": list(optimizers),
            "optimizer_parameter_groups_restored": True,
            "sampler_fields_restored": list(sampler)}


def verify_production_config(cfg: TrainConfig) -> dict:
    """Fail if a smoke-only simplification leaks into an optimization setting."""
    formal = TrainConfig()
    resolve_training_observations(cfg)
    resolve_training_observations(formal)
    wm = asdict(cfg.world_model)
    original_wm = asdict(formal.world_model)
    # Context is robot-specific; hidden networks and all optimizer settings match.
    wm["state_spec"].pop("context_layout")
    original_wm["state_spec"].pop("context_layout")
    assert wm == original_wm, "world_model optimization parameters changed"
    for name in ("wm_trainer", "critic"):
        assert asdict(getattr(cfg, name)) == asdict(getattr(formal, name)), name
    trainer = asdict(cfg.policy_trainer)
    original_trainer = asdict(formal.policy_trainer)
    trainer.pop("activation_checkpoint")
    original_trainer.pop("activation_checkpoint")
    for key in ("anchor_obs", "observation_layout", "history", "future_frames"):
        trainer.pop(key)
        original_trainer.pop(key)
    assert trainer == original_trainer, "policy optimization parameters changed"
    policy = asdict(cfg.policy)
    original = asdict(formal.policy)
    for key in ("num_joints", "num_tracked_bodies", "goal_dim", "observation_layout", "history", "future_frames",
                "obs_term_scale", "goal_term_scale"):
        policy.pop(key)
        original.pop(key)
    assert policy == original, "policy hidden architecture changed"
    schedule = asdict(cfg.schedule)
    reference_schedule = asdict(formal.schedule)
    schedule.pop("stop_after_iterations")
    reference_schedule.pop("stop_after_iterations")
    assert schedule == reference_schedule, "optimization schedule changed"
    env = asdict(cfg.env)
    reference_env = asdict(formal.env)
    env.pop("num_envs")
    reference_env.pop("num_envs")
    # Compact storage changes how references are held, not the optimization or physics.
    env.pop("compact_library")
    reference_env.pop("compact_library")
    env.pop("reference_fk_compile")
    reference_env.pop("reference_fk_compile")
    for key in ("anchor_obs", "observation_layout", "future_frames", "actuator_delay",
                "actuator_delay_max_ms", "reset_root_z_offset", "merge_short_tail_bins"):
        env.pop(key)
        reference_env.pop(key)
    assert env == reference_env, "observation/reward/randomization settings changed"
    return {
        "matches_train_config_optimization_defaults": True,
        "anchor_obs": cfg.env.anchor_obs,
        "observation_layout": cfg.env.observation_layout,
        "future_frames": cfg.env.future_frames,
        "history": cfg.policy.history,
        "execution_overrides": {"num_envs": cfg.env.num_envs,
                                "stop_after_iterations": cfg.schedule.stop_after_iterations,
                                "activation_checkpoint": cfg.policy_trainer.activation_checkpoint,
                                "compact_library": cfg.env.compact_library,
                                "reference_fk_compile": cfg.env.reference_fk_compile,
                                "actuator_delay": cfg.env.actuator_delay,
                                "merge_short_tail_bins": cfg.env.merge_short_tail_bins},
        "world_model_hidden": cfg.world_model.hidden,
        "policy_encoder_hidden": cfg.policy.encoder_hidden,
        "policy_decoder_hidden": cfg.policy.decoder_hidden,
        "critic_hidden": cfg.critic.hidden,
        "world_model_batch_size": cfg.wm_trainer.batch_size,
        "world_model_batches_per_update": cfg.wm_trainer.batches_per_update,
        "policy_batch_size": cfg.policy_trainer.batch_size,
        "imagination_horizon": cfg.policy_trainer.horizon,
        "critic_epochs": cfg.policy_trainer.critic_epochs,
        "critic_minibatches": cfg.policy_trainer.critic_minibatches,
        "utd": cfg.schedule.updates_per_env_step,
        "warmup_iterations": cfg.schedule.warmup_iterations,
        "lr_schedule_total_iterations": cfg.schedule.total_iterations,
        "compile": {"reference_fk": cfg.env.reference_fk_compile,
                    "world_model": cfg.wm_trainer.compile,
                    "policy": cfg.policy_trainer.compile},
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="khan_training_smoke")
    parser.add_argument("--log-root", type=Path, default=ROOT / "logs")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data-dir", type=Path, default=ROOT / "data/npz/khan_mini/lafan1")
    source.add_argument("--library", type=Path,
                        help="Packed 50 Hz Khan compact library; mutually exclusive with --data-dir")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--observation-layout", choices=("isaac_khan", "supermimic"), default="isaac_khan")
    parser.add_argument("--anchor-obs", action=argparse.BooleanOptionalAction, default=False,
                        help="Optional 9D goal pose for supermimic layout only; incompatible with isaac_khan")
    parser.add_argument("--activation-checkpoint", action=argparse.BooleanOptionalAction, default=False,
                        help="Optional activation recomputation; disabled to match the original training")
    parser.add_argument("--reference-fk-compile", action=argparse.BooleanOptionalAction, default=False,
                        help="Compile CUDA compact reference queries; disable for eager A/B checks")
    args = parser.parse_args(argv)
    if Path(args.run_name).name != args.run_name or args.run_name in ("", ".", ".."):
        parser.error("run-name must be one directory name")
    if args.num_envs < 1 or args.steps < 50:
        parser.error("num-envs must be positive and steps >= 50 (40 warmup + real updates)")
    if args.library is not None and (args.library.suffix != ".pt" or not args.library.is_file()):
        parser.error("--library must name an existing packed .pt file")
    return args


def configure_data_source(cfg: TrainConfig, args: argparse.Namespace) -> None:
    if args.library is not None:
        cfg.data.library = args.library.resolve()
        cfg.data.npz_dir = None
        cfg.env.compact_library = True
    else:
        cfg.data.npz_dir = args.data_dir.resolve()


def dataset_summary(trainer, cfg: TrainConfig) -> dict:
    """Report the references actually loaded, including packed storage precision."""
    library = trainer.lib
    names = [clip.name for clip in library.clips]
    lengths = [int(clip.length) for clip in library.clips]
    if len(names) != library.num_clips or len(set(names)) != len(names) or sum(lengths) != library.num_frames:
        raise ValueError("Loaded clip names/frame metadata are inconsistent")
    storage = library._ref
    for start in range(0, len(storage), 65536):
        if not bool(torch.isfinite(storage[start:start + 65536]).all().item()):
            raise ValueError("Loaded reference contains NaN/Inf after the runtime storage cast")
    packed = cfg.data.library is not None
    paths = ([str(cfg.data.library.resolve())] if packed else
             [str((cfg.data.npz_dir / (name + ".npz")).resolve()) for name in names])
    result = {
        "clips": library.num_clips, "frames": library.num_frames,
        "source_type": "packed_library" if packed else "npz_directory",
        "paths": paths, "library_path": paths[0] if packed else None,
        "compact": bool(library.compact), "fps": float(library.fps),
        "storage_dtype": str(storage.dtype), "reference_shape": list(storage.shape),
        "reference_storage_bytes": storage.numel() * storage.element_size(),
        "reference_storage_all_finite": True,
        "clip_names": names, "clip_starts": [int(c.start) for c in library.clips],
        "clip_lengths": lengths,
        "clip_length_frames": {"min": min(lengths), "max": max(lengths),
                               "mean": sum(lengths) / len(lengths)},
        "duration_hours": (library.num_frames - library.num_clips) / library.fps / 3600,
    }
    if packed:
        with cfg.data.library.open("rb") as stream:
            result["library_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    fixed_ranges = {}
    for joint, target in (("Left_wrist_yaw_joint", .3), ("Right_wrist_yaw_joint", .3),
                          ("Neck_yaw_joint", 0.), ("Neck_pitch_joint", 0.)):
        if joint not in library.joint_names:
            continue
        joint_id = library.joint_names.index(joint)
        position = storage[:, 7 + joint_id]
        velocity = storage[:, 13 + library.num_joints + joint_id]
        fixed_ranges[joint] = {
            "target_rad": target,
            "quantized_target_rad": float(torch.tensor(target, dtype=storage.dtype).item()),
            "minimum_rad": float(position.min().item()), "maximum_rad": float(position.max().item()),
            "max_abs_velocity_rad_s": float(velocity.abs().max().item()),
        }
    result["fixed_joint_reference_ranges"] = fixed_ranges
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    run_dir = args.log_root.resolve() / args.run_name
    if list(run_dir.glob("ckpt*.pt")) or (run_dir / "summary.json").exists():
        raise SystemExit(f"existing run {run_dir}; choose another run-name")
    cfg = TrainConfig(robot="khan_mini", run_name=args.run_name, log_root=args.log_root.resolve())
    configure_data_source(cfg, args)
    cfg.env.num_envs = args.num_envs
    cfg.env.reference_fk_compile = args.reference_fk_compile
    cfg.env.observation_layout = args.observation_layout
    cfg.env.anchor_obs = args.anchor_obs
    cfg.policy_trainer.anchor_obs = args.anchor_obs
    cfg.schedule.stop_after_iterations = args.steps
    cfg.policy_trainer.activation_checkpoint = args.activation_checkpoint
    report = {
        "status": "running", "started_at_utc": datetime.now(UTC).isoformat(),
        "production_config": verify_production_config(cfg),
        "versions": package_versions(), "gpu": torch.cuda.get_device_name(cfg.device),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(cfg.device).total_memory,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"[CHECK] production hyperparameters preserved; activation recomputation={args.activation_checkpoint}", flush=True)
    torch.cuda.reset_peak_memory_stats(cfg.device)
    trainer = Trainer.__new__(Trainer)
    started = time.perf_counter()
    try:
        trainer.__init__(cfg)
        free_bytes, total_bytes = torch.cuda.mem_get_info(cfg.device)
        report["memory_after_initialization"] = {
            "torch_allocated_bytes": torch.cuda.memory_allocated(cfg.device),
            "torch_reserved_bytes": torch.cuda.memory_reserved(cfg.device),
            "device_free_bytes": free_bytes, "device_total_bytes": total_bytes,
        }
        report["production_config"] = verify_production_config(cfg)
        models = {"wm": trainer.wm, "policy": trainer.policy, "critic": trainer.critic}
        before = {name: {key: value.detach().cpu().clone() for key, value in model.named_parameters()}
                  for name, model in models.items()}
        report["dataset"] = dataset_summary(trainer, cfg)
        report["dimensions"] = {
            "state": trainer.buffer.s.shape[-1], "action": cfg.policy.num_joints,
            "context": trainer.wm.spec.xi_dim, "goal": cfg.policy.goal_dim,
            "world_model_input": trainer.wm.dynamics.mlp.net[0].in_features,
            "proprioceptive_history": cfg.policy.history * (3 * cfg.policy.num_joints + 6),
            "policy_observation": policy_observation_dimension(
                cfg.policy.observation_layout, cfg.policy.num_joints, cfg.policy.history,
                cfg.policy.future_frames,
            ),
            "encoder_input": trainer.policy.encoders["motion"].mlp.net[0].in_features,
            "decoder_input": trainer.policy.decoder.mlp.net[0].in_features,
            "critic_input": trainer.critic.mlp.net[0].in_features,
            "tracked_bodies": len(trainer.robot_spec.tracked_bodies),
        }
        # Progress around expensive first compilation without changing training behavior.
        for label, branch in (("world model", trainer.wm_trainer), ("policy/critic", trainer.policy_trainer)):
            original_update = branch.update

            def logged_update(*values, _original=original_update, _label=label, **kwargs):
                report["stage"] = _label
                print(f"[UPDATE] {_label} at iteration {trainer.iteration}", flush=True)
                result = _original(*values, **kwargs)
                print(f"[UPDATE DONE] {_label}", flush=True)
                return result

            branch.update = logged_update
        trainer.run()
        assert trainer.iteration == args.steps
        expected_updates = math.floor((args.steps - cfg.schedule.warmup_iterations + 1)
                                      * cfg.schedule.updates_per_env_step + 1e-9)
        assert trainer.num_updates == expected_updates, (trainer.num_updates, expected_updates)
        report.update(iteration=trainer.iteration, num_updates=trainer.num_updates)
        report["model_checks"] = {name: verify_parameters(model, before[name])
                                  for name, model in models.items()}
        report["optimizer_checks"] = {
            "wm": verify_optimizer(trainer.wm_trainer.opt),
            "policy": verify_optimizer(trainer.policy_trainer.opt),
            "critic": verify_optimizer(trainer.policy_trainer.critic_opt),
        }
        expected = {"wm": trainer.num_updates * cfg.wm_trainer.batches_per_update,
                    "policy": trainer.num_updates * cfg.schedule.policy_updates_per_wm_update,
                    "critic": trainer.num_updates * cfg.schedule.policy_updates_per_wm_update
                              * cfg.policy_trainer.critic_epochs * cfg.policy_trainer.critic_minibatches}
        for name, steps in expected.items():
            assert report["optimizer_checks"][name]["min_step"] == steps, (name, steps)
            assert report["optimizer_checks"][name]["max_step"] == steps, (name, steps)
        checkpoint = run_dir / "ckpt_latest.pt"
        assert checkpoint.is_file()
        report["checkpoint_restore_checks"] = verify_checkpoint_restore(trainer, models, checkpoint)
        report["checkpoint"] = str(checkpoint)
        report["checkpoint_reload_passed"] = True
        trainer.writer.flush()
        report["metrics"] = read_metrics(sorted((run_dir / "tb").glob("events.out.tfevents.*")))
        report["tensorboard_all_scalars_finite"] = True
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["iteration_reached"] = getattr(trainer, "iteration", 0)
        report["completed_outer_updates"] = getattr(trainer, "num_updates", 0)
        report["config"] = asdict(cfg)
        report["wall_time_seconds"] = time.perf_counter() - started
        report["peak_torch_cuda_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(cfg.device)
        report["peak_torch_cuda_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(cfg.device)
        try:
            if getattr(trainer, "env", None) is not None:
                trainer.env.close()
        finally:
            if getattr(trainer, "writer", None) is not None:
                trainer.writer.close()
            (run_dir / "summary.json").write_text(json.dumps(report, indent=2, default=str, allow_nan=False) + "\n")
    print(f"[PASS] Khan production-configuration smoke: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

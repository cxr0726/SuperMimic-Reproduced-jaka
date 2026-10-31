"""Run a small, genuine CUDA training cycle and verify its saved artifacts.

Run from the repository with ``env -u PYTHONPATH uv run python scripts/smoke_train.py``.
This checks the training pipeline, not convergence or motion-tracking quality.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import asdict
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from supermimic.config import TrainConfig
from supermimic.training.loop import Trainer

ROOT = Path(__file__).resolve().parents[1]


def make_config(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig(run_name=args.run_name, log_root=args.log_root, seed=1)
    cfg.data.npz_dir = args.data_dir
    cfg.data.clips = ["walk1_subject1"]
    cfg.env.num_envs = 32
    cfg.env.replay_capacity = 64
    cfg.env.future_frames = 10
    cfg.world_model.hidden = (128, 128)
    cfg.policy.encoder_hidden = (128, 64)
    cfg.policy.decoder_hidden = (128, 64)
    cfg.policy.history = 10
    cfg.wm_trainer.window = 8
    cfg.wm_trainer.batch_size = 32
    cfg.wm_trainer.batches_per_update = 1
    cfg.wm_trainer.compile = None
    cfg.policy_trainer.batch_size = 32
    cfg.policy_trainer.horizon = 4
    cfg.policy_trainer.history = 10
    cfg.policy_trainer.future_frames = 10
    cfg.policy_trainer.critic_epochs = 1
    cfg.policy_trainer.critic_minibatches = 1
    cfg.policy_trainer.compile = None
    cfg.schedule.total_iterations = 16
    cfg.schedule.warmup_iterations = 12
    cfg.schedule.updates_per_env_step = 1.0
    cfg.schedule.log_interval = 4
    cfg.schedule.checkpoint_interval = 16
    return cfg


def package_versions() -> dict:
    versions = {"python": platform.python_version(), "torch_cuda_runtime": torch.version.cuda}
    for name in (
        "supermimic", "torch", "mjlab", "mujoco", "mujoco-warp", "warp-lang",
        "numpy", "tensorboard", "tyro",
    ):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    try:
        direct_url = metadata.distribution("mjlab").read_text("direct_url.json")
        if direct_url:
            versions["mjlab_source"] = json.loads(direct_url)
    except metadata.PackageNotFoundError:
        pass
    return versions


def verify_parameters(model, before: dict[str, torch.Tensor]) -> dict:
    changed = []
    for name, parameter in model.named_parameters():
        value = parameter.detach().cpu()
        assert torch.isfinite(value).all(), f"nonfinite model parameter: {name}"
        if not torch.equal(value, before[name]):
            changed.append(name)
    assert changed, "no model parameter changed during training"
    return {"all_finite": True, "changed_parameter_tensors": len(changed)}


def verify_optimizer(optimizer) -> dict:
    steps = [float(state["step"]) for state in optimizer.state.values() if "step" in state]
    assert steps and all(math.isfinite(step) and step > 0 for step in steps), (
        "optimizer did not complete a finite positive number of steps"
    )
    return {"parameter_states": len(steps), "min_step": min(steps), "max_step": max(steps)}


def read_metrics(event_files: list[Path]) -> dict:
    scalars: dict[str, list[dict]] = {}
    assert event_files, "TensorBoard event file was not created"
    for event_file in event_files:
        events = EventAccumulator(str(event_file), size_guidance={"scalars": 0}).Reload()
        for tag in events.Tags()["scalars"]:
            for event in events.Scalars(tag):
                assert math.isfinite(event.value), f"nonfinite TensorBoard scalar: {tag}"
                scalars.setdefault(tag, []).append({"step": event.step, "value": event.value})
    for tag in ("wm/loss", "wm/grad_norm", "policy/policy_grad_norm", "policy/critic_loss"):
        assert scalars.get(tag), f"training metric missing: {tag}"
    return {tag: sorted(values, key=lambda item: item["step"]) for tag, values in scalars.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="smoke_lafan1")
    parser.add_argument("--log-root", type=Path, default=ROOT / "logs")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "npz" / "lafan1")
    args = parser.parse_args()
    if args.run_name in ("", ".", "..") or Path(args.run_name).name != args.run_name:
        parser.error("--run-name must be a single directory name")
    args.log_root = args.log_root.resolve()
    args.data_dir = args.data_dir.resolve()
    run_dir = args.log_root / args.run_name
    if list(run_dir.glob("ckpt*.pt")):
        parser.error(f"checkpoints already exist in {run_dir}; choose another --run-name")
    if not (args.data_dir / "walk1_subject1.npz").is_file():
        parser.error(f"missing LAFAN1 clip: {args.data_dir / 'walk1_subject1.npz'}")
    if not torch.cuda.is_available():
        parser.error("a CUDA GPU is required for the MuJoCo Warp training environment")

    cfg = make_config(args)
    existing_events = set((run_dir / "tb").glob("events.out.tfevents.*"))
    report = {
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "config": asdict(cfg),
        "versions": package_versions(),
        "gpu": torch.cuda.get_device_name(cfg.device),
    }
    torch.cuda.reset_peak_memory_stats(cfg.device)
    started = time.perf_counter()
    # Keep access to partially constructed resources if initialization fails.
    trainer = Trainer.__new__(Trainer)
    try:
        trainer.__init__(cfg)
        models = {"wm": trainer.wm, "policy": trainer.policy, "critic": trainer.critic}
        before = {
            name: {key: value.detach().cpu().clone() for key, value in model.named_parameters()}
            for name, model in models.items()
        }
        trainer.run()
        assert trainer.iteration == 16, f"unexpected iteration: {trainer.iteration}"
        assert trainer.num_updates == 5, f"unexpected update count: {trainer.num_updates}"
        report["iteration"] = trainer.iteration
        report["num_updates"] = trainer.num_updates
        report["model_checks"] = {
            name: verify_parameters(model, before[name]) for name, model in models.items()
        }
        report["optimizer_checks"] = {
            "wm": verify_optimizer(trainer.wm_trainer.opt),
            "policy": verify_optimizer(trainer.policy_trainer.opt),
            "critic": verify_optimizer(trainer.policy_trainer.critic_opt),
        }
        checkpoint = run_dir / "ckpt_latest.pt"
        numbered = run_dir / "ckpt_0000016.pt"
        assert checkpoint.is_file() and numbered.is_file(), "checkpoint files missing"
        # Exercise actual model/optimizer/sampler restoration without a second simulator.
        trainer.iteration, trainer.num_updates = -1, -1
        trainer.load(checkpoint)
        assert trainer.iteration == 16 and trainer.num_updates == 5, "checkpoint restore failed"
        report["checkpoint"] = str(checkpoint)
        report["checkpoint_reload_passed"] = True
        trainer.writer.flush()
        new_events = sorted(set((run_dir / "tb").glob("events.out.tfevents.*")) - existing_events)
        report["metrics"] = read_metrics(new_events)
        report["tensorboard_all_scalars_finite"] = True
        report["config"] = asdict(cfg)
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["wall_time_seconds"] = time.perf_counter() - started
        report["peak_torch_cuda_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(cfg.device)
        report["peak_torch_cuda_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(cfg.device)
        try:
            if getattr(trainer, "env", None) is not None:
                trainer.env.close()
        finally:
            if getattr(trainer, "writer", None) is not None:
                trainer.writer.close()
            if run_dir.is_dir():
                (run_dir / "summary.json").write_text(
                    json.dumps(report, indent=2, default=str, allow_nan=False) + "\n"
                )
    print(f"[PASS] training smoke test: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

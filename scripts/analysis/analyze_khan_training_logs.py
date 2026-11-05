"""Export TensorBoard curves and checkpoint-aligned metrics for a trusted run.

Checkpoints are loaded with pickle enabled; only use this with trusted training
checkpoints. This script reads training artifacts and writes analysis outputs.
It does not initialize a simulation or change the training configuration.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def flatten(value: dict, prefix="") -> dict:
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        else:
            result[name] = item
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir_pos", nargs="?", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--out-dir", "--output-dir", dest="output_dir", type=Path)
    args = parser.parse_args()
    selected_run_dir = args.run_dir or args.run_dir_pos
    if selected_run_dir is None:
        parser.error("provide --run-dir or a positional run directory")
    run_dir = selected_run_dir.resolve()
    output_dir = (args.output_dir or run_dir / "analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep all events, including possible duplicates from resumed writers, so an
    # apparent monotonic curve cannot conceal an overlapping training segment.
    accumulator = EventAccumulator(
        str(run_dir / "tb"), size_guidance={"scalars": 0}, purge_orphaned_data=False
    )
    accumulator.Reload()
    tags = sorted(accumulator.Tags()["scalars"])
    curves = {tag: accumulator.Scalars(tag) for tag in tags}
    if not curves:
        raise RuntimeError(f"No TensorBoard scalar data found in {run_dir / 'tb'}")
    audit = {}
    with (output_dir / "tensorboard_scalars.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["tag", "step", "wall_time_unix_s", "value"])
        for tag, events in curves.items():
            steps = [event.step for event in events]
            audit[tag] = {
                "count": len(events),
                "first_step": min(steps),
                "last_step": max(steps),
                "duplicate_steps": len(steps) - len(set(steps)),
                "step_regressions": sum(b < a for a, b in pairwise(steps)),
                "nonfinite_values": sum(not math.isfinite(e.value) for e in events),
            }
            writer.writerows((tag, e.step, e.wall_time, e.value) for e in events)

    # Last event at a step is the final writer value when duplicated steps exist.
    by_step = {tag: {e.step: e for e in events} for tag, events in curves.items()}
    checkpoints = []
    benchmark_rows = []
    base_flat_config = None
    base_config_hash = None
    checkpoint_paths = sorted(
        path for path in run_dir.glob("ckpt_*.pt")
        if path.stem.removeprefix("ckpt_").isdigit()
    )
    for path in checkpoint_paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        iteration = int(checkpoint["iteration"])
        config = json_value(checkpoint["config"])
        reward_cfg = checkpoint.get("robot_contract", {}).get("reward_config")
        # Historical G1/14-body checkpoints used the original 0.1 cap. Modern
        # contracts record the reward weights, including Khan's neck rewards.
        max_step_reward = (reward_cfg["dt"] * sum(
            max(0.0, value) for key, value in reward_cfg.items() if key.startswith("w_")
        )) if reward_cfg else 0.1
        config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()
        flat_config = flatten(config)
        if base_flat_config is None:
            base_flat_config, base_config_hash = flat_config, config_hash
        config_differences = {
            key: {"first_checkpoint": base_flat_config.get(key), "this_checkpoint": flat_config.get(key)}
            for key in sorted(set(base_flat_config) | set(flat_config))
            if base_flat_config.get(key) != flat_config.get(key)
        }
        finite_check = {}
        for module_name in ("policy", "wm", "critic"):
            tensors = checkpoint[module_name]
            bad = [name for name, tensor in tensors.items() if not bool(torch.isfinite(tensor).all())]
            finite_check[module_name] = {
                "tensor_count": len(tensors),
                "parameter_elements": sum(t.numel() for t in tensors.values()),
                "nonfinite_tensor_names": bad,
                "all_finite": not bad,
            }
        metrics = {
            tag: events[iteration].value
            for tag, events in by_step.items() if iteration in events
        }
        prior_returns = [
            e.value for step, e in sorted(by_step["train/episode_norm_return"].items())
            if iteration - 1000 < step <= iteration
        ]
        entry = {
            "checkpoint": str(path.resolve()),
            "iteration": iteration,
            "filename_iteration_matches": int(path.stem.split("_")[-1]) == iteration,
            "num_updates": int(checkpoint["num_updates"]),
            "max_step_reward": max_step_reward,
            "tensorboard_exact_step_available": "train/episode_norm_return" in metrics,
            "metrics_at_exact_step": metrics,
            "train_return_mean_last_1000_iterations": float(np.mean(prior_returns)),
            "config_sha256": config_hash,
            "config_identical_to_first": config_hash == base_config_hash,
            "config_differences_from_first": config_differences,
            "config": config,
            "finite_tensor_checks": finite_check,
        }
        benchmark_path = output_dir / f"benchmark_{path.stem}.json"
        if benchmark_path.exists():
            benchmark = json.loads(benchmark_path.read_text())
            clips = benchmark["clips"]
            if len({clip["name"] for clip in clips}) != len(clips):
                raise ValueError(f"Duplicate motion names in {benchmark_path}")
            if benchmark["summary"]["clips"] != len(clips):
                raise ValueError(f"Inconsistent motion count in {benchmark_path}")
            total_frames = sum(clip["frames"] for clip in clips)
            total_steps = sum(clip["steps"] for clip in clips)
            for clip in clips:
                benchmark_rows.append({
                    "iteration": iteration,
                    **{key: clip[key] for key in
                       ("name", "frames", "steps", "norm_return", "failed", "fail_step")},
                    "completion_fraction": clip["steps"] / clip["frames"],
                    **{f"reward/{key}": value for key, value in (clip.get("terms") or {}).items()},
                })
            entry["benchmark_path"] = str(benchmark_path)
            entry["benchmark_summary"] = benchmark["summary"]
            entry["benchmark_extended"] = {
                "total_frames": total_frames,
                "total_executed_steps": total_steps,
                "completion_fraction": total_steps / max(total_frames, 1),
                "full_clip_weighted_return": sum(c["norm_return"] * c["steps"] for c in clips) / max(total_frames, 1),
                "failed_clip_names": [c["name"] for c in clips if c["failed"]],
            }
        else:
            entry["benchmark_path"] = None
            entry["benchmark_summary"] = None
            entry["benchmark_extended"] = None
        recheck_path = output_dir / "recheck" / f"benchmark_{path.stem}.json"
        if recheck_path.exists():
            recheck = json.loads(recheck_path.read_text())
            recheck_clips = recheck["clips"]
            entry["benchmark_recheck"] = {
                "path": str(recheck_path),
                "summary": recheck["summary"],
                "completion_fraction": sum(c["steps"] for c in recheck_clips)
                / sum(c["frames"] for c in recheck_clips),
                "successful_clip_names": [c["name"] for c in recheck_clips if not c["failed"]],
            }
        checkpoints.append(entry)
        del checkpoint
        gc.collect()
    if not checkpoints:
        raise RuntimeError(f"No numerically named ckpt_*.pt found in {run_dir}")

    intervals = [(50, 1000), (1000, 3000), (3000, 8000), (8000, 16000),
                 (16000, 40000), (40000, 100000), (100000, max(by_step["time/hours"]))]
    stage_stats = []
    for lower, upper in intervals:
        item = {"first_step_inclusive": lower, "last_step_inclusive": upper, "metrics": {}}
        for tag, points in by_step.items():
            events = [event for step, event in points.items() if lower <= step <= upper]
            if not events:
                continue
            lo = min(events, key=lambda e: e.value)
            hi = max(events, key=lambda e: e.value)
            item["metrics"][tag] = {
                "mean": float(np.mean([e.value for e in events])),
                "min": lo.value, "min_step": lo.step,
                "max": hi.value, "max_step": hi.step,
            }
        stage_stats.append(item)

    summary = {
        "run_dir": str(run_dir),
        "event_files": [str(p.resolve()) for p in sorted((run_dir / "tb").glob("*tfevents*"))],
        "scalar_audit": audit,
        "checkpoints": checkpoints,
        "stage_statistics": stage_stats,
        "metric_definitions": {
            "training_return": "Exact unsmoothed train/episode_norm_return event at checkpoint iteration; mean of finished-episode returns divided by max_step_reward * executed episode steps, over the preceding log interval (50 vectorized steps).",
            "training_hours": "Exact time/hours event; elapsed wall time recorded by this training process, including compilation. Not GPU-hours.",
            "episode_length": "train/episode_len is measured in policy control steps; seconds = steps * config.world_model.state_spec.dt.",
            "survival": "World-model-predicted product of (1 - termination probability) over the imagined horizon; not measured simulator survival.",
            "wm_errors": "wm/nse_1step comes from training batches; wm/online_nse_8step comes from sampled online replay, not a fixed held-out set.",
            "stages": "Interval endpoint inclusivity is recorded per row; adjacent stage summaries share their boundary event.",
            "benchmark_mean": "Original benchmark arithmetic mean of per-clip normalized return; each clip starts at its first frame, no noise/DR, and ends at training termination or clip end.",
            "benchmark_step_weighted": "Original benchmark sum(clip.norm_return * clip.steps) / sum(clip.steps), using executed steps rather than full reference lengths.",
            "completion_fraction": "Additional diagnostic sum(executed clip steps) / sum(full clip frame counts), retaining the original benchmark's frame/step convention.",
            "full_clip_weighted_return": "Additional diagnostic sum(clip.norm_return * clip.steps) / sum(full clip frame counts); not a replacement for the original benchmark metric.",
        },
    }
    (output_dir / "tensorboard_analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    if benchmark_rows:
        fields = list(dict.fromkeys(key for row in benchmark_rows for key in row))
        with (output_dir / "benchmark_per_clip.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(benchmark_rows)
    csv_fields = ["checkpoint", "iteration", "num_updates", "training_hours", "train_return",
                  "episode_length_steps", "episode_length_seconds", "wm_nse_1step", "wm_online_nse_8step",
                  "imag_reward_mean", "imag_survival_H", "train_return_mean_last_1000_iterations",
                  "benchmark_clips", "benchmark_failed", "benchmark_mean_norm_return",
                  "benchmark_step_weighted_norm_return", "completion_fraction", "full_clip_weighted_return"]
    with (output_dir / "checkpoint_training_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()
        for ck in checkpoints:
            m = ck["metrics_at_exact_step"]
            b = ck["benchmark_summary"] or {}
            extended = ck["benchmark_extended"] or {}
            writer.writerow({
                "checkpoint": Path(ck["checkpoint"]).name,
                "iteration": ck["iteration"], "num_updates": ck["num_updates"],
                "training_hours": m.get("time/hours"),
                "train_return": m.get("train/episode_norm_return"),
                "episode_length_steps": m.get("train/episode_len"),
                "episode_length_seconds": m.get("train/episode_len", float("nan")) * ck["config"]["world_model"]["state_spec"]["dt"],
                "wm_nse_1step": m.get("wm/nse_1step"),
                "wm_online_nse_8step": m.get("wm/online_nse_8step"),
                "imag_reward_mean": m.get("policy/imag_reward_mean"),
                "imag_survival_H": m.get("policy/imag_survival_H"),
                "train_return_mean_last_1000_iterations": ck["train_return_mean_last_1000_iterations"],
                "benchmark_clips": b.get("clips"),
                "benchmark_failed": b.get("failed"),
                "benchmark_mean_norm_return": b.get("mean_norm_return"),
                "benchmark_step_weighted_norm_return": b.get("step_weighted_norm_return"),
                "completion_fraction": extended.get("completion_fraction"),
                "full_clip_weighted_return": extended.get("full_clip_weighted_return"),
            })

    dataset_names = sorted({
        Path(ck["config"].get("data", {}).get("npz_dir") or "dataset").name
        for ck in checkpoints
    })
    dataset_label = ", ".join(dataset_names)
    report = [
        f"# Khan {dataset_label} checkpoint evaluation", "",
        f"Run directory: `{run_dir}`", "",
        ("Training return is the exact unsmoothed TensorBoard event at the checkpoint iteration. "
         "Hours are elapsed training-process wall time, not GPU-hours. Benchmark columns retain the original repository formulas."), "",
        "| Checkpoint | Training hours | Training return | Benchmark mean / step-weighted | Failed clips | Completion fraction | Full-clip-weighted return |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ck in checkpoints:
        m, b, ex = ck["metrics_at_exact_step"], ck["benchmark_summary"], ck["benchmark_extended"]
        b_return = f"{b['mean_norm_return']:.6f} / {b['step_weighted_norm_return']:.6f}" if b else "pending"
        failed = f"{b['failed']}/{b['clips']}" if b else "pending"
        complete = f"{ex['completion_fraction']:.6%}" if ex else "pending"
        weighted = f"{ex['full_clip_weighted_return']:.6f}" if ex else "pending"
        report.append(f"| {ck['iteration']} | {m['time/hours']:.6f} | {m['train/episode_norm_return']:.6f} | {b_return} | {failed} | {complete} | {weighted} |")
    report.extend(["", "Failure locations (training height/tilt termination):", ""])
    for ck in checkpoints:
        if ck["benchmark_extended"] is None:
            continue
        failed = [row for row in benchmark_rows
                  if row["iteration"] == ck["iteration"] and row["failed"]]
        locations = "; ".join(
            f"{row['name']} at step {row['steps']}/{row['frames']}" for row in failed
        ) or "none"
        report.append(f"- {ck['iteration']}: {locations}")
    rechecked = [ck for ck in checkpoints if "benchmark_recheck" in ck]
    if rechecked:
        report.extend(["", "Independent repeats (the main table retains the first evaluation):", ""])
        for ck in rechecked:
            repeated = ck["benchmark_recheck"]
            rs = repeated["summary"]
            report.append(
                f"- {ck['iteration']}: mean / executed-step weighted "
                f"{rs['mean_norm_return']:.6f} / {rs['step_weighted_norm_return']:.6f}; "
                f"failed {rs['failed']}/{rs['clips']}; completed steps "
                f"{repeated['completion_fraction']:.6%}."
            )
        report.extend(["", ("Noiseless policy evaluation does not guarantee identical numerical trajectories "
                       "across repeated GPU simulations; the repeat is kept separately rather than averaged into the main table."), ""])
    report.extend([
        "", ("Per-motion scores, failure positions and reward terms are in `benchmark_per_clip.csv`. "
         "All available TensorBoard scalar events are exported to `tensorboard_scalars.csv`; "
         "checkpoint configuration, differences, finite-tensor checks and stage statistics are in `tensorboard_analysis.json`."), "",
        ("`completion_fraction` and `full_clip_weighted_return` are additional diagnostics using full reference lengths. "
         "They retain the repository frame/step convention and do not change the original benchmark scores."), "",
        "![Training curves](training_curves.png)", "",
        "![Imagined reward components](imagined_reward_components.png)", "",
    ])
    evaluated = [ck for ck in checkpoints if ck["benchmark_summary"] is not None]
    if evaluated:
        best_full = max(evaluated, key=lambda ck: ck["benchmark_extended"]["full_clip_weighted_return"])
        fewest_failures = min(evaluated, key=lambda ck: (
            ck["benchmark_summary"]["failed"],
            -ck["benchmark_extended"]["full_clip_weighted_return"],
        ))
        report.extend([
            (f"Highest full-clip-weighted return: checkpoint {best_full['iteration']}. "
             f"Fewest failed clips: checkpoint {fewest_failures['iteration']}."), "",
            ("A high executed-step-weighted return does not imply full-clip stability: "
             "steps after an early failure are absent from its denominator."), "",
            "![Benchmark comparison](benchmark_comparison.png)", "",
        ])
    (output_dir / "checkpoint_evaluation_report.md").write_text("\n".join(report))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.22})

    def draw(axis, tag, label, factor=1.0, limit=None, **kwargs):
        points = sorted(by_step[tag].values(), key=lambda e: e.step)
        if limit is not None:
            points = [e for e in points if e.step <= limit]
        axis.plot([e.step / 1000 for e in points], [e.value * factor for e in points],
                  label=label, linewidth=1.0, **kwargs)

    fig, axes = plt.subplots(3, 2, figsize=(13.2, 11.0), constrained_layout=True)
    axis = axes[0, 0]
    draw(axis, "train/episode_norm_return", "Training return")
    for index, ck in enumerate(checkpoints):
        x, y = ck["iteration"] / 1000, ck["metrics_at_exact_step"]["train/episode_norm_return"]
        axis.scatter(x, y, s=24, color="tab:red", zorder=4)
        offset = (3, -16) if index % 2 == 0 else (3, 10)
        axis.annotate(f"{ck['iteration']//1000}k", (x, y), xytext=offset, textcoords="offset points", fontsize=8)
    axis.set(title="Training return: complete run", ylabel="Normalized return", ylim=(0, 1))
    axis = axes[0, 1]
    draw(axis, "train/episode_norm_return", "Training return", limit=12000)
    axis.set(title="Initial decline and recovery", ylabel="Normalized return", ylim=(0, 1))
    axis.legend(loc="lower right")
    axis = axes[1, 0]
    draw(axis, "wm/nse_1step", "1-step (training batch)")
    draw(axis, "wm/online_nse_8step", "8-step (online replay)")
    axis.set(title="World-model prediction error", ylabel="Normalized squared error", yscale="log")
    axis.legend()
    axis = axes[1, 1]
    dt = checkpoints[0]["config"]["world_model"]["state_spec"]["dt"]
    draw(axis, "train/episode_len", "Finished-episode mean", factor=dt)
    axis.set(title="Training episodes", ylabel="Episode length (s)")
    axis = axes[2, 0]
    reward_caps = {ck["max_step_reward"] for ck in checkpoints}
    if len(reward_caps) == 1:
        cap = next(iter(reward_caps))
        draw(axis, "policy/imag_reward_mean", f"Imagined reward / {cap:g}", factor=1 / cap)
    else:
        draw(axis, "policy/imag_reward_mean", "Imagined reward (raw; reward caps differ)")
    draw(axis, "policy/imag_survival_H", "Predicted H-step survival")
    axis.set(title="Imagination diagnostics", ylabel="Value", ylim=(0, 1.05))
    axis.legend()
    axis = axes[2, 1]
    draw(axis, "policy/critic_loss", "Critic loss")
    draw(axis, "policy/policy_grad_norm", "Actor grad norm (before clipping)", alpha=0.7)
    axis.set(title="Optimization diagnostics", ylabel="Value", yscale="log")
    axis.legend()
    for axis in axes.flat:
        axis.set_xlabel("Training iteration (thousands)")
    num_envs = checkpoints[0]["config"]["env"]["num_envs"]
    fig.suptitle(f"Khan {dataset_label}: {num_envs} environments, raw TensorBoard scalar events", fontsize=14)
    fig.savefig(output_dir / "training_curves.png", dpi=170)
    plt.close(fig)

    reward_tags = [tag for tag in tags if tag.startswith("policy/imag_r/")]
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), constrained_layout=True)
    for axis, tag in zip(axes.flat, reward_tags):
        draw(axis, tag, tag.split("/")[-1])
        axis.set(title=tag.split("/")[-1], xlabel="Training iteration (thousands)", ylabel="Mean term value")
    fig.suptitle("Imagined reward components (not measured simulator reward components)", fontsize=13)
    fig.savefig(output_dir / "imagined_reward_components.png", dpi=150)
    plt.close(fig)
    if evaluated:
        x = np.arange(len(evaluated))
        labels = [f"{ck['iteration'] / 1000:g}k" for ck in evaluated]
        clip_counts = sorted({int(ck["benchmark_summary"]["clips"]) for ck in evaluated})
        clip_count_label = "/".join(str(count) for count in clip_counts)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
        for key, label in (("mean_norm_return", "Clip mean"),
                           ("step_weighted_norm_return", "Executed-step weighted")):
            axes[0].plot(x, [ck["benchmark_summary"][key] for ck in evaluated],
                         marker="o", label=label)
        axes[0].plot(x, [ck["benchmark_extended"]["full_clip_weighted_return"] for ck in evaluated],
                     marker="o", label="Full-clip weighted (diagnostic)")
        axes[0].set(title="Benchmark returns", ylabel="Normalized return", ylim=(0, 1.03))
        axes[0].legend(loc="lower left")
        failures = [ck["benchmark_summary"]["failed"] for ck in evaluated]
        bars = axes[1].bar(x, failures, color="tab:red", alpha=0.65, label="Failed clips")
        axes[1].bar_label(bars, padding=3)
        axes[1].set(title="Full-motion stability", ylabel=f"Failed clips (out of {clip_count_label})",
                    ylim=(0, max(1, max(failures)) * 1.2))
        axes[1].yaxis.get_major_locator().set_params(integer=True)
        coverage_axis = axes[1].twinx()
        coverage_axis.grid(False)
        coverage_axis.plot(x, [100 * ck["benchmark_extended"]["completion_fraction"] for ck in evaluated],
                           color="tab:blue", marker="o", label="Executed / full steps")
        coverage_axis.set(ylabel="Executed / full reference steps (%)", ylim=(0, 103))
        axes[1].legend(loc="upper left")
        coverage_axis.legend(loc="upper right")
        for axis in axes:
            axis.set_xticks(x, labels)
            axis.set_xlabel("Checkpoint")
        fig.suptitle(f"Khan {dataset_label}: original benchmark on {clip_count_label} clips", fontsize=13)
        fig.savefig(output_dir / "benchmark_comparison.png", dpi=170)
        plt.close(fig)
    print(json.dumps({
        "output_dir": str(output_dir),
        "scalar_tag_count": len(tags),
        "checkpoints": [{"iteration": ck["iteration"], "training_hours": ck["metrics_at_exact_step"].get("time/hours"),
                          "train_return": ck["metrics_at_exact_step"].get("train/episode_norm_return"),
                          "config_identical": ck["config_identical_to_first"],
                          "finite_tensor_checks": ck["finite_tensor_checks"]} for ck in checkpoints],
    }, indent=2))


if __name__ == "__main__":
    main()

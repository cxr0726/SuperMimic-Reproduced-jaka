"""Same-process, fixed-model A/B of reference FK inside actor BPTT.

Only the compact library's reference-body callable changes. Models, start states,
noise RNG, history, horizon and activation recomputation remain identical. No
optimizer step is performed. This measures an actor update's forward/backward,
including critic bootstrap features, but excludes the subsequent critic fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from supermimic.config import EnvCfg, TrainConfig, resolve_training_observations
from supermimic.data.replay_buffer import StartBatch
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.imagination.objective import policy_objective
from supermimic.imagination.policy_trainer import PolicyTrainer, PolicyTrainerCfg
from supermimic.imagination.rollout import imagine
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Critic, CriticCfg, PolicyCfg
from supermimic.robots.registry import get_robot_spec
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


def restore_config(obj, saved):
    """Restore checkpoint values without applying fresh-policy defaults.

    A missing action/noise field denotes the original policy, not the new
    robot-specific defaults in TrainConfig's policy factory.
    """
    for field in fields(obj):
        if field.name not in saved:
            continue
        value = saved[field.name]
        current = getattr(obj, field.name)
        if isinstance(current, PolicyCfg):
            current = PolicyCfg()
            setattr(obj, field.name, current)
        elif isinstance(current, CriticCfg):
            current = CriticCfg()
            setattr(obj, field.name, current)
        elif isinstance(current, WorldModelCfg):
            current = WorldModelCfg()
            setattr(obj, field.name, current)
        elif isinstance(current, PolicyTrainerCfg):
            current = PolicyTrainerCfg()
            setattr(obj, field.name, current)
        elif isinstance(current, EnvCfg):
            current = EnvCfg(reset_sampler="legacy")
            setattr(obj, field.name, current)
        if is_dataclass(current):
            restore_config(current, value)
        else:
            setattr(obj, field.name, tuple(value) if isinstance(current, tuple) else value)


def load_benchmark_checkpoint(path, device):
    """Use the production policy loader, including residual baseline checks.

    The second checkpoint view is memory-mapped for WM/critic tensors: it does
    not copy the entire policy/optimizer payload into another CPU allocation.
    """
    policy, saved_config = load_policy_from_checkpoint(path, "cpu")
    cfg = TrainConfig()
    restore_config(cfg, saved_config)
    cfg.policy = policy.cfg
    # The production loader also resolves missing historical observation
    # fields. Do not reinterpret those as the robot's fresh-training layout.
    cfg.env.observation_layout = policy.cfg.observation_layout
    cfg.env.future_frames = policy.cfg.future_frames
    resolve_training_observations(cfg)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return cfg, policy.to(device), checkpoint


def weights_hash(models):
    result = hashlib.sha256()
    for model in models:
        for key, value in model.state_dict().items():
            result.update(key.encode())
            result.update(value.detach().contiguous().cpu().numpy().tobytes())
    return result.hexdigest()


def snapshot(ro, errors, policy, objective):
    return {
        "objective": objective.detach().cpu(),
        "terminal_state": ro.states[-1].detach().cpu(),
        "rewards": ro.rewards.detach().cpu(),
        "critic_terminal_errors": errors[-1].detach().cpu(),
        "policy_gradient": torch.cat([p.grad.detach().reshape(-1).cpu() for p in policy.parameters()]),
    }


def difference(actual, expected):
    results = {}
    for key, a in actual.items():
        b = expected[key]
        d = (a - b).double()
        results[key] = {"max_abs": float(d.abs().max()), "rms": float(d.square().mean().sqrt()),
                        "relative_l2": float(d.norm() / b.double().norm().clamp_min(1e-30))}
        if not torch.isfinite(a).all():
            raise AssertionError(f"Nonfinite {key}")
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--activation-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if min(args.samples, args.warmup) < 1 or not torch.cuda.is_available():
        parser.error("positive samples/warmup and CUDA are required")
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
    reference_events, reference_strides = [], []

    def timed_bodies(query):
        before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        before.record()
        result = original_bodies(query)
        after.record()
        reference_events.append((before, after))
        reference_strides.append({key: {"shape": list(value.shape), "stride": list(value.stride()),
                                       "contiguous": value.is_contiguous()}
                                  for key, value in vars(result).items()})
        return result

    lib.bodies = timed_bodies
    rng_cpu, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
    initial_weights = weights_hash((wm, policy, critic))

    def run(kind, capture=False):
        lib._reference_bodies_fn = eager_fn if kind == "eager" else compiled_fn
        policy.zero_grad(set_to_none=True)
        torch.set_rng_state(rng_cpu)
        torch.cuda.set_rng_state(rng_cuda)
        reference_events.clear()
        reference_strides.clear()
        torch.cuda.synchronize()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        begin = time.perf_counter()
        events[0].record()
        ro = imagine(wm, policy, lib, start, horizon, cfg.policy_trainer.future_frames,
                     obs_noise=cfg.policy_trainer.obs_noise, step_fn=trainer.step_fn,
                     anchor_obs=cfg.policy_trainer.anchor_obs,
                     activation_checkpoint=cfg.policy_trainer.activation_checkpoint)
        events[1].record()
        feature, goals, root_error, errors = trainer._critic_inputs(ro)
        events[2].record()
        value = critic(feature[-1], goals[-1], wm.spec.normalize_xi(start.xi),
                       root_error[-1], errors[-1])
        objective = policy_objective(ro, value, cfg.policy_trainer.gamma)
        (-objective / horizon).backward()
        events[3].record()
        torch.cuda.synchronize()
        result = {"mode": kind, "wall_ms": (time.perf_counter() - begin) * 1000,
                  "cuda_imagine_ms": events[0].elapsed_time(events[1]),
                  "cuda_critic_inputs_ms": events[1].elapsed_time(events[2]),
                  "cuda_bootstrap_backward_ms": events[2].elapsed_time(events[3]),
                  "cuda_total_ms": events[0].elapsed_time(events[3]),
                  "cuda_reference_fk_ms": sum(a.elapsed_time(b) for a, b in reference_events),
                  "reference_strides": reference_strides,
                  "objective": float(objective.detach()),
                  "gradient_norm": float(torch.stack([p.grad.norm().square() for p in policy.parameters()]).sum().sqrt())}
        saved = snapshot(ro, errors, policy, objective) if capture else None
        del ro, feature, goals, root_error, errors, value, objective
        return result, saved

    warm = []
    for _ in range(args.warmup):
        for kind in ("eager", "compiled"):
            metrics, _ = run(kind)
            warm.append(metrics)
            print("warmup", json.dumps({k: v for k, v in metrics.items() if k != "reference_strides"}), flush=True)
    results, snapshots = [], {}
    for repeat in range(args.samples):
        order = ("eager", "compiled") if repeat % 2 == 0 else ("compiled", "eager")
        for kind in order:
            metrics, saved = run(kind, capture=kind not in snapshots)
            if saved is not None:
                snapshots[kind] = saved
            results.append(metrics)
            print("measurement", json.dumps({k: v for k, v in metrics.items() if k != "reference_strides"}), flush=True)
    assert weights_hash((wm, policy, critic)) == initial_weights
    delta = difference(snapshots["compiled"], snapshots["eager"])
    report = {"timestamp": datetime.now(UTC).isoformat(), "checkpoint": str(args.checkpoint.resolve()),
              "library": str(args.library.resolve()), "gpu": torch.cuda.get_device_name(),
              "batch": n, "horizon": horizon, "history": history,
              "activation_checkpoint": cfg.policy_trainer.activation_checkpoint,
              "models_unchanged_sha256": initial_weights, "rng_restored_each_trial": True,
              "differences": delta, "warmup": warm, "measurements": results,
              "median": {kind: {key: statistics.median(r[key] for r in results if r["mode"] == kind)
                                for key in results[0] if key not in ("mode", "reference_strides")}
                         for kind in ("eager", "compiled")},
              "limits": "Fixed reference-based starts with neutral xi, not replay visitation. Identical model weights/RNG in one process; actor BPTT and critic bootstrap included, critic fitting and optimizer step excluded."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("differences", json.dumps(delta), flush=True)
    print("saved", args.output, flush=True)


if __name__ == "__main__":
    main()

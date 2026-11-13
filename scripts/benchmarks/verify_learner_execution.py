"""Pair execution variants on identical real replay, minibatches and permutations.

Observation noise is disabled only during this controlled learner comparison.
Fresh-noise behavior and sampler probabilities have separate tests. This checks
finite-budget numerical equivalence, not long-run tracking quality or throughput.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from benchmark_learner_updates import (
    _cpu_state,
    _precision_settings,
    _prefill,
    _release_learners,
    _state_hash,
    _steps,
)
from benchmark_reference_fk_imagination import restore_config

from supermimic.config import TrainConfig
from supermimic.imagination.policy_trainer import PolicyTrainer
from supermimic.observation_layout import (
    critic_reference_extras,
    isaac_anchor_orientation,
    isaac_goal_command,
)
from supermimic.training.loop import Trainer
from supermimic.world_model.heads import analytic_observation, flatten_history_term_major
from supermimic.world_model.wm_trainer import WorldModelTrainer, reference_at


def error(actual, expected):
    delta = actual.double() - expected.double()
    return {"max_abs": float(delta.abs().max()),
            "relative_l2": float(delta.norm() / expected.double().norm().clamp_min(1e-12))}


@torch.no_grad()
def probes(trainer, start):
    wm, lib = trainer.wm, trainer.lib
    window = lib.reference_window(start.frame, trainer.cfg.policy.future_frames)
    goal = isaac_goal_command(window.root_pos_w, window.root_quat_w, window.joint_pos)
    orientation = isaac_anchor_orientation(start.state[:, 3:7], window.root_quat_w)
    history = torch.stack([analytic_observation(s, a, start.xi, lib.default_joint_pos, "isaac_khan")
                           for s, a in zip(start.hist_states, start.hist_actions, strict=True)])
    obs = torch.cat((orientation, flatten_history_term_major(history, observation_layout="isaac_khan")), -1)
    ref = reference_at(lib, start.frame)
    critic_goal = torch.cat((goal, orientation, critic_reference_extras(
        window.joint_vel, window.root_quat_w, window.root_ang_vel_w, window.root_lin_vel_w)), -1)
    return {"obs": obs, "goal": goal, "state": start.state, "xi": start.xi,
            "feat": wm.spec.canonical(start.state), "critic_goal": critic_goal,
            "xi_norm": wm.spec.normalize_xi(start.xi), "root_error": wm.heads.root_error_features(start.state, ref),
            "tracking_error": wm.heads.critic_error_features(start.state, ref)}


@torch.no_grad()
def outputs(trainer, data, action):
    return {"actor": trainer.policy(data["obs"], data["goal"]).cpu(),
            "ema": trainer.policy_trainer.ema.module(data["obs"], data["goal"]).cpu(),
            "wm_next_state": trainer.wm.step(data["state"], action, data["xi"]).cpu(),
            "critic": trainer.critic(data["feat"], data["critic_goal"], data["xi_norm"],
                                     data["root_error"], data["tracking_error"]).cpu()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--report-only", action="store_true",
                        help="Keep finite/count checks, report rather than reject elementwise FP differences")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or args.updates < 5:
        parser.error("CUDA and at least five updates (warmup/capture/replay) are required")
    cfg = TrainConfig(robot="khan_mini")
    restore_config(cfg, json.loads(args.config.read_text()))
    cfg.device, cfg.env.num_envs, cfg.resume = args.device, args.num_envs, None
    cfg.replay_sampling = "multinomial"
    cfg.wm_trainer.cuda_graph = cfg.wm_trainer.compile_full_window = False
    cfg.policy_trainer.cuda_graph = cfg.policy_trainer.compile_networks = False
    cfg.policy_trainer.batch_reference_encoder = False
    cfg.log_root, cfg.run_name = args.output.parent, args.output.stem + "_scene"
    if args.library:
        cfg.data.library = args.library
    trainer = Trainer(cfg)
    original_permutation = torch.randperm
    baseline = []
    capturable_reference = []
    initial_values = None
    report = {"purpose": "controlled learner numerical comparison, not convergence or speed",
              "gpu": torch.cuda.get_device_name(torch.device(args.device)),
              "updates_per_mode": args.updates, "fixed_samples": True,
              "fixed_critic_permutations": True, "imagination_obs_noise": False,
              "activation_checkpoint": cfg.policy_trainer.activation_checkpoint,
              "horizon": cfg.policy_trainer.horizon, "wm_batch": cfg.wm_trainer.batch_size,
              "policy_batch": cfg.policy_trainer.batch_size, "modes": []}
    report["strict_elementwise_comparison"] = not args.report_only
    try:
        report["prefill"] = _prefill(trainer, 96)
        snapshot = {name: _cpu_state(getattr(trainer, name)) for name in ("wm", "policy", "critic")}
        report["initial_model_sha256"] = _state_hash(snapshot)
        buffer = trainer.buffer
        selections = {
            "windows": buffer.prepare_windows(cfg.wm_trainer.window),
            "starts": buffer.prepare_starts(cfg.policy_trainer.history, horizon=cfg.policy_trainer.horizon,
                                            clip_last=trainer.lib.clip_last),
        }
        generator = torch.Generator(device=args.device).manual_seed(831)
        fixed = {}
        for kind, selection in selections.items():
            n = cfg.wm_trainer.batch_size if kind == "windows" else cfg.policy_trainer.batch_size
            indices = torch.multinomial(selection.weights, n, replacement=True, generator=generator)
            fixed[kind] = (indices // buffer.T, indices % buffer.T)

        def fixed_draw(selection, n):
            env, slot = fixed[selection.kind]
            if n > env.numel():
                raise ValueError("Unprepared controlled batch size")
            return env[:n], slot[:n]

        buffer._sample_from_selection = fixed_draw
        torch.randperm = lambda n, *, device=None, **kwargs: torch.arange(n, device=device)
        start = buffer.sample_starts(32, cfg.policy_trainer.history, horizon=cfg.policy_trainer.horizon,
                                     clip_last=trainer.lib.clip_last)
        data = probes(trainer, start)
        with torch.no_grad():
            action = trainer.policy(data["obs"], data["goal"]).detach()
        modes = {"unaccelerated": (False, False, False, "multinomial"),
                 "batch_encoder_only": (True, False, False, "multinomial"),
                 "capturable_no_graph": (True, True, False, "multinomial"),
                 "d98f4df": (True, True, False, "multinomial"),
                 "cdf_fusion": (True, True, True, "cdf")}
        for mode, (batch_encoder, capture, fusion, sampler) in modes.items():
            _release_learners(trainer)
            for name, state in snapshot.items():
                getattr(trainer, name).load_state_dict(state)
            wc, pc = copy.deepcopy(cfg.wm_trainer), copy.deepcopy(cfg.policy_trainer)
            wc.cuda_graph, wc.compile_full_window = capture, fusion
            pc.cuda_graph, pc.compile_networks, pc.batch_reference_encoder = capture, fusion, batch_encoder
            pc.obs_noise = False
            buffer.sampling = sampler
            trainer.wm_trainer = WorldModelTrainer(trainer.wm, trainer.lib, wc)
            trainer.policy_trainer = PolicyTrainer(trainer.wm, trainer.policy, trainer.critic, trainer.lib, pc)
            if mode == "capturable_no_graph":
                # Keep device Adam, stable gradient buffers and step compile
                # settings, but run their kernels without capture/replay.
                trainer.wm_trainer.update_graph.enabled = False
                trainer.policy_trainer._captured_update.enabled = False
            restored = {name: _cpu_state(getattr(trainer, name)) for name in snapshot}
            assert _state_hash(restored) == report["initial_model_sha256"]
            initial = outputs(trainer, data, action)
            if initial_values is None:
                initial_values = initial
            for name in initial:
                torch.testing.assert_close(initial[name], initial_values[name], rtol=0, atol=0)
            row = {"mode": mode, "precision": _precision_settings(), "initial_outputs_bitwise_equal": True,
                   "initial_model_sha256": _state_hash(restored), "rounds": []}
            report["modes"].append(row)
            for update in range(args.updates):
                metrics = trainer.wm_trainer.update(buffer)
                wm_parameters = [value.detach().clone() for value in trainer.wm.parameters()]
                wm_gradients = [None if value.grad is None else value.grad.detach().clone()
                                for value in trainer.wm.parameters()]
                gradient_addresses = [None if value.grad is None else value.grad.data_ptr()
                                      for value in trainer.wm.parameters()]
                metrics.update(trainer.policy_trainer.update(buffer))
                for value, expected, gradient, address in zip(
                    trainer.wm.parameters(), wm_parameters, wm_gradients, gradient_addresses, strict=True,
                ):
                    assert torch.equal(value, expected), "Actor update modified WM parameters"
                    if gradient is not None:
                        assert value.grad.data_ptr() == address, "Actor update rebound WM gradient storage"
                        assert torch.equal(value.grad, gradient), "Actor update modified WM gradients"
                    else:
                        assert value.grad is None
                del wm_parameters, wm_gradients
                values = outputs(trainer, data, action)
                if mode == "unaccelerated":
                    baseline.append(values)
                if mode == "capturable_no_graph":
                    capturable_reference.append(values)
                differences = {name: error(value, baseline[update][name]) for name, value in values.items()}
                close = {name: bool(torch.isclose(value, baseline[update][name], rtol=5e-4, atol=5e-6).all())
                         for name, value in values.items()}
                row["rounds"].append({"update": update + 1, "functional_errors": differences,
                                      "within_elementwise_tolerance": close,
                                      "wm_frozen_through_actor": True,
                                      "wm_gradient_addresses": gradient_addresses})
                if mode == "d98f4df":
                    row["rounds"][-1]["difference_from_capturable_no_graph"] = {
                        name: error(value, capturable_reference[update][name]) for name, value in values.items()
                    }
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({"mode": mode, **row["rounds"][-1]}), flush=True)
                for name, value in values.items():
                    assert torch.isfinite(value).all(), (mode, update, name)
                    if not args.report_only:
                        torch.testing.assert_close(value, baseline[update][name], rtol=5e-4, atol=5e-6,
                                                   msg=f"{mode} update={update + 1} {name}")
                assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
            row["optimizer_steps"] = {"wm": _steps(trainer.wm_trainer.opt),
                                      "actor": _steps(trainer.policy_trainer.opt),
                                      "critic": _steps(trainer.policy_trainer.critic_opt)}
            expected = {"wm": args.updates * wc.batches_per_update, "actor": args.updates,
                        "critic": args.updates * pc.critic_epochs * pc.critic_minibatches}
            for name, count in expected.items():
                assert row["optimizer_steps"][name]["min"] == row["optimizer_steps"][name]["max"] == count
            row["ema_updates"] = trainer.policy_trainer.ema.num_updates
            assert row["ema_updates"] == args.updates
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    finally:
        torch.randperm = original_permutation
        if "_sample_from_selection" in vars(trainer.buffer):
            del trainer.buffer._sample_from_selection
        _release_learners(trainer)
        trainer.writer.close()
        trainer.env.close()


if __name__ == "__main__":
    main()

"""Compare execution modes on the same varying sequence from one real replay.

Each WM minibatch, Actor start batch and Critic epoch receives its own prepared
sample. The sequence is identical across modes. Device counters are advanced
by actual execution, including CUDA Graph replay, never by Python call counts.
This is a numerical diagnostic with fixed normalization statistics and replay,
not a convergence/quality test or a training-throughput benchmark.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from benchmark_learner_updates import (
    _cpu_state,
    _precision_settings,
    _prefill,
    _release_learners,
    _source_hash,
    _state_hash,
    _steps,
)
from benchmark_reference_fk_imagination import restore_config
from verify_learner_execution import error, outputs, probes

from supermimic.config import TrainConfig
from supermimic.imagination.policy_trainer import PolicyTrainer
from supermimic.training.loop import Trainer
from supermimic.world_model.wm_trainer import WorldModelTrainer

MODES = {
    "unaccelerated": (False, False, False, "multinomial"),
    "batch_encoder_only": (True, False, False, "multinomial"),
    "capturable_no_graph": (True, True, False, "multinomial"),
    "d98f4df": (True, True, False, "multinomial"),
    "cdf_fusion": (True, True, True, "cdf"),
}


class FixedReplaySequence:
    """Fixed-storage tables; their row selection remains dynamic inside graphs."""

    def __init__(self, windows, starts, permutations, original_randperm):
        for pair in (windows, starts):
            if len(pair) != 2 or pair[0].ndim != 2 or pair[0].shape != pair[1].shape:
                raise ValueError("Replay sequence tables must be matched [draw,batch] matrices")
        tensors = (*windows, *starts, permutations)
        if any(value.dtype != torch.long or value.device != windows[0].device
               or value.ndim != 2 or min(value.shape) < 1 for value in tensors):
            raise ValueError("Sequence tables must be nonempty int64 matrices on one device")
        self.windows, self.starts, self.permutations = windows, starts, permutations
        self.original_randperm = original_randperm
        self.device = windows[0].device
        self.window_index = torch.zeros(1, dtype=torch.long, device=self.device)
        self.start_index = torch.zeros_like(self.window_index)
        self.permutation_index = torch.zeros_like(self.window_index)

    def reset(self):
        for counter in (self.window_index, self.start_index, self.permutation_index):
            counter.zero_()

    @staticmethod
    def _take(table, counter):
        # Clamp only prevents a diagnostic mistake from causing a device assert.
        # Exact counters are checked after every round, so excess draws fail on
        # the host rather than silently reusing the final batch.
        index = counter.clamp(max=table.shape[0] - 1)
        return table.index_select(0, index).squeeze(0)

    def draw(self, selection, n):
        if selection.kind == "windows":
            tables, counter = self.windows, self.window_index
        elif selection.kind == "starts":
            tables, counter = self.starts, self.start_index
        else:
            raise ValueError(f"Unknown replay selection {selection.kind!r}")
        if n != tables[0].shape[1]:
            raise ValueError("Controlled training requested an unprepared replay batch size")
        env, slot = (self._take(table, counter) for table in tables)
        counter.add_(1)
        return env, slot

    def randperm(self, n, *, device=None, **kwargs):
        requested = torch.device("cpu" if device is None else device)
        if n != self.permutations.shape[1] or requested != self.device:
            # Only intercept the actual Critic shuffle, not compiler internals.
            return self.original_randperm(n, device=device, **kwargs)
        if kwargs:
            raise ValueError("Unexpected keyword arguments in controlled Critic randperm")
        result = self._take(self.permutations, self.permutation_index)
        self.permutation_index.add_(1)
        return result

    def counts(self):
        return {"wm_batches": int(self.window_index.item()),
                "actor_batches": int(self.start_index.item()),
                "critic_permutations": int(self.permutation_index.item())}


    def assert_counts(self, expected):
        actual = self.counts()
        if actual != expected:
            raise AssertionError(f"Controlled sequence counts differ: {actual} != {expected}")
        return actual


def _self_test_cpu():
    windows = (torch.arange(18).reshape(6, 3), torch.arange(18, 36).reshape(6, 3))
    starts = (torch.arange(4).reshape(2, 2), torch.arange(4, 8).reshape(2, 2))
    permutations = torch.tensor([[3, 1, 0, 2], [2, 0, 3, 1], [1, 3, 2, 0], [0, 2, 1, 3]])
    sequence = FixedReplaySequence(windows, starts, permutations, torch.randperm)
    for update in range(2):
        for batch in range(3):
            actual = sequence.draw(SimpleNamespace(kind="windows"), 3)
            for got, table in zip(actual, windows, strict=True):
                torch.testing.assert_close(got, table[update * 3 + batch], rtol=0, atol=0)
        for got, table in zip(sequence.draw(SimpleNamespace(kind="starts"), 2), starts, strict=True):
            torch.testing.assert_close(got, table[update], rtol=0, atol=0)
        for epoch in range(2):
            torch.testing.assert_close(sequence.randperm(4, device="cpu"), permutations[update * 2 + epoch],
                                       rtol=0, atol=0)
        sequence.assert_counts({"wm_batches": (update + 1) * 3, "actor_batches": update + 1,
                                "critic_permutations": (update + 1) * 2})
    # Extra reads clamp safely rather than causing a CUDA device assertion;
    # the host count check must then reject the invalid execution budget.
    sequence.draw(SimpleNamespace(kind="windows"), 3)
    try:
        sequence.assert_counts({"wm_batches": 6, "actor_batches": 2, "critic_permutations": 4})
    except AssertionError:
        pass
    else:
        raise AssertionError("An exhausted sequence was not rejected")
    sequence.reset()
    sequence.assert_counts({"wm_batches": 0, "actor_batches": 0, "critic_permutations": 0})
    torch.testing.assert_close(sequence.draw(SimpleNamespace(kind="windows"), 3)[0], windows[0][0],
                               rtol=0, atol=0)
    previous = sequence.counts()
    try:
        sequence.draw(SimpleNamespace(kind="windows"), 7)
    except ValueError:
        pass
    else:
        raise AssertionError("A mismatched batch width was not rejected")
    assert sequence.counts() == previous
    print("CPU sequence self-test passed: varying rows/permutations, reset, bounds, exact counters")


def _prepare_sequence(buffer, cfg, clip_last, updates, actor_ratio, seed, original_randperm):
    windows = buffer.prepare_windows(cfg.wm_trainer.window)
    starts = buffer.prepare_starts(cfg.policy_trainer.history, horizon=cfg.policy_trainer.horizon,
                                   clip_last=clip_last)
    generator = torch.Generator(device=buffer.device).manual_seed(seed)
    tables = {}
    for name, selection, rows, batch in (
        ("windows", windows, updates * cfg.wm_trainer.batches_per_update, cfg.wm_trainer.batch_size),
        ("starts", starts, updates * actor_ratio, cfg.policy_trainer.batch_size),
    ):
        index = torch.multinomial(selection.weights, rows * batch, replacement=True,
                                  generator=generator).reshape(rows, batch)
        tables[name] = (index // buffer.T, index % buffer.T)
    population = cfg.policy_trainer.horizon * cfg.policy_trainer.batch_size
    epochs = cfg.policy_trainer.critic_epochs if cfg.policy_trainer.use_critic else 0
    count = updates * actor_ratio * epochs
    if count:
        permutations = torch.stack([original_randperm(population, device=buffer.device, generator=generator)
                                    for _ in range(count)])
    else:
        permutations = torch.arange(population, device=buffer.device)[None]
    return FixedReplaySequence(tables["windows"], tables["starts"], permutations, original_randperm)


def _wm_snapshot(wm):
    return [(parameter.detach().clone(),
             None if parameter.grad is None else parameter.grad.detach().clone(),
             None if parameter.grad is None else parameter.grad.data_ptr())
            for parameter in wm.parameters()]


def _assert_wm_frozen(wm, saved):
    for parameter, (value, gradient, address) in zip(wm.parameters(), saved, strict=True):
        assert torch.equal(parameter, value), "Actor modified WM parameters"
        if gradient is None:
            assert parameter.grad is None
        else:
            assert parameter.grad.data_ptr() == address, "Actor rebound WM gradient storage"
            assert torch.equal(parameter.grad, gradient), "Actor modified WM gradients"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--self-test-cpu", action="store_true")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--prefill-steps", type=int, default=96)
    parser.add_argument("--sequence-seed", type=int, default=831)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.self_test_cpu:
        _self_test_cpu()
        return
    if args.config is None:
        parser.error("--config is required unless --self-test-cpu is selected")
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda" or args.updates < 5:
        parser.error("CUDA and at least five rounds (warmup/capture/replay) are required")
    if args.num_envs < 1 or args.prefill_steps < 1:
        parser.error("num-envs and prefill-steps must be positive")
    torch.cuda.set_device(torch.device(args.device))
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
    ratio = cfg.schedule.policy_updates_per_wm_update
    if not isinstance(ratio, int) or ratio < 1:
        parser.error("policy_updates_per_wm_update must be a positive integer")
    trainer = Trainer(cfg)
    original_permutation = torch.randperm
    references = {}
    report = {"purpose": "same varying samples/permutations; numerical diagnostic, not quality/speed",
              "gpu": torch.cuda.get_device_name(torch.device(args.device)),
              "python_source_sha256": _source_hash(), "rounds_per_mode": args.updates,
              "diagnostic_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "same_real_replay": True, "same_sample_sequence": True,
              "same_random_critic_permutation_sequence": True,
              "imagination_obs_noise": False, "normalizer_updates_during_comparison": False,
              "activation_checkpoint": cfg.policy_trainer.activation_checkpoint,
              "wm_batch": cfg.wm_trainer.batch_size, "horizon": cfg.policy_trainer.horizon,
              "policy_batch": cfg.policy_trainer.batch_size, "actor_updates_per_round": ratio,
              "modes": []}
    try:
        report["prefill"] = _prefill(trainer, args.prefill_steps)
        buffer = trainer.buffer
        snapshot = {name: _cpu_state(getattr(trainer, name)) for name in ("wm", "policy", "critic")}
        report["initial_model_sha256"] = _state_hash(snapshot)
        # Probe before installing the training hooks: this must not consume an
        # entry from any controlled training sequence.
        start = buffer.sample_starts(32, cfg.policy_trainer.history, horizon=cfg.policy_trainer.horizon,
                                     clip_last=trainer.lib.clip_last)
        data = probes(trainer, start)
        with torch.no_grad():
            action = trainer.policy(data["obs"], data["goal"]).detach()
        sequence = _prepare_sequence(buffer, cfg, trainer.lib.clip_last, args.updates, ratio,
                                     args.sequence_seed, original_permutation)
        report["sample_sequence_sha256"] = _state_hash({"tables": {
            "wm_env": sequence.windows[0].cpu(), "wm_slot": sequence.windows[1].cpu(),
            "actor_env": sequence.starts[0].cpu(), "actor_slot": sequence.starts[1].cpu(),
            "permutations": sequence.permutations.cpu(),
        }})
        buffer._sample_from_selection = sequence.draw
        torch.randperm = sequence.randperm
        initial_values = None
        for mode, (batch_encoder, capture, fusion, sampler) in MODES.items():
            _release_learners(trainer)
            for name, state in snapshot.items():
                getattr(trainer, name).load_state_dict(state)
            sequence.reset()
            wc, pc = copy.deepcopy(cfg.wm_trainer), copy.deepcopy(cfg.policy_trainer)
            wc.cuda_graph, wc.compile_full_window = capture, fusion
            pc.cuda_graph, pc.compile_networks, pc.batch_reference_encoder = capture, fusion, batch_encoder
            pc.obs_noise = False
            buffer.sampling = sampler
            trainer.wm_trainer = WorldModelTrainer(trainer.wm, trainer.lib, wc)
            trainer.policy_trainer = PolicyTrainer(trainer.wm, trainer.policy, trainer.critic, trainer.lib, pc)
            if mode == "capturable_no_graph":
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
            references[mode] = []
            for update in range(args.updates):
                metrics = trainer.wm_trainer.update(buffer)
                before_actor = _wm_snapshot(trainer.wm)
                for _ in range(ratio):
                    metrics.update(trainer.policy_trainer.update(buffer))
                _assert_wm_frozen(trainer.wm, before_actor)
                del before_actor
                values = outputs(trainer, data, action)
                references[mode].append(values)
                differences = {name: error(value, references["unaccelerated"][update][name])
                               for name, value in values.items()}
                close = {name: bool(torch.isclose(value, references["unaccelerated"][update][name],
                                                  rtol=5e-4, atol=5e-6).all())
                         for name, value in values.items()}
                expected_counts = {"wm_batches": (update + 1) * wc.batches_per_update,
                                   "actor_batches": (update + 1) * ratio,
                                   "critic_permutations": ((update + 1) * ratio * pc.critic_epochs
                                                           if pc.use_critic else 0)}
                counts = sequence.assert_counts(expected_counts)
                round_report = {"update": update + 1, "functional_errors": differences,
                                "within_elementwise_tolerance": close, "sample_counters": counts,
                                "wm_frozen_through_actor": True,
                                "wm_graph_phase": trainer.wm_trainer.update_graph.phase,
                                "actor_graph_phase": trainer.policy_trainer._captured_update.phase}
                for reference in ("capturable_no_graph", "d98f4df"):
                    if reference in references and reference != mode:
                        round_report["difference_from_" + reference] = {
                            name: error(value, references[reference][update][name]) for name, value in values.items()}
                row["rounds"].append(round_report)
                for name, value in values.items():
                    assert torch.isfinite(value).all(), (mode, update, name)
                assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({"mode": mode, **round_report}), flush=True)
            row["optimizer_steps"] = {"wm": _steps(trainer.wm_trainer.opt),
                                      "actor": _steps(trainer.policy_trainer.opt),
                                      "critic": _steps(trainer.policy_trainer.critic_opt)}
            expected_steps = {"wm": args.updates * wc.batches_per_update, "actor": args.updates * ratio,
                              "critic": (args.updates * ratio * pc.critic_epochs * pc.critic_minibatches
                                         if pc.use_critic else 0)}
            for name, count in expected_steps.items():
                assert row["optimizer_steps"][name]["min"] == row["optimizer_steps"][name]["max"] == count
            row["ema_updates"] = trainer.policy_trainer.ema.num_updates
            assert row["ema_updates"] == args.updates * ratio
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        torch.randperm = original_permutation
        if "_sample_from_selection" in vars(trainer.buffer):
            del trainer.buffer._sample_from_selection
        _release_learners(trainer)
        trainer.writer.close()
        trainer.env.close()


if __name__ == "__main__":
    main()

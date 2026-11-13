"""Host dispatch changes invalidate recordings before any old branch replays."""

from types import SimpleNamespace

import pytest
import torch

from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.training.cuda_graph import CapturedUpdate, replay_sampling_key

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Graphs")


def test_cpu_configuration_changes_keep_one_counted_update_per_call():
    config = {"mode": "cdf"}
    count = torch.zeros(())

    def step():
        count.add_(1 if config["mode"] == "cdf" else 10)
        return {"count": count.clone()}

    update = CapturedUpdate(step, lambda: [count], configuration=lambda: (config["mode"], 7))
    assert update()["count"].item() == 1
    assert update()["count"].item() == 2
    assert update.configuration_invalidations == 0
    config["mode"] = "multinomial"
    update.check_configuration()  # Table preparation may now safely follow.
    assert update.calls == 2 and count.item() == 2
    assert update()["count"].item() == 12
    assert update.configuration_invalidations == 1 and update.calls == 3
    config["mode"] = "cdf"
    assert update()["count"].item() == 13
    assert update.configuration_invalidations == 2 and update.calls == 4
    update.invalidate()  # checkpoint restoration forgets the old dispatch key.
    assert update()["count"].item() == 14
    assert update.configuration_invalidations == 2 and update.calls == 5
    assert not update.captured and update.phase == "eager"


def test_configuration_change_drops_old_recording_before_eager_fallback():
    config = ["cdf"]
    state = torch.zeros(())
    update = CapturedUpdate(lambda: {"count": state}, lambda: [state],
                            configuration=lambda: config[0])
    update()
    update._graph = object()  # Exercise invalidation without a CUDA runtime.
    update._outputs = {"stale": state}
    update._identity = ("old",)
    update._remaining = 0
    config[0] = "multinomial"
    assert update() == {"count": state}
    assert update._graph is None and not update._outputs and not update._identity
    assert update._remaining == update.warmup and update.configuration_invalidations == 1


@pytest.mark.parametrize("key", [torch.tensor(1), ("cdf", torch.tensor(1)), [], {}, object()])
def test_configuration_rejects_tensors_and_mutable_keys_before_executing(key):
    state = torch.zeros(())

    def step():
        state.add_(1)
        return {"count": state}

    update = CapturedUpdate(step, lambda: [state], configuration=lambda: key)
    with pytest.raises(TypeError, match="immutable CPU"):
        update()
    assert update.calls == 0 and state.item() == 0


def test_configuration_rejects_nan_and_invalid_replay_modes():
    update = CapturedUpdate(dict, list, configuration=lambda: (float("nan"),))
    with pytest.raises(ValueError, match="NaN"):
        update()
    assert replay_sampling_key(None) is None
    legacy = SimpleNamespace()
    assert replay_sampling_key(legacy) == (id(legacy), "multinomial")
    for mode in ("invalid", None, 7, torch.tensor(0)):
        with pytest.raises(ValueError, match="Replay sampling"):
            replay_sampling_key(SimpleNamespace(sampling=mode))


def test_wm_rejects_invalid_mode_before_prepare_or_optimizer_step(monkeypatch):
    from test_wm_update_capture import _trainer

    trainer, buffer = _trainer("cpu", capture=True)
    buffer.sampling = "invalid"

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid mode must fail before GPU selection preparation")

    monkeypatch.setattr(buffer, "prepare_windows", forbidden)
    with pytest.raises(ValueError, match="Replay sampling"):
        trainer.update(buffer)
    assert trainer.update_graph.calls == 0 and not trainer.opt.state


def test_actor_rejects_invalid_mode_before_prepare_or_optimizer_step(tmp_path, monkeypatch):
    from test_policy_update_capture import _components, _trainer

    components = _components(tmp_path, "cpu")
    trainer, buffer = _trainer(components, capture=True), components[-1]
    buffer.sampling = "invalid"

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid mode must fail before GPU selection preparation")

    monkeypatch.setattr(buffer, "prepare_starts", forbidden)
    with pytest.raises(ValueError, match="Replay sampling"):
        trainer.update(buffer)
    assert trainer._captured_update.calls == 0 and not trainer.opt.state and not trainer.critic_opt.state


@CUDA
def test_cuda_switch_rebuilds_before_old_cdf_can_sample_changed_legal_rows():
    buffer = ReplayBuffer(2, 32, 1, 1, 1, "cuda:0", sampling="cdf")
    buffer.s[..., 0].copy_(torch.arange(64, device="cuda:0").reshape(2, 32))
    buffer.ep_id.zero_()
    counter = torch.zeros((), device="cuda:0")
    selection = None

    def step():
        sampled = buffer.sample_windows(128, 1, prepared=selection)
        counter.add_(1)
        return {"rows": sampled.states[0, :, 0], "counter": counter}

    update = CapturedUpdate(
        step, lambda: [buffer.s, selection.weights, selection.count, selection.cdf, counter],
        warmup=1, configuration=lambda: replay_sampling_key(buffer),
    )
    try:
        patterns = (
            ("cdf", 0, ("warmup", "capture", "replay"), 0),
            # New replay contents refresh the same CDF storage, not the graph.
            ("cdf", 16, ("replay", "replay", "replay"), 0),
            ("multinomial", 8, ("warmup", "capture", "replay"), 1),
            ("cdf", 20, ("warmup", "capture", "replay"), 2),
        )
        for mode, first, phases, invalidations in patterns:
            buffer.sampling = mode
            update.check_configuration()
            buffer.valid.zero_()
            buffer.valid[:, first:first + 6] = True
            buffer.ptr.fill_(first + 6)
            selection = buffer.prepare_windows(1)
            legal = buffer.s[..., 0][buffer.window_mask(1)]
            for phase in phases:
                result = update()
                assert update.phase == phase
                assert bool(torch.isin(result["rows"], legal).all())
                assert result["counter"].item() == update.calls
            assert update.configuration_invalidations == invalidations
        assert update.calls == 12 and counter.item() == 12
        assert update.captured
    finally:
        update.invalidate()


@CUDA
def test_wm_and_actor_trainers_recapture_both_sampler_switches_without_extra_steps(tmp_path):
    from test_policy_update_capture import _components
    from test_wm_update_capture import _trainer

    from supermimic.imagination.policy_trainer import PolicyTrainer, PolicyTrainerCfg

    wm_trainer, wm_buffer = _trainer("cuda:0", capture=True, batches=2)
    wm, policy, critic, library, actor_buffer = _components(tmp_path, "cuda:0")
    actor_trainer = PolicyTrainer(wm, policy, critic, library, PolicyTrainerCfg(
        horizon=2, batch_size=2, history=5, future_frames=5, obs_noise=False,
        compile=None, compile_networks=False, observation_layout="isaac_khan",
        critic_epochs=1, critic_minibatches=1, critic_value_bound=True,
        cuda_graph=True, cuda_graph_warmup=1,
    ))
    try:
        for index, mode in enumerate(("cdf", "multinomial", "cdf")):
            wm_buffer.sampling = actor_buffer.sampling = mode
            for phase in ("warmup", "capture", "replay"):
                wm_metrics = wm_trainer.update(wm_buffer)
                pi_metrics = actor_trainer.update(actor_buffer)
                assert all(torch.isfinite(torch.tensor(value)) for value in (*wm_metrics.values(), *pi_metrics.values()))
                assert wm_trainer.update_graph.phase == actor_trainer._captured_update.phase == phase
                assert {int(v["step"]) for v in wm_trainer.opt.state.values()} == {2 * wm_trainer.update_graph.calls}
                assert {int(v["step"]) for v in actor_trainer.opt.state.values()} == {actor_trainer._captured_update.calls}
                assert {int(v["step"]) for v in actor_trainer.critic_opt.state.values()} == {actor_trainer._captured_update.calls}
                assert actor_trainer.ema.num_updates == actor_trainer._captured_update.calls
            assert wm_trainer.update_graph.configuration_invalidations == index
            assert actor_trainer._captured_update.configuration_invalidations == index
        wm_buffer.sampling = actor_buffer.sampling = "invalid"
        for trainer, buffer in ((wm_trainer, wm_buffer), (actor_trainer, actor_buffer)):
            with pytest.raises(ValueError, match="Replay sampling"):
                trainer.update(buffer)
        assert wm_trainer.update_graph.calls == actor_trainer._captured_update.calls == 9
    finally:
        wm_trainer.reset_capture()
        actor_trainer.reset_capture()

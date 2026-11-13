"""Circular Python garbage must not finalize CUDA resources during recording."""

import gc

import pytest
import torch

from supermimic.training.cuda_graph import (
    CapturedUpdate,
    _recording_gc_guard,
    graph_state,
    make_capturable_adam,
)


@pytest.mark.parametrize("initially_enabled", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_recording_gc_scope_restores_the_callers_state_even_on_failure(initially_enabled, fail, monkeypatch):
    was_enabled = gc.isenabled()
    collections = []
    monkeypatch.setattr(gc, "collect", lambda: collections.append(gc.isenabled()))
    gc.enable() if initially_enabled else gc.disable()
    try:
        def record():
            with _recording_gc_guard():
                assert not gc.isenabled()
                # Even a misbehaving body must not leak its GC-state change
                # to a caller that entered with garbage collection disabled.
                gc.enable()
                if fail:
                    raise RuntimeError("synthetic recording failure")
        if fail:
            with pytest.raises(RuntimeError, match="synthetic"):
                record()
        else:
            record()
        assert gc.isenabled() is initially_enabled
        assert collections == [initially_enabled]
    finally:
        gc.enable() if was_enabled else gc.disable()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Graphs")
@pytest.mark.parametrize("initially_enabled", [False, True])
def test_only_cold_capture_disables_gc_and_updates_are_still_counted(initially_enabled):
    was_enabled = gc.isenabled()
    gc.enable() if initially_enabled else gc.disable()
    module = torch.nn.Linear(3, 1, device="cuda:0")
    optimizer = make_capturable_adam(module.parameters(), lr=2e-4)
    inputs = torch.ones(2, 3, device="cuda:0")
    calls = []

    def step():
        calls.append((torch.cuda.is_current_stream_capturing(), gc.isenabled()))
        optimizer.zero_grad(set_to_none=False)
        loss = module(inputs).square().mean()
        loss.backward()
        optimizer.step()
        return {"loss": loss.detach()}

    update = CapturedUpdate(step, lambda: graph_state((module,), (optimizer,), (inputs,)), warmup=1)
    try:
        for _ in range(3):
            update()
            assert gc.isenabled() is initially_enabled
        assert calls == [(False, initially_enabled), (True, False)]
        assert update.phase == "replay"
        assert {int(state["step"]) for state in optimizer.state.values()} == {3}
    finally:
        update.invalidate()
        gc.enable() if was_enabled else gc.disable()

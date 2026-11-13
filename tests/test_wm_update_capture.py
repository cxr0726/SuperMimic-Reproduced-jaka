"""Whole learner capture preserves update counts, Adam state and live replay."""

import copy
import math
from types import SimpleNamespace

import pytest
import torch

from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.training.cuda_graph import (
    CapturedUpdate,
    graph_state,
    make_capturable_adam,
    prepare_adam_for_capture,
)
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.wm_trainer import WMTrainerCfg, WorldModelTrainer
from supermimic.world_model.world_model import WorldModel, WorldModelCfg

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Graphs")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_graph_safe_quaternion_constants_preserve_values_and_gradients(dtype):
    from supermimic.kinematics.rotations import quat_apply_inverse
    from supermimic.world_model.state_spec import quat_apply_inverse_quat

    torch.manual_seed(19)
    q = torch.randn(7, 4, dtype=dtype)
    q = (q / q.norm(dim=-1, keepdim=True)).requires_grad_()
    ref_q = torch.randn(7, 4, dtype=dtype)
    ref_q /= ref_q.norm(dim=-1, keepdim=True)
    old_conjugate = q * torch.tensor([1., -1., -1., -1.], dtype=dtype)
    new_conjugate = quat_apply_inverse_quat(q)
    torch.testing.assert_close(old_conjugate, new_conjugate, rtol=0, atol=0)
    reference = SimpleNamespace(root_quat_w=ref_q, root_pos_w=torch.randn(7, 3, dtype=dtype))
    p = torch.randn(7, 3, dtype=dtype)
    g = torch.tensor([0., 0., -1.], dtype=dtype).expand(7, 3)
    old_features = torch.stack((reference.root_pos_w[:, 2] - p[:, 2],
                                quat_apply_inverse(ref_q, g)[:, 2] - quat_apply_inverse(q, g)[:, 2]), -1)
    new_features = AnalyticHeads.termination_features(p, q, reference)
    torch.testing.assert_close(new_features, old_features, rtol=0, atol=0)
    old_grad, = torch.autograd.grad(old_features.sum() + old_conjugate.sum(), q, retain_graph=True)
    new_grad, = torch.autograd.grad(new_features.sum() + new_conjugate.sum(), q)
    torch.testing.assert_close(new_grad, old_grad, rtol=0, atol=0)


def test_cpu_fallback_is_one_update_per_call_and_warmup_is_not_extra(capsys):
    weight = torch.tensor(0.)

    def step():
        weight.add_(1)
        return {"count": weight.clone()}

    update = CapturedUpdate(step, lambda: [weight], warmup=3, label="unit")
    for i in range(5):
        assert update()["count"].item() == i + 1
    assert update.calls == 5 and not update.captured and not update.is_replay
    assert update.phase == "eager"
    assert capsys.readouterr().out.count("eager fallback") == 1


@pytest.mark.parametrize("warmup", [0, -1, True, 1.5])
def test_capture_rejects_invalid_warmup(warmup):
    with pytest.raises(ValueError, match="warmup"):
        CapturedUpdate(dict, list, warmup=warmup)


class _Reference:
    def __init__(self, device):
        self.root_pos = torch.tensor([[0., 0., 1.]], device=device)
        self.root_quat = torch.tensor([[1., 0., 0., 0.]], device=device)

    def ref(self, frame):
        return SimpleNamespace(
            root_pos_w=self.root_pos.expand(frame.numel(), -1),
            root_quat_w=self.root_quat.expand(frame.numel(), -1),
        )


def _fill(buffer, position=0.):
    device = buffer.device
    state = torch.zeros(buffer.E, 15, device=device)
    state[:, 2], state[:, 3], state[:, 7] = 1., 1., position
    xi = torch.zeros(buffer.E, 5, device=device)
    xi[:, 0] = .75
    for i in range(buffer.T):
        buffer.add(
            torch.arange(buffer.E, device=device), s=state,
            a=torch.full((buffer.E, 1), .1, device=device),
            a_exec=torch.full((buffer.E, 1), .1, device=device),
            xi=xi, r=torch.zeros(buffer.E, device=device),
            done=torch.zeros(buffer.E, device=device),
            frame=torch.zeros(buffer.E, dtype=torch.long, device=device),
            clip=torch.zeros(buffer.E, dtype=torch.long, device=device),
            terminal=torch.zeros(buffer.E, dtype=torch.bool, device=device),
            ep_start=torch.full((buffer.E,), i == 0, dtype=torch.bool, device=device),
        )


def _trainer(device, *, capture, batches=2):
    heads = SimpleNamespace(
        num_joints=1, default_joint_pos=torch.zeros(1, device=device),
        termination_features=AnalyticHeads.termination_features,
    )
    wm = WorldModel(
        WorldModelCfg(hidden=(8,), input_normalization=True,
                      hidden_layer_norm=True, hidden_activation="gelu", termination_mode="analytic"),
        heads, device, action_scale=torch.full((1,), .5, device=device),
    )
    # Nonzero prediction error even though all replay rows are equal. This
    # makes random-index sequences irrelevant to the optimizer parity oracle.
    with torch.no_grad():
        wm.dynamics.mlp.net[-1].bias.fill_(.02)
    trainer = WorldModelTrainer(wm, _Reference(device), WMTrainerCfg(
        window=2, batch_size=4, batches_per_update=batches, compile=None,
        cuda_graph=capture, cuda_graph_warmup=1,
    ))
    buffer = ReplayBuffer(2, 16, 15, 1, 5, device)
    _fill(buffer)
    return trainer, buffer


def test_wm_cpu_capture_request_falls_back_without_changing_counts():
    trainer, buffer = _trainer("cpu", capture=True)
    metrics = trainer.update(buffer)
    assert trainer.update_graph.calls == 1 and not trainer.update_graph.captured
    assert all(math.isfinite(value) for value in metrics.values())
    assert {int(values["step"]) for values in trainer.opt.state.values()} == {2}


@CUDA
def test_complete_wm_update_matches_eager_and_uses_fresh_replay():
    device = torch.device("cuda:0")
    torch.manual_seed(51)
    captured, captured_buffer = _trainer(device, capture=True)
    eager, eager_buffer = _trainer(device, capture=False)
    eager.wm.load_state_dict(copy.deepcopy(captured.wm.state_dict()))
    try:
        for lr in (2e-4, 1e-4, 5e-5):
            captured.set_lr(lr)
            eager.set_lr(lr)
            got, expected = captured.update(captured_buffer), eager.update(eager_buffer)
            assert got.keys() == expected.keys()
            for name in got:
                assert got[name] == pytest.approx(expected[name], rel=2e-5, abs=2e-7), name
            for actual, reference in zip(captured.wm.parameters(), eager.wm.parameters(), strict=True):
                torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-7)
        assert captured.update_graph.captured and captured.update_graph.is_replay
        assert captured.update_graph.calls == 3
        assert {int(values["step"]) for values in captured.opt.state.values()} == {6}
        lr_pointer = captured.opt.param_groups[0]["lr"].data_ptr()
        _fill(captured_buffer, .1)
        _fill(eager_buffer, .1)
        captured.set_lr(3e-5)
        eager.set_lr(3e-5)
        got, expected = captured.update(captured_buffer), eager.update(eager_buffer)
        assert captured.opt.param_groups[0]["lr"].data_ptr() == lr_pointer
        assert got["loss"] == pytest.approx(expected["loss"], rel=2e-5, abs=2e-7)
        assert {int(values["step"]) for values in captured.opt.state.values()} == {8}
        assert captured.update_graph.calls == 4
    finally:
        captured.reset_capture()


@CUDA
def test_graph_learning_rate_is_live_and_storage_replacement_is_rejected():
    model = torch.nn.Linear(2, 1, device="cuda:0")
    opt = make_capturable_adam(model.parameters(), lr=1e-3)
    x = torch.ones(4, 2, device="cuda:0")
    count = torch.zeros((), device="cuda:0")

    def step():
        opt.zero_grad(set_to_none=False)
        loss = model(x).square().mean()
        loss.backward()
        opt.step()
        count.add_(1)
        return {"loss": loss.detach(), "count": count}

    update = CapturedUpdate(step, lambda: graph_state((model,), (opt,), (x, count)), warmup=1)
    for i in range(5):
        out = update()
        assert out["count"].item() == i + 1
    assert {int(values["step"]) for values in opt.state.values()} == {5}
    opt.param_groups[0]["lr"].zero_()
    before = [p.clone() for p in model.parameters()]
    update()
    for actual, expected in zip(model.parameters(), before, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    model.weight.data = model.weight.data.clone()
    with pytest.raises(RuntimeError, match="storage changed"):
        update()


@CUDA
def test_complete_wm_update_can_restore_optimizer_and_record_again():
    device = torch.device("cuda:0")
    torch.manual_seed(37)
    captured, buffer = _trainer(device, capture=True, batches=3)
    restored_eager, reference_buffer = _trainer(device, capture=False, batches=3)
    try:
        captured.update(buffer)  # One counted warm-up block, three Adam steps.
        assert not captured.update_graph.captured
        captured.update(buffer)  # One complete graph replay, three more steps.
        assert captured.update_graph.captured
        saved_model = copy.deepcopy(captured.wm.state_dict())
        saved_optimizer = copy.deepcopy(captured.opt.state_dict())
        assert {int(values["step"]) for values in captured.opt.state.values()} == {6}
        captured.update(buffer)
        assert {int(values["step"]) for values in captured.opt.state.values()} == {9}
        # Replacing checkpoint state must invalidate the old graph and bind
        # the new moment/step tensors before any fresh capture.
        captured.wm.load_state_dict(saved_model)
        captured.opt.load_state_dict(saved_optimizer)
        captured.reset_capture()
        restored_eager.wm.load_state_dict(saved_model)
        restored_eager.opt.load_state_dict(copy.deepcopy(saved_optimizer))
        assert not captured.update_graph.captured
        for expected_phase in ("warmup", "capture"):
            got = captured.update(buffer)
            expected = restored_eager.update(reference_buffer)
            assert captured.update_graph.phase == expected_phase
            assert got["loss"] == pytest.approx(expected["loss"], rel=2e-5, abs=2e-7)
            for actual, reference in zip(captured.wm.parameters(), restored_eager.wm.parameters(), strict=True):
                torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-7)
        assert {int(values["step"]) for values in captured.opt.state.values()} == {12}
    finally:
        captured.reset_capture()


@CUDA
def test_historical_adam_state_can_be_prepared_before_recapture():
    parameter = torch.nn.Parameter(torch.ones(2, device="cuda:0"))
    opt = torch.optim.Adam((parameter,), lr=1e-3)
    parameter.square().sum().backward()
    opt.step()
    assert opt.state[parameter]["step"].device.type == "cpu"
    prepare_adam_for_capture(opt, parameter.device)
    assert opt.state[parameter]["step"].device == parameter.device
    pointer = opt.param_groups[0]["lr"].data_ptr()
    prepare_adam_for_capture(opt, parameter.device)
    assert opt.param_groups[0]["lr"].data_ptr() == pointer

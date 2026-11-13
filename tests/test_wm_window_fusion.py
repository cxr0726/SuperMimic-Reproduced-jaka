"""Whole-window fusion retains the stepwise WM objective and every BPTT path."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from supermimic.data.replay_buffer import ReplayBuffer, WindowBatch
from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame
from supermimic.world_model.wm_trainer import WMTrainerCfg, WorldModelTrainer
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


def _model(dtype, *, termination, normalized, device="cpu"):
    heads = SimpleNamespace(
        num_joints=3, default_joint_pos=torch.tensor([.1, -.2, .3], device=device),
        termination_features=AnalyticHeads.termination_features,
    )
    model = WorldModel(
        WorldModelCfg(
            hidden=(13, 11), term_hidden=(7,), input_normalization=normalized,
            hidden_layer_norm=True, hidden_activation="gelu",
            termination_mode=termination, term_loss_weight=.65,
        ), heads, device, action_scale=torch.tensor([.4, .5, .6], device=device),
    ).to(dtype=dtype)
    with torch.no_grad():
        # Avoid the zero-last initialization's zero action/hidden gradients.
        model.dynamics.mlp.net[-1].weight.normal_(std=.025)
        model.dynamics.mlp.net[-1].bias.normal_(std=.02)
        if normalized:
            for normalizer in (model.state_normalizer, model.action_normalizer):
                normalizer._mean.uniform_(-.2, .2)
                normalizer._var.uniform_(.7, 1.3)
                normalizer._std.copy_(normalizer._var.sqrt())
                normalizer.count.fill_(123)
    return model


def _window(model, dtype, *, transitions=8, batch=5):
    device = next(model.parameters()).device
    states = torch.randn(transitions + 1, batch, model.spec.state_dim, dtype=dtype, device=device) * .2
    states[..., 2] += .8
    quaternion = torch.randn(transitions + 1, batch, 4, dtype=dtype, device=device)
    states[..., 3:7] = quaternion / quaternion.norm(dim=-1, keepdim=True)
    actions = torch.randn(transitions, batch, 3, dtype=dtype, device=device) * .3
    xi = model.spec.xi_offset.to(dtype).expand(batch, -1).clone()
    xi += torch.randn_like(xi) * .005
    done = (torch.arange(transitions * batch, device=device).reshape(transitions, batch) % 3 == 0).to(dtype)
    reference_pos = torch.randn(transitions, batch, 3, dtype=dtype, device=device) * .2
    reference_pos[..., 2] += .9
    reference_quat = torch.randn(transitions, batch, 4, dtype=dtype, device=device)
    reference_quat /= reference_quat.norm(dim=-1, keepdim=True)
    return states, actions, xi, done, reference_pos, reference_quat


def _stepwise(model, inputs):
    states, actions, xi, done, ref_pos, ref_quat = inputs
    refs = [ReferenceFrame(None, None, None, None, ref_pos[t], ref_quat[t])
            for t in range(actions.shape[0])]
    return model.window_loss(states, actions, xi, done, refs)


def _gradient_error_report(actual, expected):
    difference = torch.cat([(left - right).detach().double().flatten()
                            for left, right in zip(actual, expected, strict=True)])
    reference = torch.cat([value.detach().double().flatten() for value in expected])
    return {"max_abs": float(difference.abs().max()),
            "relative_l2": float(difference.norm() / reference.norm().clamp_min(1e-12))}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("termination", ["analytic", "learned"])
@pytest.mark.parametrize("normalized", [False, True])
def test_tensor_window_matches_stepwise_loss_metrics_and_all_gradients(dtype, termination, normalized):
    torch.manual_seed(103)
    stepwise = _model(dtype, termination=termination, normalized=normalized)
    fused = copy.deepcopy(stepwise)
    inputs = _window(stepwise, dtype)
    old_inputs = [value.clone().requires_grad_(index < 3) for index, value in enumerate(inputs)]
    new_inputs = [value.clone().requires_grad_(index < 3) for index, value in enumerate(inputs)]
    initial_keys = set(fused.state_dict())
    initial_buffers = {name: value.clone() for name, value in fused.named_buffers()}
    expected, old_metrics = _stepwise(stepwise, old_inputs)
    actual, new_metrics = fused.window_loss_full(*new_inputs)
    tolerance = {"rtol": 3e-5, "atol": 3e-6} if dtype == torch.float32 else {"rtol": 1e-11, "atol": 1e-12}
    torch.testing.assert_close(actual, expected, **tolerance)
    assert new_metrics.keys() == old_metrics.keys()
    for name in old_metrics:
        # Accuracy is deliberately FP32 even for the FP64 numerical oracle.
        metric_tolerance = {"rtol": 1e-6, "atol": 1e-7} if name == "term_acc" else tolerance
        torch.testing.assert_close(new_metrics[name], old_metrics[name], **metric_tolerance)
        assert not new_metrics[name].requires_grad
    actual.backward()
    expected.backward()
    for old, new in zip(old_inputs[:3], new_inputs[:3], strict=True):
        assert new.grad is not None and torch.isfinite(new.grad).all()
        torch.testing.assert_close(new.grad, old.grad, **tolerance)
    for (old_name, old), (new_name, new) in zip(stepwise.named_parameters(), fused.named_parameters(), strict=True):
        assert old_name == new_name
        assert new.grad is not None and torch.isfinite(new.grad).all(), new_name
        torch.testing.assert_close(new.grad, old.grad, **tolerance, msg=new_name)
    assert set(fused.state_dict()) == initial_keys
    for name, value in fused.named_buffers():
        torch.testing.assert_close(value, initial_buffers[name], rtol=0, atol=0)


class _Reference:
    def __init__(self, positions, quaternions):
        self.positions, self.quaternions = positions.flatten(0, 1), quaternions.flatten(0, 1)

    def ref(self, frame):
        return SimpleNamespace(root_pos_w=self.positions[frame], root_quat_w=self.quaternions[frame])


@pytest.mark.parametrize("full_window", [False, True])
def test_trainer_selects_full_or_step_compile_and_keeps_reference_alignment(monkeypatch, full_window):
    torch.manual_seed(27)
    model = _model(torch.float32, termination="learned", normalized=True)
    inputs = _window(model, torch.float32)
    states, actions, xi, done, pos, quat = inputs
    reference_model = copy.deepcopy(model)
    expected, expected_metrics = _stepwise(reference_model, inputs)
    compiles = []

    def compile_spy(fn, **options):
        compiles.append((fn.__name__, options))
        return fn

    monkeypatch.setattr(torch, "compile", compile_spy)
    trainer = WorldModelTrainer(model, _Reference(pos, quat), WMTrainerCfg(
        compile="default", compile_full_window=full_window,
    ))
    window = WindowBatch(states, actions, actions, xi, torch.zeros_like(done), done,
                         torch.arange(done.numel()).reshape(done.shape))
    actual, actual_metrics = trainer.loss_on(window)
    torch.testing.assert_close(actual, expected)
    for key in actual_metrics:
        torch.testing.assert_close(actual_metrics[key], expected_metrics[key])
    assert [name for name, _ in compiles] == (
        ["_window_rollout", "_window_losses"] if full_window else ["_loss_step"]
    )
    assert trainer.compile_full_window is full_window


def test_compile_none_stays_eager_even_with_full_window_flag(monkeypatch):
    def forbidden_compile(*args, **kwargs):
        raise AssertionError("compile=None must not invoke torch.compile")

    monkeypatch.setattr(torch, "compile", forbidden_compile)
    model = _model(torch.float32, termination="analytic", normalized=False)
    trainer = WorldModelTrainer(model, None, WMTrainerCfg(compile=None, compile_full_window=True))
    assert not trainer.compile_full_window
    assert model._window_rollout_fn is model._window_losses_fn is None


def test_window_rollout_retains_all_eight_transitions():
    model = _model(torch.float64, termination="analytic", normalized=False)
    states, actions, xi, *_ = _window(model, torch.float64)
    predictions = model._window_rollout(states[0], actions, xi)
    assert predictions.shape == states[1:].shape
    current = states[0]
    for index in range(8):
        current = model.step(current, actions[index], xi)
        torch.testing.assert_close(predictions[index], current, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires compiled CUDA kernels")
def test_compiled_cuda_window_keeps_loss_and_parameter_input_gradients():
    torch.manual_seed(104)
    old_model = _model(torch.float32, termination="analytic", normalized=True, device="cuda:0")
    compiled_model = copy.deepcopy(old_model)
    inputs = _window(old_model, torch.float32, batch=4)
    old_inputs = [value.clone().requires_grad_(index < 3) for index, value in enumerate(inputs)]
    new_inputs = [value.clone().requires_grad_(index < 3) for index, value in enumerate(inputs)]
    trainer = WorldModelTrainer(compiled_model, None, WMTrainerCfg(
        compile="default", compile_full_window=True, cuda_graph=True,
    ))
    expected, expected_metrics = _stepwise(old_model, old_inputs)
    actual, actual_metrics = trainer.wm.window_loss_full(*new_inputs)
    torch.testing.assert_close(actual, expected, rtol=5e-4, atol=3e-5)
    for name in expected_metrics:
        torch.testing.assert_close(actual_metrics[name], expected_metrics[name], rtol=5e-4, atol=3e-5)
    expected.backward()
    actual.backward()
    for old, new in zip(old_inputs[:3], new_inputs[:3], strict=True):
        torch.testing.assert_close(new.grad, old.grad, rtol=8e-4, atol=5e-5)
    for old, new in zip(old_model.parameters(), compiled_model.parameters(), strict=True):
        torch.testing.assert_close(new.grad, old.grad, rtol=8e-4, atol=5e-5)
    print("WM_FUSION_EQUIVALENCE " + json.dumps({
        "transitions": 8, "loss_abs": float((actual - expected).detach().abs()),
        "input_gradients": _gradient_error_report(
            [value.grad for value in new_inputs[:3]], [value.grad for value in old_inputs[:3]]),
        "parameter_gradients": _gradient_error_report(
            [value.grad for value in compiled_model.parameters()],
            [value.grad for value in old_model.parameters()]),
    }, sort_keys=True))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Graphs")
def test_compiled_full_window_can_run_inside_complete_update_capture():
    torch.manual_seed(99)
    model = _model(torch.float32, termination="analytic", normalized=True, device="cuda:0")
    reference = _Reference(torch.tensor([[[0., 0., .8]]], device="cuda:0"),
                           torch.tensor([[[1., 0., 0., 0.]]], device="cuda:0"))
    eager = WorldModelTrainer(copy.deepcopy(model), reference, WMTrainerCfg(
        window=8, batch_size=4, batches_per_update=2, compile=None,
    ))
    captured = WorldModelTrainer(model, reference, WMTrainerCfg(
        window=8, batch_size=4, batches_per_update=2, compile="default",
        compile_full_window=True, cuda_graph=True, cuda_graph_warmup=1,
    ))
    buffer = ReplayBuffer(2, 16, model.spec.state_dim, 3, model.spec.xi_dim, "cuda:0")
    state = torch.zeros(2, model.spec.state_dim, device="cuda:0")
    state[:, 2], state[:, 3], state[:, 7:10] = .8, 1., .1
    ids = torch.arange(2, device="cuda:0")
    for index in range(16):
        buffer.add(ids, state, torch.full((2, 3), .2, device="cuda:0"),
                   torch.full((2, 3), .2, device="cuda:0"),
                   model.spec.xi_offset.expand(2, -1).clone(),
                   torch.zeros(2, device="cuda:0"), torch.zeros(2, device="cuda:0"),
                   torch.zeros(2, dtype=torch.long, device="cuda:0"),
                   torch.zeros(2, dtype=torch.long, device="cuda:0"),
                   torch.zeros(2, dtype=torch.bool, device="cuda:0"),
                   torch.full((2,), index == 0, dtype=torch.bool, device="cuda:0"))
    try:
        phases = []
        for index in range(3):
            expected, actual = eager.update(buffer), captured.update(buffer)
            phases.append(captured.update_graph.phase)
            for name in expected:
                assert actual[name] == pytest.approx(expected[name], rel=8e-4, abs=5e-5), name
            for old, new in zip(eager.wm.parameters(), captured.wm.parameters(), strict=True):
                torch.testing.assert_close(new, old, rtol=8e-4, atol=5e-6)
            assert {int(value["step"]) for value in captured.opt.state.values()} == {2 * (index + 1)}
        assert phases == ["warmup", "capture", "replay"]
    finally:
        captured.reset_capture()

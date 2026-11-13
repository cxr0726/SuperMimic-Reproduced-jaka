"""Official Muon/Adam parameter ownership, updates and capture-state contract.

The oracle reproduces the official optimizer partition using native PyTorch
optimizers. It deliberately does not import the untracked reference checkout.
Muon keeps its native BF16 Newton--Schulz iteration; FP32 parameters do not imply
that Muon has the same arithmetic or learning trajectory as all-Adam training.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from supermimic.config import TrainConfig
from supermimic.policy.nets import Critic, CriticCfg, MotionEncoder, Policy, PolicyCfg
from supermimic.training.cuda_graph import CapturedUpdate, graph_state
from supermimic.training.optimizers import build_training_optimizer
from supermimic.world_model.world_model import WorldModel, WorldModelCfg

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Graphs")


def _policy_config():
    return PolicyCfg(
        num_joints=27, num_tracked_bodies=15, observation_layout="isaac_khan",
        future_frames=5, history=5, goal_dim=155, latent_dim=4,
        encoder_hidden=(12, 8, 6), decoder_hidden=(16, 12, 8, 6),
        hidden_layer_norm=True, hidden_activation="gelu", running_normalization=True,
    )


def _model(role, device="cpu"):
    if role == "wm":
        heads = SimpleNamespace(num_joints=27, default_joint_pos=torch.zeros(27, device=device))
        return WorldModel(
            WorldModelCfg(hidden=(12, 8, 6), hidden_layer_norm=True,
                          hidden_activation="gelu", termination_mode="analytic"),
            heads, device,
        )
    cfg = _policy_config()
    if role == "actor":
        actor = Policy(cfg).to(device)
        # The official rule applies independently to every Sequential, not to
        # the global first/last Linear of a multi-encoder actor.
        actor.encoders["auxiliary"] = MotionEncoder(cfg).to(device)
        return actor
    if role == "critic":
        return Critic(CriticCfg(hidden=(12, 8), hidden_layer_norm=True,
                                hidden_activation="gelu"), pcfg=cfg, xi_dim=101).to(device)
    raise AssertionError(role)


def _official_partition(module):
    # HybridRobotics/supermimic learners/learner_utils.py::build_optimizers:
    # middle Linear weights in EACH Sequential use Muon; everything else Adam.
    hidden = [linear.weight for seq in module.modules() if isinstance(seq, nn.Sequential)
              for linear in [item for item in seq if isinstance(item, nn.Linear)][1:-1]]
    hidden_ids = {id(parameter) for parameter in hidden}
    rest = [parameter for parameter in module.parameters() if id(parameter) not in hidden_ids]
    return hidden, rest


def _group_parameters(optimizer, kind):
    return [parameter for group in optimizer.param_groups if group["optimizer_kind"] == kind
            for parameter in group["params"]]


def _native_reference(module, *, lr, betas, weight_decay=.1):
    hidden, rest = _official_partition(module)
    adam = torch.optim.Adam(rest, lr=torch.tensor(lr, device=rest[0].device),
                            betas=betas, fused=True, capturable=True)
    muon = (torch.optim.Muon(hidden, lr=lr, weight_decay=weight_decay,
                             adjust_lr_fn="match_rms_adamw") if hidden else None)
    return adam, muon


def _gradients(module, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return [torch.randn(parameter.shape, generator=generator).to(parameter)
            for parameter in module.parameters()]


def _set_gradients(module, gradients):
    for parameter, gradient in zip(module.parameters(), gradients, strict=True):
        parameter.grad = gradient.detach().clone()


def _assert_parameters_equal(actual, expected, *, rtol=3e-6, atol=5e-7):
    for (name, parameter), (other_name, reference) in zip(
        actual.named_parameters(), expected.named_parameters(), strict=True,
    ):
        assert name == other_name
        torch.testing.assert_close(parameter, reference, rtol=rtol, atol=atol, msg=name)


def _tree_close(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected.to(actual.device), rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _tree_close(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _tree_close(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize("role", ["wm", "actor", "critic"])
def test_real_model_parameter_groups_match_official_per_mlp_rule(role):
    model = _model(role)
    optimizer = build_training_optimizer(model, kind="muon", lr=2e-4)
    hidden, rest = _official_partition(model)
    actual_muon = _group_parameters(optimizer, "muon")
    actual_adam = _group_parameters(optimizer, "adam")
    assert {id(parameter) for parameter in actual_muon} == {id(parameter) for parameter in hidden}
    assert {id(parameter) for parameter in actual_adam} == {id(parameter) for parameter in rest}
    all_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    assert len(all_ids) == len(set(all_ids)) == len(list(model.parameters()))
    # LN vectors, biases, and each individual encoder/decoder's boundaries are
    # the Adam complement even though some other parameters are 2-D matrices.
    assert all(parameter.ndim == 2 for parameter in actual_muon)
    for seq in (item for item in model.modules() if isinstance(item, nn.Sequential)):
        linears = [item for item in seq if isinstance(item, nn.Linear)]
        for boundary in (linears[0], linears[-1]):
            assert id(boundary.weight) in {id(parameter) for parameter in actual_adam}
        for linear in linears:
            assert id(linear.bias) in {id(parameter) for parameter in actual_adam}


def test_training_defaults_select_muon_but_standalone_trainers_keep_legacy_adam():
    from supermimic.imagination.policy_trainer import PolicyTrainerCfg
    from supermimic.world_model.wm_trainer import WMTrainerCfg

    cfg = TrainConfig()
    assert cfg.wm_trainer.optimizer == "muon"
    assert cfg.policy_trainer.optimizer == cfg.policy_trainer.critic_optimizer == "muon"
    assert cfg.wm_trainer.muon_weight_decay == .1
    assert cfg.policy_trainer.muon_weight_decay == cfg.policy_trainer.critic_muon_weight_decay == .1
    assert WMTrainerCfg().optimizer == "adam"
    assert PolicyTrainerCfg().optimizer == PolicyTrainerCfg().critic_optimizer == "adam"


@pytest.mark.parametrize("role", ["wm", "actor", "critic"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=CUDA)])
def test_fixed_gradient_steps_match_official_native_torch_optimizers(role, device):
    torch.manual_seed(472)
    actual = _model(role, device)
    reference = copy.deepcopy(actual)
    betas = (.9, .95) if role == "actor" else (.9, .999)
    optimizer = build_training_optimizer(actual, kind="muon", lr=2e-4, betas=betas,
                                          capture=device != "cpu")
    adam, muon = _native_reference(reference, lr=2e-4, betas=betas)
    for step in range(5):
        gradients = _gradients(actual, 204 + step)
        _set_gradients(actual, gradients)
        _set_gradients(reference, gradients)
        optimizer.step()
        adam.step()
        if muon is not None:
            muon.step()
        _assert_parameters_equal(actual, reference)
        for actual_parameter, reference_parameter in zip(
            _group_parameters(optimizer, "muon"), _official_partition(reference)[0], strict=True,
        ):
            torch.testing.assert_close(optimizer.state[actual_parameter]["momentum_buffer"],
                                       muon.state[reference_parameter]["momentum_buffer"], rtol=0, atol=0)
    assert {int(value["step"]) for value in optimizer.state.values()} == {5}


def test_muon_weight_decay_only_applies_when_gradient_is_present():
    model = nn.Sequential(nn.Linear(4, 6), nn.GELU(), nn.Linear(6, 5), nn.GELU(), nn.Linear(5, 2))
    optimizer = build_training_optimizer(model, kind="muon", lr=.01, muon_weight_decay=.1)
    original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer.zero_grad(set_to_none=True)
    optimizer.step()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, original[name], rtol=0, atol=0)
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    muon_ids = {id(parameter) for parameter in _group_parameters(optimizer, "muon")}
    for name, parameter in model.named_parameters():
        expected = original[name] * (1 - .01 * .1) if id(parameter) in muon_ids else original[name]
        torch.testing.assert_close(parameter, expected, rtol=1e-6, atol=1e-8)


def test_muon_checkpoint_restore_matches_continuation_and_rejects_legacy_adam_before_mutation():
    torch.manual_seed(19)
    model = _model("actor")
    optimizer = build_training_optimizer(model, kind="muon", lr=2e-4, betas=(.9, .95))
    for step in range(3):
        _set_gradients(model, _gradients(model, 90 + step))
        optimizer.step()
    restored = copy.deepcopy(model)
    other = build_training_optimizer(restored, kind="muon", lr=2e-4, betas=(.9, .95))
    checkpoint = copy.deepcopy(optimizer.state_dict())
    other.validate_state_dict(checkpoint)
    other.load_state_dict(checkpoint)
    _tree_close(other.state_dict(), checkpoint)
    for step in range(3, 6):
        gradients = _gradients(model, 90 + step)
        _set_gradients(model, gradients)
        _set_gradients(restored, gradients)
        optimizer.step()
        other.step()
        _assert_parameters_equal(model, restored, rtol=0, atol=0)
        _tree_close(other.state_dict(), optimizer.state_dict())
    assert {int(value["step"]) for value in other.state.values()} == {6}
    before = copy.deepcopy(other.state_dict())
    old_adam = torch.optim.Adam(restored.parameters(), lr=2e-4)
    for load in (other.validate_state_dict, other.load_state_dict):
        with pytest.raises((ValueError, TypeError), match="(?i)adam|muon|optimizer|schema"):
            load(old_adam.state_dict())
        _tree_close(other.state_dict(), before)


@CUDA
def test_full_update_capture_uses_live_muon_lr_counts_zero_lr_and_restores():
    device = "cuda:0"
    torch.manual_seed(27)
    model = nn.Sequential(nn.Linear(8, 16), nn.LayerNorm(16), nn.GELU(),
                          nn.Linear(16, 12), nn.LayerNorm(12), nn.GELU(), nn.Linear(12, 4)).to(device)
    optimizer = build_training_optimizer(model, kind="muon", lr=1e-3, capture=True)
    inputs, targets = torch.randn(32, 8, device=device), torch.randn(32, 4, device=device)

    def update():
        loss = (model(inputs) - targets).square().mean()
        optimizer.zero_grad(set_to_none=False)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        return {"loss": loss.detach()}

    graph = CapturedUpdate(update, lambda: graph_state((model,), (optimizer,)), warmup=1)
    addresses = [group["lr"].data_ptr() for group in optimizer.param_groups]
    muon_parameter = _group_parameters(optimizer, "muon")[0]
    try:
        for count, lr in enumerate((1e-3, 5e-4, 0., 0., 2e-4), start=1):
            for group in optimizer.param_groups:
                assert isinstance(group["lr"], torch.Tensor) and group["lr"].is_cuda
                group["lr"].fill_(lr)
            parameters = [parameter.detach().clone() for parameter in model.parameters()]
            momentum = (optimizer.state[muon_parameter]["momentum_buffer"].clone()
                        if optimizer.state.get(muon_parameter) else None)
            assert torch.isfinite(graph()["loss"])
            assert addresses == [group["lr"].data_ptr() for group in optimizer.param_groups]
            assert {int(value["step"]) for value in optimizer.state.values()} == {count}
            if lr == 0:
                for parameter, before in zip(model.parameters(), parameters, strict=True):
                    torch.testing.assert_close(parameter, before, rtol=0, atol=0)
                assert not torch.equal(optimizer.state[muon_parameter]["momentum_buffer"], momentum)
            else:
                assert any(not torch.equal(parameter, before) for parameter, before in
                           zip(model.parameters(), parameters, strict=True))
        assert graph.calls == 5 and graph.captured and graph.phase == "replay"
        checkpoint = copy.deepcopy(optimizer.state_dict())
        restored = copy.deepcopy(model)
        other = build_training_optimizer(restored, kind="muon", lr=1e-3, capture=True)
        other.validate_state_dict(checkpoint)
        other.load_state_dict(checkpoint)
        other.prepare_capture()
        assert {int(value["step"]) for value in other.state.values()} == {5}

        def resumed_update():
            loss = (restored(inputs) - targets).square().mean()
            other.zero_grad(set_to_none=False)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(restored.parameters(), 1.)
            other.step()
            return {"loss": loss.detach()}

        resumed_graph = CapturedUpdate(resumed_update, lambda: graph_state((restored,), (other,)), warmup=1)
        try:
            for count in range(6, 9):
                graph()
                resumed_graph()
                _assert_parameters_equal(model, restored, rtol=1e-5, atol=2e-6)
                assert {int(value["step"]) for value in other.state.values()} == {count}
            assert resumed_graph.calls == 3 and resumed_graph.captured
        finally:
            resumed_graph.invalidate()
    finally:
        graph.invalidate()

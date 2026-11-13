"""CPU algebra, grouping, state ownership and checkpoint failure atomicity."""

import copy

import pytest
import torch
from torch import nn

from supermimic.training.optimizers import (
    MuonAdam,
    build_training_optimizer,
    validate_training_optimizer_state,
)


def _model():
    return nn.Sequential(nn.Linear(4, 8), nn.LayerNorm(8), nn.GELU(), nn.Linear(8, 6),
                         nn.GELU(), nn.Linear(6, 2))


def _gradients(model, seed=1):
    generator = torch.Generator().manual_seed(seed)
    for parameter in model.parameters():
        parameter.grad = torch.randn(parameter.shape, generator=generator)


@pytest.mark.parametrize("kind", ["adam", "muon"])
def test_parameter_ownership_survives_initial_and_temporary_freezing(kind):
    model = _model().requires_grad_(False)
    optimizer = build_training_optimizer(model, kind=kind, lr=2e-4)
    owned = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert {id(parameter) for parameter in owned} == {id(parameter) for parameter in model.parameters()}
    optimizer.step()
    assert not optimizer.state
    model.requires_grad_(True)
    _gradients(model)
    optimizer.step()
    assert {int(state["step"]) for state in optimizer.state.values()} == {1}


def test_muon_cpu_steps_match_native_pair_and_count_all_parameters():
    model = _model()
    other = copy.deepcopy(model)
    opt = build_training_optimizer(model, kind="muon", lr=2e-4, betas=(.9, .95))
    hidden = other[3].weight
    muon = torch.optim.Muon([hidden], lr=2e-4, adjust_lr_fn="match_rms_adamw")
    rest = [p for p in other.parameters() if p is not hidden]
    adam = torch.optim.Adam(rest, lr=2e-4, betas=(.9, .95))
    for step in range(3):
        _gradients(model, step)
        _gradients(other, step)
        opt.step()
        muon.step()
        adam.step()
        for actual, expected in zip(model.parameters(), other.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert {int(v["step"]) for v in opt.state.values()} == {step + 1}


def test_muon_grouping_covers_each_parameter_once_with_norms_on_adam():
    model = nn.ModuleDict({"first": _model(), "second": _model()})
    model["first"][0].bias.requires_grad_(False)
    opt = build_training_optimizer(model, kind="muon", lr=2e-4)
    selected = {id(p) for group in opt.muon.param_groups for p in group["params"]}
    assert selected == {id(model[key][3].weight) for key in ("first", "second")}
    parameters = [p for group in opt.param_groups for p in group["params"]]
    assert len(parameters) == len({id(p) for p in parameters})
    assert {id(p) for p in parameters} == {id(p) for p in model.parameters()}
    assert [g["optimizer_kind"] for g in opt.param_groups] == ["adam", "muon"]
    assert all(g["capturable"] is False for g in opt.adam.param_groups)


def test_tensor_lr_muon_formula_matches_native_cpu_with_fp32_rounding_tolerance():
    model = _model()
    other = copy.deepcopy(model)
    opt = build_training_optimizer(model, kind="muon", lr=2e-4)
    native = torch.optim.Muon([other[3].weight], lr=2e-4, adjust_lr_fn="match_rms_adamw")
    opt.muon.param_groups[0]["lr"] = torch.tensor(2e-4)
    for i, lr in enumerate((2e-4, 1e-4, 0, 3e-5)):
        _gradients(model, i)
        _gradients(other, i)
        opt.muon.param_groups[0]["lr"].fill_(lr)
        native.param_groups[0]["lr"] = lr
        opt._muon_step_tensor_lr()
        native.step()
        torch.testing.assert_close(model[3].weight, other[3].weight, rtol=2e-6, atol=3e-8)


def test_state_view_survives_load_and_clear_reaches_both_native_states():
    model = _model()
    opt = build_training_optimizer(model, kind="muon", lr=2e-4)
    assert len(opt.state) == 0
    assert model[3].weight not in opt.state
    assert opt.state.get(model[3].weight) is None
    assert len(opt.state) == 0  # Membership must not create defaultdict entries.
    _gradients(model)
    opt.step()
    view = opt.state
    saved = copy.deepcopy(opt.state_dict())
    view.clear()
    assert not opt.muon.state and not opt.adam.state and len(view) == 0
    opt.load_state_dict(saved)
    assert len(view) == len(list(model.parameters()))
    assert view[model[3].weight] is opt.muon.state[model[3].weight]
    view.clear()
    assert not opt.muon.state and not opt.adam.state


def test_legacy_adam_cannot_load_into_muon_and_reverse_is_rejected():
    model = _model()
    muon = build_training_optimizer(model, kind="muon", lr=2e-4)
    adam = build_training_optimizer(model, kind="adam", lr=2e-4)
    with pytest.raises(ValueError, match="legacy Adam"):
        muon.load_state_dict(adam.state_dict())
    with pytest.raises(ValueError, match="incompatible"):
        validate_training_optimizer_state(adam, muon.state_dict())


def test_corrupt_adam_payload_is_rejected_before_muon_mutates():
    model = _model()
    opt = build_training_optimizer(model, kind="muon", lr=2e-4)
    _gradients(model)
    opt.step()
    momentum = opt.muon.state[model[3].weight]["momentum_buffer"].clone()
    saved = copy.deepcopy(opt.state_dict())
    saved["muon"]["state"][0]["momentum_buffer"].add_(10)
    saved["adam"]["state"][0]["exp_avg"] = torch.zeros(1)
    with pytest.raises(ValueError, match="Invalid adam exp_avg"):
        opt.load_state_dict(saved)
    torch.testing.assert_close(opt.muon.state[model[3].weight]["momentum_buffer"], momentum, rtol=0, atol=0)


def test_state_dict_contract_is_not_a_mutable_alias_of_validation_contract():
    opt = build_training_optimizer(_model(), kind="muon", lr=2e-4)
    saved = opt.state_dict()
    saved["parameter_groups"]["muon"][0]["name"] = "wrong_parameter"
    with pytest.raises(ValueError, match="grouping differs"):
        opt.validate_state_dict(saved)
    opt.validate_state_dict(opt.state_dict())


def test_muon_with_no_middle_matrix_is_explicit_adam_only_wrapper():
    model = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2))
    opt = build_training_optimizer(model, kind="muon", lr=2e-4, capture=True)
    assert isinstance(opt, MuonAdam) and opt.muon is None and not opt.capturable
    _gradients(model)
    opt.step()
    opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    assert len(opt.state) == len(list(model.parameters()))


def test_adam_only_builder_retains_native_format_and_original_algorithm():
    model = _model()
    opt = build_training_optimizer(model, kind="adam", lr=2e-4, capture=True)
    assert isinstance(opt, torch.optim.Adam)
    assert not opt.param_groups[0]["capturable"]
    assert set(opt.state_dict()) == {"state", "param_groups"}
    _gradients(model)
    opt.step()
    validate_training_optimizer_state(opt, opt.state_dict())

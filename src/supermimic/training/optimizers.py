"""Official hidden-matrix Muon grouping with graph-safe, live learning rates.

Muon follows torch.optim.Muon, including its BF16 Newton--Schulz iteration.
Only its scalar-LR application is spelled out for CUDA Graph replay: native
Muon converts a Tensor LR to a host scalar. Adam-only remains the old path.
"""

from __future__ import annotations

import copy
import math
from collections.abc import MutableMapping
from typing import Literal

import torch
from torch import nn

from supermimic.training.cuda_graph import prepare_adam_for_capture


class _CombinedState(MutableMapping):
    """A live view: native load_state_dict may replace either state dictionary."""

    def __init__(self, optimizers):
        self.optimizers = optimizers
        self.owners = {p: opt for opt in optimizers for g in opt.param_groups for p in g["params"]}

    def __getitem__(self, parameter):
        return self.owners[parameter].state[parameter]

    def __setitem__(self, parameter, value):
        self.owners[parameter].state[parameter] = value

    def __delitem__(self, parameter):
        del self.owners[parameter].state[parameter]

    def __iter__(self):
        return (p for opt in self.optimizers for p in opt.state)

    def __len__(self):
        return sum(len(opt.state) for opt in self.optimizers)

    def __contains__(self, parameter):
        owner = self.owners.get(parameter)
        return owner is not None and parameter in owner.state

    def get(self, parameter, default=None):
        owner = self.owners.get(parameter)
        return default if owner is None else owner.state.get(parameter, default)

    def clear(self):
        for opt in self.optimizers:
            opt.state.clear()


def _scalar(value, label):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{label} must be scalar")
        value = value.detach().item()
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite scalar")
    return float(value)


def _validate_native_state(opt, saved, kind):
    """Preflight before either native optimizer is mutated by a composite load."""
    if not isinstance(saved, dict) or set(saved) != {"state", "param_groups"}:
        raise ValueError(f"Invalid {kind} optimizer state schema")
    groups, states = saved["param_groups"], saved["state"]
    if not isinstance(groups, list) or not isinstance(states, dict) or len(groups) != len(opt.param_groups):
        raise ValueError(f"Invalid {kind} optimizer groups/state")
    expected = opt.state_dict()["param_groups"]
    known = set()
    for group, live, schema in zip(groups, opt.param_groups, expected, strict=True):
        if (not isinstance(group, dict) or not isinstance(group.get("params"), list)
                or group["params"] != schema["params"]):
            raise ValueError(f"{kind} optimizer parameter order differs")
        ids = group["params"]
        if any(type(i) is not int or i in known for i in ids):
            raise ValueError(f"{kind} optimizer parameter IDs must be unique integers")
        known.update(ids)
        if _scalar(group.get("lr"), f"{kind} LR") < 0:
            raise ValueError(f"{kind} LR must be nonnegative")
        keys = (("momentum", "nesterov", "ns_coefficients", "ns_steps", "eps", "adjust_lr_fn", "weight_decay")
                if kind == "muon" else ("betas", "eps", "weight_decay", "amsgrad"))
        for key in keys:
            actual = group.get(key)
            reference = live[key]
            if isinstance(reference, tuple) and isinstance(actual, list):
                actual = tuple(actual)
            if actual != reference:
                raise ValueError(f"{kind} optimizer {key} differs from the configured algorithm")
        for identity, parameter in zip(ids, live["params"], strict=True):
            state = states.get(identity, {})
            if not isinstance(state, dict):
                raise ValueError(f"Invalid {kind} parameter state")  # noqa: TRY004 -- corrupt checkpoint payload
            if not state:
                continue
            names = ({"momentum_buffer", "step"} if kind == "muon"
                     else {"exp_avg", "exp_avg_sq", "step"} | ({"max_exp_avg_sq"} if live["amsgrad"] else set()))
            if set(state) != names:
                raise ValueError(f"Invalid {kind} momentum/step state")
            for name, value in state.items():
                if name == "step":
                    if not isinstance(value, torch.Tensor):
                        raise ValueError(f"{kind} step must be a scalar Tensor")
                    count = _scalar(value, f"{kind} step")
                    if count < 0 or not count.is_integer():
                        raise ValueError(f"{kind} step must be a nonnegative integer")
                elif (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                      or value.dtype != parameter.dtype or not torch.isfinite(value).all()):
                    raise ValueError(f"Invalid {kind} {name} tensor")
    if set(states) - known:
        raise ValueError(f"Unknown {kind} optimizer state parameter IDs")


class MuonAdam:
    """Muon on each Sequential's middle Linear weights; Adam on all others."""

    def __init__(self, module: nn.Module, *, lr: float, betas=(0.9, 0.999),
                 adam_weight_decay: float = 0.0, muon_weight_decay: float = 0.1,
                 capture: bool = False):
        try:
            muon_type = torch.optim.Muon
            from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz
        except (AttributeError, ImportError) as error:
            raise RuntimeError(
                "Muon training requires torch.optim.Muon and compatible torch.optim._muon helpers; "
                "this implementation is validated with PyTorch 2.14.1"
            ) from error
        self._adjust_lr, self._zeropower = _adjust_lr, _zeropower_via_newtonschulz
        # Ownership must survive temporary freeze/unfreeze around actor BPTT.
        # Native optimizers skip missing gradients; the official partition also
        # includes every parameter rather than filtering the current flag.
        named = list(module.named_parameters())
        if not named:
            raise ValueError("Training optimizer needs trainable parameters")
        self.device = named[0][1].device
        if any(p.device != self.device for _, p in named):
            raise ValueError("MuonAdam requires parameters on one device")
        if any(not p.is_floating_point() for _, p in named):
            raise ValueError("MuonAdam requires real floating-point parameters")
        hidden = {id(layer.weight) for seq in module.modules() if isinstance(seq, nn.Sequential)
                  for layer in [child for child in seq if isinstance(child, nn.Linear)][1:-1]}
        grouped = {kind: [(name, p) for name, p in named if (id(p) in hidden) == (kind == "muon")]
                   for kind in ("muon", "adam")}
        self._contract = {kind: [{"name": name, "shape": tuple(p.shape), "dtype": str(p.dtype)}
                                 for name, p in values] for kind, values in grouped.items()}
        self.muon = (muon_type([p for _, p in grouped["muon"]], lr=lr,
                                     weight_decay=muon_weight_decay, momentum=0.95, nesterov=True,
                                     ns_coefficients=(3.4445, -4.7750, 2.0315), ns_steps=5, eps=1e-7,
                                     adjust_lr_fn="match_rms_adamw") if grouped["muon"] else None)
        cuda = self.device.type == "cuda"
        self.adam = (torch.optim.Adam([p for _, p in grouped["adam"]],
                                     lr=torch.tensor(float(lr), device=self.device) if cuda else lr,
                                     betas=betas, weight_decay=adam_weight_decay,
                                     fused=cuda, capturable=cuda, foreach=False if cuda else None)
                     if grouped["adam"] else None)
        # Match the official execution order: Adam first, then Muon. The two
        # sets are disjoint, but retaining the order also simplifies audits.
        self._optimizers = tuple(opt for opt in (self.adam, self.muon) if opt is not None)
        self.state = _CombinedState(self._optimizers)
        for kind in ("muon", "adam"):
            opt = getattr(self, kind)
            if opt is not None:
                for group in opt.param_groups:
                    group["optimizer_kind"] = kind
        self.capturable = False
        if capture and cuda:
            self.prepare_capture(self.device)

    @property
    def param_groups(self):
        return [group for opt in self._optimizers for group in opt.param_groups]

    def zero_grad(self, set_to_none=True):
        for opt in self._optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def prepare_capture(self, device=None):
        device = self.device if device is None else torch.device(device)
        if device.type != "cuda":
            raise ValueError("Captured MuonAdam requires CUDA")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device != self.device:
            raise ValueError("Captured optimizer device differs from its parameters")
        if self.adam is not None:
            for group in self.adam.param_groups:
                group.update(fused=True, foreach=False, capturable=True)
            prepare_adam_for_capture(self.adam, device)
        if self.muon is not None:
            for group in self.muon.param_groups:
                lr = group["lr"]
                if not isinstance(lr, torch.Tensor):
                    group["lr"] = torch.tensor(float(lr), dtype=torch.float32, device=device)
                elif lr.device != device:
                    group["lr"] = lr.to(device)
            for values in self.muon.state.values():
                if "step" in values and values["step"].device != device:
                    values["step"] = values["step"].to(device)
        self.capturable = True

    @torch.no_grad()
    def _muon_step_tensor_lr(self):
        for group in self.muon.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse or gradient.is_complex():
                    raise RuntimeError("Muon requires dense real gradients")
                state = self.muon.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(gradient, memory_format=torch.preserve_format)
                buffer = state["momentum_buffer"]
                buffer.lerp_(gradient, 1 - group["momentum"])
                update = gradient.lerp(buffer, group["momentum"]) if group["nesterov"] else buffer
                update = self._zeropower(
                    update, group["ns_coefficients"], group["ns_steps"], group["eps"],
                )
                ratio = self._adjust_lr(1.0, group["adjust_lr_fn"], parameter.shape)
                # Convert before multiplying: BF16 * scalar Tensor otherwise
                # retains BF16, unlike the native FP32 parameter update.
                delta = update.to(parameter.dtype) * (group["lr"] * -ratio)
                if group["weight_decay"] != 0:
                    parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(delta)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self.adam is not None:
            self.adam.step()
        if self.muon is not None:
            if self.capturable:
                self._muon_step_tensor_lr()
            else:
                self.muon.step()
            for group in self.muon.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        state = self.muon.state[parameter]
                        if "step" not in state:
                            state["step"] = torch.zeros((), dtype=torch.float32, device=parameter.device)
                        state["step"].add_(1)
        return loss

    def state_dict(self):
        return {"optimizer_kind": "muon_adam", "version": 1,
                "parameter_groups": copy.deepcopy(self._contract),
                "muon": self.muon.state_dict() if self.muon is not None else None,
                "adam": self.adam.state_dict() if self.adam is not None else None}

    def validate_state_dict(self, saved):
        if (not isinstance(saved, dict) or saved.get("optimizer_kind") != "muon_adam"
                or saved.get("version") != 1):
            raise ValueError("MuonAdam checkpoint required; legacy Adam state is incompatible")
        if saved.get("parameter_groups") != self._contract:
            raise ValueError("MuonAdam parameter name/shape/dtype grouping differs")
        for kind in ("muon", "adam"):
            opt, state = getattr(self, kind), saved.get(kind)
            if opt is None:
                if state is not None:
                    raise ValueError(f"Unexpected {kind} optimizer state")
            else:
                _validate_native_state(opt, state, kind)

    def load_state_dict(self, saved):
        self.validate_state_dict(saved)
        for kind in ("muon", "adam"):
            opt = getattr(self, kind)
            if opt is not None:
                opt.load_state_dict(saved[kind])
        # Native Optimizer.load_state_dict replaces live group dictionaries.
        # Rebind capture-safe scalar LR/step tensors before recording a graph.
        if self.device.type == "cuda":
            if self.adam is not None:
                for group in self.adam.param_groups:
                    group.update(fused=True, foreach=False, capturable=True)
                prepare_adam_for_capture(self.adam, self.device)
            if self.capturable:
                self.prepare_capture(self.device)
        else:
            for opt in self._optimizers:
                for group in opt.param_groups:
                    group["lr"] = _scalar(group["lr"], "LR")
                    if opt is self.adam:
                        group.update(fused=False, foreach=None, capturable=False)


def build_training_optimizer(module: nn.Module, *, kind: Literal["adam", "muon"], lr: float,
                             betas=(0.9, 0.999), adam_weight_decay: float = 0.0,
                             muon_weight_decay: float = 0.1, capture: bool = False):
    if kind == "muon":
        return MuonAdam(module, lr=lr, betas=betas, adam_weight_decay=adam_weight_decay,
                        muon_weight_decay=muon_weight_decay, capture=capture)
    if kind != "adam":
        raise ValueError("Training optimizer must be 'adam' or 'muon'")
    parameters = list(module.parameters())
    opt = torch.optim.Adam(parameters, lr=lr, betas=betas, weight_decay=adam_weight_decay)
    if capture and parameters[0].device.type == "cuda":
        prepare_adam_for_capture(opt, parameters[0].device)
    return opt


def prepare_training_optimizer(optimizer, device):
    if isinstance(optimizer, MuonAdam):
        optimizer.prepare_capture(device)
    else:
        prepare_adam_for_capture(optimizer, device)


def validate_training_optimizer_state(optimizer, saved):
    if isinstance(optimizer, MuonAdam):
        optimizer.validate_state_dict(saved)
    else:
        if isinstance(saved, dict) and "optimizer_kind" in saved:
            raise ValueError("Muon optimizer checkpoint is incompatible with Adam-only training")
        _validate_native_state(optimizer, saved, "adam")

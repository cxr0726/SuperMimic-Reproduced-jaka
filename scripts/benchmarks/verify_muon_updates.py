"""Compare production-size graph Muon/Adam updates with the official Torch rule.

Both implementations receive identical externally supplied gradients. This
isolates optimizer grouping/arithmetic and live LR handling; it does not assert
that Muon has the learning trajectory or final quality of all-Adam training.
No simulator, reference data or repository under ref_thirdparty is required.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from supermimic.config import TrainConfig, resolve_training_observations
from supermimic.policy.nets import Critic, Policy
from supermimic.training.cuda_graph import CapturedUpdate, graph_state
from supermimic.training.optimizers import build_training_optimizer
from supermimic.world_model.world_model import WorldModel


def model_for(role, cfg, device):
    if role == "wm":
        heads = SimpleNamespace(num_joints=27, default_joint_pos=torch.zeros(27, device=device))
        return WorldModel(cfg.world_model, heads, device, action_scale=torch.full((27,), .5, device=device))
    if role == "actor":
        return Policy(cfg.policy, default_joint_pos=torch.zeros(27), action_scale=torch.full((27,), .5)).to(device)
    return Critic(cfg.critic, pcfg=cfg.policy, xi_dim=101).to(device)


def official_optimizers(model, betas, device):
    # Exact grouping/defaults from official learners/learner_utils.py.
    hidden = [linear.weight for seq in model.modules() if isinstance(seq, nn.Sequential)
              for linear in [item for item in seq if isinstance(item, nn.Linear)][1:-1]]
    ids = {id(parameter) for parameter in hidden}
    rest = [parameter for parameter in model.parameters() if id(parameter) not in ids]
    return (torch.optim.Adam(rest, lr=torch.tensor(2e-4, device=device), betas=betas,
                             fused=True, capturable=True),
            torch.optim.Muon(hidden, lr=2e-4, adjust_lr_fn="match_rms_adamw"))


def compare(actual, expected):
    maximum, numerator, denominator = 0., 0., 0.
    for (name, value), (other, reference) in zip(actual.named_parameters(), expected.named_parameters(), strict=True):
        assert name == other
        torch.testing.assert_close(value, reference, rtol=2e-6, atol=2e-7, msg=name)
        difference = value.detach().double() - reference.detach().double()
        maximum = max(maximum, float(difference.abs().max()))
        numerator += float(difference.square().sum())
        denominator += float(reference.detach().double().square().sum())
    return {"max_abs": maximum, "relative_l2": (numerator / max(denominator, 1e-30)) ** .5}


def run_role(role, cfg, device):
    torch.manual_seed(371)
    model = model_for(role, cfg, device)
    reference = copy.deepcopy(model)
    betas = cfg.policy_trainer.betas if role == "actor" else (.9, .999)
    optimizer = build_training_optimizer(model, kind="muon", lr=2e-4, betas=betas, capture=True)
    adam, muon = official_optimizers(reference, betas, device)
    parameters = list(model.parameters())
    gradients = [torch.zeros_like(parameter) for parameter in parameters]
    for parameter in parameters:
        parameter.grad = torch.zeros_like(parameter)

    def update():
        for parameter, gradient in zip(parameters, gradients, strict=True):
            parameter.grad.copy_(gradient)
        optimizer.step()
        return {"finite": parameters[0].new_ones(())}

    captured = CapturedUpdate(update, lambda: graph_state((model,), (optimizer,)), warmup=3)
    generator = torch.Generator(device=device).manual_seed(125)
    addresses = [group["lr"].data_ptr() for group in optimizer.param_groups]
    report = {"role": role, "parameters": sum(parameter.numel() for parameter in parameters),
              "muon_matrices": sum(len(group["params"]) for group in optimizer.param_groups
                                   if group["optimizer_kind"] == "muon"), "rounds": []}
    try:
        for step, lr in enumerate((2e-4, 2e-4, 2e-4, 1e-4, 0., 0., 6e-5, 6e-6), start=1):
            for gradient in gradients:
                gradient.copy_(torch.randn(gradient.shape, generator=generator, device=device) * .01)
            for group in optimizer.param_groups:
                group["lr"].fill_(lr)
            adam.param_groups[0]["lr"].fill_(lr)
            muon.param_groups[0]["lr"] = lr
            before = [parameter.detach().clone() for parameter in parameters] if lr == 0 else None
            captured()
            for parameter, gradient in zip(reference.parameters(), gradients, strict=True):
                parameter.grad = gradient.clone()
            adam.step()
            muon.step()
            error = compare(model, reference)
            assert {int(state["step"]) for state in optimizer.state.values()} == {step}
            assert addresses == [group["lr"].data_ptr() for group in optimizer.param_groups]
            if before is not None:
                for parameter, previous in zip(parameters, before, strict=True):
                    torch.testing.assert_close(parameter, previous, rtol=0, atol=0)
            reference_by_name = dict(reference.named_parameters())
            for name, parameter in model.named_parameters():
                if "momentum_buffer" in optimizer.state[parameter]:
                    torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"],
                                               muon.state[reference_by_name[name]]["momentum_buffer"], rtol=0, atol=0)
            row = {"step": step, "lr": lr, "graph_phase": captured.phase, **error}
            report["rounds"].append(row)
            print(json.dumps({"role": role, **row}), flush=True)
        assert captured.calls == 8 and captured.captured
        report["all_parameter_states_at_step"] = 8
        return report
    finally:
        captured.invalidate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    cfg = TrainConfig(robot="khan_mini")
    resolve_training_observations(cfg)
    report = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(torch.device(args.device)),
              "reference": "official parameter partition, native Torch Muon + fused capturable Adam",
              "gradient_source": "identical supplied gradients, not an end-to-end learning comparison",
              "muon_weight_decay": .1, "roles": []}
    for role in ("wm", "actor", "critic"):
        report["roles"].append(run_role(role, cfg, args.device))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

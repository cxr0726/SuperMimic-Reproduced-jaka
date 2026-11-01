"""Test saved world models on recorded real transitions; no simulation or training.

The open-loop tests replay the *real recorded action sequence* from a real state.
They diagnose prediction error, not a fresh closed-loop imagined policy rollout.
Only run this on trusted PyTorch checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from supermimic.world_model.dynamics import DynamicsMLP
from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg
from supermimic.world_model.world_model import TerminationHead


def load_model(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]["world_model"]
    spec_cfg = StateSpecCfg(**{
        key: tuple(value) if isinstance(value, list) else value
        for key, value in cfg["state_spec"].items()
    })
    nj = checkpoint["config"]["policy"]["num_joints"]
    spec = StateSpec(spec_cfg, "cpu", num_joints=nj)
    dynamics = DynamicsMLP(spec.feat_dim, nj, spec.xi_dim, cfg["hidden"], spec.delta_dim)
    termination = TerminationHead(cfg["term_hidden"], cfg["height_threshold"], cfg["tilt_threshold"])
    for module, prefix in ((dynamics, "dynamics."), (termination, "term_head.")):
        module.load_state_dict({
            key.removeprefix(prefix): value for key, value in checkpoint["wm"].items()
            if key.startswith(prefix)
        })
        module.eval()
    return spec, dynamics, termination


def features(state, pos, quat):
    ref = ReferenceFrame(None, None, None, None, pos, quat)
    return AnalyticHeads.termination_features(state[..., :3], state[..., 3:7], ref)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("logs/Khan_lafan1_4096"))
    parser.add_argument("--record-dir", type=Path)
    args = parser.parse_args()
    directory = args.record_dir or args.run_dir / "analysis/failure_videos"
    torch.set_num_threads(2)
    models = {it: load_model(args.run_dir / f"ckpt_{it:07d}.pt") for it in (40000, 200000)}
    reports, plot_traces = [], []
    for policy_it in (40000, 200000):
        path = directory / f"transitions_{policy_it:07d}.npz"
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        names = data["clip_name"]
        tensors = {key: torch.from_numpy(data[key]).float() for key in
                   ("s", "s_next", "action", "xi", "ref_root_pos", "ref_root_quat")}
        for model_it, (spec, dynamics, termination) in models.items():
            def step(state, action, xi, spec=spec, dynamics=dynamics):
                delta = dynamics(spec.canonical(state), action, spec.normalize_xi(xi))
                return spec.integrate(state, delta)

            predictions = torch.cat([
                step(tensors["s"][start:start + 256], tensors["action"][start:start + 256],
                     tensors["xi"][start:start + 256])
                for start in range(0, len(names), 256)
            ])
            nse = spec.normalized_sq_error(predictions, tensors["s_next"])
            for name in dict.fromkeys(names):
                indices = np.flatnonzero(names == name)
                indices = indices[np.argsort(data["step"][indices])]
                idx = torch.as_tensor(indices)
                row = {
                    "policy_iteration": policy_it, "model_iteration": model_it,
                    "clip": str(name), "recorded_transitions": len(indices),
                    "teacher_forced_1step_nse_mean": float(nse[idx].mean()),
                    "teacher_forced_1step_nse_last_32_mean": float(nse[idx[-32:]].mean()),
                    "windows": [],
                }
                for horizon in (1, 8, 32):
                    window = idx[-horizon:]
                    state = tensors["s"][window[0]].unsqueeze(0)
                    survival = 1.0
                    pred_features, actual_features = [], []
                    for sample in window:
                        state = step(state, tensors["action"][sample:sample + 1], tensors["xi"][sample:sample + 1])
                        pf = features(state, tensors["ref_root_pos"][sample:sample + 1],
                                      tensors["ref_root_quat"][sample:sample + 1])
                        af = features(tensors["s_next"][sample:sample + 1],
                                      tensors["ref_root_pos"][sample:sample + 1],
                                      tensors["ref_root_quat"][sample:sample + 1])
                        survival *= float(1 - termination(pf).sigmoid()[0])
                        pred_features.append(pf[0].numpy())
                        actual_features.append(af[0].numpy())
                    final = window[-1]
                    true_state = tensors["s_next"][final:final + 1]
                    af = features(true_state, tensors["ref_root_pos"][final:final + 1],
                                  tensors["ref_root_quat"][final:final + 1])
                    window_result = {
                        "horizon": horizon,
                        "start_step": int(data["step"][window[0]]),
                        "end_step": int(data["step"][final]),
                        "real_transition_terminated": bool(data["terminated"][final]),
                        "nse_at_end": float(spec.normalized_sq_error(state, true_state)[0]),
                        "actual_raw_state_height_error": float(af[0, 0]),
                        "actual_raw_state_tilt_error": float(af[0, 1]),
                        "actual_check_height_error_abs": float(data["height_error_at_check"][final]),
                        "actual_check_tilt_error_abs": float(data["tilt_error_at_check"][final]),
                        "predicted_height_error": float(pf[0, 0]),
                        "predicted_tilt_error": float(pf[0, 1]),
                        "predicted_termination_probability_at_end": float(termination(pf).sigmoid()[0]),
                        "termination_probability_on_actual_raw_state": float(termination(af).sigmoid()[0]),
                        "predicted_survival_over_recorded_actions": survival,
                    }
                    row["windows"].append(window_result)
                    if policy_it == 200000 and horizon == 32:
                        plot_traces.append({
                            "clip": str(name), "model": model_it,
                            "time": data["step"][window.numpy()] * float(data["dt"]),
                            "pred": np.array(pred_features), "actual": np.array(actual_features),
                        })
                reports.append(row)
    output = {
        "protocol": "Offline real-action replay. Each open-loop window starts from the true recorded state; actions and context are the real recorded sequence, not actions recomputed in predicted states.",
        "state_timestamp": "Raw qpos/qvel are post-integration. Captured true termination checks use the normal mjlab kinematics lag of one 0.005s physics substep. Raw-state model features can differ slightly from those checks.",
        "nse": "Normalized squared error averaged over state error dimensions using checkpoint StateSpec scales; not physical percentage error.",
        "results": reports,
    }
    (directory / "world_model_audit.json").write_text(json.dumps(output, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clip_names = list(dict.fromkeys(trace["clip"] for trace in plot_traces))
    fig, axes = plt.subplots(2, len(clip_names), figsize=(13, 7), squeeze=False, constrained_layout=True)
    for column, name in enumerate(clip_names):
        traces = [trace for trace in plot_traces if trace["clip"] == name]
        for dimension, threshold in ((0, 0.25), (1, 0.8)):
            axis = axes[dimension, column]
            axis.plot(traces[0]["time"], abs(traces[0]["actual"][:, dimension]), color="black", label="Actual integrated state")
            for trace in traces:
                axis.plot(trace["time"], abs(trace["pred"][:, dimension]), label=f"WM {trace['model']//1000}k, recorded actions")
            axis.axhline(threshold, color="red", linestyle="--", label="Termination threshold")
            axis.set(title=name if dimension == 0 else "", xlabel="Time (s)",
                     ylabel="Absolute height error (m)" if dimension == 0 else "Absolute tilt feature error")
            axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Last 32 real transitions before 200k failure: open-loop model predictions")
    fig.savefig(directory / "world_model_before_failure.png", dpi=160)
    plt.close(fig)
    print("saved", directory / "world_model_audit.json")


if __name__ == "__main__":
    main()

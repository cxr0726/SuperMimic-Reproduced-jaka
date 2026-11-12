"""Export the current Khan actor with an explicit, self-describing input contract.

The packed interface follows the user's existing Isaac ONNX: command first,
then proprioception. History, reference loading and PD control stay external.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn

from supermimic.eval.benchmark import PolicyWeights, load_policy_from_checkpoint
from supermimic.policy.nets import Policy
from supermimic.robots.contracts import robot_contract
from supermimic.robots.khan_mini import load_profile

InputLayout = Literal["packed", "split"]


class PackedActor(nn.Module):
    """obs = [goal, policy_obs], with the trained encoder and decoder intact."""

    def __init__(self, policy: Policy):
        super().__init__()
        self.policy = policy
        self.goal_dim = policy.cfg.goal_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.policy(obs[:, self.goal_dim:], obs[:, :self.goal_dim])


class SplitActor(nn.Module):
    def __init__(self, policy: Policy):
        super().__init__()
        self.policy = policy

    def forward(self, policy_obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.policy(policy_obs, goal)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deployment_schema(input_layout: InputLayout, policy_config: dict | None = None) -> dict:
    """Half-open offsets refer to flat tensors, never frame-major histories."""
    policy_config = policy_config or {}
    action_mode = policy_config.get("action_mode", "default")
    residual = action_mode == "reference_residual"
    goal_terms = [
        ("reference_root_displacement_b", 0, 15, 3, 1.0),
        ("reference_world_z", 15, 20, 1, 1.0),
        ("reference_joint_pos", 20, 155, 27, 1.0),
    ]
    policy_terms = [
        ("motion_anchor_ori_b", 0, 30, 6, 1.0),
        ("projected_gravity", 30, 45, 3, 1.0),
        ("base_ang_vel", 45, 60, 3, 0.5),
        ("joint_pos", 60, 195, 27, 1.0),
        ("joint_vel", 195, 330, 27, 0.2),
        ("actions", 330, 465, 27, 1.0),
    ]

    def terms(values, offset):
        return [{"name": n, "start": a, "stop": b, "packed_start": a+offset,
                 "packed_stop": b+offset, "per_frame_dim": d, "frames": 5,
                 "caller_scale": s} for n, a, b, d, s in values]

    inputs = ([{"name": "obs", "shape": ["batch", 620], "dtype": "float32"}]
              if input_layout == "packed" else [
                  {"name": "policy_obs", "shape": ["batch", 465], "dtype": "float32"},
                  {"name": "goal", "shape": ["batch", 155], "dtype": "float32"}])
    return {
        "version": 1, "format": "supermimic.khan.actor", "input_layout": input_layout,
        "robot": "khan_mini", "observation_layout": "isaac_khan",
        "dimensions": {"obs": 620, "goal": 155, "policy_obs": 465, "actions": 27},
        "inputs": inputs,
        "outputs": [{"name": "actions", "shape": ["batch", 27], "dtype": "float32"}],
        "packed_order": ["goal", "policy_obs"],
        "goal_terms": terms(goal_terms, 0), "policy_terms": terms(policy_terms, 155),
        "future_frames": 5, "history": 5, "future_offsets": [0, 1, 2, 3, 4],
        "history_offsets": [-4, -3, -2, -1, 0],
        "history_layout": "oldest_to_newest_per_term",
        "history_reset": "repeat first sensor sample five times; previous action zero",
        "future_orientation_history": False,
        "reference_tail": "clamp each future frame to the last frame of its clip",
        "root_displacement": "R_ref(t)^T @ (p_ref(t+k)-p_ref(t)); full reference rotation",
        "reference_joint_pos": "absolute radians, not relative to default_joint_pos",
        "joint_pos": "measured radians minus default_joint_pos, in joint_names order",
        "projected_gravity": "R_base^T @ [0,0,-1]",
        "angular_velocity_frame": "base_link body frame, radians/second",
        "quaternion_order": "wxyz",
        "orientation": "R_base(t)^T @ R_ref(t+k)",
        "rotation_6d_order": ["R00", "R01", "R10", "R11", "R20", "R21"],
        "sensor_scale_location": "caller, once; network scale buffers remain identity",
        "running_normalization": bool(policy_config.get("running_normalization", False)),
        "normalization": {
            "embedded_in_onnx": bool(policy_config.get("running_normalization", False)),
            "formula": "(raw_feature - running_mean) / (running_std + 0.01)",
            "sensor_history_statistics": "one shared statistic per raw sensor feature across history lags",
            "khan_sensor_input": "caller omega*0.5 and qd*0.2 are inverted inside the graph before normalization",
            "future_orientation_statistics": "independent flattened 30-dimensional window",
            "goal_statistics": "independent per flattened goal dimension and future offset",
            "updates_at_inference": False,
        },
        "observation_noise": False,
        "actor_stateful": False, "reference_motion_embedded": False,
        "action": {"meaning": "raw normalized joint-position action", "clip": None,
                   "tanh": False, "smoothing": False, "execution_noise": False,
                   "pd_target": "default_joint_pos + action_scale * actions",
                   "last_action": "previous raw actions, not PD target or scaled action",
                   "parameterization": action_mode,
                   "reference_residual_beta": policy_config.get("reference_residual_beta", 1.0) if residual else 0.0,
                   "reference_frame_offset": policy_config.get("reference_frame_offset", 1) if residual else None,
                   "residual_scale": "action_scale" if residual else None,
                   "reference_skip_embedded": residual,
                   "reference_skip_input": "supplied clean goal, before encoder-only noise and running normalization" if residual else None,
                   "reference_skip_units": "raw joint radians" if residual else None},
        "training_goal_noise": {
            "distribution": "independent additive uniform",
            "root_displacement_half_width": 0.0,
            "root_z_half_width_m": policy_config.get("goal_noise_z", 0.0),
            "joint_pos_half_width_rad": policy_config.get("goal_noise_joint_pos", 0.0),
            "scope": "motion encoder only; critic, reward and reference skip remain clean",
            "enabled_in_export": False,
        },
        "control_dt": 0.02, "control_frequency_hz": 50,
        "training_physics_dt": 0.005, "training_decimation": 4,
    }


def _metadata(ckpt: Path, cfg: dict, input_layout: InputLayout) -> dict[str, str]:
    profile = load_profile()
    names = profile["joint_names"]

    def array(key):
        return ",".join(format(float(profile[key][name]), ".9g") for name in names)

    contract = robot_contract("khan_mini", "isaac_khan", env_config=cfg["env"])
    # Dimensions come from the validated actor contract, not profile's historical
    # legacy SuperMimic dimensions section.
    return {
        "run_path": str(ckpt), "checkpoint_sha256": _sha256(ckpt),
        "robot": "khan_mini", "joint_names": ",".join(names),
        "joint_stiffness": array("stiffness"), "joint_damping": array("damping"),
        "default_joint_pos": array("default_joint_pos"), "action_scale": array("action_scale"),
        "effort_limit": array("effort_limit"), "velocity_limit": array("velocity_limit"),
        "armature": array("armature"),
        "joint_pos_lower": ",".join(str(profile["urdf_limits"][n]["lower"]) for n in names),
        "joint_pos_upper": ",".join(str(profile["urdf_limits"][n]["upper"]) for n in names),
        "soft_joint_pos_limit_factor": str(profile["soft_joint_pos_limit_factor"]),
        "velocity_limit_enforced_in_training": "false",
        "command_names": "motion",
        "observation_names": "command,motion_anchor_ori_b,projected_gravity,base_ang_vel,"
                             "joint_pos,joint_vel,actions",
        "observation_history": "5", "goal_frames": "5", "control_dt": "0.02",
        "anchor_body_name": profile["anchor_body"], "root_body_name": profile["root_body"],
        "com_body_name": profile["com_body"], "body_names": ",".join(profile["tracked_bodies"]),
        "observation_layout": "isaac_khan", "input_layout": input_layout,
        "onnx_deployment_schema": json.dumps(deployment_schema(input_layout, cfg["policy"]), sort_keys=True),
        "robot_contract": json.dumps(contract, sort_keys=True),
        "policy_config": json.dumps(cfg["policy"], sort_keys=True),
        "running_normalization": str(bool(cfg["policy"].get("running_normalization", False))).lower(),
        "policy_hidden_layer_norm": str(bool(cfg["policy"].get("hidden_layer_norm", False))).lower(),
        "policy_hidden_activation": cfg["policy"].get("hidden_activation", "silu"),
    }


def _motion_inputs(path: Path, samples: int, q0: torch.Tensor):
    """Reference-state offline inputs, including start reset and tail clamping.

    This is a numerical parity tape, not a physics rollout. Both backends receive
    identical nonzero previous actions and sampled reference proprioception.
    """
    from supermimic.envs.mdp.motion_library import read_npz_reference
    from supermimic.kinematics.rotations import projected_gravity, quat_apply_inverse
    from supermimic.observation_layout import isaac_anchor_orientation, isaac_goal_command
    from supermimic.world_model.heads import flatten_history_term_major

    ref, _ = read_npz_reference(path, robot_name="khan_mini", root_body_name="base_link")
    ref = torch.from_numpy(ref)
    count = ref.shape[0]
    frames = torch.linspace(0, count-1, samples).long()
    indices = (frames[:, None] + torch.arange(5)).clamp(max=count-1)
    future = ref[indices]
    goal = isaac_goal_command(future[..., :3], future[..., 3:7], future[..., 7:34])
    history_index = (frames[:, None] + torch.arange(-4, 1)).clamp(min=0)
    history = ref[history_index]
    quaternion = history[..., 3:7]
    rng = torch.Generator().manual_seed(0)
    # A single repeatable time series preserves reset backfill correlations.
    previous_actions = torch.randn(count, 27, generator=rng) * 0.25
    previous_actions[0] = 0
    sensors = torch.cat([
        projected_gravity(quaternion),
        0.5 * quat_apply_inverse(quaternion, history[..., 37:40]),
        history[..., 7:34] - q0, 0.2 * history[..., 40:67],
        previous_actions[history_index],
    ], dim=-1)
    policy_obs = torch.cat([
        isaac_anchor_orientation(ref[frames, 3:7], future[..., 3:7]),
        flatten_history_term_major(sensors.permute(1, 0, 2), 27, "isaac_khan"),
    ], dim=-1)
    return policy_obs, goal, frames.tolist()


@torch.no_grad()
def _validate(path: Path, policy: Policy, layout: InputLayout, samples: int,
              motion: Path | None) -> dict:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    rng = torch.Generator().manual_seed(0)
    profile = load_profile()
    names = profile["joint_names"]
    q0 = torch.tensor([profile["default_joint_pos"][n] for n in names], dtype=torch.float32)
    scale = np.array([profile["action_scale"][n] for n in names], dtype=np.float32)
    groups = [
        ("zero", torch.zeros(1, 465), torch.zeros(1, 155)),
        ("random_batch1", torch.randn(samples, 465, generator=rng),
         torch.randn(samples, 155, generator=rng)),
        ("random_dynamic_batch3", torch.randn(3, 465, generator=rng),
         torch.randn(3, 155, generator=rng)),
    ]
    report = {"provider": "CPUExecutionProvider", "atol": 1e-4, "rtol": 1e-4,
              "all_checks_passed": True, "groups": []}
    if motion is not None:
        proprio, goal, frames = _motion_inputs(motion, samples, q0)
        groups.append(("motion_reference_states", proprio, goal))
        report["motion"] = {"path": str(motion), "sha256": _sha256(motion),
                            "frames": frames, "physics_rollout": False}
    for name, proprio, goal in groups:
        # Random cases are evaluated one row at a time to exercise the actual
        # deployment batch=1 path, as well as the separate dynamic batch check.
        errors, target_errors = [], []
        chunk = 3 if name == "random_dynamic_batch3" else 1
        for start in range(0, proprio.shape[0], chunk):
            obs_batch, goal_batch = proprio[start:start+chunk], goal[start:start+chunk]
            expected = policy(obs_batch, goal_batch).numpy()
            feed = ({"obs": torch.cat([goal_batch, obs_batch], dim=-1).numpy()}
                    if layout == "packed" else {
                        "policy_obs": obs_batch.numpy(), "goal": goal_batch.numpy()})
            actual = session.run(["actions"], feed)[0]
            if not np.isfinite(actual).all() or not np.isfinite(expected).all():
                raise ValueError(f"Nonfinite inference outputs in {name}")
            np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
            expected_target = q0.numpy() + scale * expected
            actual_target = q0.numpy() + scale * actual
            np.testing.assert_allclose(actual_target, expected_target, atol=1e-4, rtol=1e-4)
            errors.append(float(np.max(np.abs(actual-expected))))
            target_errors.append(float(np.max(np.abs(actual_target-expected_target))))
        report["groups"].append({"name": name, "samples": proprio.shape[0],
                                  "max_action_abs_error": max(errors),
                                  "max_pd_target_abs_error_rad": max(target_errors)})
    report["max_action_abs_error"] = max(g["max_action_abs_error"] for g in report["groups"])
    report["max_pd_target_abs_error_rad"] = max(
        g["max_pd_target_abs_error_rad"] for g in report["groups"])
    return report


def export_policy(ckpt: Path, out: Path, input_layout: InputLayout = "packed",
                  validation_samples: int = 64, motion: Path | None = None,
                  overwrite: bool = False, policy_weights: PolicyWeights = "auto") -> dict:
    """Export and validate atomically; reject incompatible or unverified artifacts."""
    if input_layout not in ("packed", "split") or validation_samples < 1:
        raise ValueError("input_layout must be packed/split and validation_samples must be positive")
    ckpt, out = Path(ckpt).resolve(), Path(out).resolve()
    motion = Path(motion).resolve() if motion is not None else None
    report_path = out.with_suffix(".json")
    if out.suffix.lower() != ".onnx":
        raise ValueError("Output must have the .onnx extension")
    if not overwrite and (out.exists() or report_path.exists()):
        raise FileExistsError(f"Output exists: {out} or {report_path}; use --overwrite")
    try:
        import onnx
        import onnxruntime
    except ImportError as exc:
        raise ImportError("Install export dependencies with: uv sync --extra export") from exc
    policy, cfg = load_policy_from_checkpoint(ckpt, "cpu", policy_weights=policy_weights)
    if (cfg.get("robot") != "khan_mini" or policy.cfg.observation_layout != "isaac_khan"
            or policy.cfg.history != 5 or policy.cfg.future_frames != 5
            or policy.cfg.goal_dim != 155 or policy.decoder.scale.numel() != 465):
        raise ValueError("Exporter requires the current Khan isaac_khan 155/465 v4 actor")
    if cfg.get("world_model", {}).get("state_spec", {}).get("dt", .02) != .02:
        raise ValueError("Current deployment interface requires control dt=0.02")
    for name, value in policy.state_dict().items():
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Nonfinite actor tensor: {name}")
    for scale in (policy.decoder.scale, policy.encoders["motion"].scale):
        if not torch.equal(scale, torch.ones_like(scale)):
            raise ValueError("Khan actor scale buffers must be identity; caller sensor scales are fixed")
    policy = policy.float().eval().requires_grad_(False)
    # The loader has already checked the full robot contract. Reload only scalar
    # provenance; no WM/critic weights are added to the exported module.
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    iteration = int(saved.get("iteration", -1))
    del saved
    metadata = _metadata(ckpt, cfg, input_layout)
    metadata["checkpoint_iteration"] = str(iteration)
    metadata["policy_weights"] = policy.checkpoint_policy_weights
    metadata["policy_weights_requested"] = policy_weights
    if policy.checkpoint_policy_ema_updates is not None:
        metadata["policy_ema_updates"] = str(policy.checkpoint_policy_ema_updates)
    if policy.checkpoint_policy_ema_decay is not None:
        metadata["policy_ema_decay"] = str(policy.checkpoint_policy_ema_decay)
    model = PackedActor(policy) if input_layout == "packed" else SplitActor(policy)
    inputs = ((torch.zeros(1, 620),) if input_layout == "packed" else
              (torch.zeros(1, 465), torch.zeros(1, 155)))
    input_names = ["obs"] if input_layout == "packed" else ["policy_obs", "goal"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".onnx_export_", dir=out.parent) as directory:
        temporary = Path(directory) / out.name
        with torch.no_grad():
            torch.onnx.export(
                model.eval(), inputs, str(temporary), input_names=input_names,
                output_names=["actions"], dynamic_axes={
                    name: {0: "batch"} for name in [*input_names, "actions"]},
                opset_version=17, dynamo=False, external_data=False,
            )
        exported = onnx.load(temporary)
        onnx.helper.set_model_props(exported, metadata)
        onnx.checker.check_model(exported, full_check=True)
        if any(t.data_location == onnx.TensorProto.EXTERNAL for t in exported.graph.initializer):
            raise ValueError("Deployment ONNX must contain its weights in one file")
        onnx.save(exported, temporary)
        validation = _validate(temporary, policy, input_layout, validation_samples, motion)
        report = {
            "onnx": str(out), "onnx_sha256": _sha256(temporary),
            "checkpoint": str(ckpt), "iteration": iteration,
            "policy_weights": policy.checkpoint_policy_weights,
            "policy_weights_requested": policy_weights,
            "policy_ema_updates": policy.checkpoint_policy_ema_updates,
            "policy_ema_decay": policy.checkpoint_policy_ema_decay,
            "input_layout": input_layout, "opset": 17, "dtype": "float32",
            "metadata": metadata, "schema": deployment_schema(input_layout, cfg["policy"]),
            "validation": validation,
            "versions": {"torch": torch.__version__, "onnx": onnx.__version__,
                         "onnxruntime": onnxruntime.__version__},
        }
        temporary_report = Path(directory) / report_path.name
        temporary_report.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        os.replace(temporary, out)
        os.replace(temporary_report, report_path)
    return report

"""Benchmark evaluation (paper Appendix F.1 / plan 5.1).

Every clip is played once from its first frame with the deterministic policy:
no observation/action noise, no reset noise, no domain randomization (foot friction 1.0),
nominal physics. This does not promise bitwise-repeatable GPU contact dynamics.
A clip *fails* if the training termination (pelvis height /
tilt vs reference) fires before the clip ends. Normalised return = sum of per-step
rewards / (max step reward * number of steps).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.observation_layout import (
    goal_dimension,
    resolve_observation_layout,
    resolve_observation_sizes,
)
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.robots.contracts import validate_robot_contract
from supermimic.robots.registry import RobotName, get_robot_spec

PolicyWeights = Literal["auto", "live", "ema"]


@dataclass
class ClipResult:
    name: str
    frames: int
    steps: int
    norm_return: float
    failed: bool
    fail_step: int  # -1 if not failed
    terms: dict | None = None  # per-term mean reward per step (dt-scaled)


def benchmark_source(npz_files: list[str], num_envs: int | None) -> tuple[int, bool]:
    """Size the scene by motion count, including motions inside one packed file.

    Read packed metadata through a CPU memory map, so selecting the scene size
    does not copy the reference tensor or perform any reference FK. The runtime
    MotionLibrary remains responsible for validating the full robot/data layout.
    """
    if not npz_files:
        raise ValueError("run_benchmark needs at least one motion clip")
    requested = 1024 if num_envs is None else num_envs
    if requested < 1:
        raise ValueError("num_envs must be positive")
    packed = len(npz_files) == 1 and Path(npz_files[0]).suffix == ".pt"
    if any(Path(path).suffix == ".pt" for path in npz_files) and not packed:
        raise ValueError("Use one packed .pt library or a list of NPZ files")
    if packed:
        pack = torch.load(npz_files[0], map_location="cpu", weights_only=False, mmap=True)
        if not isinstance(pack, dict) or "names" not in pack:
            raise ValueError("Packed library needs clip names")
        count = len(pack["names"])
        if count < 1 or len(set(pack["names"])) != count:
            raise ValueError("Packed library needs nonempty, unique clip names")
    else:
        count = len(npz_files)
    return min(requested, count), packed


def load_policy_from_checkpoint(path: str | Path, device: str, *,
                                policy_weights: PolicyWeights = "auto") -> tuple[Policy, dict]:
    """Load EMA for inference when available, preserving legacy live checkpoints.

    Explicit EMA selection must be present and valid; an invalid EMA state never
    falls back to live weights. The returned policy records the resolved source
    in ``checkpoint_policy_weights`` for evaluation/export provenance.
    """
    if policy_weights not in ("auto", "live", "ema"):
        raise ValueError("policy_weights must be auto, live or ema")
    ck = torch.load(path, map_location=device, weights_only=False)
    has_ema = ck.get("policy_ema") is not None
    if policy_weights == "ema" and not has_ema:
        raise ValueError("EMA policy weights requested, but checkpoint has no policy_ema")
    use_ema = policy_weights == "ema" or (policy_weights == "auto" and has_ema)
    selected_weights = "ema" if use_ema else "live"
    state_key = "policy_ema" if selected_weights == "ema" else "policy"
    state = ck[state_key]
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Checkpoint {state_key} must be a nonempty policy state_dict")
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError(f"Checkpoint {state_key} tensor {name} is invalid or nonfinite")
    cfg = ck["config"]
    robot = get_robot_spec(cfg.get("robot", "g1"))
    pc = dict(cfg["policy"])
    # Historical checkpoints could omit these original four-layer defaults.
    # Fresh training defaults must not reinterpret their saved architecture.
    pc.setdefault("encoder_hidden", (2048, 1024, 512, 256))
    pc.setdefault("decoder_hidden", (2048, 1024, 512, 256))
    # Absence means the original fixed-scale input mapping, never today's
    # training default. The saved statistics travel with a normalized actor.
    pc.setdefault("running_normalization", False)
    if pc.get("num_joints", robot.num_joints) != robot.num_joints:
        raise ValueError(f"Checkpoint policy joint count differs from robot {robot.name}")
    pc["num_joints"] = robot.num_joints
    if pc.get("num_tracked_bodies") not in (None, len(robot.tracked_bodies)):
        raise ValueError(f"Checkpoint tracked-body count differs from robot {robot.name}")
    pc["num_tracked_bodies"] = len(robot.tracked_bodies)
    # Absence denotes the original observation schema, never the robot's new default.
    layout = resolve_observation_layout(robot.name, cfg["env"].get("observation_layout", "supermimic"))
    future, history = resolve_observation_sizes(
        layout, cfg["env"]["future_frames"], pc.get("history", 10),
        cfg["env"].get("anchor_obs", False),
    )
    if pc.get("observation_layout", layout) != layout:
        raise ValueError("Checkpoint policy and environment observation layouts differ")
    if pc.get("future_frames", future) != future:
        raise ValueError("Checkpoint policy and environment future windows differ")
    expected_goal = goal_dimension(layout, robot.num_joints, future, cfg["env"].get("anchor_obs", False))
    if pc.get("goal_dim", expected_goal) != expected_goal:
        raise ValueError(f"Checkpoint goal dimension differs from robot {robot.name} and future-frame settings")
    pc["goal_dim"] = expected_goal
    pc.update(observation_layout=layout, future_frames=future, history=history)
    validate_robot_contract(ck, robot.name, layout)
    pcfg = PolicyCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in pc.items()})
    reference_parameters = {}
    if pcfg.action_mode == "reference_residual":
        if robot.name != "khan_mini" or layout != "isaac_khan":
            raise ValueError("Reference-residual checkpoints require the Khan Isaac observation layout")
        from supermimic.robots.khan_mini import load_profile

        profile = load_profile()
        # These buffers define physical target semantics. Validate the saved
        # values before load_state_dict, which would otherwise overwrite the
        # correct robot values with a stale or corrupted checkpoint baseline.
        for argument, buffer_name, profile_key in (
            ("default_joint_pos", "reference_q_default", "default_joint_pos"),
            ("action_scale", "reference_action_scale", "action_scale"),
        ):
            expected = torch.tensor([profile[profile_key][name] for name in robot.joint_names],
                                    dtype=torch.float32, device=device)
            saved = state.get(buffer_name)
            if (not isinstance(saved, torch.Tensor) or saved.dtype != expected.dtype
                    or saved.shape != expected.shape or not torch.equal(saved.to(device), expected)):
                raise ValueError(f"Checkpoint {buffer_name} differs from the robot's joint-order {profile_key}")
            reference_parameters[argument] = expected
    policy = Policy(pcfg, **reference_parameters).to(device)
    policy.load_state_dict(state)
    policy.checkpoint_policy_weights = selected_weights
    policy.checkpoint_policy_ema_updates = ck.get("policy_ema_updates") if use_ema else None
    policy.checkpoint_policy_ema_decay = ck.get("policy_ema_decay") if use_ema else None
    policy.eval()
    return policy, cfg


@torch.no_grad()
def run_benchmark(policy: Policy, npz_files: list[str], device: str = "cuda:0", num_envs: int | None = None, future_frames: int | None = None, verbose: bool = True, anchor_obs: bool = False, robot: RobotName = "g1", history_length: int | None = None, observation_layout: str | None = None, reset_root_z_offset: float | None = None, reference_fk_compile: bool = False) -> list[ClipResult]:
    robot_spec = get_robot_spec(robot)
    if policy.cfg.num_joints != robot_spec.num_joints:
        raise ValueError(f"Policy joint count differs from benchmark robot {robot}")
    num_envs, compact = benchmark_source(npz_files, num_envs)
    layout = observation_layout or policy.cfg.observation_layout
    if layout != policy.cfg.observation_layout:
        raise ValueError("Policy and benchmark observation layouts differ")
    future_frames, history_length = resolve_observation_sizes(
        layout, policy.cfg.future_frames if future_frames is None else future_frames,
        policy.cfg.history if history_length is None else history_length, anchor_obs,
    )
    args = SuperMimicEnvCfgArgs(
        npz_files=npz_files,
        robot=robot,
        num_envs=num_envs,
        future_frames=future_frames,
        history_length=policy.cfg.history if history_length is None else history_length,
        observation_layout=layout,
        episode_length_s=1.0e5,  # clip end decides
        action_noise_std=0.0,
        obs_noise=False,
        reset_noise=False,
        domain_randomization=False,
        actuator_delay=False,
        reset_root_z_offset=reset_root_z_offset,
        sampling_mode="start",
        record_transitions=False,
        seed=0,
        anchor_obs=anchor_obs,
        compact_library=compact,
        reference_fk_compile=reference_fk_compile,
    )
    env = make_env(args, device=device)
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        n_clips = lib.num_clips
        if n_clips < num_envs:
            raise ValueError("Motion count changed after benchmark scene sizing")
        max_r = env.step_dt * sum(max(0.0, term.weight) for term in env.cfg.rewards.values())

        # Clip queue: env e plays clips e, e+N, e+2N, ...
        current = torch.arange(num_envs, device=device)
        nxt = current + num_envs
        cmd.forced_frame[:] = lib.clip_start[current]
        obs, _ = env.reset()
        cmd.forced_frame[:] = lib.clip_start[torch.clamp(nxt, max=n_clips - 1)]
        active = torch.ones(num_envs, dtype=torch.bool, device=device)
        ret_sum = torch.zeros(num_envs, device=device)
        steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        term_names = list(env.reward_manager.active_terms)
        term_sum = torch.zeros(num_envs, len(term_names), device=device)
        results: dict[int, ClipResult] = {}
        remaining = n_clips
        while remaining > 0:
            act = policy(obs["policy"], obs["goal"])
            obs, rew, term, tout, _ = env.step(act)
            ret_sum += rew
            term_sum += env.reward_manager.term_rewards
            steps += 1
            done = (term | tout) & active
            if done.any():
                ids = done.nonzero().squeeze(-1)
                for e in ids.tolist():
                    ci = int(current[e])
                    if ci < n_clips and ci not in results:
                        st = int(steps[e])
                        results[ci] = ClipResult(
                            name=lib.clips[ci].name,
                            frames=lib.clips[ci].length,
                            steps=st,
                            norm_return=float(ret_sum[e]) / (max_r * max(st, 1)),
                            failed=bool(term[e]),
                            fail_step=st if bool(term[e]) else -1,
                            terms={n: float(term_sum[e, i]) / max(st, 1) for i, n in enumerate(term_names)},
                        )
                        remaining -= 1
                        if verbose and (len(results) % 50 == 0 or len(results) == n_clips):
                            print(f"  evaluated {len(results)}/{n_clips}")
                # advance the queue for these envs (the auto-reset already used forced_frame)
                current[ids] = nxt[ids]
                nxt[ids] = nxt[ids] + num_envs
                finished = current[ids] >= n_clips
                active[ids[finished]] = False
                cmd.forced_frame[ids] = lib.clip_start[torch.clamp(nxt[ids], max=n_clips - 1)]
                ret_sum[ids] = 0.0
                term_sum[ids] = 0.0
                steps[ids] = 0
        return [results[i] for i in range(n_clips)]
    finally:
        env.close()


def summarize(results: list[ClipResult]) -> dict:
    n = len(results)
    failed = sum(r.failed for r in results)
    mean_ret = sum(r.norm_return for r in results) / max(n, 1)
    # Length-weighted return (paper Fig. 3a uses the benchmark aggregate; weighting by steps).
    tot_steps = sum(r.steps for r in results)
    w_ret = sum(r.norm_return * r.steps for r in results) / max(tot_steps, 1)
    out = {"clips": n, "failed": failed, "mean_norm_return": mean_ret, "step_weighted_norm_return": w_ret}
    if results and results[0].terms:
        out["terms"] = {k: sum(r.terms[k] * r.steps for r in results) / max(tot_steps, 1) for k in results[0].terms}
    return out


def save_results(results: list[ClipResult], path: str | Path) -> None:
    Path(path).write_text(json.dumps({"summary": summarize(results), "clips": [asdict(r) for r in results]}, indent=1))

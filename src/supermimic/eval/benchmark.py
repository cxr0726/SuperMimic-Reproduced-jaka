"""Benchmark evaluation (paper Appendix F.1 / plan 5.1).

Every clip is played once from its first frame with the deterministic policy:
no observation/action noise, no reset noise, no domain randomization (foot friction 1.0),
deterministic physics. A clip *fails* if the training termination (pelvis height /
tilt vs reference) fires before the clip ends. Normalised return = sum of per-step
rewards / (max step reward * number of steps).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

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


@dataclass
class ClipResult:
    name: str
    frames: int
    steps: int
    norm_return: float
    failed: bool
    fail_step: int  # -1 if not failed
    terms: dict | None = None  # per-term mean reward per step (dt-scaled)


def load_policy_from_checkpoint(path: str | Path, device: str) -> tuple[Policy, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    robot = get_robot_spec(cfg.get("robot", "g1"))
    pc = dict(cfg["policy"])
    if pc.get("num_joints", robot.num_joints) != robot.num_joints:
        raise ValueError(f"Checkpoint policy joint count differs from robot {robot.name}")
    pc["num_joints"] = robot.num_joints
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
    policy = Policy(pcfg).to(device)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    return policy, cfg


@torch.no_grad()
def run_benchmark(policy: Policy, npz_files: list[str], device: str = "cuda:0", num_envs: int | None = None, future_frames: int | None = None, verbose: bool = True, anchor_obs: bool = False, robot: RobotName = "g1", history_length: int | None = None, observation_layout: str | None = None) -> list[ClipResult]:
    robot_spec = get_robot_spec(robot)
    if policy.cfg.num_joints != robot_spec.num_joints:
        raise ValueError(f"Policy joint count differs from benchmark robot {robot}")
    if not npz_files:
        raise ValueError("run_benchmark needs at least one motion clip")
    num_envs = min(num_envs or 1024, len(npz_files))
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
        sampling_mode="start",
        record_transitions=False,
        seed=0,
        anchor_obs=anchor_obs,
    )
    env = make_env(args, device=device)
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        n_clips = lib.num_clips
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

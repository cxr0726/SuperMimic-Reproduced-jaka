"""Watch a trained tracking policy replay reference clips in an interactive viewer.

  uv run python scripts/play.py --ckpt logs/e5_bones_seed/ckpt_latest.pt --clips dance1_subject1
  uv run python scripts/play.py --ckpt logs/e3_lafan1_40_anchor/ckpt_latest.pt --npz-dir data/npz/lafan1 --num-envs 4

The grey/blue translucent robot is the reference pose; the solid robot is the policy.
Deterministic evaluation settings (no noise, no DR); every reset starts the clip from its first frame.
Native MuJoCo viewer when a display is available, otherwise viser (browser, http://localhost:8080).
Keys (native): space pause, right-arrow single step, +/- speed, backspace reset.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import torch
import tyro
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import PolicyWeights, load_policy_from_checkpoint
from supermimic.robots.contracts import checkpoint_context_layout, checkpoint_foot_tracking
from supermimic.robots.registry import get_robot_spec


class _DictPolicy:
    """Adapts policy(obs_policy, obs_goal) to the viewer's policy(obs) protocol."""

    def __init__(self, policy):
        self.policy = policy

    @torch.no_grad()
    def __call__(self, obs):
        return self.policy(obs["policy"], obs["goal"])


def main(
    ckpt: Path,
    npz_dir: Path | None = None,
    clips: list[str] | None = None,
    num_envs: int = 1,
    device: str = "cuda:0",
    viewer: Literal["auto", "native", "viser"] = "auto",
    ghost: bool = True,
    sampling: Literal["start", "uniform"] = "start",
    policy_weights: PolicyWeights = "auto",
) -> None:
    """--policy-weights auto prefers saved EMA; live uses the training actor."""
    policy, cfg = load_policy_from_checkpoint(ckpt, device, policy_weights=policy_weights)
    robot = get_robot_spec(cfg.get("robot", "g1"))
    npz_dir = npz_dir or robot.default_npz_dir
    files = [str(f) for f in discover_npz(npz_dir, names=clips)]
    args = SuperMimicEnvCfgArgs(
        npz_files=files,
        robot=robot.name,
        num_envs=num_envs,
        future_frames=cfg["env"]["future_frames"],
        history_length=policy.cfg.history,
        observation_layout=policy.cfg.observation_layout,
        context_layout=checkpoint_context_layout(cfg),
        foot_tracking=checkpoint_foot_tracking(cfg),
        episode_length_s=1.0e5,
        action_noise_std=0.0,
        obs_noise=False,
        reset_noise=False,
        domain_randomization=False,
        actuator_delay=False,
        reset_root_z_offset=cfg["env"].get("reset_root_z_offset"),
        sampling_mode=sampling,
        record_transitions=False,
        seed=0,
        anchor_obs=cfg["env"].get("anchor_obs", False),
        debug_vis=ghost,
    )
    env = RslRlVecEnvWrapper(make_env(args, device=device))
    pol = _DictPolicy(policy)
    if viewer == "auto":
        viewer = "native" if (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) else "viser"
    print(f"[INFO] {len(files)} clip(s), {num_envs} env(s), viewer={viewer}, "
          f"policy weights={policy.checkpoint_policy_weights}")
    try:
        if viewer == "native":
            NativeMujocoViewer(env, pol).run()
        else:
            ViserPlayViewer(env, pol).run()
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)

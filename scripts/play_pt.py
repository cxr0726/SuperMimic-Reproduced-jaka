"""Play selected motions directly from a compact PT with online reference FK.

Use exact library names (including lafan1__/mixed__ prefixes). Only selected
compact states are copied to the GPU; no NPZ or temporary library is written.
Solid robot = policy, translucent blue robot = reference. Multiple selected
motions are sampled on reset, like scripts/play.py.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Literal

import torch
import tyro
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.robots.registry import get_robot_spec


class _DictPolicy:
    def __init__(self, policy):
        self.policy = policy

    @torch.no_grad()
    def __call__(self, obs):
        return self.policy(obs["policy"], obs["goal"])


def main(
    ckpt: Path,
    library: Path,
    clips: list[str],
    num_envs: int = 1,
    device: str = "cuda:0",
    viewer: Literal["auto", "native", "viser"] = "auto",
    ghost: bool = True,
    sampling: Literal["start", "uniform"] = "start",
    reference_fk_compile: bool = False,
    steps: int | None = None,
) -> None:
    """Start at clip frame zero by default. --steps optionally bounds viewer execution."""
    if num_envs < 1 or (steps is not None and steps < 1):
        raise ValueError("num_envs and optional steps must be positive")
    if not clips or len(set(clips)) != len(clips):
        raise ValueError("clips must contain nonempty, unique exact library names")
    if library.suffix != ".pt":
        raise ValueError("--library must be a compact .pt file")
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    robot = get_robot_spec(cfg.get("robot", "g1"))
    args = SuperMimicEnvCfgArgs(
        npz_files=[str(library)], robot=robot.name, num_envs=num_envs,
        future_frames=cfg["env"]["future_frames"], history_length=policy.cfg.history,
        observation_layout=policy.cfg.observation_layout, episode_length_s=1.0e5,
        action_noise_std=0.0, obs_noise=False, reset_noise=False,
        domain_randomization=False, actuator_delay=False,
        reset_root_z_offset=cfg["env"].get("reset_root_z_offset"),
        sampling_mode=sampling, record_transitions=False, seed=0,
        anchor_obs=cfg["env"].get("anchor_obs", False), debug_vis=ghost,
        compact_library=True, packed_clip_names=tuple(clips),
        reference_fk_compile=reference_fk_compile,
        extra_command_kwargs={"library_fp16": cfg["env"].get("library_fp16", True)},
        root_height_threshold=cfg["world_model"].get("height_threshold", .25),
        root_tilt_threshold=cfg["world_model"].get("tilt_threshold", .8),
    )
    raw_env = make_env(args, device=device)
    try:
        env = RslRlVecEnvWrapper(raw_env)
        pol = _DictPolicy(policy)
        if viewer == "auto":
            viewer = "native" if (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) else "viser"
        print(f"[INFO] PT play: {len(clips)} selected motion(s), {num_envs} env(s), viewer={viewer}")
        print(f"[INFO] Selected: {', '.join(clips)}")
        player = NativeMujocoViewer(env, pol) if viewer == "native" else ViserPlayViewer(env, pol)
        previous_threads = set(threading.enumerate())
        try:
            if steps is None:
                player.run()
            else:
                player.run(num_steps=steps)
        finally:
            if viewer == "native":
                # MuJoCo's pinned passive viewer closes asynchronously. Let its
                # newly created render thread finish before freeing model/data
                # or exiting Python (otherwise short runs can segfault on exit).
                for thread in threading.enumerate():
                    if thread not in previous_threads and thread.name.endswith("(_launch_internal)"):
                        player.close()
                        thread.join(timeout=5)
    finally:
        raw_env.close()


if __name__ == "__main__":
    tyro.cli(main)

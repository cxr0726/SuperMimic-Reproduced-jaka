"""Import Khan mini into mjlab, check finite PD steps, and render its initial pose.

Default: a finite headless check. ``--viewer native`` or ``--viewer viser`` also
opens an interactive viewer after that check. A zero action holds the configured
joint targets; it is not a floating-base balance controller.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg

ROOT = Path(__file__).resolve().parents[1]


def make_preview_env_cfg(
    asset_dir: Path | None = None, num_envs: int = 1
) -> ManagerBasedRlEnvCfg:
    """Use the local Khan entity without importing G1 training dimensions or data."""
    from mjlab.envs import ManagerBasedRlEnvCfg, mdp
    from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    from mjlab.scene import SceneCfg
    from mjlab.sim import MujocoCfg, SimulationCfg
    from mjlab.terrains import TerrainEntityCfg
    from mjlab.viewer import ViewerConfig

    from supermimic.robots.khan_mini import get_khan_mini_action_cfg, get_khan_mini_cfg

    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=num_envs,
            env_spacing=2.0,
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": get_khan_mini_cfg(asset_dir=asset_dir)},
        ),
        observations={
            "policy": ObservationGroupCfg(
                terms={"joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel)},
                concatenate_terms=True,
                enable_corruption=False,
            )
        },
        actions={
            "joint_pos": get_khan_mini_action_cfg(asset_dir, noise_std=0.0)
        },
        sim=SimulationCfg(
            nconmax=128,
            njmax=512,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
        ),
        decimation=4,
        episode_length_s=1.0e6,
        seed=0,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_ROOT,
            entity_name="robot",
            distance=2.6,
            azimuth=135.0,
            elevation=-15.0,
            fovy=45.0,
            width=960,
            height=720,
            max_extra_envs=0,
            geom_group=(1, 1, 1, 0, 0, 0),
        ),
    )


def make_preview_env(
    asset_dir: Path | None = None, num_envs: int = 1, device: str = "cuda:0"
) -> ManagerBasedRlEnv:
    from mjlab.envs import ManagerBasedRlEnv

    return ManagerBasedRlEnv(
        make_preview_env_cfg(asset_dir, num_envs), device=device, render_mode="rgb_array"
    )


def _assert_finite(env: ManagerBasedRlEnv) -> None:
    import torch

    for field in ("qpos", "qvel", "qacc"):
        assert torch.isfinite(getattr(env.sim.data, field)[:]).all(), f"nonfinite {field}"


def _run_viewer(env: ManagerBasedRlEnv, name: str) -> None:
    import torch
    from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

    action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    server = None
    try:
        env.reset()
        if name == "native":
            viewer = NativeMujocoViewer(env, lambda _: action)
        else:
            import viser

            server = viser.ViserServer(host="127.0.0.1", port=8080, label="Khan mini / mjlab")
            viewer = ViserPlayViewer(env, lambda _: action, viser_server=server)
        # Begin with the imported pose available for inspection. Users can
        # resume the PD simulation with the viewer's play/pause control.
        viewer.pause()
        # BaseViewer.run closes the viewer in its own finally block.
        viewer.run()
    finally:
        if server is not None:
            server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output", type=Path, default=ROOT / "logs/khan_mini/preview.png")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--headless", action="store_true", help="Finite check only (default)")
    mode.add_argument("--viewer", choices=("native", "viser"), help="Interactive after check")
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 0:
        parser.error("--num-envs must be positive and --steps must be nonnegative")
    # Select the GL backend before mjlab/MuJoCo captures this environment variable.
    os.environ.setdefault("MUJOCO_GL", "glfw" if args.viewer == "native" else "egl")

    import numpy as np
    import torch
    from PIL import Image

    from supermimic.robots.khan_mini import load_profile

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output.with_suffix(".json")
    report = {"status": "running", "image": str(output), "render_state": "initial_pose"}
    env = None
    started = time.perf_counter()
    try:
        profile = load_profile(asset_dir=args.asset_dir)
        env = make_preview_env(args.asset_dir, args.num_envs, args.device)
        env.reset()
        _assert_finite(env)
        expected = len(profile["joint_names"])
        model = env.sim.mj_model
        assert env.action_manager.total_action_dim == expected
        assert (model.nq, model.nv, model.nu) == (expected + 7, expected + 6, expected)
        robot = env.scene["robot"]
        initial_height = float(robot.data.root_link_pos_w[:, 2].min())
        # mjlab's OffscreenRenderer uses a CPU MuJoCo model synchronized from Warp.
        rgb = env.render()
        assert rgb is not None and rgb.ndim == 3 and rgb.shape[2] == 3
        assert np.isfinite(rgb).all()
        Image.fromarray(rgb).save(output)
        report.update(
            device=str(env.device),
            num_envs=env.num_envs,
            joint_names=list(robot.joint_names),
            nq=model.nq,
            nv=model.nv,
            nu=model.nu,
            nbody=model.nbody,
            ngeom=model.ngeom,
            nmesh=model.nmesh,
            physics_dt=env.physics_dt,
            control_dt=env.step_dt,
            initial_root_height=initial_height,
            render_shape=list(rgb.shape),
            steps_requested=args.steps,
            steps_completed=0,
        )
        action = torch.zeros(env.num_envs, expected, device=env.device)
        min_height = initial_height
        with torch.no_grad():
            for step in range(args.steps):
                observations, rewards, *_ = env.step(action)
                _assert_finite(env)
                assert torch.isfinite(rewards).all()
                assert all(torch.isfinite(value).all() for value in observations.values())
                min_height = min(min_height, float(robot.data.root_link_pos_w[:, 2].min()))
                report["steps_completed"] = step + 1
        report.update(
            finite=True,
            min_root_height_during_check=min_height,
            final_root_height=float(robot.data.root_link_pos_w[:, 2].min()),
            max_abs_qvel_final=float(env.sim.data.qvel[:].abs().max()),
            status="passed",
            note="Zero relative joint targets test PD/import only; no balance policy is applied.",
        )
        report["check_wall_time_seconds"] = time.perf_counter() - started
        summary_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"[PASS] {args.steps} mjlab steps; image: {output}; summary: {summary_path}")
        if args.viewer is not None:
            _run_viewer(env, args.viewer)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if env is not None:
            env.close()  # Also closes the CPU MuJoCo offscreen renderer.
        summary_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

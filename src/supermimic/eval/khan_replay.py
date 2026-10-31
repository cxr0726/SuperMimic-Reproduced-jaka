"""Kinematic Khan NPZ replay in mjlab, without a policy or physics integration.

Each displayed frame overwrites the free root and all 27 joint states, then
calls forward kinematics. The source trajectory is never shifted vertically.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_NPZ = ROOT / "data/npz/khan_mini/lafan1/walk1_subject1.npz"


def make_replay_env(asset_dir=None, *, fps: float = 50.0, geometry: str = "both", device="cuda:0"):
    """Create a one-world mjlab scene. No manager step is used by replay."""
    from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, mdp
    from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    from mjlab.scene import SceneCfg
    from mjlab.sim import MujocoCfg, SimulationCfg
    from mjlab.terrains import TerrainEntityCfg
    from mjlab.viewer import ViewerConfig

    from supermimic.robots.khan_mini import get_khan_mini_action_cfg, get_khan_mini_cfg

    if geometry not in ("visual", "collision", "both"):
        raise ValueError("geometry must be visual, collision or both")
    groups = (1, 1, int(geometry != "collision"), int(geometry != "visual"), 0, 0)
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(num_envs=1, terrain=TerrainEntityCfg(terrain_type="plane"),
                       entities={"robot": get_khan_mini_cfg(asset_dir)}),
        observations={"policy": ObservationGroupCfg(
            terms={"joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel)},
            concatenate_terms=True, enable_corruption=False,
        )},
        actions={"joint_pos": get_khan_mini_action_cfg(asset_dir, noise_std=0)},
        sim=SimulationCfg(nconmax=128, njmax=512,
                          mujoco=MujocoCfg(timestep=1 / fps / 4, iterations=10)),
        decimation=4, episode_length_s=1e6, seed=0,
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_ROOT,
                            entity_name="robot", distance=2.5, elevation=-8,
                            azimuth=135, fovy=45, width=960, height=720,
                            max_extra_envs=0, geom_group=groups,
                            enable_reflections=False),
    )
    return ManagerBasedRlEnv(cfg, device=device, render_mode="rgb_array")


class KinematicReplayEnv:
    """Viewer-compatible adapter whose ``step`` selects the next stored frame.

    The wrapped ManagerBasedRlEnv.step and Simulation.step are never called.
    Exposing ``unwrapped`` lets mjlab's native/Viser viewers synchronize their
    usual scene and rendering resources from this real mjlab environment.
    """

    def __init__(self, npz_path, asset_dir=None, *, start_frame=0, end_frame=None,
                 loop=False, geometry="both", tolerance_m=0.001, device="cuda:0"):
        from supermimic.data.foot_clearance import load_npz_motion
        from supermimic.robots.khan_mini import load_profile

        self.path = Path(npz_path).resolve()
        self.motion = load_npz_motion(self.path, asset_dir)
        for key in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
            if key not in self.motion or not np.isfinite(self.motion[key]).all():
                raise ValueError(f"NPZ needs finite {key}")
        self.fps = float(self.motion["fps"].reshape(-1)[0])
        self.total_frames = len(self.motion["joint_pos"])
        self.start_frame = int(start_frame)
        self.end_frame = self.total_frames if end_frame is None else int(end_frame)
        if not 0 <= self.start_frame < self.end_frame <= self.total_frames:
            raise ValueError("Need 0 <= start_frame < end_frame <= source frame count")
        self.loop = bool(loop)
        self.tolerance_m = float(tolerance_m)
        if not np.isfinite(self.tolerance_m) or self.tolerance_m < 0:
            raise ValueError("tolerance_m must be finite and nonnegative")
        self.geometry = geometry
        self.profile = load_profile(asset_dir)
        self.frame = self.start_frame
        self.finished = False
        self.forward_count = 0
        self._env = make_replay_env(asset_dir, fps=self.fps, geometry=geometry, device=device)
        try:
            self._initialize_scene()
        except Exception:
            self._env.close()
            raise

    def _initialize_scene(self) -> None:
        import mujoco
        import torch

        from supermimic.data.foot_clearance import foot_geom_groups, mesh_support_vertices

        self.robot = self._env.scene["robot"]
        self._cpu_model = self._env.sim.mj_model
        self._cpu_data = mujoco.MjData(self._cpu_model)
        self.foot_groups = foot_geom_groups(self._cpu_model, self.profile)
        ids = {g for group in self.foot_groups.values() for values in group.values() for g in values}
        self._mesh_vertices = {
            g: mesh_support_vertices(self._cpu_model, g)
            for g in ids if self._cpu_model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
        }
        root_id = self.motion["body_names"].tolist().index(self.profile["root_body"])
        root = np.concatenate((self.motion["body_pos_w"][:, root_id],
                               self.motion["body_quat_w"][:, root_id],
                               self.motion["body_lin_vel_w"][:, root_id],
                               self.motion["body_ang_vel_w"][:, root_id]), axis=1)
        self._root = torch.as_tensor(root, dtype=torch.float32, device=self.device)
        self._joint_pos = torch.as_tensor(self.motion["joint_pos"], device=self.device)
        self._joint_vel = torch.as_tensor(self.motion["joint_vel"], device=self.device)
        self._base_rgba = self._cpu_model.geom_rgba.copy()
        self.clearance = {}
        self.seek(self.start_frame)

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def unwrapped(self):
        return self._env

    def seek(self, frame: int) -> None:
        import mujoco

        from supermimic.data.foot_clearance import geom_bottom_z

        frame = int(np.clip(frame, self.start_frame, self.end_frame - 1))
        self.frame = frame
        self.finished = frame == self.end_frame - 1 and not self.loop
        root = self._root[frame:frame + 1].clone()
        root[:, :3] += self._env.scene.env_origins
        self.robot.write_root_state_to_sim(root)
        self.robot.write_joint_state_to_sim(self._joint_pos[frame:frame + 1],
                                            self._joint_vel[frame:frame + 1])
        # Forward only: never env.step(), sim.step() or elapsed-time integration.
        self._env.sim.forward()
        self._env.scene.update(1 / self.fps)
        self.forward_count += 1
        self._cpu_data.qpos[:] = self._env.sim.data.qpos[0].cpu().numpy()
        mujoco.mj_kinematics(self._cpu_model, self._cpu_data)
        self.clearance = {
            foot: {
                kind: min(geom_bottom_z(self._cpu_model, self._cpu_data, gid,
                                       mesh_vertices=self._mesh_vertices.get(gid))
                          for gid in gids)
                for kind, gids in group.items()
            } for foot, group in self.foot_groups.items()
        }
        self._update_render_colors()

    def _update_render_colors(self) -> None:
        """Only CPU render copies change; collision parameters/Warp state do not."""
        import mujoco

        models = [self._cpu_model]
        renderer = getattr(self._env, "_offline_renderer", None)
        if renderer is not None:
            # mjlab 1.6 keeps a separate CPU render model; these are display-only fields.
            models.append(renderer._model)
        for model in models:
            model.geom_rgba[:] = self._base_rgba
            visual = model.geom_group == 2
            collision = model.geom_group == 3
            model.geom_rgba[visual, 3] = 0.35 if self.geometry == "both" else 1
            model.geom_rgba[collision] = [0.95, 0.57, 0.12, 0.8]
            for foot, group in self.foot_groups.items():
                for kind, ids in group.items():
                    bad = self.clearance[foot][kind] <= -self.tolerance_m
                    color = [1.0, 0.08, 0.04] if bad else [0.1, 0.65, 1.0]
                    model.geom_rgba[ids, :3] = color
            for gid in np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE):
                model.geom_rgba[gid, 3] = 0.3
                material = int(model.geom_matid[gid])
                if material >= 0:
                    model.mat_rgba[material, 3] = 0.3

    def get_observations(self):
        return self._env.get_observations()

    def reset(self):
        self.seek(self.start_frame)
        return self.get_observations(), {}

    def step(self, actions):
        import torch

        del actions
        next_frame = self.frame + 1
        if next_frame >= self.end_frame:
            next_frame = self.start_frame if self.loop else self.end_frame - 1
        self.seek(next_frame)
        zero = torch.zeros(1, device=self.device)
        done = torch.tensor([self.finished], device=self.device)
        return self.get_observations(), zero, done, torch.zeros_like(done), {"frame": self.frame}

    def hud_lines(self) -> list[str]:
        lines = [f"KINEMATIC REPLAY | {self.path.stem}",
                 f"Source frame {self.frame}/{self.total_frames - 1} | t={self.frame/self.fps:.3f}s | ground z=0"]
        for foot, values in self.clearance.items():
            side = "Left" if foot.startswith("Left") else "Right"
            lines.append(f"{side} bottom: collision {values['collision']*1000:+.2f} mm | visual {values['visual']*1000:+.2f} mm")
        lines.append("Red foot = below tolerance | transparent ground | no physics integration or height offset")
        return lines

    def render_image(self) -> np.ndarray:
        from PIL import Image, ImageDraw, ImageFont

        rgb = self._env.render()
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rectangle((0, 0, image.width, 132), fill=(10, 15, 25, 220))
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 16)
        except OSError:
            font = ImageFont.load_default(size=16)
        for i, line in enumerate(self.hud_lines()):
            draw.text((12, 8 + i * 24), line, font=font, fill=(245, 245, 245, 255))
        return np.asarray(image)

    def close(self):
        self._env.close()


def run_viewer(replay: KinematicReplayEnv, kind: str, *, speed=1.0, paused=True, max_frames=None):
    """Use the standard mjlab viewers with motion-specific seek/HUD controls."""
    import mujoco
    import torch
    from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
    from mjlab.viewer.base import ViewerAction

    class ReplayNativeViewer(NativeMujocoViewer):
        def setup(self):
            super().setup()
            with self.viewer.lock():
                self.viewer.opt.geomgroup[:] = self.cfg.geom_group

        def _safe_key_callback(self, key):
            if key in (262, 263, 268):  # GLFW right, left, home.
                target = replay.start_frame if key == 268 else replay.frame + (1 if key == 262 else -1)
                self.request_action("SEEK", target)
            else:
                super()._safe_key_callback(key)

        def _handle_custom_action(self, action, payload):
            if action == ViewerAction.CUSTOM:
                self.pause()
                replay.seek(int(payload))
                return True
            return super()._handle_custom_action(action, payload)

        def _set_status_overlay(self, viewer):
            text = "\n".join(replay.hud_lines() + ["Space: play/pause | Left/Right: frame | Home: first frame"])
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100.value,
                             mujoco.mjtGridPos.mjGRID_TOPLEFT.value, text, ""))

        def sync_env_to_viewer(self):
            if replay.finished:
                self.pause()
            super().sync_env_to_viewer()

    class ReplayViserViewer(ViserPlayViewer):
        def setup(self):
            super().setup()
            self._scene.geom_groups_visible[:] = list(self.cfg.geom_group)
            self._last_clearance_flags = None
            with self._server.gui.add_folder("NPZ kinematic replay"):
                self._motion_slider = self._server.gui.add_slider(
                    "Source frame", min=replay.start_frame, max=replay.end_frame - 1,
                    step=1, initial_value=replay.frame,
                )
                self._motion_status = self._server.gui.add_html("")

            @self._motion_slider.on_update
            def seek(event):
                # Programmatic slider updates have no client; do not enqueue them.
                if event.client is not None:
                    self.request_action("SEEK", int(self._motion_slider.value))

        def _handle_custom_action(self, action, payload):
            if action == ViewerAction.CUSTOM:
                self.pause()
                replay.seek(int(payload))
                self._scene.needs_update = True
                return True
            return super()._handle_custom_action(action, payload)

        def sync_env_to_viewer(self):
            if replay.finished:
                self.pause()
            self._motion_slider.value = replay.frame
            self._motion_status.content = "<br/>".join(html.escape(line) for line in replay.hud_lines())
            flags = tuple(value <= -replay.tolerance_m
                          for foot in replay.clearance.values() for value in foot.values())
            if flags != self._last_clearance_flags:
                self._scene.mj_model.geom_rgba[:] = replay._cpu_model.geom_rgba
                self._scene.rebuild_visual_handles()
                self._last_clearance_flags = flags
            super().sync_env_to_viewer()

    action = torch.zeros(1, 27, device=replay.device)
    server = None
    try:
        if kind == "native":
            viewer = ReplayNativeViewer(replay, lambda _: action, enable_perturbations=False)
        else:
            import viser

            server = viser.ViserServer(host="127.0.0.1", port=8080, label="Khan NPZ kinematic replay")
            viewer = ReplayViserViewer(replay, lambda _: action, viser_server=server)
        viewer._time_multiplier = speed
        viewer._speed_index = int(np.argmin(np.abs(np.asarray(viewer.SPEED_MULTIPLIERS) - speed)))
        if paused:
            viewer.pause()
        bounded_steps = max_frames
        if bounded_steps is not None and not replay.loop:
            bounded_steps = min(bounded_steps, max(1, replay.end_frame - replay.start_frame - 1))
        viewer.run(num_steps=bounded_steps)
    finally:
        if server is not None:
            server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--asset-dir", type=Path)
    parser.add_argument("--viewer", choices=("native", "viser", "none"), default="native")
    parser.add_argument("--headless", action="store_true", help="Alias for --viewer none")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, help="Exclusive source frame bound")
    parser.add_argument("--max-frames", type=int, help="Bound rendered/played frame advances")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--paused", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--geometry", choices=("visual", "collision", "both"), default="both")
    parser.add_argument("--tolerance-m", type=float, default=0.001)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--video", type=Path, nargs="?", const=Path("replay.mp4"),
                        help="Save MP4 during headless replay; relative to output-dir")
    args = parser.parse_args()
    if args.headless:
        args.viewer = "none"
    if not np.isfinite(args.speed) or args.speed <= 0 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("speed and max-frames must be positive")
    os.environ.setdefault("MUJOCO_GL", "glfw" if args.viewer == "native" else "egl")
    from PIL import Image

    from supermimic.data.foot_clearance import audit_npz

    output = (args.output_dir or ROOT / "logs/khan_replay" / args.npz.stem).resolve()
    output.mkdir(parents=True, exist_ok=True)
    audit = audit_npz(args.npz, args.asset_dir, tolerance_m=args.tolerance_m)
    per_frame = audit.pop("_per_frame")
    (output / "ground_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    replay = None
    video = None
    report = {"status": "running", "mode": "kinematic", "physics_integration_steps": 0}
    try:
        replay = KinematicReplayEnv(args.npz, args.asset_dir, start_frame=args.start_frame,
                                     end_frame=args.end_frame, loop=args.loop,
                                     geometry=args.geometry, tolerance_m=args.tolerance_m,
                                     device=args.device)
        report.update(npz=str(args.npz.resolve()), fps=replay.fps,
                      start_frame=replay.start_frame, end_frame_exclusive=replay.end_frame,
                      geometry=args.geometry, ground_z_m=0.0, height_offset_m=0.0)
        Image.fromarray(replay.render_image()).save(output / "first_frame.png")
        audit_frames = np.asarray(per_frame["frame"])
        scope = (audit_frames >= replay.start_frame) & (audit_frames < replay.end_frame)
        worst_frames = {}
        for kind in ("collision", "visual"):
            values = np.asarray(per_frame[f"both_feet_{kind}_min_z_m"])
            indices = np.flatnonzero(scope)
            index = indices[int(np.argmin(values[scope]))]
            frame = int(audit_frames[index])
            worst_frames[kind] = {"frame": frame, "bottom_z_m": float(values[index])}
            replay.seek(frame)
            Image.fromarray(replay.render_image()).save(output / f"worst_{kind}_frame.png")
        replay.reset()
        report["worst_frames"] = worst_frames
        if args.viewer == "none":
            stop = min(replay.end_frame, replay.start_frame + (args.max_frames or replay.total_frames))
            if args.video is not None:
                import imageio_ffmpeg

                path = args.video if args.video.is_absolute() else output / args.video
                path.parent.mkdir(parents=True, exist_ok=True)
                video = imageio_ffmpeg.write_frames(str(path), (960, 720), fps=replay.fps * args.speed,
                                                    codec="libx264", macro_block_size=16)
                video.send(None)
                report["video"] = str(path.resolve())
            for frame in range(replay.start_frame, stop):
                replay.seek(frame)
                if video is not None:
                    video.send(replay.render_image())
            report["frames_replayed"] = stop - replay.start_frame
            report["last_frame"] = replay.frame
        else:
            run_viewer(replay, args.viewer, speed=args.speed, paused=args.paused,
                       max_frames=args.max_frames)
        report["forward_calls"] = replay.forward_count
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if video is not None:
            video.close()
        if replay is not None:
            replay.close()
        (output / "replay_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"[PASS] kinematic replay; no integration; outputs: {output}")


if __name__ == "__main__":
    main()

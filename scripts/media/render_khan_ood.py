"""Render complete Khan OOD recordings as policy/reference comparison videos.

The left and right panels use identical camera directions and independently
follow their robot's root so both full poses remain visible. This display does
not align trajectories: the original XY discrepancy remains in the recording
and is printed on every frame. No physics or policy inference is run here.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import tyro
from PIL import Image, ImageDraw, ImageFont
from render_rollout import load_rollout_model


@lru_cache(maxsize=12)
def font(size: int, bold: bool = False):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{filename}", size)
    except OSError:
        return ImageFont.load_default()


def ffmpeg_binary() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def read_recording(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        recording = {key: archive[key] for key in archive.files}
    required = ("names", "robot_qpos", "ref_qpos", "lengths", "norm_return")
    missing = [key for key in required if key not in recording]
    if missing:
        raise ValueError(f"{path}: missing {missing}")
    robot = str(np.asarray(recording.get("robot", "khan_mini")).item())
    if robot != "khan_mini":
        raise ValueError("This renderer expects Khan_mini recording metadata")
    robot_qpos = recording["robot_qpos"]
    if robot_qpos.ndim != 3 or robot_qpos.shape[-1] != 34:
        raise ValueError("robot_qpos must have shape [clip, frame, 34]")
    if recording["ref_qpos"].shape != robot_qpos.shape:
        raise ValueError("robot/reference qpos shapes differ")
    count, frames, _ = robot_qpos.shape
    if recording["names"].shape != (count,) or recording["lengths"].shape != (count,):
        raise ValueError("Names/lengths do not match the number of clips")
    if len(set(recording["names"].tolist())) != count:
        raise ValueError("Clip names must be unique")
    if recording["norm_return"].shape != (count,):
        raise ValueError("norm_return must contain one value per clip")
    source_fps = float(np.asarray(recording.get("fps", 50.0)).item())
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("Recording fps must be finite and positive")
    for i, value in enumerate(recording["lengths"]):
        length = int(value)
        if value != length or not 1 <= length <= frames:
            raise ValueError(f"Invalid recording length for clip {i}")
        for key in ("robot_qpos", "ref_qpos"):
            valid = recording[key][i, :length]
            if not np.isfinite(valid).all():
                raise ValueError(f"Nonfinite valid poses in {key}, clip {i}")
            if not np.allclose(np.linalg.norm(valid[:, 3:7], axis=-1), 1., atol=1e-3):
                raise ValueError(f"Nonunit quaternion in {key}, clip {i}")
    if "failure_steps" not in recording:
        recording["failure_steps"] = np.full(count, -1, dtype=np.int64)
    if recording["failure_steps"].shape != (count,):
        raise ValueError("failure_steps must contain one value per clip")
    for i, step in enumerate(recording["failure_steps"]):
        if step < -1 or step >= int(recording["lengths"][i]):
            raise ValueError(f"Invalid failure step for clip {i}")
    for key in ("trace_xy_error", "trace_joint_rmse"):
        if key in recording and recording[key].shape != robot_qpos.shape[:2]:
            raise ValueError(f"{key} must have shape [clip, frame]")
    return recording


def frame_indices(length: int, source_fps: float, output_fps: int) -> np.ndarray:
    """Preserve wall-clock length within one output frame and display terminal."""
    duration = (length - 1) / source_fps
    count = max(1, math.ceil(duration * output_fps - 1e-10))
    indices = np.rint(np.arange(count) * source_fps / output_fps).astype(np.int64)
    indices = np.clip(indices, 0, length - 1)
    indices[-1] = length - 1
    return indices


def recording_mode(recording: dict[str, np.ndarray], index: int) -> str:
    value = np.asarray(recording.get("mode", "benchmark_stop"))
    return str(value.item() if value.ndim == 0 else value[index])


def status_text(mode: str, failed: bool) -> tuple[str, tuple[int, int, int]]:
    if failed and mode == "continuous_no_fall_reset":
        return "Height/tilt threshold crossed; continued WITHOUT reset", (255, 157, 126)
    if failed:
        return "TERMINATED: terminal pose held; no further policy steps", (255, 157, 126)
    if mode == "continuous_no_fall_reset":
        return "Tracking continuously; fall auto-reset disabled", (240, 219, 161)
    return "Tracking with original benchmark termination rules", (170, 233, 202)


def metric_value(recording: dict[str, np.ndarray], key: str, clip: int,
                 frame: int, fallback: float) -> float:
    value = float(recording[key][clip, frame]) if key in recording else fallback
    return value if math.isfinite(value) else fallback


def overlay(panels: list[np.ndarray], name: str, source_time: float,
            duration: float, score: float, mode: str, failed: bool,
            failure_time: float, xy: float, joint: float, width: int, height: int,
            header: int, footer: int) -> np.ndarray:
    canvas = Image.new("RGB", (width, height), (22, 29, 40))
    gap = 4
    panel_width = panels[0].shape[1]
    canvas.paste(Image.fromarray(panels[0]), (0, header))
    canvas.paste(Image.fromarray(panels[1]), (panel_width+gap, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 8), f"{name} | Khan 100k OOD | t={source_time:.2f}/{duration:.2f} s",
              font=font(24, True), fill=(242, 246, 251))
    status, color = status_text(mode, failed)
    if failed:
        status += f" @ {failure_time:.2f}s"
    draw.text((16, 40), status, font=font(17), fill=color)
    for x, title in ((0, "POLICY: simulated robot"), (panel_width+gap, "REFERENCE: kinematic target")):
        draw.rounded_rectangle((x+12, header+10, x+panel_width-12, header+43),
                               fill=(22, 29, 40, 235), radius=6)
        draw.text((x+24, header+14), title, font=font(20, True), fill=(246, 248, 252))
    y = height-footer+10
    draw.text((16, y), f"XY position error: {xy:.3f} m   |   joint RMSE: {joint:.3f} rad"
              f"   |   recorded normalized return: {score:.3f}",
              font=font(20), fill=(235, 241, 249))
    draw.text((16, y+31), "Identical camera direction; each panel follows its own root. "
              "No trajectory alignment or reference-height adjustment.",
              font=font(16), fill=(190, 205, 222))
    return np.asarray(canvas)


def main(traj: Path, out_dir: Path, fps: int = 25, width: int = 1280,
         height: int = 720, azimuth: float = 140., elevation: float = -14.,
         distance: float = 2.7) -> None:
    if fps < 1 or width < 960 or height < 540 or width % 2 or height % 2:
        raise ValueError("Use positive fps and even video dimensions of at least 960x540")
    recording = read_recording(traj)
    source_fps = float(np.asarray(recording.get("fps", 50.)).item())
    header, footer, gap = 72, 72, 4
    panel_width, panel_height = (width-gap)//2, height-header-footer
    model = load_rollout_model("khan_mini")
    if model.nq != 68:
        raise ValueError("Expected the current two-tree 27-joint Khan display model")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, panel_height, panel_width)
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[0] = 1
    option.geomgroup[2] = 1
    option.flags[mujoco.mjtVisFlag.mjVIS_LIGHT] = 1
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation, camera.distance = azimuth, elevation, distance
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        for clip, raw_name in enumerate(recording["names"]):
            name = str(raw_name)
            if Path(name).name != name:
                raise ValueError("Clip names must not contain path components")
            length = int(recording["lengths"][clip])
            duration = (length-1)/source_fps
            indices = frame_indices(length, source_fps, fps)
            mode = recording_mode(recording, clip)
            failure_step = int(recording["failure_steps"][clip])
            policy = recording["robot_qpos"][clip, :length]
            reference = recording["ref_qpos"][clip, :length]
            target = [policy[0, :3].copy(), reference[0, :3].copy()]
            out = out_dir / f"{name}.mp4"
            command = [ffmpeg_binary(), "-y", "-loglevel", "error", "-f", "rawvideo",
                       "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
                       "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                       "-crf", "19", "-movflags", "+faststart", str(out)]
            process = subprocess.Popen(command, stdin=subprocess.PIPE)
            try:
                for source_frame in indices:
                    panels = []
                    for panel, poses in enumerate((policy, reference)):
                        pose = poses[source_frame]
                        data.qpos[:34] = pose
                        data.qpos[34:] = pose
                        data.qpos[34:37] = [0., 0., -10.]
                        # mj_forward only updates display transforms here; this
                        # file never steps the simulated rollout's dynamics.
                        mujoco.mj_forward(model, data)
                        target[panel] = .85*target[panel]+.15*pose[:3]
                        camera.lookat[:] = [target[panel][0], target[panel][1], .55]
                        renderer.update_scene(data, camera, option)
                        panels.append(renderer.render().copy())
                    xy = metric_value(recording, "trace_xy_error", clip, source_frame,
                                      float(np.linalg.norm(policy[source_frame, :2]
                                                           - reference[source_frame, :2])))
                    joint = metric_value(recording, "trace_joint_rmse", clip, source_frame,
                                         float(np.sqrt(np.mean((policy[source_frame, 7:]
                                                               - reference[source_frame, 7:])**2))))
                    image = overlay(
                        panels, name, source_frame/source_fps, duration,
                        float(recording["norm_return"][clip]), mode,
                        failure_step >= 0 and source_frame >= failure_step,
                        failure_step/source_fps, xy, joint,
                        width, height, header, footer,
                    )
                    process.stdin.write(np.ascontiguousarray(image).tobytes())
            except BaseException:
                process.kill()
                process.wait()
                raise
            finally:
                if process.stdin is not None:
                    try:
                        process.stdin.close()
                    except BrokenPipeError:
                        pass
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
            result = {"name": name, "path": str(out.resolve()), "mode": mode,
                      "source_pose_frames": length, "source_duration_s": duration,
                      "video_frames": len(indices), "video_fps": fps,
                      "video_duration_s": len(indices)/fps,
                      "final_source_frame_shown": int(indices[-1]),
                      "failure_step": failure_step}
            results.append(result)
            print(json.dumps(result), flush=True)
    finally:
        renderer.close()
    (out_dir / "render_manifest.json").write_text(json.dumps({
        "trajectory": str(traj.resolve()), "width": width, "height": height,
        "panels": "policy left, reference right; separately followed roots; no alignment",
        "render_only": True, "clips": results,
    }, indent=2)+"\n")


if __name__ == "__main__":
    tyro.cli(main)

"""Render one-minute 32k/40k comparisons around recorded 32k terminations.

Only saved poses are drawn; there is no policy loading, simulation stepping or
middle-of-motion reset. The explicit frozen asset directory supplies both visual
trees. A failed panel holds its terminal robot, reference and camera.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import tyro
from PIL import Image, ImageDraw
from render_khan_checkpoint_comparison import (
    checkpoint,
    ffmpeg_binary,
    font,
    read_recording,
    shared_camera_roots,
)
from render_rollout import two_robot_xml


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def window_bounds(failure_s: float, available_s: float, seconds: float, before_s: float) -> tuple[float, float]:
    """Keep the full requested duration and move earlier near the real clip end."""
    if failure_s < 0 or failure_s > available_s or available_s < seconds:
        raise ValueError("A full window containing the left termination must be available")
    start = max(0.0, min(failure_s - before_s, available_s - seconds))
    end = start + seconds
    if not start <= failure_s <= end:
        raise ValueError("The requested pre-failure duration excludes the termination")
    return start, end


def panel_overlay(pixels: np.ndarray, iteration: int, absolute_s: float, failure_s: float) -> Image.Image:
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image, "RGBA")
    failed = failure_s >= 0 and absolute_s + 1e-8 >= failure_s
    draw.rounded_rectangle((16, 14, image.width - 16, 106), radius=10, fill=(18, 25, 36, 220))
    draw.text((32, 23), f"{iteration:,} iterations", font=font(28, True), fill=(255, 255, 255))
    if failed:
        status = f"TERMINATED at motion t={failure_s:.2f} s"
        color = (255, 151, 132)
    elif failure_s >= 0:
        status = f"Tracking | termination in {failure_s - absolute_s:.2f} s"
        color = (255, 218, 157)
    else:
        status = "Tracking | full-clip rollout completed successfully"
        color = (176, 231, 202)
    draw.text((32, 66), status, font=font(20), fill=color)
    if failed:
        draw.rounded_rectangle((16, image.height - 58, image.width - 16, image.height - 14),
                               radius=8, fill=(131, 36, 40, 225))
        draw.text((30, image.height - 50), "Terminal pose + reference + camera FROZEN | no reset",
                  font=font(20), fill=(255, 239, 234))
    return image


def canvas(panels: list[Image.Image], name: str, absolute_s: float, video_s: float,
           seconds: float, start_s: float, failure_s: float) -> Image.Image:
    gap, header, footer = 4, 64, 72
    width = panels[0].width * 2 + gap
    output = Image.new("RGB", (width, panels[0].height + header + footer), (22, 29, 40))
    output.paste(panels[0], (0, header))
    output.paste(panels[1], (panels[0].width + gap, header))
    draw = ImageDraw.Draw(output)
    draw.text((24, 12), f"Khan mini | {name}", font=font(25, True), fill=(246, 248, 252))
    clock = f"Motion t={absolute_s:06.2f}s | Video {video_s:05.2f}/{seconds:.0f}s"
    draw.text((width - draw.textlength(clock, font=font(23)) - 24, 14), clock,
              font=font(23), fill=(246, 248, 252))
    bar_y = header - 10
    draw.rectangle((24, bar_y, width - 24, bar_y + 4), fill=(67, 81, 99))
    draw.rectangle((24, bar_y, 24 + (width - 48) * video_s / seconds, bar_y + 4), fill=(103, 173, 223))
    tick = 24 + (width - 48) * (failure_s - start_s) / seconds
    draw.rectangle((tick - 2, bar_y - 4, tick + 2, bar_y + 7), fill=(255, 136, 113))
    draw.text((24, output.height - 61), "LEFT: 32k | RIGHT: 40k | Solid: simulated policy | Blue transparent: motion reference",
              font=font(21), fill=(220, 231, 243))
    draw.text((24, output.height - 31),
              f"Recorded from motion start; shown window {start_s:.2f}-{start_s + seconds:.2f}s | Failed panel holds terminal pose and camera",
              font=font(18), fill=(184, 204, 226))
    return output


def main(left32k: Path, right40k: Path, asset_dir: Path, out_dir: Path,
         fps: int = 25, seconds: float = 60.0, before_seconds: float = 45.0,
         panel_width: int = 960, panel_height: int = 640, distance: float = 2.8,
         azimuth: float = 140.0, elevation: float = -15.0, clips: list[str] | None = None) -> None:
    if fps <= 0 or seconds <= 0 or not 0 <= before_seconds <= seconds:
        raise ValueError("Use positive fps/duration with 0 <= before-seconds <= seconds")
    if panel_width < 800 or panel_height < 480 or panel_width % 2 or panel_height % 2:
        raise ValueError("Even panel dimensions of at least 800x480 are required")
    left, right = read_recording(left32k), read_recording(right40k)
    source_fps = float(left.get("fps", 50.0))
    if source_fps != float(right.get("fps", 50.0)):
        raise ValueError("Recording sample rates differ")
    if checkpoint(left, -1) != 32000 or checkpoint(right, -1) != 40000:
        raise ValueError("Expected left checkpoint32000 and right checkpoint40000")
    left_names, right_names = [str(x) for x in left["names"]], [str(x) for x in right["names"]]
    selected = clips or left_names
    if not set(selected) <= set(left_names) & set(right_names):
        raise ValueError("Requested clips must be present in both recordings")
    profile = json.loads((asset_dir / "profile.json").read_text())
    model = mujoco.MjModel.from_xml_string(two_robot_xml(asset_dir / "robot.xml", profile["root_body"]))
    nq = left["robot_qpos"].shape[-1]
    if model.nq != 2 * nq or right["robot_qpos"].shape[-1] != nq:
        raise ValueError("Frozen robot asset does not match recording qpos")
    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        if name.startswith("ref_"):
            model.geom_rgba[geom] = (0.12, 0.42, 1.0, 0.34)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, panel_width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, panel_height)
    model.vis.quality.shadowsize = 2048
    model.vis.quality.offsamples = 4
    renderer = mujoco.Renderer(model, panel_height, panel_width)
    data = mujoco.MjData(model)
    options = mujoco.MjvOption()
    options.geomgroup[:] = 0
    options.geomgroup[[0, 1, 2]] = 1
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation = azimuth, elevation
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    stills = []
    try:
        for name in selected:
            li, ri = left_names.index(name), right_names.index(name)
            length = min(int(left["lengths"][li]), int(right["lengths"][ri]))
            failure_step = int(left["failure_steps"][li])
            if failure_step < 0 or int(right["failure_steps"][ri]) >= 0:
                raise ValueError(f"{name}: require left failure and right full-clip success")
            right_end = int(right["terminal_steps"][ri])
            if right_end < 0:
                raise ValueError(f"{name}: right rollout must reach its clip-end timeout")
            available_s = min(length - 1, right_end) / source_fps
            failure_s = failure_step / source_fps
            start_s, end_s = window_bounds(failure_s, available_s, seconds, before_seconds)
            roots, camera_note = shared_camera_roots(left, right, li, ri, length)
            a, b = round(start_s * source_fps), round(end_s * source_fps)
            height = float(np.clip(np.median(roots[a:b + 1, 2]), 0.48, 0.72))
            # Use one constant distance in both panels. This envelope keeps a
            # drifting or crouching policy inside the view without following it.
            offsets = []
            for recording, index in ((left, li), (right, ri)):
                offset = recording["robot_qpos"][index, a:b + 1, :3].copy() - roots[a:b + 1]
                # The left camera freezes at termination just like its pose.
                if recording is left and b > failure_step:
                    offset[max(0, failure_step - a):] = (
                        recording["robot_qpos"][index, failure_step, :3] - roots[failure_step]
                    )
                offsets.append(float(np.linalg.norm(offset, axis=1).max()))
            fitted_distance = max(distance, (1.05 + max(offsets)) / np.sin(np.radians(model.vis.global_.fovy / 2)))
            camera.distance = fitted_distance
            output = out_dir / f"comparison_32k_vs_40k_{name}.mp4"
            frame_count = round(seconds * fps)
            snapshot_frame = min(frame_count - 1, int(np.ceil((failure_s - start_s) * fps)))
            width, total_height = 2 * panel_width + 4, panel_height + 136
            encoder = subprocess.Popen([
                ffmpeg_binary(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{total_height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
                "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
            ], stdin=subprocess.PIPE)
            try:
                for frame in range(frame_count):
                    video_s = frame / fps
                    absolute_s = start_s + video_s
                    step = min(length - 1, round(absolute_s * source_fps))
                    panels = []
                    for recording, index, iteration, fail in ((left, li, 32000, failure_step), (right, ri, 40000, -1)):
                        camera_step = min(step, fail) if fail >= 0 else step
                        root = roots[camera_step]
                        camera.lookat[:] = (root[0], root[1], height)
                        data.qpos[:nq] = recording["robot_qpos"][index, step]
                        data.qpos[nq:] = recording["ref_qpos"][index, step]
                        mujoco.mj_forward(model, data)
                        renderer.update_scene(data, camera, options)
                        panels.append(panel_overlay(renderer.render(), iteration, absolute_s, fail / source_fps if fail >= 0 else -1))
                    image = canvas(panels, name, absolute_s, video_s, seconds, start_s, failure_s)
                    if frame in {0, snapshot_frame, frame_count - 1}:
                        label = "start" if frame == 0 else "failure" if frame == snapshot_frame else "end"
                        image.save(out_dir / f"{label}_{name}.png")
                        if frame == snapshot_frame:
                            stills.append(image.copy())
                    assert encoder.stdin is not None
                    encoder.stdin.write(image.tobytes())
            finally:
                if encoder.stdin:
                    encoder.stdin.close()
                returncode = encoder.wait()
            if returncode:
                raise RuntimeError(f"Video encoder failed with exit code{returncode}")
            metadata.append({
                "name": name, "video": str(output.resolve()), "frames": frame_count,
                "fps": fps, "seconds": seconds, "size": [width, total_height],
                "motion_window_s": [start_s, end_s], "available_motion_s": available_s,
                "left_failure_s": failure_s, "failure_video_time_s": failure_s - start_s,
                "failure_snapshot_motion_s": start_s + snapshot_frame / fps,
                "right_failure_s": None, "right_completed_full_clip": True,
                "camera": camera_note, "camera_height": height, "distance": fitted_distance,
                "azimuth": azimuth, "elevation": elevation,
                "terminal_camera": "Left camera, robot and reference freeze at the first termination",
                "sha256": sha256(output),
            })
            print(f"Wrote {output} | motion {start_s:.2f}-{end_s:.2f}s | 32k failure {failure_s:.2f}s", flush=True)
    finally:
        renderer.close()
    if stills:
        tiles = [im.resize((962, 388), Image.Resampling.LANCZOS) for im in stills]
        sheet = Image.new("RGB", (1924, 388 * ((len(tiles) + 1) // 2)), (22, 29, 40))
        for i, im in enumerate(tiles):
            sheet.paste(im, (962 * (i % 2), 388 * (i // 2)))
        sheet.save(out_dir / "failure_contact_sheet.png")
    (out_dir / "comparison_manifest.json").write_text(json.dumps({
        "left_recording": str(left32k.resolve()), "left_sha256": sha256(left32k),
        "right_recording": str(right40k.resolve()), "right_sha256": sha256(right40k),
        "asset_dir": str(asset_dir.resolve()), "model_sha256": sha256(asset_dir / "robot.xml"),
        "left_iteration": 32000, "right_iteration": 40000,
        "note": "Rollouts begin at motion frame0; videos only crop saved recordings. No reset or policy resimulation occurs during rendering. Robot and reference freeze after termination. Captured reference follows recorder control-step convention (may lag absolute library index20ms); camera uses continuous reference root.",
        "clips": metadata,
    }, indent=2) + "\n")


if __name__ == "__main__":
    tyro.cli(main)

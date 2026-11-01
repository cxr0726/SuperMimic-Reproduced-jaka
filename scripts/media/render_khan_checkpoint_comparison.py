"""Render aligned, terminal-aware Khan checkpoint comparisons from saved qpos.

This script only renders recordings. It never loads a policy or steps a simulator.
Both panels use the same reference-root camera target until terminal. After terminal,
that panel holds its pose and camera, with an explicit label, while the other continues.
It displays the current visual meshes; recorded policy/physics results are not
recomputed, and this render does not identify the original collision assets.

Example::

    MUJOCO_GL=egl .venv/bin/python scripts/media/render_khan_checkpoint_comparison.py \
        --left logs/Khan_lafan1_4096/analysis/failure_videos/rollout_0040000.npz \
        --right logs/Khan_lafan1_4096/analysis/failure_videos/rollout_0200000.npz \
        --out-dir logs/Khan_lafan1_4096/analysis/failure_videos
"""

from __future__ import annotations

import json
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


def ffmpeg_binary() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def read_recording(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        recording = {key: archive[key] for key in archive.files}
    required = ("names", "robot_qpos", "ref_qpos", "lengths", "failure_steps")
    missing = [key for key in required if key not in recording]
    if missing:
        raise ValueError(f"{path}: missing {missing}")
    qpos = recording["robot_qpos"]
    if qpos.ndim != 3 or recording["ref_qpos"].shape != qpos.shape:
        raise ValueError(f"{path}: expected matching [clip, time, nq] qpos arrays")
    if len(recording["names"]) != qpos.shape[0]:
        raise ValueError(f"{path}: names and qpos clip counts differ")
    if not np.isfinite(qpos).all() or not np.isfinite(recording["ref_qpos"]).all():
        raise ValueError(f"{path}: non-finite qpos")
    source_fps = float(np.asarray(recording.get("fps", 50.0)).item())
    if not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(f"{path}: invalid fps")
    for i, length in enumerate(recording["lengths"]):
        if not 1 <= int(length) <= qpos.shape[1]:
            raise ValueError(f"{path}: invalid length for clip {i}")
        fail = int(recording["failure_steps"][i])
        if fail >= int(length) or fail < -1:
            raise ValueError(f"{path}: invalid failure step for clip {i}")
        if fail >= 0:
            for key in ("robot_qpos", "ref_qpos"):
                if not np.allclose(recording[key][i, fail:int(length)], recording[key][i, fail], atol=1e-6):
                    raise ValueError(f"{path}: {key} must stay frozen after terminal")
    return recording


@lru_cache(maxsize=12)
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{filename}", size)


def checkpoint(recording: dict[str, np.ndarray], fallback: int) -> int:
    return int(np.asarray(recording.get("checkpoint_iteration", fallback)).item())


def shared_camera_roots(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray], li: int, ri: int, length: int,
) -> tuple[np.ndarray, str]:
    for recording, index in ((left, li), (right, ri)):
        if "camera_ref_root" in recording:
            roots = recording["camera_ref_root"][index, :length]
            if roots.shape != (length, 3) or not np.isfinite(roots).all():
                raise ValueError("camera_ref_root must have finite [clip, time, 3] values")
            return roots, "continuous reference root (shared by both panels)"
    # Older recordings freeze reference qpos too. Use the side whose reference runs longer,
    # then hold that common camera position; never follow either policy's actual root.
    left_end = int(left["failure_steps"][li])
    right_end = int(right["failure_steps"][ri])
    left_end = length if left_end < 0 else left_end
    right_end = length if right_end < 0 else right_end
    recording, index = (left, li) if left_end >= right_end else (right, ri)
    return recording["ref_qpos"][index, :length, :3], "longer-running reference root (shared; holds after terminal)"


def panel_overlay(
    pixels: np.ndarray, iteration: int, time_s: float, failure_step: int, source_fps: float,
) -> Image.Image:
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image, "RGBA")
    failed = failure_step >= 0 and time_s + 1e-8 >= failure_step / source_fps
    draw.rounded_rectangle((16, 14, image.width - 16, 95), radius=10, fill=(18, 25, 36, 215))
    draw.text((32, 23), f"{iteration:,} iterations", font=font(27, True), fill=(255, 255, 255))
    if failed:
        status = f"TERMINATED at {failure_step / source_fps:.2f} s  |  pose & camera FROZEN"
        color = (255, 137, 120)
    else:
        status = "Tracking  |  no reset during this recording"
        color = (182, 225, 206)
    draw.text((32, 61), status, font=font(19), fill=color)
    if failed:
        draw.rounded_rectangle(
            (16, image.height - 52, image.width - 16, image.height - 16),
            radius=8, fill=(125, 33, 34, 210),
        )
        draw.text(
            (32, image.height - 47), "Terminal pose and camera held; no further steps / no auto-reset",
            font=font(18), fill=(255, 237, 233),
        )
    return image


def canvas(
    left: Image.Image, right: Image.Image, name: str, time_s: float, seconds: float,
) -> Image.Image:
    header, footer, gap = 60, 48, 4
    output = Image.new("RGB", (left.width + right.width + gap, left.height + header + footer), (22, 29, 40))
    output.paste(left, (0, header))
    output.paste(right, (left.width + gap, header))
    draw = ImageDraw.Draw(output)
    draw.text((24, 14), f"Khan mini  |  {name}", font=font(25, True), fill=(246, 248, 252))
    time_label = f"Time  {time_s:05.2f} s / {seconds:.2f} s"
    draw.text((output.width - draw.textlength(time_label, font=font(23)) - 24, 16), time_label, font=font(23), fill=(246, 248, 252))
    draw.text((24, output.height - 34), "Solid: simulated policy   |   Blue transparent: motion reference", font=font(19), fill=(221, 230, 242))
    camera_label = "Same camera before terminal; failed panel holds pose & camera"
    draw.text((output.width - draw.textlength(camera_label, font=font(19)) - 24, output.height - 34), camera_label, font=font(19), fill=(190, 206, 228))
    return output


def main(
    left: Path,
    right: Path,
    out_dir: Path,
    fps: int = 25,
    seconds: float = 12.0,
    panel_width: int = 960,
    panel_height: int = 640,
    azimuth: float = 140.0,
    elevation: float = -15.0,
    distance: float = 2.7,
    concat: bool = True,
) -> None:
    """Produce one side-by-side video and representative failure PNG per matching clip."""
    if fps <= 0 or seconds <= 0 or panel_width < 640 or panel_height < 480:
        raise ValueError("Use positive fps/duration and panels at least 640 x 480")
    if panel_height % 2 or panel_width % 2:
        raise ValueError("Even panel dimensions are required for yuv420p")
    left_data, right_data = read_recording(left), read_recording(right)
    source_fps = float(np.asarray(left_data.get("fps", 50.0)).item())
    if source_fps != float(np.asarray(right_data.get("fps", 50.0)).item()):
        raise ValueError("Recordings must have the same source frame rate")
    left_names = [str(value) for value in left_data["names"]]
    right_names = [str(value) for value in right_data["names"]]
    if set(left_names) != set(right_names):
        raise ValueError("Both recordings must contain the same clips")
    left_iteration, right_iteration = checkpoint(left_data, 40000), checkpoint(right_data, 200000)
    model = load_rollout_model("khan_mini")
    nq = left_data["robot_qpos"].shape[-1]
    if model.nq != nq * 2 or right_data["robot_qpos"].shape[-1] != nq:
        raise ValueError(f"Model nq={model.nq} incompatible with recording nq={nq}")
    # More legible than the original light ghost; keep geometry and alpha transparency.
    for geom in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        if geom_name.startswith("ref_"):
            model.geom_rgba[geom] = (0.12, 0.42, 1.0, 0.34)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, panel_width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, panel_height)
    renderer = mujoco.Renderer(model, panel_height, panel_width)
    render_data = mujoco.MjData(model)
    options = mujoco.MjvOption()
    options.geomgroup[:] = 0
    options.geomgroup[[0, 1, 2]] = 1  # floor, ghost, visual meshes; never collision proxies
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation, camera.distance = azimuth, elevation, distance
    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg_binary()
    outputs: list[Path] = []
    metadata: list[dict] = []
    try:
        for name in left_names:
            li, ri = left_names.index(name), right_names.index(name)
            length = min(int(left_data["lengths"][li]), int(right_data["lengths"][ri]))
            available_seconds = (length - 1) / source_fps
            if available_seconds + 1e-8 < seconds:
                raise ValueError(f"{name}: only {available_seconds:.2f}s recorded, need {seconds:.2f}s")
            camera_roots, camera_note = shared_camera_roots(left_data, right_data, li, ri, length)
            # Fixed reference-height framing, independent of policy falls or crouches.
            camera_height = float(np.clip(np.median(camera_roots[:, 2]), 0.48, 0.72))
            failure_left = int(left_data["failure_steps"][li])
            failure_right = int(right_data["failure_steps"][ri])
            if failure_right >= 0:
                snapshot_step = failure_right
            elif failure_left >= 0:
                snapshot_step = failure_left
            else:
                delta = right_data["robot_qpos"][ri, :length, :3] - right_data["ref_qpos"][ri, :length, :3]
                snapshot_step = int(np.argmax(np.linalg.norm(delta, axis=-1)))
            frame_count = round(seconds * fps)
            snapshot_frame = min(frame_count - 1, int(np.ceil(snapshot_step / source_fps * fps)))
            output = out_dir / f"comparison_{name}.mp4"
            width, height = 2 * panel_width + 4, panel_height + 108
            command = [
                ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(output),
            ]
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
            try:
                for frame in range(frame_count):
                    time_s = frame / fps
                    source_step = min(length - 1, round(time_s * source_fps))
                    panels = []
                    for recording, index, iteration, failure in (
                        (left_data, li, left_iteration, failure_left),
                        (right_data, ri, right_iteration, failure_right),
                    ):
                        camera_step = min(source_step, failure) if failure >= 0 else source_step
                        root = camera_roots[camera_step]
                        camera.lookat[:] = (root[0], root[1], camera_height)
                        render_data.qpos[:nq] = recording["robot_qpos"][index, source_step]
                        render_data.qpos[nq:] = recording["ref_qpos"][index, source_step]
                        mujoco.mj_forward(model, render_data)
                        renderer.update_scene(render_data, camera, options)
                        panels.append(panel_overlay(renderer.render(), iteration, time_s, failure, source_fps))
                    image = canvas(*panels, name, time_s, seconds)
                    if frame == snapshot_frame:
                        image.save(out_dir / f"failure_{name}.png")
                    assert encoder.stdin is not None
                    encoder.stdin.write(image.tobytes())
            finally:
                if encoder.stdin:
                    encoder.stdin.close()
                returncode = encoder.wait()
            if returncode:
                raise RuntimeError(f"ffmpeg failed with exit code {returncode}: {output}")
            outputs.append(output)
            metadata.append({
                "name": name, "video": str(output.resolve()), "frames": frame_count, "fps": fps,
                "seconds": seconds, "size": [width, height], "camera": camera_note,
                "terminal_camera": "Each failed panel freezes its camera at its own terminal time",
                "camera_height": camera_height, "azimuth": azimuth, "elevation": elevation,
                "distance": distance, "snapshot_time_s": snapshot_frame / fps,
                "failure_left_s": failure_left / source_fps if failure_left >= 0 else None,
                "failure_right_s": failure_right / source_fps if failure_right >= 0 else None,
            })
            print(f"Wrote {output} ({frame_count} frames, {seconds:.2f}s)", flush=True)
    finally:
        renderer.close()
    if concat and outputs:
        # Relative safe filenames; no shell or network inputs are interpreted.
        listing = out_dir / "comparison_concat.txt"
        listing.write_text("".join(f"file '{path.name}'\n" for path in outputs))
        summary = out_dir / "comparison_40000_vs_200000.mp4"
        subprocess.run([
            ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
            "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(summary),
        ], check=True)
        print(f"Wrote {summary}", flush=True)
    (out_dir / "comparison_manifest.json").write_text(json.dumps({
        "left": str(left.resolve()), "right": str(right.resolve()),
        "left_iteration": left_iteration, "right_iteration": right_iteration,
        "note": "Robot pose, reference pose, and panel camera freeze at each checkpoint's first terminal. No resets are shown.",
        "clips": metadata,
    }, indent=2) + "\n")


if __name__ == "__main__":
    tyro.cli(main)

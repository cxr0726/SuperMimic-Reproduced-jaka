"""Render saved 40k Khan baseline/push tests using their frozen robot asset.

Only qpos rendering is performed. Policies and training environments are never
loaded. Impulses are labelled as velocity changes, not forces in newtons.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import tyro
from PIL import Image, ImageDraw
from render_khan_checkpoint_comparison import ffmpeg_binary, font, read_recording
from render_rollout import two_robot_xml


def scalar(recording, key, default):
    return np.asarray(recording.get(key, default)).item()


def panel_label(pixels, title, subtitle, failure_step, fps, time_s, pushed=False):
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle((16, 14, image.width - 16, 106), radius=9, fill=(20, 28, 41, 220))
    draw.text((32, 24), title, font=font(26, True), fill=(249, 251, 255))
    draw.text((32, 65), subtitle, font=font(19), fill=(255, 207, 135) if pushed else (191, 220, 239))
    if failure_step >= 0 and time_s >= failure_step / fps - 1e-8:
        message = f"TERMINATED {failure_step / fps:.2f}s | pose & camera FROZEN | no reset"
        draw.rounded_rectangle((16, image.height - 55, image.width - 16, image.height - 14), radius=8, fill=(135, 39, 42, 225))
        draw.text((30, image.height - 48), message, font=font(18), fill=(255, 238, 232))
    return image


def compose(panels, title, time_s, seconds, footer):
    gap = 4
    width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    height = panels[0].height + 108
    image = Image.new("RGB", (width, height), (21, 28, 40))
    draw = ImageDraw.Draw(image)
    x = 0
    for panel in panels:
        image.paste(panel, (x, 60))
        x += panel.width + gap
    draw.text((24, 15), title, font=font(25, True), fill=(245, 248, 252))
    clock = f"t = {time_s:05.2f} / {seconds:.2f} s"
    draw.text((width - draw.textlength(clock, font=font(22)) - 24, 17), clock, font=font(22), fill=(245, 248, 252))
    draw.text((24, height - 34), footer, font=font(18), fill=(212, 223, 239))
    return image


def add_impulse_arrow(scene, root, velocity):
    velocity = np.asarray(velocity, dtype=float)
    direction = np.array((velocity[0], velocity[1], 0.0))
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return
    direction /= norm
    start = np.asarray(root) + np.array((0.0, 0.0, 0.40)) - 0.30 * direction
    end = start + 0.65 * direction
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array((1.0, 0.43, 0.04, 1.0), dtype=np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.018, start, end)
    scene.ngeom += 1


def main(
    baseline: Path,
    push: Path,
    out_dir: Path,
    asset_dir: Path | None = None,
    fps: int = 25,
    seconds: float = 15.0,
    render_baseline: bool = True,
    overview_name: str = "push_overview",
    panel_width: int = 960,
    panel_height: int = 640,
    only_failed: bool = False,
    samples: list[str] | None = None,
    max_after_failure: float | None = 1.5,
) -> None:
    """Write three baseline videos and each recorded push case compared with its baseline."""
    base, disturbed = read_recording(baseline), read_recording(push)
    source_fps = float(scalar(base, "fps", 50.0))
    if source_fps != float(scalar(disturbed, "fps", 50.0)) or source_fps <= 0 or fps <= 0:
        raise ValueError("Recordings must have the same positive fps")
    frozen = asset_dir or Path(str(scalar(base, "baseline_asset_dir", "")))
    if not (frozen / "robot.xml").is_file():
        raise ValueError("A frozen baseline asset directory is required")
    profile = json.loads((frozen / "profile.json").read_text())
    model = mujoco.MjModel.from_xml_string(two_robot_xml(frozen / "robot.xml", profile["root_body"]))
    nq = base["robot_qpos"].shape[-1]
    if model.nq != 2 * nq or disturbed["robot_qpos"].shape[-1] != nq:
        raise ValueError("Frozen robot model does not match recording qpos")
    for geom in range(model.ngeom):
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "").startswith("ref_"):
            model.geom_rgba[geom] = (0.12, 0.42, 1.0, 0.30)
    data = mujoco.MjData(model)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, panel_width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, panel_height)
    renderer = mujoco.Renderer(model, panel_height, panel_width)
    options = mujoco.MjvOption()
    options.geomgroup[:] = 0
    options.geomgroup[[0, 1, 2]] = 1
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation, camera.distance = 140.0, -15.0, 2.7
    base_names = [str(name) for name in base.get("source_clip", base["names"])]
    push_names = [str(name) for name in disturbed["names"]]
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder_exe = ffmpeg_binary()
    frames = round(fps * seconds)
    manifests = []
    push_outputs = []

    def draw_panel(recording, index, source_step, time_s, title, subtitle, roots, is_push=False, camera_distance=2.7):
        failure = int(recording["failure_steps"][index])
        camera_step = min(source_step, failure) if failure >= 0 else source_step
        root = roots[camera_step]
        camera.lookat[:] = (root[0], root[1], float(np.clip(np.median(roots[:, 2]), 0.48, 0.72)))
        camera.distance = camera_distance
        data.qpos[:nq] = recording["robot_qpos"][index, source_step]
        data.qpos[nq:] = recording["ref_qpos"][index, source_step]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera, options)
        if is_push:
            applied_at = float(recording["push_time_s"][index])
            if applied_at <= time_s < applied_at + 0.8 and (failure < 0 or source_step < failure):
                add_impulse_arrow(renderer.scene, data.qpos[:3], recording["actual_delta_v_world_mps"][index])
        return panel_label(renderer.render(), title, subtitle, failure, source_fps, time_s, is_push)

    def encode(path, callback, snapshot_time, duration=seconds):
        clip_frames = round(fps * duration)
        first = callback(0, 0.0)
        command = [encoder_exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{first.width}x{first.height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
        snapshot_frame = min(clip_frames - 1, int(np.ceil(snapshot_time * fps)))
        try:
            for frame in range(clip_frames):
                time_s = frame / fps
                source_step = round(time_s * source_fps)
                image = first if frame == 0 else callback(source_step, time_s)
                if frame == snapshot_frame:
                    image.save(path.with_suffix(".png"))
                assert encoder.stdin is not None
                encoder.stdin.write(image.tobytes())
        finally:
            if encoder.stdin:
                encoder.stdin.close()
            code = encoder.wait()
        if code:
            raise RuntimeError(f"ffmpeg failed: {path}")
        print(f"Wrote {path} ({clip_frames} frames, {clip_frames/fps:.2f}s)", flush=True)

    try:
        if render_baseline:
            for index, name in enumerate(base_names):
                if int(base["lengths"][index]) < round(seconds * source_fps) + 1:
                    raise ValueError(f"{name}: insufficient baseline duration")
                roots = base.get("camera_ref_root", base["ref_qpos"][..., :3])[index]
                def baseline_frame(source_step, time_s, index=index, name=name, roots=roots):
                    panel = draw_panel(base, index, source_step, time_s, "40,000 iterations  |  No push", "Continuous real-physics rollout; no resets shown", roots)
                    return compose([panel], f"Khan mini  |  {name}", time_s, seconds, "Solid: simulated policy   |   Blue: motion reference   |   Frozen baseline robot asset")
                output = out_dir / f"baseline_{name}.mp4"
                encode(output, baseline_frame, 8.0)
                manifests.append({"type": "baseline", "name": name, "video": str(output.resolve()), "seconds": seconds, "fps": fps, "frames": frames})
        for index, name in enumerate(push_names):
            if samples and name not in samples:
                continue
            if only_failed and int(disturbed["failure_steps"][index]) < 0:
                continue
            clip = str(disturbed["source_clip"][index])
            base_index = base_names.index(clip)
            if int(disturbed["lengths"][index]) < round(seconds * source_fps) + 1:
                raise ValueError(f"{name}: insufficient push recording duration")
            roots = base.get("camera_ref_root", base["ref_qpos"][..., :3])[base_index]
            direction = str(disturbed["direction"][index])
            dv = float(disturbed["delta_v_mps"][index])
            push_at = float(disturbed["push_time_s"][index])
            actual = disturbed["actual_delta_v_world_mps"][index]
            outcome = str(disturbed["sample_outcome"][index])
            failure = int(disturbed["failure_steps"][index])
            duration = min(seconds, failure / source_fps + max_after_failure) if failure >= 0 and max_after_failure is not None else seconds
            duration = round(duration * fps) / fps
            subtitle = f"Impulse at {push_at:.1f}s: {direction}, delta-v {dv:.1f} m/s"
            def push_frame(source_step, time_s, base_index=base_index, index=index, roots=roots, subtitle=subtitle, actual=actual, clip=clip, duration=duration):
                deviation = float(np.linalg.norm(disturbed["robot_qpos"][index, source_step, :2] - disturbed["ref_qpos"][index, source_step, :2]))
                distance = max(2.7, 1.8 * (deviation + 0.85))
                left_panel = draw_panel(base, base_index, source_step, time_s, "40,000 iterations  |  No push", "Same checkpoint, motion, starting frame, and camera", roots, camera_distance=distance)
                right_panel = draw_panel(disturbed, index, source_step, time_s, "40,000 iterations  |  Pushed", subtitle, roots, True, camera_distance=distance)
                footer = f"Solid: simulated policy | Blue: reference | World delta-v XY = ({actual[0]:+.2f}, {actual[1]:+.2f}) m/s | Impulse, not sustained force"
                return compose([left_panel, right_panel], f"Khan mini  |  {clip}", time_s, duration, footer)
            output = out_dir / f"push_comparison_{name}.mp4"
            encode(output, push_frame, failure / source_fps if failure >= 0 else push_at + 1.0, duration)
            push_outputs.append(output)
            manifests.append({"type": "push_comparison", "name": name, "clip": clip, "video": str(output.resolve()), "direction": direction, "delta_v_mps": dv, "actual_delta_v_world_mps": actual.tolist(), "push_time_s": push_at, "full_clip_sample_outcome": outcome, "failure_time_s": failure / source_fps if failure >= 0 else None, "seconds": round(duration * fps) / fps, "fps": fps, "frames": round(duration * fps)})
    finally:
        renderer.close()
    if push_outputs:
        listing = out_dir / f"{overview_name}_concat.txt"
        listing.write_text("".join(f"file '{path.name}'\n" for path in push_outputs))
        subprocess.run([encoder_exe, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(out_dir / f"{overview_name}.mp4")], check=True)
    (out_dir / f"{overview_name}_render_manifest.json").write_text(json.dumps({"baseline": str(baseline.resolve()), "push": str(push.resolve()), "frozen_asset_dir": str(frozen.resolve()), "note": "Recorded physics only; first terminal freezes pose/reference/camera. No automatic reset is presented as recovery.", "videos": manifests}, indent=2) + "\n")


if __name__ == "__main__":
    tyro.cli(main)

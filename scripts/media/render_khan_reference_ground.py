"""Render original Khan NPZ poses beside a ground-plane/foot-collision closeup.

This is kinematic reference playback: no policy, physics step, height correction,
IK, frame trimming of the source or asset modifications. The right view omits
the opaque floor so foot collisions below z=0 remain visible through the ground grid.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import tyro
from PIL import Image, ImageDraw, ImageFont

from supermimic.data.foot_clearance import (
    foot_geom_groups,
    geom_bottom_z,
    load_npz_motion,
    mesh_support_vertices,
)
from supermimic.robots.khan_mini import DEFAULT_ASSET_DIR, load_profile


@lru_cache(maxsize=8)
def font(size: int, bold: bool = False):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{filename}", size)
    except OSError:
        return ImageFont.load_default()


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ffmpeg_binary() -> str:
    result = shutil.which("ffmpeg")
    if result is not None:
        return result
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def display_model(asset_dir: Path, panel_width: int, panel_height: int):
    root = ET.fromstring((asset_dir / "robot.xml").read_text())
    root.find("compiler").set("meshdir", str(asset_dir.resolve()))
    for name in ("actuator", "sensor", "keyframe", "visual"):
        value = root.find(name)
        if value is not None:
            root.remove(value)
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(panel_width), offheight=str(panel_height))
    ET.SubElement(visual, "quality", shadowsize="2048", offsamples="4")
    ET.SubElement(visual, "map", znear="0.002")
    ET.SubElement(visual, "headlight", ambient="0.45 0.45 0.45",
                  diffuse="0.45 0.45 0.45", specular="0.08 0.08 0.08")
    asset = root.find("asset")
    ET.SubElement(asset, "texture", name="reference_sky", type="skybox", builtin="gradient",
                  rgb1="0.95 0.97 0.99", rgb2="0.78 0.83 0.90", width="256", height="256")
    ET.SubElement(asset, "texture", name="reference_grid", type="2d", builtin="checker",
                  rgb1="0.89 0.91 0.94", rgb2="0.73 0.77 0.82", width="512", height="512")
    ET.SubElement(asset, "material", name="reference_floor_material", texture="reference_grid",
                  texrepeat="2 2", texuniform="true", reflectance="0", specular="0.05")
    world = root.find("worldbody")
    ET.SubElement(world, "light", name="reference_light", pos="2 -3 4", dir="-0.2 0.3 -1",
                  directional="true", diffuse="0.5 0.5 0.5", castshadow="false")
    ET.SubElement(world, "geom", name="reference_floor", type="plane", size="0 0 .05",
                  material="reference_floor_material", contype="0", conaffinity="0", group="0")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    groups = foot_geom_groups(model)
    for group in groups.values():
        ids = group["collision"]
        model.geom_group[ids] = 4
        model.geom_rgba[ids] = (.12, .70, .95, 1.)
        model.geom_group[group["visual"]] = 5
    return model, groups


def ground_grid(scene, center: np.ndarray, extent: float = .65):
    """Display-only z=0 grid; no occluding plane in the collision closeup."""
    for direction in range(2):
        for offset in np.linspace(-extent, extent, 7):
            start = np.array([center[0]-extent, center[1]+offset, 0.])
            end = np.array([center[0]+extent, center[1]+offset, 0.])
            if direction:
                start[:2] = [center[0]+offset, center[1]-extent]
                end[:2] = [center[0]+offset, center[1]+extent]
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                               np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                               np.array([.15, .62, .28, 1.], dtype=np.float32))
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .0012, start, end)
            scene.ngeom += 1


def overlay(panels, clip, actual_time, start, end, left_z, right_z, width, height):
    output = Image.new("RGB", (width, height), (20, 28, 40))
    panel_width = panels[0].shape[1]
    output.paste(Image.fromarray(panels[0]), (0, 82))
    output.paste(Image.fromarray(panels[1]), (panel_width+4, 82))
    draw = ImageDraw.Draw(output, "RGBA")
    draw.text((18, 8), f"{clip} | Original reference kinematics", font=font(24, True), fill="white")
    draw.text((18, 43), f"Motion {actual_time:.2f}s | shown window {start:.2f}-{end:.2f}s | "
              "no policy / physics step / height lift", font=font(18), fill=(203, 219, 235))
    for x, title in ((0, "FULL BODY: original visual meshes"),
                     (panel_width+4, "FOOT COLLISIONS: green grid is z=0")):
        draw.rounded_rectangle((x+10, 91, x+panel_width-10, 124), radius=5,
                               fill=(20, 28, 40, 225))
        draw.text((x+20, 96), title, font=font(18, True), fill="white")
    worst = min(left_z, right_z)
    color = (255, 155, 108) if worst < -.001 else (155, 232, 208)
    draw.text((18, height-65), f"Exact foot collision bottom: left {100*left_z:+.2f} cm | "
              f"right {100*right_z:+.2f} cm | lowest {100*worst:+.2f} cm", font=font(23, True), fill=color)
    draw.text((18, height-32), "Orange = foot below -1 mm. Right floor is cut away to show "
              "penetration; all original robot heights are retained.", font=font(17),
              fill=(203, 219, 235))
    return output


def main(npz: Path, out: Path, start_s: float = 0., seconds: float | None = None,
         fps: int = 25, width: int = 1280, height: int = 720,
         asset_dir: Path = DEFAULT_ASSET_DIR, overwrite: bool = False) -> None:
    if not np.isfinite(start_s) or start_s < 0 or fps < 1:
        raise ValueError("Use nonnegative finite start and positive output fps")
    if seconds is not None and (not np.isfinite(seconds) or seconds <= 0):
        raise ValueError("seconds must be positive and finite")
    if width < 960 or height < 600 or width % 2 or height % 2:
        raise ValueError("Even video dimensions of at least 960x600 are required")
    if out.exists() and not overwrite:
        raise FileExistsError(f"Choose a fresh output: {out}")
    motion = load_npz_motion(npz, asset_dir)
    profile = load_profile(asset_dir)
    source_fps = float(motion["fps"].reshape(-1)[0])
    source_duration = (len(motion["joint_pos"])-1)/source_fps
    if start_s >= source_duration:
        raise ValueError("start_s is beyond the motion")
    end_s = min(source_duration, start_s+seconds) if seconds is not None else source_duration
    count = max(1, math.ceil((end_s-start_s)*fps-1e-10))
    indices = np.rint((start_s+np.arange(count)/fps)*source_fps).astype(np.int64)
    indices = np.clip(indices, 0, len(motion["joint_pos"])-1)
    indices[-1] = min(round(end_s*source_fps), len(motion["joint_pos"])-1)
    panel_width, panel_height = (width-4)//2, height-164
    model, groups = display_model(asset_dir, panel_width, panel_height)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, panel_height, panel_width)
    option = mujoco.MjvOption()
    # mjVIS_LIGHT displays the light-source icon; illumination does not require it.
    option.flags[mujoco.mjtVisFlag.mjVIS_LIGHT] = 0
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    root_id = motion["body_names"].tolist().index(profile["root_body"])
    all_collision = np.array([g for group in groups.values() for g in group["collision"]])
    support = {g: mesh_support_vertices(model, int(g)) for g in all_collision
               if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH}
    convex_hull_flag = getattr(mujoco.mjtVisFlag, "mjVIS_CONVEXHULL", None)
    # Include the true deepest source pose even when 25fps sampling would skip
    # its odd-numbered 50Hz frame. This replaces the nearest output sample only.
    full_min, full_worst = math.inf, -1
    for frame in range(round(start_s*source_fps), round(end_s*source_fps)+1):
        data.qpos[:3] = motion["body_pos_w"][frame, root_id]
        data.qpos[3:7] = motion["body_quat_w"][frame, root_id]
        data.qpos[7:] = motion["joint_pos"][frame]
        mujoco.mj_kinematics(model, data)
        bottom = min(geom_bottom_z(model, data, g, mesh_vertices=support.get(g)) for g in all_collision)
        if bottom < full_min:
            full_min, full_worst = bottom, frame
    closest = int(np.argmin(np.abs(indices-full_worst)))
    indices[closest] = full_worst
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg_binary(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
               "-pix_fmt", "yuv420p", "-crf", "19", "-movflags", "+faststart", str(out)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    selected_stills = {}
    sampled_min, sampled_frame = math.inf, -1
    try:
        for video_frame, frame in enumerate(indices):
            data.qpos[:3] = motion["body_pos_w"][frame, root_id]
            data.qpos[3:7] = motion["body_quat_w"][frame, root_id]
            data.qpos[7:] = motion["joint_pos"][frame]
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            bottom = [min(geom_bottom_z(model, data, g, mesh_vertices=support.get(g)) for g in group["collision"])
                      for group in groups.values()]
            for z, group in zip(bottom, groups.values(), strict=True):
                model.geom_rgba[group["collision"]] = ((.97, .35, .10, 1.) if z < -.001
                                                         else (.12, .70, .95, 1.))
            option.geomgroup[:] = 0
            option.geomgroup[[0, 2, 5]] = 1
            if convex_hull_flag is not None:
                option.flags[convex_hull_flag] = False
            camera.lookat[:] = [data.qpos[0], data.qpos[1], .62]
            camera.azimuth, camera.elevation, camera.distance = 140., -14., 2.65
            renderer.update_scene(data, camera, option)
            full_body = renderer.render().copy()
            option.geomgroup[:] = 0
            option.geomgroup[4] = 1
            if convex_hull_flag is not None:
                option.flags[convex_hull_flag] = True
            feet_center = data.geom_xpos[all_collision].mean(axis=0)
            spread = np.linalg.norm(data.geom_xpos[all_collision, :2]-feet_center[:2], axis=1).max()
            camera.lookat[:] = [*feet_center[:2], np.clip((min(bottom)+data.geom_xpos[all_collision, 2].max())/2, -.1, .2)]
            camera.azimuth, camera.elevation = 90., -8.
            camera.distance = max(.85, 2.4*spread+.45)
            renderer.update_scene(data, camera, option)
            ground_grid(renderer.scene, feet_center, extent=max(.55, spread+.25))
            detail = renderer.render().copy()
            image = overlay([full_body, detail], npz.stem, frame/source_fps,
                            start_s, end_s, *bottom, width, height)
            if video_frame in (0, len(indices)//2, len(indices)-1):
                selected_stills[{0: "start", len(indices)//2: "middle", len(indices)-1: "end"}[video_frame]] = image.copy()
            if min(bottom) < sampled_min:
                sampled_min, sampled_frame = min(bottom), int(frame)
                selected_stills["deepest"] = image.copy()
            encoder.stdin.write(image.tobytes())
    except BaseException:
        encoder.kill()
        encoder.wait()
        raise
    finally:
        if encoder.stdin:
            try:
                encoder.stdin.close()
            except BrokenPipeError:
                pass
        renderer.close()
    code = encoder.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)
    for name, still in selected_stills.items():
        still.save(out.with_name(f"{out.stem}_{name}.png"))
    sheet = Image.new("RGB", (width, 2*height), (20, 28, 40))
    sheet.paste(selected_stills["middle"], (0, 0))
    sheet.paste(selected_stills["deepest"], (0, height))
    sheet.save(out.with_name(f"{out.stem}_contact_sheet.png"))
    report = {"npz": str(npz.resolve()), "npz_sha256": sha256(npz),
              "model_sha256": sha256(asset_dir/"robot.xml"),
              "profile_sha256": sha256(asset_dir/"profile.json"),
              "video": str(out.resolve()), "video_sha256": sha256(out),
              "source_fps": source_fps, "source_duration_s": source_duration,
              "motion_window_s": [start_s, end_s], "output_fps": fps,
              "video_frames": len(indices), "video_duration_s": len(indices)/fps,
              "size": [width, height], "first_source_frame": int(indices[0]),
              "last_source_frame": int(indices[-1]),
              "sampled_collision_min_z_m": sampled_min, "sampled_worst_source_frame": sampled_frame,
              "full_window_collision_min_z_m": full_min,
              "full_window_worst_source_frame": full_worst,
              "deepest_source_frame_explicitly_included": bool(full_worst in indices),
              "kinematic_reference_only": True, "physics_steps": 0, "height_offset_m": 0.,
              "note": "Right view is a foot-collision cutaway with exact z=0 ground grid. "
                      "Foot collision colors are display-only; source poses and collision geometry sizes unchanged."}
    out.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    tyro.cli(main)

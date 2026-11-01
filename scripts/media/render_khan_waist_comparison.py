"""Render the real Khan waist collision box before/after an asset adjustment.

Only this process's display model is recolored. No asset, joint state, inertia,
actuator, dataset, or training configuration is written or stepped.

Example::

    MUJOCO_GL=egl .venv/bin/python scripts/media/render_khan_waist_comparison.py \
        --out-dir logs/khan_waist_collision_adjustment \
        --before-json logs/khan_waist_collision_adjustment/before/waist_collision.json

The after box is always read from the current robot.xml. The before box can be
provided as saved JSON or as --before-half-size X Y Z (MuJoCo half extents in m).
"""

from __future__ import annotations

import itertools
import json
import os
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import tyro
from PIL import Image, ImageDraw, ImageFont

from supermimic.robots.khan_mini import DEFAULT_ASSET_DIR

GEOM_NAME = "waist_yaw_Link_45_collision"
OLD_COLOR = np.array((0.96, 0.43, 0.12, 1.0), dtype=np.float32)
NEW_COLOR = np.array((0.02, 0.65, 0.67, 1.0), dtype=np.float32)
BACKGROUND = (240, 244, 249)
INK = (28, 40, 56)
VIEWS = (("Front (+X)", 0.0, -2.0), ("Side (+Y)", 90.0, -2.0), ("Three-quarter", 40.0, -10.0))


@lru_cache(maxsize=8)
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{filename}", size)


def full_size_label(half_size: np.ndarray) -> str:
    return " x ".join(f"{x:.3f}" for x in 2 * half_size) + " m"


def display_model(asset_dir: Path, width: int, height: int) -> mujoco.MjModel:
    """Compile the current MJCF with display-only illumination and a neutral floor."""
    root = ET.fromstring((asset_dir / "robot.xml").read_text())
    compiler = root.find("compiler")
    assert compiler is not None
    compiler.set("meshdir", str(asset_dir.resolve()))
    visual = root.find("visual")
    if visual is not None:
        root.remove(visual)
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(width), offheight=str(height))
    ET.SubElement(visual, "quality", shadowsize="4096", offsamples="8")
    ET.SubElement(visual, "map", znear="0.003")
    ET.SubElement(visual, "headlight", ambient="0.55 0.55 0.55", diffuse="0.45 0.45 0.45", specular="0.08 0.08 0.08")
    asset = root.find("asset")
    assert asset is not None
    ET.SubElement(asset, "texture", name="comparison_sky", type="skybox", builtin="gradient", rgb1="0.96 0.97 0.99", rgb2="0.88 0.91 0.95", width="256", height="256")
    world = root.find("worldbody")
    assert world is not None
    ET.SubElement(world, "light", name="comparison_light", pos="2 -2 4", dir="-0.3 0.3 -1", directional="true", diffuse="0.5 0.5 0.5", castshadow="false")
    ET.SubElement(world, "geom", name="comparison_floor", type="plane", size="0 0 0.05", rgba="0.92 0.94 0.97 1", contype="0", conaffinity="0", group="0")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def append_box(
    scene: mujoco.MjvScene, half_size: np.ndarray, center: np.ndarray, rotation: np.ndarray,
    color: np.ndarray, filled: bool = False,
) -> None:
    """Draw exact box edges in the scene, without creating physical collision shapes."""
    if filled:
        rgba = color.copy()
        rgba[3] = 0.10
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX, half_size, center, rotation.ravel(), rgba)
        scene.ngeom += 1
    signs = list(itertools.product((-1, 1), repeat=3))
    vertices = {sign: center + rotation @ (np.array(sign) * half_size) for sign in signs}
    for sign in signs:
        for axis in range(3):
            if sign[axis] != -1:
                continue
            adjacent = list(sign)
            adjacent[axis] = 1
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), color)
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0013, vertices[sign], vertices[tuple(adjacent)])
            scene.ngeom += 1


def annotate(image: Image.Image, title: str, subtitle: str, color: tuple[int, int, int]) -> Image.Image:
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle((16, 14, image.width - 16, 99), radius=10, fill=(255, 255, 255, 240))
    draw.rectangle((16, 14, 24, 99), fill=color)
    draw.text((38, 24), title, font=font(27, True), fill=INK)
    draw.text((38, 65), subtitle, font=font(21), fill=color)
    return image


def main(
    out_dir: Path,
    before_json: Path | None = None,
    before_half_size: tuple[float, float, float] = (0.075, 0.1, 0.15),
    before_pos: tuple[float, float, float] = (0.0, 0.0, 0.03),
    asset_dir: Path = DEFAULT_ASSET_DIR,
    width: int = 840,
    height: int = 760,
    reference_npz: Path | None = None,
    reference_frame: int = 7293,
) -> None:
    """Write front/side/oblique before-after closeups, an overlap view, and full-body views."""
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = json.loads((asset_dir / "profile.json").read_text())
    model = display_model(asset_dir, width, height)
    data = mujoco.MjData(model)
    waist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "waist_yaw_Link")
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, GEOM_NAME)
    if waist < 0 or geom_id < 0 or model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
        raise ValueError("Expected the named Khan waist box collision geom")
    after_size, after_pos, after_quat = model.geom_size[geom_id].copy(), model.geom_pos[geom_id].copy(), model.geom_quat[geom_id].copy()
    before = json.loads(before_json.read_text()) if before_json else {}
    old_size = np.array(before.get("half_size", before_half_size), dtype=np.float64)
    old_pos = np.array(before.get("pos", before_pos), dtype=np.float64)
    old_quat = np.array(before.get("quat", after_quat), dtype=np.float64)
    if old_size.shape != (3,) or np.any(old_size <= 0) or not np.isfinite(old_size).all():
        raise ValueError("Before half size must contain three finite positive values")
    # Identical source home pose in every panel; no stepping or pose optimization.
    data.qpos[:3] = profile["initial_pos"]
    for name, value in profile["default_joint_pos"].items():
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint]] = value
    visual_ids = np.flatnonzero(model.geom_group == 2)
    model.geom_rgba[visual_ids] = (0.42, 0.46, 0.52, 0.28)
    model.geom_group[geom_id] = 1
    options = mujoco.MjvOption()
    options.geomgroup[:] = 0
    options.geomgroup[[0, 1, 2]] = 1
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    renderer = mujoco.Renderer(model, height, width)
    colors = [tuple(int(x * 255) for x in color[:3]) for color in (OLD_COLOR, NEW_COLOR)]

    def render(half_size, pos, quat, color, azimuth, elevation, full_body=False, overlay=False):
        model.geom_size[geom_id] = half_size
        model.geom_pos[geom_id] = pos
        model.geom_quat[geom_id] = quat
        model.geom_rgba[geom_id] = (*color[:3], 0.12)
        mujoco.mj_forward(model, data)
        camera.lookat[:] = data.xpos[waist] + (np.array((0.0, 0.0, 0.07)) if not full_body else np.array((0.0, 0.0, -0.20)))
        camera.azimuth, camera.elevation = azimuth, elevation
        camera.distance = 2.25 if full_body else 0.95
        renderer.update_scene(data, camera, options)
        if overlay:
            # Current box remains teal; add original as a display-only orange box.
            old_rotation = np.zeros(9)
            mujoco.mju_quat2Mat(old_rotation, old_quat)
            body_rotation = data.xmat[waist].reshape(3, 3)
            old_center = data.xpos[waist] + body_rotation @ old_pos
            append_box(renderer.scene, old_size, old_center, body_rotation @ old_rotation.reshape(3, 3), OLD_COLOR, filled=True)
        append_box(renderer.scene, half_size, data.geom_xpos[geom_id], data.geom_xmat[geom_id].reshape(3, 3), color)
        return Image.fromarray(renderer.render())

    try:
        comparisons = []
        before_panels, after_panels, overlay_panels = [], [], []
        for title, azimuth, elevation in VIEWS:
            before_panel = annotate(render(old_size, old_pos, old_quat, OLD_COLOR, azimuth, elevation), f"Before  |  {title}", "Full X x Y x Z: " + full_size_label(old_size), colors[0])
            after_panel = annotate(render(after_size, after_pos, after_quat, NEW_COLOR, azimuth, elevation), f"After  |  {title}", "Full X x Y x Z: " + full_size_label(after_size), colors[1])
            before_panels.append(before_panel)
            after_panels.append(after_panel)
            pair = Image.new("RGB", (2 * width + 8, height), BACKGROUND)
            pair.paste(before_panel, (0, 0))
            pair.paste(after_panel, (width + 8, 0))
            view_name = title.split(" ")[0].lower().replace("-", "_")
            pair.save(out_dir / f"waist_box_{view_name}.png")
            comparisons.append(str((out_dir / f"waist_box_{view_name}.png").resolve()))
            overlay_panel = annotate(render(after_size, after_pos, after_quat, NEW_COLOR, azimuth, elevation, overlay=True), f"Overlay  |  {title}", "Orange = before     Teal = after", colors[1])
            overlay_panels.append(overlay_panel)
        grid = Image.new("RGB", (3 * width + 16, 2 * height + 150), BACKGROUND)
        for column in range(3):
            grid.paste(before_panels[column], (column * (width + 8), 0))
            grid.paste(after_panels[column], (column * (width + 8), height + 8))
        draw = ImageDraw.Draw(grid)
        draw.text((26, 2 * height + 28), "waist_yaw_Link collision box  |  Gray: visual mesh  |  Same source home pose and camera", font=font(26, True), fill=INK)
        pos_label = f"Local center: before {np.array2string(old_pos, precision=3)} m; after {np.array2string(after_pos, precision=3)} m. Full dimensions are shown, not half extents."
        draw.text((26, 2 * height + 70), pos_label, font=font(22), fill=(67, 83, 105))
        delta_per_side_mm = (old_size - after_size) * 1000
        reduction_label = "Reduction per side (X / Y / Z): " + " / ".join(f"{value:.1f}" for value in delta_per_side_mm) + " mm. Width change only; depth and height retained."
        draw.text((26, 2 * height + 110), reduction_label, font=font(24, True), fill=colors[1])
        grid.save(out_dir / "waist_box_comparison.png")
        overlay_grid = Image.new("RGB", (3 * width + 16, height), BACKGROUND)
        for column, panel in enumerate(overlay_panels):
            overlay_grid.paste(panel, (column * (width + 8), 0))
        overlay_grid.save(out_dir / "waist_box_overlay.png")
        whole = Image.new("RGB", (2 * width + 8, height), BACKGROUND)
        whole.paste(annotate(render(old_size, old_pos, old_quat, OLD_COLOR, 40.0, -10.0, full_body=True), "Before  |  Full robot", full_size_label(old_size), colors[0]), (0, 0))
        whole.paste(annotate(render(after_size, after_pos, after_quat, NEW_COLOR, 40.0, -10.0, full_body=True), "After  |  Full robot", full_size_label(after_size), colors[1]), (width + 8, 0))
        whole.save(out_dir / "waist_box_fullbody.png")
        reference_report = None
        if reference_npz:
            with np.load(reference_npz, allow_pickle=False) as archive:
                if not 0 <= reference_frame < len(archive["joint_pos"]):
                    raise ValueError("Reference frame outside motion length")
                names = [str(name) for name in archive["body_names"]]
                root_body = names.index(profile["root_body"])
                data.qpos[:3] = archive["body_pos_w"][reference_frame, root_body]
                data.qpos[3:7] = archive["body_quat_w"][reference_frame, root_body]
                for index, name in enumerate(archive["joint_names"]):
                    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
                    data.qpos[model.jnt_qposadr[joint]] = archive["joint_pos"][reference_frame, index]
            ref_grid = Image.new("RGB", (2 * width + 8, height + 116), BACKGROUND)
            penetration = []
            for index, (size, pos, quat, color, label) in enumerate((
                (old_size, old_pos, old_quat, OLD_COLOR, "Before"),
                (after_size, after_pos, after_quat, NEW_COLOR, "After"),
            )):
                panel = render(size, pos, quat, color, 40.0, -10.0)
                waist_contacts = [contact for contact in data.contact if geom_id in contact.geom and contact.dist < 0]
                deepest = min((float(contact.dist) for contact in waist_contacts), default=0.0)
                penetration.append(deepest)
                panel = annotate(panel, f"{label}  |  Reference hand near torso", f"Deepest waist penetration: {-deepest * 1000:.1f} mm", colors[index])
                ref_grid.paste(panel, (index * (width + 8), 0))
            draw = ImageDraw.Draw(ref_grid)
            draw.text((24, height + 16), f"{reference_npz.stem}, source frame {reference_frame}  |  Identical original reference pose", font=font(24, True), fill=INK)
            draw.text((24, height + 57), "Shrinking the box reduces some contacts; reference self-penetration can remain.", font=font(23), fill=(67, 83, 105))
            draw.text((24, height + 88), "Rendering original kinematics only; no new policy simulation or dataset modification.", font=font(19), fill=(67, 83, 105))
            ref_grid.save(out_dir / "waist_box_reference_contact.png")
            reference_report = {"npz": str(reference_npz.resolve()), "frame": reference_frame, "deepest_waist_penetration_before_m": penetration[0], "deepest_waist_penetration_after_m": penetration[1]}
    finally:
        renderer.close()
    report = {
        "geom_name": GEOM_NAME, "robot_xml": str((asset_dir / "robot.xml").resolve()),
        "before_half_size": old_size.tolist(), "after_half_size": after_size.tolist(),
        "before_full_size": (2 * old_size).tolist(), "after_full_size": (2 * after_size).tolist(),
        "before_local_pos": old_pos.tolist(), "after_local_pos": after_pos.tolist(),
        "before_quat": old_quat.tolist(), "after_quat": after_quat.tolist(),
        "volume_ratio_after_before": float(np.prod(after_size) / np.prod(old_size)),
        "pose": "Source profile default_joint_pos and initial_pos; no dynamics steps",
        "rendering": "Real MuJoCo geometry. Gray translucent visual mesh; exact orange/teal box faces and edges. Other collisions hidden for clarity.",
        "individual_comparisons": comparisons,
        "reference_contact": reference_report,
    }
    (out_dir / "render_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    tyro.cli(main)

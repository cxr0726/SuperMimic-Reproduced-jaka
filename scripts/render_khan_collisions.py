"""Render the configured Khan_mini collision geometry without changing its assets.

The upper row shows collisions alone; the lower row adds faint visual geometry
for context. Mesh collision hull visualization is enabled in MuJoCo.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render_collisions(asset_dir: Path | None, output_dir: Path) -> dict:
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    from supermimic.robots.khan_mini import get_khan_mini_cfg, load_profile

    profile = load_profile(asset_dir)
    original = get_khan_mini_cfg(asset_dir).build().spec.compile()
    source_rgba = original.geom_rgba.copy()
    model = copy.copy(original)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    mujoco.mj_forward(model, data)
    collisions = np.flatnonzero((model.geom_contype | model.geom_conaffinity) != 0)
    visual = np.flatnonzero(model.geom_group == 2)
    foot_body_ids = {model.body(name).id for name in profile["foot_bodies"]}
    foot_geoms = [int(i) for i in collisions if int(model.geom_bodyid[i]) in foot_body_ids]
    other_geoms = [int(i) for i in collisions if int(i) not in foot_geoms]
    covered = {int(model.geom_bodyid[i]) for i in collisions}

    # Render-only colors on an independent model, never the on-disk MJCF.
    model.geom_rgba[other_geoms] = [0.98, 0.55, 0.12, 1.0]
    model.geom_rgba[foot_geoms] = [0.13, 0.72, 0.97, 1.0]
    model.geom_rgba[visual] = [0.75, 0.82, 0.9, 0.18]
    model.mat_rgba[:, 3] = 1.0
    model.vis.headlight.ambient[:] = [0.6, 0.6, 0.6]
    model.vis.headlight.diffuse[:] = [0.7, 0.7, 0.7]
    model.vis.headlight.specular[:] = [0.1, 0.1, 0.1]
    model.vis.global_.fovy = 38.0
    width, height, banner = 560, 670, 58
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    lower = (data.geom_xpos - model.geom_rbound[:, None]).min(axis=0)
    upper = (data.geom_xpos + model.geom_rbound[:, None]).max(axis=0)
    lookat = (lower + upper) * 0.5
    distance = float(np.max(upper - lower)) / (2 * math.tan(math.radians(19))) * 1.16
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.elevation = -8.0

    option = mujoco.MjvOption()
    option.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = True
    option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = False
    option.sitegroup[:] = 0

    def font(size):
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default(size=size)

    title_font, label_font, note_font = font(34), font(26), font(20)
    header, footer = 126, 100
    canvas = Image.new("RGB", (width * 3, header + 2 * (height + banner) + footer), "#101722")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 16), "Khan_mini: actual collision geometry", fill="#f4f7fa", font=title_font)
    draw.text(
        (26, 66),
        "27 joints | 26 collision geoms: 18 capsules + 6 meshes + 2 boxes",
        fill="#d6e0eb", font=note_font,
    )
    draw.text(
        (26, 94),
        f"Collision coverage: {len(covered)} / {model.nbody - 1} links; head and thighs have no own colliders",
        fill="#d6e0eb", font=note_font,
    )
    cameras = []
    renderer = None
    try:
        renderer = mujoco.Renderer(model, height=height, width=width)
        for row, overlay in enumerate((False, True)):
            option.geomgroup[:] = 0
            option.geomgroup[3] = 1
            option.geomgroup[2] = int(overlay)
            for column, (label, azimuth) in enumerate(
                (("Front (+X)", 180.0), ("Side (+Y)", 270.0), ("Three-quarter (+X,+Y)", 225.0))
            ):
                camera.azimuth = azimuth
                renderer.update_scene(data, camera=camera, scene_option=option)
                rgb = renderer.render()
                x, y = column * width, header + row * (height + banner)
                row_label = "collisions only" if not overlay else "faint visual outline + collisions"
                draw.text((x + 14, y + 14), label, fill="#f4f7fa", font=label_font)
                canvas.paste(Image.fromarray(rgb), (x, y + banner))
                if row == 0:
                    cameras.append({
                        "label": label, "azimuth_deg": azimuth,
                        "eye_world": renderer.scene.camera[0].pos.tolist(),
                        "forward_world": renderer.scene.camera[0].forward.tolist(),
                    })
                # Keep the row semantics visible inside each view.
                draw.text((x + 12, y + banner + height - 30), row_label,
                          fill="#d6e0eb", font=note_font)
    finally:
        if renderer is not None:
            renderer.close()

    footer_y = canvas.height - footer
    draw.rectangle((24, footer_y + 12, 44, footer_y + 32), fill="#fa8c1f")
    draw.text((54, footer_y + 8), "Body / arm collisions", fill="#e5ecf4", font=note_font)
    draw.rectangle((370, footer_y + 12, 390, footer_y + 32), fill="#21b8f7")
    draw.text((400, footer_y + 8), "Foot collisions", fill="#e5ecf4", font=note_font)
    draw.text((24, footer_y + 48),
              "Mesh hull display enabled. Faint shapes are visuals only; no colliders were added.",
              fill="#d6e0eb", font=note_font)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "collision_views.png"
    canvas.save(image_path)

    body_names = [model.body(i).name for i in range(1, model.nbody)]
    report = {
        "image": str(image_path.resolve()), "mujoco_version": mujoco.__version__,
        "source": "get_khan_mini_cfg().build().spec.compile()",
        "pose": "init_state", "root_position_m": data.qpos[:3].tolist(),
        "nq": model.nq, "nv": model.nv, "nu": model.nu,
        "nbody": model.nbody, "ngeom": model.ngeom,
        "visual_geoms": len(visual), "collision_geoms": len(collisions),
        "collision_types": dict(Counter(mujoco.mjtGeom(int(model.geom_type[i])).name
                                        for i in collisions)),
        "foot_collision_geoms": len(foot_geoms),
        "collision_covered_links": len(covered),
        "robot_links": model.nbody - 1,
        "collision_contype_values": sorted({int(model.geom_contype[i]) for i in collisions}),
        "collision_conaffinity_values": sorted({int(model.geom_conaffinity[i])
                                               for i in collisions}),
        "explicit_exclude_count": model.nexclude,
        "disableflags": int(model.opt.disableflags),
        "self_collision_masks_enabled": bool(np.all(model.geom_contype[collisions] == 1)
                                             and np.all(model.geom_conaffinity[collisions] == 1)),
        "initial_self_contact_count": data.ncon,
        "mesh_convex_hull_visualization": True,
        "render_only_changes": "colors, visual alpha, camera, headlight, view flags",
        "links_without_own_collision_geometry": [name for i, name in enumerate(body_names, 1)
                                                if i not in covered],
        "collision_geometries": [
            {"name": model.geom(int(i)).name,
             "body": model.body(int(model.geom_bodyid[i])).name,
             "type": mujoco.mjtGeom(int(model.geom_type[i])).name,
             "group": int(model.geom_group[i]),
             "contype": int(model.geom_contype[i]),
             "conaffinity": int(model.geom_conaffinity[i]),
             "condim": int(model.geom_condim[i]),
             "priority": int(model.geom_priority[i]),
             "friction": model.geom_friction[i].tolist()}
            for i in collisions
        ],
        "camera_views": cameras,
        "notes": [
            "No ground is added; contact count refers to the robot alone at its initial pose.",
            "Self masks permit collisions; MuJoCo still filters same/welded and adjacent bodies.",
            "A link without an own collision geom can share a rigid assembly with another link.",
            "The original model and asset files are never modified by this renderer.",
        ],
    }
    # Verify that render-only recoloring did not mutate the independently built source model.
    assert np.array_equal(original.geom_rgba, source_rgba)
    (output_dir / "metadata.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "logs/khan_mini_collision")
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    report = render_collisions(args.asset_dir, args.output_dir)
    print(json.dumps({key: report[key] for key in
                      ("image", "collision_types", "self_collision_masks_enabled")}, indent=2))


if __name__ == "__main__":
    main()

"""Render collisions from the exported Khan URDF, rather than the training MJCF.

The free base and training home pose are applied for display only. No simulation
steps are taken and no robot assets are changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def render_urdf(urdf: Path, profile_path: Path, output_dir: Path) -> dict:
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    urdf = urdf.resolve()
    profile = json.loads(profile_path.read_text())
    digest = hashlib.sha256(urdf.read_bytes()).hexdigest()
    spec = mujoco.MjSpec.from_file(str(urdf))
    root = spec.body(profile["root_body"])
    root.pos = profile["initial_pos"]
    root.add_freejoint(name="render_floating_base")
    model = spec.compile()
    data = mujoco.MjData(model)
    for name, value in profile["default_joint_pos"].items():
        joint = model.joint(name)
        data.qpos[joint.qposadr[0]] = value
    mujoco.mj_forward(model, data)

    collisions = np.flatnonzero((model.geom_contype | model.geom_conaffinity) != 0)
    visuals = np.flatnonzero((model.geom_contype | model.geom_conaffinity) == 0)
    foot_ids = {model.body(name).id for name in profile["foot_bodies"]}
    feet = [int(i) for i in collisions if int(model.geom_bodyid[i]) in foot_ids]
    other = [int(i) for i in collisions if int(i) not in feet]
    foot_visuals = [int(i) for i in visuals if int(model.geom_bodyid[i]) in foot_ids]
    model.geom_group[collisions] = 3
    model.geom_group[visuals] = 2
    model.geom_rgba[other] = [0.98, 0.55, 0.12, 1.0]
    model.geom_rgba[feet] = [0.13, 0.72, 0.97, 1.0]
    model.geom_rgba[visuals] = [0.75, 0.82, 0.9, 0.16]
    model.geom_matid[:] = -1
    model.vis.headlight.ambient[:] = [0.65, 0.65, 0.65]
    model.vis.headlight.diffuse[:] = [0.65, 0.65, 0.65]
    model.vis.headlight.specular[:] = [0.08, 0.08, 0.08]
    model.vis.global_.fovy = 38
    model.vis.map.znear = 0.002
    width, height = 560, 650
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height

    def font(size: int):
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default(size=size)

    title_font, label_font, note_font = font(32), font(24), font(19)
    option = mujoco.MjvOption()
    # Collision meshes are already explicit hulls. Preserve the original visual
    # surfaces in the translucent overlay, rather than displaying their hulls.
    option.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    option.sitegroup[:] = 0
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    lower = (data.geom_xpos - model.geom_rbound[:, None]).min(axis=0)
    upper = (data.geom_xpos + model.geom_rbound[:, None]).max(axis=0)
    camera.lookat[:] = (lower + upper) * 0.5
    camera.distance = float(np.max(upper - lower)) / (2 * math.tan(math.radians(19))) * 1.14
    camera.elevation = -8
    output_dir.mkdir(parents=True, exist_ok=True)
    images = {}
    views = []
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        header, banner, footer = 112, 48, 82
        canvas = Image.new("RGB", (3 * width, header + 2 * (height + banner) + footer), "#101722")
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 16), "Khan_mini: collisions loaded from new.urdf", fill="#f4f7fa", font=title_font)
        draw.text((24, 62), f"{len(profile['joint_names'])} DOF | {model.nbody - 1} links | "
                  f"{len(collisions)} URDF collision elements", fill="#d6e0eb", font=note_font)
        draw.text((24, 88), "Blue: foot convex hulls   Orange: other collisions", fill="#d6e0eb", font=note_font)
        for row, overlay in enumerate((False, True)):
            option.geomgroup[:] = 0
            option.geomgroup[3] = 1
            option.geomgroup[2] = int(overlay)
            for col, (label, azimuth) in enumerate((("Front", 180.), ("Side", 270.), ("Three-quarter", 225.))):
                camera.azimuth = azimuth
                renderer.update_scene(data, camera=camera, scene_option=option)
                rgb = renderer.render()
                x, y = col * width, header + row * (height + banner)
                draw.text((x + 14, y + 10), label, fill="#f4f7fa", font=label_font)
                canvas.paste(Image.fromarray(rgb), (x, y + banner))
                draw.text((x + 12, y + banner + height - 28),
                          "collision only" if not overlay else "collision + faint visual",
                          fill="#d6e0eb", font=note_font)
                if row == 0:
                    views.append({"label": label, "azimuth_deg": azimuth,
                                  "lookat_m": camera.lookat.tolist(), "distance_m": float(camera.distance)})
        draw.text((24, canvas.height - 68), "Four capsules are exported as single convex meshes (surface approximation below 0.1 mm).",
                  fill="#d6e0eb", font=note_font)
        draw.text((24, canvas.height - 38), "Only rendering colors and pose are changed; no dynamics step or collider modification.",
                  fill="#d6e0eb", font=note_font)
        path = output_dir / "collision_views.png"
        canvas.save(path)
        images["collision_views"] = str(path.resolve())

        # Show one foot at a time. Other links are hidden so instep and sole
        # boundaries can be inspected without the leg occluding the camera.
        model.geom_group[:] = 0
        model.geom_group[feet] = 4
        model.geom_group[foot_visuals] = 5
        foot_canvas = Image.new("RGB", (3 * width, header + 2 * (height + banner) + footer), "#101722")
        draw = ImageDraw.Draw(foot_canvas)
        draw.text((24, 16), "new.urdf: whole-foot convex collision hulls", fill="#f4f7fa", font=title_font)
        draw.text((24, 62), "Blue: explicit collision mesh, including instep and sole (visual surfaces hidden)",
                  fill="#d6e0eb", font=note_font)
        for row, name in enumerate(profile["foot_bodies"]):
            body_id = model.body(name).id
            selected = [i for i in feet if int(model.geom_bodyid[i]) == body_id]
            shown_visual = [i for i in foot_visuals if int(model.geom_bodyid[i]) == body_id]
            model.geom_group[feet] = 0
            model.geom_group[foot_visuals] = 0
            model.geom_group[selected] = 4
            model.geom_group[shown_visual] = 5
            option.geomgroup[:] = 0
            option.geomgroup[4] = 1
            option.geomgroup[5] = 0
            centers = data.geom_xpos[selected]
            radii = model.geom_rbound[selected, None]
            camera.lookat[:] = ((centers - radii).min(axis=0) + (centers + radii).max(axis=0)) * 0.5
            camera.distance = .51
            for col, (label, azimuth, elevation) in enumerate((("Outer side", 270. if row == 0 else 90., -10.),
                                                            ("Instep", 225., -25.), ("Sole", 225., 75.))):
                camera.azimuth, camera.elevation = azimuth, elevation
                renderer.update_scene(data, camera=camera, scene_option=option)
                rgb = renderer.render()
                x, y = col * width, header + row * (height + banner)
                draw.text((x + 14, y + 10), f"{'Left' if row == 0 else 'Right'} foot: {label}",
                          fill="#f4f7fa", font=label_font)
                foot_canvas.paste(Image.fromarray(rgb), (x, y + banner))
        draw.text((24, foot_canvas.height - 66), "One convex mesh per foot. This shows the exported URDF geometry, not an MJCF screenshot.",
                  fill="#d6e0eb", font=note_font)
        draw.text((24, foot_canvas.height - 36), "The convex hull fills CAD concavities; the original visual mesh remains unchanged.",
                  fill="#d6e0eb", font=note_font)
        path = output_dir / "foot_collision_views.png"
        foot_canvas.save(path)
        images["foot_collision_views"] = str(path.resolve())
    finally:
        renderer.close()

    if hashlib.sha256(urdf.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Rendering changed the URDF")
    report = {
        "source_urdf": str(urdf), "source_urdf_sha256": digest,
        "profile": str(profile_path.resolve()), "mujoco_version": mujoco.__version__,
        "source": "MjSpec.from_file(new.urdf), compiled directly; no training MJCF used",
        "images": images, "pose": "profile default_joint_pos", "root_position_m": data.qpos[:3].tolist(),
        "robot_links": model.nbody - 1, "dof": len(profile["joint_names"]),
        "visual_geoms": len(visuals), "collision_geoms": len(collisions),
        "collision_types": dict(Counter(mujoco.mjtGeom(int(model.geom_type[i])).name for i in collisions)),
        "foot_collision_geoms": len(feet), "whole_body_views": views,
        "simulation_steps": 0, "urdf_unchanged": True,
    }
    (output_dir / "render_metadata.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=ROOT / "assets/robots/khan_mini/new.urdf")
    parser.add_argument("--profile", type=Path, default=ROOT / "assets/robots/khan_mini/profile.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "exports/khan_mini_urdf")
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    print(json.dumps(render_urdf(args.urdf, args.profile, args.output_dir), indent=2))


if __name__ == "__main__":
    main()

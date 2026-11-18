"""Compare the compiled Khan left foot colliders in the ankle-roll body frame."""

import argparse
from pathlib import Path

import matplotlib
import mujoco
import numpy as np
from scipy.spatial import ConvexHull

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from supermimic.robots.khan_mini import DEFAULT_ASSET_DIR

OLD, NEW, INK = "#9b5439", "#00776f", "#213333"
SEAM, WIDEST = 10.0, 14.6  # Measured longitudinal stations, cm from the new heel.


def foot_parts(model: mujoco.MjModel, group: int) -> list[np.ndarray]:
    """Return each mesh separately, with a common heel datum and sole plane (cm)."""
    ids = np.flatnonzero((model.geom_bodyid == model.body("Left_ankle_roll_Link").id)
                         & (model.geom_group == group))
    if not len(ids):
        raise ValueError(f"No left foot meshes in group {group}")
    parts = []
    for gid in ids:
        mesh = model.geom_dataid[gid]
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError(f"Expected a mesh for {model.geom(gid).name}")
        start = model.mesh_vertadr[mesh]
        xyz = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[gid])
        xyz = xyz @ rotation.reshape(3, 3).T + model.geom_pos[gid]
        parts.append(np.unique(np.round(100 * xyz + [7.5, 0, 4.8], 5), axis=0))
    return parts


def outline(points: np.ndarray) -> np.ndarray:
    xy = points[:, :2]
    return xy[ConvexHull(xy).vertices]


def width_at(polygons: list[np.ndarray], stations: np.ndarray) -> np.ndarray:
    lower, upper = np.full_like(stations, np.inf), np.full_like(stations, -np.inf)
    for polygon in polygons:
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
            if a[0] == b[0]:
                on = stations == a[0]
                lower[on] = np.minimum(lower[on], min(a[1], b[1]))
                upper[on] = np.maximum(upper[on], max(a[1], b[1]))
                continue
            on = (stations >= min(a[0], b[0])) & (stations <= max(a[0], b[0]))
            y = a[1] + (stations[on] - a[0]) * (b[1] - a[1]) / (b[0] - a[0])
            lower[on] = np.minimum(lower[on], y)
            upper[on] = np.maximum(upper[on], y)
    return np.where(np.isfinite(lower), upper - lower, np.nan)


def draw_plan(ax, polygons, color, style):
    for poly in polygons:
        ax.fill(*poly.T, color=color, alpha=.055, edgecolor="none")
        edges = np.stack([poly, np.roll(poly, -1, axis=0)], axis=1)
        if len(polygons) > 1:
            edges = edges[~np.all(np.isclose(edges[:, :, 0], SEAM, atol=1e-4), axis=1)]
        # A continuous polyline gives the old outline a continuous dash pattern.
        if len(polygons) == 1:
            ax.plot(*np.vstack([poly, poly[0]]).T, color=color, ls=style, lw=2)
        else:
            ax.add_collection(LineCollection(edges, colors=color, linewidths=2.4))


def top_view(old, new, destination):
    before, after = [outline(p) for p in old], [outline(p) for p in new]
    fig, (plan, section) = plt.subplots(2, 1, figsize=(8.2, 8.4), sharex=True,
                                       gridspec_kw={"height_ratios": [1.85, .95]})
    fig.subplots_adjust(left=.11, right=.96, top=.86, bottom=.10, hspace=.30)
    fig.suptitle("左脚碰撞轮廓与沿脚长的宽度", x=.11, y=.98, ha="left",
                 fontsize=21, fontweight="bold")
    fig.legend([Line2D([], [], color=OLD, ls="--", lw=2), Line2D([], [], color=NEW, lw=2.4)],
               ["原模型 · CAD 凸包", "新模型 · 实测轮廓"], loc="upper left",
               bbox_to_anchor=(.103, .935), ncol=2, frameon=False, borderaxespad=0)
    for ax in (plan, section):
        ax.set_axisbelow(True)
        ax.grid(color="#e2e8e6", lw=.65)
        ax.spines[["right", "top"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#a0aeaa")
        for x in (SEAM, WIDEST):
            ax.axvline(x, color="#81918d", ls=":", lw=1)
    draw_plan(plan, before, OLD, "--")
    draw_plan(plan, after, NEW, "-")
    plan.set_aspect("equal", adjustable="box")
    plan.set_ylabel("横向位置（cm）")
    plan.set_ylim(-6.35, 6.5)
    for x, w, label in ((SEAM, 8.7, "8.7 cm"), (WIDEST, 10.5, "10.5 cm")):
        plan.annotate("", xy=(x, w / 2), xytext=(x, -w / 2),
                      arrowprops={"arrowstyle": "<->", "color": NEW, "lw": 1.2,
                                  "shrinkA": 0, "shrinkB": 0})
        plan.text(x, 0, label, color=NEW, ha="center", va="center", weight="bold",
                  bbox={"facecolor": "white", "alpha": .94, "edgecolor": "none", "pad": 1.5})
    x = np.unique(np.concatenate([np.linspace(0, 20.2, 1100),
                                  *[poly[:, 0] for poly in before + after], [SEAM, WIDEST]]))
    old_width, new_width = width_at(before, x), width_at(after, x)
    section.fill_between(x, old_width, new_width, color="#9db5ab", alpha=.24)
    section.plot(x, old_width, color=OLD, ls="--", lw=2)
    section.plot(x, new_width, color=NEW, lw=2.4)
    section.scatter([SEAM, WIDEST], width_at(after, np.array([SEAM, WIDEST])),
                    s=26, color=NEW, zorder=4)
    section.set(xlim=(-.5, 21), ylim=(0, 11.5), xlabel="距脚跟后缘（cm，向右为脚尖）",
                ylabel="轮廓全宽（cm）", xticks=[0, 2, 4, 6, 8, 10, 12, 14.6, 16, 18, 20.2])
    # Match the physical x scale of the equal-aspect plan and the width graph.
    fig.canvas.draw()
    box, lower_box = plan.get_position(), section.get_position()
    section.set_position([box.x0, lower_box.y0, box.width, lower_box.height])
    fig.savefig(destination, dpi=165, facecolor="white")
    plt.close(fig)


def solid(ax, parts, color):
    light = np.array([-.4, -.6, .7])
    light /= np.linalg.norm(light)
    elev, azim = np.radians([24, -65])
    view = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
    triangles, colors = [], []
    for points in parts:
        hull = ConvexHull(points)
        faces, normals = points[hull.simplices], hull.equations[:, :3]
        keep = normals @ view > 0
        if len(parts) > 1:
            keep &= ~np.all(np.isclose(faces[:, :, 0], SEAM, atol=1e-4), axis=1)
        triangles.extend(faces[keep])
        shade = .75 + .25 * np.maximum(normals[keep] @ light, 0)
        colors.extend(shade[:, None] * to_rgb(color) + .14 * np.maximum(normals[keep] @ light, 0)[:, None])
    ax.add_collection3d(Poly3DCollection(triangles, facecolors=colors, edgecolors=colors,
                                         linewidths=.26, antialiased=True))
    ax.set_proj_type("ortho")
    ax.view_init(elev=24, azim=-65)
    ax.set(xlim=(-.5, 21), ylim=(-6, 6), zlim=(0, 7), xticks=[0, 5, 10, 15, 20],
           yticks=[-5, 0, 5], zticks=[0, 3, 6])
    ax.set_box_aspect((21.5, 12, 7), zoom=1.09)
    ax.set_xlabel("脚跟 → 脚尖（cm）", labelpad=6)
    ax.set_ylabel("横向（cm）", labelpad=3)
    ax.set_zlabel("高度（cm）", labelpad=2)
    ax.tick_params(labelsize=8, pad=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((.96, .97, .97, 1))
    ax.grid(False)


def forefoot_edge(ax, parts, color, style):
    # Cut the 3D hull by the widest reference plane, retaining the real bevel.
    for points in parts:
        if not points[:, 0].min() < WIDEST < points[:, 0].max():
            continue
        faces = points[ConvexHull(points).simplices]
        a = faces.reshape(-1, 3)
        b = np.roll(faces, -1, axis=1).reshape(-1, 3)
        hit = (a[:, 0] < WIDEST) != (b[:, 0] < WIDEST)
        a, b = a[hit], b[hit]
        fraction = (WIDEST - a[:, 0]) / (b[:, 0] - a[:, 0])
        yz = 10 * (a[:, 1:] + fraction[:, None] * (b[:, 1:] - a[:, 1:]))
        polygon = yz[ConvexHull(yz).vertices]
        ax.plot(*np.vstack([polygon, polygon[0]]).T, color=color, ls=style, lw=2.1)


def three_quarter(old, new, destination):
    fig = plt.figure(figsize=(12, 7.4))
    fig.suptitle("左脚三维碰撞体", x=.05, y=.97, ha="left", fontsize=21, fontweight="bold")
    for i, (parts, color, title) in enumerate(((old, OLD, "原模型 · 整脚 CAD 凸包"),
                                               (new, NEW, "新模型 · 后段与前掌分体"))):
        ax = fig.add_axes([.015 + i * .49, .38, .47, .52], projection="3d")
        solid(ax, parts, color)
        fig.text(.05 + i * .49, .87, title, color=color, fontsize=14, weight="bold")
    edge = fig.add_axes([.10, .10, .38, .25])
    forefoot_edge(edge, old, OLD, "--")
    forefoot_edge(edge, new, NEW, "-")
    edge.set(xlim=(44, 54), ylim=(-.4, 4.5), xticks=[44, 46, 48, 50, 52, 54], yticks=[0, 2, 4],
             xlabel="距脚底中心线（mm）", ylabel="距底面（mm）", aspect="equal")
    edge.spines[["right", "top"]].set_visible(False)
    edge.grid(color="#dce5e1", linewidth=.65)
    fig.text(.54, .28, "前掌外缘截面", fontsize=15, fontweight="bold")
    fig.text(.54, .15, "在距脚跟 14.6 cm 的最宽处切开两版碰撞体。\n"
                       "新轮廓从平底到外侧壁采用约 2 mm 的圆角。\n"
                       "三维图与截面均取自 MuJoCo 编译后的网格。",
             linespacing=1.6, fontsize=11, color="#475a55")
    fig.savefig(destination, dpi=165, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_ASSET_DIR / "robot.xml")
    parser.add_argument("--before", type=Path,
                        help="Old robot.xml. By default use the original visual CAD hull, "
                             "which the old collider used verbatim.")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[2] / "media")
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.model))
    new = foot_parts(model, 3)
    old = (foot_parts(mujoco.MjModel.from_xml_path(str(args.before)), 3)
           if args.before else foot_parts(model, 2))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
                         "font.size": 11, "text.color": INK, "axes.labelcolor": INK})
    top_view(old, new, args.out_dir / "khan_sole_top_view.png")
    three_quarter(old, new, args.out_dir / "khan_sole_comparison_3d.png")


if __name__ == "__main__":
    main()

"""Export the current training MJCF geometry and inertials to a portable URDF.

URDF has no capsule primitive, armature or servo gains. Capsules become single
convex meshes; parameters without a standard URDF representation go in a JSON
sidecar. Neither robot.xml nor the original source.urdf is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = ROOT / "assets/robots/khan_mini"


def vector(values) -> str:
    return " ".join(format(float(x), ".17g") for x in np.asarray(values).flat)


def rotation(quat) -> np.ndarray:
    return Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).as_matrix()


def origin(parent, pos, quat) -> None:
    ET.SubElement(parent, "origin", xyz=vector(pos),
                  rpy=vector(Rotation.from_matrix(rotation(quat)).as_euler("xyz")))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convex_obj(path: Path, vertices: np.ndarray) -> dict:
    """Use the compiled mesh frame, with outward winding and no CAD concavities."""
    hull = ConvexHull(vertices)
    indices = hull.vertices
    mapping = np.full(len(vertices), -1, dtype=int)
    mapping[indices] = np.arange(len(indices))
    faces = hull.simplices.copy()
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    reverse = np.einsum("ij,ij->i", normals, hull.equations[:, :3]) < 0
    faces[reverse] = faces[reverse][:, [0, 2, 1]]
    lines = ["# Training collision hull; coordinates in the exported collision frame."]
    lines.extend("v " + vector(v) for v in vertices[indices])
    lines.extend("f " + " ".join(str(int(i) + 1) for i in row) for row in mapping[faces])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return {"vertices": len(indices), "triangles": len(faces), "sha256": sha256(path)}


def capsule_vertices(radius: float, half_length: float) -> np.ndarray:
    # The support error is bounded by r*(1-cos(pi/64)*cos(pi/96)):
    # below 0.087 mm for the largest current capsule (r=50 mm).
    azimuth = np.arange(96) * (2 * np.pi / 96)
    points = []
    for side in (-1, 1):
        points.append([0, 0, side * (half_length + radius)])
        for theta in np.linspace(0, np.pi / 2, 17)[1:]:
            ring_radius = radius * np.sin(theta)
            z = side * (half_length + radius * np.cos(theta))
            points.extend(np.column_stack((ring_radius * np.cos(azimuth),
                                           ring_radius * np.sin(azimuth),
                                           np.full(96, z))))
    return np.asarray(points)


def mesh_vertices(model, geom_id: int) -> np.ndarray:
    mesh_id = int(model.geom_dataid[geom_id])
    start = int(model.mesh_vertadr[mesh_id])
    return np.asarray(model.mesh_vert[start:start + model.mesh_vertnum[mesh_id]], dtype=float)


def inertia_matrix(model, body_id: int) -> np.ndarray:
    r = rotation(model.body_iquat[body_id])
    return (r * model.body_inertia[body_id]) @ r.T


def support(model, geom_id: int, directions: np.ndarray) -> np.ndarray:
    """Collision support in its link frame, including MuJoCo mesh recentering."""
    local = directions @ rotation(model.geom_quat[geom_id])
    kind = model.geom_type[geom_id]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        # Bound memory when checking high-resolution source CAD meshes.
        verts = mesh_vertices(model, geom_id)
        values = np.concatenate([(verts @ part.T).max(axis=0)
                                 for part in np.array_split(local, 16)])
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        values = model.geom_size[geom_id, 0] + model.geom_size[geom_id, 1] * abs(local[:, 2])
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        values = abs(local) @ model.geom_size[geom_id]
    else:
        raise ValueError(f"Unsupported collision primitive: {kind}")
    return values + directions @ model.geom_pos[geom_id]


def validate(xml_model, path: Path, profile: dict, records: list[dict]) -> dict:
    """Reload the saved URDF and independently check FK, inertials and geometry."""
    spec = mujoco.MjSpec.from_file(str(path))
    root = spec.body(profile["root_body"])
    root.add_freejoint(name="floating_base")
    root.pos = xml_model.body(profile["root_body"]).pos
    root.quat = xml_model.body(profile["root_body"]).quat
    model = spec.compile()
    if (model.nbody, model.nq, model.nv) != (xml_model.nbody, xml_model.nq, xml_model.nv):
        raise ValueError("Export changed link/joint counts")
    rng = np.random.default_rng(0)
    directions = np.concatenate((np.eye(3), -np.eye(3), rng.normal(size=(512, 3))))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    geometry_errors = []
    for record in records:
        before = xml_model.geom(record["name"]).id
        after = model.geom(record["name"]).id
        if xml_model.body(int(xml_model.geom_bodyid[before])).name != model.body(
                int(model.geom_bodyid[after])).name:
            raise ValueError("Collision attached to a different link")
        error = float(np.max(abs(support(xml_model, before, directions)
                                 - support(model, after, directions))))
        tolerance = 1e-4 if record["source_type"] == "capsule" else 1e-6
        if error > tolerance:
            raise ValueError(f"Collision geometry differs: {record['name']}: {error} m")
        geometry_errors.append({"name": record["name"], "max_support_error_m": error})
    max_mass_error = max_com_error = max_inertia_error = 0.0
    for b in range(1, xml_model.nbody):
        other = model.body(xml_model.body(b).name).id
        max_mass_error = max(max_mass_error, abs(float(model.body_mass[other]
                                                     - xml_model.body_mass[b])))
        max_com_error = max(max_com_error, float(abs(model.body_ipos[other]
                                                    - xml_model.body_ipos[b]).max()))
        max_inertia_error = max(max_inertia_error, float(abs(inertia_matrix(model, other)
                                                            - inertia_matrix(xml_model, b)).max()))
    if max(max_mass_error, max_com_error, max_inertia_error) > 1e-10:
        raise ValueError("Inertial properties changed")
    a, b = mujoco.MjData(xml_model), mujoco.MjData(model)
    max_fk_error = max_orientation_error = 0.0
    names = profile["joint_names"]
    # Home pose, zero pose, and 16 deterministic in-range poses, with root rotation.
    for trial in range(18):
        root_q = Rotation.from_rotvec(rng.normal(size=3) * 0.3).as_quat()[[3, 0, 1, 2]]
        a.qpos[:7] = b.qpos[:7] = np.r_[profile["initial_pos"], root_q]
        for name in names:
            ja, jb = xml_model.joint(name).id, model.joint(name).id
            if trial == 0:
                value = profile["default_joint_pos"][name]
            elif trial == 1:
                value = 0.0
            else:
                value = rng.uniform(*xml_model.jnt_range[ja])
            a.qpos[xml_model.jnt_qposadr[ja]] = value
            b.qpos[model.jnt_qposadr[jb]] = value
            np.testing.assert_allclose(model.jnt_range[jb], xml_model.jnt_range[ja], atol=1e-12)
        mujoco.mj_forward(xml_model, a)
        mujoco.mj_forward(model, b)
        for body_id in range(1, xml_model.nbody):
            other = model.body(xml_model.body(body_id).name).id
            max_fk_error = max(max_fk_error, float(abs(a.xpos[body_id] - b.xpos[other]).max()))
            max_orientation_error = max(max_orientation_error,
                                        float(abs(a.xmat[body_id] - b.xmat[other]).max()))
    if max(max_fk_error, max_orientation_error) > 1e-10:
        raise ValueError("Export changed forward kinematics")
    collisions = np.count_nonzero(model.geom_contype | model.geom_conaffinity)
    if collisions != len(records):
        raise ValueError("Export changed collision count")
    return {"passed": True, "links": model.nbody - 1, "actuated_joints": len(names),
            "collision_geometries": int(collisions), "fk_test_poses": 18,
            "max_fk_position_error_m": max_fk_error,
            "max_fk_rotation_matrix_error": max_orientation_error,
            "max_mass_error_kg": max_mass_error, "max_com_error_m": max_com_error,
            "max_inertia_matrix_error_kg_m2": max_inertia_error,
            "collision_support_directions": len(directions),
            "geometry_errors": geometry_errors}


def export_urdf(asset_dir: Path, output: Path) -> dict:
    xml = asset_dir / "robot.xml"
    template = asset_dir / "source.urdf"
    profile_path = asset_dir / "profile.json"
    if output.resolve() in {p.resolve() for p in (xml, template, profile_path)}:
        raise ValueError("Output must not overwrite robot.xml, source.urdf or profile.json")
    sources = {str(p): sha256(p) for p in (xml, template, profile_path)}
    profile = json.loads(profile_path.read_text())
    spec = mujoco.MjSpec.from_file(str(xml))
    model = spec.compile()
    doc = ET.parse(template).getroot()
    doc.insert(0, ET.Comment(
        " Generated from current robot.xml. Collision meshes are explicit convex hulls; "
        "capsule surface approximation <0.1 mm. See new.urdf.json for training-only parameters. "))
    output.parent.mkdir(parents=True, exist_ok=True)
    hull_dir = output.parent / "meshes/collision_hulls"
    records = []
    for link in doc.findall("link"):
        name = link.attrib["name"]
        body_id = model.body(name).id
        for item in list(link):
            if item.tag in ("visual", "collision", "inertial"):
                link.remove(item)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz=vector(model.body_ipos[body_id]), rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=vector([model.body_mass[body_id]]))
        matrix = inertia_matrix(model, body_id)
        ET.SubElement(inertial, "inertia", **{
            key: vector([matrix[i, j]]) for key, i, j in
            (("ixx", 0, 0), ("ixy", 0, 1), ("ixz", 0, 2),
             ("iyy", 1, 1), ("iyz", 1, 2), ("izz", 2, 2))})
        for geom in spec.body(name).geoms:
            gid = model.geom(geom.name).id
            collision = bool(geom.contype | geom.conaffinity)
            element = ET.SubElement(link, "collision" if collision else "visual", name=geom.name)
            # Compiled collision pose includes CAD mesh recentering. Visuals keep
            # raw CAD files and therefore use the uncompiled specification pose.
            origin(element, model.geom_pos[gid] if collision else geom.pos,
                   model.geom_quat[gid] if collision else geom.quat)
            geometry = ET.SubElement(element, "geometry")
            kind = geom.type
            info = {"name": geom.name, "link": name,
                    "source_type": kind.name.removeprefix("mjGEOM_").lower()}
            if collision and kind in (mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_CAPSULE):
                vertices = (mesh_vertices(model, gid) if kind == mujoco.mjtGeom.mjGEOM_MESH
                            else capsule_vertices(*geom.size[:2]))
                target = hull_dir / f"{geom.name}.obj"
                info["mesh"] = str(target.relative_to(output.parent))
                info["mesh_info"] = convex_obj(target, vertices)
                ET.SubElement(geometry, "mesh", filename=info["mesh"])
                if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    info["capsule_max_surface_error_bound_m"] = float(
                        geom.size[0] * (1 - np.cos(np.pi / 64) * np.cos(np.pi / 96)))
            elif kind == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = spec.mesh(geom.meshname)
                path = (xml.parent / mesh.file).resolve()
                ET.SubElement(geometry, "mesh", filename=os.path.relpath(path, output.parent),
                              scale=vector(mesh.scale))
            elif kind == mujoco.mjtGeom.mjGEOM_BOX:
                ET.SubElement(geometry, "box", size=vector(2 * geom.size))
            else:
                raise ValueError(f"Unsupported visual/collision shape: {geom.name}, {kind}")
            if collision:
                info.update({"friction": model.geom_friction[gid].tolist(),
                             "condim": int(model.geom_condim[gid]),
                             "priority": int(model.geom_priority[gid]),
                             "contype": int(model.geom_contype[gid]),
                             "conaffinity": int(model.geom_conaffinity[gid])})
                records.append(info)
            else:
                material = ET.SubElement(element, "material", name=f"{geom.name}_material")
                ET.SubElement(material, "color", rgba=vector(geom.rgba))
    for joint in doc.findall("joint"):
        name = joint.attrib["name"]
        child = joint.find("child").attrib["link"]
        bid = model.body(child).id
        parent = joint.find("parent").attrib["link"]
        if model.body(int(model.body_parentid[bid])).name != parent:
            raise ValueError(f"Template topology differs from XML: {name}")
        for item in list(joint):
            if item.tag in ("origin", "axis", "limit", "dynamics"):
                joint.remove(item)
        origin(joint, model.body_pos[bid], model.body_quat[bid])
        if joint.attrib["type"] == "fixed":
            continue
        jid = model.joint(name).id
        if not np.allclose(model.jnt_pos[jid], 0, atol=1e-12):
            raise ValueError("Nonzero joint pivot requires a separate URDF joint frame")
        ET.SubElement(joint, "axis", xyz=vector(model.jnt_axis[jid]))
        ET.SubElement(joint, "limit", lower=vector([model.jnt_range[jid, 0]]),
                      upper=vector([model.jnt_range[jid, 1]]),
                      effort=vector([profile["effort_limit"][name]]),
                      velocity=vector([profile["velocity_limit"][name]]))
        dof = int(model.jnt_dofadr[jid])
        # Servo derivative gain is NOT passive joint damping.
        ET.SubElement(joint, "dynamics", damping=vector([model.dof_damping[dof]]),
                      friction=vector([model.dof_frictionloss[dof]]))
    for old in doc.findall("mujoco"):
        doc.remove(old)
    extension = ET.SubElement(doc, "mujoco")
    ET.SubElement(extension, "compiler", fusestatic="false", discardvisual="false",
                  strippath="false", balanceinertia="false")
    ET.indent(doc)
    ET.ElementTree(doc).write(output, encoding="utf-8", xml_declaration=True)
    report = {"urdf": str(output.resolve()), "source_sha256": sources,
              "mujoco_version": mujoco.__version__, "collisions": records,
              "training_parameters": {key: profile[key] for key in
                                      ("joint_names", "root_body", "anchor_body", "com_body",
                                       "initial_pos", "default_joint_pos", "armature", "stiffness",
                                       "damping", "action_scale", "effort_limit", "velocity_limit",
                                       "soft_joint_pos_limit_factor", "self_collisions")},
              "notes": [
                  "The root has no world pose in URDF; initial_pos is a spawn setting.",
                  ("The standard URDF does not encode armature, PD gains, action scale, "
                   "contact masks/condim/priority/friction, actuator delay or domain randomization."),
                  "Servo kd is stored in training_parameters.damping, not joint dynamics damping.",
                  "Velocity metadata comes from profile.json; training builtin servos do not enforce it.",
                  "Four capsules become single convex meshes, with surface error below 0.1 mm.",
                  "Eight CAD collision meshes are explicit convex hulls, including the foot instep.",
                  "Render/load the URDF with its meshes directory, not as a standalone file."]}
    report["validation"] = validate(model, output, profile, records)
    if any(sha256(Path(p)) != value for p, value in sources.items()):
        raise ValueError("Source assets changed during export")
    report["urdf_sha256"] = sha256(output)
    output.with_suffix(".urdf.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, default=ASSET_DIR)
    parser.add_argument("--out", type=Path, default=ASSET_DIR / "new.urdf")
    args = parser.parse_args()
    report = export_urdf(args.asset_dir.resolve(), args.out.resolve())
    print(json.dumps({"urdf": report["urdf"], "validation": report["validation"]}, indent=2))


if __name__ == "__main__":
    main()

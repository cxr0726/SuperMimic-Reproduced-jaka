"""Convert the Khan_mini URDF and Isaac Lab settings into portable mjlab assets."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from supermimic.robots.khan_mini import DEFAULT_ASSET_DIR, ROOT

DEFAULT_CONFIG = (
    ROOT / "ref_thirdparty/jaka_code/Textop/TextOpTracker/source/textop_tracker"
    "/textop_tracker/robots/K1L.py"
)
FOOT_BODIES = ("Left_ankle_roll_Link", "Right_ankle_roll_Link")
# Fit the central torso shell (126--140 mm wide), not its wider upper flanges.
# Depth, height and placement stay unchanged; this is a collision-core approximation.
WAIST_COLLISION_HALF_SIZE = (0.075, 0.07, 0.15)
TRACKED_BODIES = (
    "base_link",
    "Left_hip_roll_Link", "Left_knee_Link", "Left_ankle_roll_Link",
    "Right_hip_roll_Link", "Right_knee_Link", "Right_ankle_roll_Link",
    "waist_yaw_Link",
    "Left_shoulder_roll_Link", "Left_elbow_Link", "Left_wrist_yaw_Link",
    "Right_shoulder_roll_Link", "Right_elbow_Link", "Right_wrist_yaw__Link",
    "Neck_pitch_Link",
)


def _keywords(node: ast.AST) -> dict[str, ast.AST]:
    if not isinstance(node, ast.Call):
        raise TypeError("Expected a configuration constructor")
    return {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg is not None}


def read_source_config(path: Path) -> dict:
    """Read only literal Khan_mini_CFG values, without importing Isaac Lab."""
    assignments = [
        node.value for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "Khan_mini_CFG" for t in node.targets)
    ]
    if len(assignments) != 1:
        raise ValueError(f"Expected one Khan_mini_CFG assignment in {path}")
    cfg = _keywords(assignments[0])
    spawn = _keywords(cfg["spawn"])
    initial = _keywords(cfg["init_state"])
    articulation = _keywords(spawn["articulation_props"])
    # Resolve the source's f"{ASSET_DIR}/..." explicitly, never through eval().
    asset = spawn["asset_path"]
    if not isinstance(asset, ast.JoinedStr):
        raise TypeError("Expected Khan_mini's ASSET_DIR-based URDF path")
    parts = []
    for part in asset.values:
        if isinstance(part, ast.Constant) and isinstance(part.value, str):
            parts.append(part.value)
        elif (
            isinstance(part, ast.FormattedValue) and isinstance(part.value, ast.Name)
            and part.value.id == "ASSET_DIR"
        ):
            parts.append(str(path.parent.parent / "assets"))
        else:
            raise ValueError("Unsupported expression in asset_path")
    actuator_nodes = cfg["actuators"]
    if not isinstance(actuator_nodes, ast.Dict):
        raise TypeError("Expected a literal actuator mapping")
    actuators = {
        ast.literal_eval(key): {k: ast.literal_eval(v) for k, v in _keywords(value).items()}
        for key, value in zip(actuator_nodes.keys, actuator_nodes.values, strict=True)
    }
    return {
        "urdf": Path("".join(parts)),
        "initial_pos": ast.literal_eval(initial["pos"]),
        "default_joint_pos": ast.literal_eval(initial["joint_pos"]),
        "soft_joint_pos_limit_factor": ast.literal_eval(cfg["soft_joint_pos_limit_factor"]),
        "self_collisions": ast.literal_eval(articulation["enabled_self_collisions"]),
        "replace_cylinders_with_capsules": ast.literal_eval(spawn["replace_cylinders_with_capsules"]),
        "fix_base": ast.literal_eval(spawn["fix_base"]),
        "actuators": actuators,
    }


def _resolve(table: float | dict, name: str) -> float:
    if isinstance(table, dict):
        values = [value for pattern, value in table.items() if re.fullmatch(pattern, name)]
        if len(values) != 1:
            raise ValueError(f"Expected exactly one parameter match for {name}, got {values}")
        table = values[0]
    result = float(table)
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"Invalid actuator parameter for {name}: {result}")
    return result


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _source_label(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _precise_xml(spec: mujoco.MjSpec) -> str:
    """Avoid to_xml()'s six-digit rounding of mass, inertia and transforms."""
    model = spec.compile()
    root = ET.fromstring(spec.to_xml())

    def vector(values) -> str:
        return " ".join(format(float(value), ".17g") for value in values)

    for body in root.findall(".//body"):
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body.attrib["name"])
        body.set("pos", vector(model.body_pos[index]))
        body.set("quat", vector(model.body_quat[index]))
        inertial = body.find("inertial")
        if inertial is not None:
            inertial.set("mass", format(float(model.body_mass[index]), ".17g"))
            inertial.set("pos", vector(model.body_ipos[index]))
            inertial.set("quat", vector(model.body_iquat[index]))
            inertial.set("diaginertia", vector(model.body_inertia[index]))
    for geom in root.findall(".//geom"):
        source = spec.geom(geom.attrib["name"])
        geom.set("pos", vector(source.pos))
        geom.set("quat", vector(source.quat))
        if "size" in geom.attrib:
            geom.set("size", vector(source.size[:len(geom.attrib["size"].split())]))
    ET.indent(root)
    return ET.tostring(root, encoding="unicode") + "\n"


def _convert(urdf_path: Path, config_path: Path, source: dict, stage: Path) -> None:
    doc = ET.parse(urdf_path).getroot()
    links = [link.attrib["name"] for link in doc.findall("link")]
    joints = [joint for joint in doc.findall("joint") if joint.attrib["type"] != "fixed"]
    joint_names = [joint.attrib["name"] for joint in joints]
    if len(joint_names) != 27 or set(joint_names) != set(source["default_joint_pos"]):
        raise ValueError("URDF joints and the 27-joint Khan_mini_CFG initial pose must match")
    if source["fix_base"]:
        raise ValueError("This converter expects the floating-base Khan_mini configuration")
    profile = {
        "robot": "khan_mini", "schema_version": 1,
        "xml_path": "robot.xml", "num_joints": 27, "joint_names": joint_names,
        "body_names": links, "root_body": "base_link", "anchor_body": "base_link",
        "com_body": "waist_yaw_Link",
        "neck_joint_names": ["Neck_yaw_joint", "Neck_pitch_joint"],
        "training_semantics_version": 2,
        "tracked_bodies": list(TRACKED_BODIES), "foot_bodies": list(FOOT_BODIES),
        "foot_geom_regex": r"^(Left|Right)_ankle_roll_Link_\d+_collision$",
        "initial_pos": source["initial_pos"], "standing_height": source["initial_pos"][2],
        "default_joint_pos": source["default_joint_pos"],
        "soft_joint_pos_limit_factor": source["soft_joint_pos_limit_factor"],
        "self_collisions": source["self_collisions"],
        "replace_cylinders_with_capsules": source["replace_cylinders_with_capsules"],
        "velocity_limit_enforced": False,
        "dimensions": {"action": 27, "state": 67, "goal_per_frame": 64,
                       "goal_10_frames": 640, "proprio_10_frames": 870},
    }
    parameters = {
        "stiffness": "stiffness", "damping": "damping", "armature": "armature",
        "effort_limit": "effort_limit_sim", "velocity_limit": "velocity_limit_sim",
    }
    for key, source_key in parameters.items():
        profile[key] = {}
        for name in joint_names:
            matches = [
                actuator for actuator in source["actuators"].values()
                if any(re.fullmatch(pattern, name) for pattern in actuator["joint_names_expr"])
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected one actuator group for {name}")
            profile[key][name] = _resolve(matches[0][source_key], name)
    profile["urdf_limits"] = {
        j.attrib["name"]: {k: float(v) for k, v in j.find("limit").attrib.items()}
        for j in joints
    }
    # Local corrections requested for Khan_mini. Keep the third-party input
    # unchanged and make the overrides explicit so rebuilding preserves them.
    original_waist_velocity = profile["velocity_limit"]["waist_yaw_joint"]
    profile["velocity_limit"]["waist_yaw_joint"] = profile["urdf_limits"]["waist_yaw_joint"]["velocity"]
    profile["action_scale"] = dict.fromkeys(joint_names, 0.5)
    profile["parameter_overrides"] = {
        "waist_velocity_limit": {
            "source_config_value": original_waist_velocity,
            "effective_value": profile["velocity_limit"]["waist_yaw_joint"],
            "effective_source": "URDF joint limit velocity",
        },
        "action_scale": {"value": 0.5, "source": "user-specified uniform joint action scale"},
        "nonfoot_sliding_friction": {
            "value": 1.0, "source": "user-requested alignment with mjlab G1 non-foot contacts",
        },
    }
    profile["contact_friction"] = {
        "foot": [0.6, 0.005, 0.0001], "nonfoot": [1.0, 0.005, 0.0001],
    }
    extension = doc.find("mujoco")
    if extension is not None:
        doc.remove(extension)
    extension = ET.SubElement(doc, "mujoco")
    ET.SubElement(extension, "compiler", fusestatic="false", discardvisual="false")
    mesh_hashes = {}
    for mesh in doc.findall(".//mesh"):
        filename = mesh.attrib["filename"]
        original = (urdf_path.parent / filename).resolve()
        if not original.is_file():
            raise FileNotFoundError(original)
        relative = Path("meshes") / original.name
        if str(relative) in mesh_hashes and mesh_hashes[str(relative)] != _sha256(original):
            raise ValueError(f"Mesh basename collision: {original.name}")
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(original, destination)
        mesh_hashes[str(relative)] = _sha256(original)
        mesh.set("filename", str(relative))
    ET.indent(doc)
    converted_urdf = stage / "source.urdf"
    ET.ElementTree(doc).write(converted_urdf, encoding="utf-8", xml_declaration=True)
    spec = mujoco.MjSpec.from_file(str(converted_urdf))
    spec.body("base_link").add_freejoint(name="floating_base")
    spec.body("base_link").pos = source["initial_pos"]
    for name in joint_names:
        spec.joint(name).armature = profile["armature"][name]
    for index, geom in enumerate(spec.geoms):
        if geom.group == 1:
            geom.name = f"{geom.parent.name}_{index}_visual"
            geom.group, geom.contype, geom.conaffinity = 2, 0, 0
        else:
            geom.name = f"{geom.parent.name}_{index}_collision"
            geom.group = 3
            geom.contype = 1
            geom.conaffinity = 1 if source["self_collisions"] else 0
            foot = geom.parent.name in FOOT_BODIES
            geom.condim = 3 if foot else 1
            geom.priority = 1 if foot else 0
            # Match G1: feet use sliding friction 0.6; other contacts use 1.0.
            geom.friction = profile["contact_friction"]["foot" if foot else "nonfoot"]
            geom.rgba = [0.2, 0.6, 0.2, 0.3]
            if source["replace_cylinders_with_capsules"] and geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
    waist_boxes = [
        geom for geom in spec.geoms
        if geom.parent.name == "waist_yaw_Link" and geom.group == 3
    ]
    if len(waist_boxes) != 1 or waist_boxes[0].type != mujoco.mjtGeom.mjGEOM_BOX:
        raise ValueError("Expected one waist_yaw_Link collision box")
    waist_box = waist_boxes[0]
    original_waist_size = np.array(waist_box.size, dtype=float)
    if not np.allclose(original_waist_size, (0.075, 0.1, 0.15), rtol=0, atol=1e-9):
        raise ValueError("Source waist box changed; review the local collision-width override")
    waist_box.size[:] = WAIST_COLLISION_HALF_SIZE
    profile["parameter_overrides"]["waist_collision_box"] = {
        "geom": waist_box.name,
        "source_half_size_m": original_waist_size.tolist(),
        "effective_half_size_m": list(WAIST_COLLISION_HALF_SIZE),
        "source_full_size_m": (2 * original_waist_size).tolist(),
        "effective_full_size_m": (2 * np.array(WAIST_COLLISION_HALF_SIZE)).tolist(),
        "local_position_m": np.array(waist_box.pos).tolist(),
        "source": "user-requested narrower torso collision box",
        "rationale": "Fit the 126--140 mm central visual shell; wider upper flanges are outside this simplified collision core. Preserve X/Z size, pose and explicit inertial properties",
    }
    # Export only portable mesh paths; MuJoCo may resolve URDF filenames absolutely.
    for mesh in spec.meshes:
        mesh.file = str(Path("meshes") / Path(mesh.file).name)
    # Compile relative files against the converted URDF's directory.
    xml_path = stage / "robot.xml"
    xml_path.write_text(_precise_xml(spec))
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    actual_joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                     for i in range(1, model.njnt)]
    if (model.nq, model.nv, model.nbody) != (34, 33, 46) or actual_joints != joint_names:
        raise ValueError("Conversion changed Khan_mini's topology or joint order")
    for name in TRACKED_BODIES:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) < 0:
            raise ValueError(f"Missing tracked body: {name}")
    data = mujoco.MjData(model)
    data.qpos[:3] = source["initial_pos"]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:] = [profile["default_joint_pos"][name] for name in joint_names]
    if not ((data.qpos[7:] >= model.jnt_range[1:, 0])
            & (data.qpos[7:] <= model.jnt_range[1:, 1])).all():
        raise ValueError("Initial joint pose lies outside URDF limits")
    mujoco.mj_forward(model, data)
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.xpos).all():
        raise ValueError("Nonfinite initial forward kinematics")
    profile["model"] = {
        "nq": model.nq, "nv": model.nv, "nbody": model.nbody,
        "ngeom": model.ngeom, "nmesh": model.nmesh,
        "mass_kg": float(model.body_mass.sum()), "initial_self_contacts": data.ncon,
    }
    (stage / "profile.json").write_text(json.dumps(profile, indent=2, allow_nan=False) + "\n")
    provenance = {
        "source_urdf": _source_label(urdf_path), "source_urdf_sha256": _sha256(urdf_path),
        "source_config": _source_label(config_path), "source_config_sha256": _sha256(config_path),
        "source_configuration": "Khan_mini_CFG", "mujoco_version": mujoco.__version__,
        "mesh_sha256": mesh_hashes,
        "parameter_overrides": profile["parameter_overrides"],
        "conversion": {
            "fixed_links": "preserved", "visual_meshes": "preserved",
            "freejoint": "floating_base", "cylinders": "capsules with original radius/axis half-length",
            "actuators": "created by get_khan_mini_cfg from profile.json",
            "velocity_limits": "recorded only; not enforced by builtin position servos",
            "contacts": "self-collision masks retained; feet condim=3, other contacts condim=1",
        },
        "output_sha256": {name: _sha256(stage / name)
                          for name in ("robot.xml", "source.urdf", "profile.json")},
    }
    (stage / "source.json").write_text(json.dumps(provenance, indent=2) + "\n")


def build_khan_mini(
    urdf_path: str | Path | None = None,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Build in a staging directory; require --force before replacing edited assets."""
    config = Path(config_path or DEFAULT_CONFIG).resolve()
    source = read_source_config(config)
    urdf = Path(urdf_path or source["urdf"]).resolve()
    target = Path(output_dir or DEFAULT_ASSET_DIR).resolve()
    if target.exists() and not force:
        raise FileExistsError(f"{target} already exists; pass --force to rebuild it")
    if target in (ROOT, urdf.parent) or urdf.is_relative_to(target) or config.is_relative_to(target):
        raise ValueError("Output directory must not contain or replace source files")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".khan-mini-", dir=target.parent) as temporary:
        stage = Path(temporary) / "asset"
        stage.mkdir()
        _convert(urdf, config, source, stage)
        if target.exists():
            shutil.rmtree(target)
        stage.rename(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    target = build_khan_mini(args.urdf, args.config, args.output_dir, args.force)
    profile = json.loads((target / "profile.json").read_text())
    print(json.dumps({"asset_dir": str(target), **profile["model"]}, indent=2))


if __name__ == "__main__":
    main()

"""CPU regressions for the portable Khan_mini asset and its mjlab configuration."""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from supermimic.robots.build_khan_mini import build_khan_mini
from supermimic.robots.khan_mini import (
    get_khan_mini_action_cfg,
    get_khan_mini_cfg,
    load_profile,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (
    ROOT / "ref_thirdparty/jaka_code/Textop/TextOpTracker/source/textop_tracker/textop_tracker"
)
SOURCE_URDF = SOURCE_ROOT / "assets/Khan_mini_simplified/khan.urdf"
SOURCE_CONFIG = SOURCE_ROOT / "robots/K1L.py"

pytestmark = pytest.mark.skipif(
    not SOURCE_URDF.is_file() or not SOURCE_CONFIG.is_file(),
    reason="local Khan_mini URDF and source configuration are required",
)

# Independently transcribed from Khan_mini_CFG with the user-requested URDF
# waist velocity correction. Each row is kp, kd, armature, effort, velocity.
MOTOR_GROUPS = (
    (("hip_pitch_joint", "hip_roll_joint", "hip_yaw_joint", "knee_joint"),
     (187.0, 18.7, 0.03, 120.0, 11.52)),
    (("ankle_pitch_joint",), (100.0, 2.0, 0.016, 96.0, 14.6)),
    (("ankle_roll_joint",), (50.0, 0.5, 0.016, 96.0, 14.6)),
    (("shoulder_pitch_joint", "shoulder_roll_joint"), (102.0, 10.2, 0.016, 96.0, 11.5)),
    (("shoulder_yaw_joint", "elbow_joint"), (40.8, 4.0, 0.01, 36.0, 15.7)),
    (("wrist_roll_joint", "wrist_yaw_joint"), (6.7, 0.67, 0.01, 8.0, 12.5)),
    (("waist_yaw_joint",), (187.0, 18.7, 0.03, 120.0, 11.519)),
    (("Neck_yaw_joint", "Neck_pitch_joint"), (6.7, 0.67, 0.01, 8.0, 8.0)),
)
PARAMETER_NAMES = ("stiffness", "damping", "armature", "effort_limit", "velocity_limit")


def expected_parameters(name):
    matches = [values for suffixes, values in MOTOR_GROUPS if name.endswith(suffixes)]
    assert len(matches) == 1, name
    return dict(zip(PARAMETER_NAMES, matches[0], strict=True))


def expected_home(names):
    home = dict.fromkeys(names, 0.0)
    for side in ("Left", "Right"):
        home[f"{side}_shoulder_roll_joint"] = -1.57
        home[f"{side}_elbow_joint"] = 1.57
        home[f"{side}_wrist_yaw_joint"] = 0.3
    return home


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    output = tmp_path_factory.mktemp("khan_mini") / "robot_asset"
    result = build_khan_mini(SOURCE_URDF, SOURCE_CONFIG, output)
    assert result == output
    return output


def test_conversion_preserves_original_tree_mass_and_joint_limits(bundle):
    source = ET.parse(SOURCE_URDF).getroot()
    model = mujoco.MjModel.from_xml_path(str(bundle / "robot.xml"))
    assert (model.nq, model.nv, model.njnt, model.nbody, model.ngeom, model.nmesh) == (
        34, 33, 28, 46, 71, 44
    )
    assert model.nu == 0 and model.nkey == 0
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert (model.jnt_type[1:] == mujoco.mjtJoint.mjJNT_HINGE).all()
    np.testing.assert_allclose(model.qpos0[:7], [0, 0, 0.65, 1, 0, 0, 0])
    assert model.body_mass.sum() == pytest.approx(40.8325349886391, abs=1e-9)

    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
             for i in range(1, model.nbody)}
    assert names == {link.attrib["name"] for link in source.findall("link")}
    for link in source.findall("link"):
        body = model.body(link.attrib["name"])
        mass = float(link.find("inertial/mass").attrib["value"])
        assert body.mass[0] == pytest.approx(mass, rel=1e-5, abs=1e-10)
    for joint in source.findall("joint"):
        if joint.attrib["type"] == "fixed":
            continue
        limit = joint.find("limit")
        np.testing.assert_allclose(
            model.joint(joint.attrib["name"]).range,
            [float(limit.attrib["lower"]), float(limit.attrib["upper"])],
            atol=1e-7,
        )


def test_bundle_contains_relative_meshes_and_loads_after_relocation(bundle, tmp_path):
    assert (bundle / "profile.json").is_file()
    assert (bundle / "source.urdf").is_file()
    assert len(list((bundle / "meshes").glob("*.STL"))) == 44
    xml = ET.parse(bundle / "robot.xml").getroot()
    compiler = xml.find("compiler")
    mesh_dir = Path(compiler.attrib.get("meshdir", "")) if compiler is not None else Path()
    assert not mesh_dir.is_absolute()
    for mesh in xml.findall("./asset/mesh"):
        filename = Path(mesh.attrib["file"])
        assert not filename.is_absolute()
        assert ".." not in filename.parts and ".." not in mesh_dir.parts
        assert (bundle / mesh_dir / filename).is_file()

    relocated = tmp_path / "unrelated_location" / "khan_mini"
    shutil.copytree(bundle, relocated)
    model = mujoco.MjModel.from_xml_path(str(relocated / "robot.xml"))
    configured = get_khan_mini_cfg(relocated).build().spec.compile()
    assert (model.nq, model.nmesh, configured.nu) == (34, 44, 27)
    assert load_profile(relocated) == load_profile(bundle)


def test_saved_model_matches_original_urdf_forward_kinematics(bundle):
    # Direct URDF import is an independent baseline for the converted/saved MJCF.
    # Only retain visuals/fixed links and add the originally floating root.
    source = ET.parse(SOURCE_URDF).getroot()
    compiler = ET.SubElement(source, "mujoco")
    ET.SubElement(compiler, "compiler", discardvisual="false", fusestatic="false")
    for mesh in source.findall(".//mesh"):
        mesh.set("filename", str((SOURCE_URDF.parent / mesh.attrib["filename"]).resolve()))
    spec = mujoco.MjSpec.from_string(ET.tostring(source, encoding="unicode"))
    spec.body("base_link").add_freejoint(name="reference_root")
    reference = spec.compile()
    converted = mujoco.MjModel.from_xml_path(str(bundle / "robot.xml"))
    reference_data, converted_data = mujoco.MjData(reference), mujoco.MjData(converted)
    body_names = [link.attrib["name"] for link in source.findall("link")]
    reference_ids = [reference.body(name).id for name in body_names]
    converted_ids = [converted.body(name).id for name in body_names]
    names = [reference.joint(i).name for i in range(1, reference.njnt)]
    home = expected_home(names)
    rng = np.random.default_rng(17)
    poses = [np.array([home[name] for name in names])]
    lo, hi = reference.jnt_range[1:].T
    poses.extend(rng.uniform(lo, hi) for _ in range(3))
    for pose in poses:
        for data in (reference_data, converted_data):
            data.qpos[:7] = [0.1, -0.2, 0.73, 1, 0, 0, 0]
            data.qpos[7:] = pose
        mujoco.mj_forward(reference, reference_data)
        mujoco.mj_forward(converted, converted_data)
        np.testing.assert_allclose(
            converted_data.xpos[converted_ids], reference_data.xpos[reference_ids], atol=1e-6
        )
        actual = converted_data.xquat[converted_ids]
        expected = reference_data.xquat[reference_ids]
        sign = np.where((actual * expected).sum(axis=1, keepdims=True) >= 0, 1.0, -1.0)
        np.testing.assert_allclose(actual * sign, expected, atol=1e-6)


def test_mjlab_home_and_actuators_match_khan_configuration(bundle):
    profile = load_profile(bundle)
    cfg = get_khan_mini_cfg(bundle)
    entity = cfg.build()
    model = entity.spec.compile()
    names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                  for i in range(1, model.njnt))
    assert model.nu == 27 and len(set(names)) == 27
    assert list(names) == profile["joint_names"]
    home = expected_home(names)
    assert profile["default_joint_pos"] == home
    assert cfg.init_state.joint_pos == home
    assert cfg.init_state.pos == (0.0, 0.0, 0.65)
    assert cfg.articulation.soft_joint_pos_limit_factor == 0.9

    # MuJoCo servos enforce torque limits. The source velocity values must remain
    # available as metadata, using the URDF's corrected waist velocity.
    assert profile["velocity_limit_enforced"] is False
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        expected = expected_parameters(name)
        for parameter, value in expected.items():
            assert profile[parameter][name] == value
        assert model.actuator_gainprm[actuator_id, 0] == expected["stiffness"]
        assert model.actuator_biasprm[actuator_id, 1] == -expected["stiffness"]
        assert model.actuator_biasprm[actuator_id, 2] == -expected["damping"]
        assert model.actuator_forcelimited[actuator_id]
        np.testing.assert_allclose(
            model.actuator_forcerange[actuator_id],
            [-expected["effort_limit"], expected["effort_limit"]],
        )
        assert model.dof_armature[model.jnt_dofadr[joint_id]] == expected["armature"]

    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state")
    assert model.nkey == 1 and key >= 0
    np.testing.assert_allclose(model.key_qpos[key, :7], [0, 0, 0.65, 1, 0, 0, 0])
    np.testing.assert_allclose(model.key_qpos[key, 7:], [home[name] for name in names])
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        assert model.key_ctrl[key, actuator_id] == home[name]
    limits = model.jnt_range[1:]
    middle = limits.mean(axis=1)
    half_width = np.diff(limits, axis=1).ravel() * 0.9 / 2
    assert (model.key_qpos[key, 7:] >= middle - half_width).all()
    assert (model.key_qpos[key, 7:] <= middle + half_width).all()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    assert np.isfinite(data.xpos).all()
    np.testing.assert_allclose(data.body("base_link").xpos, [0, 0, 0.65])


def test_requested_action_scale_and_velocity_correction_survive_rebuild(bundle):
    profile = load_profile(bundle)
    assert profile["action_scale"] == dict.fromkeys(profile["joint_names"], 0.5)
    assert profile["velocity_limit"]["waist_yaw_joint"] == 11.519
    assert profile["velocity_limit"]["waist_yaw_joint"] == (
        profile["urdf_limits"]["waist_yaw_joint"]["velocity"]
    )
    cfg = get_khan_mini_action_cfg(bundle)
    assert cfg.scale == profile["action_scale"]
    assert cfg.use_default_offset and cfg.noise_std == 0.1
    assert get_khan_mini_action_cfg(bundle, noise_std=0).noise_std == 0


def test_visuals_collisions_capsules_and_feet_remain_distinct(bundle):
    for model in (
        mujoco.MjModel.from_xml_path(str(bundle / "robot.xml")),
        get_khan_mini_cfg(bundle).build().spec.compile(),
    ):
        visual = model.geom_group == 2
        collision = model.geom_group == 3
        assert visual.sum() == 45 and collision.sum() == 26
        assert (visual | collision).all()
        assert (model.geom_contype[visual] == 0).all()
        assert (model.geom_conaffinity[visual] == 0).all()
        assert (model.geom_contype[collision] == 1).all()
        assert (model.geom_conaffinity[collision] == 1).all()
        assert (model.geom_type[collision] == mujoco.mjtGeom.mjGEOM_CAPSULE).sum() == 18
        assert not (model.geom_type[collision] == mujoco.mjtGeom.mjGEOM_CYLINDER).any()
        feet = []
        for geom_id in np.flatnonzero(collision):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
            body_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])
            )
            assert name.endswith("_collision") and body_name in name
            if body_name in ("Left_ankle_roll_Link", "Right_ankle_roll_Link"):
                feet.append(geom_id)
        assert len(feet) == 14
        assert (model.geom_type[feet] == mujoco.mjtGeom.mjGEOM_CAPSULE).all()
        np.testing.assert_allclose(
            model.geom_friction[feet], np.tile([0.6, 0.005, 0.0001], (14, 1))
        )
        nonfeet = np.flatnonzero(collision & ~np.isin(np.arange(model.ngeom), feet))
        np.testing.assert_allclose(
            model.geom_friction[nonfeet], np.tile([1.0, 0.005, 0.0001], (12, 1))
        )


def test_rebuild_without_force_preserves_local_edits(bundle):
    xml = bundle / "robot.xml"
    original = xml.read_bytes()
    edited = original + b"\n<!-- local user edit must survive a refused rebuild -->\n"
    try:
        xml.write_bytes(edited)
        with pytest.raises(FileExistsError, match="force"):
            build_khan_mini(SOURCE_URDF, SOURCE_CONFIG, bundle)
        assert xml.read_bytes() == edited
    finally:
        xml.write_bytes(original)

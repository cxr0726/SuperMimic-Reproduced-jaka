"""CPU regressions for the portable Khan_mini asset and its mjlab configuration."""

import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial import ConvexHull

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
# waist velocity and ankle pitch/roll PD/armature overrides. Each row is kp, kd,
# armature, effort, velocity.
MOTOR_GROUPS = (
    (("hip_pitch_joint", "hip_roll_joint", "hip_yaw_joint", "knee_joint"),
     (187.0, 18.7, 0.03, 120.0, 11.52)),
    (("ankle_pitch_joint",), (80.0, 2.0, 0.02, 96.0, 14.6)),
    (("ankle_roll_joint",), (20.0, 1.0, 0.01, 96.0, 14.6)),
    (("shoulder_pitch_joint", "shoulder_roll_joint"), (102.0, 10.2, 0.016, 96.0, 11.5)),
    (("shoulder_yaw_joint", "elbow_joint"), (40.8, 4.0, 0.01, 36.0, 15.7)),
    (("wrist_roll_joint", "wrist_yaw_joint"), (6.7, 0.67, 0.01, 8.0, 12.5)),
    (("waist_yaw_joint",), (187.0, 18.7, 0.03, 120.0, 11.519)),
    (("Neck_yaw_joint", "Neck_pitch_joint"), (6.7, 0.67, 0.01, 8.0, 8.0)),
)
PARAMETER_NAMES = ("stiffness", "damping", "armature", "effort_limit", "velocity_limit")
KNEE_NAMES = ("Left_knee_joint", "Right_knee_joint")
KNEE_HARD_LOWER = -0.15099473684210526


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


def test_conversion_preserves_tree_except_requested_foot_mass_and_knee_overrides(bundle):
    source = ET.parse(SOURCE_URDF).getroot()
    model = mujoco.MjModel.from_xml_path(str(bundle / "robot.xml"))
    assert (model.nq, model.nv, model.njnt, model.nbody, model.ngeom, model.nmesh) == (
        34, 33, 28, 46, 61, 48
    )
    assert model.nu == 0 and model.nkey == 0
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert (model.jnt_type[1:] == mujoco.mjtJoint.mjJNT_HINGE).all()
    np.testing.assert_allclose(model.qpos0[:7], [0, 0, 0.65, 1, 0, 0, 0])
    assert model.body_mass.sum() == pytest.approx(41.0325349886391, abs=1e-9)

    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
             for i in range(1, model.nbody)}
    assert names == {link.attrib["name"] for link in source.findall("link")}
    for link in source.findall("link"):
        body = model.body(link.attrib["name"])
        mass = float(link.find("inertial/mass").attrib["value"])
        if link.attrib["name"] in ("Left_ankle_roll_Link", "Right_ankle_roll_Link"):
            mass += .1
        assert body.mass[0] == pytest.approx(mass, rel=1e-5, abs=1e-10)
    for joint in source.findall("joint"):
        if joint.attrib["type"] == "fixed":
            continue
        limit = joint.find("limit")
        name = joint.attrib["name"]
        lower = KNEE_HARD_LOWER if name in KNEE_NAMES else float(limit.attrib["lower"])
        np.testing.assert_allclose(
            model.joint(name).range, [lower, float(limit.attrib["upper"])],
            rtol=0, atol=1e-12,
        )


@pytest.mark.parametrize("asset_source", ["rebuilt", "checked_in"])
def test_knee_soft_lower_and_fixed_joint_friction_match_compiled_assets(bundle, asset_source):
    asset = bundle if asset_source == "rebuilt" else ROOT / "assets/robots/khan_mini"
    profile = load_profile(asset)
    cfg = get_khan_mini_cfg(asset)
    models = (
        mujoco.MjModel.from_xml_path(str(asset / "robot.xml")),
        cfg.build().spec.compile(),
    )
    source = ET.parse(asset / "source.urdf").getroot()
    source_limits = {
        joint.attrib["name"]: joint.find("limit").attrib
        for joint in source.findall("joint") if joint.attrib["type"] != "fixed"
    }
    assert profile["joint_frictionloss"] == dict.fromkeys(profile["joint_names"], 0.1)
    factor = cfg.articulation.soft_joint_pos_limit_factor
    assert factor == 0.9
    for model in models:
        assert not (model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS)
        np.testing.assert_array_equal(model.dof_frictionloss[:6], np.zeros(6))
        assert model.njnt == 28
        for name in profile["joint_names"]:
            joint = model.joint(name)
            assert model.jnt_type[joint.id] == mujoco.mjtJoint.mjJNT_HINGE
            dof = model.jnt_dofadr[joint.id]
            assert model.dof_frictionloss[dof] == pytest.approx(0.1, abs=1e-12)
            lower, upper = joint.range
            expected_lower = (KNEE_HARD_LOWER if name in KNEE_NAMES
                              else float(source_limits[name]["lower"]))
            np.testing.assert_allclose(
                [lower, upper], [expected_lower, float(source_limits[name]["upper"])],
                rtol=0, atol=1e-12,
            )
            assert profile["urdf_limits"][name]["lower"] == pytest.approx(lower, abs=1e-12)
            assert profile["urdf_limits"][name]["upper"] == pytest.approx(upper, abs=1e-12)
            if name in KNEE_NAMES:
                assert upper == pytest.approx(2.2689, abs=1e-12)
                soft_lower = (lower + upper) / 2 - factor * (upper - lower) / 2
                assert soft_lower == pytest.approx(-0.03, abs=1e-12)

    # The generated profile documents the source/effective distinction used
    # by export metadata; the archived source URDF remains unchanged.
    override = profile["parameter_overrides"]["knee_lower_limit"]
    assert override["soft_lower_limit"] == -0.03
    for name in KNEE_NAMES:
        assert float(source_limits[name]["lower"]) == -0.349
        assert override["source_limits"][name]["lower"] == -0.349
        assert override["effective_lower_limits"][name] == pytest.approx(KNEE_HARD_LOWER, abs=1e-12)


def test_checked_in_asset_provenance_matches_generated_physics():
    asset = ROOT / "assets/robots/khan_mini"
    provenance = json.loads((asset / "source.json").read_text())
    profile = load_profile(asset)
    assert provenance["parameter_overrides"] == profile["parameter_overrides"]
    for name, expected in provenance["output_sha256"].items():
        assert hashlib.sha256((asset / name).read_bytes()).hexdigest() == expected, name


def test_bundle_contains_relative_meshes_and_loads_after_relocation(bundle, tmp_path):
    assert (bundle / "profile.json").is_file()
    assert (bundle / "source.urdf").is_file()
    assert len(list((bundle / "meshes").glob("*.STL"))) == 44
    xml = ET.parse(bundle / "robot.xml").getroot()
    compiler = xml.find("compiler")
    mesh_dir = Path(compiler.attrib.get("meshdir", "")) if compiler is not None else Path()
    assert not mesh_dir.is_absolute()
    for mesh in xml.findall("./asset/mesh"):
        if "vertex" in mesh.attrib:
            assert mesh.attrib["name"].endswith(("_rear", "_forefoot"))
            continue
        filename = Path(mesh.attrib["file"])
        assert not filename.is_absolute()
        assert ".." not in filename.parts and ".." not in mesh_dir.parts
        assert (bundle / mesh_dir / filename).is_file()

    relocated = tmp_path / "unrelated_location" / "khan_mini"
    shutil.copytree(bundle, relocated)
    model = mujoco.MjModel.from_xml_path(str(relocated / "robot.xml"))
    configured = get_khan_mini_cfg(relocated).build().spec.compile()
    assert (model.nq, model.nmesh, configured.nu) == (34, 48, 27)
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


def test_training_metadata_separates_base_anchor_waist_com_and_tracked_head(bundle):
    profile = load_profile(bundle)
    assert profile["root_body"] == profile["anchor_body"] == "base_link"
    assert profile["com_body"] == "waist_yaw_Link"
    assert profile["neck_joint_names"] == ["Neck_yaw_joint", "Neck_pitch_joint"]
    assert len(profile["tracked_bodies"]) == 15
    assert profile["tracked_bodies"][0] == "base_link"
    assert profile["tracked_bodies"][7] == "waist_yaw_Link"
    assert profile["tracked_bodies"][-1] == "Neck_pitch_Link"
    assert len(profile["body_names"]) == 45 and "Neck_pitch_Link" in profile["body_names"]
    assert profile["dimensions"]["action"] == 27
    assert profile["dimensions"]["state"] == 67
    assert profile["dimensions"]["proprio_10_frames"] == 870
    assert profile["dimensions"]["goal_10_frames"] == 640


def test_waist_collision_width_override_keeps_pose_and_source_geometry(bundle):
    # Source/URDF provenance remains untouched; the MJCF uses a narrower Y box.
    source = ET.parse(bundle / "source.urdf").getroot()
    box = source.find("./link[@name='waist_yaw_Link']/collision/geometry/box")
    np.testing.assert_allclose(np.fromstring(box.attrib["size"], sep=" "), [.15, .20, .30])
    model = mujoco.MjModel.from_xml_path(str(bundle / "robot.xml"))
    body = model.body("waist_yaw_Link").id
    collisions = np.flatnonzero((model.geom_bodyid == body) & (model.geom_group == 3))
    assert len(collisions) == 1
    gid = collisions[0]
    assert model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX
    np.testing.assert_allclose(model.geom_size[gid], [.075, .07, .15], atol=1e-12)
    np.testing.assert_allclose(model.geom_pos[gid], [0, 0, .03], atol=1e-12)
    np.testing.assert_allclose(model.geom_quat[gid], [1, 0, 0, 0], atol=1e-12)
    override = load_profile(bundle)["parameter_overrides"]["waist_collision_box"]
    np.testing.assert_allclose(override["source_full_size_m"], [.15, .20, .30])
    np.testing.assert_allclose(override["effective_full_size_m"], [.15, .14, .30])


def test_visuals_collisions_capsules_and_feet_remain_distinct(bundle):
    for model in (
        mujoco.MjModel.from_xml_path(str(bundle / "robot.xml")),
        get_khan_mini_cfg(bundle).build().spec.compile(),
    ):
        visual = model.geom_group == 2
        collision = model.geom_group == 3
        assert visual.sum() == 45 and collision.sum() == 16
        assert (visual | collision).all()
        assert (model.geom_contype[visual] == 0).all()
        assert (model.geom_conaffinity[visual] == 0).all()
        assert (model.geom_contype[collision] == 1).all()
        assert (model.geom_conaffinity[collision] == 1).all()
        assert (model.geom_type[collision] == mujoco.mjtGeom.mjGEOM_CAPSULE).sum() == 4
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
        assert len(feet) == 4
        assert (model.geom_type[feet] == mujoco.mjtGeom.mjGEOM_MESH).all()
        np.testing.assert_allclose(
            model.geom_friction[feet], np.tile([0.6, 0.005, 0.0001], (4, 1))
        )
        nonfeet = np.flatnonzero(collision & ~np.isin(np.arange(model.ngeom), feet))
        np.testing.assert_allclose(
            model.geom_friction[nonfeet], np.tile([1.0, 0.005, 0.0001], (12, 1))
        )


def test_compiled_foot_hulls_follow_measured_sections_and_preserve_inertials(bundle):
    model = mujoco.MjModel.from_xml_path(str(bundle / "robot.xml"))
    reference = mujoco.MjModel.from_xml_path(str(bundle / "source.urdf"))
    profile = load_profile(bundle)
    for name in ("Left_ankle_roll_Link", "Right_ankle_roll_Link"):
        body = model.body(name).id
        collisions = np.flatnonzero((model.geom_bodyid == body) & (model.geom_group == 3))
        visuals = np.flatnonzero((model.geom_bodyid == body) & (model.geom_group == 2))
        assert len(collisions) == 2 and len(visuals) == 1
        assert model.mesh(model.geom_dataid[visuals[0]]).name == name
        override = profile["parameter_overrides"]["foot_collision"]["feet"][name]
        assert override["source_collision_count"] == 7
        assert override["effective_collision_count"] == 2
        assert set(override["geoms"]) == {model.geom(g).name for g in collisions}
        assert (model.geom_condim[collisions] == 3).all()
        assert (model.geom_priority[collisions] == 1).all()
        for part, xmin, xmax in (("rear", -75, 25), ("forefoot", 25, 127)):
            gid = model.geom(f"{name}_{part}_collision").id
            mesh = model.geom_dataid[gid]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, model.geom_quat[gid])
            points = 1000 * (model.mesh_vert[start:start + count] @ rotation.reshape(3, 3).T
                             + model.geom_pos[gid])
            np.testing.assert_allclose([points[:, 0].min(), points[:, 0].max(),
                                        points[:, 2].min()], [xmin, xmax, -48], atol=.01)
            # Intersect the compiled outer hull and its flat sole with vertical
            # measurement planes. A single convex hull would fill in the waist.
            for height, inset in ((-46, 0), (-48, 2)):
                ring = points[points[:, 2] <= height + .001, :2]
                poly = ring[ConvexHull(ring).vertices]
                stations = [(-35, 87), (0, 87), (24.9, 87)] if part == "rear" else [
                    (25.1, 87), (45, 98.3), (71, 105)]
                for x, expected in stations:
                    edges = np.roll(poly, -1, axis=0)
                    hit = (poly[:, 0] < x) != (edges[:, 0] < x)
                    a, b = poly[hit], edges[hit]
                    y = a[:, 1] + (x - a[:, 0]) * (b[:, 1] - a[:, 1]) / (b[:, 0] - a[:, 0])
                    assert np.ptp(y) == pytest.approx(expected - 2 * inset, abs=.08)
            if part == "rear":
                assert points[:, 2].max() == pytest.approx(14, abs=.01)
    # Contact edits change no inertials; only the explicit uniform foot mass
    # correction changes these two bodies' mass and inertia, not COM or axes.
    for body_id in range(1, model.nbody):
        name = model.body(body_id).name
        old_id = reference.body(name).id
        for field in ("body_mass", "body_ipos", "body_iquat", "body_inertia"):
            expected = getattr(reference, field)[old_id]
            if name in ("Left_ankle_roll_Link", "Right_ankle_roll_Link") and field in ("body_mass", "body_inertia"):
                expected = expected * ((reference.body_mass[old_id] + .1) / reference.body_mass[old_id])
            np.testing.assert_allclose(
                getattr(model, field)[body_id], expected,
                rtol=1e-10, atol=1e-12,
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

"""CPU checks of sole geometry height, independent of any motion integrator."""

import math

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from supermimic.data.foot_clearance import foot_geom_groups, geom_bottom_z
from supermimic.robots.khan_mini import DEFAULT_ASSET_DIR, get_khan_mini_cfg, load_profile


def primitive_model(kind, size, angle=0.0):
    model = mujoco.MjModel.from_xml_string(
        f"""<mujoco><compiler angle="radian"/>
        <worldbody><body pos="0 0 1.2"><freejoint/>
          <geom name="shape" type="{kind}" size="{size}" pos="0.1 -0.2 -0.25"
                axisangle="0 1 0 {angle}"/>
        </body></worldbody></mujoco>"""
    )
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    return model, data


@pytest.mark.parametrize("angle", [0.0, math.pi / 2, math.pi / 4, -0.61])
def test_capsule_bottom_uses_rotated_axis_end_and_radius(angle):
    model, data = primitive_model("capsule", "0.07 0.2", angle)
    expected = 1.2 - 0.25 - abs(math.cos(angle)) * 0.2 - 0.07
    assert geom_bottom_z(model, data, 0) == pytest.approx(expected, abs=1e-12)
    # The body/link origin is 1.2 m; it is not a valid proxy for the sole.
    assert abs(geom_bottom_z(model, data, 0) - data.xpos[1, 2]) > 0.3


def test_sphere_bottom_is_invariant_to_rotation():
    model, data = primitive_model("sphere", "0.07", 0.91)
    assert geom_bottom_z(model, data, 0) == pytest.approx(1.2 - 0.25 - 0.07, abs=1e-12)


@pytest.mark.parametrize("angle", [0.0, math.pi / 2, 0.43])
def test_cylinder_bottom_differs_from_capsule_hemisphere_extent(angle):
    radius, half_length = 0.07, 0.2
    model, data = primitive_model("cylinder", f"{radius} {half_length}", angle)
    expected = 0.95 - abs(math.cos(angle)) * half_length - abs(math.sin(angle)) * radius
    assert geom_bottom_z(model, data, 0) == pytest.approx(expected, abs=1e-12)


def test_rotated_box_and_ellipsoid_support_heights():
    angle = 0.37
    model, data = primitive_model("box", "0.1 0.2 0.3", angle)
    expected_box = 0.95 - abs(math.sin(angle)) * 0.1 - abs(math.cos(angle)) * 0.3
    assert geom_bottom_z(model, data, 0) == pytest.approx(expected_box, abs=1e-12)
    model, data = primitive_model("ellipsoid", "0.1 0.2 0.3", angle)
    expected_ellipsoid = 0.95 - math.sqrt((0.1 * math.sin(angle)) ** 2
                                         + (0.3 * math.cos(angle)) ** 2)
    assert geom_bottom_z(model, data, 0) == pytest.approx(expected_ellipsoid, abs=1e-12)


def test_mesh_uses_compiled_vertices_without_double_applying_scale_or_center():
    angle = 0.37
    vertices = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, 3]], dtype=float)
    scale = np.array([0.7, 1.2, 0.4])
    geom_pos = np.array([-0.1, 0.2, -0.3])
    body_pos = np.array([0, 0, 1.0])
    serialized = " ".join(str(value) for value in vertices.ravel())
    model = mujoco.MjModel.from_xml_string(
        f"""<mujoco><compiler angle="radian"/>
        <asset><mesh name="tetra" vertex="{serialized}" scale="0.7 1.2 0.4"/></asset>
        <worldbody><body pos="0 0 1"><freejoint/>
          <geom type="mesh" mesh="tetra" pos="-0.1 0.2 -0.3"
                axisangle="0 1 0 {angle}"/>
        </body></worldbody></mujoco>"""
    )
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    rotation = Rotation.from_rotvec([0, angle, 0]).as_matrix()
    expected = ((vertices * scale) @ rotation.T + geom_pos + body_pos)[:, 2].min()
    assert geom_bottom_z(model, data, 0) == pytest.approx(expected, abs=2e-7)


def test_root_translation_moves_actual_penetration_by_the_same_amount():
    model, data = primitive_model("capsule", "0.07 0.2", math.pi / 4)
    initial = geom_bottom_z(model, data, 0)
    data.qpos[2] -= initial + 0.005
    mujoco.mj_kinematics(model, data)
    assert geom_bottom_z(model, data, 0) == pytest.approx(-0.005, abs=1e-12)


@pytest.mark.skipif(not (DEFAULT_ASSET_DIR / "robot.xml").is_file(), reason="Khan assets missing")
def test_real_khan_home_uses_both_whole_foot_convex_colliders_and_has_ground_clearance():
    profile = load_profile()
    model = get_khan_mini_cfg().build().spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    mujoco.mj_kinematics(model, data)
    groups = foot_geom_groups(model, profile)
    assert set(groups) == set(profile["foot_bodies"])
    collision_ids = [geom for group in groups.values() for geom in group["collision"]]
    assert len(collision_ids) == 2 and len(set(collision_ids)) == 2
    assert all(len(group["collision"]) == 1 for group in groups.values())
    assert (model.geom_type[collision_ids] == mujoco.mjtGeom.mjGEOM_MESH).all()
    assert all(group["visual"] for group in groups.values())
    bottoms = [geom_bottom_z(model, data, geom) for geom in collision_ids]
    assert min(bottoms) == pytest.approx(0.00195, abs=1e-5)
    assert min(bottoms) > 0.001  # A 1 mm penetration tolerance must not flag home.
    data.qpos[2] -= 0.01
    mujoco.mj_kinematics(model, data)
    shifted = min(geom_bottom_z(model, data, geom) for geom in collision_ids)
    assert shifted == pytest.approx(min(bottoms) - 0.01, abs=1e-12)
    assert shifted < -0.001


def test_plane_is_not_treated_as_a_finite_foot_geometry():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="plane" size="1 1 .1"/></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    with pytest.raises(ValueError):
        geom_bottom_z(model, data, 0)

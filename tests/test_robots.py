import pytest

from jrl2.robots import ALL_ROBOTS


@pytest.fixture
def robots_classes():
    return ALL_ROBOTS


def test_create_robot(robots_classes: list):
    for robot_class in robots_classes:
        robot = robot_class()
        assert robot is not None


def test_xarm7_end_effector_variants_load_expected_links():
    from jrl2.robots import NAME_TO_ROBOT, XArm7, XArm7BioGripper, XArm7Gripper, XArm7VacuumGripper, get_robot_by_name

    assert "xarm_gripper_base_link" not in XArm7().link_names
    assert "xarm_gripper_base_link" in XArm7Gripper().link_names
    assert "bio_gripper_base_link" in XArm7BioGripper().link_names
    assert "xarm_vacuum_gripper_link" in XArm7VacuumGripper().link_names
    assert NAME_TO_ROBOT["xarm7__gripper"] is XArm7Gripper
    assert NAME_TO_ROBOT["xarm7__bio_gripper"] is XArm7BioGripper
    assert NAME_TO_ROBOT["xarm7__vacuum_gripper"] is XArm7VacuumGripper
    assert get_robot_by_name("xarm7__gripper").name == "xarm7__gripper"


def test_panda_zed_wrist_cam_attached_and_scaled():
    from jrl2.robots import Panda, PandaZedWristCam_PandaGripper, get_robot_by_name

    robot = get_robot_by_name("panda_zed_wrist_cam__panda_gripper")
    assert isinstance(robot, PandaZedWristCam_PandaGripper)
    assert "zed_wrist_cam_mount" not in Panda().link_names
    assert "zed_wrist_cam_mount" in robot.link_names
    assert "zed_mini" in robot.link_names
    assert "zed_mini::zed_mini.stl" in robot.visual_geometry_names
    assert robot.num_actuators == Panda().num_actuators
    for name in ["hand_camera_part_1.stl", "hand_camera_part_2.stl"]:
        assert f"zed_wrist_cam_mount::{name}" in robot.visual_geometry_names
        assert f"zed_wrist_cam_mount::{name}" in robot.collision_geometry_names

    # The STLs are in millimeters, so the mesh scale must be applied.
    for _, geometry, _ in (
        robot._geometry_cache["collision"]["zed_wrist_cam_mount"] + robot._geometry_cache["collision"]["zed_mini"]
    ):
        assert geometry.extents.max() < 0.2, f"Hand camera mesh is not scaled to meters: {geometry.extents}"


def test_panda_zed_wrist_cam_uses_own_collision_filtering_data():
    from jrl2.collision_detection_single_scene import SingleSceneCollisionChecker
    from jrl2.robots import get_robot_by_name

    robot = get_robot_by_name("panda_zed_wrist_cam__panda_gripper")
    # The ZED Mini sits in the mount, so they are always in contact and must be filtered
    assert robot.geometries_cant_collide("zed_mini::zed_mini.stl", "zed_wrist_cam_mount::hand_camera_part_1.stl", True)
    for use_visual in [True, False]:
        is_collision, col_pairs, _ = SingleSceneCollisionChecker(robot, use_visual=use_visual).check_collisions(
            robot.nominal_q
        )
        assert not is_collision, f"Nominal configuration is in collision (use_visual={use_visual}): {col_pairs}"

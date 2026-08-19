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

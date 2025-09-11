import pytest
import numpy as np

from jrl2.robots import get_robot_by_name
from jrl2.collision_detection_single_scene import SingleSceneCollisionChecker
from jrl2.robot import Robot

np.set_printoptions(precision=4, suppress=True)

PANDA = get_robot_by_name("panda")

"""
Run the test with:
uv run pytest  -W "ignore::DeprecationWarning" -W "ignore::UserWarning" --capture=no tests/test_collision_detection.py
"""


@pytest.mark.parametrize(
    "sphere_centers, sphere_radii, is_collision_gt",
    [
        (
            {"sphere_centers": [np.array([10, 0, 0]), np.array([11, 0, 0])]},
            {"sphere_radii": [0.25, 0.25]},
            False,  # No collision expected
        ),
        (
            {"sphere_centers": [np.array([5, 0, 0]), np.array([6, 0, 0])]},
            {"sphere_radii": [0.55, 0.55]},
            True,  # Collision expected
        ),
    ],
)
def test_check_sphere_collisions(sphere_centers, sphere_radii, is_collision_gt: bool):
    robot = PANDA
    for use_visual in [True, False]:
        collision_checker = SingleSceneCollisionChecker(robot, use_visual=use_visual)
        for sphere_center, sphere_radius in zip(sphere_centers["sphere_centers"], sphere_radii["sphere_radii"]):
            collision_checker.add_sphere(sphere_center, sphere_radius)
        is_collision, colliding_links = collision_checker.check_collisions(robot.midpoint_configuration)
        assert is_collision == is_collision_gt, f"Expected collision: {is_collision_gt}, got: {is_collision}"
        if is_collision:
            assert colliding_links == {("sphere_0", "sphere_1")}


@pytest.fixture
def robot() -> Robot:
    return PANDA


def test_get_all_link_geometry_poses_non_batched(robot: Robot):
    """There should be no collision between the robot and itself at the midpoint configuration.
    For both visual and collision geometries.
    """
    for use_visual in [True, False]:
        collision_checker = SingleSceneCollisionChecker(robot, use_visual=use_visual)
        is_collision, names = collision_checker.check_collisions(robot.midpoint_configuration)
        assert not is_collision, f"Expected no collision, got {is_collision}"
        assert len(names) == 0, f"Expected no collision, got {names}"
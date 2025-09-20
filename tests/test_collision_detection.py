from time import time

import pytest
import numpy as np
from trimesh.primitives import Box, Sphere
from jrl2.robots import get_robot_by_name
from jrl2.collision_detection_single_scene import SingleSceneCollisionChecker
from jrl2.robot import Robot
from jrl2.math_utils import get_translated_pose

np.set_printoptions(precision=4, suppress=True)

PANDA = get_robot_by_name("panda")

"""
Run the test with:
uv run pytest  -W "ignore::DeprecationWarning" -W "ignore::UserWarning" --capture=no tests/test_collision_detection.py
"""


@pytest.mark.parametrize(
    "spheres, is_collision_gt",
    [
        (
            [
                Sphere(radius=0.25, center=np.array([10, 0, 0])),
                Sphere(radius=0.25, center=np.array([11, 0, 0]))
            ],
            False,  # No collision expected
        ),
        (
            [
                Sphere(radius=0.55, center=np.array([5, 0, 0])),
                Sphere(radius=0.55, center=np.array([6, 0, 0]))
            ],
            True,  # Collision expected
        ),
    ],
)
def test_check_sphere_collisions(spheres, is_collision_gt: bool):
    robot = PANDA
    for use_visual in [True, False]:
        collision_checker = SingleSceneCollisionChecker(robot, use_visual=use_visual)
        for sphere in spheres:
            collision_checker.add_sphere(sphere)
        is_collision, colliding_links = collision_checker.check_collisions(robot.midpoint_configuration)
        assert is_collision == is_collision_gt, f"Expected collision: {is_collision_gt}, got: {is_collision}"
        if is_collision:
            assert colliding_links == {("sphere_0", "sphere_1")}


def test_get_contacts_runtime_increase():
    robot = PANDA
    collision_checker = SingleSceneCollisionChecker(robot, use_visual=False)
    box = Box(extents=(0.25, 0.25, 0.05), transform=get_translated_pose(np.array([0.25, -0.4, 0.5])))
    collision_checker.add_box(box)
    print()
    N = 100
    q_rands = [robot.sample_random_q_non_batched() for _ in range(N)]
    for return_contacts in [True, False]:
        print("~~~~~~~~~~")
        print(f"{return_contacts=}")
        t0 = time()
        for i in range(N):
            collision_checker.check_collisions(q_rands[i], return_contacts=return_contacts)
        t1 = time()
        print(f"Time taken: {t1 - t0}")


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

import pytest
import numpy as np

from jrl2.robots import get_robot_by_name
from jrl2.collision_detection import NonBatchedCollisionChecker

np.set_printoptions(precision=4, suppress=True)

PANDA = get_robot_by_name("panda")


@pytest.fixture(
    params=[
        (
            {"sphere_centers": [np.array([1, 1, 1]), np.array([2, 2, 2])]},  # configuration
            {
                "sphere_radii": [1.0, 2.0],
            },  # origin
        ),
    ]
)
def non_batched_collision_checker(request) -> NonBatchedCollisionChecker:
    sphere_centers, sphere_radii = request.param

    robot = PANDA
    collision_checker = NonBatchedCollisionChecker(robot)
    for sphere_center, sphere_radius in zip(sphere_centers["sphere_centers"], sphere_radii["sphere_radii"]):
        print(f"Adding sphere with center {sphere_center} and radius {sphere_radius}")
        collision_checker.add_sphere(sphere_center, sphere_radius)
    return collision_checker


def test_check_collisions(non_batched_collision_checker: NonBatchedCollisionChecker):
    q_dict = {joint.name: 0.0 for joint in PANDA.actuated_joints}
    non_batched_collision_checker.check_collisions(q_dict)

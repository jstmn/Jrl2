import argparse

import numpy as np
from trimesh.primitives import Sphere, Box

from jrl2.visualization import visualize_scene
from jrl2.robots import get_robot_by_name
from jrl2.collision_detection_single_scene import SingleSceneCollisionChecker

"""
uv run scripts/visualize_robot.py --robot panda --use_visual
uv run scripts/visualize_robot.py --robot panda --use_collision
uv run scripts/visualize_robot.py --robot panda --use_collision --h5_filepath '~/Projects/mpcm2/data/08-18_20:02:30__push-T, 2 (Aug 18)/data.h5'
"""


def _get_translated_pose(offset: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = offset
    return pose


OBSTACLES = {
    "spheres": [
        Sphere(center=np.array([0.5, 0, 0.25]), radius=0.125),
        Sphere(center=np.array([0.25, 0.5, 0.25]), radius=0.1),
    ],
    "boxes": [
        Box(extents=(0.25, 0.25, 0.05), transform=_get_translated_pose(np.array([0.25, -0.4, 0.5]))),
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, required=True, help="The name of the robot to visualize")
    parser.add_argument("--use_visual", action="store_true", help="Whether to use visual or collision geometries")
    parser.add_argument("--use_collision", action="store_true", help="Whether to use visual or collision geometries")
    parser.add_argument("--h5_filepath", type=str, required=False, help="The path to the h5 file to visualize")
    args = parser.parse_args()

    assert not (args.use_visual and args.use_collision), "Cannot use both visual and collision geometries"
    assert (
        args.use_visual or args.use_collision
    ), "Must use either visual or collision geometries: --use_collision or --use_visual"

    robot = get_robot_by_name(args.robot.lower())
    collision_checker = SingleSceneCollisionChecker(robot, use_visual=args.use_visual)

    # Add obstacles to the scene
    for obs in OBSTACLES["spheres"]:
        collision_checker.add_sphere(obs)
    for obs in OBSTACLES["boxes"]:
        collision_checker.add_box(obs)

    # Visualize the robot
    visualize_scene(robot, collision_checker, q_dict=robot.nominal_q, use_visual=args.use_visual)


if __name__ == "__main__":
    main()

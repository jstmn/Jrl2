import argparse

from jrl2.visualization import robot_scene
from jrl2.robots import get_robot_by_name
from jrl2.collision_detection_single_scene import SingleSceneCollisionChecker

"""
uv run scripts/visualize_robot.py --robot panda
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, required=True, help="The name of the robot to visualize")
    args = parser.parse_args()

    robot = get_robot_by_name(args.robot.lower())
    q_dict = robot.midpoint_configuration
    collision_checker = SingleSceneCollisionChecker(robot)
    robot_scene(robot, collision_checker, q_dict=q_dict, use_visual=True)


if __name__ == "__main__":
    main()

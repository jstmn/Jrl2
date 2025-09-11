import argparse

from jrl2.visualization import robot_scene
from jrl2.robots import get_robot_by_name
from jrl2.collision_detection_single_scene import SingleSceneCollisionChecker

"""
uv run scripts/visualize_robot.py --robot panda --use_visual
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, required=True, help="The name of the robot to visualize")
    parser.add_argument("--use_visual", action="store_true", help="Whether to use visual or collision geometries")
    parser.add_argument("--use_collision", action="store_true", help="Whether to use visual or collision geometries")
    args = parser.parse_args()

    assert not (args.use_visual and args.use_collision), "Cannot use both visual and collision geometries"
    assert args.use_visual or args.use_collision, "Must use either visual or collision geometries"

    robot = get_robot_by_name(args.robot.lower())
    q_dict = robot.midpoint_configuration
    collision_checker = SingleSceneCollisionChecker(robot, use_visual=args.use_visual)
    robot_scene(robot, collision_checker, q_dict=q_dict, use_visual=args.use_visual)


if __name__ == "__main__":
    main()

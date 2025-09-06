import argparse

from jrl2.robots import get_robot_by_name


"""
uv run scripts/visualize_robot.py --robot panda
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, required=True, help="The name of the robot to visualize")
    args = parser.parse_args()

    robot = get_robot_by_name(args.robot.lower())
    q_dict = robot.midpoint_configuration
    robot.visualize(q_dict=q_dict)


if __name__ == "__main__":
    main()

"""
This script empirically tests which link geometries are always colliding, sometimes colliding, and never colliding
according to trimesh (fcl really) for a given robot. For example, we know that link6 and link7 will for the Panda robot
because they intersect.

To collect the raw data, we will sample n random joint configurations. For each configuration, collision checking is
performed and the list of geometries in contact is saved.

After collecting the raw data, pairs of geometries in contact greater than (100-epsilon)% are considered always
colliding and pairs of geometries in contact less than epsilon% are considered never colliding. This data will then be
saved to src/jrl2/collision_filtering_data/robot_name.yaml

Note that we use the visual geometries for collision checking, because they provide a far more accurate model
of the actual robot.

# Example usage:
uv run scripts/generate_filtering_data.py --robot panda --n 10000
"""

import argparse
import os

from tqdm import tqdm
import yaml

from jrl2.robots import get_robot_by_name
from jrl2.collision_detection_single_scene import SingleSceneCollisionChecker
from jrl2.robot import Robot

assert os.path.exists("src/jrl2/robot.py"), "Current working directory must be the project root"

EPSILON_ALWAYS = 0.1  # percent
EPSILON_NEVER = 0.01  # percent


def print_data(data: dict, count_total: int):
    sorted_items = sorted(data.items(), key=lambda item: 100 * (item[1] / count_total), reverse=True)
    for pair, count in sorted_items:
        pct = 100 * (count / count_total)
        is_always = pct > (100 - EPSILON_ALWAYS)
        is_never = pct < EPSILON_NEVER
        is_sometimes = pct >= EPSILON_NEVER and pct <= (100 - EPSILON_ALWAYS)
        label = "always" if is_always else "sometimes" if is_sometimes else "never"
        if label == "never":
            assert is_never
        if label == "always":
            assert is_always
        if label == "sometimes":
            assert is_sometimes
        print(f"  {pair}:\t{count}\t{pct:.2f}% -> {label}")


def get_data(robot: Robot, n: int, collision_checker: SingleSceneCollisionChecker, geom_names: list[str]):
    data = {}
    for i in tqdm(range(n)):
        q = robot.sample_random_q_non_batched()
        _, names = collision_checker.check_collisions(q, dont_filter=True, print_timing=i % 1000 == 0)
        for name in names:
            assert len(name) == 2
            assert name[0] in geom_names
            assert name[1] in geom_names
            name_ordered = robot.return_ordered_geometry_name_pair(name[0], name[1])
            if name_ordered not in data:
                data[name_ordered] = 0
            data[name_ordered] += 1

        if i % 1000 == 0 and i > 0:
            print("--------------------------------")
            print(f"Progress: {i}/{n}")
            print_data(data, i + 1)
            print("--------------------------------")
    return data


def main(robot: Robot, n: int):
    """
    Generate collision filtering data for a given robot. If EPSILON_ALWAYS is 0.1, then pairs of geometries in contact
    greater than 99.9% of the time are considered always colliding. Similarly, if EPSILON_NEVER is 0.01, then pairs of
    geometries in contact less than 0.01% of the time are considered never colliding.

    Args:
        robot (Robot): The robot to generate filtering data for.
        n (int): The number of samples to generate.
    """
    collision_checker_visual = SingleSceneCollisionChecker(robot, use_visual=True)
    collision_checker_collision = SingleSceneCollisionChecker(robot, use_visual=False)

    robot_visual_names = robot.visual_geometry_names
    robot_collision_names = robot.collision_geometry_names
    data_visual = get_data(robot, n, collision_checker_visual, robot_visual_names)
    data_collision = get_data(robot, n, collision_checker_collision, robot_collision_names)

    print("--------------------------------")
    print("Data collection complete")
    print("--------------------------------")

    yaml_data = {
        "visual": {"always": [], "sometimes": [], "never": []},
        "collision": {"always": [], "sometimes": [], "never": []},
    }
    print_data(data_visual, n)
    print_data(data_collision, n)

    print("--------------------------------")
    print("Data processing complete")
    print("--------------------------------")

    for data, key in zip([data_visual, data_collision], ["visual", "collision"]):
        for pair, count in data.items():
            pct = 100 * (count / n)
            is_always = pct > (100 - EPSILON_ALWAYS)
            is_never = pct < EPSILON_NEVER
            is_sometimes = pct >= EPSILON_NEVER and pct <= (100 - EPSILON_ALWAYS)
            if is_always:
                yaml_data[key]["always"].append(pair)
            elif is_never:
                yaml_data[key]["never"].append(pair)
            elif is_sometimes:
                yaml_data[key]["sometimes"].append(pair)
            else:
                raise ValueError(f"Pair {pair} has {pct:.2f}% of contact, which is not always, sometimes, or never")
            label = "always" if is_always else "sometimes" if is_sometimes else "never"
            if label == "never":
                assert is_never
            if label == "always":
                assert is_always
            if label == "sometimes":
                assert is_sometimes
            print(f"  {key} {pair}:\t{count}\t{pct:.2f}% -> {label}")

    with open(f"src/jrl2/collision_filtering_data/{robot.name}.yaml", "w") as f:
        yaml.dump(yaml_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, required=True, help="The name of the robot to generate filtering data for")
    parser.add_argument("--n", type=int, required=True, help="The number of samples to generate")
    args = parser.parse_args()

    robot = get_robot_by_name(args.robot.lower())
    main(robot, args.n)

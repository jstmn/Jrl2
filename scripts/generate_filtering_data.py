"""
This script empirically tests which link geometries are always colliding, sometimes colliding, and never colliding
according to trimesh (fcl really) for a given robot. For example, we know that link6 and link7 will for the Panda robot
because they intersect.

To collect the raw data, we will sample n random joint configurations. For each configuration, collision checking is
performed and the list of geometries in contact is saved.

After collecting the raw data, pairs of geometries in contact greater than (100-epsilon)% are considered always
colliding and pairs of geometries in contact less than epsilon% are considered never colliding. This data will then be
saved to src/jrl2/collision_filtering_data/robot_name.yaml

Example usage:
uv run scripts/generate_filtering_data.py --robot panda --n 100
"""

import argparse
from tqdm import tqdm
import yaml

from jrl2.robots import get_robot_by_name
from jrl2.collision_detection_single_scene import SingleSceneCollisionChecker
from jrl2.robot import Robot


EPSILON = 0.5  # percent


def main(robot: Robot, n: int):
    """
    Generate collision filtering data for a given robot. If EPSILON is 5, then pairs of geometries in contact greater
    than 95% of the time are considered always colliding, while pairs of geometries in contact less than 5% of the time
    are considered never colliding. Note that 5% is a very high number. In practice, we'll want more like 1 or 0.1.

    Args:
        robot (Robot): The robot to generate filtering data for.
        n (int): The number of samples to generate.
    """
    collision_checker = SingleSceneCollisionChecker(robot)

    robot_visual_names = robot.visual_geometry_names
    robot_collision_names = robot.collision_geometry_names

    data = {}
    for i in tqdm(range(n)):
        q = robot.sample_random_q_non_batched()
        _, names, _ = collision_checker.check_collisions(q, dont_filter=True, use_visual=False)
        for name in names:
            assert len(name) == 2
            assert name[0] in robot_visual_names or name[0] in robot_collision_names
            assert name[1] in robot_visual_names or name[1] in robot_collision_names
            name_ordered = robot.return_ordered_geometry_name_pair(name[0], name[1])
            if name_ordered not in data:
                data[name_ordered] = 0
            data[name_ordered] += 1

        if i % 500 == 0 and i > 0:
            print("--------------------------------")
            print(f"Progress: {i}/{n}")
            for pair, count in data.items():
                print(f"  {pair}:\t{count}\t{100*(count/n):.2f}%")
            print("--------------------------------")

    yaml_data = {
        "collision": {"always": [], "sometimes": [], "never": []},
        "visual": {},
    }

    for pair, count in data.items():
        pct = 100 * (count / n)
        is_always = pct > (100 - EPSILON)
        is_never = pct < EPSILON
        is_sometimes = pct >= EPSILON and pct <= (100 - EPSILON)
        if is_always:
            yaml_data["collision"]["always"].append(pair)
        elif is_never:
            yaml_data["collision"]["never"].append(pair)
        elif is_sometimes:
            yaml_data["collision"]["sometimes"].append(pair)
        else:
            raise ValueError(f"Pair {pair} has {pct:.2f}% of contact, which is not always, sometimes, or never")
        print(
            f"  {pair}:\t{count}\t{pct:.2f}% -> {'always' if is_always else 'sometimes' if is_sometimes else 'never'}"
        )

    with open(f"src/jrl2/collision_filtering_data/{robot.name}.yaml", "w") as f:
        yaml.dump(yaml_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, required=True, help="The name of the robot to generate filtering data for")
    parser.add_argument("--n", type=int, required=True, help="The number of samples to generate")
    args = parser.parse_args()

    robot = get_robot_by_name(args.robot.lower())
    main(robot, args.n)

import numpy as np

from jrl2.robot import Robot


class Panda(Robot):

    NOMINAL_Q = {
        "panda_joint1": 0,
        "panda_joint2": -np.pi / 4,
        "panda_joint3": 0,
        "panda_joint4": -3 * np.pi / 4,
        "panda_joint5": 0,
        "panda_joint6": np.pi / 2,
        "panda_joint7": np.pi / 4,
        "panda_finger_joint2": 0.0,
        "panda_finger_joint1": 0.0,
    }
    ADDITIONAL_IGNORED_GEOMS = {
        "collision": {
            "always": [
                ("panda_leftfinger::box_4", "panda_rightfinger::box_4"),
            ]
        },
        "visual": {
            "always": [
                ("panda_leftfinger::finger.dae", "panda_rightfinger::finger.dae"),
            ]
        },
    }

    def __init__(self):
        super().__init__(
            "panda_description", nominal_q=self.NOMINAL_Q, additional_ignored_geoms=self.ADDITIONAL_IGNORED_GEOMS
        )


# TODO: add Fr3 to the robot_descriptions.py library and then add this class back in
# class Fr3(Robot):
#     def __init__(self):
#         super().__init__("fr3_description")


class UR5(Robot):
    def __init__(self):
        super().__init__("ur5_description")


class UR10(Robot):
    def __init__(self):
        super().__init__("ur10_description")


ALL_ROBOTS = [Panda, UR5, UR10]
NAME_TO_ROBOT = {robot.__name__.lower(): robot for robot in ALL_ROBOTS}


def get_robot_by_name(name: str) -> Robot:
    try:
        return NAME_TO_ROBOT[name]()
    except KeyError:
        raise ValueError(f"Robot with name {name} not found. Available robots: {NAME_TO_ROBOT.keys()}")


if __name__ == "__main__":
    for robot in ALL_ROBOTS:
        print(robot())

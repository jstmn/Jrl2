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


class XArm7(Robot):
    ROBOT_ID = "xarm7"

    def __init__(self):
        super().__init__("xarm7_description", robot_id=self.ROBOT_ID)


class XArm7Gripper(Robot):
    """xArm7 with the UFACTORY parallel-jaw gripper (`add_gripper:=true`)."""

    ROBOT_ID = "xarm7__gripper"

    def __init__(self):
        super().__init__(
            "xarm7_description",
            xacro_args={"add_gripper": "true"},
            robot_id=self.ROBOT_ID,
            collision_filtering_stem="xarm7",
        )


class XArm7BioGripper(Robot):
    """xArm7 with the UFACTORY BIO gripper (`add_bio_gripper:=true`)."""

    ROBOT_ID = "xarm7__bio_gripper"

    def __init__(self):
        super().__init__(
            "xarm7_description",
            xacro_args={"add_bio_gripper": "true"},
            robot_id=self.ROBOT_ID,
            collision_filtering_stem="xarm7",
        )


class XArm7VacuumGripper(Robot):
    """xArm7 with the UFACTORY vacuum gripper (`add_vacuum_gripper:=true`)."""

    ROBOT_ID = "xarm7__vacuum_gripper"

    def __init__(self):
        super().__init__(
            "xarm7_description",
            xacro_args={"add_vacuum_gripper": "true"},
            robot_id=self.ROBOT_ID,
            collision_filtering_stem="xarm7",
        )


ALL_ROBOTS = [Panda, UR5, UR10, XArm7, XArm7Gripper, XArm7BioGripper, XArm7VacuumGripper]
NAME_TO_ROBOT = {getattr(robot, "ROBOT_ID", robot.__name__.lower()): robot for robot in ALL_ROBOTS}
assert len(NAME_TO_ROBOT) == len(ALL_ROBOTS), f"Duplicate robot ids: {list(NAME_TO_ROBOT)}"


def get_robot_by_name(name: str) -> Robot:
    try:
        return NAME_TO_ROBOT[name.lower()]()
    except KeyError:
        raise ValueError(f"Robot with name {name} not found. Available robots: {NAME_TO_ROBOT.keys()}")


if __name__ == "__main__":
    for robot in ALL_ROBOTS:
        print(robot())

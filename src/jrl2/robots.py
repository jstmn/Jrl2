from jrl2.robot import Robot


class Panda(Robot):
    def __init__(self):
        super().__init__("panda_description")


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

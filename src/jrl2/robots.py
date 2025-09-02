from jrl2.robot import Robot


class Panda(Robot):

    def __init__(self):
        super().__init__("panda_description")


class Fr3(Robot):
    def __init__(self):
        super().__init__("fr3_description")


class PR2(Robot):
    def __init__(self):
        super().__init__("pr2_description")


class UR5(Robot):
    def __init__(self):
        super().__init__("ur5_description")


class UR10(Robot):
    def __init__(self):
        super().__init__("ur10_description")


class Baxter(Robot):
    def __init__(self):
        super().__init__("baxter_description")


ALL_ROBOTS = [Panda, Fr3, PR2, UR5, UR10, Baxter]

if __name__ == "__main__":
    for robot in ALL_ROBOTS:
        print(robot())

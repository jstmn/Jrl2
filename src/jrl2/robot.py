from yourdfpy import Robot as YourDFRobot
from robot_descriptions.loaders.yourdfpy import load_robot_description


class Robot:
    def __init__(self, name: str):
        self._name = name.replace("_description", "")
        self._yourdfpy_model: YourDFRobot = load_robot_description(name)

    def __str__(self):
        return f"Robot('{self._name}')"

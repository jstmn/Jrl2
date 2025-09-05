import pytest

from jrl2.robots import ALL_ROBOTS


@pytest.fixture
def robots_classes():
    return ALL_ROBOTS


def test_create_robot(robots_classes: list):
    for robot_class in robots_classes:
        robot = robot_class()
        assert robot is not None

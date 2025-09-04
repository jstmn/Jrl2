import pytest

from jrl2.robots import Panda


@pytest.fixture
def panda_robot():
    return Panda()


def test_create_robot(panda_robot: Panda):
    assert panda_robot is not None

from jrl2.scene import SingleEnvironmentScene

from jrl2.robots import Panda

import pytest


@pytest.fixture
def single_environment_scene():
    return SingleEnvironmentScene(Panda())


def test_single_environment_scene(single_environment_scene):
    assert single_environment_scene is not None


# @pytest.fixture
# def batched_environment_scene():
#     return BatchedEnvironmentScene(Panda())

# def test_batched_environment_scene(batched_environment_scene):
#     assert batched_environment_scene is not None

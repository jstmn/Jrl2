import torch
import pytest

from jrl2.scene import SingleEnvironmentScene
from jrl2.robots import Panda


@pytest.fixture
def single_environment_scene():
    return SingleEnvironmentScene(Panda())


def test_single_environment_scene(single_environment_scene: SingleEnvironmentScene):
    assert single_environment_scene is not None

    q = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    single_environment_scene.add_sphere(torch.tensor([2.0, 0.0, 0.0]), 1.0)
    single_environment_scene.add_sphere(torch.tensor([2.2, 0.0, 0.0]), 0.1)
    collisions = single_environment_scene.find_robot_environment_collisions(q)
    assert len(collisions) == 0



# @pytest.fixture
# def batched_environment_scene():
#     return BatchedEnvironmentScene(Panda())

# def test_batched_environment_scene(batched_environment_scene):
#     assert batched_environment_scene is not None

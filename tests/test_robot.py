import pytest
from yourdfpy import Link, Joint
from yourdfpy import Robot as YourdfpyRobot
import numpy as np
from scipy.spatial.transform import Rotation

from jrl2.robot import Robot

np.set_printoptions(precision=4, suppress=True)


def _get_translated_pose(offset: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = offset
    return pose


def _get_rotated_pose(axis: np.ndarray, angle: float) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec(angle * axis).as_matrix()
    return pose


@pytest.fixture(
    params=[
        (
            {"left_joint": 0.0, "right_joint": 0.0},  # configuration
            {"left_joint": np.eye(4), "right_joint": np.eye(4)},  # origin
            {
                "base_link": np.eye(4),
                "left_link": np.eye(4),
                "right_link": np.eye(4),  # ground truth poses
            },
        ),
        (
            {"left_joint": 1.0, "right_joint": 1.0},
            {
                "left_joint": _get_translated_pose(np.array([1.0, 0.0, 0.0])),
                "right_joint": _get_translated_pose(np.array([1.0, 0.0, 0.0])),
            },  # origin
            {
                "base_link": np.eye(4),
                "left_link": _get_translated_pose(np.array([1.0, 0.0, 1.0])),
                "right_link": _get_translated_pose(np.array([1.0, 0.0, -1.0])),
            },
        ),
    ]
)
def mock_robot_prismatic(request) -> tuple[Robot, dict[str, float], dict[str, np.ndarray]]:
    q_dict, origin, gt_poses = request.param
    links = [
        Link(name="base_link"),
        Link(name="left_link"),
        Link(name="right_link"),
    ]
    joints = [
        Joint(
            name="left_joint",
            parent="base_link",
            child="left_link",
            axis=[0, 0, 1],
            origin=origin["left_joint"],
            type="prismatic",
        ),
        Joint(
            name="right_joint",
            parent="base_link",
            child="right_link",
            axis=[0, 0, -1],
            origin=origin["right_joint"],
            type="prismatic",
        ),
    ]
    yourdfpy_robot = YourdfpyRobot(name="mock_robot_prismatic", links=links, joints=joints)
    robot = Robot(name="mock_robot_prismatic", yourdfpy_robot=yourdfpy_robot)
    return (robot, q_dict, gt_poses)


def test_get_all_link_poses_non_batched_prismatic(mock_robot_prismatic: Robot):
    robot, q_dict, gt_poses = mock_robot_prismatic
    poses = robot.get_all_link_poses_non_batched(q_dict)
    for link_name in gt_poses.keys():
        assert link_name in poses
    for link_name, pose in poses.items():
        assert np.allclose(pose, gt_poses[link_name])


@pytest.fixture(
    params=[
        (
            {"left_joint": 0.0, "right_joint": 0.0},  # configuration
            {
                "left_joint": _get_translated_pose(np.array([1, 1, 1])),
                "right_joint": _get_translated_pose(np.array([1, 1, 1])),
            },  # origin
            {
                "base_link": np.eye(4),
                "left_link": _get_translated_pose(np.array([1, 1, 1])),
                "right_link": _get_translated_pose(np.array([1, 1, 1])),  # ground truth poses
            },
        ),
        (
            {"left_joint": np.pi / 2, "right_joint": np.pi / 2},
            {
                "left_joint": _get_translated_pose(np.array([1, 1, 1])),  # origin left
                "right_joint": _get_translated_pose(np.array([1, 1, 1])),
            },  # origin right
            {
                "base_link": np.eye(4),
                "left_link": _get_translated_pose(np.array([1.0, 1.0, 1.0]))
                @ _get_rotated_pose(axis=np.array([0, 0, 1]), angle=np.pi / 2),  # ground truth left
                "right_link": _get_translated_pose(np.array([1.0, 1.0, 1.0]))
                @ _get_rotated_pose(axis=np.array([0, 0, 1]), angle=-np.pi / 2),  # ground truth right
            },
        ),
    ]
)
def mock_robot_revolute(request) -> tuple[Robot, dict[str, float], dict[str, np.ndarray]]:
    q_dict, origin, gt_poses = request.param
    links = [
        Link(name="base_link"),
        Link(name="left_link"),
        Link(name="right_link"),
    ]
    joints = [
        Joint(
            name="left_joint",
            parent="base_link",
            child="left_link",
            axis=[0, 0, 1],
            origin=origin["left_joint"],
            type="revolute",
        ),
        Joint(
            name="right_joint",
            parent="base_link",
            child="right_link",
            axis=[0, 0, -1],
            origin=origin["right_joint"],
            type="revolute",
        ),
    ]
    yourdfpy_robot = YourdfpyRobot(name="mock_robot_prismatic", links=links, joints=joints)
    robot = Robot(name="mock_robot_prismatic", yourdfpy_robot=yourdfpy_robot)
    return (robot, q_dict, gt_poses)


def test_get_all_link_poses_non_batched_revolute(mock_robot_revolute: Robot):
    robot, q_dict, gt_poses = mock_robot_revolute
    poses = robot.get_all_link_poses_non_batched(q_dict)
    for link_name in gt_poses.keys():
        assert link_name in poses
    for link_name, pose in poses.items():
        assert isinstance(pose, np.ndarray)
        assert isinstance(gt_poses[link_name], np.ndarray)
        assert np.allclose(pose, gt_poses[link_name], rtol=0.0, atol=0.001)

import os
import tempfile
from pathlib import Path

import kinpy as kp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from yourdfpy import Joint, Limit, Link
from yourdfpy import Robot as YourdfpyRobot

from jrl2.robot import Robot
from jrl2.robots import ALL_ROBOTS

np.set_printoptions(precision=4, suppress=True)


""" Run the test with:
uv run pytest  -W "ignore::DeprecationWarning" -W "ignore::UserWarning" --capture=no tests/test_robot.py

"""


def _get_translated_pose(offset: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = offset
    return pose


def _get_rotated_pose(axis: np.ndarray, angle: float) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec(angle * axis).as_matrix()
    return pose


def _kinpy_chain_from_urdf(urdf_filepath: Path):
    with open(urdf_filepath) as f:
        return kp.build_chain_from_urdf(f.read().encode("utf-8"))


def forward_kinematics_kinpy(kinpy_fk_chain, q_dict: dict[str, float], link_name: str) -> np.ndarray:
    zero_transform = kp.transform.Transform()
    return kinpy_fk_chain.forward_kinematics(q_dict, world=zero_transform)[link_name].matrix()


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
            limit=Limit(lower=0.0, upper=1.0),
        ),
        Joint(
            name="right_joint",
            parent="base_link",
            child="right_link",
            axis=[0, 0, -1],
            origin=origin["right_joint"],
            type="prismatic",
            limit=Limit(lower=0.0, upper=1.0),
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
            limit=Limit(lower=-10.0, upper=10.0),
        ),
        Joint(
            name="right_joint",
            parent="base_link",
            child="right_link",
            axis=[0, 0, -1],
            origin=origin["right_joint"],
            type="revolute",
            limit=Limit(lower=-10.0, upper=10.0),
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


@pytest.mark.parametrize("robot_cls", ALL_ROBOTS, ids=lambda cls: cls.__name__)
def test_get_all_link_poses_non_batched_matches_kinpy(robot_cls):
    robot = robot_cls()
    assert robot is not None
    has_prismatic = any(joint.type == "prismatic" for joint in robot.actuated_joints)

    with tempfile.NamedTemporaryFile(suffix=".urdf", delete=False) as tmp_file:
        urdf_filepath = Path(tmp_file.name)
        robot._yourdfpy_model.write_xml_file(urdf_filepath)

    try:
        kinpy_fk_chain = None if has_prismatic else _kinpy_chain_from_urdf(urdf_filepath)
        for _ in range(10):
            q_dict = robot.sample_random_q_non_batched()
            jrl2_link_poses = robot.get_all_link_poses_non_batched(q_dict)

            for link_name, tf in jrl2_link_poses.items():
                assert tf is not None, f"JRL2 forward kinematics returned None for link {link_name}"
                assert isinstance(tf, np.ndarray), f"JRL2 forward kinematics returned {type(tf)} for link {link_name}"
                assert tf.shape == (4, 4), f"JRL2 forward kinematics returned {tf.shape} for link {link_name}"

            if has_prismatic:
                # Kinpy does not support prismatic joints, so only check pose formatting.
                continue

            for link_name in robot.link_names:
                assert link_name in jrl2_link_poses
                link_tf_kinpy = forward_kinematics_kinpy(kinpy_fk_chain, q_dict, link_name)
                link_tf_jrl2 = jrl2_link_poses[link_name]
                assert link_tf_kinpy is not None, f"Kinpy forward kinematics returned None for link {link_name}"
                assert link_tf_jrl2 is not None, f"JRL2 forward kinematics returned None for link {link_name}"
                assert np.allclose(link_tf_jrl2, link_tf_kinpy), f"FK mismatch for {robot.name} link {link_name}"
    finally:
        if urdf_filepath.exists():
            os.unlink(urdf_filepath)

import numpy as np

from jrl2.types import NP_SE3_TYPE


def get_translated_pose(offset: np.ndarray) -> NP_SE3_TYPE:
    pose = np.eye(4)
    pose[:3, 3] = offset
    return pose

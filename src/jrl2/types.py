import numpy as np

# 4x4 SE3 matrix
NP_SE3_TYPE = np.ndarray

# [ndof]
NP_Q_TYPE = np.ndarray

# {joint_name_1: joint_angle_1, joint_name_2: joint_angle_2, ...}
NP_Q_DICT_TYPE = dict[str, float]
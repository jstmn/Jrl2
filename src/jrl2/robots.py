import importlib.resources

import numpy as np
from yourdfpy import Collision, Geometry, Joint, Link, Mesh, Visual

from jrl2.robot import Robot


class Panda(Robot):

    NOMINAL_Q = {
        "panda_joint1": 0,
        "panda_joint2": -np.pi / 4,
        "panda_joint3": 0,
        "panda_joint4": -3 * np.pi / 4,
        "panda_joint5": 0,
        "panda_joint6": np.pi / 2,
        "panda_joint7": np.pi / 4,
        "panda_finger_joint2": 0.0,
        "panda_finger_joint1": 0.0,
    }
    ADDITIONAL_IGNORED_GEOMS = {
        "collision": {
            "always": [
                ("panda_leftfinger::box_4", "panda_rightfinger::box_4"),
            ]
        },
        "visual": {
            "always": [
                ("panda_leftfinger::finger.dae", "panda_rightfinger::finger.dae"),
            ]
        },
    }

    def __init__(self):
        super().__init__(
            "panda_description", nominal_q=self.NOMINAL_Q, additional_ignored_geoms=self.ADDITIONAL_IGNORED_GEOMS
        )


def _translation(xyz: tuple[float, float, float]) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = xyz
    return pose


def _mount_T_zed_mini() -> np.ndarray:
    """Pose of the ZED Mini body frame in the mount frame. The camera body sits in the sleeve's U channel cavity with
    its flat back on the channel floor, which runs from (x=69.71, z=-0.21mm) to (x=95.07, z=9.04mm). The lenses
    protrude out of the channel opening toward the gripper axis and down at the fingers. The ZED Mini mesh has its
    stereo baseline along x, its back at y=0, its body front at y=14.5mm with the lenses protruding to y=30.25mm (so it
    looks along +y), and is centered in z.
    """
    floor_start, floor_end = np.array([0.06971, -0.00021]), np.array([0.09507, 0.00904])
    floor_dir = (floor_end - floor_start) / np.linalg.norm(floor_end - floor_start)
    floor_mid = (floor_start + floor_end) / 2
    # Columns are the ZED Mini x, y, z axes expressed in the mount frame
    mount_R_zed = np.array(
        [
            [0.0, -floor_dir[1], floor_dir[0]],
            [1.0, 0.0, 0.0],
            [0.0, floor_dir[0], floor_dir[1]],
        ]
    )
    zed_back_center = np.array([-0.00225, 0.0, 0.0])
    # Centered along the assembled sleeve, which spans y=[-74, 58]mm in the mount frame
    mount_back_center = np.array([floor_mid[0], -0.008, floor_mid[1]])
    pose = np.eye(4)
    pose[:3, :3] = mount_R_zed
    pose[:3, 3] = mount_back_center - mount_R_zed @ zed_back_center
    return pose

def _mesh_geometries(
    name: str, mesh_filepath: str, scale: float, link_T_mesh: np.ndarray
) -> tuple[Visual, Collision]:
    geometry = Geometry(mesh=Mesh(filename=mesh_filepath, scale=np.array([scale, scale, scale])))
    return Visual(name=name, origin=link_T_mesh, geometry=geometry), Collision(
        name=name, origin=link_T_mesh, geometry=geometry
    )
    
class PandaZedWristCam_PandaGripper(Robot):
    """Panda with the DROID wrist camera: a ZED Mini held by a two piece printed sleeve attached to the hand.

    The sleeve STLs are in millimeters and laid out for printing. In the STL frame the flange axis is at (x=0,
    y=-280mm), the mounting tab is the z=[0, 3mm] plane and the sleeve extends along +x. Part 1 is shifted +39mm in y so
    that its prongs seat into part 2. The STL axes line up with the panda_hand axes: the tab sits in the flange plane
    with the sleeve off the +x side of the hand, its long axis (the stereo baseline) parallel to the finger motion, and
    its outer end tilted down toward the fingers. The ZED Mini is its own link so that its frame can be looked up.

    NOTE: the mounting transform is estimated, not taken from the DROID CAD assembly.
    """

    ROBOT_ID = "panda_zed_wrist_cam__panda_gripper"
    HAND_CAMERA_LINK_NAME = "zed_wrist_cam_mount"
    ZED_MINI_LINK_NAME = "zed_mini"
    PANDA_HAND_T_HAND_CAMERA = np.eye(4)
    HAND_CAMERA_T_ZED_MINI = _mount_T_zed_mini()
    # (filename, link_T_mesh)
    HAND_CAMERA_MESHES = [
        ("hand_camera/hand_camera_part_1.stl", _translation((0.0, 0.280 + 0.039, 0.0))),
        ("hand_camera/hand_camera_part_2.stl", _translation((0.0, 0.280, 0.0))),
    ]

    def __init__(self):
        asset_dir = importlib.resources.files("jrl2") / "assets" / "droid"

        mount_visuals, mount_collisions = [], []
        for filename, link_T_mesh in self.HAND_CAMERA_MESHES:
            visual, collision = _mesh_geometries(filename, str(asset_dir / filename), 0.001, link_T_mesh)
            mount_visuals.append(visual)
            mount_collisions.append(collision)
        mount_link = Link(name=self.HAND_CAMERA_LINK_NAME, visuals=mount_visuals, collisions=mount_collisions)
        mount_joint = Joint(
            name=f"{self.HAND_CAMERA_LINK_NAME}_joint",
            type="fixed",
            parent="panda_hand",
            child=self.HAND_CAMERA_LINK_NAME,
            origin=self.PANDA_HAND_T_HAND_CAMERA,
        )

        # zed_mini.stl was converted from the Stereolabs Zed_mini.step and is already in meters
        zed_visual, zed_collision = _mesh_geometries(
            "zed_mini.stl", str(asset_dir / "zed_mini.stl"), 1.0, np.eye(4)
        )
        zed_link = Link(name=self.ZED_MINI_LINK_NAME, visuals=[zed_visual], collisions=[zed_collision])
        zed_joint = Joint(
            name=f"{self.ZED_MINI_LINK_NAME}_joint",
            type="fixed",
            parent=self.HAND_CAMERA_LINK_NAME,
            child=self.ZED_MINI_LINK_NAME,
            origin=self.HAND_CAMERA_T_ZED_MINI,
        )
        super().__init__(
            "panda_description",
            nominal_q=Panda.NOMINAL_Q,
            additional_ignored_geoms=Panda.ADDITIONAL_IGNORED_GEOMS,
            robot_id=self.ROBOT_ID,
            collision_filtering_stem=self.ROBOT_ID,
            extra_links=[(mount_link, mount_joint), (zed_link, zed_joint)],
        )


# TODO: add Fr3 to the robot_descriptions.py library and then add this class back in
# class Fr3(Robot):
#     def __init__(self):
#         super().__init__("fr3_description")


class UR5(Robot):
    def __init__(self):
        super().__init__("ur5_description")


class UR10(Robot):
    def __init__(self):
        super().__init__("ur10_description")


class XArm7(Robot):
    ROBOT_ID = "xarm7"

    def __init__(self):
        super().__init__("xarm7_description", robot_id=self.ROBOT_ID)


class XArm7Gripper(Robot):
    """xArm7 with the UFACTORY parallel-jaw gripper (`add_gripper:=true`)."""

    ROBOT_ID = "xarm7__gripper"

    def __init__(self):
        super().__init__(
            "xarm7_description",
            xacro_args={"add_gripper": "true"},
            robot_id=self.ROBOT_ID,
            collision_filtering_stem="xarm7",
        )


class XArm7BioGripper(Robot):
    """xArm7 with the UFACTORY BIO gripper (`add_bio_gripper:=true`)."""

    ROBOT_ID = "xarm7__bio_gripper"

    def __init__(self):
        super().__init__(
            "xarm7_description",
            xacro_args={"add_bio_gripper": "true"},
            robot_id=self.ROBOT_ID,
            collision_filtering_stem="xarm7",
        )


class XArm7VacuumGripper(Robot):
    """xArm7 with the UFACTORY vacuum gripper (`add_vacuum_gripper:=true`)."""

    ROBOT_ID = "xarm7__vacuum_gripper"

    def __init__(self):
        super().__init__(
            "xarm7_description",
            xacro_args={"add_vacuum_gripper": "true"},
            robot_id=self.ROBOT_ID,
            collision_filtering_stem="xarm7",
        )


ALL_ROBOTS = [
    Panda,
    PandaZedWristCam_PandaGripper,
    UR5,
    UR10,
    XArm7,
    XArm7Gripper,
    XArm7BioGripper,
    XArm7VacuumGripper,
]
NAME_TO_ROBOT = {getattr(robot, "ROBOT_ID", robot.__name__.lower()): robot for robot in ALL_ROBOTS}
assert len(NAME_TO_ROBOT) == len(ALL_ROBOTS), f"Duplicate robot ids: {list(NAME_TO_ROBOT)}"


def get_robot_by_name(name: str) -> Robot:
    try:
        return NAME_TO_ROBOT[name.lower()]()
    except KeyError:
        raise ValueError(f"Robot with name {name} not found. Available robots: {NAME_TO_ROBOT.keys()}")


if __name__ == "__main__":
    for robot in ALL_ROBOTS:
        print(robot())

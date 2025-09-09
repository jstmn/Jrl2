from importlib import import_module  # type: ignore
from pathlib import Path
from time import sleep

from yourdfpy import Robot as YourdfpyRobot
from yourdfpy import URDF as YourdfpyURDF
from yourdfpy import Link, Joint
from robot_descriptions.loaders.yourdfpy import load_robot_description
import numpy as np
from scipy.spatial.transform import Rotation
import viser
import trimesh


ACTUATED_JOINT_TYPES = ["revolute", "prismatic", "continuous"]

NP_SE3_TYPE = np.ndarray
NP_Q_TYPE = np.ndarray  # [ndof array]
NP_Q_DICT_TYPE = dict[str, float]  # {joint_name: joint_angle}


# TODO: replace scipy rotations with numba jitted rotations
def _fk_step_non_batched(joint: Joint, joint_angle: float) -> NP_SE3_TYPE:
    assert joint.type not in ["floating", "planar"], f"Joint type '{joint.type}' is not implemented"
    assert joint.type in (
        "revolute",
        "continuous",
        "prismatic",
        "fixed",
    ), f"Unknown joint type '{joint.type}'"
    assert joint.origin is not None
    parent_T_child_fixed = np.eye(4)

    # joint.origin is a 4x4 transformation matrix
    # Extract translation from the last column
    parent_T_child_fixed[0:3, 3] = joint.origin[0:3, 3]

    # Extract rotation from the 3x3 submatrix
    parent_T_child_fixed[:3, :3] = joint.origin[:3, :3]

    if joint.type in ("revolute", "continuous"):
        assert joint.axis is not None
        joint_rotation = Rotation.from_rotvec(joint_angle * np.array(joint.axis)).as_matrix()
        joint_T = np.eye(4)
        joint_T[:3, :3] = joint_rotation
        return parent_T_child_fixed @ joint_T

    if joint.type == "prismatic":
        assert joint.axis is not None
        joint_T = np.eye(4)
        joint_T[:3, 3] = joint_angle * np.array(joint.axis)
        return parent_T_child_fixed @ joint_T

    if joint.type == "fixed":
        return parent_T_child_fixed

    raise RuntimeError(f"I shouldn't be here {joint.type}")


def _get_successor_links(urdfpy_robot: YourdfpyRobot) -> dict:
    """Maps a Link to a [(Link[parent], Joint, Link[child]), ...] tuple for each successor link. The parent link is the
    link that the joint is attached to, and the child link is the link that the joint is attached to.

    Args:
        urdfpy_robot (YourdfpyRobot): The URDF robot to get the successor links for.

    Returns:
        dict: A dictionary mapping a Link to a [(Link[parent], Joint, Link[child]), ...] tuple for each successor link.
    """
    result = {}
    links_by_name = {link.name: link for link in urdfpy_robot.links}

    for link in urdfpy_robot.links:
        result[link.name] = []

    for joint in urdfpy_robot.joints:
        result[joint.parent].append((links_by_name[joint.parent], joint, links_by_name[joint.child]))
        assert isinstance(
            result[joint.parent], list
        ), f"Expected list for {joint.parent}, got {type(result[joint.parent])}"
        assert isinstance(
            result[joint.parent][0], tuple
        ), f"Expected tuple for {joint.parent} first element, got {type(result[joint.parent][0])}"
        assert isinstance(
            result[joint.parent][0][0], Link
        ), f"Expected Link for {joint.parent} first element first element, got {type(result[joint.parent][0][0])}"
        assert isinstance(
            result[joint.parent][0][1], Joint
        ), f"Expected Joint for {joint.parent} first element second element, got {type(result[joint.parent][0][1])}"
        assert isinstance(
            result[joint.parent][0][2], Link
        ), f"Expected Link for {joint.parent} first element third element, got {type(result[joint.parent][0][2])}"
    return result


class Robot:
    def __init__(self, name: str, yourdfpy_robot: YourdfpyRobot | None = None):
        if yourdfpy_robot is None:
            assert "_description" in name, f"Name {name} should contain _description"
            self._name = name.replace("_description", "")
            self._yourdfpy_model: YourdfpyURDF = load_robot_description(name, build_collision_scene_graph=True)
            self._urdfpy_robot: YourdfpyRobot = self._yourdfpy_model.robot
            module = import_module(f"robot_descriptions.{name}")
            # Example:
            # self._urdf_filepath='~/.cache/robot_descriptions/example-robot-data/robots/panda_description/urdf/panda.urdf'
            # self._robot_description_dir='~/.cache/robot_descriptions/example-robot-data/robots/panda_description'
            # self._repository_path='~/.cache/robot_descriptions/example-robot-data'
            self._urdf_filepath = Path(module.URDF_PATH)
            self._robot_description_dir = Path(module.PACKAGE_PATH)
            self._repository_path = Path(module.REPOSITORY_PATH)
            assert self._urdf_filepath.exists(), f"URDF file {self._urdf_filepath} does not exist"
            assert (
                self._robot_description_dir.exists()
            ), f"Robot description directory {self._robot_description_dir} does not exist"
            assert self._repository_path.exists(), f"Repository path {self._repository_path} does not exist"
        else:
            self._name = name.replace("_description", "")
            self._urdfpy_robot = yourdfpy_robot
            self._yourdfpy_model = YourdfpyURDF(robot=yourdfpy_robot, build_collision_scene_graph=True)

        assert isinstance(
            self._yourdfpy_model, YourdfpyURDF
        ), f"Expected YourdfpyURDF, got {type(self._yourdfpy_model)}"
        assert isinstance(self._urdfpy_robot, YourdfpyRobot), f"Expected YourdfpyRobot, got {type(self._urdfpy_robot)}"

        # Store links and joints by name for easy access
        self._links_by_name: dict[str, Link] = {link.name: link for link in self._urdfpy_robot.links}
        self._joints_by_name: dict[str, Joint] = {joint.name: joint for joint in self._urdfpy_robot.joints}
        # self._successor_links maps a Link to a [(Link[parent], Joint, Link[child]), ...] tuple for every link.
        self._successor_links: dict[str, list[tuple[Link, Joint, Link]]] = _get_successor_links(self._urdfpy_robot)

    @property
    def actuated_joints(self) -> list[Joint]:
        return [joint for joint in self._urdfpy_robot.joints if joint.type in ACTUATED_JOINT_TYPES]

    @property
    def actuated_joint_names(self) -> list[str]:
        return [joint.name for joint in self.actuated_joints]

    @property
    def link_names(self) -> list[str]:
        return list(self._links_by_name.keys())

    @property
    def links(self) -> list[Link]:
        return list(self._links_by_name.values())

    @property
    def num_actuators(self) -> int:
        return len(self.actuated_joints)

    @property
    def midpoint_configuration(self) -> NP_Q_DICT_TYPE:
        return {joint.name: (joint.limit.lower + joint.limit.upper) / 2.0 for joint in self.actuated_joints}

    def get_all_link_poses_non_batched(
        self, q_dict: NP_Q_DICT_TYPE, root_link_pose: NP_SE3_TYPE = np.eye(4)
    ) -> NP_SE3_TYPE:
        """
        Get the poses of all links in the robot.
        """
        assert len(q_dict) == self.num_actuators
        assert set(q_dict.keys()) == set(self.actuated_joint_names)
        poses = {name: None for name in self._links_by_name.keys()}
        q_dict_full = q_dict.copy()
        q_dict_full.update(
            {joint.name: 0.0 for joint in self._urdfpy_robot.joints if joint.type not in ACTUATED_JOINT_TYPES}
        )

        #
        root_link = self._links_by_name[self._yourdfpy_model.base_link]
        poses[root_link.name] = root_link_pose
        search_queue = []  # (current_link, joint, child_link)
        for parent_link, joint, child_link in self._successor_links[root_link.name]:
            assert parent_link == root_link, f"Parent link is not the root link: {parent_link} != {root_link}"
            search_queue.append((root_link, joint, child_link))

        while len(search_queue) > 0:
            current_link, joint, child_link = search_queue.pop(0)

            # Compute the transformation for this joint
            link_T_successor = _fk_step_non_batched(joint, q_dict_full[joint.name])
            poses[child_link.name] = poses[current_link.name] @ link_T_successor
            #
            for child_link_repeated, next_joint, next_child_link in self._successor_links[child_link.name]:
                assert child_link_repeated == child_link, f"Child link repeated: {child_link_repeated} != {child_link}"
                search_queue.append((child_link_repeated, next_joint, next_child_link))

        # Verify the output
        for link_name, pose in poses.items():
            assert pose is not None, f"Pose is None for link {link_name}"
            assert isinstance(pose, np.ndarray), f"Pose is not a numpy array for link {link_name}"
            assert pose.shape == (4, 4), f"Pose has shape {pose.shape} for link {link_name}"

        return poses

    def get_all_link_mesh_poses_non_batched(
        self, q_dict: NP_Q_DICT_TYPE, use_visual: bool, only_poses: bool = False
    ) -> dict[str, list[tuple[str, trimesh.Trimesh, NP_SE3_TYPE]]]:
        """
        Get the collision or visual meshes of all links in the robot. Note that there can be several meshes for each
        link.

        If only_poses is True:
            1. only the poses of the meshes will be returned
            2. The return type will be dict[str, list[tuple[None, NP_SE3_TYPE]]]
        """
        link_poses = self.get_all_link_poses_non_batched(q_dict)
        link_meshes = {}
        for link_name, link_pose in link_poses.items():

            # Setup variables
            link_meshes[link_name] = []
            link = self._links_by_name[link_name]

            # Get the mesh
            visual_or_collision = link.visuals if use_visual else link.collisions
            use_indexing = len(visual_or_collision) > 1
            for i in range(len(visual_or_collision)):
                name = link_name
                if use_indexing:
                    name += f"__{i}"
                link_T_mesh = visual_or_collision[i].origin
                geom = visual_or_collision[i].geometry  # May be one of the following:
                if link_T_mesh is None:
                    link_T_mesh = np.eye(4)
                mesh_pose = link_pose @ link_T_mesh
                trimesh_obj = None

                if not only_poses:
                    if geom.mesh is not None:
                        new_filename = self._yourdfpy_model._filename_handler(fname=geom.mesh.filename)
                        assert Path(new_filename).exists(), f"File {new_filename} does not exist"
                        trimesh_obj = trimesh.load(
                            new_filename,
                            ignore_broken=True,
                            force="mesh",
                            skip_materials=True,
                        )
                    elif geom.box is not None:
                        trimesh_obj = trimesh.primitives.Box(geom.box.size)
                    elif geom.cylinder is not None:
                        trimesh_obj = trimesh.primitives.Cylinder(geom.cylinder.radius, geom.cylinder.length)
                    elif geom.sphere is not None:
                        trimesh_obj = trimesh.primitives.Sphere(geom.sphere.radius)
                    else:
                        raise ValueError(f"Link {link_name} has an unsupported geometry. Geometry: {geom}")
                link_meshes[link_name].append((name, trimesh_obj, mesh_pose))
        return link_meshes

    def visualize(self, q_dict: NP_Q_DICT_TYPE | None = None, show_frames: bool = False):
        assert show_frames is False, "There's a bug with the frames, they are not being updated correctly"
        server = viser.ViserServer()
        if q_dict is None:
            q_dict = self.midpoint_configuration

        import numpy as np

        np.set_printoptions(precision=4, suppress=True)

        # Set the camera view to a good position + angle
        def client_connect_callback(client: viser.ClientHandle):
            print(f"Client {client.client_id} connected")
            camera_handle = client.camera
            camera_handle.position = np.array([1.5, 1.5, 1.5])
            camera_handle.wxyz = np.array([-0.1006611, 0.17216605, 0.84593253, -0.49459513])
            camera_handle.fov = 1.3089969389957472
            camera_handle.look_at = np.array([0.0, 0.0, 0.5])
            camera_handle.up_direction = np.array([0.0, 0.0, 1.0])

        server.on_client_connect(client_connect_callback)

        # Add the grid and robot to the scene
        server.add_grid("/grid", width=5.0, height=5.0)
        meshes_added = False
        mesh_handles = {}

        def update_configuration(q_dict_in: NP_Q_DICT_TYPE):
            nonlocal meshes_added
            link_mesh_poses = self.get_all_link_mesh_poses_non_batched(
                q_dict_in, use_visual=True, only_poses=True if meshes_added else False
            )
            for _, link_trimesh_list in link_mesh_poses.items():
                for mesh_name, link_trimesh_object, link_trimesh_pose in link_trimesh_list:
                    print(type(link_trimesh_object))
                    wxyz = Rotation.from_matrix(link_trimesh_pose[:3, :3]).as_quat(scalar_first=True)
                    position = link_trimesh_pose[:3, 3]
                    name = f"/{mesh_name}"
                    frame_name = name + "/frame"
                    if not meshes_added:
                        # scalar-first order is (w, x, y, z)
                        mesh_handles[name] = server.add_mesh_trimesh(
                            name=name, mesh=link_trimesh_object, position=position, wxyz=wxyz
                        )
                        if show_frames:
                            mesh_handles[frame_name] = server.add_frame(
                                name=frame_name, position=position, wxyz=wxyz, axes_length=0.075, axes_radius=0.005
                            )
                        print(type(mesh_handles[name]))
                    else:
                        mesh_handles[name].position = position
                        mesh_handles[name].wxyz = wxyz
                        if show_frames:
                            mesh_handles[frame_name].position = position
                            mesh_handles[frame_name].wxyz = wxyz

            if not meshes_added:
                for mesh, mesh_handle in mesh_handles.items():

                    def make_click_handler(handle: viser.GlbHandle):
                        def _on_click(event: viser.GuiEvent):
                            print(f"Clicked on {handle.name}")
                            # toggle between red and default (white)
                            if handle.color is None or (handle.color == (1.0, 1.0, 1.0)):
                                handle.color = (1.0, 0.0, 0.0)  # red
                            else:
                                handle.color = (1.0, 1.0, 1.0)  # white

                        return _on_click

                    mesh_handle.on_click(make_click_handler(mesh_handle))

            meshes_added = True

        update_configuration(q_dict)

        # Add a slider for each joint to the gui. There may be a more elegent way to do this without having a dictionary
        # of callbacks, but ey - works well enough.
        def on_slider_update(event: viser.GuiEvent):
            value = callbacks[event.target.label].value
            q_dict[event.target.label] = value
            update_configuration(q_dict)

        callbacks = {}
        for joint in self.actuated_joints:
            range_ = joint.limit.upper - joint.limit.lower
            callbacks[joint.name] = server.gui.add_slider(
                label=joint.name,
                min=joint.limit.lower,
                max=joint.limit.upper,
                step=range_ / 100.0,
                initial_value=q_dict[joint.name],
            )
            callbacks[joint.name].on_update(on_slider_update)

        print("Open your browser to http://localhost:8080")
        print("Press Ctrl+C to exit")

        while True:
            sleep(2.0)

    def __str__(self):
        return f"Robot('{self._name}')"

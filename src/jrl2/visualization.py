from jrl2.robot import Robot, NP_Q_DICT_TYPE
import viser
from scipy.spatial.transform import Rotation
from time import sleep
from jrl2.robots import Panda


def visualize(robot: Robot, q_dict: NP_Q_DICT_TYPE | None = None, show_frames: bool = False):
    assert show_frames is False, "There's a bug with the frames, they are not being updated correctly"
    server = viser.ViserServer()
    if q_dict is None:
        q_dict = robot.midpoint_configuration

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
        link_mesh_poses = robot.get_all_link_mesh_poses_non_batched(
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
    for joint in robot.actuated_joints:
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

""""
uv run src/jrl2/visualization.py
"""


if __name__ == "__main__":
    robot = Panda()
    visualize(robot)
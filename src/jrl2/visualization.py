import socket

import viser
from scipy.spatial.transform import Rotation
import numpy as np
from time import sleep

from jrl2.collision_detection_single_scene import SingleSceneCollisionChecker
from jrl2.robot import Robot, NP_Q_DICT_TYPE

np.set_printoptions(precision=4, suppress=True)

VISER_PORT = 8080


def assert_no_existing_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("localhost", VISER_PORT)) == 0:
            raise RuntimeError(f"Port {VISER_PORT} is already in use. View processes with 'lsof -i :{VISER_PORT}'")


def _wxyz_from_transform(transform: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(transform[:3, :3]).as_quat(scalar_first=True)


def _position_from_transform(transform: np.ndarray) -> np.ndarray:
    return transform[:3, 3]


def robot_scene(
    robot: Robot,
    collision_checker: SingleSceneCollisionChecker,
    q_dict: NP_Q_DICT_TYPE | None = None,
    show_frames: bool = False,
    use_visual: bool = True,
) -> None:
    assert show_frames is False, "There's a bug with the frames, they are not being updated correctly"
    assert_no_existing_server()
    server = viser.ViserServer(port=VISER_PORT)
    if q_dict is None:
        q_dict = robot.nominal_q
    use_collision = not use_visual

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
    obstacle_color = [1.0, 0.85, 0.0]
    obstacle_opacity = 0.85
    robot_link_color = [1.0, 1.0, 1.0]
    robot_link_opacity = 0.95

    # Add the obstacles to the scene
    for i, sphere in enumerate(collision_checker.spheres):
        name = f"/sphere_{i}"
        server.add_icosphere(
            name=name,
            position=_position_from_transform(sphere.transform),
            radius=sphere.primitive.radius,
            opacity=obstacle_opacity,
            color=obstacle_color,
        )
    for i, box in enumerate(collision_checker.boxes):
        name = f"/box_{i}"
        server.add_mesh_simple(
            name=name,
            vertices=box.vertices,
            faces=box.faces,
            opacity=obstacle_opacity,
            color=obstacle_color,
        )

    def update_configuration(q_dict_in: NP_Q_DICT_TYPE):
        nonlocal meshes_added
        link_mesh_poses = robot.get_all_link_geometry_poses_non_batched(
            q_dict_in, use_visual=use_visual, only_poses=True if meshes_added else False
        )
        for _, link_trimesh_list in link_mesh_poses.items():
            for mesh_name, link_trimesh_object, link_trimesh_pose in link_trimesh_list:
                # scalar_first order is (w, x, y, z)
                wxyz = _wxyz_from_transform(link_trimesh_pose)
                position = _position_from_transform(link_trimesh_pose)
                name = f"/{mesh_name}"
                frame_name = name + "/frame"
                if not meshes_added:
                    if use_visual:
                        mesh_handles[name] = server.add_mesh_trimesh(
                            name=name, mesh=link_trimesh_object, position=position, wxyz=wxyz
                        )
                    else:
                        mesh_handles[name] = server.add_mesh_simple(
                            name=name,
                            vertices=link_trimesh_object.vertices,
                            faces=link_trimesh_object.faces,
                            position=position,
                            opacity=robot_link_opacity,
                            color=robot_link_color,
                            wxyz=wxyz,
                        )
                    if show_frames:
                        mesh_handles[frame_name] = server.add_frame(
                            name=frame_name, position=position, wxyz=wxyz, axes_length=0.075, axes_radius=0.005
                        )
                else:
                    mesh_handles[name].position = position
                    mesh_handles[name].wxyz = wxyz
                    if show_frames:
                        mesh_handles[frame_name].position = position
                        mesh_handles[frame_name].wxyz = wxyz

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
            step=range_ / 500.0,
            initial_value=q_dict[joint.name],
        )
        callbacks[joint.name].on_update(on_slider_update)

    print("Open your browser to http://localhost:8080")
    print("Press Ctrl+C to exit")

    #
    counter = 0
    icosphere_handles = []
    # last_geoms_in_contact = set()

    while True:
        _, colliding_geom_names, contacts = collision_checker.check_collisions(
            q_dict, return_contacts=True, print_timing=counter % 500 == 0
        )
        counter += 1
        if len(colliding_geom_names) > 0:
            print("------------")
            print(len(colliding_geom_names), len(contacts))
            for geom_pair in colliding_geom_names:
                print("geom_pair: ", geom_pair)

        for handle in icosphere_handles:
            handle.visible = False

        # TODO(@jstmn): for some reason the meshes become black when the color is updated. I'm not sure why
        # I was unable to successfully debug.
        # if use_collision:
        #     # Update robot link colors based on collision status
        #     current_geoms_in_contact = set()
        #     for name_pair in colliding_geom_names:
        #         current_geoms_in_contact.add(name_pair[0])
        #         current_geoms_in_contact.add(name_pair[1])

        #     # Reset colors for geoms that are no longer in contact
        #     for geom_name in last_geoms_in_contact - current_geoms_in_contact:
        #         mesh_handle_name = f"/{geom_name}"
        #         if mesh_handle_name in mesh_handles:
        #             print("resetting", mesh_handle_name, " to ", robot_link_color)
        #             mesh_handles[mesh_handle_name].color = robot_link_color

        #     # Set red color for geoms currently in contact
        #     for geom_name in current_geoms_in_contact:
        #         mesh_handle_name = f"/{geom_name}"
        #         if mesh_handle_name in mesh_handles:
        #             print("(contact) setting", mesh_handle_name, " to ", [1.0, 0.0, 0.0])
        #             mesh_handles[mesh_handle_name].color = [1.0, 0.0, 0.0]
        #     last_geoms_in_contact = current_geoms_in_contact

        # for i, contact in enumerate(contacts):
        while len(contacts) > len(icosphere_handles):
            icosphere_handles.append(
                server.scene.add_icosphere(
                    name=f"contact_{len(icosphere_handles)}",
                    position=contacts[len(icosphere_handles)].point,
                    radius=0.01,
                )
            )

        for i, contact in enumerate(contacts):
            icosphere_handles[i].position = contact.point
            icosphere_handles[i].visible = True

        sleep(1 / 100)

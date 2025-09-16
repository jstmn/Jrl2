import numpy as np
from time import time

from trimesh.primitives import Sphere, Box
import trimesh

from jrl2.robot import Robot


def _translation_to_SE3(translation: np.ndarray) -> np.ndarray:
    se3 = np.eye(4)
    se3[:3, 3] = translation
    return se3


class SingleSceneCollisionChecker:
    def __init__(self, robot: Robot, use_visual: bool):
        self._robot = robot
        self._use_visual = use_visual
        self._spheres: list[Sphere] = []
        self._boxes: list[Box] = []
        self._collision_manager = trimesh.collision.CollisionManager()

        # Add meshes to scene
        link_mesh_poses = self._robot.get_all_link_geometry_poses_non_batched(
            robot.midpoint_configuration, use_visual=self._use_visual
        )
        for _, link_trimesh_list in link_mesh_poses.items():
            for mesh_name, link_trimesh_object, link_trimesh_pose in link_trimesh_list:
                self._collision_manager.add_object(
                    mesh=link_trimesh_object,
                    name=mesh_name,
                    transform=link_trimesh_pose,
                )

    @property
    def spheres(self) -> list[Sphere]:
        return self._spheres

    @property
    def boxes(self) -> list[Box]:
        return self._boxes

    def check_collisions(
        self,
        q_dict: dict[str, float],
        dont_filter: bool = False,
        print_timing: bool = False,
        return_contacts: bool = False,
        q_range_padding: float | None = None,
    ) -> tuple[bool, set[tuple[str, str]]]:
        """Check for collisions at the given robot configuration."""
        self._robot.assert_valid_configuration(q_dict, padding=q_range_padding)
        t0 = time()

        # Run FK
        link_mesh_poses = self._robot.get_all_link_geometry_poses_non_batched(
            q_dict, use_visual=self._use_visual, only_poses=True
        )
        for _, link_trimesh_list in link_mesh_poses.items():
            for mesh_name, _, link_trimesh_pose in link_trimesh_list:
                self._collision_manager.set_transform(name=mesh_name, transform=link_trimesh_pose)
        t_fk = time()

        # Run collision checking
        if return_contacts:
            is_collision, names, contacts = self._collision_manager.in_collision_internal(
                return_names=True, return_data=True
            )
        else:
            is_collision, names = self._collision_manager.in_collision_internal(return_names=True, return_data=False)

        # Timing
        t_collision = time()
        if print_timing:
            print(f"t_fk, t_collision-check (ms): {1000*(t_fk - t0):0.3f},\t{1000*(t_collision - t_fk):0.3f}")
        if dont_filter:
            if return_contacts:
                return is_collision, names, contacts
            else:
                return is_collision, names

        # Filter out pairs of geometries that are known a priori to be always colliding
        pairs_filtered = set()
        for name in names:
            if not self._robot.geometries_cant_collide(name[0], name[1], use_visual=self._use_visual):
                pairs_filtered.add(name)

        contacts_filtered = []
        if return_contacts:
            for contact in contacts:
                assert len(contact.names) == 2
                name_pair = tuple(contact.names)
                if not self._robot.geometries_cant_collide(name_pair[0], name_pair[1], use_visual=self._use_visual):
                    contacts_filtered.append(contact)
            return len(pairs_filtered) > 0, pairs_filtered, contacts_filtered
        return len(pairs_filtered) > 0, pairs_filtered

    def add_sphere(self, sphere: Sphere) -> str:
        sphere_name = f"sphere_{len(self._spheres)}"
        self._spheres.append(sphere)
        self._collision_manager.add_object(
            mesh=sphere,
            name=sphere_name,
        )
        return sphere_name

    def add_box(self, box: Box) -> str:
        box_name = f"box_{len(self._boxes)}"
        self._boxes.append(box)
        self._collision_manager.add_object(
            mesh=box,
            name=box_name,
        )
        return box_name

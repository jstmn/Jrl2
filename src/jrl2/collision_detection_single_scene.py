from fcl import Contact
import numpy as np
from time import time

from trimesh.collision import ContactData
from trimesh.primitives import Sphere, Box
import trimesh

from jrl2.robot import Robot


class SingleSceneCollisionChecker:
    def __init__(self, robot: Robot, use_visual: bool):
        self._robot = robot
        self._use_visual = use_visual
        self._spheres: list[Sphere] = []
        self._boxes: list[Box] = []
        self._collision_manager = trimesh.collision.CollisionManager()
        self._user_ignored_pairs: set[tuple[str, str]] = set()

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
        filter_ignorable: bool = True,
        print_timing: bool = False,
        return_contacts: bool = False,
        q_range_padding: float | None = None,
    ) -> tuple[bool, list[tuple[str, str]], list[ContactData]]:
        """Check for collisions at the given robot configuration.

        Return:
            tuple[bool, list[tuple[str, str]], list[ContactData]]:
                - is_collision: Whether there is a collision
                - col_pairs: List of names of the colliding objects (pairs of names)
                - contacts: List of ContactData objects. Will be empty if return_contacts is False.
        """
        self._robot.assert_valid_configuration(q_dict, padding=q_range_padding)
        
        def pair_should_be_ignored(name_pair: tuple[str, str]) -> bool:
            assert len(name_pair) == 2, f"Expected tuple of length 2, got {len(name_pair)}"
            name_pair_ordered = self._robot.return_ordered_geometry_name_pair(name_pair[0], name_pair[1])
            if name_pair_ordered in self._user_ignored_pairs:
                return True
            return self._robot.geometries_cant_collide(
                name_pair_ordered[0], name_pair_ordered[1], use_visual=self._use_visual
            )

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
        contacts = []
        if return_contacts:
            # col_pairs : set of 2-tuples. The set of pairwise collisions. Each tuple contains two names in alphabetical
            # order indicating that the two corresponding objects are in collision.
            # contacts : list of ContactData
            _, col_pairs, contacts = self._collision_manager.in_collision_internal(return_names=True, return_data=True)
        else:
            _, col_pairs = self._collision_manager.in_collision_internal(return_names=True, return_data=False)
        assert True if return_contacts else (len(contacts) == 0), "contacts should be empty if return_contacts is False"

        # Timing
        t_collision = time()
        if print_timing:
            print(f"t_fk, t_collision-check (ms): {1000*(t_fk - t0):0.3f},\t{1000*(t_collision - t_fk):0.3f}")

        if not filter_ignorable:
            return len(col_pairs) > 0, col_pairs, contacts

        # Filter out pairs of geometries that are known a priori to be always colliding
        pairs_updated = set()
        for col_tuple in col_pairs:
            if pair_should_be_ignored(col_tuple):
                continue
            pairs_updated.add(col_tuple)

        contacts_updated = []
        if return_contacts:
            for contact in contacts:
                assert len(contact.names) == 2, f"Expected 2 names, got {len(contact.names)} (names={contact.names})"
                col_tuple = tuple(contact.names)
                if pair_should_be_ignored(col_tuple):
                    continue
                contacts_updated.append(contact)
        return len(pairs_updated) > 0, pairs_updated, contacts_updated

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

    def ignore_pair(self, name_pair: tuple[str, str]):
        assert len(name_pair) == 2, f"Expected tuple of length 2, got {len(name_pair)}"
        self._user_ignored_pairs.add(self._robot.return_ordered_geometry_name_pair(name_pair[0], name_pair[1]))

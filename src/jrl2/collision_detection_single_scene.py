from time import time

import fcl
import trimesh
from trimesh.collision import ContactData
from trimesh.primitives import Box, Sphere

from jrl2.robot import Robot


class SingleSceneCollisionChecker:
    def __init__(self, robot: Robot, use_visual: bool):
        self._robot = robot
        self._use_visual = use_visual
        self._spheres: list[Sphere] = []
        self._boxes: list[Box] = []
        self._collision_manager = trimesh.collision.CollisionManager()
        self._user_ignored_pairs: set[tuple[str, str]] = set()
        self._ignored_pairs: set[tuple[str, str]] = set()

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
        self._rebuild_ignored_pairs()

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
    ) -> tuple[bool, set[tuple[str, str]], list[ContactData]]:
        """Check for collisions at the given robot configuration.

        Return:
            tuple[bool, set[tuple[str, str]], list[ContactData]]:
                - is_collision: Whether there is a collision
                - col_pairs: Pairwise colliding object names
                - contacts: ContactData objects. Empty if return_contacts is False.
        """
        self._robot.assert_valid_configuration(q_dict, padding=q_range_padding)

        t0 = time()
        link_mesh_poses = self._robot.get_all_link_geometry_poses_non_batched(
            q_dict, use_visual=self._use_visual, only_poses=True
        )
        self._set_transforms(link_mesh_poses)
        t_fk = time()

        is_collision, col_pairs, contacts = self._in_collision_internal(
            return_contacts=return_contacts, filter_ignorable=filter_ignorable
        )
        t_collision = time()
        if print_timing:
            print(f"t_fk, t_collision-check (ms): {1000*(t_fk - t0):0.3f},\t{1000*(t_collision - t_fk):0.3f}")
        return is_collision, col_pairs, contacts

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
        self._rebuild_ignored_pairs()

    def _rebuild_ignored_pairs(self):
        ignored = set(self._user_ignored_pairs)
        names = list(self._collision_manager._objs.keys())
        for i, name_1 in enumerate(names):
            for name_2 in names[i + 1 :]:
                if self._robot.geometries_cant_collide(name_1, name_2, use_visual=self._use_visual):
                    ignored.add(self._robot.return_ordered_geometry_name_pair(name_1, name_2))
        self._ignored_pairs = ignored

    def _set_transforms(self, link_mesh_poses: dict):
        """Set all robot geometry poses. FCL collide reads each object's transform directly."""
        for _, link_trimesh_list in link_mesh_poses.items():
            for mesh_name, _, link_trimesh_pose in link_trimesh_list:
                collision_object = self._collision_manager._objs[mesh_name]["obj"]
                collision_object.setRotation(link_trimesh_pose[:3, :3])
                collision_object.setTranslation(link_trimesh_pose[:3, 3])

    def _in_collision_internal(
        self, return_contacts: bool, filter_ignorable: bool
    ) -> tuple[bool, set[tuple[str, str]], list[ContactData]]:
        """FCL self-collision, inlined from trimesh.CollisionManager.in_collision_internal.

        Ignored pairs (always/never colliding, plus user ignores) are skipped before the
        narrow-phase check. If we are not collecting contact points, FCL is asked for a
        single contact and we stop after the first non-ignored collision.
        """
        # Visualization gets up to 100 contacts per pair. Boolean queries only need one real hit.
        # filter_ignorable=False (e.g. filtering-data generation) still needs every pair.
        collect_all_pairs = return_contacts or not filter_ignorable
        request = fcl.CollisionRequest(
            num_max_contacts=100 if return_contacts else 1,
            enable_contact=return_contacts,
        )
        ignored_pairs = self._ignored_pairs if filter_ignorable else set()
        objects = [(name, data["obj"]) for name, data in self._collision_manager._objs.items()]
        col_pairs: set[tuple[str, str]] = set()
        contacts: list[ContactData] = []

        for i, (name_1, object_1) in enumerate(objects):
            for name_2, object_2 in objects[i + 1 :]:
                name_pair = self._robot.return_ordered_geometry_name_pair(name_1, name_2)
                if name_pair in ignored_pairs:
                    continue

                pair_result = fcl.CollisionResult()
                fcl.collide(object_1, object_2, request, pair_result)
                if not pair_result.is_collision:
                    continue

                col_pairs.add(name_pair)
                if return_contacts:
                    for contact in pair_result.contacts:
                        contacts.append(ContactData(name_pair, contact))
                if not collect_all_pairs:
                    return True, col_pairs, contacts

        return len(col_pairs) > 0, col_pairs, contacts

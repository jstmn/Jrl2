import numpy as np
from jrl2.robot import Robot
from dataclasses import dataclass
import trimesh


def _translation_to_SE3(translation: np.ndarray) -> np.ndarray:
    se3 = np.eye(4)
    se3[:3, 3] = translation
    return se3


@dataclass
class Sphere:
    world_T_center: np.ndarray
    radius: float
    name: str


class SingleSceneCollisionChecker:
    def __init__(self, robot: Robot):
        self._robot = robot
        self._robots = []
        self._spheres = []
        self._capsules = []
        self._boxes = []
        self._collision_manager = trimesh.collision.CollisionManager()

    def clear_scene(self):
        for sphere in self._spheres:
            self._collision_manager.remove_object(sphere.name)
        for capsule in self._capsules:
            self._collision_manager.remove_object(capsule.name)
        for box in self._boxes:
            self._collision_manager.remove_object(box.name)
        self._spheres = []
        self._capsules = []
        self._boxes = []

    def check_collisions(
        self, q_dict: dict[str, float]
    ) -> tuple[bool, set[tuple[str, str]], list[trimesh.collision.ContactData]]:
        link_mesh_poses = self._robot.get_all_link_mesh_poses_non_batched(q_dict, use_visual=False)
        # TODO: Add robots to the collision manager
        is_collision, names, contacts = self._collision_manager.in_collision_internal(
            return_names=True, return_data=True
        )
        return is_collision, names, contacts

    def add_sphere(self, center: np.ndarray, radius: float):
        world_T_center = _translation_to_SE3(center)
        sphere_name = f"sphere_{len(self._spheres)}"
        self._spheres.append(Sphere(world_T_center=world_T_center, radius=radius, name=sphere_name))
        trimesh_sphere = trimesh.primitives.Sphere(radius=radius, center=center)
        self._collision_manager.add_object(
            mesh=trimesh_sphere,
            name=sphere_name,
            # transform=world_T_center,
        )

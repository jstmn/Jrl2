import numpy as np
from time import time
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

        # Add meshes to scene
        link_mesh_poses = self._robot.get_all_link_geometry_poses_non_batched(
            robot.midpoint_configuration, use_visual=False
        )
        for _, link_trimesh_list in link_mesh_poses.items():
            for mesh_name, link_trimesh_object, link_trimesh_pose in link_trimesh_list:
                self._collision_manager.add_object(
                    mesh=link_trimesh_object,
                    name=mesh_name,
                    transform=link_trimesh_pose,
                )

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
        self, q_dict: dict[str, float], use_visual: bool = False, dont_filter: bool = False, print_timing: bool = False
    ) -> tuple[bool, set[tuple[str, str]]]:
        """Check for collisions at the given robot configuration."""
        t0 = time()

        # Run FK
        link_mesh_poses = self._robot.get_all_link_geometry_poses_non_batched(q_dict, use_visual=use_visual)
        for _, link_trimesh_list in link_mesh_poses.items():
            for mesh_name, _, link_trimesh_pose in link_trimesh_list:
                self._collision_manager.set_transform(name=mesh_name, transform=link_trimesh_pose)
        t_fk = time()

        # Run collision checking
        is_collision, names = self._collision_manager.in_collision_internal(return_names=True, return_data=False)

        # Timing
        t_collision = time()
        if print_timing:
            print(f"runtime fk, collision: {t_fk - t0},\t{t_collision - t_fk}")
        if dont_filter:
            return is_collision, names

        # Filter out pairs of geometries that are known a priori to be always colliding
        names_real = set()
        for name in names:
            if not self._robot.links_cant_collide(name[0], name[1]):
                names_real.add(name)
        return len(names_real) > 0, names_real

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

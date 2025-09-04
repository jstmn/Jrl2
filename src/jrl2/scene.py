import trimesh
import torch
from trimesh.primitives import Sphere, Capsule, Box
from trimesh.collision import scene_to_collision
import random
from abc import ABC, abstractmethod
from jrl2.robot import Robot


def get_random_hex() -> str:
    return hex(random.randint(0, 1e8))


class Scene(ABC):
    def __init__(self, robot: Robot):
        self._robot = robot

    @abstractmethod
    def add_sphere(self, center: torch.Tensor, radius: float | torch.Tensor):
        pass

    @abstractmethod
    def add_capsule(self, start: torch.Tensor, end: torch.Tensor, radius: float | torch.Tensor, sections: int = 32):
        pass

    @abstractmethod
    def add_box(self, extents: torch.Tensor, transform: torch.Tensor, bounds: torch.Tensor):
        pass

    @abstractmethod
    def find_robot_environment_collisions(
        self, robot_q: torch.Tensor
    ) -> list[tuple[trimesh.primitives.Primitive, trimesh.primitives.Primitive]]:
        pass


class SingleEnvironmentScene(Scene):
    def __init__(self, robot: Robot):
        super().__init__(robot)
        self._spheres = []
        self._capsules = []
        self._boxes = []
        self._spheres_hash = get_random_hex()
        self._capsules_hash = get_random_hex()
        self._boxes_hash = get_random_hex()
        self._trimesh_collision_manager, self._trimesh_collision_nodes = scene_to_collision(
            self._robot._yourdfpy_model.collision_scene
        )
        self._scene_hash = self._sum_hash

    def _delete_scene(self):
        self._spheres = []
        self._capsules = []
        self._boxes = []
        self._spheres_hash = get_random_hex()
        self._capsules_hash = get_random_hex()
        self._boxes_hash = get_random_hex()
        for i in range(len(self._spheres)):
            self._trimesh_collision_manager.remove_object(f"sphere_{i}")
        for i in range(len(self._capsules)):
            self._trimesh_collision_manager.remove_object(f"capsule_{i}")
        for i in range(len(self._boxes)):
            self._trimesh_collision_manager.remove_object(f"box_{i}")

    # fcl.fcl.CollisionObject:
    # ['getNodeType', 'getObjectType', 'getQuatRotation', 'getRotation', 'getTransform', 'getTranslation', 'isFree', 'isOccupied', 'isUncertain', 'setQuatRotation', 'setRotation', 'setTransform', 'setTranslation']

    #

    def _update_scene(self):
        if self._sum_hash != self._scene_hash:
            assert self._robot._yourdfpy_model.collision_scene is not None
            print(self._robot._yourdfpy_model.collision_scene)
            # self._trimesh_collision_nodes:  {'node': <fcl.fcl.CollisionObject object>, 'node_1': <fcl.fcl.CollisionObject object>}
            self._trimesh_collision_manager, self._trimesh_collision_nodes = scene_to_collision(
                self._robot._yourdfpy_model.collision_scene
            )
            # scene_to_collision returns:
            #   1. manager (CollisionManager) – CollisionManager for objects in scene
            #   2. objects ({node name: CollisionObject}) – Collision objects for nodes in scene
            self._scene_hash = self._sum_hash

            print(f"self._trimesh_collision_manager: {self._trimesh_collision_manager}")

            assert len(self._spheres) > 0
            for i in range(len(self._spheres)):
                self._trimesh_collision_manager.add_object(
                    mesh=self._spheres[i], name=f"sphere_{i}", transform=self._spheres[i].transform
                )

            for i in range(len(self._capsules)):
                self._trimesh_collision_manager.add_object(
                    mesh=self._capsules[i], name=f"capsule_{i}", transform=self._capsules[i].transform
                )

            for i in range(len(self._boxes)):
                self._trimesh_collision_manager.add_object(
                    mesh=self._boxes[i], name=f"box_{i}", transform=self._boxes[i].transform
                )

            print()
            print("self._trimesh_collision_nodes: ", self._trimesh_collision_nodes)
            for collision_node in self._trimesh_collision_nodes.values():
                print("collision_node: ", collision_node)
                print(dir(collision_node))
            print()

    @property
    def _sum_hash(self) -> str:
        return self._spheres_hash + self._capsules_hash + self._boxes_hash

    def add_sphere(self, center: torch.Tensor, radius: float | torch.Tensor):
        """
        Add a sphere to the scene. Internally, this creates a trimesh Sphere Primitive.

        Parameters:
            center: torch.Tensor[(B, 3)], The center point of the sphere.
            radius: float or torch.Tensor, Radius of the sphere.
        """
        assert (
            len(center.shape) == 1 and center.shape[0] == 3
        ), "SingleEnvironmentScene only supports single spheres, use BatchedEnvironmentScene for batched operations"
        self._spheres_hash = get_random_hex()
        self._spheres.append(Sphere(radius=radius, center=center, mutable=False))
        assert len(self._spheres) > 0

    def add_capsule(self, start: torch.Tensor, end: torch.Tensor, radius: float | torch.Tensor, sections: int = 32):
        """
        Add a capsule to the scene. Internally, this creates a trimesh Capsule Primitive.

        Parameters:
            start: torch.Tensor[(B, 3)], The starting point of the capsule.
            end: torch.Tensor[(B, 3)], The ending point of the capsule.
            radius: float or torch.Tensor, Radius of the cylinder.
            sections: int, Number of facets in the circle.
        """
        raise NotImplementedError(
            "Need to use rotation of capsules. Currently assumes that the capsule is aligned with the z-axis."
        )
        self._capsules_hash = get_random_hex()
        height = torch.norm(end - start).item()
        transform = torch.eye(4)
        transform[:3, 3] = (start + end) / 2
        self._capsules.append(
            Capsule(radius=radius, height=height, transform=transform, sections=sections, mutable=False)
        )

    def add_box(self, extents: torch.Tensor, transform: torch.Tensor, bounds: torch.Tensor):
        """
        Add a box to the scene. Internally, this creates a trimesh Box Primitive. The description of the parameters below are partly copied from the [trimesh documentation](https://trimesh.org/trimesh.primitives.html#trimesh.primitives.Box).

        Parameters:
            extents: torch.Tensor[(B, 3)], Length of each side of the 3D box.
            transform: torch.Tensor[(4, 4) or (B, 4, 4)], Homogeneous transformation matrix for box center.
            bounds: torch.Tensor[(2, 3) or (B, 2, 3)], Axis aligned bounding box, if passed extents and transform will be derived from this.
        """
        assert extents.shape == (3,)
        assert transform.shape == (4, 4)
        assert bounds.shape == (2, 3)
        self._boxes_hash = get_random_hex()
        self._boxes.append(Box(extents, transform, bounds, mutable=False))

    def find_robot_environment_collisions(
        self, robot_q: torch.Tensor
    ) -> list[tuple[trimesh.primitives.Primitive, trimesh.primitives.Primitive]]:
        assert len(self._spheres) > 0
        self._update_scene()
        is_contact, names = self._trimesh_collision_manager.in_collision_internal(return_names=True, return_data=False)
        print()
        print("find_robot_environment_collisions()")
        print(f"names: {names}")
        return is_contact


class BatchedEnvironmentScene(Scene):
    def __init__(self, robot: Robot):
        super().__init__(robot)
        assert NotImplementedError(
            "Not implemented yet. Need to consider how to save the capsules for each environment"
        )

    def add_sphere(self, center: torch.Tensor, radius: float | torch.Tensor):
        """
        Add a sphere to the scene. Internally, this creates a trimesh Sphere Primitive.

        Parameters:
            center: torch.Tensor[(B, 3)], The center point of the sphere.
            radius: float or torch.Tensor, Radius of the sphere.
        """
        self._spheres_hash = get_random_hex()
        B = center.shape[0]

    def add_capsule(self, start: torch.Tensor, end: torch.Tensor, radius: float | torch.Tensor, sections: int = 32):
        """
        Add a capsule to the scene. Internally, this creates a trimesh Capsule Primitive.

        Parameters:
            start: torch.Tensor[(B, 3)], The starting point of the capsule.
            end: torch.Tensor[(B, 3)], The ending point of the capsule.
            radius: float or torch.Tensor, Radius of the cylinder.
            sections: int, Number of facets in the circle.
        """
        raise NotImplementedError(
            "Need to use rotation of capsules. Currently assumes that the capsule is aligned with the z-axis."
        )

    def add_box(self, extents: torch.Tensor, transform: torch.Tensor, bounds: torch.Tensor):
        """
        Add a box to the scene. Internally, this creates a trimesh Box Primitive. The description of the parameters below are partly copied from the [trimesh documentation](https://trimesh.org/trimesh.primitives.html#trimesh.primitives.Box).

        Parameters:
            extents: torch.Tensor[(B, 3)], Length of each side of the 3D box.
            transform: torch.Tensor[(B, 4, 4)], Homogeneous transformation matrix for box center.
            bounds: torch.Tensor[(B, 2, 3)], Axis aligned bounding box, if passed extents and transform will be derived from this.
        """
        self._boxes_hash = get_random_hex()
        B = bounds.shape[0]
        assert extents.shape == (B, 3)
        assert transform.shape == (B, 4, 4)
        assert bounds.shape == (B, 2, 3)

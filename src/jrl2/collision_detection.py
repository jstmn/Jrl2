import coal
import numpy as np
from jrl2.robot import Robot
from dataclasses import dataclass


def _translation_to_SE3(translation: np.ndarray) -> np.ndarray:
    se3 = np.eye(4)
    se3[:3, 3] = translation
    return se3


@dataclass
class Sphere:
    world_T_sphere: np.ndarray
    radius: float
    name: str


class NonBatchedCollisionChecker:
    def __init__(self, robot: Robot):
        self._robot = robot
        self._spheres = []
        self._capsules = []
        self._boxes = []

    def check_collisions(self, q: np.ndarray) -> bool:
        link_poses = self._robot.get_all_link_poses_non_batched(q)

        shape1 = coal.Ellipsoid(0.7, 1.0, 0.8)
        shape2 = coal.Sphere(0.5)

        # Define the shapes' placement in 3D space
        T1 = coal.Transform3s()
        T1.setTranslation(np.random.rand(3))
        T1.setRotation(np.random.rand(3, 3))
        T2 = coal.Transform3s()
        # Using np arrays also works
        T1.setTranslation(np.random.rand(3))
        T2.setRotation(np.random.rand(3, 3))

        # Define collision requests and results
        col_req = coal.CollisionRequest()
        col_res = coal.CollisionResult()

        # Collision call
        coal.collide(shape1, T1, shape2, T2, col_req, col_res)

        # Accessing the collision result once it has been populated
        print("Is collision? ", {col_res.isCollision()})
        if col_res.isCollision():
            contact: coal.Contact = col_res.getContact(0)
            print("Penetration depth: ", contact.penetration_depth)
            print(
                "Distance between the shapes including the security margin: ",
                contact.penetration_depth + col_req.security_margin,
            )
            print("Witness point shape1: ", contact.getNearestPoint1())
            print("Witness point shape2: ", contact.getNearestPoint2())
            print("Normal: ", contact.normal)

        # Before running another collision call, it is important to clear the old one
        col_res.clear()

    def add_sphere(self, center: np.ndarray, radius: float):
        self._spheres.append(Sphere(_translation_to_SE3(center), radius, name=f"sphere_{len(self._spheres)}"))

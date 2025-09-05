import coal
import os
from pathlib import Path
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

        def loadConvexMesh(file_name: str):
            loader = coal.MeshLoader()
            bvh: coal.BVHModelBase = loader.load(file_name)
            bvh.buildConvexHull(True, "Qt")
            return bvh.convex

        # Test loading
        for link in self._robot.links:
            mesh_filepath_raw = str(link.visuals[0].geometry.mesh.filename)
            assert mesh_filepath_raw.startswith(
                "package://"
            ), f"Mesh file {mesh_filepath_raw} does not start with 'package://'"
            package_url = mesh_filepath_raw.replace("package://", "")
            assert "meshes" in package_url, f"Mesh file {package_url} does not contain 'meshes'"
            mesh_relative_path = package_url.split("meshes", 1)[1]  # Gets "/visual/link0.dae"
            # Need to remove leading slash from mesh_relative_path to avoid absolute path issues
            mesh_relative_path = mesh_relative_path.lstrip("/")
            mesh_filepath = os.path.join(str(self._robot._robot_description_dir), "meshes", mesh_relative_path)
            assert Path(mesh_filepath).exists(), f"Mesh file {mesh_filepath} does not exist"

        np.random.seed(0)
        shape1 = coal.Sphere(0.5)
        shape2 = loadConvexMesh(
            "/home/jstm/.cache/robot_descriptions/example-robot-data/robots/panda_description/meshes/collision/link0.stl"
        )

        # Define the shapes' placement in 3D space
        T1 = coal.Transform3s()
        T1.setTranslation(np.array([0.0, 0.0, 0.0]))
        T1.setRotation(np.eye(3))
        T2 = coal.Transform3s()
        T1.setTranslation(np.array([0.0, 0.0, 0.0]))
        T2.setRotation(np.eye(3))

        # Define collision requests and results
        col_req = coal.CollisionRequest()
        col_req.security_margin = 0.25
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

# Jrl2
This is an updated version of [Jrl](https://github.com/jstmn) with the following changes:
* Single query self and environment collision checks will be performed by [coal](https://github.com/coal-library/coal). This will allow for checking for collisions between cylinders, spheres, ellipsoids, cuboids, pointclouds, and potentially other meshes in the environment with the robots collision meshes. This will remove the projects Klampt dependency. This will be in addition to the batched collision checking available as well.
* Batched collision checking will be limited to only sphere-capsule, sphere-cuboid, sphere-sphere collisions. These checks can all be performed analytically (i.e. no optimization required) which will make them very fast. JIT compiling can be used here as well.
* Numpy will be completely removed. This will be a (nearly) entirely pytorch library.
* 3D math operations using a thirdparty, like [PyTorch3d](https://pytorch3d.org/) or [RoMa](https://naver.github.io/roma/). Altternatively, the 3d operations will be cleaned up, standardized, and be rewritten using JIT compilation. This can be done with the [warp](https://github.com/NVIDIA/warp) or [torchdynamo](https://github.com/pytorch/torchdynamo)
* The API for specifying which portion of the kinematic tree an operation should perform on will be reformatted. Currently, this is fixed in advance by specifying a base frame and end effector frame in the `Robot` class initializer. This is severly limiting however, because often times you may want the transform from different frames to one another. Concretely, this means that Forward/Inverse Kinematics is performed for one fixed kinematic chain per `Robot` subclass. There are two options of how to rewrite this. 
  1. For the first, all APIs will accept a start and end link. A search will be performed to find the connecting path between these two, and that portion of the kinmatic chain will be used in the function. 
  2. For the second, joint groups will need to be created ahead of time and saved to the class. This is the same as how MoveIt! works - there are predefined planning groups, you use one per planning query. 

 I think the first approach - everything requires a base and target frame is more elegant and intuitive. I'm not aware of any clear downsides to that approach. A default base and end effector frame should probably be set.
* A new visualization library will be used. Options include [Pybullet](Pybullet), [RoboMeshCat/Meshcat](https://github.com/petrikvladimir/RoboMeshCat), [scikit-robot](https://github.com/iory/scikit-robot). Alternatively it would be a fun exercise to write my own or extend a current one like [kiss3d](kiss3d). This will remove the klampt dep.



# Installation

How to install the project?
```
# uv sync ?!
```

# uv cheat sheet

Creating and working on Python projects, i.e., with a pyproject.toml.

- `uv add`: Add a dependency to the project.
- `uv remove`: Remove a dependency from the project.
- `uv sync`: Sync the project's dependencies with the environment.
- `uv lock`: Create a lockfile for the project's dependencies.
- `uv run`: Run a command in the project environment.
- `uv tree`: View the dependency tree for the project.


**Example commands:**
```
uv run scripts/hello.py  # run a script
```

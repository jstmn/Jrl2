# Jrl2

<figure>
  <img src="media/hero_gif.gif" width="600">
  <figcaption>The Panda robot visualized with non-batched (single scene) collision checking running.</figcaption>
</figure>



**This is an updated version of [Jrl](https://github.com/jstmn) with the following changes:**
* Visualization with [Viser](https://viser.studio/main/) rather than []
* Robot urdfs from [robot-descriptions](https://github.com/robot-descriptions/robot_descriptions.py)
* Non-batched (single scene) collision checking using [trimesh](https://trimesh.org/) and [fcl](https://github.com/BerkeleyAutomation/python-fcl). The supported collision shapes are: TriangleP, Box, Sphere, Ellipsoid, Capsule, Cone, Convex, Cylinder, Half-Space, Plane, Mesh, OcTree. Note that the (coal)[https://github.com/coal-library/coal] library allows for adding a margin between objects. This would be quite useful for this package.
* Batched collision checking will be limited to only sphere-capsule, sphere-cuboid, sphere-sphere collisions. These checks can all be performed analytically (i.e. no optimization required) which will make them very fast. JIT compiling can be used here as well.
* 3D math operations using a thirdparty library, such as [PyTorch3d](https://pytorch3d.org/) or [RoMa](https://naver.github.io/roma/) rather than self written functions. Alternatively, the 3d operations will be cleaned up, standardized, and be rewritten using JIT compilation. This can be done with the [warp](https://github.com/NVIDIA/warp) or [torchdynamo](https://github.com/pytorch/torchdynamo)
* The API for specifying which portion of the kinematic tree an operation should perform on will be reformatted. Currently, this is fixed in advance by specifying a base frame and end effector frame in the `Robot` class initializer. This is severly limiting however, because often times you may want the transform from different frames to one another. Instead, joint configurations will be provided via a dictuanary from joint name to joint angle. This should alleviate all uncertianty.


# Installation
``` bash
git clone https://github.com/jstmn/Jrl2.git && cd Jrl2/
uv sync --no-dev         # Remove '--no-dev' to install with development dependencies: `uv sync`
uv pip install -e .
```

# uv cheat sheet

Creating and working on Python projects, i.e., with a pyproject.toml.

- `uv add`: Add a dependency to the project.
- `uv remove`: Remove a dependency from the project.
- `uv sync`: Sync the project's dependencies with the environment.
- `uv lock`: Create a lockfile for the project's dependencies.
- `uv run ...`: Run a command in the project environment. (Examples: `uv run black`, `uv run pytest tests/`)
- `uv tree`: View the dependency tree for the project.


**Example commands:**
```
uv run scripts/hello.py  # run a script
```

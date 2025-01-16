import dataclasses
import torch

@dataclasses.dataclass
class Spheres:
    """ centers should be [B, 3], radii should be [B] """
    centers: torch.Tensor
    radii: torch.Tensor


@dataclasses.dataclass
class Capsules:
    """ endpoints_1 and endpoints_2 should be [B, 3], radii should be [B] """
    endpoints_1: torch.Tensor
    endpoints_2: torch.Tensor
    radii: torch.Tensor


@dataclasses.dataclass
class AABBs:
    """ lower_corners should be [B, 3], centers should be [B, 3] """
    centers: torch.Tensor
    lower_corners: torch.Tensor
    upper_corners: torch.Tensor


@dataclasses.dataclass
class Pointclouds:
    """ points should be [B, N, 3]
    """
    points: torch.Tensor


@dataclasses.dataclass
class Environment:
    name: str
    spheres: Spheres
    capsules: Capsules
    aabbs: AABBs
    pointclouds: Pointclouds


from typing import TypedDict

from src.my_types import ArrayFloat32Nx2


class NeighboringParticles(TypedDict):
    left: ArrayFloat32Nx2
    right: ArrayFloat32Nx2
    top: ArrayFloat32Nx2
    bottom: ArrayFloat32Nx2

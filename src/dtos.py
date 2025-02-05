from dataclasses import dataclass, field

from src.my_types import ArrayFloat32Nx2


@dataclass
class NeighboringParticles:
    left: ArrayFloat32Nx2
    right: ArrayFloat32Nx2
    top: ArrayFloat32Nx2
    bottom: ArrayFloat32Nx2

    delta_top_bottom: ArrayFloat32Nx2 = field(init=False)
    delta_right_left: ArrayFloat32Nx2 = field(init=False)

    def __post_init__(self) -> None:
        self.delta_top_bottom = self.top - self.bottom
        self.delta_right_left = self.right - self.left

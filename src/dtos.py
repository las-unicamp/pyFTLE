from dataclasses import dataclass, field

from src.my_types import ArrayFloat32Nx2


@dataclass
class NeighboringParticles:
    left: ArrayFloat32Nx2
    right: ArrayFloat32Nx2
    top: ArrayFloat32Nx2
    bottom: ArrayFloat32Nx2

    initial_delta_top_bottom: ArrayFloat32Nx2 = field(init=False)
    initial_delta_right_left: ArrayFloat32Nx2 = field(init=False)

    def __post_init__(self) -> None:
        self.initial_delta_top_bottom = self.top - self.bottom
        self.initial_delta_right_left = self.right - self.left
        self.initial_centroid = (self.left + self.right + self.top + self.bottom) / 4.0

    @property
    def delta_top_bottom(self) -> ArrayFloat32Nx2:
        return self.top - self.bottom

    @property
    def delta_right_left(self) -> ArrayFloat32Nx2:
        return self.right - self.left

    @property
    def centroid(self) -> ArrayFloat32Nx2:
        return (self.left + self.right + self.top + self.bottom) / 4.0

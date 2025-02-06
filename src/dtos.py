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
        object.__setattr__(self, "initial_delta_top_bottom", self.delta_top_bottom)
        object.__setattr__(self, "initial_delta_right_left", self.delta_right_left)
        object.__setattr__(self, "initial_centroid", self.centroid)

    @property
    def delta_top_bottom(self) -> ArrayFloat32Nx2:
        return self.top - self.bottom

    @property
    def delta_right_left(self) -> ArrayFloat32Nx2:
        return self.right - self.left

    @property
    def centroid(self) -> ArrayFloat32Nx2:
        return (self.left + self.right + self.top + self.bottom) / 4.0

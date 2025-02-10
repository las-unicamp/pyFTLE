from dataclasses import dataclass, field

import numpy as np

from src.my_types import ArrayFloat32N4x2, ArrayFloat32Nx2


@dataclass
class NeighboringParticles:
    """
    Stores the positions of neighboring particles for a set of particles.

    The `positions` array has shape (4*N, 2), where:
    - First N rows → Left  neighbors (x, y)
    - Next  N rows → Right neighbors (x, y)
    - Next  N rows → Top   neighbors (x, y)
    - Last  N rows → Bottom neighbors (x, y)

    This structure allows efficient vectorized computations
    """

    positions: ArrayFloat32N4x2  # Flattened representation with shape (4*N, 2)

    initial_delta_top_bottom: ArrayFloat32Nx2 = field(init=False)
    initial_delta_right_left: ArrayFloat32Nx2 = field(init=False)
    initial_centroid: ArrayFloat32Nx2 = field(init=False)

    def __post_init__(self) -> None:
        assert (
            self.positions.ndim == 2 and self.positions.shape[1] == 2
        ), f"positions must have shape (4*N, 2), got {self.positions.shape}"
        assert (
            self.positions.shape[0] % 4 == 0
        ), "positions.shape[0] must be a multiple of 4 (N groups of 4 neighbors)"

        left, right, top, bottom = np.split(self.positions, 4)
        self.initial_delta_top_bottom = top - bottom
        self.initial_delta_right_left = right - left

        n_particles = self.positions.shape[0] // 4
        self.initial_centroid = np.mean(
            self.positions.reshape(n_particles, 4, 2), axis=1
        )

    def __len__(self) -> int:
        """Returns the number of particle groups (N)."""
        return self.positions.shape[0] // 4

    @property
    def delta_top_bottom(self) -> ArrayFloat32Nx2:
        """Compute the vector difference between the top and bottom neighbors."""
        n_particles = self.positions.shape[0] // 4  # n_particles = N
        return (
            self.positions[2 * n_particles : 3 * n_particles, :]
            - self.positions[3 * n_particles :, :]
        )

    @property
    def delta_right_left(self) -> ArrayFloat32Nx2:
        """Compute the vector difference between the right and left neighbors."""
        n_particles = self.positions.shape[0] // 4  # n_particles = N
        return (
            self.positions[n_particles : 2 * n_particles, :]
            - self.positions[:n_particles, :]
        )

    @property
    def centroid(self) -> ArrayFloat32Nx2:
        """Compute the centroid of the four neighboring positions."""
        n_particles = self.positions.shape[0] // 4  # n_particles = N
        return np.mean(self.positions.reshape(n_particles, 4, 2), axis=1)

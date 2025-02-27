from dataclasses import dataclass, field

import numpy as np
from numba import njit  # type: ignore

from src.my_types import ArrayFloat64N4x2, ArrayFloat64Nx2


@dataclass
class NeighboringParticles:
    positions: ArrayFloat64N4x2  # Shape (4*N, 2)

    initial_delta_top_bottom: ArrayFloat64Nx2 = field(init=False)
    initial_delta_right_left: ArrayFloat64Nx2 = field(init=False)
    initial_centroid: ArrayFloat64Nx2 = field(init=False)

    def __post_init__(self) -> None:
        assert (
            self.positions.shape[0] % 4 == 0
        ), "positions.shape[0] must be multiple of 4"

        n_particles = self.positions.shape[0] // 4
        self.initial_delta_top_bottom = compute_initial_delta_top_bottom(
            self.positions, n_particles
        )
        self.initial_delta_right_left = compute_initial_delta_right_left(
            self.positions, n_particles
        )
        self.initial_centroid = compute_centroid(self.positions, n_particles)

    def __len__(self) -> int:
        return self.positions.shape[0] // 4

    @property
    def delta_top_bottom(self) -> ArrayFloat64Nx2:
        """Compute the vector difference between the top and bottom neighbors."""
        return compute_delta_top_bottom(self.positions, len(self))

    @property
    def delta_right_left(self) -> ArrayFloat64Nx2:
        """Compute the vector difference between the right and left neighbors."""
        return compute_delta_right_left(self.positions, len(self))

    @property
    def centroid(self) -> ArrayFloat64Nx2:
        """Compute the centroid of the four neighboring positions."""
        return compute_centroid(self.positions, len(self))


@njit
def compute_initial_delta_top_bottom(
    positions: ArrayFloat64N4x2, n_particles: int
) -> ArrayFloat64Nx2:
    """Compute the initial difference between top and bottom neighbors."""
    top = positions[2 * n_particles : 3 * n_particles]
    bottom = positions[3 * n_particles :]
    return top - bottom


@njit
def compute_initial_delta_right_left(
    positions: ArrayFloat64N4x2, n_particles: int
) -> ArrayFloat64Nx2:
    """Compute the initial difference between right and left neighbors."""
    right = positions[n_particles : 2 * n_particles]
    left = positions[:n_particles]
    return right - left


@njit
def compute_delta_top_bottom(
    positions: ArrayFloat64N4x2, n_particles: int
) -> ArrayFloat64Nx2:
    """Compute the vector difference between the top and bottom neighbors."""
    return positions[2 * n_particles : 3 * n_particles] - positions[3 * n_particles :]


@njit
def compute_delta_right_left(
    positions: ArrayFloat64N4x2, n_particles: int
) -> ArrayFloat64Nx2:
    """Compute the vector difference between the right and left neighbors."""
    return positions[n_particles : 2 * n_particles] - positions[:n_particles]


@njit
def compute_centroid(positions: ArrayFloat64N4x2, n_particles: int) -> ArrayFloat64Nx2:
    """Compute the centroid of the four neighboring positions."""
    centroid = np.zeros((n_particles, 2), dtype=np.float64)
    for i in range(n_particles):
        for j in range(2):
            centroid[i, j] = (
                positions[i, j]
                + positions[n_particles + i, j]
                + positions[2 * n_particles + i, j]
                + positions[3 * n_particles + i, j]
            ) / 4.0
    return centroid

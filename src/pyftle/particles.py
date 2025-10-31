from dataclasses import dataclass, field

import numpy as np

from pyftle.my_types import (
    ArrayFloat64N4x2,
    ArrayFloat64N6x3,
    ArrayFloat64Nx2,
    ArrayFloat64Nx3,
)


@dataclass
class NeighboringParticles:
    positions: ArrayFloat64N4x2 | ArrayFloat64N6x3  # Shape (4*N, 2) or (6*N, 3)

    initial_delta_top_bottom: ArrayFloat64Nx2 | ArrayFloat64Nx3 = field(init=False)
    initial_delta_right_left: ArrayFloat64Nx2 | ArrayFloat64Nx3 = field(init=False)
    initial_delta_front_back: ArrayFloat64Nx2 | ArrayFloat64Nx3 = field(init=False)
    initial_centroid: ArrayFloat64Nx2 | ArrayFloat64Nx3 = field(init=False)
    num_neighbors: int = field(init=False)  # (4 if 2D 6 if 3D)

    def __post_init__(self) -> None:
        assert self.positions.shape[0] % 4 == 0 or self.positions.shape[0] % 6 == 0, (
            "positions.shape[0] must be multiple of 4 or 6"
        )
        self.num_neighbors = self.positions.shape[1] * 2

        self.initial_delta_top_bottom = compute_delta_top_bottom(
            self.positions, self.num_neighbors
        )
        self.initial_delta_right_left = compute_delta_right_left(
            self.positions, self.num_neighbors
        )
        if (
            self.positions.shape[1] == 3
        ):  # handle 3D case and compute front and back property
            self.initial_delta_front_back = compute_delta_front_back(
                self.positions, self.num_neighbors
            )
        else:
            self.initial_delta_front_back = np.zeros(0)  # no-ops

        self.initial_centroid = compute_centroid(self.positions, self.num_neighbors)

    def __len__(self) -> int:
        return self.positions.shape[0] // self.num_neighbors

    @property
    def delta_top_bottom(self) -> ArrayFloat64Nx2 | ArrayFloat64Nx3:
        """Compute the vector difference between the top and bottom neighbors."""
        return compute_delta_top_bottom(self.positions, self.num_neighbors)

    @property
    def delta_right_left(self) -> ArrayFloat64Nx2 | ArrayFloat64Nx3:
        """Compute the vector difference between the right and left neighbors."""
        return compute_delta_right_left(self.positions, self.num_neighbors)

    @property
    def delta_front_back(self) -> ArrayFloat64Nx2 | ArrayFloat64Nx3:
        """Compute the vector difference between the right and left neighbors."""
        return compute_delta_front_back(self.positions, self.num_neighbors)

    @property
    def centroid(self) -> ArrayFloat64Nx2 | ArrayFloat64Nx3:
        """Compute the centroid of the four neighboring positions."""
        return compute_centroid(self.positions, self.num_neighbors)


def compute_delta_right_left(positions, num_neighbors):
    left, right, *_ = np.split(positions, num_neighbors, axis=0)
    return right - left


def compute_delta_top_bottom(positions, num_neighbors):
    _, _, top, bottom, *_ = np.split(positions, num_neighbors, axis=0)
    return top - bottom


def compute_delta_front_back(positions, num_neighbors):
    *_, front, back = np.split(positions, num_neighbors, axis=0)
    return front - back


def compute_centroid(
    positions: ArrayFloat64N4x2 | ArrayFloat64N6x3, num_neighbors: int
) -> ArrayFloat64Nx2 | ArrayFloat64Nx3:
    """Compute the centroid of the four neighboring positions."""
    parts = np.split(positions, num_neighbors, axis=0)
    centroid = np.mean(parts, axis=0)  # vectorized mean over neighbor axis
    return centroid

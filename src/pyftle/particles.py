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

        n_particles = self.positions.shape[0] // self.num_neighbors
        self.initial_delta_top_bottom = compute_initial_delta_top_bottom(
            self.positions, self.num_neighbors
        )
        self.initial_delta_right_left = compute_initial_delta_right_left(
            self.positions, self.num_neighbors
        )
        if (
            self.positions.shape[1] == 3
        ):  # handle 3D case and compute front and back property
            self.initial_delta_front_back = compute_initial_delta_front_back(
                self.positions, self.num_neighbors
            )
        else:
            self.initial_delta_front_back = np.zeros(0)  # no-ops

        self.initial_centroid = compute_centroid(
            self.positions, n_particles, self.num_neighbors
        )

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
        return compute_centroid(self.positions, len(self), self.num_neighbors)


def compute_initial_delta_right_left(
    positions: ArrayFloat64N4x2 | ArrayFloat64N6x3, dimension: int
) -> ArrayFloat64Nx2 | ArrayFloat64Nx3:
    """Compute the initial difference between right and left neighbors."""
    split_array = np.split(positions, dimension, axis=0)
    right = split_array[1]
    left = split_array[0]
    return right - left


def compute_initial_delta_top_bottom(
    positions: ArrayFloat64N4x2 | ArrayFloat64N6x3, dimension: int
) -> ArrayFloat64Nx2:
    """Compute the initial difference between top and bottom neighbors."""
    split_array = np.split(positions, dimension, axis=0)
    top = split_array[2]
    bottom = split_array[3]
    return top - bottom


def compute_initial_delta_front_back(
    positions: ArrayFloat64N4x2 | ArrayFloat64Nx3, dimension: int
) -> ArrayFloat64Nx2:
    """Compute the initial difference between right and left neighbors."""
    split_array = np.split(positions, dimension, axis=0)
    front = split_array[4]
    back = split_array[5]
    return front - back


def compute_delta_right_left(
    positions: ArrayFloat64N4x2 | ArrayFloat64N6x3, dimension: int
) -> ArrayFloat64Nx2:
    """Compute the vector difference between the right and left neighbors."""
    split_array = np.split(positions, dimension, axis=0)
    right = split_array[1]
    left = split_array[0]
    return right - left


def compute_delta_top_bottom(
    positions: ArrayFloat64N4x2 | ArrayFloat64N6x3, dimension: int
) -> ArrayFloat64Nx2:
    """Compute the vector difference between the top and bottom neighbors."""
    split_array = np.split(positions, dimension, axis=0)
    top = split_array[2]
    bottom = split_array[3]
    return top - bottom


def compute_delta_front_back(
    positions: ArrayFloat64N4x2 | ArrayFloat64Nx3, dimension: int
) -> ArrayFloat64Nx2:
    """Compute the vector difference between the right and left neighbors."""
    split_array = np.split(positions, dimension, axis=0)
    front = split_array[4]
    back = split_array[5]

    return front - back


def compute_centroid(
    positions: ArrayFloat64N4x2 | ArrayFloat64N6x3, n_particles: int, dimension: int
) -> ArrayFloat64Nx2 | ArrayFloat64Nx3:
    """Compute the centroid of the four neighboring positions."""
    centroid = np.zeros((n_particles, int(dimension / 2)), dtype=np.float64)
    if dimension == 4:
        for i in range(n_particles):
            for j in range(2):
                centroid[i, j] = (
                    positions[i, j]
                    + positions[n_particles + i, j]
                    + positions[2 * n_particles + i, j]
                    + positions[3 * n_particles + i, j]
                ) / 4.0
    else:
        left, right, top, bottom, front, back = np.split(positions, dimension, axis=0)

        centroid[:, 0] = (
            left[:, 0]
            + right[:, 0]
            + top[:, 0]
            + bottom[:, 0]
            + front[:, 0]
            + back[:, 0]
        ) / 6  # x_centroid
        centroid[:, 1] = (
            left[:, 1]
            + right[:, 1]
            + top[:, 1]
            + bottom[:, 1]
            + front[:, 1]
            + back[:, 1]
        ) / 6  # y_centroid
        centroid[:, 2] = (
            left[:, 2]
            + right[:, 2]
            + top[:, 2]
            + bottom[:, 2]
            + front[:, 2]
            + back[:, 2]
        ) / 6  # z_centroid

    return centroid

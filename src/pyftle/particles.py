from dataclasses import dataclass, field

import numpy as np

from pyftle.my_types import Array4Nx2, Array6Nx3, ArrayNx2, ArrayNx3


@dataclass
class NeighboringParticles:
    # Ruff:
    r"""
    Represents groups of neighboring particles used to approximate local
    flow gradients (Jacobian / deformation gradient tensor).

    Each group contains one seed particle and its surrounding neighbors:
    four neighbors in 2D (right, left, top, bottom) or six neighbors in 3D
    (right, left, top, bottom, front, back). This configuration enables the
    computation of directional displacement differences and flow map
    derivatives for FTLE/LCS analysis.

    Parameters
    ----------
    positions : Array4Nx2 | Array6Nx3
        Particle positions at the current time step, arranged as consecutive
        groups of 4 (for 2D) or 6 (for 3D). The ordering within each group
        must be:
            - 2D: [right, left, top, bottom]
            - 3D: [right, left, top, bottom, front, back]

    Attributes
    ----------
    initial_delta_top_bottom : ArrayNx2 | ArrayNx3
        Vector difference between top and bottom neighbors at initialization,
        computed as :math:`\Delta \mathbf{X}_{TB} = \mathbf{X}_T - \mathbf{X}_B`.
    initial_delta_right_left : ArrayNx2 | ArrayNx3
        Vector difference between right and left neighbors at initialization,
        :math:`\Delta \mathbf{X}_{RL} = \mathbf{X}_R - \mathbf{X}_L`.
    initial_delta_front_back : ArrayNx2 | ArrayNx3
        Vector difference between front and back neighbors at initialization,
        :math:`\Delta \mathbf{X}_{FB} = \mathbf{X}_F - \mathbf{X}_B`. Empty in 2D.
    initial_centroid : ArrayNx2 | ArrayNx3
        Mean position of each group of neighboring particles.
    num_neighbors : int
        Number of neighbors per group (4 for 2D, 6 for 3D).

    Notes
    -----
    The class precomputes all initial (reference) geometry in `__post_init__`.
    Properties such as `delta_*` provide current (deformed) displacements:
    e.g., :math:`\Delta \mathbf{x}_{RL} = \mathbf{x}_R - \mathbf{x}_L`.

    Examples
    --------
    >>> positions = np.random.rand(8, 2)  # 2 groups of 4 neighbors in 2D
    >>> p = NeighboringParticles(positions)
    >>> len(p)
    2
    >>> p.delta_right_left.shape
    (2, 2)
    >>> p.centroid.shape
    (2, 2)
    """

    positions: Array4Nx2 | Array6Nx3  # Shape (4*N, 2) or (6*N, 3)

    initial_delta_top_bottom: ArrayNx2 | ArrayNx3 = field(init=False)
    initial_delta_right_left: ArrayNx2 | ArrayNx3 = field(init=False)
    initial_delta_front_back: ArrayNx2 | ArrayNx3 = field(init=False)
    initial_centroid: ArrayNx2 | ArrayNx3 = field(init=False)
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
        """Return the number of particle groups (N)."""
        return self.positions.shape[0] // self.num_neighbors

    @property
    def delta_top_bottom(self) -> ArrayNx2 | ArrayNx3:
        r"""
        Compute the current vector difference between top and bottom neighbors.

        Returns
        -------
        ArrayNx2 | ArrayNx3
            The top-bottom vector difference for each group,
            :math:`\Delta \mathbf{x}_{TB} = \mathbf{x}_T - \mathbf{x}_B`.
        """
        return compute_delta_top_bottom(self.positions, self.num_neighbors)

    @property
    def delta_right_left(self) -> ArrayNx2 | ArrayNx3:
        r"""
        Compute the current vector difference between right and left neighbors.

        Returns
        -------
        ArrayNx2 | ArrayNx3
            The right-left vector difference for each group,
            :math:`\Delta \mathbf{x}_{RL} = \mathbf{x}_R - \mathbf{x}_L`.
        """
        return compute_delta_right_left(self.positions, self.num_neighbors)

    @property
    def delta_front_back(self) -> ArrayNx2 | ArrayNx3:
        r"""
        Compute the current vector difference between front and back neighbors.

        Returns
        -------
        ArrayNx2 | ArrayNx3
            The front-back vector difference for each group,
            :math:`\Delta \mathbf{x}_{FB} = \mathbf{x}_F - \mathbf{x}_B`.
            Returns an empty array in 2D cases.
        """
        return compute_delta_front_back(self.positions, self.num_neighbors)

    @property
    def centroid(self) -> ArrayNx2 | ArrayNx3:
        r"""
        Compute the centroid (mean position) of each group of neighbors.

        Returns
        -------
        ArrayNx2 | ArrayNx3
            The centroid coordinates of each group,
            :math:`\bar{\mathbf{x}} = \frac{1}{n}\sum_i \mathbf{x}_i`.
        """
        return compute_centroid(self.positions, self.num_neighbors)


def compute_delta_right_left(positions, num_neighbors):
    r"""
    Compute the vector difference between right and left neighbors.

    Parameters
    ----------
    positions : Array4Nx2 | Array6Nx3
        Particle positions grouped in sets of `num_neighbors`.
    num_neighbors : int
        Number of neighbors per group (4 in 2D, 6 in 3D).

    Returns
    -------
    ArrayNx2 | ArrayNx3
        The vector difference :math:`\Delta \mathbf{x}_{RL} = \mathbf{x}_R -
        \mathbf{x}_L`.
    """
    left, right, *_ = np.split(positions, num_neighbors, axis=0)
    return right - left


def compute_delta_top_bottom(positions, num_neighbors):
    r"""
    Compute the vector difference between top and bottom neighbors.

    Parameters
    ----------
    positions : Array4Nx2 | Array6Nx3
        Particle positions grouped in sets of `num_neighbors`.
    num_neighbors : int
        Number of neighbors per group (4 in 2D, 6 in 3D).

    Returns
    -------
    ArrayNx2 | ArrayNx3
        The vector difference :math:`\Delta \mathbf{x}_{TB} = \mathbf{x}_T -
        \mathbf{x}_B`.
    """
    _, _, top, bottom, *_ = np.split(positions, num_neighbors, axis=0)
    return top - bottom


def compute_delta_front_back(positions, num_neighbors):
    r"""
    Compute the vector difference between front and back neighbors (3D only).

    Parameters
    ----------
    positions : Array6Nx3
        Particle positions grouped in sets of 6 neighbors.
    num_neighbors : int
        Number of neighbors (must be 6 in 3D).

    Returns
    -------
    ArrayNx3
        The vector difference :math:`\Delta \mathbf{x}_{FB} = \mathbf{x}_F -
        \mathbf{x}_B`. Returns an empty array in 2D cases.
    """
    *_, front, back = np.split(positions, num_neighbors, axis=0)
    return front - back


def compute_centroid(
    positions: Array4Nx2 | Array6Nx3, num_neighbors: int
) -> ArrayNx2 | ArrayNx3:
    r"""
    Compute the centroid (mean position) of each neighbor group.

    Parameters
    ----------
    positions : Array4Nx2 | Array6Nx3
        Particle positions grouped in sets of `num_neighbors`.
    num_neighbors : int
        Number of neighbors per group (4 in 2D, 6 in 3D).

    Returns
    -------
    ArrayNx2 | ArrayNx3
        Centroid coordinates,
        :math:`\bar{\mathbf{x}} = \frac{1}{n}\sum_i \mathbf{x}_i`.
    """
    parts = np.split(positions, num_neighbors, axis=0)
    centroid = np.mean(parts, axis=0)  # vectorized mean over neighbor axis
    return centroid

from typing import Protocol

import numpy as np
from scipy.interpolate import (
    CloughTocher2DInterpolator,
    LinearNDInterpolator,
    NearestNDInterpolator,
    RegularGridInterpolator,
)

from src.caching import cache_last_n_files
from src.file_readers import read_coordinates, read_velocity_data
from src.my_types import ArrayFloat32N, ArrayFloat32Nx2


class InterpolationStrategy(Protocol):
    def interpolate(
        self,
        new_points: ArrayFloat32Nx2,
    ) -> ArrayFloat32Nx2:
        """Implements the interpolation strategy."""
        ...


class CubicInterpolatorStrategy:
    """Piecewise cubic, C1 smooth, curvature-minimizing interpolator in 2D
    for the velocity field using Clough-Tocher interpolation.

    Pros:
    - Produces smooth, high-quality interpolation.
    - Suitable for smoothly varying velocity fields.

    Cons:
    - Computationally expensive due to Delaunay triangulation.
    - Slower than simpler interpolation methods.

    Parameters
    ----------
    points : NDArray
        Array of shape `(n_points, 2)` representing the coordinates.
    velocities_u : NDArray
        Array of shape `(n_points,)` representing the u-velocity values.
    velocities_v : NDArray
        Array of shape `(n_points,)` representing the v-velocity values.
    """

    def __init__(
        self,
        points: ArrayFloat32Nx2,
        velocities_u: ArrayFloat32N,
        velocities_v: ArrayFloat32N,
    ):
        self.interpolator_u = CloughTocher2DInterpolator(points, velocities_u)
        self.interpolator_v = CloughTocher2DInterpolator(points, velocities_v)

    def interpolate(self, new_points: ArrayFloat32Nx2) -> ArrayFloat32Nx2:
        u_interpolated = self.interpolator_u(new_points)
        v_interpolated = self.interpolator_v(new_points)
        return np.column_stack((u_interpolated, v_interpolated))


class LinearInterpolatorStrategy:
    """Piecewise linear interpolator using Delaunay triangulation.

    Pros:
    - Faster than Clough-Tocher.
    - Still provides reasonably smooth interpolation.

    Cons:
    - Not as smooth as cubic interpolation.
    - May introduce discontinuities in derivatives.
    """

    def __init__(
        self,
        points: ArrayFloat32Nx2,
        velocities_u: ArrayFloat32N,
        velocities_v: ArrayFloat32N,
    ):
        self.interpolator_u = LinearNDInterpolator(points, velocities_u)
        self.interpolator_v = LinearNDInterpolator(points, velocities_v)

    def interpolate(self, new_points: ArrayFloat32Nx2) -> ArrayFloat32Nx2:
        u_interpolated = self.interpolator_u(new_points)
        v_interpolated = self.interpolator_v(new_points)
        return np.column_stack((u_interpolated, v_interpolated))


class NearestNeighborInterpolatorStrategy:
    """Nearest neighbor interpolation, assigning the value of the closest known point.

    Pros:
    - Very fast and computationally cheap.
    - No triangulation required.

    Cons:
    - Produces a blocky, discontinuous field.
    - Not suitable for smoothly varying velocity fields.
    """

    def __init__(
        self,
        points: ArrayFloat32Nx2,
        velocities_u: ArrayFloat32N,
        velocities_v: ArrayFloat32N,
    ):
        self.interpolator_u = NearestNDInterpolator(points, velocities_u)
        self.interpolator_v = NearestNDInterpolator(points, velocities_v)

    def interpolate(self, new_points: ArrayFloat32Nx2) -> ArrayFloat32Nx2:
        u_interpolated = self.interpolator_u(new_points)
        v_interpolated = self.interpolator_v(new_points)
        return np.column_stack((u_interpolated, v_interpolated))


class GridInterpolatorStrategy:
    """Grid-based interpolation using RegularGridInterpolator.

    Pros:
    - Extremely fast when data is structured on a regular grid.
    - Memory efficient compared to unstructured methods.

    Cons:
    - Only works with structured grids.
    - Requires careful handling of grid spacing and boundaries.
    """

    def __init__(self, grid_x, grid_y, velocities_u, velocities_v):
        self.interpolator_u = RegularGridInterpolator((grid_x, grid_y), velocities_u)
        self.interpolator_v = RegularGridInterpolator((grid_x, grid_y), velocities_v)

    def interpolate(self, new_points: ArrayFloat32Nx2) -> ArrayFloat32Nx2:
        u_interpolated = self.interpolator_u(new_points)
        v_interpolated = self.interpolator_v(new_points)
        return np.column_stack((u_interpolated, v_interpolated))


@cache_last_n_files(num_cached_files=2)
def create_interpolator(snapshot_file: str, grid_file: str, strategy: str = "cubic"):
    """
    Reads velocity and coordinate data from the given files and creates an interpolator
    based on the selected strategy.

    Supported strategies:
    - "cubic": Clough-Tocher interpolation (default, high-quality but slow).
    - "linear": Linear interpolation (faster, but less smooth).
    - "nearest": Nearest-neighbor interpolation (fastest, but lowest quality).
    - "grid": Grid-based interpolation (fastest for structured grids).

    Args:
        snapshot_file (str): Path to the velocity data file.
        grid_file (str): Path to the coordinate data file.
        strategy (str): Interpolation strategy to use ("cubic", "linear", "nearest",
        "grid").

    Returns:
        (InterpolationStrategy): The selected interpolator object.
    """
    velocities = read_velocity_data(snapshot_file)
    coordinates = read_coordinates(grid_file)

    if strategy == "cubic":
        return CubicInterpolatorStrategy(
            coordinates, velocities[:, 0], velocities[:, 1]
        )
    elif strategy == "linear":
        return LinearInterpolatorStrategy(
            coordinates, velocities[:, 0], velocities[:, 1]
        )
    elif strategy == "nearest":
        return NearestNeighborInterpolatorStrategy(
            coordinates, velocities[:, 0], velocities[:, 1]
        )
    elif strategy == "grid":
        grid_x, grid_y = np.unique(coordinates[:, 0]), np.unique(coordinates[:, 1])
        return GridInterpolatorStrategy(
            grid_x,
            grid_y,
            velocities[:, 0].reshape(len(grid_x), len(grid_y)),
            velocities[:, 1].reshape(len(grid_x), len(grid_y)),
        )
    else:
        raise ValueError(f"Unknown interpolation strategy: {strategy}")

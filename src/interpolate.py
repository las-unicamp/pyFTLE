from typing import Protocol

import numpy as np
from scipy.interpolate import (
    CloughTocher2DInterpolator,
    LinearNDInterpolator,
    NearestNDInterpolator,
    RegularGridInterpolator,
)

from src.caching import cache_last_n_files
from src.file_readers import CoordinateDataReader, VelocityDataReader
from src.metrics import Metrics
from src.my_types import (
    ArrayFloat32MxN,
    ArrayFloat32N,
    ArrayFloat32Nx2,
)


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
        velocities = velocities_u + 1j * velocities_v
        self.interpolator = CloughTocher2DInterpolator(points, velocities)

    def interpolate(self, new_points: ArrayFloat32Nx2) -> ArrayFloat32Nx2:
        interp_velocities = self.interpolator(new_points)
        return np.column_stack((interp_velocities.real, interp_velocities.imag))


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
        velocities = velocities_u + 1j * velocities_v
        self.interpolator = LinearNDInterpolator(points, velocities)

    def interpolate(self, new_points: ArrayFloat32Nx2) -> ArrayFloat32Nx2:
        interp_velocities = self.interpolator(new_points)
        return np.column_stack((interp_velocities.real, interp_velocities.imag))


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
        velocities = velocities_u + 1j * velocities_v
        self.interpolator = NearestNDInterpolator(points, velocities)

    def interpolate(self, new_points: ArrayFloat32Nx2) -> ArrayFloat32Nx2:
        interp_velocities = self.interpolator(new_points)
        return np.column_stack((interp_velocities.real, interp_velocities.imag))


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
        velocities = velocities_u + 1j * velocities_v
        self.interpolator = RegularGridInterpolator((grid_x, grid_y), velocities)

    def interpolate(self, new_points: ArrayFloat32Nx2) -> ArrayFloat32Nx2:
        interp_velocities = self.interpolator(new_points)
        return np.column_stack((interp_velocities.real, interp_velocities.imag))




class InterpolatorFactory:
    def __init__(
        self,
        coordinate_reader: CoordinateDataReader,
        velocity_reader: VelocityDataReader,
    ):
        self.coordinate_reader = coordinate_reader
        self.velocity_reader = velocity_reader

    @cache_last_n_files(num_cached_files=2)
    def create_interpolator(
        self, snapshot_file: str, grid_file: str, strategy: str = "cubic"
    ):
        """
        Reads velocity and coordinate data from the given files and creates an
        interpolator based on the selected strategy.

        Supported strategies:
        - "cubic": Clough-Tocher interpolation (default, high-quality but slow).
        - "linear": Linear interpolation (faster, but less smooth).
        - "nearest": Nearest-neighbor interpolation (fastest, but lowest quality).
        - "grid": Grid-based interpolation (fastest for structured grids).

        Args:
            snapshot_file (str): Path to the velocity data file.
            grid_file (str): Path to the coordinate data file.
            strategy (str): Interpolation strategy to use ("cubic", "linear",
            "nearest", "grid").

        Returns:
            (InterpolationStrategy): The selected interpolator object.
        """
        flatten = strategy != "grid"

        # Choose the appropriate method dynamically
        read_velocity = getattr(
            self.velocity_reader, "read_flatten" if flatten else "read_raw"
        )
        read_coordinates = getattr(
            self.coordinate_reader, "read_flatten" if flatten else "read_raw"
        )

        velocities = read_velocity(snapshot_file)
        coordinates = read_coordinates(grid_file)

        match strategy:
            case "cubic":
                return CubicInterpolatorStrategy(
                    coordinates, velocities[:, 0], velocities[:, 1]
                )
            case "linear":
                return LinearInterpolatorStrategy(
                    coordinates, velocities[:, 0], velocities[:, 1]
                )
            case "nearest":
                return NearestNeighborInterpolatorStrategy(
                    coordinates, velocities[:, 0], velocities[:, 1]
                )
            case "grid":
                return GridInterpolatorStrategy(
                    coordinates, velocities[0], velocities[1]
                )
            case _:
                raise ValueError(f"Unknown interpolation strategy: {strategy}")

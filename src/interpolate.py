# ruff: noqa: N806
from typing import Protocol

import numpy as np
from scipy.interpolate import (
    CloughTocher2DInterpolator,
    LinearNDInterpolator,
    NearestNDInterpolator,
    RegularGridInterpolator,
)

from src.file_readers import CoordinateReader, VelocityReader
from src.my_types import (
    Array2xMxN,
    Array3xMxN,
    ArrayFloat64Nx2,
    ArrayFloat64Nx3,
)


class InterpolationStrategy(Protocol):
    def interpolate(
        self,
        new_points: ArrayFloat64Nx2 | ArrayFloat64Nx3,
    ) -> ArrayFloat64Nx2 | ArrayFloat64Nx3:
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
        Array of shape `(n_points, 2)` representing the velocities values.
    """

    def __init__(
        self,
        points: ArrayFloat64Nx2 | ArrayFloat64Nx3,
        velocities: ArrayFloat64Nx2 | ArrayFloat64Nx3,
    ):
        velocities_cmplx = velocities[:, 0] + 1j * velocities[:, 1]
        self.interpolator = CloughTocher2DInterpolator(points, velocities_cmplx)
        if points.shape[1] == 3:
            self.interpolator_z = CloughTocher2DInterpolator(points, velocities[:, 2])

    def interpolate(self, new_points: ArrayFloat64Nx2) -> ArrayFloat64Nx2:
        interp_velocities = self.interpolator(new_points)
        if new_points.shape[1] == 2:
            return np.column_stack((interp_velocities.real, interp_velocities.imag))
        else:
            interp_velocities_z = self.interpolator_z(new_points)
            return np.column_stack(
                (interp_velocities.real, interp_velocities.imag, interp_velocities_z)
            )


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
        points: ArrayFloat64Nx2 | ArrayFloat64Nx3,
        velocities: ArrayFloat64Nx2 | ArrayFloat64Nx3,
    ):
        velocities_cmplx = velocities[:, 0] + 1j * velocities[:, 1]
        self.interpolator = LinearNDInterpolator(points, velocities_cmplx)

        if points.shape[1] == 3:
            self.interpolator_z = LinearNDInterpolator(
                points, velocities[:, 2], fill_value=0.0
            )

    def interpolate(
        self, new_points: ArrayFloat64Nx2 | ArrayFloat64Nx3
    ) -> ArrayFloat64Nx2 | ArrayFloat64Nx3:
        interp_velocities = self.interpolator(new_points)

        if new_points.shape[1] == 2:
            return np.column_stack((interp_velocities.real, interp_velocities.imag))
        else:
            interp_velocities_z = self.interpolator_z(new_points)
            return np.column_stack(
                (interp_velocities.real, interp_velocities.imag, interp_velocities_z)
            )


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
        points: ArrayFloat64Nx2 | ArrayFloat64Nx3,
        velocities: ArrayFloat64Nx2 | ArrayFloat64Nx3,
    ):
        velocities_cmplx = velocities[:, 0] + 1j * velocities[:, 1]
        self.interpolator = NearestNDInterpolator(points, velocities_cmplx)
        if points.shape[1] == 3:
            self.interpolator_z = NearestNDInterpolator(points, velocities[:, 2])

    def interpolate(
        self, new_points: ArrayFloat64Nx2 | ArrayFloat64Nx3
    ) -> ArrayFloat64Nx2 | ArrayFloat64Nx3:
        interp_velocities = self.interpolator(new_points)
        if new_points.shape[1] == 2:
            return np.column_stack((interp_velocities.real, interp_velocities.imag))
        else:
            interp_velocities_z = self.interpolator_z(new_points)
            return np.column_stack(
                (interp_velocities.real, interp_velocities.imag, interp_velocities_z)
            )


class GridInterpolatorStrategy:
    """Grid-based interpolation using RegularGridInterpolator.

    Pros:
    - Extremely fast when data is structured on a regular grid.
    - Memory efficient compared to unstructured methods.

    Cons:
    - Only works with structured grids.
    - Requires careful handling of grid spacing and boundaries.
    """

    def __init__(
        self,
        coordinates: Array3xMxN | Array2xMxN,
        velocities: Array3xMxN | Array2xMxN,
    ):
        grid_shape = coordinates[0].shape

        grid_x = np.linspace(
            np.min(coordinates[0]),
            np.max(coordinates[0]),
            grid_shape[0],
        )
        grid_y = np.linspace(
            np.min(coordinates[1]),
            np.max(coordinates[1]),
            grid_shape[1],
        )

        if len(grid_shape) == 2:
            self.interpolator_u = RegularGridInterpolator(
                (grid_x, grid_y), velocities[0], bounds_error=False, fill_value=None
            )
            self.interpolator_v = RegularGridInterpolator(
                (grid_x, grid_y), velocities[1], bounds_error=False, fill_value=None
            )
        else:
            grid_z = np.linspace(
                np.min(coordinates[2]),
                np.max(coordinates[2]),
                grid_shape[2],
            )

            self.interpolator_u = RegularGridInterpolator(
                (grid_x, grid_y, grid_z),
                velocities[0],
                bounds_error=False,
                fill_value=0.0,
            )
            self.interpolator_v = RegularGridInterpolator(
                (grid_x, grid_y, grid_z),
                velocities[1],
                bounds_error=False,
                fill_value=0.0,
            )
            self.interpolator_z = RegularGridInterpolator(
                (grid_x, grid_y, grid_z),
                velocities[2],
                bounds_error=False,
                fill_value=0.0,
            )

    def interpolate(
        self, new_points: ArrayFloat64Nx2 | ArrayFloat64Nx3
    ) -> ArrayFloat64Nx2 | ArrayFloat64Nx3:
        """Interpolates velocity field at given Cartesian points."""
        u_interp = self.interpolator_u(new_points)
        v_interp = self.interpolator_v(new_points)
        if new_points.shape[1] == 3:
            w_interp = self.interpolator_z(new_points)
            return np.column_stack((u_interp, v_interp, w_interp))
        else:
            return np.column_stack((u_interp, v_interp))


class InterpolatorFactory:
    def __init__(
        self,
        coordinate_reader: CoordinateReader,
        velocity_reader: VelocityReader,
    ):
        self.coordinate_reader = coordinate_reader
        self.velocity_reader = velocity_reader

    def create_interpolator(
        self, snapshot_file: str, grid_file: str, strategy: str = "cubic"
    ) -> InterpolationStrategy:
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
                    coordinates,
                    velocities,
                )

            case "linear":
                return LinearInterpolatorStrategy(
                    coordinates,
                    velocities,
                )

            case "nearest":
                return NearestNeighborInterpolatorStrategy(
                    coordinates,
                    velocities,
                )

            case "grid":
                return GridInterpolatorStrategy(
                    coordinates,
                    velocities,
                )
            case _:
                raise ValueError(f"Unknown interpolation strategy: {strategy}")

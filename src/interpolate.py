# ruff: noqa: N806
from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np
from matplotlib import ExecutableNotFoundError
from scipy.interpolate import (
    CloughTocher2DInterpolator,
    LinearNDInterpolator,
    NearestNDInterpolator,
    RegularGridInterpolator,
)

from src.my_types import (
    ArrayFloat64Nx2,
    ArrayFloat64Nx3,
)


class Interpolator(ABC):
    def __init__(
        self,
    ):
        """Lazy initialization: need to call update() once the velocities and
        points are available
        """

        self.velocities: Optional[ArrayFloat64Nx2 | ArrayFloat64Nx3] = None
        self.points: Optional[ArrayFloat64Nx2 | ArrayFloat64Nx3] = None
        self.interpolator = None  # Placeholder for the actual interpolator instance
        self.velocity_fn: Optional[Callable] = None  # Used only by AnalyticalInterp

    def _initialize_interpolator(self) -> None:
        """Initialize the actual interpolator object based on velocity and points."""

        if self.velocities is None or self.points is None:
            raise ValueError("Velocities and points must be set before initialization.")

        # Ensure to check that the velocities and points are properly shaped
        if self.velocities.shape[0] != self.points.shape[0]:
            raise ValueError("Number of velocities must match the number of points.")

        raise NotImplementedError("This method should be implemented by subclasses")

    def update(
        self,
        velocities: ArrayFloat64Nx2 | ArrayFloat64Nx3,
        points: Optional[ArrayFloat64Nx2 | ArrayFloat64Nx3] = None,
    ) -> None:
        """Updates the interpolation function for a new field."""

        self.velocities = velocities
        if points is not None:
            self.points = points

        self._initialize_interpolator()

    @abstractmethod
    def interpolate(
        self,
        new_points: ArrayFloat64Nx2 | ArrayFloat64Nx3,
    ) -> ArrayFloat64Nx2 | ArrayFloat64Nx3:
        """Implements the interpolation strategy."""
        pass


class CubicInterpolator(Interpolator):
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

    def _initialize_interpolator(self) -> None:
        if self.velocities is None or self.points is None:
            raise ValueError("Velocities and points must be set before initialization.")

        velocities_cmplx = self.velocities[:, 0] + 1j * self.velocities[:, 1]
        self.interpolator = CloughTocher2DInterpolator(self.points, velocities_cmplx)
        if self.points.shape[1] == 3:
            self.interpolator_z = CloughTocher2DInterpolator(
                self.points, self.velocities[:, 2]
            )

    def interpolate(self, new_points: ArrayFloat64Nx2) -> ArrayFloat64Nx2:
        interp_velocities = self.interpolator(new_points)
        if new_points.shape[1] == 2:
            return np.column_stack((interp_velocities.real, interp_velocities.imag))
        else:
            interp_velocities_z = self.interpolator_z(new_points)
            return np.column_stack(
                (interp_velocities.real, interp_velocities.imag, interp_velocities_z)
            )


class LinearInterpolator(Interpolator):
    """Piecewise linear interpolator using Delaunay triangulation.

    Pros:
    - Faster than Clough-Tocher.
    - Still provides reasonably smooth interpolation.

    Cons:
    - Not as smooth as cubic interpolation.
    - May introduce discontinuities in derivatives.
    """

    def _initialize_interpolator(self) -> None:
        if self.velocities is None or self.points is None:
            raise ValueError("Velocities and points must be set before initialization.")

        velocities_cmplx = self.velocities[:, 0] + 1j * self.velocities[:, 1]
        self.interpolator = LinearNDInterpolator(self.points, velocities_cmplx)

        if self.points.shape[1] == 3:
            self.interpolator_z = LinearNDInterpolator(
                self.points, self.velocities[:, 2], fill_value=0.0
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


class NearestNeighborInterpolator(Interpolator):
    """Nearest neighbor interpolation, assigning the value of the closest known point.

    Pros:
    - Very fast and computationally cheap.
    - No triangulation required.

    Cons:
    - Produces a blocky, discontinuous field.
    - Not suitable for smoothly varying velocity fields.
    """

    def _initialize_interpolator(self) -> None:
        if self.velocities is None or self.points is None:
            raise ValueError("Velocities and points must be set before initialization.")

        velocities_cmplx = self.velocities[:, 0] + 1j * self.velocities[:, 1]
        self.interpolator = NearestNDInterpolator(self.points, velocities_cmplx)
        if self.points.shape[1] == 3:
            self.interpolator_z = NearestNDInterpolator(
                self.points, self.velocities[:, 2]
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


class GridInterpolator(Interpolator):
    """Grid-based interpolation using RegularGridInterpolator.

    Pros:
    - Extremely fast when data is structured on a regular grid.
    - Memory efficient compared to unstructured methods.

    Cons:
    - Only works with structured grids.
    - Requires careful handling of grid spacing and boundaries.
    """

    def __init__(self):
        super().__init__()
        self.interpolator_u = None
        self.interpolator_v = None
        self.interpolator_z = None

    def _initialize_interpolator(self) -> None:
        """Initializes the actual interpolator for grid-based interpolation."""

        if self.velocities is None or self.points is None:
            raise ValueError("Velocities and points must be set before initialization.")

        # Shape of velocities and coordinates
        dimension, *grid_shape = self.velocities.shape

        grid_x = np.linspace(
            np.min(self.points[:, 0]), np.max(self.points[:, 0]), grid_shape[0]
        )
        grid_y = np.linspace(
            np.min(self.points[:, 1]), np.max(self.points[:, 1]), grid_shape[1]
        )

        if dimension == 2:  # 2D grid (nx, ny)
            self.interpolator_u = RegularGridInterpolator(
                (grid_x, grid_y),
                self.velocities[0],
                bounds_error=False,
                fill_value=None,
            )
            self.interpolator_v = RegularGridInterpolator(
                (grid_x, grid_y),
                self.velocities[1],
                bounds_error=False,
                fill_value=None,
            )
        else:  # 3D grid (nx, ny, nz)
            grid_z = np.linspace(
                np.min(self.points[2]),
                np.max(self.points[2]),
                grid_shape[2],
            )

            self.interpolator_u = RegularGridInterpolator(
                (grid_x, grid_y, grid_z),
                self.velocities[0],
                bounds_error=False,
                fill_value=0.0,
            )
            self.interpolator_v = RegularGridInterpolator(
                (grid_x, grid_y, grid_z),
                self.velocities[1],
                bounds_error=False,
                fill_value=0.0,
            )
            self.interpolator_z = RegularGridInterpolator(
                (grid_x, grid_y, grid_z),
                self.velocities[2],
                bounds_error=False,
                fill_value=0.0,
            )

    def interpolate(
        self, new_points: ArrayFloat64Nx2 | ArrayFloat64Nx3
    ) -> ArrayFloat64Nx2 | ArrayFloat64Nx3:
        """Interpolates velocity field at given Cartesian points."""
        if self.interpolator_u is None or self.interpolator_v is None:
            raise ValueError(
                "Interpolator has not been initialized. Call `update()` first."
            )

        u_interp = self.interpolator_u(new_points)
        v_interp = self.interpolator_v(new_points)

        if new_points.shape[1] == 3:
            if self.interpolator_z is None:
                raise ValueError("3D interpolator is not initialized properly.")

            w_interp = self.interpolator_z(new_points)
            return np.column_stack((u_interp, v_interp, w_interp))
        else:
            return np.column_stack((u_interp, v_interp))


class AnalyticalInterpolator(Interpolator):
    def __init__(
        self,
        time: float = 0.0,
    ):
        self.time = time
        self.velocity_fn: Optional[Callable] = None

    def interpolate(
        self, new_points: ArrayFloat64Nx2 | ArrayFloat64Nx3
    ) -> ArrayFloat64Nx2 | ArrayFloat64Nx3:
        """Evaluates the velocity field at the given coordinates."""

        if not callable(self.velocity_fn):
            raise ExecutableNotFoundError("velocity_fn was not assigned properly")
        return self.velocity_fn(self.time, new_points)

    def update(
        self,
        velocities: ArrayFloat64Nx2 | ArrayFloat64Nx3,
        points: Optional[ArrayFloat64Nx2 | ArrayFloat64Nx3] = None,
    ) -> None:
        """Override parent to do nothing (no state update necessary).
        Arguments are required by the interface, but they are not used here."""
        pass


def create_interpolator(
    interpolation_type: str, velocity_fn: Optional[Callable] = None
) -> Interpolator:
    """
    Factory function to return an interpolator constructor based on the type.

    Supported types of interpolators:
    - "cubic": Clough-Tocher interpolation (default, high-quality but slow).
    - "linear": Linear interpolation (faster, but less smooth).
    - "nearest": Nearest-neighbor interpolation (fastest, but lowest quality).
    - "grid": Grid-based interpolation (fastest for structured grids).

    Args:
        interpolation_type (str): Specifies which interpolator to use
            ('cubic', 'linear', 'nearest', or 'grid').

    Returns:
        An instance of the appropriate interpolator.
    """
    interpolation_type = interpolation_type.lower()  # Normalize input to lowercase

    interpolation_map: dict[str, type[Interpolator]] = {
        "cubic": CubicInterpolator,  # Uses Euler for the first step, then AB2
        "linear": LinearInterpolator,
        "nearest": NearestNeighborInterpolator,
        "grid": GridInterpolator,
        "analytical": AnalyticalInterpolator,
    }

    if interpolation_type not in interpolation_map:
        raise ValueError(
            f"Invalid interpolation type '{interpolation_type}'. "
            f"Choose from {list(interpolation_map.keys())}."
        )

    interpolator = interpolation_map[interpolation_type]()

    if interpolation_type == "analytical":
        interpolator.velocity_fn = velocity_fn

    return interpolator

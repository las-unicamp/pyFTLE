# ruff: noqa: N806
from abc import ABC, abstractmethod
from typing import Callable, Literal, Optional

import numpy as np
from matplotlib import ExecutableNotFoundError
from scipy.interpolate import (
    CloughTocher2DInterpolator,
    LinearNDInterpolator,
    NearestNDInterpolator,
    RegularGridInterpolator,
)
from scipy.spatial import Delaunay

from src.my_types import ArrayFloat64Nx2, ArrayFloat64Nx3


class Interpolator(ABC):
    def __init__(self):
        """Lazy initialization: need to call update() once the velocities and
        points are available
        """

        self.velocities: Optional[ArrayFloat64Nx2 | ArrayFloat64Nx3] = None
        self.points: Optional[ArrayFloat64Nx2 | ArrayFloat64Nx3] = None
        self.interpolator = None  # Placeholder for the actual interpolator instance
        self.velocity_fn: Optional[Callable] = None  # Used only by AnalyticalInterp
        self.grid_shape: Optional[tuple[int, ...]] = None  # Used only by GridInterp

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

    def __init__(self):
        self.tri: Optional[Delaunay] = None  # pre-computed Delaunay for faster updates

    def _initialize_interpolator(self) -> None:
        if self.velocities is None or self.points is None:
            raise ValueError("Velocities and points must be set before initialization.")

        if self.points.shape[1] == 3:
            raise ValueError("cubic interpolator is only valid for 2D cases")

        velocities_cmplx = self.velocities[:, 0] + 1j * self.velocities[:, 1]

        if self.tri is None:
            self.interpolator = CloughTocher2DInterpolator(
                self.points, velocities_cmplx
            )
            self.tri = self.interpolator.tri  # type: ignore[attr-defined]
        else:
            self.interpolator = CloughTocher2DInterpolator(self.tri, velocities_cmplx)

    def interpolate(self, new_points: ArrayFloat64Nx2) -> ArrayFloat64Nx2:
        interp_velocities = self.interpolator(new_points)
        return np.column_stack((interp_velocities.real, interp_velocities.imag))


class LinearInterpolator(Interpolator):
    """Piecewise linear interpolator using Delaunay triangulation.

    Pros:
    - Faster than Clough-Tocher.
    - Still provides reasonably smooth interpolation.

    Cons:
    - Not as smooth as cubic interpolation.
    - May introduce discontinuities in derivatives.
    """

    def __init__(self):
        self.tri: Optional[Delaunay] = None  # pre-computed Delaunay for faster updates

    def _initialize_interpolator(self) -> None:
        if self.velocities is None or self.points is None:
            raise ValueError("Velocities and points must be set before initialization.")

        velocities_cmplx = self.velocities[:, 0] + 1j * self.velocities[:, 1]

        if self.tri is None:
            self.interpolator = LinearNDInterpolator(self.points, velocities_cmplx)
            self.tri = self.interpolator.tri  # type: ignore[attr-defined]
        else:
            self.interpolator = LinearNDInterpolator(self.tri, velocities_cmplx)

        if self.points.shape[1] == 3:
            if self.tri is None:
                raise RuntimeError("self.tri not initialized")
            self.interpolator_z = LinearNDInterpolator(self.tri, self.velocities[:, 2])

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
    - Requires grid_shape and structured coordinate data.
    """

    VALID_METHODS = {"linear", "nearest", "slinear", "cubic", "quintic"}

    def __init__(
        self,
        grid_shape: tuple[int, ...],
        method: Literal["linear", "nearest", "slinear", "cubic", "quintic"] = "linear",
    ):
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"Invalid method '{method}'. Must be one of {self.VALID_METHODS}"
            )
        self.method = method
        self.grid_shape = grid_shape
        self.interpolator_x: Optional[RegularGridInterpolator] = None
        self.interpolator_y: Optional[RegularGridInterpolator] = None
        self.interpolator_z: Optional[RegularGridInterpolator] = None
        self.grid: Optional[tuple[np.ndarray, ...]] = None  # cached grid axes
        self.ndim: int

    def _initialize_interpolator(self) -> None:
        """Initializes the actual interpolator for grid-based interpolation."""

        if self.velocities is None or self.points is None:
            raise ValueError("Velocities and points must be set before initialization.")

        if self.grid_shape is None:
            raise ValueError("grid_shape must be provided before initialization.")

        n_points, self.ndim = self.points.shape
        expected_points = np.prod(self.grid_shape)

        if expected_points != n_points:
            raise ValueError(
                f"grid_shape {self.grid_shape} implies {expected_points} points, "
                f"but got {n_points}."
            )

        if self.ndim not in (2, 3):
            raise ValueError("Velocity field must have 2 or 3 components (u, v, [w]).")

        if self.ndim == 2:
            nx, ny = self.grid_shape
            nz = 1
        else:
            nx, ny, nz = self.grid_shape

        x = np.linspace(self.points[:, 0].min(), self.points[:, 0].max(), nx)
        y = np.linspace(self.points[:, 1].min(), self.points[:, 1].max(), ny)

        if self.ndim == 2:
            vel_shape = (ny, nx)
            self.grid = (x, y)
        else:
            z = np.linspace(self.points[:, 2].min(), self.points[:, 2].max(), nz)

            vel_shape = (ny, nx, nz)
            self.grid = (x, y, z)

        self.interpolator_x = RegularGridInterpolator(
            self.grid,
            np.swapaxes(self.velocities[:, 0].reshape(vel_shape), 0, 1),
            method=self.method,  # type: ignore
            bounds_error=False,
            fill_value=0.0,  # type: ignore[arg-type]
        )

        self.interpolator_y = RegularGridInterpolator(
            self.grid,
            np.swapaxes(self.velocities[:, 1].reshape(vel_shape), 0, 1),
            method=self.method,  # type: ignore
            bounds_error=False,
            fill_value=0.0,  # type: ignore[arg-type]
        )

        if self.ndim == 3:
            self.interpolator_z = RegularGridInterpolator(
                self.grid,
                np.swapaxes(self.velocities[:, 2].reshape(vel_shape), 0, 1),
                method=self.method,  # type: ignore
                bounds_error=False,
                fill_value=0.0,  # type: ignore[arg-type]
            )

    def interpolate(
        self, new_points: ArrayFloat64Nx2 | ArrayFloat64Nx3
    ) -> ArrayFloat64Nx2 | ArrayFloat64Nx3:
        """Interpolates velocity field at given Cartesian points."""
        if self.interpolator_x is None:
            raise ValueError(
                "Interpolator has not been initialized. Call `update()` first."
            )

        result = np.empty_like(new_points)

        result[..., 0] = self.interpolator_x(new_points)
        result[..., 1] = self.interpolator_y(new_points)  # type: ignore

        if self.ndim == 3:
            result[..., 2] = self.interpolator_z(new_points)  # type: ignore

        return result

    def update(
        self,
        velocities: ArrayFloat64Nx2 | ArrayFloat64Nx3,
        points: Optional[ArrayFloat64Nx2 | ArrayFloat64Nx3] = None,
    ) -> None:
        # If interp already initialized and don't need to update grid, then
        # just update the velocity field
        if self.interpolator_x is not None and points is None:
            velocity_field = velocities.reshape((*self.grid_shape, self.ndim))  # type: ignore

            if self.ndim == 2:
                self.interpolator_x.values[:, :] = velocity_field[:, :, 0]
                self.interpolator_y.values[:, :] = velocity_field[:, :, 1]  # type: ignore
            else:
                self.interpolator_x.values[:, :, :] = velocity_field[:, :, :, 0]  # type: ignore
                self.interpolator_y.values[:, :, :] = velocity_field[:, :, :, 1]  # type: ignore
                self.interpolator_z.values[:, :, :] = velocity_field[:, :, :, 2]  # type: ignore

        else:
            # initialize interpolator
            super().update(velocities, points)


class AnalyticalInterpolator(Interpolator):
    def __init__(self, velocity_fn: Callable):
        self.velocity_fn = velocity_fn
        self.time = 0.0

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
    interpolation_type: str,
    grid_shape: Optional[tuple[int, ...]] = None,
    velocity_fn: Optional[Callable] = None,
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
        grid_shape: used to choose GridInterpolator and passed to constructor
        velocity_fn: used only with 'analytical'

    Returns:
        An instance of the appropriate interpolator.
    """
    interpolation_type = interpolation_type.lower()

    interpolation_map: dict[str, type[Interpolator]] = {
        "cubic": CubicInterpolator,
        "linear": LinearInterpolator,
        "nearest": NearestNeighborInterpolator,
        "analytical": AnalyticalInterpolator,
    }

    if interpolation_type not in interpolation_map:
        raise ValueError(
            f"Invalid interpolation type '{interpolation_type}'. "
            f"Choose from {list(interpolation_map.keys())}."
        )

    if interpolation_type == "analytical":
        if not callable(velocity_fn):
            raise ExecutableNotFoundError("velocity_fn was not assigned properly")
        return AnalyticalInterpolator(velocity_fn)

    # If structured grid: use GridInterpolator with given method
    if grid_shape is not None:
        return GridInterpolator(grid_shape, interpolation_type)  # type: ignore

    # Fallback: construct the requested unstructured interpolator
    return interpolation_map[interpolation_type]()

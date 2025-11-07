# ruff: noqa: N806
from abc import ABC, abstractmethod
from typing import Callable, Literal, Optional, cast

import numpy as np
from matplotlib import ExecutableNotFoundError
from scipy.interpolate import (
    CloughTocher2DInterpolator,
    LinearNDInterpolator,
    NearestNDInterpolator,
    RegularGridInterpolator,
)
from scipy.spatial import Delaunay

from pyftle.grid_interp.grid_interp import Interp2D, Interp3D
from pyftle.my_types import Array2xN, Array3xN


class Interpolator(ABC):
    def __init__(self):
        """Lazy initialization: need to call update() once the velocities and
        points are available
        """

        self.velocities: Optional[Array2xN | Array3xN] = None
        self.points: Optional[Array2xN | Array3xN] = None
        self.interpolator = None  # Placeholder for the actual interpolator instance
        self.velocity_fn: Optional[Callable] = None  # Used only by AnalyticalInterp
        self.grid_shape: Optional[tuple[int, ...]] = None  # Used only by GridInterp
        self._velocities_buffer: Optional[np.ndarray] = None  # in-place ops

    def _initialize_interpolator(self) -> None:
        """Initialize the actual interpolator object based on velocity and points."""

        if self.velocities is None or self.points is None:
            raise ValueError("Velocities and points must be set before initialization.")

        # Ensure to check that the velocities and points are properly shaped
        if self.velocities.shape[-1] != self.points.shape[-1]:
            raise ValueError("Number of velocities must match the number of points.")

        raise NotImplementedError("This method should be implemented by subclasses")

    def update(
        self,
        velocities: Array2xN | Array3xN,
        points: Optional[Array2xN | Array3xN] = None,
    ) -> None:
        """Updates the interpolation function for a new field."""

        self.velocities = velocities
        if points is not None:
            self.points = points

        self._initialize_interpolator()

    @abstractmethod
    def interpolate(
        self,
        new_points: Array2xN | Array3xN,
    ) -> Array2xN | Array3xN:
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
        super().__init__()
        self.tri: Optional[Delaunay] = None  # pre-computed Delaunay for faster updates

    def _initialize_interpolator(self) -> None:
        if self.velocities is None or self.points is None:
            raise ValueError("Velocities and points must be set before initialization.")

        if len(self.points) == 3:
            raise ValueError("cubic interpolator is only valid for 2D cases")

        velocities_cmplx = self.velocities[0] + 1j * self.velocities[1]

        if self.tri is None:
            self.interpolator = CloughTocher2DInterpolator(
                self.points.T, velocities_cmplx
            )
            self.tri = self.interpolator.tri  # type: ignore[attr-defined]
        else:
            self.interpolator = CloughTocher2DInterpolator(self.tri, velocities_cmplx)

    def interpolate(
        self,
        new_points: Array2xN,
        out: Optional[Array2xN] = None,
    ) -> Array2xN:
        if out is None:
            out = np.empty_like(new_points)

        interp_velocities = self.interpolator(new_points)

        out[:, 0] = interp_velocities.real
        out[:, 1] = interp_velocities.imag

        return out


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
        super().__init__()
        self.tri: Optional[Delaunay] = None  # pre-computed Delaunay for faster updates

    def _initialize_interpolator(self) -> None:
        if self.velocities is None or self.points is None:
            raise ValueError("Velocities and points must be set before initialization.")

        velocities_cmplx = self.velocities[0] + 1j * self.velocities[1]

        if self.tri is None:
            self.interpolator = LinearNDInterpolator(self.points.T, velocities_cmplx)
            self.tri = self.interpolator.tri  # type: ignore[attr-defined]
        else:
            self.interpolator = LinearNDInterpolator(self.tri, velocities_cmplx)

        if len(self.points) == 3:
            if self.tri is None:
                raise RuntimeError("self.tri not initialized")
            self.interpolator_z = LinearNDInterpolator(self.tri, self.velocities[2])

    def interpolate(
        self, new_points: Array2xN | Array3xN, out=None
    ) -> Array2xN | Array3xN:
        if out is None:
            out = np.empty_like(new_points)

        interp_velocities = self.interpolator(new_points)

        out[:, 0] = interp_velocities.real
        out[:, 1] = interp_velocities.imag
        if len(new_points) == 3:
            out[:, 2] = self.interpolator_z(new_points)

        return out


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

        velocities_cmplx = self.velocities[0] + 1j * self.velocities[1]
        self.interpolator = NearestNDInterpolator(self.points.T, velocities_cmplx)
        if len(self.points) == 3:
            self.interpolator_z = NearestNDInterpolator(
                self.points.T, self.velocities[2]
            )

    def interpolate(
        self, new_points: Array2xN | Array3xN, out=None
    ) -> Array2xN | Array3xN:
        if out is None:
            out = np.empty_like(new_points)

        interp_velocities = self.interpolator(new_points)

        out[:, 0] = interp_velocities.real
        out[:, 1] = interp_velocities.imag
        if len(new_points) == 3:
            out[:, 2] = self.interpolator_z(new_points)

        return out


class HighPerfInterpolator(Interpolator):
    """Grid-based interpolation (2D or 3D) using C++/Eigen backends.

    Supports both bilinear (2D) and trilinear (3D) interpolation on
    regular rectangular grids.

    Automatically dispatches to Interp2D or Interp3D depending on
    the dimensionality of `grid_shape`.
    """

    def __init__(self, grid_shape: tuple[int, ...]):
        super().__init__()
        self.grid_shape = grid_shape
        self._u_buffer = None
        self._v_buffer = None
        self._w_buffer = None

        self.interpolator_x: Optional[Interp2D | Interp3D] = None
        self.interpolator_y: Optional[Interp2D | Interp3D] = None
        self.interpolator_z: Optional[Interp3D] = None

        self.velocities: Optional[np.ndarray] = None
        self.points: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def _initialize_interpolator(self) -> None:
        if self.velocities is None or self.points is None:
            raise ValueError("Velocities and points must be set before initialization.")

        grid_shape = cast(tuple[int, ...], self.grid_shape)
        dim = len(grid_shape)

        coordinates = self.points.reshape((dim, *grid_shape))

        # ---------------------------------------------------------------
        if dim == 2:
            velocities = self.velocities.reshape((dim, *grid_shape), order="F")

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

            # Initialize 2D interpolators
            self.interpolator_x = Interp2D(velocities[0], grid_x, grid_y)
            self.interpolator_y = Interp2D(velocities[1], grid_x, grid_y)
            self.interpolator_z = None  # not used in 2D

        # ---------------------------------------------------------------
        elif dim == 3:
            velocities = np.transpose(
                self.velocities.reshape((dim, *grid_shape)), axes=(0, 2, 1, 3)
            )

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
            grid_z = np.linspace(
                np.min(coordinates[2]),
                np.max(coordinates[2]),
                grid_shape[2],
            )

            # Initialize 3D interpolators
            self.interpolator_x = Interp3D(
                velocities[0],
                grid_x,
                grid_y,
                grid_z,
            )
            self.interpolator_y = Interp3D(
                velocities[1],
                grid_x,
                grid_y,
                grid_z,
            )
            self.interpolator_z = Interp3D(
                velocities[2],
                grid_x,
                grid_y,
                grid_z,
            )

        else:
            raise ValueError(f"Unsupported grid dimensionality: {dim}")

    # ------------------------------------------------------------------
    def _ensure_buffers(self, n: int, dim: int) -> None:
        """Allocate or resize buffers if necessary."""
        if self._u_buffer is None or self._u_buffer.shape[0] != n:
            self._u_buffer = np.empty(n)
            self._v_buffer = np.empty(n)
            if dim == 3:
                self._w_buffer = np.empty(n)

    # ------------------------------------------------------------------
    def interpolate(
        self,
        new_points: Array2xN | Array3xN,
        out=None,
    ) -> Array2xN | Array3xN:
        """Interpolates velocity field at given Cartesian points."""
        dim = new_points.shape[1]
        n = new_points.shape[0]

        if out is None:
            out = np.empty_like(new_points)

        self._ensure_buffers(n, dim)

        if dim == 2:
            assert self.interpolator_x is not None and self.interpolator_y is not None

            self.interpolator_x(new_points, out=self._u_buffer)
            self.interpolator_y(new_points, out=self._v_buffer)

            out[:, 0] = self._u_buffer
            out[:, 1] = self._v_buffer

        elif dim == 3:
            assert (
                self.interpolator_x is not None
                and self.interpolator_y is not None
                and self.interpolator_z is not None
            )

            self.interpolator_x(new_points, out=self._u_buffer)
            self.interpolator_y(new_points, out=self._v_buffer)
            self.interpolator_z(new_points, out=self._w_buffer)

            out[:, 0] = self._u_buffer
            out[:, 1] = self._v_buffer
            out[:, 2] = self._w_buffer

        else:
            raise ValueError(f"Unsupported point dimension: {dim}")

        return out


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
        super().__init__()
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

        self.ndim, n_points = self.points.shape
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

        x = np.linspace(self.points[0].min(), self.points[0].max(), nx)
        y = np.linspace(self.points[1].min(), self.points[1].max(), ny)

        if self.ndim == 2:
            vel_shape = (nx, ny)
            self.grid = (x, y)

            self.interpolator_x = RegularGridInterpolator(
                self.grid,
                self.velocities[0].reshape(vel_shape),
                method=self.method,  # type: ignore
                bounds_error=False,
                fill_value=0.0,  # type: ignore[arg-type]
            )

            self.interpolator_y = RegularGridInterpolator(
                self.grid,
                self.velocities[1].reshape(vel_shape),
                method=self.method,  # type: ignore
                bounds_error=False,
                fill_value=0.0,  # type: ignore[arg-type]
            )
        else:
            z = np.linspace(self.points[2].min(), self.points[2].max(), nz)

            vel_shape = (nx, ny, nz)
            self.grid = (x, y, z)

            self.interpolator_x = RegularGridInterpolator(
                self.grid,
                self.velocities[0].reshape(vel_shape),
                method=self.method,  # type: ignore
                bounds_error=False,
                fill_value=0.0,  # type: ignore[arg-type]
            )

            self.interpolator_y = RegularGridInterpolator(
                self.grid,
                self.velocities[1].reshape(vel_shape),
                method=self.method,  # type: ignore
                bounds_error=False,
                fill_value=0.0,  # type: ignore[arg-type]
            )

            self.interpolator_z = RegularGridInterpolator(
                self.grid,
                self.velocities[2].reshape(vel_shape),
                method=self.method,  # type: ignore
                bounds_error=False,
                fill_value=0.0,  # type: ignore[arg-type]
            )

    def interpolate(self, new_points: Array2xN | Array3xN) -> Array2xN | Array3xN:
        """Interpolates velocity field at given Cartesian points."""
        if self.interpolator_x is None:
            raise ValueError(
                "Interpolator has not been initialized. Call `update()` first."
            )

        result = np.empty_like(new_points)

        result[:, 0] = self.interpolator_x(new_points)
        result[:, 1] = self.interpolator_y(new_points)  # type: ignore

        if self.ndim == 3:
            result[:, 2] = self.interpolator_z(new_points)  # type: ignore

        return result

    def update(
        self,
        velocities: Array2xN | Array3xN,
        points: Optional[Array2xN | Array3xN] = None,
    ) -> None:
        # If interp already initialized and don't need to update grid, then
        # just update the velocity field
        if self.interpolator_x is not None and points is None:
            if self.ndim == 2:
                velocity_field = velocities.reshape(
                    (self.ndim, *self.grid_shape),  # type: ignore
                    order="F",  # Row-wise access in memory layout
                )
                self.interpolator_x.values = velocity_field[0]
                self.interpolator_y.values = velocity_field[1]  # type: ignore
            else:
                velocity_field = velocities.reshape((self.ndim, *self.grid_shape))  # type: ignore
                self.interpolator_x.values = velocity_field[0]
                self.interpolator_y.values = velocity_field[1]  # type: ignore
                self.interpolator_z.values = velocity_field[2]  # type: ignore

        else:
            # initialize interpolator
            super().update(velocities, points)


class AnalyticalInterpolator(Interpolator):
    def __init__(self, velocity_fn: Callable):
        self.velocity_fn = velocity_fn
        self.time = 0.0

    def interpolate(self, new_points: Array2xN | Array3xN) -> Array2xN | Array3xN:
        """Evaluates the velocity field at the given coordinates."""
        if not callable(self.velocity_fn):
            raise ExecutableNotFoundError("velocity_fn was not assigned properly")
        return self.velocity_fn(self.time, new_points)

    def update(
        self,
        velocities: Array2xN | Array3xN,
        points: Optional[Array2xN | Array3xN] = None,
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
        "grid": HighPerfInterpolator,
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
    if grid_shape is not None and interpolation_type != "grid":
        return GridInterpolator(grid_shape, interpolation_type)  # type: ignore

    if grid_shape is not None and interpolation_type == "grid":
        return HighPerfInterpolator(grid_shape)

    # Fallback: construct the requested unstructured interpolator
    return interpolation_map[interpolation_type]()

# ruff: noqa: N806
from abc import ABC, abstractmethod
from typing import Callable, Optional, cast

import numpy as np
from scipy.interpolate import (
    CloughTocher2DInterpolator,
    LinearNDInterpolator,
    NearestNDInterpolator,
    RegularGridInterpolator,
)

from src.interp3d import interp_3d
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
        self, new_points: ArrayFloat64Nx2 | ArrayFloat64Nx3, out=None
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

        self.interpolator = CloughTocher2DInterpolator(
            self.points, self.velocities[:, 0] + 1j * self.velocities[:, 1]
        )
        if self.points.shape[1] == 3:
            self.interpolator_z = CloughTocher2DInterpolator(
                self.points, self.velocities[:, 2]
            )

    def interpolate(
        self, new_points: ArrayFloat64Nx2 | ArrayFloat64Nx3, out=None
    ) -> ArrayFloat64Nx2 | ArrayFloat64Nx3:
        if out is None:
            if new_points.shape[1] == 2:
                out = np.empty((len(new_points), 2), dtype=float)
            else:
                out = np.empty((len(new_points), 3), dtype=float)

        # Get complex interpolation from self.interpolator
        interp_val = self.interpolator(new_points)  # complex-valued

        if new_points.shape[1] == 2:
            out[:, 0] = interp_val.real
            out[:, 1] = interp_val.imag
        else:
            interp_z = self.interpolator_z(new_points)
            out[:, 0] = interp_val.real
            out[:, 1] = interp_val.imag
            out[:, 2] = interp_z
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

    def _initialize_interpolator(self) -> None:
        if self.velocities is None or self.points is None:
            raise ValueError("Velocities and points must be set before initialization.")

        self.interpolator = LinearNDInterpolator(
            self.points, self.velocities[:, 0] + 1j * self.velocities[:, 1]
        )

        if self.points.shape[1] == 3:
            self.interpolator_z = LinearNDInterpolator(
                self.points, self.velocities[:, 2], fill_value=0.0
            )

    def interpolate(
        self, new_points: ArrayFloat64Nx2 | ArrayFloat64Nx3, out=None
    ) -> ArrayFloat64Nx2 | ArrayFloat64Nx3:
        if out is None:
            if new_points.shape[1] == 2:
                out = np.empty((len(new_points), 2), dtype=float)
            else:
                out = np.empty((len(new_points), 3), dtype=float)

        # Get complex interpolation from self.interpolator
        interp_val = self.interpolator(new_points)  # complex-valued

        if new_points.shape[1] == 2:
            out[:, 0] = interp_val.real
            out[:, 1] = interp_val.imag
        else:
            interp_z = self.interpolator_z(new_points)
            out[:, 0] = interp_val.real
            out[:, 1] = interp_val.imag
            out[:, 2] = interp_z

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
        self.interpolator = NearestNDInterpolator(
            self.points, self.velocities[:, 0] + 1j * self.velocities[:, 1]
        )
        if self.points.shape[1] == 3:
            self.interpolator_z = NearestNDInterpolator(
                self.points, self.velocities[:, 2]
            )

    def interpolate(
        self, new_points: ArrayFloat64Nx2 | ArrayFloat64Nx3, out=None
    ) -> ArrayFloat64Nx2 | ArrayFloat64Nx3:
        if out is None:
            if new_points.shape[1] == 2:
                out = np.empty((len(new_points), 2), dtype=float)
            else:
                out = np.empty((len(new_points), 3), dtype=float)

        # Get complex interpolation from self.interpolator
        interp_val = self.interpolator(new_points)  # complex-valued

        if new_points.shape[1] == 2:
            out[:, 0] = interp_val.real
            out[:, 1] = interp_val.imag
        else:
            interp_z = self.interpolator_z(new_points)
            out[:, 0] = interp_val.real
            out[:, 1] = interp_val.imag
            out[:, 2] = interp_z

        return out


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
        self.interpolator_z: Optional[RegularGridInterpolator] = None

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
        self, new_points: ArrayFloat64Nx2 | ArrayFloat64Nx3, out=None
    ) -> ArrayFloat64Nx2 | ArrayFloat64Nx3:
        """Interpolates velocity field at given Cartesian points."""
        assert self.interpolator_u is not None, (
            "Interpolator not initialized. Call _initialize_interpolator() first."
        )
        assert self.interpolator_v is not None, (
            "Interpolator not initialized. Call _initialize_interpolator() first."
        )

        if out is None:
            out = np.empty_like(new_points)
        out[:, 0] = self.interpolator_u(new_points)
        out[:, 1] = self.interpolator_v(new_points)
        if new_points.shape[1] == 3:
            assert self.interpolator_z is not None, (
                "3D interpolation attempted on a 2D-initialized field."
            )
            out[:, 2] = self.interpolator_z(new_points)
        return out


class CythonInterpolator(Interpolator):
    """Grid-based interpolation based on https://github.com/jglaser/interp3d.
    the implementation was vectorized to handle multiple particles at the same time.

    this method works for regular rectangular grids

    """

    def __init__(self):
        super().__init__()
        self._u_buffer = None
        self._v_buffer = None
        self._w_buffer = None

    def _initialize_interpolator(self) -> None:
        if self.velocities is None or self.points is None:
            raise ValueError("Velocities and points must be set before initialization.")

        coordinates = self.points
        velocities = self.velocities
        grid_shape = coordinates[0].shape

        if len(grid_shape) == 2:
            raise NotImplementedError("grid_cython only works for 3D cases.")

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

        grid_x = make_safe_array(grid_x)
        grid_y = make_safe_array(grid_y)
        grid_z = make_safe_array(grid_z)

        self.interpolator_u = interp_3d.Interp3D(
            make_safe_array(velocities[0]), grid_x, grid_y, grid_z
        )

        self.interpolator_v = interp_3d.Interp3D(
            make_safe_array(velocities[1]), grid_x, grid_y, grid_z
        )
        self.interpolator_z = interp_3d.Interp3D(
            make_safe_array(velocities[2]), grid_x, grid_y, grid_z
        )

    def _ensure_buffers(self, n):
        """Allocate or resize buffers if Ne"""
        if self._u_buffer is None or self._u_buffer.shape[0] != n:
            self._u_buffer = np.empty(n, dtype=np.float64)
            self._v_buffer = np.empty(n, dtype=np.float64)
            self._w_buffer = np.empty(n, dtype=np.float64)

    def interpolate(
        self, new_points: ArrayFloat64Nx2 | ArrayFloat64Nx3, out=None
    ) -> ArrayFloat64Nx2 | ArrayFloat64Nx3:
        """Interpolates velocity field at given Cartesian points."""

        n = new_points.shape[0]

        if out is None:
            out = np.empty_like(new_points)
        self._ensure_buffers(n)

        self.interpolator_u(new_points, out=self._u_buffer)
        self.interpolator_v(new_points, out=self._v_buffer)
        self.interpolator_z(new_points, out=self._w_buffer)
        out[:, 0] = self._u_buffer
        out[:, 1] = self._v_buffer
        out[:, 2] = self._w_buffer
        return out


class AnalyticalInterpolator(Interpolator):
    pass


class InMemoryInterpolator(Interpolator):
    """Interpolation wrapper for in-memory velocity fields."""

    def __init__(
        self, velocity_field: Callable | np.ndarray, coordinates, time: float = 0.0
    ):
        self.time = time
        self.velocity_field = velocity_field
        self.coordinates = coordinates
        self._is_callable = callable(velocity_field)

        if not self._is_callable:
            vf_array = cast(np.ndarray, velocity_field)
            X, Y, Z = coordinates
            u, v, w = (
                vf_array[..., 0],
                vf_array[..., 1],
                vf_array[..., 2],
            )
            self._interp_u = RegularGridInterpolator(
                (X[:, 0, 0], Y[0, :, 0], Z[0, 0, :]), u
            )
            self._interp_v = RegularGridInterpolator(
                (X[:, 0, 0], Y[0, :, 0], Z[0, 0, :]), v
            )
            self._interp_w = RegularGridInterpolator(
                (X[:, 0, 0], Y[0, :, 0], Z[0, 0, :]), w
            )

    def interpolate(
        self, new_points: ArrayFloat64Nx2 | ArrayFloat64Nx3, out=None
    ) -> ArrayFloat64Nx2 | ArrayFloat64Nx3:
        """Evaluates the velocity field at the given coordinates."""
        if out is None:
            out = np.empty_like(new_points)  ### need to fix this to inplace as well
        if self._is_callable:
            vf_func = cast(Callable[..., np.ndarray], self.velocity_field)

            x, y = new_points[:, 0], new_points[:, 1]
            z = new_points[:, 2] if new_points.shape[1] == 3 else None
            result = vf_func(x, y, z, self.time)
            return np.asarray(result)

        # Array-based interpolation
        x, y, z = new_points[:, 0], new_points[:, 1], new_points[:, 2]
        u = self._interp_u((x, y, z))
        v = self._interp_v((x, y, z))
        w = self._interp_w((x, y, z))

        return np.column_stack((u, v, w))


def create_interpolator(interpolation_type: str) -> Interpolator:
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
        "cubic": CubicInterpolator,
        "linear": LinearInterpolator,
        "nearest": NearestNeighborInterpolator,
        "grid": GridInterpolator,
        "grid_cython": CythonInterpolator,
    }

    if interpolation_type not in interpolation_map:
        raise ValueError(
            f"Invalid interpolation type '{interpolation_type}'. "
            f"Choose from {list(interpolation_map.keys())}."
        )

    return interpolation_map[interpolation_type]()


def make_safe_array(arr):
    return np.ascontiguousarray(arr, dtype=np.float64)

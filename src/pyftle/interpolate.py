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
    """
    Abstract base class for velocity field interpolation strategies.

    This class defines a common interface for interpolators used to estimate
    velocities at arbitrary spatial points, either from discrete samples or
    analytical expressions.

    Subclasses must implement the :meth:`interpolate` method and may override
    the :meth:`_initialize_interpolator` method to define their specific behavior.
    """

    def __init__(self):
        """
        Initialize an uninitialized interpolator instance.

        This class uses *lazy initialization*: the actual interpolation object
        is created only when :meth:`update` is called with velocity and coordinate data.
        """

        self.velocities: Optional[Array2xN | Array3xN] = None
        self.points: Optional[Array2xN | Array3xN] = None
        self.interpolator = None  # Placeholder for the actual interpolator instance
        self.velocity_fn: Optional[Callable] = None  # Used only by AnalyticalInterp
        self.grid_shape: Optional[tuple[int, ...]] = None  # Used only by GridInterp
        self._velocities_buffer: Optional[np.ndarray] = None  # in-place ops

    def _initialize_interpolator(self) -> None:
        """
        Create the concrete interpolator object.

        This method must be overridden by subclasses. It is called by :meth:`update`
        once velocity and coordinate data are set.

        Raises
        ------
        ValueError
            If `velocities` or `points` are not set, or their shapes are inconsistent.
        NotImplementedError
            Always, unless overridden by subclasses.
        """

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
        """
        Update the interpolator with new velocity and coordinate data.

        Parameters
        ----------
        velocities : Array2xN or Array3xN
            Velocity components at each grid or sample point.
        points : Array2xN or Array3xN, optional
            Spatial coordinates corresponding to each velocity sample.
            If omitted, previously stored points are reused.
        """

        self.velocities = velocities
        if points is not None:
            self.points = points

        self._initialize_interpolator()

    @abstractmethod
    def interpolate(
        self,
        new_points: Array2xN | Array3xN,
    ) -> Array2xN | Array3xN:
        """
        Interpolate the velocity field at given spatial coordinates.

        Parameters
        ----------
        new_points : Array2xN or Array3xN
            Coordinates where the velocity should be interpolated.

        Returns
        -------
        Array2xN or Array3xN
            Interpolated velocity components at the queried points.
        """
        pass


class CubicInterpolator(Interpolator):
    """
    Clough-Tocher piecewise cubic interpolator for 2D velocity fields.

    Provides C¹-smooth, curvature-minimizing interpolation using Delaunay
    triangulation of the input points.

    Pros
    ----
    - Produces smooth, high-quality interpolation.
    - Suitable for smoothly varying velocity fields.

    Cons
    ----
    - Computationally expensive (requires triangulation).
    - Limited to 2D domains.
    """

    def __init__(self):
        super().__init__()
        self.tri: Optional[Delaunay] = None  # pre-computed Delaunay for faster updates

    def _initialize_interpolator(self) -> None:
        """Construct the Clough-Tocher interpolator."""

        if self.velocities is None or self.points is None:
            raise ValueError("Velocities and points must be set before initialization.")

        if len(self.points) == 3:
            raise ValueError("cubic interpolator is only valid for 2D cases")

        velocities_cmplx = self.velocities[0] + 1j * self.velocities[1]

        # If there is no previous triangulation or if we have new points, then
        # recreate the interpolator from scratch, else use existing triangulation
        if self.tri is None or self.points is not None:
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
        """
        Interpolate velocities at new points using Clough-Tocher interpolation.

        Parameters
        ----------
        new_points : Array2xN
            Query coordinates.
        out : Array2xN, optional
            Output array for in-place storage.

        Returns
        -------
        Array2xN
            Interpolated velocity components.
        """
        if out is None:
            out = np.empty_like(new_points)

        interp_velocities = self.interpolator(new_points)

        out[:, 0] = interp_velocities.real
        out[:, 1] = interp_velocities.imag

        return out


class LinearInterpolator(Interpolator):
    """
    Piecewise linear interpolator for 2D or 3D velocity fields.

    Uses Delaunay triangulation for unstructured grids. Faster than Clough-Tocher
    but less smooth.

    Pros
    ----
    - Computationally efficient.
    - Handles scattered (unstructured) points.

    Cons
    ----
    - Discontinuous derivatives.
    - Less accurate for smooth fields.
    """

    def __init__(self):
        super().__init__()
        self.tri: Optional[Delaunay] = None  # pre-computed Delaunay for faster updates

    def _initialize_interpolator(self) -> None:
        """Construct the linear interpolator (2D or 3D)."""

        if self.velocities is None or self.points is None:
            raise ValueError("Velocities and points must be set before initialization.")

        velocities_cmplx = self.velocities[0] + 1j * self.velocities[1]

        # If there is no previous triangulation or if we have new points, then
        # recreate the interpolator from scratch, else use existing triangulation
        if self.tri is None or self.points is not None:
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
        """Perform linear interpolation on unstructured data."""

        if out is None:
            out = np.empty_like(new_points)

        interp_velocities = self.interpolator(new_points)

        out[:, 0] = interp_velocities.real
        out[:, 1] = interp_velocities.imag
        if len(new_points) == 3:
            out[:, 2] = self.interpolator_z(new_points)

        return out


class NearestNeighborInterpolator(Interpolator):
    """
    Nearest-neighbor interpolation for discrete velocity data.

    Assigns each query point the velocity of its nearest known sample.

    Pros
    ----
    - Extremely fast.
    - No triangulation or precomputation required.

    Cons
    ----
    - Produces blocky, discontinuous velocity fields.
    - Not suitable for smooth physical flows.
    """

    def _initialize_interpolator(self) -> None:
        """Construct nearest-neighbor interpolator."""

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
        """Interpolate velocities using nearest-neighbor strategy."""

        if out is None:
            out = np.empty_like(new_points)

        interp_velocities = self.interpolator(new_points)

        out[:, 0] = interp_velocities.real
        out[:, 1] = interp_velocities.imag
        if len(new_points) == 3:
            out[:, 2] = self.interpolator_z(new_points)

        return out


class HighPerfInterpolator(Interpolator):
    """
    High-performance grid-based interpolator using C++/Eigen backends.

    Supports both 2D (bilinear) and 3D (trilinear) interpolation on structured,
    rectangular grids. This class dispatches automatically to `Interp2D` or
    `Interp3D` implementations depending on the dimensionality of
    `grid_shape`.

    It is optimized for speed and suitable for large-scale flow field
    interpolation, such as FTLE or particle advection computations.

    Parameters
    ----------
    grid_shape : tuple[int, ...]
        The number of grid points along each spatial axis, e.g.
        `(nx, ny)` for 2D or `(nx, ny, nz)` for 3D grids.

    Attributes
    ----------
    grid_shape : tuple[int, ...]
        Grid resolution in each dimension.
    interpolator_x, interpolator_y, interpolator_z : Interp2D or Interp3D, optional
        C++-based interpolator instances for each velocity component.
    velocities : np.ndarray or None
        Flattened velocity components used to initialize the interpolators.
    points : np.ndarray or None
        Flattened coordinate array corresponding to the velocity data.
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
        """
        Initialize low-level C++/Eigen interpolators based on grid dimensionality.

        Raises
        ------
        ValueError
            If `velocities` or `points` are not set, or the grid dimensionality
            is not supported (only 2D and 3D are valid).
        """

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
        """
        Allocate or resize interpolation buffers when necessary.

        Parameters
        ----------
        n : int
            Number of query points for interpolation.
        dim : int
            Spatial dimension of the velocity field (2 or 3).
        """

        if self._u_buffer is None or self._u_buffer.shape[0] != n:
            self._u_buffer = np.empty(n)
            self._v_buffer = np.empty(n)
            if dim == 3:
                self._w_buffer = np.empty(n)

    # ------------------------------------------------------------------
    def update(
        self,
        velocities: Array2xN | Array3xN,
        points: Optional[Array2xN | Array3xN] = None,
    ) -> None:
        """
        Update the interpolator with new velocity and coordinate data.

        Parameters
        ----------
        velocities : Array2xN or Array3xN
            Velocity components at each grid or sample point.
        points : Array2xN or Array3xN, optional
            Spatial coordinates corresponding to each velocity sample.
            If omitted, previously stored points are reused.
        """

        self.velocities = velocities
        if points is not None:
            self.points = points
            self._initialize_interpolator()
        else:
            if self.interpolator_x and self.interpolator_y:
                self.interpolator_x.v = velocities[0, :]
                self.interpolator_y.v = velocities[1, :]
            if self.interpolator_z:
                self.interpolator_z.v = velocities[2, :]

    # ------------------------------------------------------------------
    def interpolate(
        self,
        new_points: Array2xN | Array3xN,
        out=None,
    ) -> Array2xN | Array3xN:
        """
        Interpolate the velocity field at arbitrary Cartesian coordinates.

        Parameters
        ----------
        new_points : Array2xN or Array3xN
            Coordinates where the velocity field should be interpolated.
        out : np.ndarray, optional
            Optional preallocated output array for efficiency.

        Returns
        -------
        out : Array2xN or Array3xN
            Interpolated velocity vectors at the requested points.

        Raises
        ------
        ValueError
            If the dimensionality of `new_points` is not supported.
        AssertionError
            If the internal interpolators were not properly initialized.

        Notes
        -----
        - Uses preallocated NumPy buffers for minimal memory overhead.
        - Supports both 2D and 3D velocity fields.
        """

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
    """
    Grid-based interpolator using SciPy's `RegularGridInterpolator`.

    This interpolator is designed for structured (rectangular) grids and supports
    linear, nearest, cubic, and higher-order methods. It is slower than
    `HighPerfInterpolator` but more flexible for prototyping or when C++ bindings
    are not available.

    Pros
    ----
    - Fast for structured, regular data.
    - Lower memory usage compared to unstructured methods.

    Cons
    ----
    - Requires structured grid coordinates.
    - Not suited for scattered data.

    Parameters
    ----------
    grid_shape : tuple[int, ...]
        Number of grid points along each spatial dimension.
    method : {'linear', 'nearest', 'slinear', 'cubic', 'quintic'}, optional
        Interpolation scheme to use. Default is `'linear'`.

    Attributes
    ----------
    interpolator_x, interpolator_y, interpolator_z : RegularGridInterpolator, optional
        Interpolator objects for each velocity component.
    grid : tuple[np.ndarray, ...] or None
        Cached coordinate arrays defining the grid axes.
    ndim : int
        Dimensionality of the velocity field (2 or 3).
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
        """
        Initialize SciPy `RegularGridInterpolator` instances for each component.

        Raises
        ------
        ValueError
            If `velocities` or `points` are not set, or the grid shape does not
            match the number of provided points.
        """

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
        """
        Interpolate the velocity field at given Cartesian coordinates.

        Parameters
        ----------
        new_points : Array2xN or Array3xN
            Target coordinates for interpolation.

        Returns
        -------
        velocities : Array2xN or Array3xN
            Interpolated velocity components at `new_points`.

        Raises
        ------
        ValueError
            If the interpolator has not been initialized.
        """
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
        """
        Update the internal velocity field or reinitialize the interpolator.

        Parameters
        ----------
        velocities : Array2xN or Array3xN
            Updated velocity components in flattened order.
        points : Array2xN or Array3xN, optional
            Optional grid coordinates. If provided, reinitializes the
            interpolator; otherwise, only updates the values.

        Notes
        -----
        - Efficiently updates preinitialized interpolators by replacing `.values`.
        - Calls `super().update()` if reinitialization is required.
        """
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
    """
    Interpolator that evaluates an analytical velocity field function.

    Useful for synthetic or theoretical flow fields where velocity values
    are defined by an analytical expression rather than discrete data.

    Parameters
    ----------
    velocity_fn : Callable
        A function of the form `velocity_fn(t: float, points: np.ndarray) -> np.ndarray`
        returning velocity vectors for given time `t` and coordinates.

    Attributes
    ----------
    velocity_fn : Callable
        User-defined analytical velocity function.
    time : float
        Current simulation time.
    """

    def __init__(self, velocity_fn: Callable):
        self.velocity_fn = velocity_fn
        self.time = 0.0

    def interpolate(self, new_points: Array2xN | Array3xN) -> Array2xN | Array3xN:
        """
        Evaluate the analytical velocity field at the given points.

        Parameters
        ----------
        new_points : Array2xN or Array3xN
            Cartesian coordinates where the velocity is to be evaluated.

        Returns
        -------
        velocities : Array2xN or Array3xN
            Velocity vectors computed from the analytical function.

        Raises
        ------
        ExecutableNotFoundError
            If `velocity_fn` has not been properly initialized.
        """

        if not callable(self.velocity_fn):
            raise ExecutableNotFoundError("velocity_fn was not assigned properly")
        return self.velocity_fn(self.time, new_points)

    def update(
        self,
        velocities: Array2xN | Array3xN,
        points: Optional[Array2xN | Array3xN] = None,
    ) -> None:
        """
        Do nothing (analytical interpolators have no internal state).

        Parameters
        ----------
        velocities : Array2xN or Array3xN
            Ignored. Present to satisfy the interpolator interface.
        points : Array2xN or Array3xN, optional
            Ignored. Present to satisfy the interpolator interface.
        """
        pass


def create_interpolator(
    interpolation_type: str,
    grid_shape: Optional[tuple[int, ...]] = None,
    velocity_fn: Optional[Callable] = None,
) -> Interpolator:
    """
    Factory function to construct an interpolator instance.

    Parameters
    ----------
    interpolation_type : str
        Type of interpolation to perform. Supported options:
        - `"cubic"`: Clough-Tocher cubic interpolation (smooth but slower).
        - `"linear"`: Linear interpolation (fast, moderately smooth).
        - `"nearest"`: Nearest-neighbor interpolation (very fast, discontinuous).
        - `"grid"`: Grid-based C++/Eigen interpolation (`HighPerfInterpolator`).
        - `"analytical"`: Analytical function-based interpolation.
    grid_shape : tuple[int, ...], optional
        Shape of the structured grid, required for `"grid"` or `"linear"` types.
    velocity_fn : Callable, optional
        Analytical velocity function, required for `"analytical"` type.

    Returns
    -------
    interpolator : Interpolator
        An initialized interpolator matching the requested configuration.

    Raises
    ------
    ValueError
        If an unsupported interpolation type is requested.
    ExecutableNotFoundError
        If `velocity_fn` is not provided for analytical interpolation.

    Examples
    --------
    >>> interp = create_interpolator("grid", grid_shape=(64, 64))
    >>> interp.update(velocities, points)
    >>> v_interp = interp.interpolate(new_points)
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

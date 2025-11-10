import numpy as np

from pyftle.ginterp import interp2d_vec_inplace, interp3d_vec_inplace


class Interp2D:
    """
    Efficient 2D interpolator for regularly spaced scalar fields.

    This class performs vectorized bilinear interpolation on a regular
    Cartesian grid using a precompiled C++ backend via `pybind11`.

    It converts physical coordinates `(x, y)` to normalized grid-space
    indices and performs interpolation in-place into a provided output array.

    Parameters
    ----------
    v : ArrayNx2
        2D scalar field values defined on a regular `(x, y)` grid.
    x, y : ndarray of float
        1D coordinate arrays defining the regular grid in each dimension.
        Must be monotonically increasing.

    Notes
    -----
    The interpolation uses preallocated internal buffers to minimize
    allocations and improve performance during repeated evaluations.
    """

    def __init__(self, v, x, y):
        self.v = np.ascontiguousarray(v)
        self.min_x, self.max_x = x[0], x[-1]
        self.min_y, self.max_y = y[0], y[-1]
        self.delta_x = (self.max_x - self.min_x) / (x.shape[0] - 1)
        self.delta_y = (self.max_y - self.min_y) / (y.shape[0] - 1)

        # Preallocated buffer for grid coordinates (to avoid reallocations)
        self._grid_points_buffer = None

    def __call__(self, t, out):
        """
        Interpolate scalar values at arbitrary 2D physical-space coordinates.

        Parameters
        ----------
        t : ArrayNx2
            Query points in physical coordinates of shape `(N, 2)`.
        out : ArrayN
            Preallocated output array of shape `(N,)` where interpolated
            values are written in-place.

        Returns
        -------
        out : ArrayN or float
            If `out.shape[0] == 1`, returns a scalar float; otherwise,
            returns the same `out` array after being filled in-place.

        Notes
        -----
        The physical coordinates are internally mapped to grid-space indices
        according to:

            i = (x - x_min) / Δx
            j = (y - y_min) / Δy

        Then, `interp2d_vec_inplace` performs bilinear interpolation
        using the surrounding grid values in `v`.
        """
        arr = np.atleast_2d(t)
        n = arr.shape[0]

        # Reuse or create buffer for index-space coordinates
        if self._grid_points_buffer is None or self._grid_points_buffer.shape[0] != n:
            self._grid_points_buffer = np.empty_like(arr)

        # Convert physical coordinates to normalized grid indices (in-place)
        points = self._grid_points_buffer
        points[:, 0] = (arr[:, 0] - self.min_x) / self.delta_x
        points[:, 1] = (arr[:, 1] - self.min_y) / self.delta_y

        interp2d_vec_inplace(self.v, points, out)

        if out.shape[0] == 1:
            return out[0]
        return out


class Interp3D:
    """
    Efficient 3D interpolator for regularly spaced scalar fields.

    This class performs vectorized trilinear interpolation on a regular
    Cartesian grid using a precompiled C++ backend via `pybind11`.

    It converts physical coordinates `(x, y, z)` to normalized grid-space
    indices and performs interpolation in-place into a provided output array.

    Parameters
    ----------
    v : ArrayNx3
        3D scalar field values defined on a regular `(x, y, z)` grid.
    x, y, z : ndarray of float
        1D coordinate arrays defining the regular grid in each dimension.
        Must be monotonically increasing.

    Notes
    -----
    The class uses an internal reusable buffer to reduce memory allocations
    when called repeatedly for multiple queries.
    """

    def __init__(self, v, x, y, z):
        self.v = np.ascontiguousarray(v)
        self.min_x, self.max_x = x[0], x[-1]
        self.min_y, self.max_y = y[0], y[-1]
        self.min_z, self.max_z = z[0], z[-1]
        self.delta_x = (self.max_x - self.min_x) / (x.shape[0] - 1)
        self.delta_y = (self.max_y - self.min_y) / (y.shape[0] - 1)
        self.delta_z = (self.max_z - self.min_z) / (z.shape[0] - 1)
        # Buffer for grid point coordinates to prevent reallocation
        self._grid_points_buffer = None

    def __call__(self, t, out):
        """
        Interpolate scalar values at arbitrary 3D physical-space coordinates.

        Parameters
        ----------
        t : ArrayNx3
            Query points in physical coordinates of shape `(N, 3)`.
        out : ArrayN
            Preallocated output array of shape `(N,)` where interpolated
            values are written in-place.

        Returns
        -------
        out : ArrayN or float
            If `out.shape[0] == 1`, returns a scalar float; otherwise,
            returns the same `out` array after being filled in-place.

        Notes
        -----
        The physical coordinates are mapped to grid-space indices via:

            i = (x - x_min) / Δx
            j = (y - y_min) / Δy
            k = (z - z_min) / Δz

        The function `interp3d_vec_inplace` then performs trilinear interpolation
        using the surrounding grid values in `v`.

        For best performance, `out` should be a preallocated `float64` array
        with contiguous memory layout.
        """
        arr = np.atleast_2d(t)
        n = arr.shape[0]

        # Reuse or create the buffer for grid coordinates
        if self._grid_points_buffer is None or self._grid_points_buffer.shape[0] != n:
            self._grid_points_buffer = np.empty_like(arr)

        # Convert physical coordinates to grid indices (in-place)
        points = self._grid_points_buffer
        points[:, 0] = (arr[:, 0] - self.min_x) / self.delta_x
        points[:, 1] = (arr[:, 1] - self.min_y) / self.delta_y
        points[:, 2] = (arr[:, 2] - self.min_z) / self.delta_z

        interp3d_vec_inplace(self.v, points, out)

        if out.shape[0] == 1:
            return out[0]
        return out

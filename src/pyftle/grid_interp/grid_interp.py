import numpy as np

from pyftle.ginterp import interp2d_vec_inplace, interp3d_vec_inplace


class Interp2D:
    def __init__(self, v, x, y):
        """
        v : 2D numpy array (float64)
            Grid values defined on a regular (x, y) mesh.
        x, y : 1D numpy arrays
            Grid coordinate vectors for each dimension.
        """
        self.v = np.ascontiguousarray(v)
        self.min_x, self.max_x = x[0], x[-1]
        self.min_y, self.max_y = y[0], y[-1]
        self.delta_x = (self.max_x - self.min_x) / (x.shape[0] - 1)
        self.delta_y = (self.max_y - self.min_y) / (y.shape[0] - 1)

        # Preallocated buffer for grid coordinates (to avoid reallocations)
        self._grid_points_buffer = None

    def __call__(self, t, out):
        """
        Interpolates the grid at given physical-space coordinates.

        Parameters
        ----------
        t : (N,2) array-like
            Query points in physical coordinates.
        out : (N,) array-like
            Pre-allocated output array to store the interpolated values in-place.

        Returns
        -------
        If out.shape[0] == 1, returns scalar float. Otherwise, returns `out`.
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
        t: (N,3) array for multiple points.
        out: Optional pre-allocated array (N,) to store the results in-place.
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

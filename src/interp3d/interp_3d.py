import numpy as np

# Import the new in-place function
from .interp import interp3D_vec, interp3D_vec_inplace


class Interp3D:
    def __init__(self, v, x, y, z):
        self.v = np.ascontiguousarray(v, dtype=np.float64)
        self.min_x, self.max_x = x[0], x[-1]
        self.min_y, self.max_y = y[0], y[-1]
        self.min_z, self.max_z = z[0], z[-1]
        self.delta_x = (self.max_x - self.min_x) / (x.shape[0] - 1)
        self.delta_y = (self.max_y - self.min_y) / (y.shape[0] - 1)
        self.delta_z = (self.max_z - self.min_z) / (z.shape[0] - 1)
        # Buffer for grid point coordinates to prevent reallocation
        self._grid_points_buffer = None

    def __call__(self, t, out=None):
        """
        t: (N,3) array for multiple points.
        out: Optional pre-allocated array (N,) to store the results in-place.
        """
        arr = np.atleast_2d(t)
        N = arr.shape[0]

        # Reuse or create the buffer for grid coordinates
        if self._grid_points_buffer is None or self._grid_points_buffer.shape[0] != N:
            self._grid_points_buffer = np.empty_like(arr, dtype=np.float64)

        # Convert physical coordinates to grid indices (in-place)
        points = self._grid_points_buffer
        points[:, 0] = (arr[:, 0] - self.min_x) / self.delta_x
        points[:, 1] = (arr[:, 1] - self.min_y) / self.delta_y
        points[:, 2] = (arr[:, 2] - self.min_z) / self.delta_z

        # If an output array is provided, use the in-place Cython function
        if out is not None:
            interp3D_vec_inplace(self.v, points, out)
            if out.shape[0] == 1:
                return out[0]
            return out
        # Otherwise, fall back to the original behavior
        else:
            result = interp3D_vec(self.v, points)
            if result.shape[0] == 1:
                return result[0]
            return result

from typing import Protocol

import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator

from src.my_types import ArrayFloat32N, ArrayFloat32Nx2


class InterpolationStrategy(Protocol):
    def interpolate(
        self,
        new_points: ArrayFloat32Nx2,
    ) -> ArrayFloat32Nx2:
        """Implements the integration strategy"""
        ...


class CubicInterpolatorStrategy:
    """Piecewise cubic, C1 smooth, curvature-minimizing interpolator in 2D
    for the velocity field.

    Parameters
    ----------
    points : NDArray
        array of shape `(n_points, 2)` representing the coordinates.

    velocities_u : NDArray
        array of shape `(n_points)` representing the u-velocity values.

    velocities_v : NDArray
        array of shape `(n_points)` representing the u-velocity values.
    """

    def __init__(
        self,
        points: ArrayFloat32Nx2,
        velocities_u: ArrayFloat32N,
        velocities_v: ArrayFloat32N,
    ):
        self.interpolator_u = CloughTocher2DInterpolator(points, velocities_u)
        self.interpolator_v = CloughTocher2DInterpolator(points, velocities_v)

    def interpolate(
        self,
        new_points: ArrayFloat32Nx2,
    ) -> ArrayFloat32Nx2:
        """Interpolate the field to the new points.

        Parameters
        ----------
        new_points : NDArray
            array of shape `(n_points, 2)` representing the coordinates .

        Returns
        -------
        interpolated_velocities : NDArray
            array of shape `(n_points, 2)` representing the stacked u- and
            v-velocities.
        """
        u_interpolated = self.interpolator_u(new_points)
        v_interpolated = self.interpolator_v(new_points)

        return np.column_stack((u_interpolated, v_interpolated))

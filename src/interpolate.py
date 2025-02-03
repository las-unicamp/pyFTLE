import numpy as np
from scipy.interpolate import griddata

from src.my_types import ArrayFloat32Nx2


def velocity_interpolation(
    velocities: ArrayFloat32Nx2,
    points: ArrayFloat32Nx2,
    new_points: ArrayFloat32Nx2,
) -> ArrayFloat32Nx2:
    """
    Interpolates velocity data to new points.
    """
    u_interp = griddata(
        (points[:, 0], points[:, 1]),
        velocities[:, 0],
        (new_points[:, 0], new_points[:, 1]),
        method="cubic",
        fill_value=0.0,
    )
    v_interp = griddata(
        (points[:, 0], points[:, 1]),
        velocities[:, 1],
        (new_points[:, 0], new_points[:, 1]),
        method="cubic",
        fill_value=0.0,
    )

    return np.column_stack((u_interp, v_interp))

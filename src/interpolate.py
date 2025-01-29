import numpy as np
from scipy.interpolate import griddata

from src.file_readers import read_coordinates, read_velocity_data
from src.my_types import ArrayFloat32Nx2


def velocity_interpolation(
    velocity_file_name: str,
    grid_file_name: str,
    new_points: ArrayFloat32Nx2,
) -> ArrayFloat32Nx2:
    """
    Reads velocity data from a file and interpolates it to the given points.
    The functions that read the velocity and grid data are decorated to use a
    rolling cache to avoid unneccesary IO streams.
    """
    velocity_data = read_velocity_data(velocity_file_name)
    grid_data = read_coordinates(grid_file_name)

    print("grid_data shape:", grid_data.shape)
    print("velocity_data shape:", velocity_data.shape)
    print("new_points shape:", new_points.shape)

    # Perform interpolation
    u_interp = griddata(
        (grid_data[:, 0], grid_data[:, 1]),
        velocity_data[:, 0],
        (new_points[:, 0], new_points[:, 1]),
        method="cubic",
        fill_value=0.0,
    )
    v_interp = griddata(
        (grid_data[:, 0], grid_data[:, 1]),
        velocity_data[:, 1],
        (new_points[:, 0], new_points[:, 1]),
        method="cubic",
        fill_value=0.0,
    )

    return np.column_stack((u_interp, v_interp))

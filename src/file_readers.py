import numpy as np
from scipy.io import loadmat

from src.caching import cache_last_n_files
from src.dtos import NeighboringParticles
from src.my_types import ArrayFloat32Nx2


@cache_last_n_files(num_cached_files=2)
def read_velocity_data(file_path: str) -> ArrayFloat32Nx2:
    """
    Reads velocity data from a MATLAB file and returns it as a numpy array
    with shape [n_points, 2] (velocity_x, velocity_y).

    Args:
        file_path (str): Path to the MATLAB file.

    Returns:
        ArrayFloat32Nx2: Array of shape [n_points, 2].
    """
    data = loadmat(file_path)
    velocity_x = data["velocity_x"].flatten()
    velocity_y = data["velocity_y"].flatten()
    return np.column_stack((velocity_x, velocity_y))


@cache_last_n_files(num_cached_files=2)
def read_coordinates(file_path: str) -> ArrayFloat32Nx2:
    """
    Reads coordinate data from a MATLAB file and returns it as a numpy array
    with shape [n_points, 2] (coordinate_x, coordinate_y).

    Args:
        file_path (str): Path to the MATLAB file.

    Returns:
        np.ndarray: Array of shape [n_points, 2].
    """
    data = loadmat(file_path)
    coordinate_x = data["coordinate_x"].flatten()
    coordinate_y = data["coordinate_y"].flatten()
    return np.column_stack((coordinate_x, coordinate_y))


@cache_last_n_files(num_cached_files=2)
def read_seed_particles_coordinates(file_path: str) -> NeighboringParticles:
    """
    Reads seeded particle coordinates from a MATLAB file and returns a dictionary
    (NeighboringParticles) whose keys identify the 4 neighboring particles as
    "left", "right", "top" and "bottom". The key values are coordinate arrays
    of shape [n_particles, 2].

    Args:
        file_path (str): Path to the MATLAB file.

    Returns:
        NeighboringParticles: Dictionary of neighboring particles, whose values
        are arrays of shape [n_particles, 2].
    """
    data = loadmat(file_path)
    neighboring_particles: NeighboringParticles = {
        "top": data["top"],
        "bottom": data["bottom"],
        "left": data["left"],
        "right": data["right"],
    }
    return neighboring_particles

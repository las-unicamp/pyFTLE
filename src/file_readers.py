import numpy as np
from scipy.io import loadmat

from src.my_types import ArrayFloat64Nx2, ArrayFloat64Nx3
from src.particles import NeighboringParticles


def read_velocity(file_path: str) -> ArrayFloat64Nx2 | ArrayFloat64Nx3:
    """
    Reads velocity data from a MATLAB file.
    """
    data = loadmat(file_path)

    if "velocity_x" not in data or "velocity_y" not in data:
        raise ValueError(
            "The MATLAB file does not contain the expected keys 'velocity_x' and"
            "'velocity_y'."
        )

    velocity_x = np.asarray(data["velocity_x"]).flatten().astype(np.float64)
    velocity_y = np.asarray(data["velocity_y"]).flatten().astype(np.float64)

    if "velocity_z" in data:
        velocity_z = np.asarray(data["velocity_z"]).flatten().astype(np.float64)

        return np.column_stack((velocity_x, velocity_y, velocity_z))
    else:
        return np.column_stack((velocity_x, velocity_y))


def read_coordinate(file_path: str) -> ArrayFloat64Nx2:
    """
    Reads coordinate data from a MATLAB file.
    """
    data = loadmat(file_path)

    if "coordinate_x" not in data or "coordinate_y" not in data:
        raise ValueError(
            "The MATLAB file does not contain the expected keys 'coordinate_x' and"
            "'coordinate_y'."
        )

    coordinate_x = np.asarray(data["coordinate_x"]).flatten().astype(np.float64)
    coordinate_y = np.asarray(data["coordinate_y"]).flatten().astype(np.float64)

    if "coordinate_z" in data:
        coordinate_z = np.asarray(data["coordinate_z"]).flatten().astype(np.float64)
        return np.column_stack((coordinate_x, coordinate_y, coordinate_z))
    else:
        return np.column_stack((coordinate_x, coordinate_y))


def read_seed_particles_coordinates(file_path: str) -> NeighboringParticles:
    """
    Reads seeded particle coordinates from a MATLAB file containing `left`, `right`
    `top` and `bottom` keys to identify the 4 neighboring particles. Then, returns
    a NeighboringParticles object that holds the coordinate array and other
    useful attributes.

    Args:
        file_path (str): Path to the MATLAB file.

    Returns:
        NeighboringParticles: Dataclass of neighboring particles.
    """
    data = loadmat(file_path)

    if (
        "left" not in data
        or "right" not in data
        or "top" not in data
        or "bottom" not in data
    ):
        raise ValueError(
            "The MATLAB file must contain at least "
            "'left', 'right', 'top' and 'bottom' keys. ('front' and 'back' for 3D)"
        )

    left = np.asarray(data["left"])
    right = np.asarray(data["right"])
    top = np.asarray(data["top"])
    bottom = np.asarray(data["bottom"])

    if "front" in data and "back" in data:
        front = np.asarray(data["front"])
        back = np.asarray(data["back"])
        positions = np.stack((left, right, top, bottom, front, back), axis=1)
        positions = positions.reshape(
            -1, 3, order="F"
        )  # Convert (N, 4, 2) → (4*N, 2) | (N, 6, 3) → (6*N, 3)
    else:
        positions = np.stack((left, right, top, bottom), axis=1)
        positions = positions.reshape(
            -1, 2, order="F"
        )  # Convert (N, 4, 2) → (4*N, 2) | (N, 6, 3) → (6*N, 3)

    return NeighboringParticles(positions=positions)

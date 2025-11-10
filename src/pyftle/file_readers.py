"""
File readers for velocity, coordinate, and seed particle data.

This module provides utilities for loading MATLAB `.mat` files containing
velocity fields, spatial coordinates, and seed particle positions used in
FTLE (Finite-Time Lyapunov Exponent) computations.

Supported data layouts
----------------------
Each `.mat` file must contain specific variable names:

- **Velocity data**:
  - 2D: `velocity_x`, `velocity_y`
  - 3D: `velocity_x`, `velocity_y`, `velocity_z` (optional)

- **Coordinate data**:
  - 2D: `coordinate_x`, `coordinate_y`
  - 3D: `coordinate_x`, `coordinate_y`, `coordinate_z` (optional)

- **Seed particle data**:
  - 2D: `left`, `right`, `top`, `bottom`
  - 3D: `left`, `right`, `top`, `bottom`, `front`, `back`

All arrays are automatically flattened and stacked in column-major order to
produce consistent NumPy arrays compatible with `pyftle`'s internal types.
"""

import numpy as np
from scipy.io import loadmat

from pyftle.my_types import Array2xN, Array3xN
from pyftle.particles import NeighboringParticles


def read_velocity(file_path: str) -> Array2xN | Array3xN:
    """
    Read velocity components from a MATLAB `.mat` file.

    The file must contain at least the keys `'velocity_x'` and `'velocity_y'`.
    If `'velocity_z'` is also present, the velocity is treated as 3D.

    Parameters
    ----------
    file_path : str
        Path to the MATLAB file containing the velocity data.

    Returns
    -------
    velocities : Array2xN or Array3xN
        A 2xN or 3xN array where each row corresponds to one velocity
        component (`x`, `y`, and optionally `z`), and `N` is the number
        of spatial grid points.

    Raises
    ------
    ValueError
        If the required keys (`velocity_x`, `velocity_y`) are not present
        in the `.mat` file.
    """
    data = loadmat(file_path)

    if "velocity_x" not in data or "velocity_y" not in data:
        raise ValueError(
            "The MATLAB file does not contain the expected keys 'velocity_x' and"
            "'velocity_y'."
        )

    velocity_x = np.asarray(data["velocity_x"]).ravel()
    velocity_y = np.asarray(data["velocity_y"]).ravel()

    if "velocity_z" in data:
        velocity_z = np.asarray(data["velocity_z"]).ravel()

        return np.stack((velocity_x, velocity_y, velocity_z))

    return np.stack((velocity_x, velocity_y))


def read_coordinate(file_path: str) -> Array2xN | Array3xN:
    """
    Read coordinate grid data from a MATLAB `.mat` file.

    The file must contain at least the keys `'coordinate_x'` and `'coordinate_y'`.
    If `'coordinate_z'` is present, the coordinates are treated as 3D.

    Parameters
    ----------
    file_path : str
        Path to the MATLAB file containing coordinate data.

    Returns
    -------
    coordinates : Array2xN or Array3xN
        A 2xN or 3xN array where each row corresponds to one spatial
        component (`x`, `y`, and optionally `z`), and `N` is the number
        of grid points.

    Raises
    ------
    ValueError
        If the required coordinate keys are missing from the file.
    """
    data = loadmat(file_path)

    if "coordinate_x" not in data or "coordinate_y" not in data:
        raise ValueError(
            "The MATLAB file does not contain the expected keys 'coordinate_x' and"
            "'coordinate_y'."
        )

    coordinate_x = np.asarray(data["coordinate_x"]).ravel()
    coordinate_y = np.asarray(data["coordinate_y"]).ravel()

    if "coordinate_z" in data:
        coordinate_z = np.asarray(data["coordinate_z"]).ravel()
        return np.stack((coordinate_x, coordinate_y, coordinate_z))

    return np.stack((coordinate_x, coordinate_y))


def read_seed_particles_coordinates(file_path: str) -> NeighboringParticles:
    """
    Read seed particle coordinates from a MATLAB `.mat` file and return a
    `NeighboringParticles` dataclass instance.

    The MATLAB file must contain at least the 2D neighbor keys:
    `'left'`, `'right'`, `'top'`, `'bottom'`.

    If `'front'` and `'back'` are also present, the dataset is interpreted as 3D.

    Parameters
    ----------
    file_path : str
        Path to the MATLAB file containing seed particle data.

    Returns
    -------
    particles : NeighboringParticles
        Dataclass representing the grouped particle positions. The total shape is:
        - (4xN, 2) for 2D datasets (left, right, top, bottom)
        - (6xN, 3) for 3D datasets (left, right, top, bottom, front, back)

    Raises
    ------
    ValueError
        If the required neighbor keys are missing from the `.mat` file.

    Notes
    -----
    The returned `NeighboringParticles` object contains the concatenated
    coordinates of all neighboring particles used to compute flow map
    Jacobians and FTLE fields.
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
        positions = np.concatenate(
            (left, right, top, bottom, front, back),  # shape (6*N, 3)
            axis=0,
        )
    else:
        positions = np.concatenate((left, right, top, bottom), axis=0)  # shape (4*N, 2)

    return NeighboringParticles(positions=positions)

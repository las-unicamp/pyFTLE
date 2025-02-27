import numpy as np
from scipy.io import loadmat

from src.my_types import ArrayFloat64MxN, ArrayFloat64Nx2
from src.particles import NeighboringParticles


class VelocityDataReader:
    def read_raw(self, file_path: str) -> tuple[ArrayFloat64MxN, ArrayFloat64MxN]:
        """
        Reads velocity data from a MATLAB file and returns it as a tuple of numpy
        arrays (grids of velocity_x and velocity_y) with shapes [M, N].

        Args:
            file_path (str): Path to the MATLAB file.

        Returns:
            tuple[ArrayFloat64MxN, ArrayFloat64MxN]: Tuple of arrays of shape [M, N].
        """
        data = loadmat(file_path)

        if "velocity_x" not in data or "velocity_y" not in data:
            raise ValueError(
                "The MATLAB file does not contain the expected keys 'velocity_x' and"
                "'velocity_y'."
            )

        velocity_x: ArrayFloat64MxN = np.asarray(data["velocity_x"], dtype=np.float64)
        velocity_y: ArrayFloat64MxN = np.asarray(data["velocity_y"], dtype=np.float64)

        return velocity_x, velocity_y

    def read_flatten(self, file_path: str) -> ArrayFloat64Nx2:
        """
        Reads velocity data from a MATLAB file and returns it as a numpy array
        with shape [n_points, 2] (velocity_x, velocity_y).

        Args:
            file_path (str): Path to the MATLAB file.

        Returns:
            ArrayFloat64Nx2: Array of shape [n_points, 2].
        """
        data = loadmat(file_path)

        if "velocity_x" not in data or "velocity_y" not in data:
            raise ValueError(
                "The MATLAB file does not contain the expected keys 'velocity_x' and"
                "'velocity_y'."
            )

        velocity_x = np.asarray(data["velocity_x"]).flatten().astype(np.float64)
        velocity_y = np.asarray(data["velocity_y"]).flatten().astype(np.float64)

        return np.column_stack((velocity_x, velocity_y))


class CoordinateDataReader:
    def read_raw(self, file_path: str) -> tuple[ArrayFloat64MxN, ArrayFloat64MxN]:
        """
        Reads coordinate data from a MATLAB file and returns it as a tuple of numpy
        arrays (grids of coordinate_x and coordinate_y) with shapes [M, N].

        Args:
            file_path (str): Path to the MATLAB file.

        Returns:
            tuple[ArrayFloat64MxN, ArrayFloat64MxN]: Tuple of arrays of shape [M, N].
        """
        data = loadmat(file_path)

        if "coordinate_x" not in data or "coordinate_y" not in data:
            raise ValueError(
                "The MATLAB file does not contain the expected keys 'coordinate_x' and"
                "'coordinate_y'."
            )

        coordinate_x: ArrayFloat64MxN = np.asarray(
            data["coordinate_x"], dtype=np.float64
        )
        coordinate_y: ArrayFloat64MxN = np.asarray(
            data["coordinate_y"], dtype=np.float64
        )

        return coordinate_x, coordinate_y

    def read_flatten(self, file_path: str) -> ArrayFloat64Nx2:
        """
        Reads coordinate data from a MATLAB file and returns it as a numpy array
        with shape [n_points, 2] (coordinate_x, coordinate_y).

        Args:
            file_path (str): Path to the MATLAB file.

        Returns:
            np.ndarray: Array of shape [n_points, 2].
        """
        data = loadmat(file_path)

        if "coordinate_x" not in data or "coordinate_y" not in data:
            raise ValueError(
                "The MATLAB file does not contain the expected keys 'coordinate_x' and"
                "'coordinate_y'."
            )

        coordinate_x = np.asarray(data["coordinate_x"]).flatten().astype(np.float64)
        coordinate_y = np.asarray(data["coordinate_y"]).flatten().astype(np.float64)

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
            "The MATLAB file must contain 'left', 'right', 'top', and 'bottom' keys."
        )

    left = np.asarray(data["left"])
    right = np.asarray(data["right"])
    top = np.asarray(data["top"])
    bottom = np.asarray(data["bottom"])

    positions = np.stack((left, right, top, bottom), axis=1)
    positions = positions.reshape(-1, 2, order="F")  # Convert (N, 4, 2) → (4*N, 2)

    return NeighboringParticles(positions=positions)

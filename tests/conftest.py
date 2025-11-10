from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.io import savemat

from pyftle.interpolate import Interpolator
from pyftle.my_types import Array2xN, Array3xN, Array4Nx2, Array6Nx3
from pyftle.particles import NeighboringParticles


def create_mock_matlab_file(file_path, data):
    """Utility to write a MATLAB .mat file with given data.

    Args:
        file_path (Path): The path to the .mat file.
        data (dict): A dictionary where keys are variable names and values are numpy arrays.
    """
    savemat(file_path, data)


@pytest.fixture
def mock_velocity_file_2d(tmp_path):
    """Fixture to create a mock 2D velocity .mat file.

    Args:
        tmp_path (Path): pytest fixture for a temporary directory.

    Returns:
        Path: The path to the created mock 2D velocity file.

    Flow:
        tmp_path -> create_mock_matlab_file -> file_path
    """
    file_path = tmp_path / "velocity_2d.mat"
    data = {
        "velocity_x": np.array([1.0, 2.0, 3.0]),
        "velocity_y": np.array([4.0, 5.0, 6.0]),
    }
    create_mock_matlab_file(file_path, data)
    return file_path


@pytest.fixture
def mock_velocity_file_3d(tmp_path):
    """Fixture to create a mock 3D velocity .mat file.

    Args:
        tmp_path (Path): pytest fixture for a temporary directory.

    Returns:
        Path: The path to the created mock 3D velocity file.

    Flow:
        tmp_path -> create_mock_matlab_file -> file_path
    """
    file_path = tmp_path / "velocity_3d.mat"
    data = {
        "velocity_x": np.array([1.0, 2.0, 3.0]),
        "velocity_y": np.array([4.0, 5.0, 6.0]),
        "velocity_z": np.array([7.0, 8.0, 9.0]),
    }
    create_mock_matlab_file(file_path, data)
    return file_path


@pytest.fixture
def mock_coordinate_file_2d(tmp_path):
    """Fixture to create a mock 2D coordinate .mat file.

    Args:
        tmp_path (Path): pytest fixture for a temporary directory.

    Returns:
        Path: The path to the created mock 2D coordinate file.

    Flow:
        tmp_path -> create_mock_matlab_file -> file_path
    """
    file_path = tmp_path / "coordinate_2d.mat"
    data = {
        "coordinate_x": np.array([10.0, 11.0, 12.0]),
        "coordinate_y": np.array([13.0, 14.0, 15.0]),
    }
    create_mock_matlab_file(file_path, data)
    return file_path


@pytest.fixture
def mock_coordinate_file_3d(tmp_path):
    """Fixture to create a mock 3D coordinate .mat file.

    Args:
        tmp_path (Path): pytest fixture for a temporary directory.

    Returns:
        Path: The path to the created mock 3D coordinate file.

    Flow:
        tmp_path -> create_mock_matlab_file -> file_path
    """
    file_path = tmp_path / "coordinate_3d.mat"
    data = {
        "coordinate_x": np.array([10.0, 11.0, 12.0]),
        "coordinate_y": np.array([13.0, 14.0, 15.0]),
        "coordinate_z": np.array([16.0, 17.0, 18.0]),
    }
    create_mock_matlab_file(file_path, data)
    return file_path


@pytest.fixture
def mock_seed_particle_file_2d(tmp_path):
    """Fixture to create a mock 2D seed particle .mat file.

    Args:
        tmp_path (Path): pytest fixture for a temporary directory.

    Returns:
        Path: The path to the created mock 2D seed particle file.

    Flow:
        tmp_path -> create_mock_matlab_file -> file_path
    """
    file_path = tmp_path / "seed_particles_2d.mat"
    data = {
        "top": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "bottom": np.array([[5.0, 6.0], [7.0, 8.0]]),
        "left": np.array([[9.0, 10.0], [11.0, 12.0]]),
        "right": np.array([[13.0, 14.0], [15.0, 16.0]]),
    }
    create_mock_matlab_file(file_path, data)
    return file_path


@pytest.fixture
def mock_seed_particle_file_3d(tmp_path):
    """Fixture to create a mock 3D seed particle .mat file.

    Args:
        tmp_path (Path): pytest fixture for a temporary directory.

    Returns:
        Path: The path to the created mock 3D seed particle file.

    Flow:
        tmp_path -> create_mock_matlab_file -> file_path
    """
    file_path = tmp_path / "seed_particles_3d.mat"
    data = {
        "top": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "bottom": np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]),
        "left": np.array([[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]),
        "right": np.array([[19.0, 20.0, 21.0], [22.0, 23.0, 24.0]]),
        "front": np.array([[25.0, 26.0, 27.0], [28.0, 29.0, 30.0]]),
        "back": np.array([[31.0, 32.0, 33.0], [34.0, 35.0, 36.0]]),
    }
    create_mock_matlab_file(file_path, data)
    return file_path


@pytest.fixture
def generate_2x2_jacobians() -> Array2xN:
    """Fixture to generate a small array of 2x2 Jacobians for testing.

    Returns:
        Array2xN: A numpy array of 2x2 Jacobians.

    Flow:
        None -> numpy array of 2x2 Jacobians
    """
    return np.array(
        [
            [[2.0, 0.0], [0.0, 1.0]],
            [[1.5, 0.5], [0.5, 1.5]],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def generate_3x3_jacobians() -> Array3xN:
    """Fixture to generate a small array of 3x3 Jacobians for testing.

    Returns:
        Array3xN: A numpy array of 3x3 Jacobians.

    Flow:
        None -> numpy array of 3x3 Jacobians
    """
    return np.array(
        [
            np.eye(3),
            [[1.0, 0.2, 0.0], [0.0, 1.5, 0.1], [0.0, 0.0, 1.2]],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def mock_interpolator():
    """Fixture to create a mock Interpolator object.

    Returns:
        MagicMock: A mock Interpolator object with a fake velocity field.

    Flow:
        None -> MagicMock(Interpolator)
    """
    mock = MagicMock(spec=Interpolator)
    mock.interpolate.side_effect = lambda x: x * 0.1
    return mock


@pytest.fixture
def initial_conditions():
    """Fixture to create a NeighboringParticles object with predefined initial positions.

    Returns:
        NeighboringParticles: An instance of NeighboringParticles with mock positions.

    Flow:
        None -> NeighboringParticles
    """
    positions = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
            [13.0, 14.0],
            [15.0, 16.0],
        ]
    )
    return NeighboringParticles(positions=positions)


@pytest.fixture
def generate_mock_data_2d() -> tuple[Array2xN, Array2xN]:
    """Fixture to generate mock 2D points and velocities.

    Returns:
        tuple[Array2xN, Array2xN]: A tuple containing mock 2D points and velocities.

    Flow:
        None -> (mock 2D points, mock 2D velocities)
    """
    points = np.array(
        [
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    velocities = np.array(
        [
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    return points, velocities


@pytest.fixture
def generate_mock_data_3d() -> tuple[Array3xN, Array3xN]:
    """Fixture to generate mock 3D points and velocities.

    Returns:
        tuple[Array3xN, Array3xN]: A tuple containing mock 3D points and velocities.

    Flow:
        None -> (mock 3D points, mock 3D velocities)
    """
    points = np.array(
        [
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    velocities = np.array(
        [
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    return points, velocities


@pytest.fixture(params=["2D", "3D"])
def sample_particles(request):
    """Fixture to create a NeighboringParticles object for 2D or 3D test cases.

    Args:
        request (FixtureRequest): Pytest's fixture request object to access parameters.

    Returns:
        NeighboringParticles: An instance of NeighboringParticles with mock positions
                              based on the '2D' or '3D' parameter.

    Flow:
        request.param ('2D' or '3D') -> numpy array of positions -> NeighboringParticles
    """
    if request.param == "2D":
        positions: Array4Nx2 = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [0.5, 0.0],
            ],
            dtype=Array4Nx2,
        )
    else:
        positions: Array6Nx3 = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, -1.0, 0.0],
                [0.5, 0.0, 1.0],
                [0.5, 0.0, -1.0],
            ],
            dtype=Array6Nx3,
        )
    return NeighboringParticles(positions=positions)
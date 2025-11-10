import numpy as np
import pytest
from scipy.io import savemat
from unittest.mock import MagicMock

from pyftle.interpolate import Interpolator
from pyftle.my_types import Array2xN, Array3xN, Array4Nx2, Array6Nx3
from pyftle.particles import NeighboringParticles


# ==========================================================
# Helper functions for creating mock files
# ==========================================================
def create_mock_matlab_file(file_path, data):
    """Utility to write a MATLAB .mat file with given data."""
    savemat(file_path, data)


# ==========================================================
# Fixtures for file readers
# ==========================================================
@pytest.fixture
def mock_velocity_file_2d(tmp_path):
    file_path = tmp_path / "velocity_2d.mat"
    data = {
        "velocity_x": np.array([1.0, 2.0, 3.0]),
        "velocity_y": np.array([4.0, 5.0, 6.0]),
    }
    create_mock_matlab_file(file_path, data)
    return file_path


@pytest.fixture
def mock_velocity_file_3d(tmp_path):
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
    file_path = tmp_path / "coordinate_2d.mat"
    data = {
        "coordinate_x": np.array([10.0, 11.0, 12.0]),
        "coordinate_y": np.array([13.0, 14.0, 15.0]),
    }
    create_mock_matlab_file(file_path, data)
    return file_path


@pytest.fixture
def mock_coordinate_file_3d(tmp_path):
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


# ==========================================================
# Fixtures for FTLE calculations
# ==========================================================
@pytest.fixture
def generate_2x2_jacobians() -> Array2xN:
    """Return a small array of 2×2 Jacobians for testing."""
    return np.array(
        [
            [[2.0, 0.0], [0.0, 1.0]],
            [[1.5, 0.5], [0.5, 1.5]],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def generate_3x3_jacobians() -> Array3xN:
    """Return a small array of 3×3 Jacobians for testing."""
    return np.array(
        [
            np.eye(3),
            [[1.0, 0.2, 0.0], [0.0, 1.5, 0.1], [0.0, 0.0, 1.2]],
        ],
        dtype=np.float64,
    )


# ==========================================================
# Fixtures for Integrator tests
# ==========================================================
@pytest.fixture
def mock_interpolator():
    mock = MagicMock(spec=Interpolator)
    mock.interpolate.side_effect = lambda x: x * 0.1  # Fake velocity field
    return mock


@pytest.fixture
def initial_conditions():
    # Create a positions array with shape (4 * N, 2)
    positions = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],  # Left neighbors
            [5.0, 6.0],
            [7.0, 8.0],  # Right neighbors
            [9.0, 10.0],
            [11.0, 12.0],  # Top neighbors
            [13.0, 14.0],
            [15.0, 16.0],  # Bottom neighbors
        ]
    )
    return NeighboringParticles(positions=positions)


# ==========================================================
# Fixtures for Interpolator tests
# ==========================================================
@pytest.fixture
def generate_mock_data_2d() -> tuple[Array2xN, Array2xN]:
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


# ==========================================================
# Fixtures for NeighboringParticles tests
# ==========================================================
@pytest.fixture(params=["2D", "3D"])
def sample_particles(request):
    """Creates a NeighboringParticles object for 2D or 3D test cases."""
    if request.param == "2D":
        positions: Array4Nx2 = np.array(
            [
                [0.0, 0.0],  # Left neighbor
                [1.0, 0.0],  # Right neighbor
                [0.5, 1.0],  # Top neighbor
                [0.5, 0.0],  # Bottom neighbor
            ],
            dtype=Array4Nx2,
        )
    else:  # 3D
        positions: Array6Nx3 = np.array(
            [
                [0.0, 0.0, 0.0],  # Left
                [1.0, 0.0, 0.0],  # Right
                [0.5, 1.0, 0.0],  # Top
                [0.5, -1.0, 0.0],  # Bottom
                [0.5, 0.0, 1.0],  # Front
                [0.5, 0.0, -1.0],  # Back
            ],
            dtype=Array6Nx3,
        )
    return NeighboringParticles(positions=positions)

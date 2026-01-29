"""Shared pytest fixtures and helpers for pyFTLE tests.

Fixtures defined here are automatically discovered by pytest and available
to all test modules in this directory and subdirectories.
"""

import numpy as np
import pytest
from scipy.io import savemat

from pyftle.my_types import ArrayNx2x2, ArrayNx3x3
from pyftle.particles import NeighboringParticles

# ---------------------------------------------------------------------------
# Helpers (used by fixtures or tests)
# ---------------------------------------------------------------------------


def create_mock_matlab_file(file_path, data):
    """Write a MATLAB .mat file with the given data."""
    savemat(file_path, data)


# ---------------------------------------------------------------------------
# File reader fixtures (2D/3D velocity, coordinate, seed particle files)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_velocity_file_2d(tmp_path):
    """Path to a temporary 2D velocity .mat file."""
    file_path = tmp_path / "velocity_2d.mat"
    data = {
        "velocity_x": np.array([1.0, 2.0, 3.0]),
        "velocity_y": np.array([4.0, 5.0, 6.0]),
    }
    create_mock_matlab_file(file_path, data)
    return file_path


@pytest.fixture
def mock_velocity_file_3d(tmp_path):
    """Path to a temporary 3D velocity .mat file."""
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
    """Path to a temporary 2D coordinate .mat file."""
    file_path = tmp_path / "coordinate_2d.mat"
    data = {
        "coordinate_x": np.array([10.0, 11.0, 12.0]),
        "coordinate_y": np.array([13.0, 14.0, 15.0]),
    }
    create_mock_matlab_file(file_path, data)
    return file_path


@pytest.fixture
def mock_coordinate_file_3d(tmp_path):
    """Path to a temporary 3D coordinate .mat file."""
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
    """Path to a temporary 2D seed particles .mat file."""
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
    """Path to a temporary 3D seed particles .mat file."""
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


# ---------------------------------------------------------------------------
# File writer fixtures (particle centroids and FTLE fields)
# ---------------------------------------------------------------------------


@pytest.fixture
def particles_centroid_2d():
    """Sample 2D particle centroids (4 points)."""
    return np.array([[0, 0], [1, 0], [0, 1], [1, 1]])


@pytest.fixture
def particles_centroid_3d():
    """Sample 3D particle centroids (8 points)."""
    return np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]
    )


@pytest.fixture
def ftle_field_2d():
    """Sample 2D FTLE field (4 values)."""
    return np.array([1.0, 2.0, 3.0, 4.0])


@pytest.fixture
def ftle_field_3d():
    """Sample 3D FTLE field (8 values)."""
    return np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)


# ---------------------------------------------------------------------------
# Integration / interpolator fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_interpolator():
    """Mock Interpolator with a simple velocity field (v = 0.1 * position)."""
    from unittest.mock import MagicMock

    from pyftle.interpolate import Interpolator

    mock = MagicMock(spec=Interpolator)
    mock.interpolate.side_effect = lambda x: x * 0.1
    return mock


@pytest.fixture
def initial_conditions():
    """NeighboringParticles with 8 positions (4 groups × 2) for integration tests."""
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


# ---------------------------------------------------------------------------
# FTLE test data (Jacobians)
# ---------------------------------------------------------------------------


@pytest.fixture
def jacobians_2x2() -> ArrayNx2x2:
    """Small array of 2×2 Jacobians for FTLE tests."""
    return np.array(
        [
            [[2.0, 0.0], [0.0, 1.0]],
            [[1.5, 0.5], [0.5, 1.5]],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def jacobians_3x3() -> ArrayNx3x3:
    """Small array of 3×3 Jacobians for FTLE tests."""
    return np.array(
        [
            np.eye(3),
            [[1.0, 0.2, 0.0], [0.0, 1.5, 0.1], [0.0, 0.0, 1.2]],
        ],
        dtype=np.float64,
    )

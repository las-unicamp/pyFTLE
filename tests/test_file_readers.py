import numpy as np
import pytest
from scipy.io import loadmat, savemat

from src.file_readers import (
    read_coordinates,
    read_seed_particles_coordinates,
    read_velocity_data,
)


# Helper to create a mock MATLAB file for testing
def create_mock_matlab_file(file_path, data):
    savemat(file_path, data)


@pytest.fixture
def mock_velocity_file(tmp_path):
    file_path = tmp_path / "velocity_data.mat"
    data = {
        "velocity_x": np.array([[1.0, 2.0, 3.0]]).T,
        "velocity_y": np.array([[4.0, 5.0, 6.0]]).T,
    }
    create_mock_matlab_file(file_path, data)
    return file_path


@pytest.fixture
def mock_coordinate_file(tmp_path):
    file_path = tmp_path / "coordinate_data.mat"
    data = {
        "coordinate_x": np.array([[7.0, 8.0, 9.0]]).T,
        "coordinate_y": np.array([[10.0, 11.0, 12.0]]).T,
    }
    create_mock_matlab_file(file_path, data)
    return file_path


def test_read_velocity_data(mock_velocity_file):
    expected = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    result = read_velocity_data(mock_velocity_file)
    assert result.shape == (3, 2)
    np.testing.assert_array_equal(result, expected)


def test_read_coordinates(mock_coordinate_file):
    expected = np.array([[7.0, 10.0], [8.0, 11.0], [9.0, 12.0]])
    result = read_coordinates(mock_coordinate_file)
    assert result.shape == (3, 2)
    np.testing.assert_array_equal(result, expected)


@pytest.fixture
def mock_seed_particle_file(tmp_path):
    file_path = tmp_path / "seed_particles.mat"
    data = {
        "top": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "bottom": np.array([[5.0, 6.0], [7.0, 8.0]]),
        "left": np.array([[9.0, 10.0], [11.0, 12.0]]),
        "right": np.array([[13.0, 14.0], [15.0, 16.0]]),
    }
    create_mock_matlab_file(file_path, data)
    return file_path


def test_read_seed_particles_coordinates(mock_seed_particle_file):
    result = read_seed_particles_coordinates(mock_seed_particle_file)

    # Expected output after reshaping to (4*N, 2)
    expected_positions = np.array(
        [
            # Left
            [9.0, 10.0],
            [11.0, 12.0],
            # Right
            [13.0, 14.0],
            [15.0, 16.0],
            # Top
            [1.0, 2.0],
            [3.0, 4.0],
            # Bottom
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )

    # Ensure positions match
    np.testing.assert_array_equal(result.positions, expected_positions)


def test_read_seed_particles_coordinates_caching(mock_seed_particle_file, mocker):
    # Patch loadmat in the src.file_readers namespace
    mocked_loadmat = mocker.patch("src.file_readers.loadmat", wraps=loadmat)

    # Call the function twice with the same file
    result1 = read_seed_particles_coordinates(mock_seed_particle_file)
    result2 = read_seed_particles_coordinates(mock_seed_particle_file)

    # Verify the results are the same
    np.testing.assert_array_equal(result1, result2)

    # Ensure loadmat is only called once (subsequent calls used the cache)
    read_seed_particles_coordinates(mock_seed_particle_file)
    read_seed_particles_coordinates(mock_seed_particle_file)
    read_seed_particles_coordinates(mock_seed_particle_file)
    read_seed_particles_coordinates(mock_seed_particle_file)
    mocked_loadmat.assert_called_once_with(mock_seed_particle_file)

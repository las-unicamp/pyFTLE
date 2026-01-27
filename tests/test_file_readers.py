import numpy as np

from pyftle.file_readers import (
    read_coordinate,
    read_seed_particles_coordinates,
    read_velocity,
)


# ==========================================================
# Velocity Reader Tests
# ==========================================================
def test_read_velocity_2d(mock_velocity_file_2d):
    result = read_velocity(str(mock_velocity_file_2d))
    expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (2, 3)  # Shape is (2, N) for 2D


def test_read_velocity_3d(mock_velocity_file_3d):
    result = read_velocity(str(mock_velocity_file_3d))
    expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (3, 3)  # Shape is (3, N) for 3D


# ==========================================================
# Coordinate Reader Tests
# ==========================================================
def test_read_coordinate_2d(mock_coordinate_file_2d):
    result = read_coordinate(str(mock_coordinate_file_2d))
    expected = np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]])
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (2, 3)  # Shape is (2, N) for 2D


def test_read_coordinate_3d(mock_coordinate_file_3d):
    result = read_coordinate(str(mock_coordinate_file_3d))
    expected = np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]])
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (3, 3)  # Shape is (3, N) for 3D


# ==========================================================
# Seed Particle Reader Tests (2D & 3D)
# ==========================================================
def test_read_seed_particles_coordinates_2d(mock_seed_particle_file_2d):
    result = read_seed_particles_coordinates(mock_seed_particle_file_2d)
    expected = np.array(
        [
            [9.0, 10.0],
            [11.0, 12.0],
            [13.0, 14.0],
            [15.0, 16.0],
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )
    np.testing.assert_array_equal(result.positions, expected)
    assert result.positions.shape == (8, 2)  # Shape is (4*N, 2) for 2D


def test_read_seed_particles_coordinates_3d(mock_seed_particle_file_3d):
    result = read_seed_particles_coordinates(mock_seed_particle_file_3d)
    expected = np.array(
        [
            # Left
            [13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0],
            # Right
            [19.0, 20.0, 21.0],
            [22.0, 23.0, 24.0],
            # Top
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            # Bottom
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            # Front
            [25.0, 26.0, 27.0],
            [28.0, 29.0, 30.0],
            # Back
            [31.0, 32.0, 33.0],
            [34.0, 35.0, 36.0],
        ]
    )
    np.testing.assert_array_equal(result.positions, expected)
    assert result.positions.shape == (12, 3)  # Shape is (6*N, 3) for 3D

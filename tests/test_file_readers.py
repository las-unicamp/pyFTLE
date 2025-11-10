import numpy as np

from pyftle.file_readers import (
    read_coordinate,
    read_seed_particles_coordinates,
    read_velocity,
)


def test_read_velocity_2d(mock_velocity_file_2d):
    """Tests reading 2D velocity data from a mock file.

    Args:
        mock_velocity_file_2d (Path): Path to the mock 2D velocity file.

    Flow:
        mock_velocity_file_2d -> read_velocity -> result
        result (shape 2x3) == expected (shape 2x3)
    """
    result = read_velocity(str(mock_velocity_file_2d))
    expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (2, 3)


def test_read_velocity_3d(mock_velocity_file_3d):
    """Tests reading 3D velocity data from a mock file.

    Args:
        mock_velocity_file_3d (Path): Path to the mock 3D velocity file.

    Flow:
        mock_velocity_file_3d -> read_velocity -> result
        result (shape 3x3) == expected (shape 3x3)
    """
    result = read_velocity(str(mock_velocity_file_3d))
    expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (3, 3)


def test_read_coordinate_2d(mock_coordinate_file_2d):
    """Tests reading 2D coordinate data from a mock file.

    Args:
        mock_coordinate_file_2d (Path): Path to the mock 2D coordinate file.

    Flow:
        mock_coordinate_file_2d -> read_coordinate -> result
        result (shape 2x3) == expected (shape 2x3)
    """
    result = read_coordinate(str(mock_coordinate_file_2d))
    expected = np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]])
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (2, 3)


def test_read_coordinate_3d(mock_coordinate_file_3d):
    """Tests reading 3D coordinate data from a mock file.

    Args:
        mock_coordinate_file_3d (Path): Path to the mock 3D coordinate file.

    Flow:
        mock_coordinate_file_3d -> read_coordinate -> result
        result (shape 3x3) == expected (shape 3x3)
    """
    result = read_coordinate(str(mock_coordinate_file_3d))
    expected = np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]])
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (3, 3)


def test_read_seed_particles_coordinates_2d(mock_seed_particle_file_2d):
    """Tests reading 2D seed particle coordinates from a mock file.

    Args:
        mock_seed_particle_file_2d (Path): Path to the mock 2D seed particle file.

    Flow:
        mock_seed_particle_file_2d -> read_seed_particles_coordinates -> result
        result.positions (shape 8x2) == expected (shape 8x2)
    """
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
    assert result.positions.shape == (8, 2)


def test_read_seed_particles_coordinates_3d(mock_seed_particle_file_3d):
    """Tests reading 3D seed particle coordinates from a mock file.

    Args:
        mock_seed_particle_file_3d (Path): Path to the mock 3D seed particle file.

    Flow:
        mock_seed_particle_file_3d -> read_seed_particles_coordinates -> result
        result.positions (shape 12x3) == expected (shape 12x3)
    """
    result = read_seed_particles_coordinates(mock_seed_particle_file_3d)
    expected = np.array(
        [
            [13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0],
            [19.0, 20.0, 21.0],
            [22.0, 23.0, 24.0],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [25.0, 26.0, 27.0],
            [28.0, 29.0, 30.0],
            [31.0, 32.0, 33.0],
            [34.0, 35.0, 36.0],
        ]
    )
    np.testing.assert_array_equal(result.positions, expected)
    assert result.positions.shape == (12, 3)

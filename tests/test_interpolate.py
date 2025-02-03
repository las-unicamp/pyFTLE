import numpy as np
import pytest
from scipy.interpolate import griddata

from src.interpolate import velocity_interpolation


@pytest.fixture
def mock_velocity_data():
    """Mock velocity data (n_points, 2)"""
    return np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 4.0], [3.0, 9.0]], dtype=np.float32)


@pytest.fixture
def mock_coordinates():
    """Mock coordinate data (n_points, 2)"""
    return np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]], dtype=np.float32)


@pytest.fixture
def mock_new_points():
    """Mock points to interpolate velocity data"""
    return np.array([[0.25, 0.25], [0.5, 0.5], [0.0, 0.5]], dtype=np.float32)


def test_velocity_interpolation(mock_velocity_data, mock_coordinates, mock_new_points):
    """
    Test velocity_interpolation with mock data.
    """
    # Call function
    result = velocity_interpolation(
        mock_velocity_data, mock_coordinates, mock_new_points
    )

    # Manually interpolate expected values using griddata
    expected_u = griddata(
        (mock_coordinates[:, 0], mock_coordinates[:, 1]),
        mock_velocity_data[:, 0],
        (mock_new_points[:, 0], mock_new_points[:, 1]),
        method="cubic",
        fill_value=0.0,
    )

    expected_v = griddata(
        (mock_coordinates[:, 0], mock_coordinates[:, 1]),
        mock_velocity_data[:, 1],
        (mock_new_points[:, 0], mock_new_points[:, 1]),
        method="cubic",
        fill_value=0.0,
    )

    expected = np.column_stack((expected_u, expected_v))

    # Assert shape
    assert result.shape == (mock_new_points.shape[0], 2)

    # Assert values
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_velocity_interpolation_empty_grid(mock_new_points):
    """
    Test velocity_interpolation with an empty grid (should return an error).
    """
    empty_velocities = np.empty((0, 2))
    empty_coordinates = np.empty((0, 2))

    with pytest.raises(ValueError, match="No points given"):
        velocity_interpolation(empty_velocities, empty_coordinates, mock_new_points)

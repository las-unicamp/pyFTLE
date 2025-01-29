from unittest.mock import patch

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


@patch("src.interpolate.read_velocity_data")
@patch("src.interpolate.read_coordinates")
def test_velocity_interpolation(
    mock_read_coordinates,
    mock_read_velocity_data,
    mock_velocity_data,
    mock_coordinates,
    mock_new_points,
):
    """
    Test velocity_interpolation with mock data.
    """

    # Mock return values of the file reading functions
    mock_read_velocity_data.return_value = mock_velocity_data
    mock_read_coordinates.return_value = mock_coordinates

    # Call function
    result = velocity_interpolation(
        "mock_velocity.mat", "mock_grid.mat", mock_new_points
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

    # Ensure file reading functions were called once
    mock_read_velocity_data.assert_called_once_with("mock_velocity.mat")
    mock_read_coordinates.assert_called_once_with("mock_grid.mat")


def test_velocity_interpolation_empty_grid(mock_new_points):
    """
    Test velocity_interpolation with an empty grid (should return zeros).
    """
    with (
        patch("src.interpolate.read_velocity_data", return_value=np.empty((0, 2))),
        patch("src.interpolate.read_coordinates", return_value=np.empty((0, 2))),
    ):
        with pytest.raises(ValueError, match="No points given"):
            velocity_interpolation(
                "empty_velocity.mat", "empty_grid.mat", mock_new_points
            )

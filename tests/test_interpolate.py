from unittest.mock import patch

import numpy as np
import pytest

from src.interpolate import (
    CubicInterpolatorStrategy,
    GridInterpolatorStrategy,
    LinearInterpolatorStrategy,
    NearestNeighborInterpolatorStrategy,
    create_interpolator,
)


def generate_mock_data():
    points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32
    )
    velocities = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32
    )
    return points, velocities


@pytest.mark.parametrize(
    "strategy_class",
    [
        CubicInterpolatorStrategy,
        LinearInterpolatorStrategy,
        NearestNeighborInterpolatorStrategy,
    ],
)
def test_interpolators(strategy_class):
    points, velocities = generate_mock_data()
    interpolator = strategy_class(points, velocities[:, 0], velocities[:, 1])

    new_points = np.array([[0.5, 0.5], [0.25, 0.75]], dtype=np.float32)
    interpolated_values = interpolator.interpolate(new_points)

    assert interpolated_values.shape == (new_points.shape[0], 2)
    assert np.isfinite(interpolated_values).all()


def test_grid_interpolator():
    grid_x = np.array([0.0, 1.0], dtype=np.float32)
    grid_y = np.array([0.0, 1.0], dtype=np.float32)
    velocities_u = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    velocities_v = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    interpolator = GridInterpolatorStrategy(grid_x, grid_y, velocities_u, velocities_v)

    new_points = np.array([[0.5, 0.5], [0.25, 0.75]], dtype=np.float32)
    interpolated_values = interpolator.interpolate(new_points)

    assert interpolated_values.shape == (new_points.shape[0], 2)
    assert np.isfinite(interpolated_values).all()


@patch("src.interpolate.read_velocity_data")
@patch("src.interpolate.read_coordinates")
def test_create_interpolator(mock_read_coordinates, mock_read_velocity_data):
    points, velocities = generate_mock_data()
    mock_read_coordinates.return_value = points
    mock_read_velocity_data.return_value = velocities

    for strategy in ["cubic", "linear", "nearest"]:
        interpolator = create_interpolator(
            "dummy_snapshot.mat", "dummy_grid.mat", strategy
        )
        new_points = np.array([[0.5, 0.5]], dtype=np.float32)
        interpolated_values = interpolator.interpolate(new_points)

        assert interpolated_values.shape == (new_points.shape[0], 2)
        assert np.isfinite(interpolated_values).all()


def test_caching():
    with (
        patch("src.interpolate.read_velocity_data") as mock_read_velocity_data,
        patch("src.interpolate.read_coordinates") as mock_read_coordinates,
    ):
        points, velocities = generate_mock_data()
        mock_read_coordinates.return_value = points
        mock_read_velocity_data.return_value = velocities

        interpolator_1 = create_interpolator(
            "dummy_snapshot.mat", "dummy_grid.mat", "cubic"
        )
        interpolator_2 = create_interpolator(
            "dummy_snapshot.mat", "dummy_grid.mat", "cubic"
        )

        assert interpolator_1 is interpolator_2  # Should return cached instance

        interpolator_3 = create_interpolator(
            "dummy_snapshot3.mat", "dummy_grid.mat", "cubic"
        )
        assert interpolator_1 is not interpolator_3  # Should create new instance


if __name__ == "__main__":
    pytest.main()

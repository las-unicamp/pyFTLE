from unittest.mock import patch

import numpy as np
import pytest

from src.file_readers import CoordinateDataReader, VelocityDataReader
from src.interpolate import (
    CubicInterpolatorStrategy,
    InterpolatorFactory,
    LinearInterpolatorStrategy,
    NearestNeighborInterpolatorStrategy,
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


# def test_grid_interpolator():
#     grid_x = np.array(
#         [[0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0]], dtype=np.float32
#     )
#     grid_y = np.array(
#         [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]], dtype=np.float32
#     )

#     velocities_u = np.array(
#         [[0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0]], dtype=np.float32
#     )
#     velocities_v = np.array(
#         [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]], dtype=np.float32
#     )

#     interpolator = GridInterpolatorStrategy(
#         (grid_x, grid_y), velocities_u, velocities_v
#     )

#     new_points = np.array([[0.5, 0.5], [0.25, 0.75]], dtype=np.float32)
#     interpolated_values = interpolator.interpolate(new_points)

#     assert interpolated_values.shape == (new_points.shape[0], 2)
#     assert np.isfinite(interpolated_values).all()


@patch("src.file_readers.CoordinateDataReader.read_flatten")
@patch("src.file_readers.VelocityDataReader.read_flatten")
def test_create_interpolator(mock_read_velocity, mock_read_coordinates):
    points, velocities = generate_mock_data()
    mock_read_coordinates.return_value = points
    mock_read_velocity.return_value = velocities

    factory = InterpolatorFactory(CoordinateDataReader(), VelocityDataReader())

    for strategy in ["cubic", "linear", "nearest"]:
        interpolator = factory.create_interpolator(
            "dummy_snapshot.mat", "dummy_grid.mat", strategy
        )
        new_points = np.array([[0.5, 0.5]], dtype=np.float32)
        interpolated_values = interpolator.interpolate(new_points)

        assert interpolated_values.shape == (new_points.shape[0], 2)
        assert np.isfinite(interpolated_values).all()


def test_caching():
    with (
        patch(
            "src.file_readers.CoordinateDataReader.read_flatten"
        ) as mock_read_coordinates,
        patch("src.file_readers.VelocityDataReader.read_flatten") as mock_read_velocity,
    ):
        points, velocities = generate_mock_data()
        mock_read_coordinates.return_value = points
        mock_read_velocity.return_value = velocities

        factory = InterpolatorFactory(CoordinateDataReader(), VelocityDataReader())
        interpolator_1 = factory.create_interpolator(
            "dummy_snapshot.mat", "dummy_grid.mat", "cubic"
        )
        interpolator_2 = factory.create_interpolator(
            "dummy_snapshot.mat", "dummy_grid.mat", "cubic"
        )

        assert interpolator_1 is interpolator_2  # Should return cached instance

        interpolator_3 = factory.create_interpolator(
            "dummy_snapshot3.mat", "dummy_grid.mat", "cubic"
        )
        assert interpolator_1 is not interpolator_3  # Should create a new instance


if __name__ == "__main__":
    pytest.main()

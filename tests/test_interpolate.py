from unittest.mock import Mock

import numpy as np
import pytest

from src.interpolate import (
    CubicInterpolatorStrategy,
    GridInterpolatorStrategy,
    InterpolatorFactory,
    LinearInterpolatorStrategy,
    NearestNeighborInterpolatorStrategy,
)
from src.my_types import (
    Array2xMxN,
    Array3xMxN,
    ArrayFloat64Nx2,
    ArrayFloat64Nx3,
)


# -----------------------
# Helper: mock data
# -----------------------
def generate_mock_data_2d() -> tuple[ArrayFloat64Nx2, ArrayFloat64Nx2]:
    points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    velocities = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    return points, velocities


def generate_mock_data_3d() -> tuple[ArrayFloat64Nx3, ArrayFloat64Nx3]:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    velocities = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    return points, velocities


# -----------------------
# Interpolator tests (2D)
# -----------------------
@pytest.mark.parametrize(
    "strategy_class",
    [
        CubicInterpolatorStrategy,
        LinearInterpolatorStrategy,
        NearestNeighborInterpolatorStrategy,
    ],
)
def test_interpolators_2d(strategy_class):
    points, velocities = generate_mock_data_2d()
    interpolator = strategy_class(points, velocities)

    new_points = np.array([[0.5, 0.5], [0.25, 0.75]], dtype=np.float64)
    interpolated_values = interpolator.interpolate(new_points)

    assert interpolated_values.shape == (new_points.shape[0], 2)
    assert np.isfinite(interpolated_values).all()


# -----------------------
# Interpolator tests (3D)
# -----------------------
@pytest.mark.parametrize(
    "strategy_class",
    [
        LinearInterpolatorStrategy,
        NearestNeighborInterpolatorStrategy,
    ],
)
def test_interpolators_3d(strategy_class):
    points, velocities = generate_mock_data_3d()
    interpolator = strategy_class(points, velocities)

    new_points = np.array([[0.5, 0.5, 0.5], [0.25, 0.25, 0.75]], dtype=np.float64)
    interpolated_values = interpolator.interpolate(new_points)

    assert interpolated_values.shape == (new_points.shape[0], 3)
    assert np.isfinite(interpolated_values).all()


# -----------------------
# GridInterpolator tests
# -----------------------
def test_grid_interpolator_2d():
    grid_x, grid_y = np.mgrid[0:1:3j, 0:1:3j]
    velocities_u = np.copy(grid_x)
    velocities_v = np.copy(grid_y)

    coordinates: Array2xMxN = np.array([grid_x, grid_y], dtype=np.float64)
    velocities: Array2xMxN = np.array([velocities_u, velocities_v], dtype=np.float64)

    interpolator = GridInterpolatorStrategy(coordinates, velocities)
    new_points = np.array([[0.5, 0.5], [0.25, 0.75]], dtype=np.float64)
    interpolated_values = interpolator.interpolate(new_points)

    assert interpolated_values.shape == (new_points.shape[0], 2)
    assert np.isfinite(interpolated_values).all()


def test_grid_interpolator_3d():
    grid_x, grid_y, grid_z = np.mgrid[0:1:3j, 0:1:3j, 0:1:3j]
    velocities_u = np.copy(grid_x)
    velocities_v = np.copy(grid_y)
    velocities_w = np.copy(grid_z)

    coordinates: Array3xMxN = np.array([grid_x, grid_y, grid_z], dtype=np.float64)
    velocities: Array3xMxN = np.array(
        [velocities_u, velocities_v, velocities_w], dtype=np.float64
    )

    interpolator = GridInterpolatorStrategy(coordinates, velocities)
    new_points = np.array([[0.5, 0.5, 0.5], [0.25, 0.25, 0.75]], dtype=np.float64)
    interpolated_values = interpolator.interpolate(new_points)

    assert interpolated_values.shape == (new_points.shape[0], 3)
    assert np.isfinite(interpolated_values).all()


# -----------------------
# Factory tests
# -----------------------
def test_create_interpolator_flattened():
    points, velocities = generate_mock_data_2d()

    # Create mocks that follow the protocol
    mock_coordinate_reader = Mock()
    mock_velocity_reader = Mock()

    mock_coordinate_reader.read_flatten.return_value = points
    mock_velocity_reader.read_flatten.return_value = velocities

    # Inject mocks into the factory instead of the concrete classes
    factory = InterpolatorFactory(mock_coordinate_reader, mock_velocity_reader)

    for strategy in ["cubic", "linear", "nearest"]:
        interpolator = factory.create_interpolator(
            "dummy_snapshot.mat", "dummy_grid.mat", strategy
        )
        new_points = np.array([[0.5, 0.5]], dtype=np.float64)
        interpolated_values = interpolator.interpolate(new_points)
        assert interpolated_values.shape == (1, 2)
        assert np.isfinite(interpolated_values).all()


def test_create_interpolator_grid():
    grid_x, grid_y = np.mgrid[0:1:3j, 0:1:3j]
    velocities_u = np.copy(grid_x)
    velocities_v = np.copy(grid_y)

    coordinates = np.array([grid_x, grid_y], dtype=np.float64)
    velocities = np.array([velocities_u, velocities_v], dtype=np.float64)

    # Create mocks that follow the protocol
    mock_coordinate_reader = Mock()
    mock_velocity_reader = Mock()

    # For the grid strategy, the factory calls `.read_raw()`, not `.read_flatten()`
    mock_coordinate_reader.read_raw.return_value = coordinates
    mock_velocity_reader.read_raw.return_value = velocities

    factory = InterpolatorFactory(mock_coordinate_reader, mock_velocity_reader)

    interpolator = factory.create_interpolator(
        "dummy_snapshot.mat", "dummy_grid.mat", "grid"
    )

    new_points = np.array([[0.5, 0.5], [0.1, 0.9]], dtype=np.float64)
    interpolated_values = interpolator.interpolate(new_points)

    assert interpolated_values.shape == (new_points.shape[0], 2)
    assert np.isfinite(interpolated_values).all()


def test_invalid_strategy_raises():
    """Ensure ValueError is raised for an unknown interpolation strategy."""
    mock_coordinate_reader = Mock()
    mock_velocity_reader = Mock()

    factory = InterpolatorFactory(mock_coordinate_reader, mock_velocity_reader)

    with pytest.raises(ValueError, match="Unknown interpolation strategy"):
        factory.create_interpolator(
            "dummy_snapshot.mat", "dummy_grid.mat", "invalid_strategy"
        )

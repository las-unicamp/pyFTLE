import numpy as np
import pytest

from pyftle.interpolate import (
    CubicInterpolator,
    GridInterpolator,
    LinearInterpolator,
    NearestNeighborInterpolator,
    create_interpolator,
)


@pytest.mark.parametrize(
    "strategy_class",
    [
        CubicInterpolator,
        LinearInterpolator,
        NearestNeighborInterpolator,
    ],
)
def test_interpolators_2d(strategy_class, generate_mock_data_2d):
    """Tests various 2D interpolator strategies.

    Args:
        strategy_class (type): The interpolator class to test.
        generate_mock_data_2d (tuple): Fixture providing mock 2D points and
            velocities.

    Flow:
        generate_mock_data_2d -> interpolator.update -> interpolator.interpolate
        -> interpolated_values
        interpolated_values shape == expected shape
        All interpolated_values are finite.
    """
    points, velocities = generate_mock_data_2d
    interpolator = strategy_class()
    interpolator.update(velocities, points)

    new_points = np.array([[0.5, 0.5], [0.25, 0.75]])
    interpolated_values = interpolator.interpolate(new_points)

    assert interpolated_values.shape == (new_points.shape[0], 2)
    assert np.isfinite(interpolated_values).all()


@pytest.mark.parametrize(
    "strategy_class",
    [
        LinearInterpolator,
        NearestNeighborInterpolator,
    ],
)
def test_interpolators_3d(strategy_class, generate_mock_data_3d):
    """Tests various 3D interpolator strategies.

    Args:
        strategy_class (type): The interpolator class to test.
        generate_mock_data_3d (tuple): Fixture providing mock 3D points and
            velocities.

    Flow:
        generate_mock_data_3d -> interpolator.update -> interpolator.interpolate
        -> interpolated_values
        interpolated_values shape == expected shape
        All interpolated_values are finite.
    """
    points, velocities = generate_mock_data_3d
    interpolator = strategy_class()
    interpolator.update(velocities, points)

    new_points = np.array([[0.5, 0.5, 0.5], [0.25, 0.25, 0.75]])
    interpolated_values = interpolator.interpolate(new_points)

    assert interpolated_values.shape == (new_points.shape[0], 3)
    assert np.isfinite(interpolated_values).all()


def test_grid_interpolator_2d():
    """Tests the 2D GridInterpolator.

    Flow:
        Generate 2D grid data -> GridInterpolator initialized
        -> interpolator.update -> interpolator.interpolate -> interpolated_values
        interpolated_values shape == expected shape
        All interpolated_values are finite.
    """
    grid_x, grid_y = np.mgrid[0:1:3j, 0:1:3j]
    grid_shape = grid_x.shape

    points = np.stack((grid_x.ravel(), grid_y.ravel()))

    velocities = np.stack((grid_x.ravel(), grid_y.ravel()))

    interpolator = GridInterpolator(grid_shape=grid_shape, method="linear")
    interpolator.update(velocities, points)

    new_points = np.array([[0.5, 0.5], [0.25, 0.75]])
    interpolated_values = interpolator.interpolate(new_points)

    assert interpolated_values.shape == (new_points.shape[0], 2)
    assert np.isfinite(interpolated_values).all()


def test_grid_interpolator_3d():
    """Tests the 3D GridInterpolator.

    Flow:
        Generate 3D grid data -> GridInterpolator initialized
        -> interpolator.update -> interpolator.interpolate -> interpolated_values
        interpolated_values shape == expected shape
        All interpolated_values are finite.
    """
    grid_x, grid_y, grid_z = np.mgrid[0:1:3j, 0:1:3j, 0:1:3j]
    grid_shape = grid_x.shape

    points = np.stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel()))
    velocities = np.stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel()))

    interpolator = GridInterpolator(grid_shape=grid_shape, method="linear")
    interpolator.update(velocities, points)

    new_points = np.array([[0.5, 0.5, 0.5], [0.25, 0.25, 0.75]])
    interpolated_values = interpolator.interpolate(new_points)

    assert interpolated_values.shape == (new_points.shape[0], 3)
    assert np.isfinite(interpolated_values).all()


def test_create_interpolator_with_grid_shape():
    """Tests that create_interpolator returns a GridInterpolator when
    grid_shape is provided.

    Flow:
        grid_shape, "linear" -> create_interpolator -> interpolator
        interpolator is instance of GridInterpolator
        interpolator.grid_shape == grid_shape
        interpolator.method == "linear"
    """
    grid_shape = (3, 3)
    interpolator = create_interpolator("linear", grid_shape=grid_shape)
    assert isinstance(interpolator, GridInterpolator)
    assert interpolator.grid_shape == grid_shape
    assert interpolator.method == "linear"


@pytest.mark.parametrize("kind", ["cubic", "linear", "nearest"])
def test_create_interpolator_returns_correct_type(kind):
    """Tests that create_interpolator returns the correct interpolator type
    for various kinds.

    Args:
        kind (str): The interpolation type string.

    Flow:
        kind -> create_interpolator -> interpolator
        interpolator is instance of expected type.
    """
    interpolator = create_interpolator(kind)
    assert isinstance(
        interpolator,
        (
            CubicInterpolator,
            LinearInterpolator,
            NearestNeighborInterpolator,
            GridInterpolator,
        ),
    )


def test_create_interpolator_invalid_type():
    """Tests that create_interpolator raises ValueError for an invalid
    interpolation type.

    Flow:
        "nonsense" -> create_interpolator -> raises ValueError
    """
    with pytest.raises(ValueError, match="Invalid interpolation type"):
        create_interpolator("nonsense")

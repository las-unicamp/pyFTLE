import numpy as np
import pytest

from src.interpolate import (
    CubicInterpolator,
    GridInterpolator,
    LinearInterpolator,
    NearestNeighborInterpolator,
    create_interpolator,
)
from src.my_types import (
    ArrayFloat64Nx2,
    ArrayFloat64Nx3,
)


# -----------------------
# Helper: mock data
# -----------------------
def generate_mock_data_2d() -> tuple[ArrayFloat64Nx2, ArrayFloat64Nx2]:
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    velocities = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
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
        CubicInterpolator,
        LinearInterpolator,
        NearestNeighborInterpolator,
    ],
)
def test_interpolators_2d(strategy_class):
    points, velocities = generate_mock_data_2d()
    interpolator = strategy_class()
    interpolator.update(velocities, points)

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
        LinearInterpolator,
        NearestNeighborInterpolator,
    ],
)
def test_interpolators_3d(strategy_class):
    points, velocities = generate_mock_data_3d()
    interpolator = strategy_class()
    interpolator.update(velocities, points)

    new_points = np.array([[0.5, 0.5, 0.5], [0.25, 0.25, 0.75]], dtype=np.float64)
    interpolated_values = interpolator.interpolate(new_points)

    assert interpolated_values.shape == (new_points.shape[0], 3)
    assert np.isfinite(interpolated_values).all()


# -----------------------
# GridInterpolator tests
# -----------------------
def test_grid_interpolator_2d():
    grid_x, grid_y = np.mgrid[0:1:3j, 0:1:3j]
    grid_shape = grid_x.shape  # (3, 3)

    # Flatten grid points to shape (n_points, ndim)
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    # Create velocities array shape (n_points, ndim)
    velocities = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    interpolator = GridInterpolator(grid_shape=grid_shape, method="linear")
    interpolator.update(velocities, points)

    new_points = np.array([[0.5, 0.5], [0.25, 0.75]], dtype=np.float64)
    interpolated_values = interpolator.interpolate(new_points)

    assert interpolated_values.shape == (new_points.shape[0], 2)
    assert np.isfinite(interpolated_values).all()


def test_grid_interpolator_3d():
    grid_x, grid_y, grid_z = np.mgrid[0:1:3j, 0:1:3j, 0:1:3j]
    grid_shape = grid_x.shape  # (3, 3, 3)

    points = np.column_stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel()))
    velocities = np.column_stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel()))

    interpolator = GridInterpolator(grid_shape=grid_shape, method="linear")
    interpolator.update(velocities, points)

    new_points = np.array([[0.5, 0.5, 0.5], [0.25, 0.25, 0.75]], dtype=np.float64)
    interpolated_values = interpolator.interpolate(new_points)

    assert interpolated_values.shape == (new_points.shape[0], 3)
    assert np.isfinite(interpolated_values).all()


def test_create_interpolator_with_grid_shape():
    grid_shape = (3, 3)
    interpolator = create_interpolator("linear", grid_shape=grid_shape)
    assert isinstance(interpolator, GridInterpolator)
    assert interpolator.grid_shape == grid_shape
    assert interpolator.method == "linear"


# -----------------------
# Factory tests
# -----------------------
@pytest.mark.parametrize("kind", ["cubic", "linear", "nearest"])
def test_create_interpolator_returns_correct_type(kind):
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
    with pytest.raises(ValueError, match="Invalid interpolation type"):
        create_interpolator("nonsense")

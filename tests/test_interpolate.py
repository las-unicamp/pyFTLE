# test_interpolate.py

import numpy as np
import pytest

from src.interpolate import (
    CubicInterpolator,
    CythonInterpolator,
    GridInterpolator,
    LinearInterpolator,
    NearestNeighborInterpolator,
    create_interpolator,
)
from src.my_types import Array2xMxN, Array3xMxN, ArrayFloat64Nx2, ArrayFloat64Nx3

# #######################
# ## Helper Fixtures   ##
# #######################


@pytest.fixture
def scatter_data_2d() -> tuple[ArrayFloat64Nx2, ArrayFloat64Nx2]:
    """Provides 2D scattered points and a corresponding velocity field (u=x, v=y)."""
    points = np.array(
        [[0, 0], [1, 0], [0, 1], [1, 1]],
        dtype=np.float64,
    )
    velocities = np.copy(points)
    return points, velocities


@pytest.fixture
def scatter_data_3d() -> tuple[ArrayFloat64Nx3, ArrayFloat64Nx3]:
    """Provides 3D scattered points and
    a corresponding velocity field (u=x, v=y, w=z)."""
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    velocities = np.copy(points)
    return points, velocities


@pytest.fixture
def grid_data_2d() -> tuple[Array2xMxN, Array2xMxN]:
    """Provides a 2D grid and a corresponding velocity field (u=x, v=y)."""
    grid_x, grid_y = np.mgrid[0:1:5j, 0:1:5j]
    coordinates = np.array([grid_x, grid_y], dtype=np.float64)
    velocities = np.array([np.copy(grid_x), np.copy(grid_y)], dtype=np.float64)
    return coordinates, velocities


@pytest.fixture
def grid_data_3d() -> tuple[Array3xMxN, Array3xMxN]:
    """Provides a 3D grid and a corresponding velocity field (u=x, v=y, w=z)."""
    grid_x, grid_y, grid_z = np.mgrid[0:1:5j, 0:1:5j, 0:1:5j]
    coordinates = np.array([grid_x, grid_y, grid_z], dtype=np.float64)
    velocities = np.array(
        [np.copy(grid_x), np.copy(grid_y), np.copy(grid_z)], dtype=np.float64
    )
    return coordinates, velocities


# ######################################
# ## Tests for Factory Interpolators  ##
# ######################################


@pytest.mark.parametrize("kind", ["cubic", "linear", "nearest"])
def test_unstructured_2d_from_factory(kind, scatter_data_2d):
    """Tests 2D interpolators for scattered data created via the factory."""
    points, velocities = scatter_data_2d
    new_points = np.array([[0.5, 0.5], [0.25, 0.75]], dtype=np.float64)

    interpolator = create_interpolator(kind)
    interpolator.update(velocities=velocities, points=points)
    interpolated_values = interpolator.interpolate(new_points)

    assert interpolated_values.shape == (new_points.shape[0], 2)
    assert np.isfinite(interpolated_values).all()


@pytest.mark.parametrize("kind", ["linear", "nearest"])
def test_unstructured_3d_from_factory(kind, scatter_data_3d):
    """Tests 3D interpolators for scattered data created via the factory."""
    points, velocities = scatter_data_3d
    new_points = np.array([[0.5, 0.5, 0.5], [0.25, 0.75, 0.1]], dtype=np.float64)

    interpolator = create_interpolator(kind)
    interpolator.update(velocities=velocities, points=points)
    interpolated_values = interpolator.interpolate(new_points)

    assert interpolated_values.shape == (new_points.shape[0], 3)
    assert np.isfinite(interpolated_values).all()


def test_grid_2d_from_factory(grid_data_2d):
    """Tests the 2D grid interpolator created via the factory."""
    coordinates, velocities = grid_data_2d
    new_points = np.array([[0.5, 0.5], [0.25, 0.75]], dtype=np.float64)

    interpolator = create_interpolator("grid")
    interpolator.update(velocities=velocities, points=coordinates)
    interpolated_values = interpolator.interpolate(new_points)

    assert interpolated_values.shape == (new_points.shape[0], 2)
    assert np.isfinite(interpolated_values).all()
    # For a simple u=x, v=y field, interpolated values should match the points
    assert np.allclose(interpolated_values, new_points)


@pytest.mark.parametrize("kind", ["grid", "grid_cython"])
def test_grid_3d_from_factory(kind, grid_data_3d):
    """Tests 3D grid interpolators (standard and Cython) from the factory."""
    coordinates, velocities = grid_data_3d
    new_points = np.array([[0.5, 0.5, 0.5], [0.25, 0.75, 0.1]], dtype=np.float64)

    interpolator = create_interpolator(kind)
    interpolator.update(velocities=velocities, points=coordinates)
    interpolated_values = interpolator.interpolate(new_points)

    assert interpolated_values.shape == (new_points.shape[0], 3)
    assert np.isfinite(interpolated_values).all()
    # For a simple u=x, v=y, w=z field, interpolated values should match the points
    assert np.allclose(interpolated_values, new_points, atol=1e-9)


# ##################################
# ## Tests for Factory Logic      ##
# ##################################


@pytest.mark.parametrize(
    "kind, expected_class",
    [
        ("cubic", CubicInterpolator),
        ("linear", LinearInterpolator),
        ("nearest", NearestNeighborInterpolator),
        ("grid", GridInterpolator),
        ("grid_cython", CythonInterpolator),
    ],
)
def test_factory_returns_correct_types(kind, expected_class):
    """Ensures the factory returns an instance of the correct class for each key."""
    interpolator = create_interpolator(kind)
    assert isinstance(interpolator, expected_class)


def test_factory_raises_error_for_invalid_type():
    """Ensures the factory raises a ValueError for an unknown interpolator type."""
    with pytest.raises(ValueError, match="Invalid interpolation type"):
        create_interpolator("nonexistent_interpolator")

import numpy as np
import pytest

from src.interpolate import CubicInterpolatorStrategy


def test_cubic_interpolator():
    # Define sample input data
    points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32
    )
    velocities_u = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    velocities_v = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

    # Create the interpolator
    interpolator = CubicInterpolatorStrategy(points, velocities_u, velocities_v)

    # Define new points for interpolation
    new_points = np.array([[0.5, 0.5], [0.25, 0.75]], dtype=np.float32)

    # Perform interpolation
    interpolated_values = interpolator.interpolate(new_points)

    # Assert the shape of the interpolated values
    assert interpolated_values.shape == (new_points.shape[0], 2)

    # Assert that interpolated values are within expected bounds
    assert np.all(interpolated_values[:, 0] >= 0.0) and np.all(
        interpolated_values[:, 0] <= 1.0
    )
    assert np.all(interpolated_values[:, 1] >= 0.0) and np.all(
        interpolated_values[:, 1] <= 1.0
    )

    # Assert interpolation consistency (optional, depending on expectations)
    assert np.isfinite(interpolated_values).all()


if __name__ == "__main__":
    pytest.main()

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.integrate import (
    AdamsBashforth2Integrator,
    AdaptiveIntegrator,
    EulerIntegrator,
    RungeKutta4Integrator,
    get_integrator,
)
from src.interpolate import InterpolationStrategy
from src.particles import NeighboringParticles


@pytest.fixture
def mock_interpolator():
    mock = MagicMock(spec=InterpolationStrategy)
    mock.interpolate.side_effect = lambda x: x * 0.1  # Fake velocity field
    return mock


@pytest.fixture
def initial_conditions():
    # Create a positions array with shape (4 * N, 2)
    positions = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],  # Left neighbors
            [5.0, 6.0],
            [7.0, 8.0],  # Right neighbors
            [9.0, 10.0],
            [11.0, 12.0],  # Top neighbors
            [13.0, 14.0],
            [15.0, 16.0],  # Bottom neighbors
        ]
    )
    return NeighboringParticles(positions=positions)


@pytest.fixture
def previous_conditions():
    # Create a positions array with shape (4 * N, 2)
    positions = np.array(
        [
            [0.9, 1.8],
            [3.0, 3.8],  # Left neighbors
            [4.8, 5.8],
            [6.9, 7.8],  # Right neighbors
            [8.8, 9.9],
            [10.8, 11.9],  # Top neighbors
            [12.8, 13.9],
            [14.8, 15.9],  # Bottom neighbors
        ]
    )
    return positions


def test_euler_integrator(mock_interpolator, initial_conditions):
    integrator = EulerIntegrator()
    h = 0.1
    initial_positions = initial_conditions.positions.copy()

    integrator.integrate(h, initial_conditions, mock_interpolator)

    expected_positions = initial_positions + h * initial_positions * 0.1

    assert np.allclose(initial_conditions.positions, expected_positions)


def test_runge_kutta4_integrator(mock_interpolator, initial_conditions):
    integrator = RungeKutta4Integrator()
    h = 0.1
    integrator.integrate(h, initial_conditions, mock_interpolator)

    assert np.all(np.isfinite(initial_conditions.positions))


def test_adams_bashforth2_integrator(
    mock_interpolator, initial_conditions, previous_conditions
):
    integrator = AdamsBashforth2Integrator()
    h = 0.1
    integrator.integrate(h, initial_conditions, previous_conditions, mock_interpolator)

    assert np.all(np.isfinite(initial_conditions.positions))


def test_get_integrator():
    # Test valid integrator names
    assert isinstance(get_integrator("ab2"), AdaptiveIntegrator)
    assert isinstance(get_integrator("euler"), EulerIntegrator)
    assert isinstance(get_integrator("rk4"), RungeKutta4Integrator)

    # Test case insensitivity
    assert isinstance(get_integrator("AB2"), AdaptiveIntegrator)
    assert isinstance(get_integrator("EULER"), EulerIntegrator)
    assert isinstance(get_integrator("rK4"), RungeKutta4Integrator)

    # Test invalid input
    with pytest.raises(ValueError, match="Invalid integrator name 'invalid'.*"):
        get_integrator("invalid")

    with pytest.raises(ValueError, match="Invalid integrator name ''.*"):
        get_integrator("")


if __name__ == "__main__":
    pytest.main()

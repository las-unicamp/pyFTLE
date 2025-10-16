from unittest.mock import MagicMock

import numpy as np
import pytest

from src.integrate import (
    AdamsBashforth2Integrator,
    EulerIntegrator,
    RungeKutta4Integrator,
    create_integrator,
)
from src.interpolate import Interpolator
from src.particles import NeighboringParticles


@pytest.fixture
def mock_interpolator():
    """
    Creates a mock interpolator that simulates a velocity field v = 0.1 * p.
    It correctly handles the 'out' argument for in-place operations.
    """
    mock = MagicMock(spec=Interpolator)

    def mock_interpolate(positions, out=None):
        # Calculate the velocity
        velocity = positions * 0.1
        # If an output buffer is provided, copy the result into it
        if out is not None:
            np.copyto(out, velocity)
        return velocity

    mock.interpolate.side_effect = mock_interpolate
    return mock


@pytest.fixture
def initial_conditions():
    """Provides a standard set of initial particle positions."""
    positions = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
            [13.0, 14.0],
            [15.0, 16.0],
        ],
        dtype=np.float64,
    )
    return NeighboringParticles(positions=positions)


def test_euler_integrator(mock_interpolator, initial_conditions):
    """Tests the EulerIntegrator for one step."""
    integrator = EulerIntegrator(mock_interpolator)
    h = 0.1
    p0 = initial_conditions.positions.copy()

    # Perform integration
    integrator.integrate(h, initial_conditions)

    # Calculate expected positions
    # p1 = p0 + h * v(p0) = p0 + h * (p0 * 0.1)
    v0 = p0 * 0.1
    expected_positions = p0 + h * v0

    assert np.allclose(initial_conditions.positions, expected_positions)


def test_runge_kutta4_integrator(mock_interpolator, initial_conditions):
    """Tests the RungeKutta4Integrator for one step."""
    integrator = RungeKutta4Integrator(mock_interpolator)
    h = 0.1
    p = initial_conditions.positions.copy()

    # Perform integration
    integrator.integrate(h, initial_conditions)

    # Manually calculate the expected result for comparison

    def v(pos):
        return pos * 0.1

    k1 = v(p)
    k2 = v(p + 0.5 * h * k1)
    k3 = v(p + 0.5 * h * k2)
    k4 = v(p + h * k3)
    expected_positions = p + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    assert np.allclose(initial_conditions.positions, expected_positions)


def test_adams_bashforth2_integrator(mock_interpolator, initial_conditions):
    """Tests the AdamsBashforth2Integrator for two steps."""
    integrator = AdamsBashforth2Integrator(mock_interpolator)
    h = 0.1
    p0 = initial_conditions.positions.copy()

    # --- First step (uses Euler for initialization) ---
    integrator.integrate(h, initial_conditions)
    p1 = initial_conditions.positions.copy()

    # Check first step result
    v0 = p0 * 0.1
    expected_p1 = p0 + h * v0
    assert np.allclose(p1, expected_p1)

    # --- Second step (uses the actual AB2 formula) ---
    integrator.integrate(h, initial_conditions)
    p2 = initial_conditions.positions

    # Calculate expected result for the second step
    v1 = p1 * 0.1
    # p2 = p1 + h * (1.5 * v1 - 0.5 * v0)
    expected_p2 = p1 + h * (1.5 * v1 - 0.5 * v0)

    assert np.allclose(p2, expected_p2)


def test_create_integrator(mock_interpolator):
    """Tests the factory function for creating integrators."""
    # Valid names
    assert isinstance(
        create_integrator("ab2", mock_interpolator), AdamsBashforth2Integrator
    )
    assert isinstance(create_integrator("euler", mock_interpolator), EulerIntegrator)
    assert isinstance(
        create_integrator("rk4", mock_interpolator), RungeKutta4Integrator
    )

    # Case-insensitivity
    assert isinstance(
        create_integrator("AB2", mock_interpolator), AdamsBashforth2Integrator
    )
    assert isinstance(create_integrator("EULER", mock_interpolator), EulerIntegrator)
    assert isinstance(
        create_integrator("rK4", mock_interpolator), RungeKutta4Integrator
    )

    # Invalid input
    with pytest.raises(ValueError, match="Invalid integrator name 'invalid'.*"):
        create_integrator("invalid", mock_interpolator)
    with pytest.raises(ValueError, match="Invalid integrator name ''.*"):
        create_integrator("", mock_interpolator)

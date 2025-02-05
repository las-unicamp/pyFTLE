from unittest.mock import MagicMock

import numpy as np
import pytest

from src.dtos import NeighboringParticles
from src.integrate import (
    AdamsBashforth2Integrator,
    EulerIntegrator,
    RungeKutta4Integrator,
)
from src.interpolate import InterpolationStrategy
from src.particle_position import PositionDict


@pytest.fixture
def mock_interpolator():
    mock = MagicMock(spec=InterpolationStrategy)
    mock.interpolate.side_effect = lambda x: x * 0.1  # Fake velocity field
    return mock


@pytest.fixture
def initial_conditions():
    return PositionDict(
        NeighboringParticles(
            left=np.array([[1.0, 2.0], [3.0, 4.0]]),
            right=np.array([[5.0, 6.0], [7.0, 8.0]]),
            top=np.array([[9.0, 10.0], [11.0, 12.0]]),
            bottom=np.array([[13.0, 14.0], [15.0, 16.0]]),
        )
    )


@pytest.fixture
def previous_conditions():
    return PositionDict(
        NeighboringParticles(
            left=np.array([[0.9, 1.8], [3.0, 3.8]]),
            right=np.array([[4.8, 5.8], [6.9, 7.8]]),
            top=np.array([[8.8, 9.9], [10.8, 11.9]]),
            bottom=np.array([[12.8, 13.9], [14.8, 15.9]]),
        )
    )


def test_euler_integrator(mock_interpolator, initial_conditions):
    integrator = EulerIntegrator()
    h = 0.1
    result = integrator.integrate(h, initial_conditions, mock_interpolator)

    assert isinstance(result, PositionDict)
    assert np.allclose(
        result.data.top,
        initial_conditions.data.top + h * initial_conditions.data.top * 0.1,
    )
    assert np.allclose(
        result.data.bottom,
        initial_conditions.data.bottom + h * initial_conditions.data.bottom * 0.1,
    )
    assert np.allclose(
        result.data.left,
        initial_conditions.data.left + h * initial_conditions.data.left * 0.1,
    )
    assert np.allclose(
        result.data.right,
        initial_conditions.data.right + h * initial_conditions.data.right * 0.1,
    )


def test_runge_kutta4_integrator(mock_interpolator, initial_conditions):
    integrator = RungeKutta4Integrator()
    h = 0.1
    result = integrator.integrate(h, initial_conditions, mock_interpolator)

    assert isinstance(result, PositionDict)
    assert np.all(np.isfinite(result.data.top))
    assert np.all(np.isfinite(result.data.bottom))
    assert np.all(np.isfinite(result.data.left))
    assert np.all(np.isfinite(result.data.right))


def test_adams_bashforth2_integrator(
    mock_interpolator, initial_conditions, previous_conditions
):
    integrator = AdamsBashforth2Integrator()
    h = 0.1
    result = integrator.integrate(
        h, previous_conditions, initial_conditions, mock_interpolator
    )

    assert isinstance(result, PositionDict)
    assert np.all(np.isfinite(result.data.top))
    assert np.all(np.isfinite(result.data.bottom))
    assert np.all(np.isfinite(result.data.left))
    assert np.all(np.isfinite(result.data.right))

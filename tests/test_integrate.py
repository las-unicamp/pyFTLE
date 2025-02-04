from unittest.mock import MagicMock

import numpy as np
import pytest

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
    return PositionDict({"x": np.array([1.0, 2.0]), "y": np.array([3.0, 4.0])})


@pytest.fixture
def previous_conditions():
    return PositionDict({"x": np.array([0.9, 1.9]), "y": np.array([2.9, 3.9])})


def test_euler_integrator(mock_interpolator, initial_conditions):
    integrator = EulerIntegrator()
    h = 0.1
    result = integrator.integrate(h, initial_conditions, mock_interpolator)

    assert isinstance(result, PositionDict)
    assert np.allclose(
        result.position["x"],
        initial_conditions.position["x"] + h * initial_conditions.position["x"] * 0.1,
    )
    assert np.allclose(
        result.position["y"],
        initial_conditions.position["y"] + h * initial_conditions.position["y"] * 0.1,
    )


def test_runge_kutta4_integrator(mock_interpolator, initial_conditions):
    integrator = RungeKutta4Integrator()
    h = 0.1
    result = integrator.integrate(h, initial_conditions, mock_interpolator)

    assert isinstance(result, PositionDict)
    assert np.all(np.isfinite(result.position["x"]))
    assert np.all(np.isfinite(result.position["y"]))


def test_adams_bashforth2_integrator(
    mock_interpolator, initial_conditions, previous_conditions
):
    integrator = AdamsBashforth2Integrator()
    h = 0.1
    result = integrator.integrate(
        h, previous_conditions, initial_conditions, mock_interpolator
    )

    assert isinstance(result, PositionDict)
    assert np.all(np.isfinite(result.position["x"]))
    assert np.all(np.isfinite(result.position["y"]))

import numpy as np
import pytest

from pyftle.integrate import (
    AdamsBashforth2Integrator,
    EulerIntegrator,
    RungeKutta4Integrator,
    create_integrator,
)


def test_euler_integrator(mock_interpolator, initial_conditions):
    """Tests the Euler integrator.

    Args:
        mock_interpolator (MagicMock): Mock object for Interpolator.
        initial_conditions (NeighboringParticles): Fixture providing initial
            particle conditions.

    Flow:
        EulerIntegrator initialized with mock_interpolator
        -> integrator.integrate with initial_conditions
        initial_conditions.positions == expected_positions (calculated manually)
    """
    integrator = EulerIntegrator(mock_interpolator)
    h = 0.1
    initial_positions = initial_conditions.positions.copy()

    integrator.integrate(h, initial_conditions)

    expected_positions = initial_positions + h * initial_positions * 0.1

    assert np.allclose(initial_conditions.positions, expected_positions)


def test_runge_kutta4_integrator(mock_interpolator, initial_conditions):
    """Tests the Runge-Kutta 4 integrator.

    Args:
        mock_interpolator (MagicMock): Mock object for Interpolator.
        initial_conditions (NeighboringParticles): Fixture providing initial
            particle conditions.

    Flow:
        RungeKutta4Integrator initialized with mock_interpolator
        -> integrator.integrate with initial_conditions
        All positions in initial_conditions.positions are finite.
    """
    integrator = RungeKutta4Integrator(mock_interpolator)
    h = 0.1
    integrator.integrate(h, initial_conditions)

    assert np.all(np.isfinite(initial_conditions.positions))


def test_adams_bashforth2_integrator(mock_interpolator, initial_conditions):
    """Tests the Adams-Bashforth 2 integrator.

    Args:
        mock_interpolator (MagicMock): Mock object for Interpolator.
        initial_conditions (NeighboringParticles): Fixture providing initial
            particle conditions.

    Flow:
        AdamsBashforth2Integrator initialized with mock_interpolator
        -> integrator.integrate with initial_conditions
        All positions in initial_conditions.positions are finite.
    """
    integrator = AdamsBashforth2Integrator(mock_interpolator)
    h = 0.1
    integrator.integrate(h, initial_conditions)

    assert np.all(np.isfinite(initial_conditions.positions))


def test_get_integrator(mock_interpolator):
    """Tests the create_integrator factory function.

    Args:
        mock_interpolator (MagicMock): Mock object for Interpolator.

    Flow:
        create_integrator with valid names -> returns correct integrator type
        create_integrator with case-insensitive names -> returns correct
            integrator type
        create_integrator with invalid names -> raises ValueError
    """
    assert isinstance(
        create_integrator("ab2", mock_interpolator), AdamsBashforth2Integrator
    )
    assert isinstance(create_integrator("euler", mock_interpolator), EulerIntegrator)
    assert isinstance(
        create_integrator("rk4", mock_interpolator), RungeKutta4Integrator
    )

    assert isinstance(
        create_integrator("AB2", mock_interpolator), AdamsBashforth2Integrator
    )
    assert isinstance(create_integrator("EULER", mock_interpolator), EulerIntegrator)
    assert isinstance(
        create_integrator("rK4", mock_interpolator), RungeKutta4Integrator
    )

    with pytest.raises(ValueError, match="Invalid integrator name 'invalid'.*"):
        create_integrator("invalid", mock_interpolator)
    with pytest.raises(ValueError, match="Invalid integrator name ''.*"):
        create_integrator("", mock_interpolator)

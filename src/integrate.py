from typing import Protocol

from src.interpolate import InterpolationStrategy
from src.particles import NeighboringParticles


class IntegratorStrategy(Protocol):
    def integrate(
        self,
        h: float,
        particles: NeighboringParticles,
        interpolator: InterpolationStrategy,
    ) -> None:
        """
        Perform a single integration step (Euler, Runge-Kutta, Adams-Bashforth 2).
        WARNING: This method performs in-place mutations of the particle positions.

        Args:
            h (float): Step size for integration.
            particles (NeighboringParticles): Dataclass instance containing the
                coordinates of the particles at the current step.
            interpolator (InterpolationStrategy):
                An instance of an interpolation strategy that computes the velocity
                given the position values.
        """
        ...


class AdamsBashforth2Integrator:
    """
    Perform a single step of the second-order Adams-Bashforth method
    for solving ordinary differential equations (ODEs).

    The Adams-Bashforth method is an explicit multistep method that uses the
    values of the function (velocity) at the current and previous steps to
    approximate the solution.

    y_{n+2} = y_{n+1} + h * [3/2 * f(t_{n+1}, y_{n+1}) - 1/2 * f(t_{n}, y_{n})]

    Here we use the convention:
    - n+2 → Future timestep, to be be stored in `particles` after integration
    - n+1 → Current timestep, obtained from `particles`
    - n   → Previous timestep, obtained from `particles_previous`
    """

    def __init__(self):
        self.previous_velocity = None  # Stores f(t_n, y_n) for next iteration

    def integrate(
        self,
        h: float,
        particles: NeighboringParticles,
        interpolator: InterpolationStrategy,
    ) -> None:
        current_velocity = interpolator.interpolate(particles.positions)

        if self.previous_velocity is None:
            # First step: fallback to Euler method
            particles.positions += h * current_velocity
        else:
            # Adams-Bashforth 2-step method
            particles.positions += h * (
                1.5 * current_velocity - 0.5 * self.previous_velocity
            )

        # Store current velocity for the next step
        self.previous_velocity = current_velocity


class EulerIntegrator:
    """
    Perform a single step of the Euler method for solving ordinary differential
    equations (ODEs).

    The Euler method is a first-order numerical procedure for solving ODEs
    by approximating the solution using the derivative (velocity) at the current
    point in time.
    """

    def integrate(
        self,
        h: float,
        particles: NeighboringParticles,
        interpolator: InterpolationStrategy,
    ) -> None:
        particles.positions += h * interpolator.interpolate(particles.positions)


class RungeKutta4Integrator:
    """
    Perform a single step of the 4th-order Runge-Kutta method for solving
    ordinary differential equations (ODEs).

    The Runge-Kutta 4 method is a widely used numerical method for solving ODEs
    by approximating the solution using four intermediate slopes, providing a
    higher-order accuracy than the Euler or Adams-Bashforth methods.
    """

    def integrate(
        self,
        h: float,
        particles: NeighboringParticles,
        interpolator: InterpolationStrategy,
    ) -> None:
        # Compute the four slopes (k1, k2, k3, k4)
        k1 = interpolator.interpolate(particles.positions)
        k2 = interpolator.interpolate(particles.positions + 0.5 * h * k1)
        k3 = interpolator.interpolate(particles.positions + 0.5 * h * k2)
        k4 = interpolator.interpolate(particles.positions + h * k3)

        # Update the solution in-place using the weighted average of the slopes
        particles.positions += (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def get_integrator(integrator_name: str) -> IntegratorStrategy:
    """Factory to create IntegratorStrategy instances"""

    integrator_name = integrator_name.lower()  # Normalize input to lowercase

    integrator_map = {
        "ab2": AdamsBashforth2Integrator,  # Uses Euler for the first step, then AB2
        "euler": EulerIntegrator,
        "rk4": RungeKutta4Integrator,
    }

    if integrator_name not in integrator_map:
        raise ValueError(
            f"Invalid integrator name '{integrator_name}'. "
            f"Choose from {list(integrator_map.keys())}."
        )

    return integrator_map[integrator_name]()

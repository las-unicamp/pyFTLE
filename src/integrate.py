from typing import Protocol, overload

from src.decorators import format_position_dict
from src.interpolate import InterpolationStrategy
from src.particle_position import PositionDict


class IntegratorStrategy(Protocol):
    @overload
    def integrate(
        self, h: float, y_n: PositionDict, interpolator: InterpolationStrategy
    ) -> PositionDict:
        """
        Perform a single integration step (Euler, Runge-Kutta).

        Args:
            h (float): Step size for integration.
            y_n (PositionDict): Dictionary of arrays containing the solution values
                (positions) at the current step.
            interpolator (InterpolationStrategy):
                An instance of an interpolation strategy that computes the velocity
                (derivative) given the position values.

        Returns:
            PositionDict: The updated solution values after one integration step.
        """
        ...

    @overload
    def integrate(
        self,
        h: float,
        y_n: PositionDict,
        y_np1: PositionDict,
        interpolator: InterpolationStrategy,
    ) -> PositionDict:
        """
        Perform a single integration step (Adams-Bashforth 2).

        Args:
            h (float): Step size for integration.
            y_n (PositionDict): Dictionary of arrays containing the solution values
                (i.e, positions) at the current step.
            y_np1 (PositionDict): Dictionary of arrays containing the solution values
                (i.e, positions) at the next step.
            interpolator (InterpolationStrategy):
                An instance of an interpolation strategy that computes the velocity
                (derivative) given the position values.

        Returns:
            PositionDict: The updated solution values after one integration step.
        """
        ...

    def integrate(self, *args, **kwargs) -> PositionDict:
        """Actual implementation to be overridden by subclasses."""
        pass


class AdamsBashforth2Integrator:
    """
    Perform a single step of the second-order Adams-Bashforth method
    for solving ordinary differential equations (ODEs).

    The Adams-Bashforth method is an explicit multistep method that uses the
    values of the function (velocity) at the current and previous steps to
    approximate the solution.
    """

    @format_position_dict
    def integrate(
        self,
        h: float,
        y_n: PositionDict,
        y_np1: PositionDict,
        interpolator: InterpolationStrategy,
    ) -> PositionDict:
        return y_np1 + h * (
            1.5 * interpolator.interpolate(y_np1) - 0.5 * interpolator.interpolate(y_n)
        )


class EulerIntegrator:
    """
    Perform a single step of the Euler method for solving ordinary differential
    equations (ODEs).

    The Euler method is a first-order numerical procedure for solving ODEs
    by approximating the solution using the derivative (velocity) at the current
    point in time.
    """

    @format_position_dict
    def integrate(
        self,
        h: float,
        y_n: PositionDict,
        interpolator: InterpolationStrategy,
    ) -> PositionDict:
        return y_n + h * interpolator.interpolate(y_n)


class RungeKutta4Integrator:
    """
    Perform a single step of the 4th-order Runge-Kutta method for solving
    ordinary differential equations (ODEs).

    The Runge-Kutta 4 method is a widely used numerical method for solving ODEs
    by approximating the solution using four intermediate slopes, providing a
    higher-order accuracy than the Euler or Adams-Bashforth methods.
    """

    @format_position_dict
    def integrate(
        self,
        h: float,
        y_n: PositionDict,
        interpolator: InterpolationStrategy,
    ) -> PositionDict:
        # Compute the four slopes (k1, k2, k3, k4)
        k1 = interpolator.interpolate(y_n)
        k2 = interpolator.interpolate(y_n + 0.5 * h * k1)
        k3 = interpolator.interpolate(y_n + 0.5 * h * k2)
        k4 = interpolator.interpolate(y_n + h * k3)

        # Update the solution using the weighted average of the slopes
        return y_n + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def get_integrator(integrator_name: str) -> IntegratorStrategy:
    """Factory to create IntegratorStrategy instances"""

    integrator_name = integrator_name.lower()  # Normalize input to lowercase

    integrator_map = {
        "ab2": AdamsBashforth2Integrator,
        "euler": EulerIntegrator,
        "rk4": RungeKutta4Integrator,
    }

    if integrator_name not in integrator_map:
        raise ValueError(
            f"Invalid integrator name '{integrator_name}'. "
            f"Choose from {list(integrator_map.keys())}."
        )

    return integrator_map[integrator_name]()

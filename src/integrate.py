from abc import ABC, abstractmethod
from typing import cast

import numpy as np
from numba import njit  # type: ignore
from numpy.typing import ArrayLike

from src.cython_integrators import (
    adams_bashforth_2_step,
    euler_step,
    runge_kutta_4_step,
)
from src.interpolate import Interpolator
from src.my_types import ArrayFloat64Nx2, ArrayFloat64Nx3
from src.particles import NeighboringParticles


class Integrator(ABC):
    def __init__(self, interpolator: Interpolator, **kwargs) -> None:  # noqa: ARG002
        """
        Constructor for the Integrator. All integrators must receive an
        interpolator during initialization.

        Args:
            interpolator (Interpolator): The interpolator used to compute
                velocity based on particle positions.
            **kwargs: Additional parameters specific to the integrator type.
        """
        self.interpolator = interpolator
        self.current_velocity = None

    @abstractmethod
    def integrate(self, h: float, particles: NeighboringParticles) -> None:
        """
        Perform a single integration step (Euler, Runge-Kutta, Adams-Bashforth 2).
        WARNING: This method performs in-place mutations of the particle positions.

        Args:
            h (float): Step size for integration.
            particles (NeighboringParticles): Dataclass instance containing the
                coordinates of the particles at the current step.
            interpolator (Interpolation):
                An instance of an interpolator that computes the velocity given the
                position values.
        """
        pass


@njit
def adams_bashforth_2_step_n(
    h: float,
    current_velocity: ArrayFloat64Nx2 | ArrayFloat64Nx3,
    previous_velocity: ArrayFloat64Nx2 | ArrayFloat64Nx3,
) -> ArrayFloat64Nx2 | ArrayFloat64Nx3:
    return h * (1.5 * current_velocity - 0.5 * previous_velocity)


class AdamsBashforth2Integrator(Integrator):
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

    def __init__(self, interpolator: Interpolator, **kwargs) -> None:
        super().__init__(interpolator, **kwargs)
        self.previous_velocity = None  # Stores f(t_n, y_n) for next iteration
        self._initialized = False
        self._step_result = None

    def _initialize_first_step(self, h, particles, interpolator):
        self.previous_velocity = np.empty_like(particles.positions)
        self._current_velocity = np.empty_like(particles.positions)
        self._step_result = np.empty_like(particles.positions)
        interpolator.interpolate(particles.positions, out=self._current_velocity)
        self._step_result = euler_step(
            h,
            np.ascontiguousarray(self._current_velocity),
            out=np.ascontiguousarray(self._step_result),
        )
        particles.positions += self._step_result
        np.copyto(self.previous_velocity, self._current_velocity)
        self._initialized = True

    def integrate(self, h: float, particles: NeighboringParticles) -> None:
        if not self._initialized:
            self._initialize_first_step(h, particles, self.interpolator)
            return

        self.interpolator.interpolate(particles.positions, out=self._current_velocity)

        adams_bashforth_2_step(
            h,
            np.ascontiguousarray(self._current_velocity),
            np.ascontiguousarray(self.previous_velocity),
            out=np.ascontiguousarray(self._step_result),
        )
        if particles.positions.shape[1] == 2:
            self._step_result = cast(ArrayFloat64Nx2, self._step_result)
            particles.positions += self._step_result
            self.previous_velocity = cast(ArrayFloat64Nx2, self.previous_velocity)
            self._current_velocity = cast(ArrayFloat64Nx2, self._current_velocity)
            np.copyto(self.previous_velocity, self._current_velocity)
        else:
            self._step_result = cast(ArrayFloat64Nx3, self._step_result)
            particles.positions += self._step_result
            self.previous_velocity = cast(ArrayFloat64Nx3, self.previous_velocity)
            self._current_velocity = cast(ArrayFloat64Nx3, self._current_velocity)
            np.copyto(self.previous_velocity, self._current_velocity)


@njit
def euler_step_n(
    h: float, current_velocity: ArrayFloat64Nx2 | ArrayFloat64Nx3
) -> ArrayFloat64Nx2 | ArrayFloat64Nx3:
    return h * current_velocity


class EulerIntegrator(Integrator):
    """
    Perform a single step of the Euler method for solving ordinary differential
    equations (ODEs).

    The Euler method is a first-order numerical procedure for solving ODEs
    by approximating the solution using the derivative (velocity) at the current
    point in time.
    """

    _initialized = False

    def integrate(self, h: float, particles: NeighboringParticles) -> None:
        if not self._initialized:
            self.current_velocity = np.empty_like(particles.positions)
            self._initialized = True

        self.interpolator.interpolate(particles.positions, out=self.current_velocity)
        particles.positions += euler_step(
            h, np.ascontiguousarray(self.current_velocity)
        )


@njit
def runge_kutta_4_step_n(
    h: float,
    k1: ArrayFloat64Nx2 | ArrayFloat64Nx3,
    k2: ArrayFloat64Nx2 | ArrayFloat64Nx3,
    k3: ArrayFloat64Nx2 | ArrayFloat64Nx3,
    k4: ArrayFloat64Nx2 | ArrayFloat64Nx3,
) -> ArrayFloat64Nx2 | ArrayFloat64Nx3:
    return (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


class RungeKutta4Integrator(Integrator):
    """
    Perform a single step of the 4th-order Runge-Kutta method for solving
    ordinary differential equations (ODEs).

    The Runge-Kutta 4 method is a widely used numerical method for solving ODEs
    by approximating the solution using four intermediate slopes, providing a
    higher-order accuracy than the Euler or Adams-Bashforth methods.
    """

    def __init__(self, interpolator: Interpolator, **kwargs) -> None:
        super().__init__(interpolator, **kwargs)
        self.k1 = None
        self.k2 = None
        self.k3 = None
        self.k4 = None
        self.temp_positions = None
        self.delta_p = None

    def _ensure_buffers(self, shape):
        """Allocate buffers on first run or if particle count changes."""
        if self.k1 is None or self.k1.shape != shape:
            self.k1 = np.empty(shape, dtype=np.float64)
            self.k2 = np.empty(shape, dtype=np.float64)
            self.k3 = np.empty(shape, dtype=np.float64)
            self.k4 = np.empty(shape, dtype=np.float64)
            self.temp_positions = np.empty(shape, dtype=np.float64)
            self.delta_p = np.empty(shape, dtype=np.float64)

    def integrate(self, h: float, particles: NeighboringParticles) -> None:
        p = particles.positions
        self._ensure_buffers(p.shape)
        k1, k2, k3, k4 = self.k1, self.k2, self.k3, self.k4
        delta_p = self.delta_p
        temp_p = self.temp_positions
        self.interpolator.interpolate(p, out=k1)
        p = cast(ArrayLike, p)
        delta_p = cast(ArrayLike, delta_p)
        temp_p = cast(ArrayFloat64Nx3, temp_p)
        k1 = cast(ArrayLike, k1)
        k2 = cast(ArrayLike, k2)
        k3 = cast(ArrayLike, k3)
        k4 = cast(ArrayLike, k4)

        # --- Compute k2 (temp_p = p + 0.5 * h * k1) ---
        euler_step(0.5 * h, k1, out=delta_p)
        np.add(p, delta_p, out=temp_p)
        self.interpolator.interpolate(temp_p, out=k2)

        # --- Compute k3 (temp_p = p + 0.5 * h * k2) ---
        euler_step(0.5 * h, k2, out=delta_p)
        np.add(p, delta_p, out=temp_p)
        self.interpolator.interpolate(temp_p, out=k3)

        # --- Compute k4 (temp_p = p + h * k3) ---
        euler_step(h, k3, out=delta_p)
        np.add(p, delta_p, out=temp_p)
        self.interpolator.interpolate(temp_p, out=k4)
        runge_kutta_4_step(h, k1, k2, k3, k4, delta_p)
        # Update the solution in-place using the weighted average of the slopes
        np.add(p, delta_p, out=cast(ArrayFloat64Nx3, p))


def create_integrator(integrator_name: str, interpolator: Interpolator) -> Integrator:
    """Factory to create Integrator instances"""

    integrator_name = integrator_name.lower()  # Normalize input to lowercase

    integrator_map: dict[str, type[Integrator]] = {
        "ab2": AdamsBashforth2Integrator,  # Uses Euler for the first step, then AB2
        "euler": EulerIntegrator,
        "rk4": RungeKutta4Integrator,
    }

    if integrator_name not in integrator_map:
        raise ValueError(
            f"Invalid integrator name '{integrator_name}'. "
            f"Choose from {list(integrator_map.keys())}."
        )

    return integrator_map[integrator_name](interpolator)

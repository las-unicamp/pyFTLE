from abc import ABC, abstractmethod
from typing import Optional, cast

import numpy as np
from numba import njit  # type: ignore

from pyftle.interpolate import Interpolator
from pyftle.my_types import ArrayFloat64Nx2, ArrayFloat64Nx3
from pyftle.particles import NeighboringParticles

# ============================================================
# Low-level numerical kernels (Numba-accelerated)
# ============================================================


@njit(inline="always")
def euler_step_inplace(
    positions: ArrayFloat64Nx2 | ArrayFloat64Nx3,
    h: float,
    velocity: ArrayFloat64Nx2 | ArrayFloat64Nx3,
):
    """In-place Euler step: positions += h * velocity"""
    positions += h * velocity


@njit(inline="always")
def adams_bashforth_2_step_inplace(
    positions: ArrayFloat64Nx2 | ArrayFloat64Nx3,
    h: float,
    v_current: ArrayFloat64Nx2 | ArrayFloat64Nx3,
    v_prev: ArrayFloat64Nx2 | ArrayFloat64Nx3,
):
    """In-place AB2 step: positions += h * (1.5*v_current - 0.5*v_prev)"""
    positions += h * (1.5 * v_current - 0.5 * v_prev)


@njit(inline="always")
def runge_kutta_4_step_inplace(
    positions: ArrayFloat64Nx2 | ArrayFloat64Nx3,
    h: float,
    k1: ArrayFloat64Nx2 | ArrayFloat64Nx3,
    k2: ArrayFloat64Nx2 | ArrayFloat64Nx3,
    k3: ArrayFloat64Nx2 | ArrayFloat64Nx3,
    k4: ArrayFloat64Nx2 | ArrayFloat64Nx3,
):
    """In-place RK4 update: positions += (h/6)*(k1 + 2*k2 + 2*k3 + k4)"""
    positions += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ============================================================
# Integrator Base Class
# =========================================================


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


# ============================================================
# Euler Integrator
# ============================================================


class EulerIntegrator(Integrator):
    """
    Perform a single step of the Euler method for solving ordinary differential
    equations (ODEs).

    The Euler method is a first-order numerical procedure for solving ODEs
    by approximating the solution using the derivative (velocity) at the current
    point in time.
    """

    def __init__(self, interpolator: Interpolator, **kwargs) -> None:
        super().__init__(interpolator, **kwargs)
        # Preallocate a temporary velocity buffer
        self._velocity = None

    def integrate(self, h: float, particles: NeighboringParticles) -> None:
        # Get or allocate buffer
        if self._velocity is None or self._velocity.shape != particles.positions.shape:
            self._velocity = np.empty_like(particles.positions)

        # Interpolate velocity directly into the preallocated buffer
        np.copyto(self._velocity, self.interpolator.interpolate(particles.positions))

        # In-place update using numba kernel
        euler_step_inplace(particles.positions, h, self._velocity)


# ============================================================
# Adams-Bashforth 2 Integrator
# ============================================================


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
        self.previous_velocity = None
        self._velocity = None

    def integrate(self, h: float, particles: NeighboringParticles) -> None:
        if self._velocity is None or self._velocity.shape != particles.positions.shape:
            self._velocity = np.empty_like(particles.positions)

        # Compute current velocity
        np.copyto(self._velocity, self.interpolator.interpolate(particles.positions))

        if self.previous_velocity is None:
            # First step: Euler fallback
            euler_step_inplace(particles.positions, h, self._velocity)
        else:
            # Use AB2 formula in-place
            adams_bashforth_2_step_inplace(
                particles.positions, h, self._velocity, self.previous_velocity
            )

        # Swap references to avoid reallocations
        if self.previous_velocity is None:
            self.previous_velocity = np.empty_like(self._velocity)
        np.copyto(self.previous_velocity, self._velocity)


# ============================================================
# Runge-Kutta 4 Integrator
# ============================================================


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
        # Preallocate buffers for intermediate slopes
        self._k1: Optional[ArrayFloat64Nx2 | ArrayFloat64Nx3 | None] = None
        self._k2: Optional[ArrayFloat64Nx2 | ArrayFloat64Nx3 | None] = None
        self._k3: Optional[ArrayFloat64Nx2 | ArrayFloat64Nx3 | None] = None
        self._k4: Optional[ArrayFloat64Nx2 | ArrayFloat64Nx3 | None] = None
        # temporary array for intermediate positions
        self._tmp: Optional[ArrayFloat64Nx2 | ArrayFloat64Nx3 | None] = None

    def integrate(self, h: float, particles: NeighboringParticles) -> None:
        npos = particles.positions.shape

        # Lazily allocate working buffers
        if self._k1 is None or self._k1.shape != npos:
            self._k1 = np.empty_like(particles.positions)
            self._k2 = np.empty_like(particles.positions)
            self._k3 = np.empty_like(particles.positions)
            self._k4 = np.empty_like(particles.positions)
            self._tmp = np.empty_like(particles.positions)

        k1 = cast(ArrayFloat64Nx2 | ArrayFloat64Nx3, self._k1)
        k2 = cast(ArrayFloat64Nx2 | ArrayFloat64Nx3, self._k2)
        k3 = cast(ArrayFloat64Nx2 | ArrayFloat64Nx3, self._k3)
        k4 = cast(ArrayFloat64Nx2 | ArrayFloat64Nx3, self._k4)
        tmp = cast(ArrayFloat64Nx2 | ArrayFloat64Nx3, self._tmp)

        # k1 = f(t, y)
        np.copyto(k1, self.interpolator.interpolate(particles.positions))

        # k2 = f(t + h/2, y + h/2 * k1)
        np.copyto(tmp, particles.positions)
        tmp += 0.5 * h * k1
        np.copyto(k2, self.interpolator.interpolate(tmp))

        # k3 = f(t + h/2, y + h/2 * k2)
        np.copyto(tmp, particles.positions)
        tmp += 0.5 * h * k2
        np.copyto(k3, self.interpolator.interpolate(tmp))

        # k4 = f(t + h, y + h * k3)
        np.copyto(tmp, particles.positions)
        tmp += h * k3
        np.copyto(k4, self.interpolator.interpolate(tmp))

        # In-place RK4 update
        runge_kutta_4_step_inplace(particles.positions, h, k1, k2, k3, k4)


# ============================================================
# Factory
# ============================================================


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

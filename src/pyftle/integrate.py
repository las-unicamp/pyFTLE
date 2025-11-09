from abc import ABC, abstractmethod
from typing import Optional, cast

import numpy as np
from numba import njit  # type: ignore

from pyftle.interpolate import Interpolator
from pyftle.my_types import Array2xN, Array3xN, ArrayNx2, ArrayNx3
from pyftle.particles import NeighboringParticles

# ============================================================
# Low-level numerical kernels (Numba-accelerated)
# ============================================================


@njit(inline="always")
def euler_step_inplace(
    positions: ArrayNx2 | ArrayNx3,
    h: float,
    velocity: Array2xN | Array3xN,
):
    r"""
    Perform an in-place Euler integration step.

    Updates particle positions according to:

    .. math::
        x_{n+1} = x_n + h \, v_n

    Parameters
    ----------
    positions : ArrayNx2 | ArrayNx3
        Particle positions at the current time. Updated in-place.
    h : float
        Time step size.
    velocity : Array2xN | Array3xN
        Velocity field evaluated at the current positions.

    Notes
    -----
    This is a low-level kernel designed for performance with Numba.
    It does not allocate memory or perform error checking.
    """
    positions += h * velocity


@njit(inline="always")
def adams_bashforth_2_step_inplace(
    positions: ArrayNx2 | ArrayNx3,
    h: float,
    v_current: Array2xN | Array3xN,
    v_prev: Array2xN | Array3xN,
):
    """
    Perform an in-place second-order Adams–Bashforth (AB2) integration step.

    Updates particle positions according to:

    .. math::

        x_{n+1} = x_n + h \\left( \\tfrac{3}{2} v_n - \\tfrac{1}{2} v_{n-1} \\right)

    Parameters
    ----------
    positions : ArrayNx2 or ArrayNx3
        Particle positions at the current time. Updated in-place.

    h : float
        Time step size.

    v_current : Array2xN or Array3xN
        Velocity field evaluated at the current positions.

    v_prev : Array2xN or Array3xN
        Velocity field evaluated at the previous positions.

    """
    positions += h * (1.5 * v_current - 0.5 * v_prev)


@njit(inline="always")
def runge_kutta_4_step_inplace(
    positions: ArrayNx2 | ArrayNx3,
    h: float,
    k1: ArrayNx2 | ArrayNx3,
    k2: ArrayNx2 | ArrayNx3,
    k3: ArrayNx2 | ArrayNx3,
    k4: ArrayNx2 | ArrayNx3,
):
    """
    Perform an in-place fourth-order Runge-Kutta (RK4) integration step.

    Updates particle positions according to:

    .. math::

        x_{n+1} = x_n + \\frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)

    Parameters
    ----------
    positions : ArrayNx2 | ArrayNx3
        Particle positions at the current time. Updated in-place.
    h : float
        Time step size.
    k1, k2, k3, k4 : ArrayNx2 | ArrayNx3
        Intermediate slope (velocity) evaluations for RK4.

    """
    positions += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ============================================================
# Integrator Base Class
# =========================================================


class Integrator(ABC):
    """
    Abstract base class for numerical ODE integrators.

    All specific integrator implementations (Euler, AB2, RK4)
    must derive from this class and implement the :meth:`integrate`
    method.

    Parameters
    ----------
    interpolator : Interpolator
        Interpolator used to evaluate velocity from particle positions.
    **kwargs
        Optional extra arguments specific to the integrator implementation.
    """

    def __init__(self, interpolator: Interpolator, **kwargs) -> None:  # noqa: ARG002
        self.interpolator = interpolator

    @abstractmethod
    def integrate(self, h: float, particles: NeighboringParticles) -> None:
        """
        Perform a single integration step.

        Parameters
        ----------
        h : float
            Time step size.
        particles : NeighboringParticles
            Dataclass containing particle positions and neighboring data.

        Notes
        -----
        This method must mutate ``particles.positions`` in-place.
        Implementations should use Numba-accelerated kernels for efficiency.
        """
        pass


# ============================================================
# Euler Integrator
# ============================================================


class EulerIntegrator(Integrator):
    r"""
    First-order explicit Euler integrator.

    The Euler method approximates the solution of an ODE by
    advancing the state using the current derivative (velocity):

    .. math::
        x_{n+1} = x_n + h \, v_n
    """

    def __init__(self, interpolator: Interpolator, **kwargs) -> None:
        super().__init__(interpolator, **kwargs)
        # Preallocate a temporary velocity buffer
        self._velocity = None

    def integrate(self, h: float, particles: NeighboringParticles) -> None:
        """
        Perform a single Euler integration step in-place.

        Parameters
        ----------
        h : float
            Time step size.
        particles : NeighboringParticles
            Particle data with current positions to be updated.
        """
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
    Second-order explicit Adams-Bashforth (AB2) integrator.

    Uses a linear combination of the current and previous velocities
    to achieve second-order accuracy:

    .. math::
        x_{n+1} = x_n + h \\left( \\frac{3}{2} v_n - \\frac{1}{2} v_{n-1} \\right)

    On the first iteration (when no previous velocity is available),
    it defaults to a single Euler step.
    """

    def __init__(self, interpolator: Interpolator, **kwargs) -> None:
        super().__init__(interpolator, **kwargs)
        self.previous_velocity = None
        self._velocity = None

    def integrate(self, h: float, particles: NeighboringParticles) -> None:
        """
        Perform a single AB2 integration step in-place.

        Parameters
        ----------
        h : float
            Time step size.
        particles : NeighboringParticles
            Particle data with current positions to be updated.

        Notes
        -----
        Uses Euler integration for the first step, when no
        previous velocity field is available.
        """
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
    Fourth-order explicit Runge-Kutta (RK4) integrator.

    Uses four slope evaluations per step to achieve fourth-order accuracy:

    .. math::
        x_{n+1} = x_n + \\frac{h}{6} (k_1 + 2k_2 + 2k_3 + k_4)

    where:
        k₁ = f(t, x)
        k₂ = f(t + h/2, x + h k₁/2)
        k₃ = f(t + h/2, x + h k₂/2)
        k₄ = f(t + h, x + h k₃)
    """

    def __init__(self, interpolator: Interpolator, **kwargs) -> None:
        super().__init__(interpolator, **kwargs)
        # Preallocate buffers for intermediate slopes
        self._k1: Optional[ArrayNx2 | ArrayNx3 | None] = None
        self._k2: Optional[ArrayNx2 | ArrayNx3 | None] = None
        self._k3: Optional[ArrayNx2 | ArrayNx3 | None] = None
        self._k4: Optional[ArrayNx2 | ArrayNx3 | None] = None
        # temporary array for intermediate positions
        self._tmp: Optional[ArrayNx2 | ArrayNx3 | None] = None

    def integrate(self, h: float, particles: NeighboringParticles) -> None:
        """
        Perform a single RK4 integration step in-place.

        Parameters
        ----------
        h : float
            Time step size.
        particles : NeighboringParticles
            Particle data with current positions to be updated.

        Notes
        -----
        This method allocates intermediate buffers lazily and reuses them
        between calls to minimize memory allocations.
        """
        npos = particles.positions.shape

        # Lazily allocate working buffers
        if self._k1 is None or self._k1.shape != npos:
            self._k1 = np.empty_like(particles.positions)
            self._k2 = np.empty_like(particles.positions)
            self._k3 = np.empty_like(particles.positions)
            self._k4 = np.empty_like(particles.positions)
            self._tmp = np.empty_like(particles.positions)

        k1 = cast(ArrayNx2 | ArrayNx3, self._k1)
        k2 = cast(ArrayNx2 | ArrayNx3, self._k2)
        k3 = cast(ArrayNx2 | ArrayNx3, self._k3)
        k4 = cast(ArrayNx2 | ArrayNx3, self._k4)
        tmp = cast(ArrayNx2 | ArrayNx3, self._tmp)

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
    """
    Factory function to instantiate a numerical integrator.

    Parameters
    ----------
    integrator_name : str
        Name of the integrator to create.
        Supported values: ``"euler"``, ``"ab2"``, ``"rk4"``.
    interpolator : Interpolator
        Interpolator used to compute velocity from particle positions.

    Returns
    -------
    Integrator
        An instance of the requested integrator type.

    Raises
    ------
    ValueError
        If the provided ``integrator_name`` is not recognized.
    """

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

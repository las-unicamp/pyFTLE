# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, language_level=3
import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

# We DO NOT redeclare DTYPE_t here.
# Cython automatically knows about it from the integrators.pxd file.

# ---- Runge-kutta 4 step ----
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object runge_kutta_4_step(double h,
                                DTYPE_t[:, ::1] k1,
                                DTYPE_t[:, ::1] k2,
                                DTYPE_t[:, ::1] k3,
                                DTYPE_t[:, ::1] k4,
                                DTYPE_t[:, ::1] out=None):
    if out is None:
        out = np.empty_like(k1)

    cdef Py_ssize_t i, j
    cdef int n = k1.shape[0], m = k1.shape[1]
    cdef double coeff = h / 6.0
    
    with nogil:
        for i in prange(n):
            for j in range(m):
                out[i, j] = coeff * (k1[i, j] + 2.0*k2[i, j] + 2.0*k3[i, j] + k4[i, j])
    return np.asarray(out)


# ---- Euler step ----
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object euler_step(double h,
                        DTYPE_t[:, ::1] current_velocity,
                        DTYPE_t[:, ::1] out=None):
    if out is None:
        out = np.empty_like(current_velocity)

    cdef Py_ssize_t i, j
    cdef int n = current_velocity.shape[0], m = current_velocity.shape[1]

    with nogil:
        for i in prange(n):
            for j in range(m):
                out[i, j] = h * current_velocity[i, j]
    return np.asarray(out)


# ---- Adams–Bashforth 2 ----
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object adams_bashforth_2_step(double h,
                                    DTYPE_t[:, ::1] current_velocity,
                                    DTYPE_t[:, ::1] previous_velocity,
                                    DTYPE_t[:, ::1] out=None):
    if out is None:
        out = np.empty_like(current_velocity)

    cdef Py_ssize_t i, j
    cdef int n = current_velocity.shape[0], m = current_velocity.shape[1]

    with nogil:
        for i in prange(n):
            for j in range(m):
                out[i, j] = h * (1.5*current_velocity[i, j] - 0.5*previous_velocity[i, j])
    return np.asarray(out)
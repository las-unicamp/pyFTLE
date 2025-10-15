# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

cimport numpy as np
import numpy as np
from libc.math cimport floor

cpdef np.ndarray[np.float64_t, ndim=1] interp3D_vec(np.float_t[:, :, ::1] v,
                                                   np.ndarray[np.float64_t, ndim=2] points):
    """
    Vectorized 3D interpolation.
    points: (N,3) array of coordinates in physical space
    returns: (N,) array of interpolated values
    """
    cdef:
        int N = points.shape[0]
        int i, X, Y, Z
        int x0, x1, y0, y1, z0, z1
        double xd, yd, zd, c00, c01, c10, c11, c0, c1, c
        double px, py, pz
        np.float_t *v_c
        np.ndarray[np.float64_t, ndim=1] out = np.zeros(N, dtype=np.float64)

    X, Y, Z = v.shape[0], v.shape[1], v.shape[2]
    v_c = &v[0,0,0]

    for i in range(N):
        px, py, pz = points[i,0], points[i,1], points[i,2]

        x0 = <int>floor(px)
        x1 = x0 + 1
        y0 = <int>floor(py)
        y1 = y0 + 1
        z0 = <int>floor(pz)
        z1 = z0 + 1

        # NOTE: The original division was redundant as (x1-x0) is always 1.
        # This is a minor optimization.
        xd = px - x0
        yd = py - y0
        zd = pz - z0

        if 0 <= x0 < X-1 and 0 <= y0 < Y-1 and 0 <= z0 < Z-1:
            c00 = v_c[Y*Z*x0 + Z*y0 + z0]*(1-xd) + v_c[Y*Z*x1 + Z*y0 + z0]*xd
            c01 = v_c[Y*Z*x0 + Z*y0 + z1]*(1-xd) + v_c[Y*Z*x1 + Z*y0 + z1]*xd
            c10 = v_c[Y*Z*x0 + Z*y1 + z0]*(1-xd) + v_c[Y*Z*x1 + Z*y1 + z0]*xd
            c11 = v_c[Y*Z*x0 + Z*y1 + z1]*(1-xd) + v_c[Y*Z*x1 + Z*y1 + z1]*xd

            c0 = c00*(1-yd) + c10*yd
            c1 = c01*(1-yd) + c11*yd

            c = c0*(1-zd) + c1*zd
        else:
            c = 0.0

        out[i] = c

    return out


cpdef void interp3D_vec_inplace(np.float_t[:, :, ::1] v,
                                np.ndarray[np.float64_t, ndim=2] points,
                                np.ndarray[np.float64_t, ndim=1] out):
    """
    Vectorized 3D interpolation with in-place computation.
    points: (N,3) array of coordinates in physical space.
    out: (N,) pre-allocated array to store the results.
    """
    cdef:
        int N = points.shape[0]
        int i, X, Y, Z
        int x0, x1, y0, y1, z0, z1
        double xd, yd, zd, c00, c01, c10, c11, c0, c1, c
        double px, py, pz
        np.float_t *v_c

    assert out.shape[0] == N, "Output array has incorrect shape."

    X, Y, Z = v.shape[0], v.shape[1], v.shape[2]
    v_c = &v[0,0,0]

    for i in range(N):
        px, py, pz = points[i,0], points[i,1], points[i,2]

        x0 = <int>floor(px)
        x1 = x0 + 1
        y0 = <int>floor(py)
        y1 = y0 + 1
        z0 = <int>floor(pz)
        z1 = z0 + 1

        xd = px - x0
        yd = py - y0
        zd = pz - z0

        if 0 <= x0 < X-1 and 0 <= y0 < Y-1 and 0 <= z0 < Z-1:
            c00 = v_c[Y*Z*x0 + Z*y0 + z0]*(1-xd) + v_c[Y*Z*x1 + Z*y0 + z0]*xd
            c01 = v_c[Y*Z*x0 + Z*y0 + z1]*(1-xd) + v_c[Y*Z*x1 + Z*y0 + z1]*xd
            c10 = v_c[Y*Z*x0 + Z*y1 + z0]*(1-xd) + v_c[Y*Z*x1 + Z*y1 + z0]*xd
            c11 = v_c[Y*Z*x0 + Z*y1 + z1]*(1-xd) + v_c[Y*Z*x1 + Z*y1 + z1]*xd
            
            c0 = c00*(1-yd) + c10*yd
            c1 = c01*(1-yd) + c11*yd

            c = c0*(1-zd) + c1*zd
        else:
            c = 0.0

        out[i] = c

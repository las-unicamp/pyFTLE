cimport numpy as np

# Declare the original function with its full signature
cpdef np.ndarray[np.float64_t, ndim=1] interp3D_vec(np.float_t[:, :, ::1] v,
                                                   np.ndarray[np.float64_t, ndim=2] points)

# Declare the new in-place function with its full signature
cpdef void interp3D_vec_inplace(np.float_t[:, :, ::1] v,
                                np.ndarray[np.float64_t, ndim=2] points,
                                np.ndarray[np.float64_t, ndim=1] out)

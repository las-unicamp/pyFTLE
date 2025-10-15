cimport numpy as np

ctypedef np.float64_t DTYPE_t

cpdef object runge_kutta_4_step(double h,
                                DTYPE_t[:, ::1] k1,
                                DTYPE_t[:, ::1] k2,
                                DTYPE_t[:, ::1] k3,
                                DTYPE_t[:, ::1] k4,
                                DTYPE_t[:, ::1] out=*)

cpdef object euler_step(double h,
                        DTYPE_t[:, ::1] current_velocity,
                        DTYPE_t[:, ::1] out=*)

cpdef object adams_bashforth_2_step(double h,
                                    DTYPE_t[:, ::1] current_velocity,
                                    DTYPE_t[:, ::1] previous_velocity,
                                    DTYPE_t[:, ::1] out=*)


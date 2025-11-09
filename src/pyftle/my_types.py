"""
Type aliases for NumPy arrays used throughout the ``pyftle`` package.

These aliases are defined using ``nptyping`` to make array shapes and
element types explicit in function signatures. They serve as lightweight,
readable type hints for scientific arrays representing physical fields,
particle positions, and tensors.

All arrays use ``numpy.float64`` as the element type.

Examples
--------
>>> from pyftle.my_types import ArrayNx2
>>> import numpy as np
>>> a: ArrayNx2 = np.zeros((100, 2), dtype=np.float64)

Notes
-----
These aliases are **not runtime checks**; they are primarily for static type
checking (e.g., with ``pyright``, ``mypy``, or IDEs with type support).

Shapes follow the convention:

- ``N``: number of particles, points, or samples.
- Dimensions such as ``2``, ``3``, ``2x2``, and ``3x3`` indicate spatial or
  tensor components.

Aliases
-------
``ArrayN`` : (N,) array of float64
    1D array of scalar values.

``Array2xN`` : (2, N) array of float64
    2D array representing two components across N samples.

``Array3xN`` : (3, N) array of float64
    3D array representing three components across N samples.

``Array4Nx2`` : (N, 2) array of float64
    2D array for four groups of 2D particle positions (e.g., right, left, top,
    bottom).

``Array6Nx3`` : (N, 3) array of float64
    3D array for six groups of 3D particle positions (e.g., right, left, top,
    bottom, front, back).

``ArrayNx2`` : (N, 2) array of float64
    2D array with N vectors in 2D space.

``ArrayNx3`` : (N, 3) array of float64
    2D array with N vectors in 3D space.

``ArrayNx2x2`` : (N, 2, 2) array of float64
    Stack of N 2×2 tensors (e.g., Jacobians in 2D).

``ArrayNx3x3`` : (N, 3, 3) array of float64
    Stack of N 3×3 tensors (e.g., Jacobians in 3D).
"""

from nptyping import Float64, NDArray, Shape

ArrayN = NDArray[Shape["*"], Float64]
Array2xN = NDArray[Shape["2, *"], Float64]
Array3xN = NDArray[Shape["3, *"], Float64]
Array4Nx2 = NDArray[Shape["*, 2"], Float64]
Array6Nx3 = NDArray[Shape["*, 3"], Float64]
ArrayNx2 = NDArray[Shape["*, 2"], Float64]
ArrayNx3 = NDArray[Shape["*, 3"], Float64]
ArrayNx2x2 = NDArray[Shape["*, 2, 2"], Float64]
ArrayNx3x3 = NDArray[Shape["*, 3, 3"], Float64]

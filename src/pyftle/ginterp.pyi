"""
2D and 3D interpolation (bilinear and trilinear) using Eigen and pybind11
"""

from __future__ import annotations

import typing

import numpy
import numpy.typing

__all__: list[str] = [
    "interp2d_vec",
    "interp2d_vec_inplace",
    "interp3d_vec",
    "interp3d_vec_inplace",
]

def interp2d_vec(
    arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"],
) -> numpy.typing.NDArray[numpy.float64]:
    """
    Vectorized 2D interpolation
    """

def interp2d_vec_inplace(
    arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"],
    arg2: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
    ],
) -> None:
    """
    In-place 2D interpolation
    """

def interp3d_vec(
    arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"],
) -> numpy.typing.NDArray[numpy.float64]:
    """
    Vectorized 3D interpolation
    """

def interp3d_vec_inplace(
    arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"],
    arg2: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
    ],
) -> None:
    """
    In-place 3D interpolation
    """

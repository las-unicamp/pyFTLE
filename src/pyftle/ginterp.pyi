"""
Bi and trilinear interpolation (float32 grid, in-place output)
"""

from __future__ import annotations

import typing

import numpy
import numpy.typing

__all__: list[str] = ["interp2d_vec_inplace", "interp3d_vec_inplace"]

def interp2d_vec_inplace(
    v: typing.Annotated[numpy.typing.ArrayLike, numpy.float32],
    points: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    out: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
) -> None: ...
def interp3d_vec_inplace(
    v: typing.Annotated[numpy.typing.ArrayLike, numpy.float32],
    points: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    out: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
) -> None: ...

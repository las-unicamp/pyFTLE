import os
from abc import ABC, abstractmethod
from typing import Optional, Union, cast

import numpy as np
import pyvista as pv
from scipy.io import savemat

from pyftle.my_types import ArrayN, ArrayNx2, ArrayNx3


class FTLEWriter(ABC):
    """
    Abstract base class for writing Finite-Time Lyapunov Exponent (FTLE)
    fields to different file formats.

    This class defines a unified interface for exporting computed FTLE data
    along with particle centroid coordinates, either as structured or
    unstructured datasets.

    Parameters
    ----------
    directory_path : str or os.PathLike
        Directory where output files will be saved. The folder will be created
        automatically if it does not exist.
    grid_shape : tuple of int, optional
        Shape of the underlying grid (e.g., ``(nx, ny)`` or ``(nx, ny, nz)``).
        If omitted, the data are assumed to represent an unstructured point
        cloud.
    """

    def __init__(
        self,
        directory_path: Union[str, os.PathLike],
        grid_shape: Optional[tuple[int, ...]] = None,
    ) -> None:
        self.path = directory_path
        try:
            os.makedirs(self.path, exist_ok=True)
        except OSError as e:
            print(f"Error creating output folder: {e}")

        self.grid_shape = grid_shape
        self.dim: Optional[int] = None

    @abstractmethod
    def write(
        self,
        filename: str,
        ftle_field: ArrayN,
        particles_centroid: ArrayNx2 | ArrayNx3,
    ) -> None:
        """
        Write the FTLE field to a file.

        Parameters
        ----------
        filename : str
            File name (without extension) or full file path for the output.
        ftle_field : ArrayN
            Array containing FTLE scalar values for all particle centroids.
        particles_centroid : ArrayNx2 or ArrayNx3
            Coordinates of particle centroids in 2D or 3D space.
        """
        ...


class MatWriter(FTLEWriter):
    """
    Writer class to export FTLE fields to MATLAB ``.mat`` files.

    The writer supports both structured and unstructured datasets. Structured
    grids are reshaped according to the specified ``grid_shape`` and saved as
    multidimensional arrays. Unstructured data are saved as flattened arrays.

    Parameters
    ----------
    directory_path : str or os.PathLike
        Directory where the ``.mat`` files will be stored.
    grid_shape : tuple of int, optional
        Shape of the computational grid (``(nx, ny)`` or ``(nx, ny, nz)``).
        If not provided, data are assumed to be unstructured.
    """

    def __init__(
        self,
        directory_path: Union[str, os.PathLike],
        grid_shape: Optional[tuple[int, ...]] = None,
    ) -> None:
        super().__init__(directory_path, grid_shape)

    def write(
        self,
        filename: str,
        ftle_field: ArrayN,
        particles_centroid: ArrayNx2 | ArrayNx3,
    ) -> None:
        """
        Save the FTLE field and particle centroid coordinates in a MATLAB
        ``.mat`` file.

        Parameters
        ----------
        filename : str
            Base name (without extension) for the output file.
        ftle_field : ArrayN
            Flattened FTLE values corresponding to each particle centroid.
        particles_centroid : ArrayNx2 or ArrayNx3
            Array of centroid coordinates in 2D or 3D space.

        Raises
        ------
        ValueError
            If ``grid_shape`` is provided but its length is not 2 or 3.
        """

        # Determine the dimensionality (2D or 3D)
        if self.dim is None:
            self.dim = particles_centroid.shape[1]

        mat_filename = os.path.join(self.path, filename + ".mat")

        if self.grid_shape:
            if len(self.grid_shape) == 2:
                nx, ny = self.grid_shape
                nz = 1
            elif len(self.grid_shape) == 3:
                nx, ny, nz = self.grid_shape
            else:
                raise ValueError(
                    f"Invalid grid_shape length {len(self.grid_shape)}. Must be 2 or 3."
                )

            # Use typing.cast to tell the linter that self.dim is now integer
            self.dim = cast(int, self.dim)

            ftle_field = ftle_field.reshape(nx, ny, nz)
            particles_centroid = particles_centroid.reshape(nx, ny, nz, self.dim)

            # Prepare MATLAB dictionary
            data = {
                "ftle": ftle_field,
                "x": particles_centroid[..., 0],
                "y": particles_centroid[..., 1],
            }

            # Add z only if 3D
            if self.dim == 3:
                data["z"] = particles_centroid[..., 2]

            savemat(mat_filename, data)

        else:
            # Unstructured grid
            data = {
                "ftle": ftle_field.ravel(),
                "x": particles_centroid[:, 0],
                "y": particles_centroid[:, 1],
            }

            if self.dim == 3:
                data["z"] = particles_centroid[:, 2]

            savemat(mat_filename, data)


class VTKWriter(FTLEWriter):
    """
    Writer class to export FTLE fields to VTK files for visualization with
    ParaView or other visualization tools.

    Structured grids are written as ``.vts`` files (VTK StructuredGrid),
    whereas unstructured data are written as ``.vtp`` files (VTK PolyData).

    Parameters
    ----------
    directory_path : str or os.PathLike
        Directory where the VTK files will be saved.
    grid_shape : tuple of int, optional
        Shape of the computational grid (``(nx, ny)`` or ``(nx, ny, nz)``).
        If omitted, data are treated as an unstructured cloud of points.
    """

    def __init__(
        self,
        directory_path: Union[str, os.PathLike],
        grid_shape: Optional[tuple[int, ...]] = None,
    ) -> None:
        super().__init__(directory_path, grid_shape)

    def write(
        self,
        filename: str,
        ftle_field: ArrayN,
        particles_centroid: ArrayNx2 | ArrayNx3,
    ) -> None:
        """
        Save the FTLE field and particle centroid coordinates as a VTK file.

        Parameters
        ----------
        filename : str
            Base name (without extension) for the output file.
        ftle_field : ArrayN
            Flattened FTLE values corresponding to each particle centroid.
        particles_centroid : ArrayNx2 or ArrayNx3
            Array of centroid coordinates in 2D or 3D space.

        Raises
        ------
        ValueError
            If ``grid_shape`` is provided but its length is not 2 or 3.
        """

        # Determine the dimensionality (2D or 3D)
        if self.dim is None:
            self.dim = particles_centroid.shape[1]

        vtk_filename = os.path.join(self.path, filename)

        # Structured grid
        if self.grid_shape is not None:
            if len(self.grid_shape) == 2:
                nx, ny = self.grid_shape
                nz = 1
                particles_centroid = particles_centroid.reshape(
                    nx, ny, nz, self.dim, order="F"
                )

                x = particles_centroid[..., 0]
                y = particles_centroid[..., 1]
                z = np.zeros_like(x)

                grid = pv.StructuredGrid(x, y, z)
                grid["ftle"] = ftle_field.ravel(order="F")
                grid.save(vtk_filename + ".vts")

            elif len(self.grid_shape) == 3:
                nx, ny, nz = self.grid_shape
                particles_centroid = particles_centroid.reshape(nx, ny, nz, self.dim)

                x = particles_centroid[..., 0]
                y = particles_centroid[..., 1]
                z = particles_centroid[..., 2]

                grid = pv.StructuredGrid(x, y, z)

                ftle_matrix = ftle_field.reshape((nx, ny, nz))
                ftle_cartesian = np.transpose(ftle_matrix, axes=(1, 0, 2))
                grid["ftle"] = ftle_cartesian.ravel(order="F")

                grid.save(vtk_filename + ".vts")
            else:
                raise ValueError(
                    f"Invalid grid_shape length {len(self.grid_shape)}. Must be 2 or 3."
                )
        else:
            if self.dim == 2:
                points = np.hstack(
                    [particles_centroid, np.zeros((particles_centroid.shape[0], 1))]
                )
            else:
                points = particles_centroid
            mesh = pv.PolyData(points)
            mesh["ftle"] = ftle_field.ravel()
            mesh.save(vtk_filename + ".vtp")


def create_writer(
    output_format: str,
    directory_path: str,
    grid_shape: Optional[tuple[int, ...]] = None,
) -> FTLEWriter:
    """
    Factory function to create an FTLE writer for the desired output format.

    Parameters
    ----------
    output_format : str
        Output format for the FTLE data. Must be either ``"mat"`` or ``"vtk"``.
    directory_path : str
        Directory path where the output files will be stored.
    grid_shape : tuple of int, optional
        Shape of the structured grid (``(nx, ny)`` or ``(nx, ny, nz)``).
        If omitted, the writer assumes an unstructured dataset.

    Returns
    -------
    FTLEWriter
        An instance of either :class:`MatWriter` or :class:`VTKWriter`.

    Raises
    ------
    ValueError
        If ``output_format`` is not recognized.
    """
    if output_format == "mat":
        return MatWriter(directory_path, grid_shape)
    elif output_format == "vtk":
        return VTKWriter(directory_path, grid_shape)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

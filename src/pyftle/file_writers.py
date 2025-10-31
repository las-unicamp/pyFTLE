import os
from abc import ABC, abstractmethod
from typing import Optional, Union, cast

import numpy as np
import pyvista as pv
from scipy.io import savemat

from pyftle.my_types import ArrayN, ArrayNx2, ArrayNx3


class FTLEWriter(ABC):
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

        Args:
            filename (str): Full name/path to the file.
            ftle_field: Array of the FTLE field to be saved.
            particles_centroid: Centroid coordinates of the particles.
        """
        ...


class MatWriter(FTLEWriter):
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
                "ftle": ftle_field.flatten(),
                "x": particles_centroid[:, 0],
                "y": particles_centroid[:, 1],
            }

            if self.dim == 3:
                data["z"] = particles_centroid[:, 2]

            savemat(mat_filename, data)


class VTKWriter(FTLEWriter):
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
                grid["ftle"] = ftle_field.flatten(order="F")
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
                grid["ftle"] = ftle_cartesian.flatten(order="F")

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
            mesh["ftle"] = ftle_field.flatten()
            mesh.save(vtk_filename + ".vtp")


def create_writer(
    output_format: str,
    directory_path: str,
    grid_shape: Optional[tuple[int, ...]] = None,
) -> FTLEWriter:
    """
    Creates an FTLE writer based on the specified output format.

    Args:
        output_format (str): The output format, either "mat" or "vtk".
        directory_path (str): Directory path to store output files.
        grid_shape (tuple): Shape of the grid-based data (optional, default is None).

    Returns:
        FTLEWriter: The corresponding writer object (MatWriter or VTKWriter).
    """
    if output_format == "mat":
        return MatWriter(directory_path, grid_shape)
    elif output_format == "vtk":
        return VTKWriter(directory_path, grid_shape)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

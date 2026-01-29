from unittest.mock import patch

import numpy as np
import pyvista as pv

from pyftle.file_writers import MatWriter, VTKWriter


# ----------------------------
# MatWriter tests
# ----------------------------
@patch("pyftle.file_writers.savemat")
def test_mat_writer_2d(mock_savemat, tmp_path, particles_centroid_2d, ftle_field_2d):
    writer = MatWriter(str(tmp_path), grid_shape=(2, 2, 1))
    writer.write("test_2d", ftle_field_2d, particles_centroid_2d)

    mock_savemat.assert_called_once()
    args, _ = mock_savemat.call_args
    mat_filename, saved_data = args

    assert mat_filename.endswith(".mat")
    assert "ftle" in saved_data
    assert "x" in saved_data
    assert "y" in saved_data
    assert "z" not in saved_data
    assert saved_data["ftle"].shape == (2, 2, 1)


@patch("pyftle.file_writers.savemat")
def test_mat_writer_3d(mock_savemat, tmp_path, particles_centroid_3d, ftle_field_3d):
    writer = MatWriter(str(tmp_path), grid_shape=(2, 2, 2))
    writer.write("test_3d", ftle_field_3d, particles_centroid_3d)

    mock_savemat.assert_called_once()
    args, _ = mock_savemat.call_args
    mat_filename, saved_data = args

    assert mat_filename.endswith(".mat")
    assert saved_data["ftle"].shape == (2, 2, 2)
    assert "x" in saved_data
    assert "y" in saved_data
    assert "z" in saved_data


@patch("pyftle.file_writers.savemat")
def test_mat_writer_unstructured(
    mock_savemat, tmp_path, particles_centroid_2d, ftle_field_2d
):
    writer = MatWriter(str(tmp_path), grid_shape=None)
    writer.write("test_unstructured", ftle_field_2d, particles_centroid_2d)

    mock_savemat.assert_called_once()
    args, _ = mock_savemat.call_args
    _, saved_data = args

    assert saved_data["x"].ndim == 1
    assert saved_data["y"].ndim == 1


# ----------------------------
# VTKWriter tests
# ----------------------------
@patch.object(pv.StructuredGrid, "save")
def test_vtk_writer_2d(mock_save, tmp_path, particles_centroid_2d, ftle_field_2d):
    writer = VTKWriter(str(tmp_path), grid_shape=(2, 2))
    writer.write("test_3d", ftle_field_2d, particles_centroid_2d)

    mock_save.assert_called_once()
    saved_path = mock_save.call_args[0][0]
    assert saved_path.endswith(".vts")


@patch.object(pv.StructuredGrid, "save")
def test_vtk_writer_3d(mock_save, tmp_path, particles_centroid_3d, ftle_field_3d):
    writer = VTKWriter(str(tmp_path), grid_shape=(2, 2, 2))
    writer.write("test_3d", ftle_field_3d, particles_centroid_3d)

    mock_save.assert_called_once()
    saved_path = mock_save.call_args[0][0]
    assert saved_path.endswith(".vts")


@patch.object(pv.PolyData, "save")
def test_vtk_writer_unstructured(
    mock_save, tmp_path, particles_centroid_2d, ftle_field_2d
):
    writer = VTKWriter(str(tmp_path), grid_shape=None)
    writer.write("test_unstructured", ftle_field_2d, particles_centroid_2d)

    mock_save.assert_called_once()
    saved_path = mock_save.call_args[0][0]
    assert saved_path.endswith(".vtp")


# ----------------------------
# Edge case: empty centroid
# ----------------------------
@patch("pyftle.file_writers.savemat")
def test_empty_particles_centroid(mock_savemat, tmp_path):
    writer = MatWriter(str(tmp_path), grid_shape=None)
    empty_centroid = np.empty((0, 2))
    writer.write("test_empty", np.array([]), empty_centroid)
    mock_savemat.assert_called_once()

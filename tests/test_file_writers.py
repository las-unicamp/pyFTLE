import os
import unittest
from unittest.mock import patch

import numpy as np
import pyvista as pv

from pyftle.file_writers import MatWriter, VTKWriter


class TestFTLEWriter(unittest.TestCase):
    def setUp(self):
        # Prepare mock directory path
        self.test_dir = "test_output"
        os.makedirs(self.test_dir, exist_ok=True)

        # Mock particle centroids (2D and 3D)
        self.particles_centroid_2d = np.array(
            [[0, 0], [1, 0], [0, 1], [1, 1]]  # 4 particles in 2D
        )
        self.particles_centroid_3d = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ]  # 8 particles in 3D
        )

        # Mock FTLE fields
        self.ftle_field_2d = np.array([1.0, 2.0, 3.0, 4.0])
        self.ftle_field_3d = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)

    def tearDown(self):
        # Clean up test directory after test
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)

    # ----------------------------
    # Tests for MatWriter
    # ----------------------------
    @patch("pyftle.file_writers.savemat")
    def test_mat_writer_2d(self, mock_savemat):
        writer = MatWriter(self.test_dir, grid_shape=(2, 2, 1))
        writer.write("test_2d", self.ftle_field_2d, self.particles_centroid_2d)

        mock_savemat.assert_called_once()
        args, _ = mock_savemat.call_args
        mat_filename, saved_data = args

        self.assertTrue(mat_filename.endswith(".mat"))
        self.assertIn("ftle", saved_data)
        self.assertIn("x", saved_data)
        self.assertIn("y", saved_data)
        self.assertNotIn("z", saved_data)  # Should not save z data
        self.assertEqual(saved_data["ftle"].shape, (2, 2, 1))

    @patch("pyftle.file_writers.savemat")
    def test_mat_writer_3d(self, mock_savemat):
        writer = MatWriter(self.test_dir, grid_shape=(2, 2, 2))
        writer.write("test_3d", self.ftle_field_3d, self.particles_centroid_3d)

        mock_savemat.assert_called_once()
        args, _ = mock_savemat.call_args
        mat_filename, saved_data = args

        self.assertTrue(mat_filename.endswith(".mat"))
        self.assertEqual(saved_data["ftle"].shape, (2, 2, 2))
        self.assertIn("x", saved_data)
        self.assertIn("y", saved_data)
        self.assertIn("z", saved_data)

    @patch("pyftle.file_writers.savemat")
    def test_mat_writer_unstructured(self, mock_savemat):
        writer = MatWriter(self.test_dir, grid_shape=None)
        writer.write(
            "test_unstructured", self.ftle_field_2d, self.particles_centroid_2d
        )

        mock_savemat.assert_called_once()
        args, _ = mock_savemat.call_args
        _, saved_data = args

        # Unstructured case → 1D arrays
        self.assertEqual(saved_data["x"].ndim, 1)
        self.assertEqual(saved_data["y"].ndim, 1)

    # ----------------------------
    # Tests for VTKWriter
    # ----------------------------
    @patch.object(pv.StructuredGrid, "save")
    def test_vtk_writer_2d(self, mock_save):
        writer = VTKWriter(self.test_dir, grid_shape=(2, 2))
        writer.write("test_3d", self.ftle_field_2d, self.particles_centroid_2d)

        mock_save.assert_called_once()
        saved_path = mock_save.call_args[0][0]
        self.assertTrue(saved_path.endswith(".vts"))

    @patch.object(pv.StructuredGrid, "save")
    def test_vtk_writer_3d(self, mock_save):
        writer = VTKWriter(self.test_dir, grid_shape=(2, 2, 2))
        writer.write("test_3d", self.ftle_field_3d, self.particles_centroid_3d)

        mock_save.assert_called_once()
        saved_path = mock_save.call_args[0][0]
        self.assertTrue(saved_path.endswith(".vts"))

    @patch.object(pv.PolyData, "save")
    def test_vtk_writer_unstructured(self, mock_save):
        writer = VTKWriter(self.test_dir, grid_shape=None)
        writer.write(
            "test_unstructured", self.ftle_field_2d, self.particles_centroid_2d
        )

        mock_save.assert_called_once()
        saved_path = mock_save.call_args[0][0]
        self.assertTrue(saved_path.endswith(".vtp"))

    # ----------------------------
    # Edge case: empty centroid
    # ----------------------------
    @patch("pyftle.file_writers.savemat")
    def test_empty_particles_centroid(self, mock_savemat):
        writer = MatWriter(self.test_dir, grid_shape=None)
        empty_centroid = np.empty((0, 2))
        writer.write("test_empty", np.array([]), empty_centroid)
        mock_savemat.assert_called_once()


if __name__ == "__main__":
    unittest.main()

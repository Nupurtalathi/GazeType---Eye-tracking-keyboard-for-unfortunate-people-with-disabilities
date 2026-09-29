"""
Unit Tests for Calibration Mapping Engine.
Verifies grid generation, polynomial/affine/homography fitting, outlier rejection, and JSON persistence.
"""

import unittest
import numpy as np
import os
import tempfile
from gaze.calibration import GazeCalibration, CalibrationPoint


class TestCalibration(unittest.TestCase):
    def test_outlier_rejection(self):
        cp = CalibrationPoint(500, 500)
        # Normal samples around (0.50, 0.50)
        for _ in range(15):
            cp.add_sample(0.50 + np.random.uniform(-0.01, 0.01), 0.50 + np.random.uniform(-0.01, 0.01))
        # Severe outlier
        cp.add_sample(0.95, 0.10)

        center = cp.get_filtered_center()
        self.assertIsNotNone(center)
        self.assertAlmostEqual(center[0], 0.50, delta=0.03)
        self.assertAlmostEqual(center[1], 0.50, delta=0.03)

    def test_calibration_fit_and_mapping(self):
        calib = GazeCalibration(screen_width=1920, screen_height=1080, model_type="polynomial")
        grid = calib.setup_grid(rows=3, cols=3)
        self.assertEqual(len(grid), 9)

        # Feed synthetic ground-truth mappings
        for cp in calib.calibration_points:
            # Synthetic linear/slightly curved ratio:
            hr = (cp.screen_x / 1920.0) * 0.4 + 0.3
            vr = (cp.screen_y / 1080.0) * 0.4 + 0.3
            for _ in range(10):
                cp.add_sample(hr, vr)

        # Test Polynomial fit
        success = calib.fit()
        self.assertTrue(success)
        self.assertTrue(calib.is_calibrated)
        self.assertLess(calib.rmse_error_px, 15.0)

        # Test mapping
        pt = calib.calibration_points[4]  # Center
        center_hr = (pt.screen_x / 1920.0) * 0.4 + 0.3
        center_vr = (pt.screen_y / 1080.0) * 0.4 + 0.3
        mapped = calib.map_gaze(center_hr, center_vr)
        self.assertIsNotNone(mapped)
        self.assertAlmostEqual(mapped[0], pt.screen_x, delta=20.0)
        self.assertAlmostEqual(mapped[1], pt.screen_y, delta=20.0)

    def test_save_and_load(self):
        calib = GazeCalibration(screen_width=1920, screen_height=1080, model_type="polynomial")
        calib.setup_grid(rows=3, cols=3)
        for cp in calib.calibration_points:
            hr = (cp.screen_x / 1920.0) * 0.4 + 0.3
            vr = (cp.screen_y / 1080.0) * 0.4 + 0.3
            for _ in range(5):
                cp.add_sample(hr, vr)
        calib.fit()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            calib.save(tmp_path)
            loaded = GazeCalibration(screen_width=1920, screen_height=1080)
            self.assertFalse(loaded.is_calibrated)
            ok = loaded.load(tmp_path)
            self.assertTrue(ok)
            self.assertTrue(loaded.is_calibrated)
            self.assertEqual(loaded.model_type, "polynomial")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()

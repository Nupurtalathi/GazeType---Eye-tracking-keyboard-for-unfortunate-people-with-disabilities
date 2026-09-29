"""
Full End-to-End Pipeline Integration Test.
Verifies coordination among GazeDetector, Calibration, FixationDetector,
DwellEngine, RegionManager, GazeLogger, and Heatmap generation.
"""

import unittest
import numpy as np
import tempfile
import os
import time

from config.config import TrackerConfig
from gaze.calibration import GazeCalibration
from gaze.smoothing import GazeSmoother
from fixation.fixation_detector import FixationDetector
from dwell.dwell_engine import DwellEngine, DwellState
from regions.region_manager import RegionManager, Region
from logging_util.gaze_logger import GazeLogger
from visualization.heatmap import GazeHeatmap


class TestPipelineIntegration(unittest.TestCase):
    def test_full_pipeline_flow(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. Config
            config = TrackerConfig(
                screen_width=1920,
                screen_height=1080,
                session_log_dir=tmp_dir,
                dwell_threshold=1.00
            )

            # 2. Calibration
            calib = GazeCalibration(config.screen_width, config.screen_height, model_type="polynomial")
            grid = calib.setup_grid(rows=3, cols=3)
            for cp in calib.calibration_points:
                hr = (cp.screen_x / 1920.0) * 0.4 + 0.3
                vr = (cp.screen_y / 1080.0) * 0.4 + 0.3
                for _ in range(5):
                    cp.add_sample(hr, vr)
            self.assertTrue(calib.fit())

            # 3. Smoother
            smoother = GazeSmoother()

            # 4. Fixation & Dwell
            fixation = FixationDetector(fixation_radius=50.0, min_fixation_time=0.10)
            dwell_triggered_flag = False

            def on_dwell_cb(anchor, t):
                nonlocal dwell_triggered_flag
                dwell_triggered_flag = True

            dwell = DwellEngine(dwell_threshold=0.50, on_dwell_callback=on_dwell_cb)

            # 5. Region Manager
            regions = RegionManager(default_dwell_time=0.50)
            key_activated = None

            def on_key_cb(key_id):
                nonlocal key_activated
                key_activated = key_id

            regions.register_region(Region(id="KEY_A", x=900, y=500, width=120, height=80, callback=on_key_cb))

            # 6. Logger
            logger = GazeLogger(log_dir=tmp_dir)
            logger.start_session("integration_test_session")

            # 7. Simulate Gaze looking at KEY_A for 0.7 seconds
            t_base = 100.0
            dt = 0.033  # ~30 FPS

            # Mapping input that yields ~ (960, 540)
            sim_hr = (960.0 / 1920.0) * 0.4 + 0.3
            sim_vr = (540.0 / 1080.0) * 0.4 + 0.3

            for step in range(25):  # 25 * 0.033s = ~0.825s
                cur_t = t_base + step * dt
                # Gaze mapping
                raw_pos = calib.map_gaze(sim_hr, sim_vr)
                self.assertIsNotNone(raw_pos)

                # Smoothing
                sx, sy = smoother.filter(raw_pos[0], raw_pos[1], cur_t)

                # Fixation
                is_fix = fixation.update(sx, sy, cur_t, is_valid=True)

                # Dwell
                res = dwell.update(
                    gaze_x=sx, gaze_y=sy, is_fixating=is_fix,
                    is_blinking=False, is_face_detected=True,
                    is_valid_confidence=True, timestamp=cur_t
                )

                # Region
                regions.update(sx, sy, is_valid=True, timestamp=cur_t)

                # Log
                logger.record_gaze_point(sx, sy, cur_t)
                if res.triggered:
                    logger.log_event("dwell_triggered", sx, sy, sim_hr, sim_vr, res.dwell_time)

            # Verify outcomes
            self.assertTrue(dwell_triggered_flag, "Dwell engine should have triggered")
            self.assertEqual(key_activated, "KEY_A", "RegionManager should have triggered KEY_A")

            # Verify logger output
            logger.stop_session()
            self.assertTrue(os.path.exists(logger.csv_path))
            with open(logger.csv_path, "r", encoding="utf-8") as f:
                csv_content = f.read()
            self.assertIn("dwell_triggered", csv_content)

            # Verify Heatmap generation
            pts = logger.get_session_points()
            self.assertGreater(len(pts), 0)
            heatmap_img = GazeHeatmap.generate(pts, width=640, height=360)
            self.assertEqual(heatmap_img.shape, (360, 640, 3))


if __name__ == "__main__":
    unittest.main()

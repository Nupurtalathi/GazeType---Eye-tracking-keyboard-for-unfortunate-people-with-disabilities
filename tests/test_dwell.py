"""
Unit Tests for Dwell Engine and Region Manager.
Verifies time accumulation, threshold triggers, blink grace period pauses,
face loss resets, and region manager callbacks.
"""

import unittest
from dwell.dwell_engine import DwellEngine, DwellState, DwellResult
from regions.region_manager import RegionManager, Region


class TestDwellEngine(unittest.TestCase):
    def setUp(self):
        self.callback_called = False
        self.callback_point = None

        def dummy_cb(anchor, dwell_t):
            self.callback_called = True
            self.callback_point = anchor

        self.engine = DwellEngine(
            dwell_threshold=1.50,
            blink_reset_time=0.50,
            cooldown_time=0.60,
            on_dwell_callback=dummy_cb
        )

    def test_dwell_accumulation_and_trigger(self):
        t = 0.0
        # Initialize
        res = self.engine.update(500, 300, is_fixating=True, is_blinking=False, is_face_detected=True, is_valid_confidence=True, timestamp=t)
        self.assertEqual(res.state, DwellState.FIXATING)
        self.assertFalse(res.triggered)

        # Advance to 1.0s (dwelling)
        t = 1.0
        res = self.engine.update(500, 300, is_fixating=True, is_blinking=False, is_face_detected=True, is_valid_confidence=True, timestamp=t)
        self.assertEqual(res.state, DwellState.DWELLING)
        self.assertAlmostEqual(res.dwell_time, 1.0, places=2)
        self.assertFalse(res.triggered)

        # Advance to 1.55s (threshold reached)
        t = 1.55
        res = self.engine.update(500, 300, is_fixating=True, is_blinking=False, is_face_detected=True, is_valid_confidence=True, timestamp=t)
        self.assertEqual(res.state, DwellState.TRIGGERED)
        self.assertTrue(res.triggered)
        self.assertTrue(self.callback_called)
        self.assertEqual(self.callback_point, (500, 300))

    def test_fixation_loss_resets_dwell(self):
        # Build 1.0s of dwell
        self.engine.update(500, 300, is_fixating=True, is_blinking=False, is_face_detected=True, is_valid_confidence=True, timestamp=0.0)
        self.engine.update(500, 300, is_fixating=True, is_blinking=False, is_face_detected=True, is_valid_confidence=True, timestamp=1.0)
        self.assertAlmostEqual(self.engine.dwell_time, 1.0, places=2)

        # Fixation breaks (user looked away)
        res = self.engine.update(700, 400, is_fixating=False, is_blinking=False, is_face_detected=True, is_valid_confidence=True, timestamp=1.05)
        self.assertEqual(res.state, DwellState.TRACKING)
        self.assertEqual(res.dwell_time, 0.0)
        self.assertFalse(self.callback_called)

    def test_short_blink_pauses_without_reset(self):
        # Accumulate 0.8s dwell
        self.engine.update(500, 300, is_fixating=True, is_blinking=False, is_face_detected=True, is_valid_confidence=True, timestamp=0.0)
        self.engine.update(500, 300, is_fixating=True, is_blinking=False, is_face_detected=True, is_valid_confidence=True, timestamp=0.8)
        self.assertAlmostEqual(self.engine.dwell_time, 0.8, places=2)

        # User blinks for 200ms (< 500ms threshold)
        res = self.engine.update(500, 300, is_fixating=True, is_blinking=True, is_face_detected=True, is_valid_confidence=True, timestamp=0.9)
        self.assertEqual(res.state, DwellState.PAUSED_BLINK)
        self.assertAlmostEqual(self.engine.dwell_time, 0.8, places=2)

        res = self.engine.update(500, 300, is_fixating=True, is_blinking=True, is_face_detected=True, is_valid_confidence=True, timestamp=1.0)
        self.assertEqual(res.state, DwellState.PAUSED_BLINK)
        # Gaze time did not increment during blink
        self.assertAlmostEqual(self.engine.dwell_time, 0.8, places=2)

        # Eye re-opens at 1.05
        res = self.engine.update(500, 300, is_fixating=True, is_blinking=False, is_face_detected=True, is_valid_confidence=True, timestamp=1.05)
        # Continues accumulating from 0.8s
        self.assertGreaterEqual(self.engine.dwell_time, 0.8)

    def test_long_blink_resets_dwell(self):
        # Accumulate 0.8s dwell
        self.engine.update(500, 300, is_fixating=True, is_blinking=False, is_face_detected=True, is_valid_confidence=True, timestamp=0.0)
        self.engine.update(500, 300, is_fixating=True, is_blinking=False, is_face_detected=True, is_valid_confidence=True, timestamp=0.8)

        # Eye closed at 0.85
        self.engine.update(500, 300, is_fixating=True, is_blinking=True, is_face_detected=True, is_valid_confidence=True, timestamp=0.85)
        # Eye remains closed past blink_reset_time (0.5s -> 0.85 + 0.55 = 1.40)
        res = self.engine.update(500, 300, is_fixating=True, is_blinking=True, is_face_detected=True, is_valid_confidence=True, timestamp=1.45)
        self.assertEqual(res.state, DwellState.INVALID)
        self.assertEqual(self.engine.dwell_time, 0.0)

    def test_face_loss_resets_immediately(self):
        self.engine.update(500, 300, is_fixating=True, is_blinking=False, is_face_detected=True, is_valid_confidence=True, timestamp=0.0)
        self.engine.update(500, 300, is_fixating=True, is_blinking=False, is_face_detected=True, is_valid_confidence=True, timestamp=0.9)

        res = self.engine.update(500, 300, is_fixating=True, is_blinking=False, is_face_detected=False, is_valid_confidence=True, timestamp=0.95)
        self.assertEqual(res.state, DwellState.FACE_LOST)
        self.assertEqual(self.engine.dwell_time, 0.0)


class TestRegionManager(unittest.TestCase):
    def test_region_dwell_trigger(self):
        triggered_region = None

        def on_dwell(reg_id):
            nonlocal triggered_region
            triggered_region = reg_id

        manager = RegionManager(default_dwell_time=1.0)
        manager.on_dwell(on_dwell)

        # Register button "KEY_A" at (100, 100), width 200, height 100
        btn = Region(id="KEY_A", x=100, y=100, width=200, height=100, dwell_time=1.0)
        manager.register_region(btn)

        # Gaze enters region at (150, 150)
        manager.update(150, 150, is_valid=True, timestamp=0.0)
        self.assertIsNone(triggered_region)

        # Gaze remains for 0.5s
        manager.update(155, 145, is_valid=True, timestamp=0.5)
        self.assertIsNone(triggered_region)

        # Gaze reaches 1.05s inside region
        res = manager.update(152, 148, is_valid=True, timestamp=1.05)
        self.assertEqual(res, "KEY_A")
        self.assertEqual(triggered_region, "KEY_A")


if __name__ == "__main__":
    unittest.main()

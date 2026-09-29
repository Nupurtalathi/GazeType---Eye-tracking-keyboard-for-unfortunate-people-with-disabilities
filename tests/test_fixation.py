"""
Unit Tests for Fixation Detection.
Verifies spatial clustering within radius and temporal thresholding.
"""

import unittest
from fixation.fixation_detector import FixationDetector


class TestFixationDetector(unittest.TestCase):
    def setUp(self):
        # 50 px radius, 100 ms min fixation time
        self.detector = FixationDetector(fixation_radius=50.0, min_fixation_time=0.10)

    def test_single_point_not_fixating(self):
        result = self.detector.update(500, 300, timestamp=1.00)
        self.assertFalse(result)
        self.assertFalse(self.detector.is_active())

    def test_stable_cluster_achieves_fixation(self):
        # Cluster within 5px jitter over 150 ms
        points = [
            (500, 300, 1.00),
            (502, 301, 1.04),
            (501, 299, 1.08),
            (503, 302, 1.12),
            (500, 300, 1.16)  # 160 ms total > 100 ms
        ]
        for x, y, t in points:
            res = self.detector.update(x, y, timestamp=t)

        self.assertTrue(self.detector.is_active())
        self.assertAlmostEqual(self.detector.get_duration(), 0.16, places=2)
        centroid = self.detector.get_centroid()
        self.assertIsNotNone(centroid)
        self.assertAlmostEqual(centroid[0], 501.2, places=1)
        self.assertAlmostEqual(centroid[1], 300.4, places=1)

    def test_saccade_breaks_fixation(self):
        # First achieve fixation
        self.detector.update(500, 300, timestamp=1.00)
        self.detector.update(501, 301, timestamp=1.12)
        self.assertTrue(self.detector.is_active())

        # Saccade jump 300 px away
        res = self.detector.update(800, 300, timestamp=1.15)
        self.assertFalse(res)
        self.assertFalse(self.detector.is_active())
        self.assertEqual(self.detector.get_duration(), 0.0)
        # New centroid should be at the saccade landing point
        self.assertEqual(self.detector.get_centroid(), (800.0, 300.0))

    def test_invalid_gaze_ignored(self):
        self.detector.update(500, 300, timestamp=1.00)
        self.detector.update(501, 301, timestamp=1.12)
        self.assertTrue(self.detector.is_active())

        # Frame where face was lost (is_valid=False)
        res = self.detector.update(501, 301, timestamp=1.15, is_valid=False)
        # Should not disrupt previously established state immediately
        self.assertTrue(res)


if __name__ == "__main__":
    unittest.main()

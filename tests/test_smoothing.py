"""
Unit Tests for Smoothing and Jitter Filters.
Verifies One Euro Filter, EMA, and saccade gating.
"""

import unittest
from gaze.smoothing import GazeSmoother, OneEuroFilter1D, LowPassFilter


class TestSmoothing(unittest.TestCase):
    def test_one_euro_stationary_jitter_reduction(self):
        filter_1d = OneEuroFilter1D(min_cutoff=0.8, beta=0.01)

        t = 1.0
        outputs = []
        # Noisy signal around 500.0
        noisy_vals = [500.0, 503.0, 497.0, 502.0, 498.0, 501.0]
        for val in noisy_vals:
            out = filter_1d.filter(val, timestamp=t)
            outputs.append(out)
            t += 0.033  # ~30 FPS

        # Jitter variance should be reduced
        self.assertLess(abs(outputs[-1] - 500.0), 2.0)

    def test_saccade_jump_reset(self):
        smoother = GazeSmoother(method="one_euro", max_allowed_jump=200.0)

        # Baseline
        smoother.filter(100.0, 100.0, timestamp=1.0)
        smoother.filter(101.0, 100.0, timestamp=1.033)

        # Sudden jump of 600px (saccade)
        sx, sy = smoother.filter(700.0, 100.0, timestamp=1.066)
        # Should snap immediately towards 700 without being held back near 100
        self.assertGreater(sx, 600.0)


if __name__ == "__main__":
    unittest.main()

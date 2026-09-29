"""Vertical accuracy: calibration learns this user's eyelid-vs-gaze relation."""
import unittest

import numpy as np

from gaze.calibration import GazeCalibration
from keyboard.layout import build_layout
from keyboard.selector import KeySelector

W, H = 1920, 1080


def observe(sx, sy, rng):
    u = (sx / W - 0.5) * 2
    v = (sy / H - 0.5) * 2
    hr = 0.5 + 0.16 * u + rng.normal(0, 0.002)
    # iris vertical signal is weak and saturates looking down (lids cover the iris) ...
    vr = 0.5 + 0.05 * np.tanh(1.5 * v) + rng.normal(0, 0.004)
    # ... but the eyelid aperture keeps changing
    lid = 0.30 - 0.07 * v + rng.normal(0, 0.002)
    return hr, vr, (0.0 + rng.normal(0, 1e-4), 0.0 + rng.normal(0, 1e-4), lid)


class TestVertical(unittest.TestCase):
    def _cal(self, with_lid, seed=0):
        rng = np.random.default_rng(seed)
        c = GazeCalibration(W, H)
        c.setup_grid(4, 4, 0.05, 0.05)
        c.add_validation_points(0.03)
        for cp in c.calibration_points:
            for _ in range(30):
                hr, vr, extra = observe(cp.screen_x, cp.screen_y, rng)
                cp.add_sample(hr, vr, extra if with_lid else extra[:2])
        self.assertTrue(c.fit(), c.last_fit_message)
        return c

    def test_lid_feature_selected_and_improves_vertical(self):
        c_lid, c_no = self._cal(True), self._cal(False)
        self.assertTrue(c_lid._use_head)
        self.assertTrue(c_lid._extra_mask[2])
        self.assertLess(c_lid.loo_error_px, c_no.loo_error_px)
        self.assertIn("eyelid", c_lid.last_fit_message)

    def test_bottom_row_keys_hit(self):
        c = self._cal(True)
        keys = build_layout(W, H)
        sel = KeySelector(keys, W, H)
        bottom = [k for k in keys if k.kind != "display" and k.y + k.h >= H - 1]
        rng = np.random.default_rng(5)
        hits = 0
        for k in bottom:
            hr, vr, extra = observe(k.cx, k.cy, rng)
            hits += sel.key_at(*c.map_gaze(hr, vr, extra)).id == k.id
        self.assertEqual(hits, len(bottom))


if __name__ == "__main__":
    unittest.main()

"""Edge reach: calibration must map gaze at the extreme borders onto the border keys."""
import unittest

import numpy as np

from gaze.calibration import GazeCalibration
from keyboard.layout import build_layout
from keyboard.selector import KeySelector

W, H = 1920, 1080


def eye_response(sx, sy):
    """Synthetic eye: iris movement saturates towards the edges (as real eyes/lids do)."""
    u = (sx / W - 0.5) * 2
    v = (sy / H - 0.5) * 2
    hr = 0.5 + 0.16 * np.tanh(1.3 * u)
    vr = 0.5 + 0.12 * np.tanh(1.6 * v)        # stronger compression looking down/up
    return hr, vr


def run_calibration(validation=True, drop_bottom=False, seed=0):
    rng = np.random.default_rng(seed)
    c = GazeCalibration(W, H)
    c.setup_grid(4, 4, 0.05, 0.05)
    if validation:
        c.add_validation_points(0.03)
    for cp in c.calibration_points:
        if drop_bottom and cp.screen_y > H * 0.9:
            continue
        hr, vr = eye_response(cp.screen_x, cp.screen_y)
        for _ in range(30):
            cp.add_sample(hr + rng.normal(0, 0.002), vr + rng.normal(0, 0.002))
    return c, c.fit()


class TestEdgeReach(unittest.TestCase):
    def test_border_keys_reachable(self):
        c, ok = run_calibration()
        self.assertTrue(ok, c.last_fit_message)
        keys = build_layout(W, H)
        sel = KeySelector(keys, W, H)
        # looking at the centre of every key in the bottom 2 rows and right 2 columns
        border = [k for k in keys if k.kind != "display" and
                  (k.y >= H - 2 * keys[-1].h - 1 or k.x >= W - 2 * keys[-1].w - 1)]
        self.assertGreater(len(border), 15)
        for k in border:
            gx, gy = c.map_gaze(*eye_response(k.cx, k.cy))
            self.assertEqual(sel.key_at(gx, gy).id, k.id, f"{k.id}: mapped to ({gx:.0f},{gy:.0f})")

    def test_validation_reports_edge_error(self):
        c, ok = run_calibration()
        self.assertTrue(ok)
        self.assertIsNotNone(c.validation_error_px)
        self.assertLess(c.validation_error_px, 120)

    def test_cubic_beats_quadratic_at_edges(self):
        c, ok = run_calibration()
        self.assertEqual(c._poly_degree, 3)
        c2, ok2 = run_calibration(validation=False)          # 16 grid points only
        self.assertTrue(ok2)
        errs = lambda cal: [np.hypot(*(np.array(cal.map_gaze(*eye_response(x, y))) - (x, y)))
                            for x in (60, 1860) for y in (40, 1040)]
        self.assertLess(np.mean(errs(c)), np.mean(errs(c2)))

    def test_missing_bottom_row_is_rejected(self):
        c, ok = run_calibration(drop_bottom=True)
        self.assertFalse(ok)
        self.assertIn("bottom edge", c.last_fit_message)


if __name__ == "__main__":
    unittest.main()


class TestHeadCompensation(unittest.TestCase):
    """User turns the head towards the edges; eyes rotate correspondingly less."""

    @staticmethod
    def observe(sx, sy, head_share, rng=None):
        u = (sx / W - 0.5) * 2
        v = (sy / H - 0.5) * 2
        yaw, pitch = 0.05 * head_share * u, 0.04 * head_share * v          # head proxies
        eye_u, eye_v = u * (1 - 0.6 * head_share), v * (1 - 0.6 * head_share)  # eyes do the rest
        n = (lambda: rng.normal(0, 0.0015)) if rng is not None else (lambda: 0.0)
        return 0.5 + 0.16 * eye_u + n(), 0.5 + 0.12 * eye_v + n(), (yaw + n() * 0.3, pitch + n() * 0.3)

    def _calibrate(self, head_share_during_calibration):
        rng = np.random.default_rng(3)
        c = GazeCalibration(W, H)
        c.setup_grid(4, 4, 0.05, 0.05)
        c.add_validation_points(0.03)
        for cp in c.calibration_points:
            # head share varies a bit from target to target, as it does in real life
            share = head_share_during_calibration * rng.uniform(0.5, 1.5)
            for _ in range(30):
                hr, vr, head = self.observe(cp.screen_x, cp.screen_y, share, rng)
                cp.add_sample(hr, vr, head)
        self.assertTrue(c.fit(), c.last_fit_message)
        return c

    def test_head_terms_used_and_fix_under_reach(self):
        c = self._calibrate(0.4)
        self.assertTrue(c._use_head, c.last_fit_message)
        # at use time the user turns the head MORE than during calibration
        bottom_right = (1800.0, 1000.0)
        hr, vr, head = self.observe(*bottom_right, head_share=0.9)
        with_head = np.array(c.map_gaze(hr, vr, head))
        without = np.array(c.map_gaze(hr, vr, None))
        err_with = np.hypot(*(with_head - bottom_right))
        err_without = np.hypot(*(without - bottom_right))
        self.assertLess(err_with, 0.5 * err_without)
        self.assertLess(err_with, 120)

    def test_still_head_disables_head_terms(self):
        c = self._calibrate(0.0)
        self.assertFalse(c._use_head)

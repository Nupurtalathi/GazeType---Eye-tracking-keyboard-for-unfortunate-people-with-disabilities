"""Reach tuning: outer keys must be reachable at comfortable eye angles."""
import unittest

import numpy as np

from gaze.reach import ReachModel, fit_reach, reach_targets
from keyboard.layout import build_layout
from keyboard.selector import KeySelector

W, H = 1920, 1080


def compressed(x, y, kx=0.72, ky=0.62):
    """Live gaze lands short of the edges (edge effect, not a uniform scale)."""
    cx, cy = W / 2, H / 2
    dx, dy = (x - cx) / cx, (y - cy) / cy
    return cx + cx * dx * (1 - (1 - kx) * abs(dx)), cy + cy * dy * (1 - (1 - ky) * abs(dy))


class TestReach(unittest.TestCase):
    def setUp(self):
        self.keys = build_layout(W, H)
        self.sel = KeySelector(self.keys, W, H)
        meas = {n: (compressed(tx, ty), (tx, ty)) for n, tx, ty in reach_targets(self.keys, W, H)}
        self.model = fit_reach(meas, W, H)

    def _hit(self, fn):
        ks = [k for k in self.keys if k.kind != "display"]
        return sum(self.sel.key_at(*fn(k.cx, k.cy)).id == k.id for k in ks), len(ks)

    def test_without_tuning_outer_keys_are_missed(self):
        hit, n = self._hit(compressed)
        self.assertLess(hit, n - 8)

    def test_with_tuning_every_key_is_reached(self):
        hit, n = self._hit(lambda x, y: self.model.apply(*compressed(x, y)))
        self.assertEqual(hit, n)

    def test_gains_are_per_direction(self):
        self.assertGreater(self.model.g_right, 1.1)
        self.assertGreater(self.model.g_down, 1.1)
        self.assertLess(self.model.corner_error_px, 60)

    def test_centre_untouched(self):
        x, y = self.model.apply(W / 2, H / 2)
        self.assertAlmostEqual(x, W / 2, delta=1)
        self.assertAlmostEqual(y, H / 2, delta=1)

    def test_bad_measurement_keeps_gain_1(self):
        meas = {"centre": ((W / 2, H / 2), (W / 2, H / 2)), "right": ((W / 2 + 5, H / 2), (W - 120, H / 2))}
        m = fit_reach(meas, W, H)
        self.assertEqual(m.g_right, 1.0)

    def test_roundtrip(self):
        m2 = ReachModel.from_dict(self.model.to_dict())
        self.assertEqual(m2.apply(100, 900), self.model.apply(100, 900))


class TestSticky(unittest.TestCase):
    def test_jitter_across_row_boundary_keeps_key(self):
        keys = build_layout(W, H)
        sel = KeySelector(keys, W, H)
        a = [k for k in keys if k.id == "Q"][0]
        ev = None
        rng = np.random.default_rng(0)
        for i in range(60):   # gaze centred 20 px above Q's top edge region with 45 px vertical noise
            y = a.y + 0.15 * a.h + rng.normal(0, 45)
            ev = ev or sel.update(a.cx, y, True, i / 30)
        self.assertIsNotNone(ev)
        self.assertEqual(ev.key.id, "Q")


if __name__ == "__main__":
    unittest.main()

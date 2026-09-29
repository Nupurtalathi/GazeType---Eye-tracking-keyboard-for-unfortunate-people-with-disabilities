"""v7 typing stability: stabiliser + selector rules."""
import unittest

import numpy as np

from gaze.stabilizer import GazeStabilizer
from keyboard.layout import build_layout
from keyboard.selector import KeySelector

W, H = 1920, 1080


def ar_noise(n, sdx, sdy, rng):
    ex = ey = 0.0
    out = []
    for _ in range(n):
        ex = 0.6 * ex + 0.8 * rng.normal(0, sdx)
        ey = 0.6 * ey + 0.8 * rng.normal(0, sdy)
        out.append((ex, ey))
    return np.array(out)


class TestStabilizer(unittest.TestCase):
    def test_fixation_jitter_suppressed_and_no_key_flicker(self):
        rng = np.random.default_rng(0)
        s = GazeStabilizer(240, 163)
        e = ar_noise(1800, 48, 39, rng)
        out = np.array([s.update(1000 + e[i, 0], 500 + e[i, 1], i / 30) for i in range(1800)])[30:]
        kx = np.floor((out[:, 0] - 880) / 240)
        ky = np.floor((out[:, 1] - 418.5) / 163)
        switches = int(np.sum((np.diff(kx) != 0) | (np.diff(ky) != 0)))
        self.assertLessEqual(switches, 3)                         # per minute (v6 filter: ~24)
        self.assertLess(out[:, 1].std(), 30)

    def test_small_deliberate_move_is_fast(self):
        rng = np.random.default_rng(1)
        for dx, dy in [(240, 0), (0, 163), (-240, 0), (0, -163)]:
            s = GazeStabilizer(240, 163)
            e = ar_noise(90, 48, 39, rng)
            for i in range(40):
                s.update(1000 + e[i, 0], 500 + e[i, 1], i / 30)
            for j in range(40, 90):
                o = s.update(1000 + dx + e[j, 0], 500 + dy + e[j, 1], j / 30)
                if abs(o[0] - 1000 - dx) < 120 and abs(o[1] - 500 - dy) < 81:
                    break
            self.assertLessEqual((j - 40) / 30, 0.25, (dx, dy))       # one key in <= 250 ms

    def test_single_spike_rejected(self):
        s = GazeStabilizer(240, 163)
        for i in range(30):
            s.update(1000, 500, i / 30)
        o = s.update(1400, 500, 1.0)                                 # one-frame glitch
        for i in range(5):
            o = s.update(1000, 500, 1.03 + i / 30)
        self.assertAlmostEqual(o[0], 1000, delta=15)
        self.assertEqual(s.saccade_count, 0)

    def test_threshold_adapts_to_noise(self):
        rng = np.random.default_rng(2)
        quiet, noisy = GazeStabilizer(240, 163), GazeStabilizer(240, 163)
        for i, (a, b) in enumerate(zip(ar_noise(600, 10, 10, rng), ar_noise(600, 120, 110, rng))):
            quiet.update(1000 + a[0], 500 + a[1], i / 30)
            noisy.update(1000 + b[0], 500 + b[1], i / 30)
        self.assertGreater(noisy.thresholds()[1], quiet.thresholds()[1])


class TestSelectorStability(unittest.TestCase):
    def setUp(self):
        self.keys = build_layout(W, H)
        self.k = {k.id: k for k in self.keys}

    def _look(self, sel, key, seconds, t0, **kw):
        evs = []
        for i in range(int(seconds * 30)):
            e = sel.update(key.cx, key.cy, True, t0 + i / 30, stable=True, **kw)
            if e:
                evs.append(e.key.id)
        return evs, t0 + seconds

    def test_lingering_after_selection_does_not_retype(self):
        sel = KeySelector(self.keys, W, H, repeat_factor=1.8)
        evs, _ = self._look(sel, self.k["E"], 2.5, 0.0)            # fires once at ~1.1 s
        self.assertEqual(evs, ["E"])                                 # v6 fired again at ~2.5 s

    def test_double_letter_by_holding(self):
        sel = KeySelector(self.keys, W, H, repeat_factor=1.8)
        evs, _ = self._look(sel, self.k["E"], 3.6, 0.0)
        self.assertEqual(evs, ["E", "E"])

    def test_repeat_requires_exit_when_disabled(self):
        sel = KeySelector(self.keys, W, H, repeat_factor=0)
        evs, t = self._look(sel, self.k["L"], 5.0, 0.0)
        self.assertEqual(evs, ["L"])
        _, t = self._look(sel, self.k["K"], 0.4, t)                 # look away (proper exit)
        evs, _ = self._look(sel, self.k["L"], 1.5, t)
        self.assertEqual(evs, ["L"])

    def test_small_move_after_saccade_switches_quickly(self):
        sel = KeySelector(self.keys, W, H)
        a, b = self.k["A"], self.k["B"]
        self._look(sel, a, 0.5, 0.0)
        # land just inside B, well within A's sticky zone, right after a saccade
        x, y = b.x + 0.1 * b.w, b.cy
        for i in range(3):
            sel.update(x, y, True, 0.5 + i / 30, stable=True, last_saccade_t=0.5)
        self.assertIs(sel.state.active_key, b)

    def test_jitter_without_saccade_stays_on_key(self):
        sel = KeySelector(self.keys, W, H)
        a, b = self.k["A"], self.k["B"]
        self._look(sel, a, 0.5, 0.0)
        for i in range(20):
            sel.update(b.x + 0.1 * b.w, b.cy, True, 0.5 + i / 30, stable=True)   # no saccade reported
        self.assertIs(sel.state.active_key, a)

    def test_unstable_frames_freeze_dwell(self):
        sel = KeySelector(self.keys, W, H)
        a = self.k["C"]
        for i in range(60):
            e = sel.update(a.cx, a.cy, True, i / 30, stable=False)
            self.assertIsNone(e)
        self.assertEqual(sel.state.progress, 0.0)

    def test_bottom_row_controls_exist_and_are_tall(self):
        for kid in ("SPACE", "DEL", "ENTER", "CLEAR"):
            self.assertIn(kid, self.k)
        self.assertEqual(self.k["DEL"].label, "BACKSPACE")
        self.assertGreater(self.k["ENTER"].h, self.k["A"].h)
        self.assertAlmostEqual(self.k["ENTER"].y + self.k["ENTER"].h, H, delta=1)


if __name__ == "__main__":
    unittest.main()

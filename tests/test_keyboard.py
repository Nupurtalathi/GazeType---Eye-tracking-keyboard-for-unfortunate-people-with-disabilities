"""Tests for the full-screen gaze keyboard layout and key selector."""
import unittest

from keyboard.layout import build_layout, DEFAULT_PHRASES
from keyboard.selector import KeySelector
from gaze.drift_correction import DriftCorrector

W, H = 1920, 1080


class TestLayout(unittest.TestCase):
    def setUp(self):
        self.keys = build_layout(W, H)

    def test_all_letters_and_phrases_present(self):
        ids = {k.id for k in self.keys}
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            self.assertIn(c, ids)
        for label, _ in DEFAULT_PHRASES:
            self.assertIn(f"PHRASE_{label}", ids)
        for c in ("SPACE", "DEL", "CLEAR", "SPEAK", "PAUSE"):
            self.assertIn(c, ids)

    def test_phrase_row_is_above_letters(self):
        phrase_y = max(k.y for k in self.keys if k.kind == "phrase")
        letter_y = min(k.y for k in self.keys if k.kind == "letter")
        self.assertLess(phrase_y, letter_y)

    def test_screen_fully_tiled_including_edges(self):
        # every pixel sample (incl. borders) belongs to exactly one key
        for x in list(range(0, W, 37)) + [0, W - 1]:
            for y in list(range(0, H, 23)) + [0, H - 1]:
                hits = [k for k in self.keys if k.contains(x, y)]
                self.assertEqual(len(hits), 1, (x, y, [h.id for h in hits]))

    def test_keys_large_enough_for_webcam_gaze(self):
        for k in self.keys:
            if k.kind in ("letter", "phrase"):
                self.assertGreaterEqual(k.w, 200)
                self.assertGreaterEqual(k.h, 140)


class TestSelector(unittest.TestCase):
    def setUp(self):
        self.keys = build_layout(W, H)
        self.sel = KeySelector(self.keys, W, H)
        self.k = {k.id: k for k in self.keys}

    def _run(self, x, y, seconds, t0=0.0, valid=True, fps=30):
        ev, t = None, t0
        for i in range(int(seconds * fps)):
            t = t0 + i / fps
            e = self.sel.update(x, y, valid, t, raw_xy=(x, y))
            ev = ev or e
        return ev, t

    def test_dwell_selects_letter(self):
        a = self.k["A"]
        ev, _ = self._run(a.cx, a.cy, 1.3)
        self.assertIsNotNone(ev)
        self.assertEqual(ev.key.id, "A")
        self.assertIsNotNone(ev.raw_gaze_median)

    def test_edge_clamped_gaze_hits_corner_key(self):
        ev, _ = self._run(-80, H + 60, 1.4)      # off-screen bottom-left -> "Y"
        self.assertEqual(ev.key.id, "Y")
        ev, _ = self._run(W + 50, -40, 1.5, t0=5)  # off-screen top-right -> PAUSE
        self.assertEqual(ev.key.id, "PAUSE")

    def test_boundary_jitter_does_not_reset(self):
        a, b = self.k["A"], self.k["B"]
        edge_x = a.x + a.w
        ev, t = None, 0.0
        for i in range(45):          # 1.5 s, flicker one frame across boundary every 5th frame
            t = i / 30
            x = edge_x + 20 if i % 5 == 4 else edge_x - 30
            ev = ev or self.sel.update(x, a.cy, True, t)
        self.assertIsNotNone(ev)
        self.assertEqual(ev.key.id, "A")

    def test_short_blink_pauses_long_gap_resets(self):
        a = self.k["C"]
        self._run(a.cx, a.cy, 0.6)
        self._run(None, None, 0.2, t0=0.6, valid=False)       # blink < grace
        ev, _ = self._run(a.cx, a.cy, 0.6, t0=0.8)
        self.assertIsNotNone(ev)          # (0.6 - 0.12 settle) + 0.6 >= 1.0 s: blink paused, did not reset

    def test_pause_blocks_typing(self):
        p = self.k["PAUSE"]
        ev, _ = self._run(p.cx, p.cy, 1.4)
        self.assertEqual(ev.key.id, "PAUSE")
        self.assertTrue(self.sel.paused_by_user)
        a = self.k["A"]
        ev, _ = self._run(a.cx, a.cy, 2.0, t0=3)
        self.assertIsNone(ev)


class TestDrift(unittest.TestCase):
    def test_learns_constant_offset_and_undo(self):
        d = DriftCorrector()
        for x, y in [(300, 300), (900, 500), (1500, 800), (600, 900)]:
            d.add((x - 80, y + 50), (x, y))                 # tracker reads 80 px left, 50 px low
        cx, cy = d.correct((1000, 600))
        self.assertAlmostEqual(cx, 1080, delta=5)
        self.assertAlmostEqual(cy, 550, delta=5)
        for _ in range(4):
            d.undo_last()
        self.assertEqual(d.correct((1000, 600)), (1000, 600))

    def test_rejects_wild_residual(self):
        d = DriftCorrector()
        self.assertFalse(d.add((0, 0), (900, 900)))


if __name__ == "__main__":
    unittest.main()

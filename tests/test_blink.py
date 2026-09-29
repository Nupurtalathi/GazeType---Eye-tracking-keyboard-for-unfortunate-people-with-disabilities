"""Regression test: looking down (lowered lids) must not be treated as a blink."""
import unittest

from gaze.blink import BlinkFilter


class TestBlinkFilter(unittest.TestCase):
    def _feed(self, f, ear, seconds, t0, fps=30):
        out = []
        for i in range(int(seconds * fps)):
            out.append(f.update(ear, t0 + i / fps))
        return out, t0 + seconds

    def test_sustained_downward_gaze_becomes_valid(self):
        f = BlinkFilter(hard_floor=0.10)
        _, t = self._feed(f, 0.36, 1.0, 0.0)            # looking at top rows
        dropped, t = self._feed(f, 0.21, 2.0, t)          # looking at bottom rows (lids 42% lower)
        # only the first ~0.35 s may be discarded, the rest must be usable
        self.assertLessEqual(sum(dropped), 12)
        self.assertFalse(any(dropped[-40:]))

    def test_real_blink_is_rejected(self):
        f = BlinkFilter(hard_floor=0.10)
        _, t = self._feed(f, 0.33, 1.0, 0.0)
        dropped, t = self._feed(f, 0.15, 0.15, t)         # 150 ms blink
        self.assertTrue(all(dropped))
        after, _ = self._feed(f, 0.33, 0.2, t)
        self.assertTrue(after[0])                          # post-blink hold-off
        self.assertFalse(after[-1])

    def test_closed_eyes_always_rejected(self):
        f = BlinkFilter(hard_floor=0.10)
        self._feed(f, 0.33, 1.0, 0.0)
        dropped, _ = self._feed(f, 0.06, 2.0, 1.0)
        self.assertTrue(all(dropped))

    def test_blink_while_looking_down(self):
        f = BlinkFilter(hard_floor=0.10)
        _, t = self._feed(f, 0.36, 1.0, 0.0)
        _, t = self._feed(f, 0.21, 2.0, t)                 # baseline adapts to lowered lids
        dropped, _ = self._feed(f, 0.11, 0.15, t)
        self.assertTrue(all(dropped))


if __name__ == "__main__":
    unittest.main()

"""
Blink filter shared by both gaze backends (v4).

v3 bug (why the bottom keyboard rows could not be selected): looking DOWN closes the
eyelids by 30-45%. v3 flagged any frame with EAR < 62% of the recent open-eye EAR as a
blink and never updated the baseline while "blinking" - so a user looking at the bottom
two rows was classified as blinking for as long as they looked there, and no key could
be selected.

v4 rule:
  * EAR below the hard floor           -> eyes closed (always invalid)
  * EAR below REL x baseline, briefly  -> blink (invalid) for at most MAX_BLINK_S
  * still low after MAX_BLINK_S        -> NOT a blink: sustained downward gaze / squint.
                                          Valid again, and the baseline adapts to it.
  * after a real blink ends            -> POST_BLINK_HOLDOFF frames discarded (lids moving)
"""

import time
from collections import deque
from typing import Optional


class BlinkFilter:
    REL = 0.62
    MAX_BLINK_S = 0.35        # natural blinks last 0.1-0.3 s
    BASELINE_FRAMES = 30
    POST_BLINK_HOLDOFF = 3

    def __init__(self, hard_floor: float):
        self.hard_floor = hard_floor
        self._hist: deque = deque(maxlen=self.BASELINE_FRAMES)
        self._low_since: Optional[float] = None
        self._holdoff = 0
        self.last_state = "open"      # open | blink | closed | lowered | holdoff

    @property
    def baseline(self) -> Optional[float]:
        if len(self._hist) < 5:
            return None
        s = sorted(self._hist)
        return s[len(s) // 2]

    def update(self, ear: float, t: Optional[float] = None) -> bool:
        """Returns True if this frame must be discarded (blink / closed / hold-off)."""
        t = time.time() if t is None else t
        if ear < self.hard_floor:
            self._low_since = self._low_since or t
            self._holdoff = self.POST_BLINK_HOLDOFF
            self.last_state = "closed"
            return True

        base = self.baseline
        low = base is not None and ear < self.REL * base
        if low:
            if self._low_since is None:
                self._low_since = t
            if t - self._low_since <= self.MAX_BLINK_S:
                self._holdoff = self.POST_BLINK_HOLDOFF
                self.last_state = "blink"
                return True
            # sustained -> downward gaze, accept and let baseline follow
            self._holdoff = 0
            self._hist.append(ear)
            self.last_state = "lowered"
            return False

        self._low_since = None
        self._hist.append(ear)
        if self._holdoff > 0:
            self._holdoff -= 1
            self.last_state = "holdoff"
            return True
        self.last_state = "open"
        return False

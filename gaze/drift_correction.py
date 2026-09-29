"""
Implicit (online) drift correction.

Problem: a calibration is only exact for the head position at calibration time. Users
slump, turn or lean, and the error grows over minutes (on the candidate video the head
drifted ~16 px in 25 s).

Idea: every confirmed keyboard selection is a free calibration sample - the user was
looking at that key. We store (raw mapped gaze, key centre) pairs and correct new gaze
by a distance-weighted average of recent residuals. Undo (BACK/DEL) removes the last
sample so wrong selections do not poison the correction.

Safety limits: needs >= MIN_SAMPLES, uses only the last MAX_SAMPLES, residuals larger
than MAX_RESIDUAL_PX are ignored (probably a wrong key), and total correction is capped.
"""

from collections import deque
from typing import Optional, Tuple

import numpy as np


class DriftCorrector:
    MAX_SAMPLES = 20
    MIN_SAMPLES = 3
    MAX_RESIDUAL_PX = 260.0
    MAX_CORRECTION_PX = 200.0

    def __init__(self, kernel_px: float = 450.0, enabled: bool = True):
        self.kernel_px = kernel_px
        self.enabled = enabled
        self._obs: deque = deque(maxlen=self.MAX_SAMPLES)   # (raw_x, raw_y, res_x, res_y)

    def add(self, raw_xy: Tuple[float, float], true_xy: Tuple[float, float]) -> bool:
        rx, ry = true_xy[0] - raw_xy[0], true_xy[1] - raw_xy[1]
        if np.hypot(rx, ry) > self.MAX_RESIDUAL_PX:
            return False
        self._obs.append((raw_xy[0], raw_xy[1], rx, ry))
        return True

    def undo_last(self) -> None:
        if self._obs:
            self._obs.pop()

    def reset(self) -> None:
        self._obs.clear()

    @property
    def n_samples(self) -> int:
        return len(self._obs)

    def correct(self, xy: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        if xy is None or not self.enabled or len(self._obs) < self.MIN_SAMPLES:
            return xy
        o = np.array(self._obs)
        d2 = (o[:, 0] - xy[0]) ** 2 + (o[:, 1] - xy[1]) ** 2
        recency = np.linspace(0.5, 1.0, len(o))                 # newer samples count more
        w = np.exp(-d2 / (2 * self.kernel_px ** 2)) * recency
        if w.sum() < 1e-6:
            return xy
        cx, cy = float((w * o[:, 2]).sum() / w.sum()), float((w * o[:, 3]).sum() / w.sum())
        mag = np.hypot(cx, cy)
        if mag > self.MAX_CORRECTION_PX:
            cx, cy = cx * self.MAX_CORRECTION_PX / mag, cy * self.MAX_CORRECTION_PX / mag
        return (xy[0] + cx, xy[1] + cy)

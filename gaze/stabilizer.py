"""
Fixation-locked gaze stabiliser for typing (v7).

Idea (standard in eye-typing systems): the eye alternates between FIXATIONS (the user
looks at one key; everything that moves is noise) and SACCADES (a fast, deliberate jump to
another key). The two need opposite treatment:

  * during a fixation: output a heavily smoothed, *locked* position (running median + slow
    EMA over ~0.3 s). Micro-jitter, tremor and landmark noise do not move it at all, so the
    active key cannot flicker.
  * on a saccade: drop the history and jump straight to the new position - no lag, no
    rubber-banding - so a SMALL deliberate move to the neighbouring key is enough.

A saccade is only accepted when the gaze leaves the locked position by more than a fraction
of a key (default 50% of a key width / height) on `confirm_frames` consecutive frames AND the
mean of those frames is also that far away - single-frame spikes are rejected.

Stage 0 is a One Euro filter (low min_cutoff = strong smoothing, beta keeps real motion).
Vertical uses its own, lower cutoff (noisier axis).
"""

from collections import deque
from typing import Optional, Tuple

import numpy as np

from .smoothing import OneEuroFilter1D


class GazeStabilizer:
    def __init__(self, key_w: float = 240.0, key_h: float = 166.0,
                 saccade_frac: float = 0.50, confirm_frames: int = 2, detect_avg: int = 5,
                 window: int = 9, fix_alpha: float = 0.30,
                 min_cutoff_x: float = 0.25, min_cutoff_y: float = 0.18,
                 beta: float = 0.006, d_cutoff: float = 1.0,
                 noise_k: float = 3.0, noise_alpha: float = 0.03):
        self.key_w, self.key_h = key_w, key_h
        self.saccade_frac = saccade_frac
        self.confirm_frames = confirm_frames
        self.detect_avg = detect_avg
        self._raw: deque = deque(maxlen=detect_avg)
        self.window = window
        self.fix_alpha = fix_alpha
        self._params = (min_cutoff_x, min_cutoff_y, beta, d_cutoff)
        # Adaptive part: the saccade threshold never drops below noise_k x the measured
        # fixation noise of the detection signal -> a noisier user/camera/lighting automatically
        # gets stronger jitter rejection instead of a flood of false jumps.
        self.noise_k, self.noise_alpha = noise_k, noise_alpha
        self._var = None                   # running variance (x, y) of det - anchor during fixation
        self._fx = OneEuroFilter1D(min_cutoff_x, beta, d_cutoff)
        self._fy = OneEuroFilter1D(min_cutoff_y, beta, d_cutoff)
        self._buf: deque = deque(maxlen=window)
        self._pending = []
        self._out: Optional[Tuple[float, float]] = None
        self.state = "idle"               # idle | fixation | pending | saccade
        self.last_saccade_t: Optional[float] = None
        self.saccade_count = 0

    def reset(self):
        mcx, mcy, beta, dc = self._params
        self._fx = OneEuroFilter1D(mcx, beta, dc)
        self._fy = OneEuroFilter1D(mcy, beta, dc)
        self._buf.clear()
        self._raw.clear()
        self._pending = []
        self._out = None
        self.state = "idle"

    def set_key_size(self, w: float, h: float):
        self.key_w, self.key_h = w, h

    def thresholds(self) -> Tuple[float, float]:
        tx, ty = self.saccade_frac * self.key_w, self.saccade_frac * self.key_h
        if self._var is not None:
            tx = max(tx, self.noise_k * float(np.sqrt(self._var[0])))
            ty = max(ty, self.noise_k * float(np.sqrt(self._var[1])))
        return tx, ty

    def _ndist(self, a, b) -> float:
        tx, ty = self.thresholds()
        return float(np.hypot((a[0] - b[0]) / tx, (a[1] - b[1]) / ty))

    @property
    def is_stable(self) -> bool:
        """True while the gaze is in a confirmed fixation (safe to accumulate dwell)."""
        return self.state == "fixation"

    def update(self, x: Optional[float], y: Optional[float], t: float) -> Optional[Tuple[float, float]]:
        if x is None or y is None:
            return self._out
        p = (self._fx.filter(x, t), self._fy.filter(y, t))
        self._raw.append((x, y))
        det = tuple(np.mean(np.array(self._raw), axis=0))    # short raw average: fast but de-noised
        if self._out is None:
            self._buf.append(p)
            self._out = p
            self.state = "fixation"
            return self._out

        anchor = tuple(np.median(np.array(self._buf), axis=0)) if self._buf else self._out
        # saccade detection uses a short RAW average (the smoothed signal lags a jump by many frames)
        if self._ndist(det, anchor) > 1.0:
            self._pending.append((det[0], det[1], p))
            self.state = "pending"
            if len(self._pending) >= self.confirm_frames:
                raw = np.array([q[:2] for q in self._pending])
                mean = tuple(raw.mean(axis=0))
                spread = max(self._ndist(r, mean) for r in raw)
                if self._ndist(mean, anchor) > 1.0 and spread < 1.5:
                    # confirmed deliberate jump: restart everything at the new place
                    mcx, mcy, beta, dc = self._params
                    self._fx = OneEuroFilter1D(mcx, beta, dc)
                    self._fy = OneEuroFilter1D(mcy, beta, dc)
                    for rx, ry in raw:
                        self._fx.filter(rx, t); self._fy.filter(ry, t)
                    self._buf.clear()
                    self._buf.extend([tuple(r) for r in raw])
                    self._raw.clear()
                    self._raw.append(mean)
                    self._out = mean
                    self._pending = []
                    self.state = "saccade"
                    self.last_saccade_t = t
                    self.saccade_count += 1
                    return self._out
                self._pending = self._pending[1:]          # inconsistent: slide the window
            return self._out                                   # hold still while undecided

        # inside the fixation zone -> update the noise estimate
        d2 = np.array([(det[0] - anchor[0]) ** 2, (det[1] - anchor[1]) ** 2])
        self._var = d2 if self._var is None else (1 - self.noise_alpha) * self._var + self.noise_alpha * d2
        self._pending = []
        self._buf.append(p)
        med = np.median(np.array(self._buf), axis=0)
        a = self.fix_alpha
        self._out = (self._out[0] + a * (med[0] - self._out[0]), self._out[1] + a * (med[1] - self._out[1]))
        self.state = "fixation"
        return self._out

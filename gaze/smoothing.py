"""
Gaze Smoothing and Jitter Suppression Module.
Implements the 1€ (One Euro) Filter for low-latency adaptive jitter removal,
along with Exponential Moving Average (EMA) and saccade jump gating.
"""

import math
import time
from typing import Optional, Tuple


class LowPassFilter:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.s: Optional[float] = None

    def filter(self, value: float, alpha: Optional[float] = None) -> float:
        if alpha is not None:
            self.alpha = alpha
        if self.s is None:
            self.s = value
        else:
            self.s = self.alpha * value + (1.0 - self.alpha) * self.s
        return self.s

    def reset(self):
        self.s = None


class OneEuroFilter1D:
    """
    Casiez et al. (2012) 1€ Filter.
    Adapts cutoff frequency based on rate of change:
    high smoothing during stationary fixations, instantaneous response during saccades.
    """
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.01, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_filt = LowPassFilter()
        self.dx_filt = LowPassFilter()
        self.last_time: Optional[float] = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-4))

    def filter(self, x: float, timestamp: Optional[float] = None) -> float:
        if timestamp is None:
            timestamp = time.time()

        if self.last_time is None:
            self.last_time = timestamp
            return self.x_filt.filter(x)

        dt = max(1e-4, timestamp - self.last_time)
        self.last_time = timestamp

        prev_x = self.x_filt.s if self.x_filt.s is not None else x
        dx = (x - prev_x) / dt
        edx = self.dx_filt.filter(dx, self._alpha(self.d_cutoff, dt))

        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self.x_filt.filter(x, self._alpha(cutoff, dt))

    def reset(self):
        self.x_filt.reset()
        self.dx_filt.reset()
        self.last_time = None


class GazeSmoother:
    def __init__(
        self,
        method: str = "one_euro",
        ema_alpha: float = 0.25,
        one_euro_min_cutoff: float = 0.8,
        one_euro_beta: float = 0.02,
        max_allowed_jump: float = 350.0,
        one_euro_min_cutoff_y: Optional[float] = None
    ):
        self.method = method
        self.ema_alpha = ema_alpha
        self.max_allowed_jump = max_allowed_jump

        # Filters for X and Y coordinates
        self.euro_x = OneEuroFilter1D(min_cutoff=one_euro_min_cutoff, beta=one_euro_beta)
        # vertical gets its own (usually lower = smoother) cutoff: it is the noisier axis
        self.euro_y = OneEuroFilter1D(min_cutoff=one_euro_min_cutoff_y if one_euro_min_cutoff_y else one_euro_min_cutoff,
                                      beta=one_euro_beta)

        self.ema_x: Optional[float] = None
        self.ema_y: Optional[float] = None
        self.last_output: Optional[Tuple[float, float]] = None

    def filter(self, x: float, y: float, timestamp: Optional[float] = None) -> Tuple[float, float]:
        """
        Filters raw screen coordinates (x, y). Detects sudden saccadic leaps
        to immediately snap without rubber-banding delay.
        """
        if timestamp is None:
            timestamp = time.time()

        # Check for saccade (large sudden jump)
        if self.last_output is not None:
            dist = math.hypot(x - self.last_output[0], y - self.last_output[1])
            if dist > self.max_allowed_jump:
                self.reset()

        if self.method == "one_euro":
            sx = self.euro_x.filter(x, timestamp)
            sy = self.euro_y.filter(y, timestamp)
        else:
            # EMA
            if self.ema_x is None or self.ema_y is None:
                self.ema_x = x
                self.ema_y = y
            else:
                self.ema_x = self.ema_alpha * x + (1.0 - self.ema_alpha) * self.ema_x
                self.ema_y = self.ema_alpha * y + (1.0 - self.ema_alpha) * self.ema_y
            sx, sy = self.ema_x, self.ema_y

        self.last_output = (sx, sy)
        return sx, sy

    def reset(self):
        """Clears filter internal memory."""
        self.euro_x.reset()
        self.euro_y.reset()
        self.ema_x = None
        self.ema_y = None
        self.last_output = None

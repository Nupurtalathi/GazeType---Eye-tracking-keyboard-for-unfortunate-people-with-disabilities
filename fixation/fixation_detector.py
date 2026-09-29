"""
Fixation Detection Module.
Detects when the gaze remains within a spatial dispersion threshold (radius)
for a continuous minimum time window.
"""

import math
import time
from collections import deque
from typing import Optional, Tuple, List


class FixationDetector:
    def __init__(self, fixation_radius: float = 50.0, min_fixation_time: float = 0.10, max_history: int = 60):
        self.fixation_radius = fixation_radius
        self.min_fixation_time = min_fixation_time
        self.max_history = max_history

        # State
        self.centroid: Optional[Tuple[float, float]] = None
        self.start_time: Optional[float] = None
        self.last_update_time: Optional[float] = None
        self.duration: float = 0.0
        self.is_fixating: bool = False
        self.samples: deque = deque(maxlen=max_history)
        self.fixation_count: int = 0

    def update(self, x: float, y: float, timestamp: Optional[float] = None, is_valid: bool = True) -> bool:
        """
        Updates the fixation state with a new gaze point.
        Returns True if the gaze is currently classified as fixating.
        """
        if timestamp is None:
            timestamp = time.time()

        if not is_valid or x is None or y is None:
            # Invalid gaze temporarily disrupts accumulation
            return self.is_fixating

        if self.centroid is None:
            # First point initializes potential fixation
            self.centroid = (float(x), float(y))
            self.start_time = timestamp
            self.last_update_time = timestamp
            self.duration = 0.0
            self.is_fixating = False
            self.samples.clear()
            self.samples.append((float(x), float(y)))
            return False

        # Calculate Euclidean distance from current centroid
        dist = math.hypot(x - self.centroid[0], y - self.centroid[1])

        if dist <= self.fixation_radius:
            # Point is inside cluster
            self.samples.append((float(x), float(y)))
            self.last_update_time = timestamp

            # Update rolling centroid
            sx = sum(p[0] for p in self.samples)
            sy = sum(p[1] for p in self.samples)
            n = len(self.samples)
            self.centroid = (sx / n, sy / n)

            self.duration = timestamp - self.start_time
            was_fixating = self.is_fixating
            self.is_fixating = self.duration >= self.min_fixation_time

            if self.is_fixating and not was_fixating:
                self.fixation_count += 1

            return self.is_fixating

        else:
            # Gaze moved outside fixation radius (saccade / shift)
            self.reset(new_point=(x, y), timestamp=timestamp)
            return False

    def reset(self, new_point: Optional[Tuple[float, float]] = None, timestamp: Optional[float] = None):
        """Resets the current fixation."""
        if timestamp is None:
            timestamp = time.time()

        if new_point is not None:
            self.centroid = (float(new_point[0]), float(new_point[1]))
            self.start_time = timestamp
            self.last_update_time = timestamp
            self.duration = 0.0
            self.is_fixating = False
            self.samples.clear()
            self.samples.append(self.centroid)
        else:
            self.centroid = None
            self.start_time = None
            self.last_update_time = None
            self.duration = 0.0
            self.is_fixating = False
            self.samples.clear()

    def get_centroid(self) -> Optional[Tuple[float, float]]:
        return self.centroid

    def get_duration(self) -> float:
        return self.duration

    def is_active(self) -> bool:
        return self.is_fixating

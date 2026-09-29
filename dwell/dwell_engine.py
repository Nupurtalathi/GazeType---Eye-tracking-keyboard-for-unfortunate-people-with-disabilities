"""
Dwell-Time Detection Engine.
Calculates duration of continuous gaze fixations, incorporates blink-pause grace periods,
updates circular progress indicators, and fires callbacks upon threshold completion.
"""

from enum import Enum
import time
from typing import Optional, Tuple, Callable
from dataclasses import dataclass


class DwellState(str, Enum):
    TRACKING = "Tracking"
    FIXATING = "Fixating"
    DWELLING = "Dwell in progress"
    TRIGGERED = "Dwell Triggered"
    PAUSED_BLINK = "Blink (Paused)"
    INVALID = "Invalid Gaze"
    FACE_LOST = "Face Lost"


@dataclass
class DwellResult:
    state: DwellState
    dwell_time: float
    progress: float  # 0.0 to 1.0
    threshold: float
    triggered: bool
    anchor_point: Optional[Tuple[float, float]]


class DwellEngine:
    def __init__(
        self,
        dwell_threshold: float = 1.50,
        blink_reset_time: float = 0.50,
        cooldown_time: float = 0.60,
        on_dwell_callback: Optional[Callable[[Tuple[float, float], float], None]] = None
    ):
        self.dwell_threshold = dwell_threshold
        self.blink_reset_time = blink_reset_time
        self.cooldown_time = cooldown_time
        self.on_dwell_callback = on_dwell_callback

        # State tracking
        self.dwell_time: float = 0.0
        self.state: DwellState = DwellState.TRACKING
        self.anchor_point: Optional[Tuple[float, float]] = None
        self.last_timestamp: Optional[float] = None

        self.blink_start_time: Optional[float] = None
        self.is_blinking_paused: bool = False

        self.has_triggered: bool = False
        self.last_trigger_time: float = 0.0

    def update(
        self,
        gaze_x: Optional[float],
        gaze_y: Optional[float],
        is_fixating: bool,
        is_blinking: bool,
        is_face_detected: bool,
        is_valid_confidence: bool,
        timestamp: Optional[float] = None
    ) -> DwellResult:
        """
        Advances the dwell-time engine by dt.
        Handles blinking pauses, loss of face, fixation breaks, and event dispatch.
        """
        if timestamp is None:
            timestamp = time.time()

        dt = (timestamp - self.last_timestamp) if self.last_timestamp is not None else 0.0
        self.last_timestamp = timestamp

        # Guard 1: Face loss -> immediate reset
        if not is_face_detected:
            self.reset()
            self.state = DwellState.FACE_LOST
            return self._build_result(False)

        # Guard 2: Blinking grace period
        if is_blinking:
            if self.blink_start_time is None:
                self.blink_start_time = timestamp
                self.is_blinking_paused = True

            blink_duration = timestamp - self.blink_start_time
            if blink_duration > self.blink_reset_time:
                # Eye closed too long -> reset dwell
                self.reset()
                self.state = DwellState.INVALID
                return self._build_result(False)
            else:
                # Normal short blink -> pause timer without accumulating dt
                self.state = DwellState.PAUSED_BLINK
                return self._build_result(False)
        else:
            # Not blinking -> clear blink tracker
            self.blink_start_time = None
            self.is_blinking_paused = False

        # Guard 3: Low confidence / invalid gaze
        if not is_valid_confidence or gaze_x is None or gaze_y is None:
            self.reset()
            self.state = DwellState.INVALID
            return self._build_result(False)

        # Check cooldown after trigger
        if self.has_triggered:
            if (timestamp - self.last_trigger_time) < self.cooldown_time:
                return self._build_result(False)
            else:
                self.has_triggered = False

        # Fixation check
        if not is_fixating:
            # Gaze moved or hasn't established minimum fixation
            self.reset()
            self.state = DwellState.TRACKING
            return self._build_result(False)

        # Active fixation: initialize or update anchor point
        if self.anchor_point is None:
            self.anchor_point = (gaze_x, gaze_y)
            self.dwell_time = 0.0

        # Accumulate dwell time
        self.dwell_time += dt

        # Check if threshold reached
        just_triggered = False
        if self.dwell_time >= self.dwell_threshold and not self.has_triggered:
            self.state = DwellState.TRIGGERED
            self.has_triggered = True
            self.last_trigger_time = timestamp
            just_triggered = True

            # Fire callback
            if self.on_dwell_callback and self.anchor_point:
                try:
                    self.on_dwell_callback(self.anchor_point, self.dwell_time)
                except Exception as e:
                    print(f"[DwellEngine] Callback error: {e}")

        elif self.dwell_time > 0.3:
            self.state = DwellState.DWELLING
        else:
            self.state = DwellState.FIXATING

        return self._build_result(just_triggered)

    def _build_result(self, triggered: bool) -> DwellResult:
        progress = min(1.0, max(0.0, self.dwell_time / max(1e-4, self.dwell_threshold)))
        return DwellResult(
            state=self.state,
            dwell_time=self.dwell_time,
            progress=progress,
            threshold=self.dwell_threshold,
            triggered=triggered,
            anchor_point=self.anchor_point
        )

    def reset(self):
        """Resets dwell timer and anchor position."""
        self.dwell_time = 0.0
        self.anchor_point = None
        self.blink_start_time = None
        self.is_blinking_paused = False
        self.has_triggered = False
        self.state = DwellState.TRACKING

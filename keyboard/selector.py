"""
Key selection state machine for the gaze keyboard.

Improves *effective* accuracy on top of the raw gaze estimate:
  * Snap-to-key: gaze is clamped to the screen and assigned to the key that contains
    it, so there is no dead space and the screen edges always hit a key.
  * Hysteresis: a new key must hold the gaze for `switch_time` before it becomes the
    active key -> jitter across a key boundary no longer resets progress.
  * Blink / dropout tolerance: gaps shorter than `grace_time` pause the dwell, longer
    gaps reset it.
  * Re-trigger lock: after a selection the same key needs a fresh full dwell.
The raw (un-corrected) gaze samples collected during a dwell are returned with the
event so the drift corrector can learn from each confirmed selection.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .layout import Key


@dataclass
class SelectionEvent:
    key: Key
    timestamp: float
    raw_gaze_median: Optional[Tuple[float, float]] = None


@dataclass
class SelectorState:
    active_key: Optional[Key] = None
    progress: float = 0.0          # 0..1
    paused: bool = False
    candidate_key: Optional[Key] = None
    repeat: bool = False           # dwelling again on the key that was just typed (longer dwell)
    settling: bool = False         # key just became active, dwell not counting yet


class KeySelector:
    """
    v7 typing-stability rules (on top of snap-to-key / edge clamping):
      * Saccade-aware hysteresis: while the eye is fixating, the active key is "sticky"
        (gaze must leave it by sticky_x / sticky_y of its size) and a new key must win for
        switch_time. Right after a deliberate saccade (reported by GazeStabilizer) stickiness
        is switched off and switching is fast, so a SMALL intentional move to a neighbour is enough.
      * Stable-gaze dwell: dwell only accumulates while the stabiliser reports a fixation, and
        only after `settle_time` on the new key. Unstable frames freeze (not reset) the dwell.
      * Short glance-away memory: returning to the previous key within `memory_time` restores
        most of its progress (no punishment for a brief look at the text bar).
      * No accidental repeats: after a key fires it is locked. To type it again the user either
        looks away and back (proper key exit) or keeps dwelling for repeat_factor x the dwell
        (needed for double letters like the EE in NEED). repeat_factor=0 -> exit is mandatory.
    """

    def __init__(self, keys: List[Key], screen_w: int, screen_h: int,
                 switch_time: float = 0.15, grace_time: float = 0.35, cooldown: float = 0.4,
                 sticky_x: float = 0.15, sticky_y: float = 0.25,
                 settle_time: float = 0.12, saccade_window: float = 0.30, fast_switch_time: float = 0.05,
                 repeat_factor: float = 1.8, memory_time: float = 0.6, memory_keep: float = 0.8):
        self.keys = [k for k in keys if k.kind != "display"]
        self.display_keys = [k for k in keys if k.kind == "display"]
        self.sticky_x, self.sticky_y = sticky_x, sticky_y
        self.screen_w, self.screen_h = screen_w, screen_h
        self.switch_time = switch_time
        self.fast_switch_time = fast_switch_time
        self.saccade_window = saccade_window
        self.settle_time = settle_time
        self.grace_time = grace_time
        self.cooldown = cooldown
        self.repeat_factor = repeat_factor
        self.memory_time, self.memory_keep = memory_time, memory_keep
        self.paused_by_user = False
        self.state = SelectorState()
        self._dwell = 0.0
        self._last_t: Optional[float] = None
        self._last_valid_t: Optional[float] = None
        self._candidate_since: Optional[float] = None
        self._active_since: Optional[float] = None
        self._cooldown_until = 0.0
        self._raw: List[Tuple[float, float]] = []
        self._locked: Optional[Key] = None       # key that just fired and has not been exited
        self._prev = None                        # (key, dwell, time_left) for glance-away memory

    def key_at(self, x: float, y: float) -> Optional[Key]:
        x = min(max(x, 0.0), self.screen_w - 1e-3)
        y = min(max(y, 0.0), self.screen_h - 1e-3)
        for k in self.keys:
            if k.contains(x, y):
                return k
        return None                      # on the text display

    def _reset_dwell(self):
        self._dwell = 0.0
        self._raw = []
        self.state.progress = 0.0

    def _activate(self, k: Optional[Key], t: float):
        old = self.state.active_key
        if old is not None and self._dwell > 0:
            self._prev = (old, self._dwell, t)
        self.state.active_key = k
        self.state.candidate_key = None
        self._active_since = t
        self._reset_dwell()
        if k is not self._locked:
            self._locked = None                  # proper exit happened -> key re-armed
        if self._prev and k is self._prev[0] and t - self._prev[2] <= self.memory_time:
            self._dwell = self._prev[1] * self.memory_keep
        self._prev = None if (self._prev and k is self._prev[0]) else self._prev

    def update(self, x: Optional[float], y: Optional[float], valid: bool,
               t: Optional[float] = None, raw_xy: Optional[Tuple[float, float]] = None,
               stable: Optional[bool] = None, last_saccade_t: Optional[float] = None
               ) -> Optional[SelectionEvent]:
        t = time.time() if t is None else t
        dt = 0.0 if self._last_t is None else max(0.0, min(0.2, t - self._last_t))
        self._last_t = t

        if not valid or x is None or y is None:
            gap = t - (self._last_valid_t if self._last_valid_t is not None else t)
            self.state.paused = True
            if gap > self.grace_time:
                self.state.active_key = None
                self._locked = None
                self._reset_dwell()
            return None
        self._last_valid_t = t
        self.state.paused = False

        recent_saccade = last_saccade_t is not None and (t - last_saccade_t) <= self.saccade_window
        k = self.key_at(x, y)
        a = self.state.active_key
        if a is not None and k is not a and not recent_saccade:
            mx, my = a.w * self.sticky_x, a.h * self.sticky_y
            xc = min(max(x, 0.0), self.screen_w - 1e-3)
            yc = min(max(y, 0.0), self.screen_h - 1e-3)
            if a.x - mx <= xc < a.x + a.w + mx and a.y - my <= yc < a.y + a.h + my:
                k = a                    # still inside the sticky zone of the active key
        # temporal hysteresis on key changes (fast right after a deliberate saccade)
        if k is not self.state.active_key:
            if k is not self.state.candidate_key:
                self.state.candidate_key = k
                self._candidate_since = t
            need = self.fast_switch_time if recent_saccade else self.switch_time
            if t - self._candidate_since >= need or self.state.active_key is None:
                self._activate(k, t)
            else:
                return None              # still on previous key; do not advance its dwell
        else:
            self.state.candidate_key = None

        key = self.state.active_key
        if key is None or t < self._cooldown_until:
            return None
        if not getattr(key, "enabled", True):
            self._reset_dwell()
            return None
        if self.paused_by_user and key.id != "PAUSE":
            self.state.progress = 0.0
            return None
        since = self._active_since if self._active_since is not None else t
        self.state.settling = (t - since) < self.settle_time
        if self.state.settling or stable is False:
            return None                  # freeze, don't reset: wait for a stable fixation

        self._dwell += dt
        if raw_xy is not None:
            self._raw.append(raw_xy)
        self.state.repeat = key is self._locked
        target = key.dwell_time * (self.repeat_factor if self.state.repeat else 1.0)
        if self.state.repeat and self.repeat_factor <= 0:
            self.state.progress = 0.0
            return None                  # repeats disabled: must look away first
        self.state.progress = min(1.0, self._dwell / target)
        if self._dwell >= target:
            med = tuple(np.median(np.array(self._raw), axis=0)) if len(self._raw) >= 5 else None
            ev = SelectionEvent(key=key, timestamp=t, raw_gaze_median=med)
            self._reset_dwell()
            self._cooldown_until = t + self.cooldown
            self._locked = key
            if key.id == "PAUSE":
                self.paused_by_user = not self.paused_by_user
            return ev
        return None

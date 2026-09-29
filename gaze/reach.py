"""
Reach tuning: make every key reachable with COMFORTABLE eye angles (v6).

Problem on small screens (14" laptop at ~50 cm): the whole keyboard spans only about
+/-16 deg horizontally and +/-10 deg vertically. Webcam iris estimates are compressed
towards the centre (the iris model "under-reports" large rotations, lids cover the iris
looking down, people turn the head instead of the eyes). After calibration the cursor
therefore often stops short of the outer keys, and users compensate by rolling their
eyes to extreme angles - tiring and inaccurate.

Fix: after calibration the user looks at 5 key centres (centre, far-left column,
far-right column, top row, bottom row). We measure where the calibrated gaze actually
lands and fit, independently for LEFT / RIGHT / UP / DOWN:

    x' = cx + d * (1 + (g - 1) * |d| / D)          d = x - cx,  D = half screen width

i.e. no change at the centre, growing towards the edge (the compression is an edge
effect, so inner keys must not be pushed around). g is solved so the measured edge
point lands exactly on its key centre. Plus a small centre offset. 4 corner targets
are measured too and reported as a check, never used for fitting.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np

GAIN_LIMITS = (0.8, 2.2)


@dataclass
class ReachModel:
    screen_w: float = 1920.0
    screen_h: float = 1080.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    g_left: float = 1.0
    g_right: float = 1.0
    g_up: float = 1.0
    g_down: float = 1.0
    corner_error_px: Optional[float] = None
    report: Dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------ apply
    @staticmethod
    def _ramp(d: float, g: float, D: float) -> float:
        return d * (1.0 + (g - 1.0) * min(1.0, abs(d) / D))

    def apply(self, x: float, y: float) -> Tuple[float, float]:
        cx, cy = self.screen_w / 2.0, self.screen_h / 2.0
        dx = x + self.offset_x - cx
        dy = y + self.offset_y - cy
        dx = self._ramp(dx, self.g_right if dx >= 0 else self.g_left, cx)
        dy = self._ramp(dy, self.g_down if dy >= 0 else self.g_up, cy)
        return cx + dx, cy + dy

    @property
    def is_identity(self) -> bool:
        return (self.offset_x == 0 and self.offset_y == 0 and
                self.g_left == self.g_right == self.g_up == self.g_down == 1.0)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "ReachModel":
        if not d:
            return cls()
        m = cls()
        for k, v in d.items():
            if hasattr(m, k):
                setattr(m, k, v)
        return m


def _solve_gain(measured_d: float, target_d: float, D: float) -> Optional[float]:
    """Gain so that _ramp(measured_d) == target_d. None if the measurement is unusable."""
    if abs(measured_d) < 0.05 * D or np.sign(measured_d) != np.sign(target_d):
        return None                     # gaze barely moved / went the wrong way
    frac = min(1.0, abs(measured_d) / D)
    g = 1.0 + (target_d / measured_d - 1.0) / frac
    return float(np.clip(g, *GAIN_LIMITS))


def reach_targets(keys, screen_w: float, screen_h: float) -> List[Tuple[str, float, float]]:
    """Fitting targets = real key centres of the keyboard (comfortable angles, not the bezel)."""
    sel = [k for k in keys if k.kind not in ("display",)]
    left = min(k.cx for k in sel)
    right = max(k.cx for k in sel)
    top = min(k.cy for k in sel if k.kind in ("suggestion", "phrase", "letter"))
    bottom = max(k.cy for k in sel)
    cx, cy = screen_w / 2.0, screen_h / 2.0
    return [
        ("centre", cx, cy),
        ("left", left, cy), ("right", right, cy),
        ("top", cx, top), ("bottom", cx, bottom),
        ("top-left", left, top), ("top-right", right, top),
        ("bottom-left", left, bottom), ("bottom-right", right, bottom),
    ]


def fit_reach(measurements: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]],
              screen_w: float, screen_h: float) -> ReachModel:
    """measurements: name -> ((measured_x, measured_y), (target_x, target_y)), measured WITHOUT reach."""
    m = ReachModel(screen_w=screen_w, screen_h=screen_h)
    cx, cy = screen_w / 2.0, screen_h / 2.0
    if "centre" in measurements:
        (mx, my), (tx, ty) = measurements["centre"]
        m.offset_x, m.offset_y = float(np.clip(tx - mx, -150, 150)), float(np.clip(ty - my, -150, 150))

    def edge(name, axis, attr):
        if name not in measurements:
            return
        meas, tgt = measurements[name]
        c = cx if axis == 0 else cy
        off = m.offset_x if axis == 0 else m.offset_y
        g = _solve_gain(meas[axis] + off - c, tgt[axis] - c, c)
        if g is not None:
            setattr(m, attr, g)
        m.report[name] = 100.0 * (meas[axis] + off - c) / (tgt[axis] - c)   # % of the way reached

    edge("left", 0, "g_left"); edge("right", 0, "g_right")
    edge("top", 1, "g_up"); edge("bottom", 1, "g_down")

    errs = []
    for name in ("top-left", "top-right", "bottom-left", "bottom-right"):
        if name in measurements:
            (mx, my), (tx, ty) = measurements[name]
            px, py = m.apply(mx, my)
            errs.append(np.hypot(px - tx, py - ty))
    m.corner_error_px = float(np.median(errs)) if errs else None
    return m

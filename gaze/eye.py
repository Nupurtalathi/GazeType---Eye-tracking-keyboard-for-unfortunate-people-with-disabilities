"""
Eye Analysis and Landmark Geometry Module (v2).

Key change vs v1 — VERTICAL RATIO:
  v1: vr = (pupil_y - top_lid_y) / lid_height
      The eyelids follow the eyeball when looking up/down, so this ratio barely
      moves. On the candidate video it correlated +0.39 with eye openness (wrong
      sign) and had SNR ~2.4 -> vertical gaze was mostly noise.
  v2: vr = signed distance of iris centre from the line joining the two eye
      corners, divided by corner-to-corner eye width. Corners do not move with
      the lids. On the same video: correlation -0.91 with openness (physically
      correct), SNR ~11.

Horizontal ratio is now measured along the corner axis (head-roll invariant).
Blink decision is taken by GazeDetector (adaptive per user); Eye only reports EAR.
"""

import math
from typing import List, Tuple, Optional

import numpy as np

from .pupil import Pupil

# vr = VR_OFFSET + VR_SCALE * (offset / eye_width). Keeps vr roughly in [0.1, 0.9]
# so existing 0..1 checks elsewhere keep working. Calibration absorbs the scale.
VR_OFFSET = 0.5
VR_SCALE = 2.5
HARD_CLOSED_EAR = 0.12   # below this we don't even try to find a pupil


class Eye:
    # Landmark indices for 68-point model
    LEFT_EYE_POINTS = [42, 43, 44, 45, 46, 47]
    RIGHT_EYE_POINTS = [36, 37, 38, 39, 40, 41]

    def __init__(self, original_frame: np.ndarray, landmarks, points: List[int],
                 blink_threshold: float = HARD_CLOSED_EAR, gray_frame: Optional[np.ndarray] = None):
        self.frame = original_frame
        self.blink_threshold = blink_threshold
        self.pupil: Optional[Pupil] = None
        self.landmarks_points: np.ndarray = np.array([])
        self.ear: float = 0.0
        self.is_blinking: bool = False          # may be overridden by GazeDetector
        self.horizontal_ratio: Optional[float] = None
        self.vertical_ratio: Optional[float] = None
        self.pupil_frame_coords: Optional[Tuple[float, float]] = None

        if gray_frame is None:
            import cv2
            gray_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY) if original_frame.ndim == 3 else original_frame
        self._analyze(gray_frame, landmarks, points)

    def _analyze(self, gray: np.ndarray, landmarks, points: List[int]) -> None:
        pts = np.array([(landmarks.part(p).x, landmarks.part(p).y) for p in points], np.float64)
        self.landmarks_points = pts.astype(np.int32)

        a, b = pts[0], pts[3]                      # eye corners (image-left, image-right)
        eye_w = float(np.linalg.norm(b - a))
        if eye_w < 4:
            return
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        self.ear = float((v1 + v2) / (2.0 * eye_w))
        self.is_blinking = self.ear < self.blink_threshold
        if self.is_blinking:
            return

        self.pupil = Pupil(gray, pts)
        c = self.pupil.center
        if c is None:
            return
        self.pupil_frame_coords = c

        ex = (b - a) / eye_w                       # unit vector along corner axis
        ey = np.array([-ex[1], ex[0]])             # perpendicular, +ve = image-down
        d = np.array(c) - a
        hr = float(d @ ex) / eye_w
        vr_raw = float(d @ ey) / eye_w
        self.horizontal_ratio = float(np.clip(hr, 0.0, 1.0))
        self.vertical_ratio = float(np.clip(VR_OFFSET + VR_SCALE * vr_raw, 0.0, 1.0))

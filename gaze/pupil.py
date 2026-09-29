"""
Pupil / Iris Centre Localisation (v2).

Changes vs v1 (tuned on the recorded candidate video, see tools/replay_video.py):
  * Works on a 4x up-sampled eye ROI -> sub-pixel centre instead of integer pixels.
  * Threshold = DARK_PERCENTILE of pixels *inside the lid polygon* (v1 used the
    whole crop incl. white fill, capped at 80 -> unstable under lighting changes).
  * Darkness-weighted centroid of the largest dark blob (robust to partial lid
    occlusion and specular highlights).
  * No "darkest pixel" fallback: if nothing plausible is found we return None and
    the frame is dropped rather than injecting a random point.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

DARK_PERCENTILE = 42     # swept 8..50 on the candidate video; 30-50 is a flat optimum
UPSAMPLE = 4
MIN_MASK_PIXELS = 30


class Pupil:
    def __init__(self, gray_frame: np.ndarray, eye_points: np.ndarray,
                 dark_percentile: float = DARK_PERCENTILE):
        # Absolute frame coordinates (float, sub-pixel)
        self.x: Optional[float] = None
        self.y: Optional[float] = None
        self.confidence: float = 0.0
        self.iris_frame: Optional[np.ndarray] = None
        self.dark_percentile = dark_percentile
        if gray_frame is not None and gray_frame.size > 0 and len(eye_points) == 6:
            self._detect(gray_frame, eye_points.astype(np.float64))

    def _detect(self, gray: np.ndarray, pts: np.ndarray) -> None:
        h, w = gray.shape[:2]
        x0, y0 = np.floor(pts.min(0) - 3).astype(int)
        x1, y1 = np.ceil(pts.max(0) + 3).astype(int)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 - x0 < 6 or y1 - y0 < 3:
            return

        up = UPSAMPLE
        roi = cv2.resize(gray[y0:y1, x0:x1], None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
        roi = cv2.GaussianBlur(roi, (7, 7), 0)

        mask = np.zeros_like(roi)
        cv2.fillPoly(mask, [((pts - [x0, y0]) * up).astype(np.int32)], 255)
        inside = mask > 0
        vals = roi[inside]
        if vals.size < MIN_MASK_PIXELS:
            return

        th = np.percentile(vals, self.dark_percentile)
        bw = ((roi <= th) & inside).astype(np.uint8)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        self.iris_frame = bw * 255

        n, labels, stats, _ = cv2.connectedComponentsWithStats(bw)
        if n < 2:
            return
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        ys, xs = np.nonzero(labels == k)
        weights = (th - roi[ys, xs].astype(np.float64)) + 1.0
        cx = (xs * weights).sum() / weights.sum()
        cy = (ys * weights).sum() / weights.sum()

        self.x = x0 + cx / up
        self.y = y0 + cy / up

        # Confidence: share of the visible eye taken by the blob (iris ~25-45%) + contrast
        area_ratio = stats[k, cv2.CC_STAT_AREA] / float(inside.sum())
        shape_score = float(np.clip(1.0 - abs(area_ratio - 0.35) / 0.35, 0.0, 1.0))
        contrast = float(np.median(vals) - np.median(roi[ys, xs]))
        contrast_score = float(np.clip(contrast / 40.0, 0.0, 1.0))
        self.confidence = 0.5 * shape_score + 0.5 * contrast_score

    @property
    def center(self) -> Optional[Tuple[float, float]]:
        if self.x is None or self.y is None:
            return None
        return (self.x, self.y)

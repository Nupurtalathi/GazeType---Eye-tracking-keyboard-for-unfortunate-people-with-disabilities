"""
Gaze Detection Engine.
Wraps dlib facial landmark prediction, pupil isolation, and eye geometry
to produce continuous gaze ratios and diagnostic annotated frames.
Based on the architectural principles of antoinelame/GazeTracking,
extended with binocular weighting, confidence scoring, and continuous coordinates.
"""

import cv2
import dlib
import numpy as np
import os
from typing import Optional, Tuple
from collections import deque
from .eye import Eye
from .confidence import GazeConfidence
from .blink import BlinkFilter
from .model_downloader import ensure_model_exists


class GazeDetector:
    # Blink handling: see gaze/blink.py (duration-aware; sustained low EAR = looking down).

    def __init__(self, model_path: Optional[str] = None, blink_threshold: float = 0.15, min_confidence: float = 0.45):
        if model_path is None or not os.path.exists(model_path):
            model_path = ensure_model_exists(model_path)

        self.model_path = model_path
        self.blink_threshold = blink_threshold
        self.min_confidence = min_confidence

        # Initialize dlib detectors
        self._face_detector = dlib.get_frontal_face_detector()
        self._landmark_predictor = dlib.shape_predictor(model_path)

        # State attributes
        self.frame: Optional[np.ndarray] = None
        self.face: Optional[dlib.rectangle] = None
        self.landmarks = None
        self.eye_left: Optional[Eye] = None
        self.eye_right: Optional[Eye] = None

        self._confidence_score: float = 0.0
        self._is_valid: bool = False
        self._status_text: str = "Uninitialized"
        self._blink = BlinkFilter(hard_floor=min(blink_threshold, 0.13))

    def refresh(self, frame: np.ndarray, timestamp: Optional[float] = None) -> None:
        """
        Refreshes frame analysis. Detects faces, localizes landmarks,
        and computes pupil positions and gaze metrics.
        """
        self.frame = frame
        self._timestamp = timestamp
        if frame is None or frame.size == 0:
            self._reset_state("No frame")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect face (scale 0 for high FPS on CPU)
        faces = self._face_detector(gray, 0)
        if len(faces) == 0:
            self._reset_state("Face not detected")
            return

        # Pick largest face if multiple detected
        self.face = max(faces, key=lambda rect: rect.width() * rect.height())
        self.landmarks = self._landmark_predictor(gray, self.face)

        # Extract left and right eyes (dlib indices: 42-47 left, 36-41 right)
        self.eye_left = Eye(frame, self.landmarks, Eye.LEFT_EYE_POINTS, gray_frame=gray)
        self.eye_right = Eye(frame, self.landmarks, Eye.RIGHT_EYE_POINTS, gray_frame=gray)
        self._apply_adaptive_blink()

        # Evaluate overall confidence and validity
        self._confidence_score, self._is_valid, self._status_text = GazeConfidence.evaluate(
            face_detected=True,
            left_eye=self.eye_left,
            right_eye=self.eye_right,
            min_threshold=self.min_confidence
        )

    def _apply_adaptive_blink(self) -> None:
        ear = (self.eye_left.ear + self.eye_right.ear) / 2.0
        if self._blink.update(ear, getattr(self, '_timestamp', None)):
            for eye in (self.eye_left, self.eye_right):
                eye.is_blinking = True
                eye.horizontal_ratio = None
                eye.vertical_ratio = None

    def _reset_state(self, reason: str) -> None:
        self.face = None
        self.landmarks = None
        self.eye_left = None
        self.eye_right = None
        self._confidence_score = 0.0
        self._is_valid = False
        self._status_text = reason

    def pupil_left_coords(self) -> Optional[Tuple[int, int]]:
        """Returns (x, y) coordinates of left pupil in webcam frame coordinates."""
        if self.eye_left and self.eye_left.pupil_frame_coords:
            return self.eye_left.pupil_frame_coords
        return None

    def pupil_right_coords(self) -> Optional[Tuple[int, int]]:
        """Returns (x, y) coordinates of right pupil in webcam frame coordinates."""
        if self.eye_right and self.eye_right.pupil_frame_coords:
            return self.eye_right.pupil_frame_coords
        return None

    def horizontal_ratio(self) -> Optional[float]:
        """
        Returns continuous horizontal gaze ratio (0.0 = looking left, 1.0 = looking right).
        Blends both eyes or falls back to whichever eye has valid data.
        """
        if self.is_blinking():
            return None

        hr_l = self.eye_left.horizontal_ratio if (self.eye_left and not self.eye_left.is_blinking) else None
        hr_r = self.eye_right.horizontal_ratio if (self.eye_right and not self.eye_right.is_blinking) else None

        if hr_l is not None and hr_r is not None:
            return float((hr_l + hr_r) / 2.0)
        elif hr_l is not None:
            return hr_l
        elif hr_r is not None:
            return hr_r
        return None

    def vertical_ratio(self) -> Optional[float]:
        """
        Returns continuous vertical gaze ratio (0.0 = looking up, 1.0 = looking down).
        """
        if self.is_blinking():
            return None

        vr_l = self.eye_left.vertical_ratio if (self.eye_left and not self.eye_left.is_blinking) else None
        vr_r = self.eye_right.vertical_ratio if (self.eye_right and not self.eye_right.is_blinking) else None

        if vr_l is not None and vr_r is not None:
            return float((vr_l + vr_r) / 2.0)
        elif vr_l is not None:
            return vr_l
        elif vr_r is not None:
            return vr_r
        return None

    def is_left(self, threshold: float = 0.38) -> bool:
        hr = self.horizontal_ratio()
        return hr is not None and hr <= threshold

    def is_right(self, threshold: float = 0.62) -> bool:
        hr = self.horizontal_ratio()
        return hr is not None and hr >= threshold

    def is_center(self, left_thresh: float = 0.38, right_thresh: float = 0.62) -> bool:
        hr = self.horizontal_ratio()
        return hr is not None and left_thresh < hr < right_thresh

    def lid_aperture(self):
        if self.eye_left is None or self.eye_right is None or self.horizontal_ratio() is None:
            return None
        return (self.eye_left.ear + self.eye_right.ear) / 2.0

    def pose_features(self):
        h, l = self.head_pose(), self.lid_aperture()
        if h is None:
            return None
        return (h[0], h[1], l) if l is not None else h

    def head_pose(self):
        """(yaw, pitch) proxies from nose tip vs outer eye corners (same definition as MediaPipe backend)."""
        if self.landmarks is None:
            return None
        from .mediapipe_detector import _head_proxy
        pt = lambda i: np.array([self.landmarks.part(i).x, self.landmarks.part(i).y], float)
        return _head_proxy(pt(30), pt(36), pt(45))

    def is_blinking(self) -> bool:
        """Returns True if eyes are closed."""
        if self.eye_left and self.eye_right:
            return self.eye_left.is_blinking and self.eye_right.is_blinking
        return False

    def is_face_detected(self) -> bool:
        return self.face is not None

    def confidence(self) -> float:
        return self._confidence_score

    def status(self) -> str:
        return self._status_text

    def is_valid(self) -> bool:
        return self._is_valid

    def annotated_frame(self) -> np.ndarray:
        """Returns a copy of the frame decorated with tracking diagnostics."""
        if self.frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        annotated = self.frame.copy()

        # Draw face bounding box
        if self.face is not None:
            x1, y1 = self.face.left(), self.face.top()
            x2, y2 = self.face.right(), self.face.bottom()
            color = (0, 220, 100) if self._is_valid else (0, 165, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw Eye Contours
            for eye, name in [(self.eye_left, "L"), (self.eye_right, "R")]:
                if eye is not None and len(eye.landmarks_points) > 0:
                    cv2.polylines(annotated, [eye.landmarks_points], True, (255, 255, 0), 1)

                    if eye.pupil_frame_coords is not None:
                        px, py = int(round(eye.pupil_frame_coords[0])), int(round(eye.pupil_frame_coords[1]))
                        # Pupil marker
                        cv2.circle(annotated, (px, py), 3, (0, 0, 255), -1)
                        cv2.circle(annotated, (px, py), 6, (0, 255, 255), 1)

                        # Small gaze direction indicator
                        if eye.horizontal_ratio is not None and eye.vertical_ratio is not None:
                            dx = int((eye.horizontal_ratio - 0.5) * 30)
                            dy = int((eye.vertical_ratio - 0.5) * 20)
                            cv2.arrowedLine(annotated, (px, py), (px + dx, py + dy), (0, 255, 0), 1, tipLength=0.3)

        return annotated

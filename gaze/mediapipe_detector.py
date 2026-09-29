"""
MediaPipe Face Mesh + Iris gaze backend (v3, default).

Why: dlib's 68-point landmarks jitter by 1-2 px per frame and have no iris point, so the
pupil had to be found by thresholding. MediaPipe Face Mesh with refine_landmarks=True
tracks 478 points including 5 iris points per eye, with temporal tracking built in.

Measured on the candidate video (same segments, same metric = target spread / within-
fixation SD, higher is better):

                         640 px input     320 px input (simulates sitting further away)
    horizontal  dlib v1      22.7             10.8
                MediaPipe    35.7             41.8
    vertical    dlib v2      11.5              6.3
                MediaPipe    25   (iris offset fused with lid aperture)

Drop-in replacement for GazeDetector: same public methods, so the dashboard, calibration,
keyboard and replay tool work unchanged. Runs fully offline (models ship inside the
mediapipe wheel) - no frames leave the machine.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from .blink import BlinkFilter

try:
    import mediapipe as mp
    _FACE_MESH = mp.solutions.face_mesh
    MEDIAPIPE_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    _FACE_MESH = None
    MEDIAPIPE_AVAILABLE = False

# (corner_a, corner_b, iris_centre, upper_lid, lower_lid) - corner_a is image-left
_EYES = {
    "image_left": (33, 133, 468, 159, 145),
    "image_right": (362, 263, 473, 386, 374),
}
_CONTOURS = {
    "image_left": [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
    "image_right": [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382],
}

# Feature scaling -> keeps ratios inside [0,1] with head-room for screen edges.
HR_GAIN = 2.0          # hr = 0.5 + HR_GAIN * (h_raw - 0.5)
VR_GAIN = 4.0          # vr = 0.5 + VR_GAIN * (fused_v - VR_CENTER)
VR_CENTER = -0.075
LID_WEIGHT = 0.5       # vertical = (1-w)*iris_offset + w*(LID_SLOPE*aperture); w=0.5 best on video
LID_SLOPE = -0.35      # looking down closes the lids; fitted on the candidate video


def _head_proxy(nose, outer_a, outer_b) -> Optional[Tuple[float, float]]:
    iod = float(np.linalg.norm(outer_b - outer_a))
    if iod < 10:
        return None
    ex = (outer_b - outer_a) / iod
    ey = np.array([-ex[1], ex[0]])
    d = nose - (outer_a + outer_b) / 2.0
    return (float(d @ ex / iod), float(d @ ey / iod))


class _EyeView:
    """Minimal per-eye view so existing UI code (eye_left / eye_right) keeps working."""

    def __init__(self):
        self.ear = 0.0
        self.is_blinking = False
        self.horizontal_ratio: Optional[float] = None
        self.vertical_ratio: Optional[float] = None
        self.pupil_frame_coords: Optional[Tuple[float, float]] = None
        self.landmarks_points = np.array([])
        self.pupil = None


class MediaPipeGazeDetector:
    MAX_LR_DISAGREEMENT = 0.20   # raw hr units; eyes legitimately disagree more at screen edges

    def __init__(self, model_path: Optional[str] = None, blink_threshold: float = 0.10,
                 min_confidence: float = 0.45):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("mediapipe is not installed (pip install -r requirements.txt)")
        self.blink_threshold = blink_threshold
        self.min_confidence = min_confidence
        self._mesh = _FACE_MESH.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.frame: Optional[np.ndarray] = None
        self.face = None
        self.landmarks: Optional[np.ndarray] = None
        self.eye_left: Optional[_EyeView] = None
        self.eye_right: Optional[_EyeView] = None
        self._hr: Optional[float] = None
        self._vr: Optional[float] = None
        self._confidence = 0.0
        self._valid = False
        self._status = "Uninitialized"
        self._blink = BlinkFilter(hard_floor=blink_threshold)
        self._blinking = False
        self._head = None

    # ------------------------------------------------------------------ core
    def refresh(self, frame: np.ndarray, timestamp: Optional[float] = None) -> None:
        self.frame = frame
        self._hr = self._vr = None
        self._head = None
        self._blinking = False
        if frame is None or frame.size == 0:
            return self._reset("No frame")
        h, w = frame.shape[:2]
        res = self._mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return self._reset("Face not detected")

        L = np.array([(p.x * w, p.y * h) for p in res.multi_face_landmarks[0].landmark], np.float64)
        self.landmarks = L
        self.face = (int(L[:, 0].min()), int(L[:, 1].min()), int(L[:, 0].max()), int(L[:, 1].max()))
        self._head = _head_proxy(L[1], L[33], L[263])

        views, hs, vs, ears = {}, [], [], []
        for name, (a, b, c, t, bo) in _EYES.items():
            A, B, C = L[a], L[b], L[c]
            ew = float(np.linalg.norm(B - A))
            ev = _EyeView()
            ev.landmarks_points = L[_CONTOURS[name]].astype(np.int32)
            ev.pupil_frame_coords = (float(C[0]), float(C[1]))
            if ew < 4:
                views[name] = ev
                continue
            ex = (B - A) / ew
            ey = np.array([-ex[1], ex[0]])
            d = C - A
            ev.ear = float(np.linalg.norm(L[t] - L[bo]) / ew)
            h_raw, v_raw = float(d @ ex / ew), float(d @ ey / ew)
            fused = (1 - LID_WEIGHT) * v_raw + LID_WEIGHT * LID_SLOPE * ev.ear
            ev.horizontal_ratio = float(np.clip(0.5 + HR_GAIN * (h_raw - 0.5), 0, 1))
            ev.vertical_ratio = float(np.clip(0.5 + VR_GAIN * (fused - VR_CENTER), 0, 1))
            hs.append(h_raw); vs.append(fused); ears.append(ev.ear)
            views[name] = ev
        # keep the old naming: eye_left = subject's left eye = image-right in an un-mirrored frame
        self.eye_left, self.eye_right = views.get("image_right"), views.get("image_left")

        if len(hs) < 2:
            return self._reset("Eyes not localised", keep_face=True)

        # ---- blink filter (duration-aware: sustained low EAR = looking down, not a blink)
        ear = float(np.mean(ears))
        self._ear = ear
        if self._blink.update(ear, timestamp):
            self._blinking = True
            for ev in (self.eye_left, self.eye_right):
                ev.is_blinking = True
                ev.horizontal_ratio = ev.vertical_ratio = None
            self._confidence, self._valid = 0.1, False
            self._status = "Blinking"
            return

        # ---- binocular fusion
        disagreement = abs(hs[0] - hs[1])
        h_raw, fused = float(np.mean(hs)), float(np.mean(vs))
        self._hr = float(np.clip(0.5 + HR_GAIN * (h_raw - 0.5), 0, 1))
        self._vr = float(np.clip(0.5 + VR_GAIN * (fused - VR_CENTER), 0, 1))

        # NOTE: eye openness is deliberately NOT part of confidence any more - low openness is
        # normal when looking at the bottom of the screen and used to push those frames out.
        agreement = float(np.clip(1 - disagreement / self.MAX_LR_DISAGREEMENT, 0, 1))
        self._confidence = 0.5 + 0.5 * agreement
        self._valid = self._confidence >= self.min_confidence
        self._status = "Valid" if self._valid else "Low Confidence"

    def _reset(self, reason: str, keep_face: bool = False) -> None:
        if not keep_face:
            self.face = None
            self.landmarks = None
            self.eye_left = self.eye_right = None
        self._hr = self._vr = None
        if not keep_face:
            self._head = None
        self._confidence, self._valid, self._status = 0.0, False, reason

    # ------------------------------------------------------------- public API
    def horizontal_ratio(self) -> Optional[float]:
        return self._hr

    def lid_aperture(self) -> Optional[float]:
        """Mean eyelid opening / eye width of the last valid frame (lower when looking down)."""
        return getattr(self, "_ear", None) if self._hr is not None else None

    def pose_features(self):
        """Extra calibration features: (yaw, pitch, lid). None if unavailable."""
        h, l = self.head_pose(), self.lid_aperture()
        if h is None:
            return None
        return (h[0], h[1], l) if l is not None else h

    def head_pose(self) -> Optional[Tuple[float, float]]:
        """(yaw, pitch) proxies: nose-tip offset from the eye-corner midpoint / inter-ocular distance."""
        return self._head

    def vertical_ratio(self) -> Optional[float]:
        return self._vr

    def is_blinking(self) -> bool:
        return self._blinking

    def is_face_detected(self) -> bool:
        return self.face is not None

    def confidence(self) -> float:
        return self._confidence

    def is_valid(self) -> bool:
        return self._valid

    def status(self) -> str:
        return self._status

    def pupil_left_coords(self):
        return self.eye_left.pupil_frame_coords if self.eye_left else None

    def pupil_right_coords(self):
        return self.eye_right.pupil_frame_coords if self.eye_right else None

    def annotated_frame(self) -> np.ndarray:
        if self.frame is None:
            return np.zeros((480, 640, 3), np.uint8)
        out = self.frame.copy()
        if self.face is not None:
            x1, y1, x2, y2 = self.face
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 100) if self._valid else (0, 165, 255), 2)
        for ev in (self.eye_left, self.eye_right):
            if ev is None or len(ev.landmarks_points) == 0:
                continue
            cv2.polylines(out, [ev.landmarks_points], True, (255, 255, 0), 1)
            if ev.pupil_frame_coords and not ev.is_blinking:
                px, py = (int(round(v)) for v in ev.pupil_frame_coords)
                cv2.circle(out, (px, py), 2, (0, 0, 255), -1)
                cv2.circle(out, (px, py), 5, (0, 255, 255), 1)
        return out


def create_detector(backend: str, model_path: Optional[str], blink_threshold: float, min_confidence: float):
    """Factory used by the app. Falls back to dlib if mediapipe is missing."""
    if backend == "mediapipe" and MEDIAPIPE_AVAILABLE:
        return MediaPipeGazeDetector(model_path, min_confidence=min_confidence)
    if backend == "mediapipe":
        print("[Gaze] mediapipe not available - falling back to dlib backend (lower precision).")
    from .gaze_detector import GazeDetector
    return GazeDetector(model_path, blink_threshold, min_confidence)

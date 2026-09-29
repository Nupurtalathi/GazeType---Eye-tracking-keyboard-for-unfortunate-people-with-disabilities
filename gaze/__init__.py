"""Gaze tracking and computer vision modules."""
try:
    from .gaze_detector import GazeDetector          # needs dlib (legacy backend)
except ImportError:                                   # mediapipe-only install
    GazeDetector = None
from .calibration import GazeCalibration
from .smoothing import GazeSmoother
from .confidence import GazeConfidence
from .model_downloader import ensure_model_exists

__all__ = [
    "GazeDetector",
    "GazeCalibration",
    "GazeSmoother",
    "GazeConfidence",
    "ensure_model_exists",
]

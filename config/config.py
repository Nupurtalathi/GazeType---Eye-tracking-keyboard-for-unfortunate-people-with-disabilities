"""
Central Configuration for Real-Time Webcam Eye-Gaze Tracker.
Provides adjustable parameters for camera capture, gaze estimation,
smoothing filters, fixation detection, dwell thresholds, and logging.
"""

from dataclasses import dataclass, asdict, field
import json
import os
from typing import Tuple, List


@dataclass
class TrackerConfig:
    # Camera settings
    camera_index: int = 0
    camera_width: int = 1280        # 720p: eyes are ~2x more pixels than at 640x480 (MacBook / most laptop cams)
    camera_height: int = 720
    target_fps: int = 30

    # Screen resolution (auto-detected by GUI if 0)
    screen_width: int = 1920
    screen_height: int = 1080

    # Gaze backend: "mediapipe" (default, ~2x more precise, offline) or "dlib" (legacy)
    detector_backend: str = "mediapipe"

    # Eye & Blink detection
    blink_ear_threshold: float = 0.15  # Absolute floor only; main blink logic is adaptive (GazeDetector.BLINK_REL)
    blink_reset_time: float = 0.75     # Seconds of eye closure before resetting dwell (short blinks pause)

    # Gaze confidence
    min_gaze_confidence: float = 0.35  # Relaxed to allow smooth continuous tracking

    # Smoothing settings
    smoothing_method: str = "one_euro"  # "one_euro" or "ema"
    ema_alpha: float = 0.30
    one_euro_min_cutoff: float = 0.25    # v7: stronger smoothing (was 0.4)
    one_euro_beta: float = 0.006         # v7: less speed-boost -> jitter stays suppressed (was 0.01)
    one_euro_min_cutoff_y: float = 0.18  # vertical: even more smoothing (noisier axis; was 0.3)

    # v7 fixation-locked stabiliser (gaze/stabilizer.py) - used for typing
    stabilizer_enabled: bool = True
    saccade_frac: float = 0.50           # a jump > 50% of a key (on 2 frames) = deliberate move
    saccade_confirm_frames: int = 2
    saccade_detect_avg: int = 5
    fixation_window: int = 9             # frames in the fixation median (~0.3 s)
    fixation_alpha: float = 0.30

    # v7 key-selection stability (keyboard/selector.py)
    key_sticky_x: float = 0.15           # while fixating: leave a key by 15% of its width ...
    key_sticky_y: float = 0.25           # ... / 25% of its height before a neighbour takes over
    key_switch_time: float = 0.15
    key_settle_time: float = 0.12        # dwell starts after the gaze settled on the key
    key_repeat_factor: float = 1.8       # same key again without looking away: 1.8x dwell (0 = must look away)
    key_memory_time: float = 0.6         # brief glance away keeps 80% of the progress
    one_euro_d_cutoff: float = 1.0
    max_allowed_jump: float = 400.0  # Max pixel jump before treating as saccade

    # Fixation detector settings
    fixation_radius: float = 85.0     # Radius in pixels on screen (tolerant of micro-tremors)
    min_fixation_time: float = 0.08   # Minimum seconds to establish fixation (80 ms)

    # Dwell-time engine settings
    dwell_threshold: float = 0.90     # Responsive 0.90s for accessible typing
    dwell_cooldown: float = 0.50      # Seconds to pause after dwell trigger before re-triggering

    # Gaze range & sensitivity gains
    gain_x: float = 2.4               # Multiplier to expand human ocular range to full screen
    gain_y: float = 2.4
    center_hr: float = 0.43           # Resting center horizontal ratio
    center_vr: float = 0.39           # Resting center vertical ratio

    # Calibration settings
    calibration_model_type: str = "polynomial"  # "polynomial", "affine", "homography", "linear"
    calibration_samples_per_point: int = 30
    calibration_prep_delay: float = 1.0        # Seconds to settle gaze on point before recording
    calibration_sample_delay: float = 1.5      # Seconds to record samples
    calibration_grid_rows: int = 4        # 4x4 = 16 points (was 3x3 = 9)
    calibration_grid_cols: int = 4
    calibration_margin: float = 0.05      # targets 5% from the border (was 12%) -> edges are measured, not extrapolated

    calibration_validation: bool = True   # +8 edge targets: measure & correct reach to the screen borders
    reach_tuning_after_calibration: bool = True   # 9-key reach check right after calibration (~15 s)
    screen_diagonal_in: float = 14.0      # used to report errors in degrees / key size in degrees
    viewing_distance_cm: float = 50.0
    edge_gain_x: float = 1.0              # manual stretch about screen centre if edges are still hard to reach
    edge_gain_y: float = 1.0              #   (e.g. 1.10 = 10% further); normally leave at 1.0

    # Online drift correction from confirmed keyboard selections
    drift_correction_enabled: bool = True
    drift_kernel_px: float = 450.0

    # Gaze keyboard
    keyboard_letter_dwell: float = 1.0
    keyboard_phrase_dwell: float = 1.2
    keyboard_show_cursor: bool = True
    keyboard_phrases_file: str = ""
    keyboard_user_vocab_file: str = ""       # learned words (data/user_vocab.json by default)
    keyboard_speak_on_sentence_end: bool = True
    speech_echo: str = "word"          # "word" (speak each word as typed) | "letter" | "sentence" | "off"
    speech_rate_wpm: int = 160
    speech_backend: str = "auto"       # auto | say (macOS) | sapi | powershell (Windows) | espeak-ng | pyttsx3

    # File paths
    base_dir: str = field(default_factory=lambda: os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path: str = ""
    calibration_file: str = ""
    session_log_dir: str = ""

    def __post_init__(self):
        if not self.model_path:
            self.model_path = os.path.join(self.base_dir, "models", "shape_predictor_68_face_landmarks.dat")
        if not self.calibration_file:
            self.calibration_file = os.path.join(self.base_dir, "data", "calibration_profile.json")
        if not self.keyboard_phrases_file:
            self.keyboard_phrases_file = os.path.join(self.base_dir, "config", "keyboard_phrases.json")
        if not self.keyboard_user_vocab_file:
            self.keyboard_user_vocab_file = os.path.join(self.base_dir, "data", "user_vocab.json")
        if not self.session_log_dir:
            self.session_log_dir = os.path.join(self.base_dir, "data", "sessions")
        os.makedirs(self.session_log_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.calibration_file), exist_ok=True)

    def save_to_file(self, filepath: str = None) -> None:
        if filepath is None:
            filepath = os.path.join(self.base_dir, "config", "settings.json")
        data = asdict(self)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load_from_file(cls, filepath: str) -> "TrackerConfig":
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(**data)
        return cls()


# Default singleton instance
DEFAULT_CONFIG = TrackerConfig()

"""
Structured Data and Event Logger.
Logs significant gaze and dwell events (fixation changes, dwell activations,
region triggers, calibration steps) to CSV and JSON formats without excessive disk I/O.
Maintains a spatial buffer for session heatmap generation.
"""

import csv
import json
import os
import time
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Any


class GazeLogger:
    CSV_HEADERS = ["timestamp", "gaze_x", "gaze_y", "horizontal_ratio", "vertical_ratio", "dwell_time", "event", "details"]

    def __init__(self, log_dir: str = "data/sessions", auto_flush_interval: float = 5.0):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.auto_flush_interval = auto_flush_interval
        self.is_recording = False
        self.session_id: Optional[str] = None
        self.csv_path: Optional[str] = None

        self._event_buffer: List[Dict[str, Any]] = []
        self._gaze_points: List[Tuple[float, float, float]] = []  # (x, y, timestamp)
        self._last_flush = time.time()
        self._csv_file = None
        self._csv_writer = None

    def start_session(self, session_name: Optional[str] = None) -> str:
        """Starts a new recording session."""
        self.stop_session()

        now = datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        self.session_id = session_name if session_name else f"session_{timestamp_str}"
        self.csv_path = os.path.join(self.log_dir, f"{self.session_id}.csv")

        self._csv_file = open(self.csv_path, mode="w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.CSV_HEADERS)
        self._csv_writer.writeheader()
        self._csv_file.flush()

        self._event_buffer.clear()
        self._gaze_points.clear()
        self.is_recording = True

        self.log_event("session_start", 0, 0, 0, 0, 0.0, f"Session {self.session_id} initiated")
        return self.session_id

    def stop_session(self) -> None:
        """Flushes remaining buffers and closes the active file handle."""
        if not self.is_recording:
            return

        self.log_event("session_stop", 0, 0, 0, 0, 0.0, "Session finalized")
        self._flush()

        if self._csv_file is not None:
            try:
                self._csv_file.close()
            except OSError:
                pass
            self._csv_file = None
            self._csv_writer = None

        self.is_recording = False

    def log_event(
        self,
        event: str,
        gaze_x: Optional[float],
        gaze_y: Optional[float],
        hr: Optional[float],
        vr: Optional[float],
        dwell_time: float,
        details: str = ""
    ) -> None:
        """Appends a meaningful state event."""
        if not self.is_recording or self._csv_writer is None:
            return

        now_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        record = {
            "timestamp": now_str,
            "gaze_x": f"{gaze_x:.1f}" if gaze_x is not None else "None",
            "gaze_y": f"{gaze_y:.1f}" if gaze_y is not None else "None",
            "horizontal_ratio": f"{hr:.3f}" if hr is not None else "None",
            "vertical_ratio": f"{vr:.3f}" if vr is not None else "None",
            "dwell_time": f"{dwell_time:.2f}",
            "event": event,
            "details": details
        }

        self._event_buffer.append(record)

        if (time.time() - self._last_flush) >= self.auto_flush_interval or len(self._event_buffer) >= 20:
            self._flush()

    def record_gaze_point(self, gaze_x: float, gaze_y: float, timestamp: Optional[float] = None) -> None:
        """Stores gaze samples for heatmap reconstruction."""
        if not self.is_recording:
            return
        if timestamp is None:
            timestamp = time.time()
        self._gaze_points.append((float(gaze_x), float(gaze_y), float(timestamp)))

    def _flush(self) -> None:
        if self._csv_writer is not None and self._csv_file is not None and self._event_buffer:
            try:
                self._csv_writer.writerows(self._event_buffer)
                self._csv_file.flush()
                self._event_buffer.clear()
            except OSError:
                pass
        self._last_flush = time.time()

    def get_session_points(self) -> List[Tuple[float, float]]:
        """Returns list of (x, y) coordinates collected during this session."""
        return [(p[0], p[1]) for p in self._gaze_points]

    def export_json(self, target_path: Optional[str] = None) -> str:
        """Exports full session trajectory and events to structured JSON."""
        if target_path is None:
            target_path = os.path.join(self.log_dir, f"{self.session_id or 'session'}.json")

        data = {
            "session_id": self.session_id,
            "total_points": len(self._gaze_points),
            "events_count": len(self._event_buffer),
            "gaze_samples": [{"x": p[0], "y": p[1], "t": p[2]} for p in self._gaze_points]
        }

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return target_path

    def __del__(self):
        self.stop_session()

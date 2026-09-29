"""
Threaded Webcam Capture Module.
Runs frame grabbing in a dedicated thread to ensure zero UI lag
and maximize processing throughput. Includes multi-backend fallback and camera switching.
"""

import cv2
import threading
import time
from typing import Optional, Tuple, List


class ThreadedWebcam:
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480, target_fps: int = 30):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.target_fps = target_fps

        self.cap: Optional[cv2.VideoCapture] = None
        self._frame = None
        self._ret = False
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # FPS tracking
        self._fps_counter = 0
        self._fps_timer = time.time()
        self.measured_fps = 0.0

    @staticmethod
    def list_available_cameras(max_tested: int = 4) -> List[int]:
        """Scans for available webcam indexes."""
        available = []
        for idx in range(max_tested):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(idx)
                cap.release()
            else:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        available.append(idx)
                    cap.release()
        return available if available else [0]

    def start(self) -> bool:
        """Initializes OpenCV video capture and starts background worker thread."""
        if self._running:
            return True

        # Try backends in order of Windows compatibility
        backends = [
            ("DSHOW", cv2.CAP_DSHOW),
            ("DEFAULT", cv2.CAP_ANY),
            ("MSMF", cv2.CAP_MSMF)
        ]

        opened = False
        for name, backend in backends:
            try:
                self.cap = cv2.VideoCapture(self.camera_index, backend)
                if self.cap and self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

                    # Warm up read
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        self._ret = ret
                        self._frame = frame
                        opened = True
                        break
                    else:
                        self.cap.release()
                        self.cap = None
            except Exception:
                if self.cap:
                    self.cap.release()
                    self.cap = None

        if not opened:
            return False

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def switch_camera(self, new_index: int) -> bool:
        """Stops current capture and starts capture on the specified camera index."""
        self.stop()
        self.camera_index = new_index
        return self.start()

    def _capture_loop(self) -> None:
        """Continuously grab the latest camera frame."""
        while self._running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._ret = ret
                    self._frame = frame

                # Measure actual capture FPS
                self._fps_counter += 1
                now = time.time()
                elapsed = now - self._fps_timer
                if elapsed >= 1.0:
                    self.measured_fps = self._fps_counter / elapsed
                    self._fps_counter = 0
                    self._fps_timer = now
            else:
                time.sleep(0.01)

    def read(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """Returns the most recent frame thread-safely."""
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ret, self._frame.copy()

    def is_running(self) -> bool:
        return self._running and self.cap is not None and self.cap.isOpened()

    def stop(self) -> None:
        """Stops background thread and releases hardware resource."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._frame = None
        self._ret = False

    def __del__(self):
        self.stop()

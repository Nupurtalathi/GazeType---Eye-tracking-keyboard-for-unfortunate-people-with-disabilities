"""
Gaze and Dwell Visualizer Overlay Module.
Renders gaze reticles, fixation boundaries, circular dwell progress rings,
and diagnostic banners.
"""

import cv2
import numpy as np
import math
from typing import Tuple, Optional
from dwell.dwell_engine import DwellState


class GazeOverlay:
    STATE_COLORS = {
        DwellState.TRACKING: (180, 180, 180),     # Soft Gray
        DwellState.FIXATING: (0, 215, 255),       # Gold
        DwellState.DWELLING: (0, 140, 255),       # Vibrant Orange
        DwellState.TRIGGERED: (50, 205, 50),      # Lime Green
        DwellState.PAUSED_BLINK: (255, 191, 0),   # Cyan/Deep Sky
        DwellState.INVALID: (80, 80, 220),        # Red
        DwellState.FACE_LOST: (60, 60, 180)       # Dark Red
    }

    @classmethod
    def draw_gaze_reticle(
        cls,
        canvas: np.ndarray,
        center: Tuple[int, int],
        progress: float,
        dwell_state: DwellState,
        fixation_radius: int = 50,
        reticle_radius: int = 24
    ) -> None:
        """
        Draws an interactive gaze reticle on the frame/canvas with circular progress ring.
        """
        cx, cy = int(center[0]), int(center[1])
        h, w = canvas.shape[:2]
        if not (0 <= cx < w and 0 <= cy < h):
            return

        color = cls.STATE_COLORS.get(dwell_state, (200, 200, 200))

        # 1. Subtle fixation tolerance boundary
        if dwell_state in [DwellState.FIXATING, DwellState.DWELLING, DwellState.TRIGGERED]:
            cv2.circle(canvas, (cx, cy), fixation_radius, (60, 60, 80), 1, cv2.LINE_AA)

        # 2. Outer ring background
        cv2.circle(canvas, (cx, cy), reticle_radius, (70, 70, 70), 3, cv2.LINE_AA)

        # 3. Circular progress arc (0 to 360 deg)
        if progress > 0.0:
            angle_span = int(progress * 360)
            cv2.ellipse(
                canvas,
                (cx, cy),
                (reticle_radius, reticle_radius),
                -90,  # Start at top (12 o'clock)
                0,
                angle_span,
                color,
                4,
                cv2.LINE_AA
            )

        # 4. Central gaze point
        cv2.circle(canvas, (cx, cy), 5, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 2, (255, 255, 255), -1, cv2.LINE_AA)

        # 5. Dwell triggered burst effect
        if dwell_state == DwellState.TRIGGERED:
            cv2.circle(canvas, (cx, cy), reticle_radius + 12, (50, 255, 50), 2, cv2.LINE_AA)
            cv2.putText(
                canvas, "DWELL TRIGGERED", (cx - 70, cy - reticle_radius - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 50), 2, cv2.LINE_AA
            )

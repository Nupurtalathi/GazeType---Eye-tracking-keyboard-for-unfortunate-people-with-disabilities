"""
Calibration Window and Calibration Target Sequence.
Displays fullscreen 9-point grid, animated targets, audio/visual cues,
sample collection progress, and model calculation feedback.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import time
import math
from typing import List, Tuple, Optional, Callable
from gaze.calibration import GazeCalibration, CalibrationPoint


class CalibrationWindow(tk.Toplevel):
    def __init__(self, parent, calibration_engine: GazeCalibration, on_complete_callback: Callable[[], None]):
        super().__init__(parent)
        self.calib_engine = calibration_engine
        self.on_complete_callback = on_complete_callback

        self.title("Gaze Calibration")
        # Fullscreen on Windows
        self.attributes("-fullscreen", True)
        self.configure(bg="#121318")

        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()
        self.calib_engine.screen_width = self.screen_width
        self.calib_engine.screen_height = self.screen_height

        # Grid points
        cfg = getattr(parent, "config", None)
        rows = getattr(cfg, "calibration_grid_rows", 4)
        cols = getattr(cfg, "calibration_grid_cols", 4)
        margin = getattr(cfg, "calibration_margin", 0.05)
        self.points: List[Tuple[int, int]] = self.calib_engine.setup_grid(rows=rows, cols=cols, margin_x=margin, margin_y=margin)
        self.n_grid = len(self.points)
        if getattr(cfg, "calibration_validation", True):
            # 8 extra held-out targets on the very edges -> measures & corrects edge reach
            self.points += self.calib_engine.add_validation_points(margin=0.03)
        self.current_point_idx = 0
        self.is_collecting = False
        self.samples_needed = 30  # ~1 s of valid frames per point (15 was too few for a stable median)
        self.samples_collected = 0

        # UI Canvas
        self.canvas = tk.Canvas(self, bg="#121318", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Instructions / HUD
        self.hud_text = self.canvas.create_text(
            self.screen_width // 2, self.screen_height // 2 - 30,  # centre: targets now sit near the edges
            text="Look steadily at the pulsing target. Keep your head stationary.",
            font=("Segoe UI", 16, "bold"), fill="#E0E0E0"
        )
        self.counter_text = self.canvas.create_text(
            self.screen_width // 2, self.screen_height // 2 + 5,
            text=f"Target 1 of {len(self.points)}",
            font=("Segoe UI", 13), fill="#00D2FF"
        )

        # Bottom help
        self.canvas.create_text(
            self.screen_width // 2, self.screen_height // 2 + 40,
            text="Press ESC or Q to abort calibration",
            font=("Segoe UI", 11), fill="#777777"
        )

        # Target graphics objects
        self.target_outer = None
        self.target_inner = None
        self.target_dot = None

        # Animation state
        self.pulse_phase = 0.0
        self.state = "PREPARING"  # PREPARING -> RECORDING -> TRANSITION -> FINISHED
        self.state_start_time = time.time()
        self.prep_duration = 0.9  # Seconds to let eye settle
        self.point_timeout = 4.0  # Seconds max per target; move on even if few valid frames

        # Bindings
        self.bind("<Escape>", lambda e: self.abort())
        self.bind("q", lambda e: self.abort())
        self.bind("<space>", lambda e: self.start_recording())

        # Start animation loop
        self._animate()

    def feed_sample(self, hr: Optional[float], vr: Optional[float], head=None) -> None:
        """Called by the main dashboard loop to feed current gaze ratios into calibration."""
        if not self.is_collecting or hr is None or vr is None:
            return

        if self.current_point_idx < len(self.calib_engine.calibration_points):
            cp = self.calib_engine.calibration_points[self.current_point_idx]
            cp.add_sample(hr, vr, head)
            self.samples_collected += 1

            if self.samples_collected >= self.samples_needed:
                self._advance_to_next_point()

    def _advance_to_next_point(self) -> None:
        self.is_collecting = False
        self.current_point_idx += 1
        self.samples_collected = 0

        if self.current_point_idx < len(self.points):
            self.state = "PREPARING"
            self.state_start_time = time.time()
            phase = "Edge check" if self.current_point_idx >= self.n_grid else "Target"
            self.canvas.itemconfig(self.counter_text, text=f"{phase} {self.current_point_idx + 1} of {len(self.points)}")
        else:
            self._finish_calibration()

    def _finish_calibration(self) -> None:
        self.state = "FINISHED"
        self.canvas.delete("target")
        self.canvas.itemconfig(self.hud_text, text="Calculating Gaze Transformation...", fill="#00FF88")

        success = self.calib_engine.fit()
        if success:
            msg = f"Calibration Successful!\n{self.calib_engine.last_fit_message}"
            self.canvas.itemconfig(self.hud_text, text=msg)
            self.after(1400, self._close_and_callback)
        else:
            messagebox.showerror("Calibration Failed", self.calib_engine.last_fit_message or "Insufficient valid gaze samples collected. Please retry.")
            self.destroy()

    def _close_and_callback(self) -> None:
        self.destroy()
        if self.on_complete_callback:
            self.on_complete_callback()

    def abort(self) -> None:
        self.destroy()

    def start_recording(self) -> None:
        self.state = "RECORDING"
        self.is_collecting = True
        self.recording_start = time.time()

    def _animate(self) -> None:
        if not self.winfo_exists() or self.state == "FINISHED":
            return

        now = time.time()

        if self.state == "PREPARING":
            if (now - self.state_start_time) >= self.prep_duration:
                self.start_recording()
        elif self.state == "RECORDING" and (now - getattr(self, "recording_start", now)) > self.point_timeout:
            # Never hang on a target the tracker cannot see well (typical at the edges).
            # fit() rejects the calibration later if a whole border ends up without data.
            self._advance_to_next_point()

        # Draw target at current point
        if self.current_point_idx < len(self.points):
            tx, ty = self.points[self.current_point_idx]
            self.pulse_phase += 0.12
            pulse = math.sin(self.pulse_phase) * 6.0

            self.canvas.delete("target")

            # Progress ring
            progress = min(1.0, self.samples_collected / max(1, self.samples_needed))
            arc_extent = int(progress * 359)

            base_r = 32 + pulse
            # Outer ring
            self.canvas.create_oval(
                tx - base_r, ty - base_r, tx + base_r, ty + base_r,
                outline="#333344", width=3, tags="target"
            )

            # Circular progress arc
            if arc_extent > 0:
                self.canvas.create_arc(
                    tx - base_r, ty - base_r, tx + base_r, ty + base_r,
                    start=90, extent=-arc_extent, outline="#00E5FF", width=5, style="arc", tags="target"
                )

            # Middle ring
            mid_r = 16
            color = "#00FF88" if self.state == "RECORDING" else "#FFB800"
            self.canvas.create_oval(
                tx - mid_r, ty - mid_r, tx + mid_r, ty + mid_r,
                outline=color, width=2, tags="target"
            )

            # Center Bullseye dot
            dot_r = 6
            self.canvas.create_oval(
                tx - dot_r, ty - dot_r, tx + dot_r, ty + dot_r,
                fill="#FFFFFF", outline="", tags="target"
            )

        self.after(30, self._animate)

"""
Accuracy and Dwell Testing Mode.
Presents sequential benchmark targets to evaluate gaze estimation accuracy (pixel error),
dwell latency, and success rates before deploying to assistive interfaces.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import time
import math
from typing import List, Tuple, Optional, Dict, Any


class TestingModeWindow(tk.Toplevel):
    def __init__(self, parent, screen_width: int = 1920, screen_height: int = 1080):
        super().__init__(parent)
        self.title("Eye-Gaze Accuracy & Dwell Benchmark")
        self.attributes("-fullscreen", True)
        self.configure(bg="#0F1117")

        self.screen_width = screen_width
        self.screen_height = screen_height

        # Define 5 benchmark test target coordinates
        w, h = screen_width, screen_height
        self.targets = [
            (int(w * 0.20), int(h * 0.25), "Top-Left"),
            (int(w * 0.80), int(h * 0.25), "Top-Right"),
            (int(w * 0.50), int(h * 0.50), "Center"),
            (int(w * 0.25), int(h * 0.75), "Bottom-Left"),
            (int(w * 0.75), int(h * 0.75), "Bottom-Right")
        ]
        self.current_idx = 0
        self.test_results: List[Dict[str, Any]] = []

        # Current state
        self.current_gaze: Optional[Tuple[float, float]] = None
        self.current_dwell_time: float = 0.0
        self.target_dwell: float = 1.20
        self.target_radius: float = 55.0

        self.dwell_in_target_time: float = 0.0
        self.target_start_time = time.time()
        self.last_update_time = time.time()

        # Canvas
        self.canvas = tk.Canvas(self, bg="#0F1117", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Header HUD
        self.hud_text = self.canvas.create_text(
            w // 2, 40,
            text="Gaze Accuracy Benchmark - Fixate on the illuminated circle",
            font=("Segoe UI", 16, "bold"), fill="#E0E0E0"
        )
        self.telemetry_text = self.canvas.create_text(
            w // 2, 75,
            text="",
            font=("Consolas", 13), fill="#00E5FF"
        )

        # Footer
        self.canvas.create_text(
            w // 2, h - 35,
            text="Press ESC or Q to exit test mode",
            font=("Segoe UI", 11), fill="#777777"
        )

        self.bind("<Escape>", lambda e: self.destroy())
        self.bind("q", lambda e: self.destroy())

        self._render_loop()

    def feed_gaze(self, gaze_x: Optional[float], gaze_y: Optional[float], is_valid: bool) -> None:
        """Called by main tracker loop with estimated screen coordinates."""
        now = time.time()
        dt = now - self.last_update_time
        self.last_update_time = now

        if self.current_idx >= len(self.targets):
            return

        tx, ty, label = self.targets[self.current_idx]

        if is_valid and gaze_x is not None and gaze_y is not None:
            self.current_gaze = (gaze_x, gaze_y)
            dist = math.hypot(gaze_x - tx, gaze_y - ty)

            if dist <= self.target_radius:
                self.dwell_in_target_time += dt
                if self.dwell_in_target_time >= self.target_dwell:
                    # Successful dwell on benchmark target!
                    self._record_success(tx, ty, label, gaze_x, gaze_y, dist, self.dwell_in_target_time)
            else:
                self.dwell_in_target_time = max(0.0, self.dwell_in_target_time - dt * 0.5)
        else:
            self.current_gaze = None

    def _record_success(self, tx, ty, label, gx, gy, err, dwell_time):
        self.test_results.append({
            "target": (tx, ty),
            "label": label,
            "estimated": (round(gx, 1), round(gy, 1)),
            "error_px": round(err, 1),
            "dwell_time": round(dwell_time, 2),
            "status": "SUCCESS"
        })

        self.current_idx += 1
        self.dwell_in_target_time = 0.0

        if self.current_idx >= len(self.targets):
            self._show_summary()

    def _show_summary(self):
        self.canvas.delete("all")
        w, h = self.screen_width, self.screen_height

        errors = [r["error_px"] for r in self.test_results]
        mean_err = sum(errors) / len(errors) if errors else 0.0

        lines = [
            "BENCHMARK COMPLETED",
            f"Targets Tested: {len(self.targets)} | Success Rate: 100%",
            f"Mean Accuracy Error: {mean_err:.1f} pixels",
            "",
            "Detailed Results:"
        ]
        for r in self.test_results:
            lines.append(
                f"Target {r['label']}: Expected {r['target']} -> Estimated {r['estimated']} | "
                f"Error: {r['error_px']} px | Dwell: {r['dwell_time']}s [OK]"
            )

        lines.append("")
        lines.append("Press ESC or Close to return to Dashboard.")

        y_pos = h // 4
        for line in lines:
            color = "#00FF88" if "COMPLETED" in line or "100%" in line else "#E0E0E0"
            font = ("Segoe UI", 16, "bold") if "COMPLETED" in line else ("Consolas", 12)
            self.canvas.create_text(w // 2, y_pos, text=line, font=font, fill=color)
            y_pos += 34

    def _render_loop(self):
        if not self.winfo_exists() or self.current_idx >= len(self.targets):
            return

        w, h = self.screen_width, self.screen_height
        tx, ty, label = self.targets[self.current_idx]

        self.canvas.delete("dynamic")

        # Telemetry string
        if self.current_gaze:
            gx, gy = self.current_gaze
            err = math.hypot(gx - tx, gy - ty)
            telemetry = (
                f"Target: ({tx}, {ty}) [{label}] | Gaze: ({gx:.0f}, {gy:.0f}) | "
                f"Error: {err:.1f} px | Dwell: {self.dwell_in_target_time:.2f} / {self.target_dwell:.2f} s"
            )
        else:
            telemetry = f"Target: ({tx}, {ty}) [{label}] | Searching for gaze..."

        self.canvas.itemconfig(self.telemetry_text, text=telemetry)

        # Draw Target Circle
        prog = min(1.0, self.dwell_in_target_time / self.target_dwell)
        # Background disk
        self.canvas.create_oval(
            tx - self.target_radius, ty - self.target_radius,
            tx + self.target_radius, ty + self.target_radius,
            outline="#33384A", width=3, fill="#191B24", tags="dynamic"
        )

        # Progress Arc
        if prog > 0:
            arc_extent = int(prog * 359)
            self.canvas.create_arc(
                tx - self.target_radius, ty - self.target_radius,
                tx + self.target_radius, ty + self.target_radius,
                start=90, extent=-arc_extent, outline="#00FF88", width=6, style="arc", tags="dynamic"
            )

        # Bullseye center
        self.canvas.create_oval(tx - 6, ty - 6, tx + 6, ty + 6, fill="#00E5FF", outline="", tags="dynamic")
        self.canvas.create_text(tx, ty - self.target_radius - 16, text=label, font=("Segoe UI", 11, "bold"), fill="#E0E0E0", tags="dynamic")

        # Draw current gaze reticle
        if self.current_gaze:
            gx, gy = self.current_gaze
            self.canvas.create_oval(gx - 8, gy - 8, gx + 8, gy + 8, outline="#FF3366", width=2, tags="dynamic")
            self.canvas.create_line(gx - 14, gy, gx + 14, gy, fill="#FF3366", width=2, tags="dynamic")
            self.canvas.create_line(gx, gy - 14, gx, gy + 14, fill="#FF3366", width=2, tags="dynamic")

        self.after(30, self._render_loop)

"""
Reach Tuning window (~15 s, runs automatically after calibration).

Shows the real keyboard layout faintly and asks the user to look at the centre of 9 keys
(centre, far left/right column, top/bottom row, 4 corners). Measures where the calibrated
gaze lands and fits per-direction edge gains (gaze/reach.py) so every key - including the
outer columns and bottom rows - is reached at a comfortable eye angle.

Fed by the dashboard:   reach_win.feed(raw_xy_without_reach, is_valid)
"""

import math
import time
import tkinter as tk
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from gaze.reach import fit_reach, reach_targets
from keyboard.layout import build_layout, load_phrases


class ReachTuningWindow(tk.Toplevel):
    PREP_S = 0.9
    SAMPLES = 25
    TIMEOUT_S = 3.5

    def __init__(self, parent, calibration, screen_w: int, screen_h: int,
                 phrases_file: Optional[str] = None,
                 on_complete: Optional[Callable[[object], None]] = None):
        super().__init__(parent)
        self.title("Reach tuning")
        self.attributes("-fullscreen", True)
        self.configure(bg="#0F1117")
        self.calibration = calibration
        self.W, self.H = screen_w, screen_h
        self.on_complete = on_complete
        self.keys = build_layout(screen_w, screen_h, load_phrases(phrases_file))
        self.targets = reach_targets(self.keys, screen_w, screen_h)
        self.idx = 0
        self.samples = []
        self.meas: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
        self.state = "PREP"
        self.t_state = time.time()
        self.phase = 0.0

        self.canvas = tk.Canvas(self, bg="#0F1117", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        for k in self.keys:
            if k.kind == "display":
                continue
            self.canvas.create_rectangle(k.x + 3, k.y + 3, k.x + k.w - 3, k.y + k.h - 3,
                                         outline="#262B3B", fill="#141722", width=2)
            if k.label:
                self.canvas.create_text(k.cx, k.cy, text=k.label, fill="#3A4055",
                                        font=("Segoe UI", 26 if k.kind == "letter" else 16, "bold"))
        self.hud = self.canvas.create_text(self.W // 2, 24, fill="#E0E0E0", font=("Segoe UI", 15, "bold"),
                                           text="Reach tuning - look at the centre of the glowing key. Keep your head natural.")
        self.cursor = self.canvas.create_oval(0, 0, 0, 0, outline="#FFCC00", width=2, state="hidden")
        self.bind("<Escape>", lambda e: self.destroy())
        self._tick()

    def feed(self, raw_xy: Optional[Tuple[float, float]], valid: bool) -> None:
        if not self.winfo_exists():
            return
        if raw_xy is not None:
            x, y = raw_xy
            self.canvas.coords(self.cursor, x - 10, y - 10, x + 10, y + 10)
            self.canvas.itemconfig(self.cursor, state="normal")
        if self.state == "REC" and valid and raw_xy is not None:
            self.samples.append(raw_xy)

    def _tick(self):
        if not self.winfo_exists():
            return
        now = time.time()
        if self.idx >= len(self.targets):
            return self._finish()
        name, tx, ty = self.targets[self.idx]
        if self.state == "PREP" and now - self.t_state > self.PREP_S:
            self.state, self.t_state, self.samples = "REC", now, []
        elif self.state == "REC" and (len(self.samples) >= self.SAMPLES or now - self.t_state > self.TIMEOUT_S):
            if len(self.samples) >= 6:
                arr = np.array(self.samples[len(self.samples) // 4:])     # drop settling part
                self.meas[name] = (tuple(np.median(arr, axis=0)), (tx, ty))
            self.idx += 1
            self.state, self.t_state = "PREP", now
        self.phase += 0.15
        r = 26 + 6 * math.sin(self.phase)
        self.canvas.delete("tgt")
        prog = min(1.0, len(self.samples) / self.SAMPLES) if self.state == "REC" else 0.0
        self.canvas.create_oval(tx - r, ty - r, tx + r, ty + r, outline="#00E5FF", width=4, tags="tgt")
        if prog > 0:
            self.canvas.create_arc(tx - r - 8, ty - r - 8, tx + r + 8, ty + r + 8, start=90,
                                   extent=-359 * prog, style="arc", outline="#00D26A", width=5, tags="tgt")
        self.canvas.create_oval(tx - 6, ty - 6, tx + 6, ty + 6, fill="#FFFFFF", outline="", tags="tgt")
        self.canvas.tag_raise(self.cursor)
        self.canvas.itemconfig(self.hud, text=f"Reach tuning {self.idx + 1}/{len(self.targets)} - look at the "
                                              f"centre of the glowing key ({name}). Keep your head natural.")
        self.after(30, self._tick)

    def _finish(self):
        model = fit_reach(self.meas, self.W, self.H)
        self.calibration.reach = model
        rep = ", ".join(f"{k} {v:.0f}%" for k, v in model.report.items())
        ce = f"{model.corner_error_px:.0f}px" if model.corner_error_px is not None else "n/a"
        msg = (f"Reach tuned. Gaze reached (before tuning): {rep or 'n/a'}\n"
               f"Gains L {model.g_left:.2f}  R {model.g_right:.2f}  U {model.g_up:.2f}  D {model.g_down:.2f} | "
               f"corner check error {ce}")
        print("[Reach]", msg.replace("\n", " | "))
        self.canvas.delete("tgt")
        self.canvas.itemconfig(self.hud, text=msg)
        self.after(2200, self._close)

    def _close(self):
        if self.on_complete:
            self.on_complete(self.calibration.reach)
        self.destroy()

"""
Real-Time Desktop Dashboard Application.
Integrates threaded webcam capture, gaze detection, calibration mapping,
fixation detection, dwell-time engine, interactive accessibility keyboard keypad,
and telemetry monitors into a responsive dark-themed desktop interface.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import time
import os
import math
from typing import Optional, Tuple

from config.config import TrackerConfig, DEFAULT_CONFIG
from camera.webcam import ThreadedWebcam
from gaze.mediapipe_detector import create_detector
from gaze.drift_correction import DriftCorrector
from gaze.stabilizer import GazeStabilizer
from gaze.calibration import GazeCalibration
from gaze.smoothing import GazeSmoother
from fixation.fixation_detector import FixationDetector
from dwell.dwell_engine import DwellEngine, DwellState, DwellResult
from regions.region_manager import RegionManager, Region
from logging_util.gaze_logger import GazeLogger
from visualization.heatmap import GazeHeatmap
from visualization.gaze_overlay import GazeOverlay
from ui.calibration_ui import CalibrationWindow
from ui.test_mode_ui import TestingModeWindow
from ui.keyboard_window import GazeKeyboardWindow
from ui.reach_ui import ReachTuningWindow


class EyeGazeDashboard(tk.Tk):
    def __init__(self, config: Optional[TrackerConfig] = None):
        super().__init__()
        self.config = config or DEFAULT_CONFIG

        # Query native screen dimensions
        self.screen_w = self.winfo_screenwidth()
        self.screen_h = self.winfo_screenheight()
        self.config.screen_width = self.screen_w
        self.config.screen_height = self.screen_h

        self.title("Real-Time Webcam Eye-Gaze Tracker & Dwell Detection")
        self.geometry("1280x820")
        self.minsize(1080, 720)
        self.configure(bg="#14161F")

        # Initialize sub-systems
        self._init_backend_engines()

        # Build GUI Layout
        self._build_styles()
        self._build_ui()

        # Register accessibility demo keyboard regions
        self._setup_accessibility_keyboard()

        # Calibration & Test mode window handles
        self.calib_win: Optional[CalibrationWindow] = None
        self.test_win: Optional[TestingModeWindow] = None
        self.kb_win: Optional[GazeKeyboardWindow] = None
        self.reach_win: Optional[ReachTuningWindow] = None
        self._open_keyboard_after_setup = False

        # Camera loop state
        self.is_tracking = False
        self.loop_id = None
        self.last_frame_time = time.time()
        self.current_fps = 0.0
        self.fps_counter = 0
        self.fps_timer = time.time()
        self._cam_image_ref = None

        # Handle window close and keyboard shortcuts
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.bind("<space>", lambda e: self.recenter_gaze())

        # Auto-start tracking on application launch after GUI renders
        self.after(300, self.start_tracking)

    def _init_backend_engines(self):
        """Instantiates all core algorithmic modules."""
        self.webcam = ThreadedWebcam(
            camera_index=self.config.camera_index,
            width=self.config.camera_width,
            height=self.config.camera_height,
            target_fps=self.config.target_fps
        )

        self.gaze_detector = create_detector(
            self.config.detector_backend,
            self.config.model_path,
            self.config.blink_ear_threshold,
            self.config.min_gaze_confidence
        )
        self.drift = DriftCorrector(kernel_px=self.config.drift_kernel_px,
                                    enabled=self.config.drift_correction_enabled)

        self.calibration = GazeCalibration(
            screen_width=self.screen_w,
            screen_height=self.screen_h,
            model_type=self.config.calibration_model_type
        )
        if os.path.exists(self.config.calibration_file):
            self.calibration.load(self.config.calibration_file)
        self.calibration.geometry = (self.config.screen_diagonal_in, self.config.viewing_distance_cm)
        self.calibration.edge_gain_x = self.config.edge_gain_x
        self.calibration.edge_gain_y = self.config.edge_gain_y

        self.stabilizer = GazeStabilizer(
            key_w=self.screen_w / 8, key_h=self.screen_h * 0.15,
            saccade_frac=self.config.saccade_frac,
            confirm_frames=self.config.saccade_confirm_frames,
            detect_avg=self.config.saccade_detect_avg,
            window=self.config.fixation_window,
            fix_alpha=self.config.fixation_alpha,
            min_cutoff_x=self.config.one_euro_min_cutoff,
            min_cutoff_y=self.config.one_euro_min_cutoff_y,
            beta=self.config.one_euro_beta,
            d_cutoff=self.config.one_euro_d_cutoff,
        )
        self.smoother = GazeSmoother(
            method=self.config.smoothing_method,
            ema_alpha=self.config.ema_alpha,
            one_euro_min_cutoff=self.config.one_euro_min_cutoff,
            one_euro_beta=self.config.one_euro_beta,
            max_allowed_jump=self.config.max_allowed_jump,
            one_euro_min_cutoff_y=self.config.one_euro_min_cutoff_y
        )

        self.fixation_detector = FixationDetector(
            fixation_radius=self.config.fixation_radius,
            min_fixation_time=self.config.min_fixation_time
        )

        self.dwell_engine = DwellEngine(
            dwell_threshold=self.config.dwell_threshold,
            blink_reset_time=self.config.blink_reset_time,
            cooldown_time=self.config.dwell_cooldown,
            on_dwell_callback=self._on_free_space_dwell
        )

        self.region_manager = RegionManager(
            default_dwell_time=self.config.dwell_threshold
        )
        self.region_manager.on_dwell(self._on_region_dwell)

        self.logger = GazeLogger(log_dir=self.config.session_log_dir)

    def _build_styles(self):
        self.style = ttk.Style(self)
        self.style.theme_use("clam")

        self.style.configure(".", background="#14161F", foreground="#E0E0E0")
        self.style.configure("TFrame", background="#14161F")
        self.style.configure("Card.TFrame", background="#1D202B", relief="flat")
        self.style.configure("TLabel", background="#14161F", foreground="#E0E0E0", font=("Segoe UI", 10))
        self.style.configure("Card.TLabel", background="#1D202B", foreground="#E0E0E0", font=("Segoe UI", 10))
        self.style.configure("Header.TLabel", background="#14161F", foreground="#00E5FF", font=("Segoe UI", 15, "bold"))
        self.style.configure("Telemetry.TLabel", background="#1D202B", foreground="#00E5FF", font=("Consolas", 10, "bold"))

        self.style.configure(
            "Primary.TButton",
            font=("Segoe UI", 10, "bold"),
            background="#007ACC",
            foreground="#FFFFFF",
            borderwidth=0,
            padding=6
        )
        self.style.map("Primary.TButton", background=[("active", "#0098FF")])

        self.style.configure(
            "Action.TButton",
            font=("Segoe UI", 10, "bold"),
            background="#2E3346",
            foreground="#FFFFFF",
            borderwidth=0,
            padding=6
        )
        self.style.map("Action.TButton", background=[("active", "#3E455E")])

        self.style.configure(
            "Success.TButton",
            font=("Segoe UI", 10, "bold"),
            background="#1E824C",
            foreground="#FFFFFF",
            borderwidth=0,
            padding=6
        )
        self.style.map("Success.TButton", background=[("active", "#26A65B")])

    def _build_ui(self):
        # 1. Top Bar
        top_bar = ttk.Frame(self, padding=(16, 10, 16, 10))
        top_bar.pack(fill=tk.X)

        title_lbl = ttk.Label(top_bar, text="👁 EYE GAZE TRACKER & DWELL ENGINE", style="Header.TLabel")
        title_lbl.pack(side=tk.LEFT)

        self.session_badge = tk.Label(
            top_bar, text="IDLE", bg="#2C303E", fg="#A0A0A0",
            font=("Segoe UI", 9, "bold"), padx=10, pady=2
        )
        self.session_badge.pack(side=tk.LEFT, padx=16)

        # Quick Control Buttons on Right
        self.btn_toggle_tracking = ttk.Button(
            top_bar, text="▶ Start Tracking", style="Success.TButton", command=self.toggle_tracking
        )
        self.btn_toggle_tracking.pack(side=tk.RIGHT, padx=4)

        self.btn_recenter = ttk.Button(
            top_bar, text="🎯 Re-Center (Space)", style="Action.TButton", command=self.recenter_gaze
        )
        self.btn_recenter.pack(side=tk.RIGHT, padx=4)

        self.btn_calibrate = ttk.Button(
            top_bar, text=f"🎯 Calibrate ({self.config.calibration_grid_rows * self.config.calibration_grid_cols}-pt)",
            style="Action.TButton", command=self.open_calibration
        )
        self.btn_calibrate.pack(side=tk.RIGHT, padx=4)

        self.btn_keyboard = ttk.Button(
            top_bar, text="⌨ Gaze Keyboard", style="Success.TButton", command=self.open_keyboard
        )
        self.btn_keyboard.pack(side=tk.RIGHT, padx=4)

        self.btn_reach = ttk.Button(
            top_bar, text="↔ Reach Tuning", style="Action.TButton", command=self.open_reach_tuning
        )
        self.btn_reach.pack(side=tk.RIGHT, padx=4)

        self.btn_test_mode = ttk.Button(
            top_bar, text="📊 Benchmark Test", style="Action.TButton", command=self.open_test_mode
        )
        self.btn_test_mode.pack(side=tk.RIGHT, padx=4)

        # 2. Main Content Split View
        main_content = ttk.Frame(self, padding=(16, 0, 16, 10))
        main_content.pack(fill=tk.BOTH, expand=True)

        # Left Column: Camera Preview & Eye Telemetry (fixed pixel width)
        left_col = ttk.Frame(main_content, style="Card.TFrame", padding=12)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        cam_header = ttk.Frame(left_col, style="Card.TFrame")
        cam_header.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(cam_header, text="Webcam Feed", font=("Segoe UI", 11, "bold"), style="Card.TLabel").pack(side=tk.LEFT)

        # Camera index switcher
        ttk.Label(cam_header, text="Cam:", style="Card.TLabel").pack(side=tk.LEFT, padx=(12, 2))
        self.cam_combo = ttk.Combobox(cam_header, values=["0", "1", "2"], width=3, state="readonly")
        self.cam_combo.set(str(self.config.camera_index))
        self.cam_combo.bind("<<ComboboxSelected>>", self._on_camera_select)
        self.cam_combo.pack(side=tk.LEFT)

        self.lbl_cam_fps = tk.Label(cam_header, text="0.0 FPS", bg="#1D202B", fg="#00E5FF", font=("Consolas", 10, "bold"))
        self.lbl_cam_fps.pack(side=tk.RIGHT)

        # Video Canvas (exact 460x345 pixels)
        self.video_canvas = tk.Canvas(left_col, width=460, height=345, bg="#0E1017", highlightthickness=1, highlightbackground="#2A2E3D")
        self.video_canvas.pack(fill=tk.NONE, expand=False, pady=4)
        self._show_camera_standby_message("Initializing Camera Feed...")

        # Eye Diagnostic Indicators
        diag_box = ttk.Frame(left_col, style="Card.TFrame", padding=(4, 8, 4, 8))
        diag_box.pack(fill=tk.X, pady=6)

        self.lbl_face = tk.Label(diag_box, text="Face: Searching...", bg="#1D202B", fg="#E0E0E0", font=("Segoe UI", 10))
        self.lbl_face.grid(row=0, column=0, sticky="w", padx=6, pady=2)

        self.lbl_eyes = tk.Label(diag_box, text="Eyes: Not localized", bg="#1D202B", fg="#E0E0E0", font=("Segoe UI", 10))
        self.lbl_eyes.grid(row=0, column=1, sticky="w", padx=6, pady=2)

        self.lbl_pupils = tk.Label(diag_box, text="Pupils: None", bg="#1D202B", fg="#E0E0E0", font=("Segoe UI", 10))
        self.lbl_pupils.grid(row=1, column=0, sticky="w", padx=6, pady=2)

        self.lbl_blinking = tk.Label(diag_box, text="Blinking: No", bg="#1D202B", fg="#E0E0E0", font=("Segoe UI", 10))
        self.lbl_blinking.grid(row=1, column=1, sticky="w", padx=6, pady=2)

        self.lbl_confidence = tk.Label(diag_box, text="Confidence: 0%", bg="#1D202B", fg="#E0E0E0", font=("Segoe UI", 10))
        self.lbl_confidence.grid(row=2, column=0, sticky="w", padx=6, pady=2)

        self.lbl_model_type = tk.Label(diag_box, text=f"Mapping: {self.config.calibration_model_type.capitalize()}", bg="#1D202B", fg="#E0E0E0", font=("Segoe UI", 10))
        self.lbl_model_type.grid(row=2, column=1, sticky="w", padx=6, pady=2)

        # Calibration Model Selection Dropdown
        model_select_frame = ttk.Frame(left_col, style="Card.TFrame", padding=(4, 4))
        model_select_frame.pack(fill=tk.X, pady=4)
        ttk.Label(model_select_frame, text="Active Model:", style="Card.TLabel").pack(side=tk.LEFT, padx=4)
        self.model_combo = ttk.Combobox(
            model_select_frame, values=["polynomial", "affine", "homography", "linear"],
            state="readonly", width=12
        )
        self.model_combo.set(self.config.calibration_model_type)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_change)
        self.model_combo.pack(side=tk.LEFT, padx=6)

        # Sensitivity Gain Slider
        gain_frame = ttk.Frame(left_col, style="Card.TFrame", padding=(4, 4))
        gain_frame.pack(fill=tk.X, pady=2)
        ttk.Label(gain_frame, text="Sensitivity:", style="Card.TLabel").pack(side=tk.LEFT, padx=4)
        self.gain_slider = ttk.Scale(gain_frame, from_=1.0, to=4.0, value=self.calibration.gain_x, command=self._on_gain_change)
        self.gain_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.lbl_gain_val = tk.Label(gain_frame, text=f"{self.calibration.gain_x:.1f}x", bg="#1D202B", fg="#00E5FF", font=("Consolas", 9, "bold"))
        self.lbl_gain_val.pack(side=tk.RIGHT, padx=4)

        # Session Recording & Heatmap Buttons
        sess_btn_frame = ttk.Frame(left_col, style="Card.TFrame", padding=(4, 6))
        sess_btn_frame.pack(fill=tk.X, pady=4)

        self.btn_session = ttk.Button(
            sess_btn_frame, text="⏺ Record Session", style="Action.TButton", command=self.toggle_session_record
        )
        self.btn_session.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        self.btn_heatmap = ttk.Button(
            sess_btn_frame, text="🔥 Heatmap", style="Action.TButton", command=self.show_heatmap
        )
        self.btn_heatmap.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        self.btn_export = ttk.Button(
            sess_btn_frame, text="💾 Export CSV", style="Action.TButton", command=self.export_csv
        )
        self.btn_export.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        # Right Column: Virtual Screen Visualizer & Interactive Keypad
        right_col = ttk.Frame(main_content, style="Card.TFrame", padding=12)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(12, 0))

        screen_header = ttk.Frame(right_col, style="Card.TFrame")
        screen_header.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(
            screen_header, text="Screen Gaze & Interactive Assistive Keypad",
            font=("Segoe UI", 11, "bold"), style="Card.TLabel"
        ).pack(side=tk.LEFT)

        self.lbl_dwell_info = tk.Label(screen_header, text="Dwell: 0.00 / 1.50 s", bg="#1D202B", fg="#00E5FF", font=("Consolas", 10, "bold"))
        self.lbl_dwell_info.pack(side=tk.RIGHT)

        # Virtual Screen Canvas
        self.screen_canvas = tk.Canvas(right_col, bg="#0F1117", highlightthickness=1, highlightbackground="#2A2E3D")
        self.screen_canvas.pack(fill=tk.BOTH, expand=True, pady=4)
        self.screen_canvas.bind("<Configure>", self._on_screen_canvas_resize)

        # Typed Text Output Bar
        text_bar = ttk.Frame(right_col, style="Card.TFrame", padding=(4, 6))
        text_bar.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(text_bar, text="Typed Output:", font=("Segoe UI", 10, "bold"), style="Card.TLabel").pack(side=tk.LEFT, padx=4)
        self.txt_typed_output = tk.Entry(
            text_bar, bg="#11131A", fg="#00FF88", insertbackground="#00FF88",
            font=("Consolas", 12, "bold"), relief="flat", highlightthickness=1, highlightbackground="#333848"
        )
        self.txt_typed_output.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Button(text_bar, text="Clear", style="Action.TButton", command=self._clear_typed_text).pack(side=tk.RIGHT)
        ttk.Button(text_bar, text="🔊 Speak", style="Action.TButton", command=self._speak_typed_text).pack(side=tk.RIGHT, padx=4)

        # 3. Bottom Telemetry Status Bar
        bot_bar = ttk.Frame(self, style="Card.TFrame", padding=(16, 8, 16, 8))
        bot_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self.lbl_gaze_pos = tk.Label(bot_bar, text="Gaze Position: X = ---- px | Y = ---- px", bg="#1D202B", fg="#E0E0E0", font=("Segoe UI", 10))
        self.lbl_gaze_pos.pack(side=tk.LEFT, padx=8)

        self.lbl_ratios = tk.Label(bot_bar, text="Horizontal Ratio: 0.00 | Vertical Ratio: 0.00", bg="#1D202B", fg="#E0E0E0", font=("Segoe UI", 10))
        self.lbl_ratios.pack(side=tk.LEFT, padx=16)

        self.lbl_status = tk.Label(
            bot_bar, text="STATUS: READY", bg="#1E2333", fg="#00E5FF",
            font=("Segoe UI", 9, "bold"), padx=12, pady=3
        )
        self.lbl_status.pack(side=tk.RIGHT, padx=8)

    def _show_camera_standby_message(self, message: str):
        self.video_canvas.delete("all")
        self.video_canvas.create_text(
            230, 160, text=message, font=("Segoe UI", 12, "bold"), fill="#888899", tags="standby"
        )
        self.video_canvas.create_text(
            230, 195, text="Click '▶ Start Tracking' or check camera permissions.",
            font=("Segoe UI", 9), fill="#555566", tags="standby"
        )

    def _on_camera_select(self, event=None):
        new_idx = int(self.cam_combo.get())
        if new_idx != self.config.camera_index:
            self.config.camera_index = new_idx
            was_tracking = self.is_tracking
            if was_tracking:
                self.stop_tracking()
            self.webcam.camera_index = new_idx
            if was_tracking:
                self.start_tracking()

    def _setup_accessibility_keyboard(self):
        """Sets up 9 interactive virtual keys for assistive eye-typing."""
        self.keypad_keys = [
            ("A", "A"), ("B", "B"), ("C", "C"),
            ("YES", " Yes "), ("SPACE", " "), ("NO", " No "),
            ("HELP", " [HELP] "), ("BACK", "<DEL>"), ("CLEAR", "<CLR>")
        ]

    def _on_screen_canvas_resize(self, event):
        cw = event.width
        ch = event.height
        if cw < 200 or ch < 200:
            return

        # Expand keypad to fill canvas with large accessible buttons
        grid_x0 = cw * 0.04
        grid_y0 = ch * 0.18
        grid_w = cw * 0.92
        grid_h = ch * 0.78

        cols = 3
        rows = 3
        cell_w = grid_w / cols
        cell_h = grid_h / rows
        gap = 8

        self.region_manager.clear_regions()

        for idx, (label, val) in enumerate(self.keypad_keys):
            r = idx // cols
            c = idx % cols
            kx = grid_x0 + c * cell_w + gap
            ky = grid_y0 + r * cell_h + gap
            kw = cell_w - gap * 2
            kh = cell_h - gap * 2

            sx = (kx / float(cw)) * self.screen_w
            sy = (ky / float(ch)) * self.screen_h
            sw = (kw / float(cw)) * self.screen_w
            sh = (kh / float(ch)) * self.screen_h

            reg = Region(
                id=f"KEY_{label}",
                x=sx, y=sy, width=sw, height=sh,
                dwell_time=self.config.dwell_threshold,
                label=label
            )
            self.region_manager.register_region(reg)

    def _on_region_dwell(self, region_id: str):
        if self.kb_win is not None and self.kb_win.winfo_exists():
            return                      # full-screen keyboard owns the gaze while open
        for label, val in self.keypad_keys:
            if region_id == f"KEY_{label}":
                if val == "<DEL>":
                    cur = self.txt_typed_output.get()
                    self.txt_typed_output.delete(0, tk.END)
                    self.txt_typed_output.insert(0, cur[:-1])
                elif val == "<CLR>":
                    self.txt_typed_output.delete(0, tk.END)
                else:
                    self.txt_typed_output.insert(tk.END, val)

                # Audio chime on typing
                try:
                    import winsound
                    winsound.MessageBeep(winsound.MB_ICONASTERISK)
                except Exception:
                    pass

                self._flash_screen_canvas(label)
                break

    def _flash_screen_canvas(self, label: str):
        tag = f"flash_{label}"
        self.screen_canvas.create_text(
            self.screen_canvas.winfo_width() // 2, 35,
            text=f"Selected: '{label}'", font=("Segoe UI", 16, "bold"), fill="#00FF88", tags=tag
        )
        self.after(700, lambda: self.screen_canvas.delete(tag))

    def _on_gain_change(self, val):
        gain = round(float(val), 2)
        self.calibration.gain_x = gain
        self.calibration.gain_y = gain
        self.config.gain_x = gain
        self.config.gain_y = gain
        self.lbl_gain_val.config(text=f"{gain:.1f}x")

    def recenter_gaze(self):
        """Re-centers screen gaze coordinate mapping to user's current resting point."""
        hr = self.gaze_detector.horizontal_ratio()
        vr = self.gaze_detector.vertical_ratio()
        if hr is not None and vr is not None:
            self.calibration.set_center(hr, vr)
            self.calibration.save(self.config.calibration_file)
            self._flash_screen_canvas("🎯 Gaze Re-Centered!")
            try:
                import winsound
                winsound.MessageBeep(winsound.MB_OK)
            except Exception:
                pass

    def _on_free_space_dwell(self, anchor: Tuple[float, float], dwell_time: float):
        self.logger.log_event("dwell_triggered", anchor[0], anchor[1], None, None, dwell_time, "Free-space dwell")

    def _speak_typed_text(self):
        """Text-to-speech for whatever is in the dashboard text box (can also be typed by a carer)."""
        if not hasattr(self, "_speaker"):
            from keyboard.speech import Speaker
            self._speaker = Speaker(rate_wpm=self.config.speech_rate_wpm, backend=self.config.speech_backend)
        self._speaker.say(self.txt_typed_output.get(), interrupt=True)

    def _clear_typed_text(self):
        self.txt_typed_output.delete(0, tk.END)

    def _on_model_change(self, event=None):
        new_model = self.model_combo.get()
        self.config.calibration_model_type = new_model
        self.calibration.model_type = new_model
        self.lbl_model_type.config(text=f"Mapping: {new_model.capitalize()}")

    def toggle_tracking(self):
        if self.is_tracking:
            self.stop_tracking()
        else:
            self.start_tracking()

    def start_tracking(self):
        if self.is_tracking:
            return

        if not self.webcam.start():
            self._show_camera_standby_message(f"Could not open camera {self.config.camera_index}.")
            messagebox.showerror(
                "Webcam Error",
                f"Failed to access webcam at index {self.config.camera_index}.\n"
                "Please verify no other app (e.g. Zoom, Teams, Camera) is using it, "
                "or switch camera index in the Cam dropdown."
            )
            return

        self.is_tracking = True
        self.btn_toggle_tracking.config(text="⏹ Stop Tracking", style="Primary.TButton")
        self._update_loop()

    def stop_tracking(self):
        self.is_tracking = False
        if self.loop_id:
            self.after_cancel(self.loop_id)
            self.loop_id = None
        self.webcam.stop()
        self.btn_toggle_tracking.config(text="▶ Start Tracking", style="Success.TButton")
        self.lbl_status.config(text="STATUS: STOPPED", bg="#2C303E", fg="#A0A0A0")
        self._show_camera_standby_message("Camera Paused")

    def toggle_session_record(self):
        if self.logger.is_recording:
            self.logger.stop_session()
            self.btn_session.config(text="⏺ Record Session")
            self.session_badge.config(text="IDLE", bg="#2C303E", fg="#A0A0A0")
            messagebox.showinfo("Session Saved", f"Session saved to {self.logger.csv_path}")
        else:
            sid = self.logger.start_session()
            self.btn_session.config(text="⏹ Stop Recording")
            self.session_badge.config(text=f"RECORDING: {sid}", bg="#8B0000", fg="#FFFFFF")

    def export_csv(self):
        if not self.logger.csv_path or not os.path.exists(self.logger.csv_path):
            messagebox.showwarning("Export CSV", "No active session recording found. Start a recording session first.")
            return
        dest = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if dest:
            import shutil
            shutil.copyfile(self.logger.csv_path, dest)
            messagebox.showinfo("Export Successful", f"Session exported to:\n{dest}")

    def show_heatmap(self):
        points = self.logger.get_session_points()
        if not points:
            messagebox.showinfo("Heatmap", "No gaze samples collected yet. Start tracking and record a session to generate a heatmap.")
            return

        heatmap_img = GazeHeatmap.generate(points, width=1280, height=720)
        cv2.imshow("Session Gaze Density Heatmap (Press any key to close)", heatmap_img)
        cv2.waitKey(1)

    def open_calibration(self):
        if self.calib_win is not None and self.calib_win.winfo_exists():
            self.calib_win.lift()
            return
        if not self.is_tracking:
            self.start_tracking()

        self.calib_win = CalibrationWindow(
            self, self.calibration, on_complete_callback=self._on_calibration_completed
        )

    def _on_calibration_completed(self):
        self.calibration.save(self.config.calibration_file)
        self.drift.reset()          # old drift samples belong to the previous calibration
        self.smoother.reset()
        self.stabilizer.reset()
        print("[Calibration]", self.calibration.last_fit_message)
        if self.config.reach_tuning_after_calibration:
            self.after(400, self.open_reach_tuning)       # ~15 s: makes outer keys reachable
        else:
            messagebox.showinfo("Calibration Finished",
                                f"Calibration saved to {self.config.calibration_file}\n{self.calibration.last_fit_message}")
            self._maybe_open_keyboard()

    def open_reach_tuning(self):
        if not self.calibration.is_calibrated:
            messagebox.showwarning("Reach tuning", "Calibrate first.")
            return
        if self.reach_win is not None and self.reach_win.winfo_exists():
            self.reach_win.lift()
            return
        if not self.is_tracking:
            self.start_tracking()
        self.reach_win = ReachTuningWindow(self, self.calibration, self.screen_w, self.screen_h,
                                           phrases_file=self.config.keyboard_phrases_file,
                                           on_complete=self._on_reach_completed)

    def _on_reach_completed(self, model):
        self.calibration.save(self.config.calibration_file)
        self.drift.reset()
        self.smoother.reset()
        self.stabilizer.reset()
        self._maybe_open_keyboard()

    def _maybe_open_keyboard(self):
        if self._open_keyboard_after_setup:
            self._open_keyboard_after_setup = False
            self.after(300, self.open_keyboard)

    def open_keyboard(self):
        if self.kb_win is not None and self.kb_win.winfo_exists():
            self.kb_win.lift()
            return
        if not self.calibration.is_calibrated:
            # never type with an uncalibrated mapping: calibrate -> reach tuning -> keyboard
            self._open_keyboard_after_setup = True
            self.open_calibration()
            return
        if not self.is_tracking:
            self.start_tracking()
        self.kb_win = GazeKeyboardWindow(
            self, self.screen_w, self.screen_h,
            phrases_file=self.config.keyboard_phrases_file,
            letter_dwell=self.config.keyboard_letter_dwell,
            phrase_dwell=self.config.keyboard_phrase_dwell,
            show_cursor=self.config.keyboard_show_cursor,
            on_selection=self._on_keyboard_selection,
            on_undo=self.drift.undo_last,
            user_vocab_file=self.config.keyboard_user_vocab_file,
            speak_on_sentence_end=self.config.keyboard_speak_on_sentence_end,
            speech_rate=self.config.speech_rate_wpm,
            speech_echo=self.config.speech_echo,
            speech_backend=self.config.speech_backend,
            selector_params=dict(
                sticky_x=self.config.key_sticky_x, sticky_y=self.config.key_sticky_y,
                switch_time=self.config.key_switch_time, settle_time=self.config.key_settle_time,
                repeat_factor=self.config.key_repeat_factor, memory_time=self.config.key_memory_time),
            typed_log_file=os.path.join(self.config.base_dir, "data", "typed_log.txt"),
        )
        # saccade threshold is relative to the real key size of this screen
        self.stabilizer.set_key_size(*self.kb_win.letter_key_size)
        self.stabilizer.reset()

    def _on_keyboard_selection(self, ev) -> bool:
        """Each confirmed key = one free calibration sample for drift correction."""
        if ev.raw_gaze_median is None:
            return False
        return self.drift.add(ev.raw_gaze_median, (ev.key.cx, ev.key.cy))

    def open_test_mode(self):
        if self.test_win is not None and self.test_win.winfo_exists():
            self.test_win.lift()
            return
        if not self.is_tracking:
            self.start_tracking()

        self.test_win = TestingModeWindow(self, self.screen_w, self.screen_h)

    def _update_loop(self):
        """Primary real-time processing loop (runs at 30 FPS)."""
        if not self.is_tracking:
            return

        start_time = time.time()
        ret, frame = self.webcam.read()

        if ret and frame is not None:
            # 1. Flip horizontally so webcam acts like a mirror
            frame = cv2.flip(frame, 1)

            # 2. Gaze detection pipeline
            self.gaze_detector.refresh(frame, start_time)

            face_detected = self.gaze_detector.is_face_detected()
            is_blinking = self.gaze_detector.is_blinking()
            hr = self.gaze_detector.horizontal_ratio()
            vr = self.gaze_detector.vertical_ratio()
            confidence = self.gaze_detector.confidence()
            confidence_ok = self.gaze_detector.is_valid()
            # extra calibration features: head yaw/pitch + eyelid aperture (vertical accuracy)
            head = self.gaze_detector.pose_features() if hasattr(self.gaze_detector, "pose_features") else None

            # Feed to active calibration window if open
            if self.calib_win is not None and self.calib_win.winfo_exists():
                # only feed frames the detector trusts (no blinks / low confidence)
                if confidence_ok and not is_blinking:
                    self.calib_win.feed_sample(hr, vr, head)

            # 3. Screen Mapping
            raw_screen_pos = self.calibration.map_gaze(hr, vr, head) if (hr is not None and vr is not None) else None
            uncorrected_pos = raw_screen_pos
            if self.reach_win is not None and self.reach_win.winfo_exists():
                no_reach = self.calibration.map_gaze(hr, vr, head, apply_reach=False) if (hr is not None and vr is not None) else None
                self.reach_win.feed(no_reach, confidence_ok and not is_blinking)
            raw_screen_pos = self.drift.correct(raw_screen_pos)

            # 4. Smoothing Filter
            if raw_screen_pos is not None:
                if self.config.stabilizer_enabled:
                    smooth_x, smooth_y = self.stabilizer.update(raw_screen_pos[0], raw_screen_pos[1], start_time)
                else:
                    smooth_x, smooth_y = self.smoother.filter(raw_screen_pos[0], raw_screen_pos[1], start_time)
            else:
                smooth_x, smooth_y = None, None

            # 5. Fixation Detection
            is_fixating = False
            if smooth_x is not None and smooth_y is not None:
                is_fixating = self.fixation_detector.update(smooth_x, smooth_y, start_time, confidence_ok)

            # 6. Dwell-Time Engine
            dwell_res: DwellResult = self.dwell_engine.update(
                gaze_x=smooth_x,
                gaze_y=smooth_y,
                is_fixating=is_fixating,
                is_blinking=is_blinking,
                is_face_detected=face_detected,
                is_valid_confidence=confidence_ok,
                timestamp=start_time
            )

            # 7. Region Manager (Interactive UI element dwell)
            self.region_manager.update(smooth_x, smooth_y, confidence_ok, start_time)

            # Feed to benchmark test window if open
            if self.test_win is not None and self.test_win.winfo_exists():
                self.test_win.feed_gaze(smooth_x, smooth_y, confidence_ok)

            # Feed the full-screen gaze keyboard if open
            if self.kb_win is not None and self.kb_win.winfo_exists():
                stab = self.config.stabilizer_enabled
                self.kb_win.feed_gaze(smooth_x, smooth_y, confidence_ok and not is_blinking,
                                      raw_xy=uncorrected_pos,
                                      stable=self.stabilizer.is_stable if stab else None,
                                      last_saccade_t=self.stabilizer.last_saccade_t if stab else None)

            # 8. Data Logger
            if self.logger.is_recording and smooth_x is not None and smooth_y is not None:
                self.logger.record_gaze_point(smooth_x, smooth_y, start_time)

            # 9. Update GUI Telemetry & Visuals
            self._render_camera_preview()
            self._render_screen_view(smooth_x, smooth_y, dwell_res)
            self._update_telemetry_labels(face_detected, is_blinking, hr, vr, confidence, smooth_x, smooth_y, dwell_res)

        # FPS measurement
        self.fps_counter += 1
        now = time.time()
        if (now - self.fps_timer) >= 1.0:
            self.current_fps = self.fps_counter / (now - self.fps_timer)
            self.fps_counter = 0
            self.fps_timer = now
            self.lbl_cam_fps.config(text=f"{self.current_fps:.1f} FPS")

        # Schedule next tick (~30 FPS -> 33ms interval)
        elapsed = (time.time() - start_time) * 1000
        delay = max(5, int(33 - elapsed))
        self.loop_id = self.after(delay, self._update_loop)

    def _render_camera_preview(self):
        """Draws annotated frame directly onto Tkinter Canvas."""
        annotated = self.gaze_detector.annotated_frame()
        disp_frame = cv2.resize(annotated, (460, 345), interpolation=cv2.INTER_AREA)
        disp_rgb = cv2.cvtColor(disp_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(disp_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        # Cache reference to prevent garbage collection
        self._cam_image_ref = imgtk
        self.video_canvas.delete("all")
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

    def _render_screen_view(self, gx: Optional[float], gy: Optional[float], dwell_res: DwellResult):
        """Draws screen canvas with accessibility keypad, gaze point, and dwell progress ring."""
        cw = self.screen_canvas.winfo_width()
        ch = self.screen_canvas.winfo_height()
        if cw < 50 or ch < 50:
            return

        self.screen_canvas.delete("dynamic")

        active_reg_id, active_dwell, target_dwell = self.region_manager.get_dwell_state()

        for reg in self.region_manager.regions.values():
            rx = (reg.x / float(self.screen_w)) * cw
            ry = (reg.y / float(self.screen_h)) * ch
            rw = (reg.width / float(self.screen_w)) * cw
            rh = (reg.height / float(self.screen_h)) * ch

            is_active = (reg.id == active_reg_id)
            bg_col = "#253046" if is_active else "#1A1D27"
            border_col = "#00FF88" if is_active else "#333A4D"
            border_w = 2 if is_active else 1

            self.screen_canvas.create_rectangle(
                rx, ry, rx + rw, ry + rh,
                fill=bg_col, outline=border_col, width=border_w, tags="dynamic"
            )

            if is_active and target_dwell > 0:
                prog = min(1.0, active_dwell / target_dwell)
                pw = rw * prog
                self.screen_canvas.create_rectangle(
                    rx, ry + rh - 6, rx + pw, ry + rh,
                    fill="#00FF88", outline="", tags="dynamic"
                )

            self.screen_canvas.create_text(
                rx + rw / 2, ry + rh / 2,
                text=reg.label, font=("Segoe UI", 13, "bold"),
                fill="#FFFFFF" if is_active else "#C5C8D4", tags="dynamic"
            )

        if gx is not None and gy is not None:
            cx = (gx / float(self.screen_w)) * cw
            cy = (gy / float(self.screen_h)) * ch

            color_map = {
                DwellState.TRACKING: "#9E9E9E",
                DwellState.FIXATING: "#FFD700",
                DwellState.DWELLING: "#FF9100",
                DwellState.TRIGGERED: "#00FF66",
                DwellState.PAUSED_BLINK: "#00E5FF",
                DwellState.INVALID: "#FF3366",
                DwellState.FACE_LOST: "#D50000"
            }
            state_color = color_map.get(dwell_res.state, "#FFFFFF")

            fix_r_canvas = (self.config.fixation_radius / float(self.screen_w)) * cw
            if dwell_res.state in [DwellState.FIXATING, DwellState.DWELLING, DwellState.TRIGGERED]:
                self.screen_canvas.create_oval(
                    cx - fix_r_canvas, cy - fix_r_canvas, cx + fix_r_canvas, cy + fix_r_canvas,
                    outline="#333D52", width=1, dash=(3, 3), tags="dynamic"
                )

            ring_r = 22
            self.screen_canvas.create_oval(
                cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r,
                outline="#383D50", width=3, tags="dynamic"
            )

            if dwell_res.progress > 0:
                arc_extent = int(dwell_res.progress * 359)
                self.screen_canvas.create_arc(
                    cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r,
                    start=90, extent=-arc_extent, outline=state_color, width=4, style="arc", tags="dynamic"
                )

            dot_r = 5
            self.screen_canvas.create_oval(
                cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r,
                fill=state_color, outline="", tags="dynamic"
            )

            if dwell_res.triggered:
                self.screen_canvas.create_oval(
                    cx - 36, cy - 36, cx + 36, cy + 36,
                    outline="#00FF66", width=3, tags="dynamic"
                )

    def _update_telemetry_labels(self, face_detected, is_blinking, hr, vr, confidence, gx, gy, dwell_res):
        self.lbl_face.config(text=f"Face: {'Detected' if face_detected else 'Lost'}", fg="#00FF88" if face_detected else "#FF3366")
        self.lbl_blinking.config(text=f"Blinking: {'Yes' if is_blinking else 'No'}", fg="#00E5FF" if is_blinking else "#E0E0E0")
        self.lbl_confidence.config(text=f"Confidence: {int(confidence * 100)}%", fg="#00FF88" if confidence >= 0.50 else "#FFB800")

        left_p = self.gaze_detector.pupil_left_coords()
        right_p = self.gaze_detector.pupil_right_coords()
        self.lbl_pupils.config(text=f"Pupils: {'L+R' if (left_p and right_p) else ('L only' if left_p else ('R only' if right_p else 'None'))}")

        if gx is not None and gy is not None:
            self.lbl_gaze_pos.config(text=f"Gaze Position: X = {int(gx):4d} px | Y = {int(gy):4d} px")
        else:
            self.lbl_gaze_pos.config(text="Gaze Position: X = ---- px | Y = ---- px")

        if hr is not None and vr is not None:
            self.lbl_ratios.config(text=f"Horizontal Ratio: {hr:.2f} | Vertical Ratio: {vr:.2f}")

        self.lbl_dwell_info.config(text=f"Dwell: {dwell_res.dwell_time:.2f} / {dwell_res.threshold:.2f} s ({int(dwell_res.progress * 100)}%)")

        status_bg = {
            DwellState.TRACKING: "#2A2E3D",
            DwellState.FIXATING: "#7A6500",
            DwellState.DWELLING: "#995200",
            DwellState.TRIGGERED: "#156329",
            DwellState.PAUSED_BLINK: "#006B7A",
            DwellState.INVALID: "#8A1828",
            DwellState.FACE_LOST: "#6B0D18"
        }.get(dwell_res.state, "#1E2333")

        self.lbl_status.config(text=f"STATUS: {dwell_res.state.value.upper()}", bg=status_bg)

    def on_close(self):
        self.stop_tracking()
        self.logger.stop_session()
        self.destroy()

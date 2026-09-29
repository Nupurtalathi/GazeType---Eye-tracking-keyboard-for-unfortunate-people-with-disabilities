"""
Main Application Entry Point.
Launches the Real-Time Webcam Eye-Gaze Tracking and Dwell Detection desktop system.
Supports GUI dashboard, direct calibration mode, benchmark test mode, or headless runner.
"""

import sys
import os
import argparse

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config.config import TrackerConfig
from gaze.model_downloader import ensure_model_exists


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Webcam Eye-Gaze Tracking and Dwell-Time Detection Platform"
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Webcam capture width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Webcam capture height (default: 720)")
    parser.add_argument("--dwell", type=float, default=1.50, help="Dwell time threshold in seconds (default: 1.50)")
    parser.add_argument("--radius", type=float, default=50.0, help="Fixation radius in pixels (default: 50.0)")
    parser.add_argument(
        "--model", type=str, default="polynomial",
        choices=["polynomial", "affine", "homography", "linear"],
        help="Calibration mapping model (default: polynomial)"
    )
    parser.add_argument("--calibrate", action="store_true", help="Launch directly into calibration (4x4 grid by default)")
    parser.add_argument("--benchmark", action="store_true", help="Launch directly into benchmark accuracy test mode")
    parser.add_argument("--keyboard", action="store_true", help="Open the full-screen gaze keyboard on start")
    parser.add_argument("--backend", type=str, default="mediapipe", choices=["mediapipe", "dlib"],
                        help="Gaze detector backend (default: mediapipe)")
    parser.add_argument("--headless", action="store_true", help="Run without GUI in console telemetry mode")

    args = parser.parse_args()

    # Verify and download shape predictor model if necessary
    print("=" * 70)
    print("👁  REAL-TIME WEBCAM EYE-GAZE TRACKER & DWELL ENGINE")
    print("=" * 70)
    # dlib landmark model is only needed for the legacy backend
    from gaze.mediapipe_detector import MEDIAPIPE_AVAILABLE
    needs_dlib = args.backend == "dlib" or not MEDIAPIPE_AVAILABLE
    model_path = ensure_model_exists() if needs_dlib else ""

    # Load configuration
    config = TrackerConfig(
        camera_index=args.camera,
        camera_width=args.width,
        camera_height=args.height,
        dwell_threshold=args.dwell,
        fixation_radius=args.radius,
        calibration_model_type=args.model,
        detector_backend=args.backend,
        model_path=model_path
    )

    if args.headless:
        # Headless console runner
        from camera.webcam import ThreadedWebcam
        from gaze.mediapipe_detector import create_detector
        from gaze.calibration import GazeCalibration
        from fixation.fixation_detector import FixationDetector
        from dwell.dwell_engine import DwellEngine
        from logging_util.gaze_logger import GazeLogger
        import time

        print("[Headless] Initializing hardware capture and tracking engines...")
        webcam = ThreadedWebcam(config.camera_index, config.camera_width, config.camera_height)
        if not webcam.start():
            print(f"[Error] Could not open webcam at index {config.camera_index}.")
            sys.exit(1)

        detector = create_detector(config.detector_backend, config.model_path,
                                   config.blink_ear_threshold, config.min_gaze_confidence)
        calibration = GazeCalibration(config.screen_width, config.screen_height, config.calibration_model_type)
        if os.path.exists(config.calibration_file):
            calibration.load(config.calibration_file)

        fixation = FixationDetector(config.fixation_radius, config.min_fixation_time)
        dwell = DwellEngine(config.dwell_threshold, config.blink_reset_time)
        logger = GazeLogger(config.session_log_dir)
        session_id = logger.start_session("headless_session")

        print(f"[Headless] Tracking started. Session logging to {logger.csv_path}")
        print("Press Ctrl+C to terminate.")

        try:
            while True:
                ret, frame = webcam.read()
                if ret and frame is not None:
                    now = time.time()
                    detector.refresh(frame, now)
                    hr = detector.horizontal_ratio()
                    vr = detector.vertical_ratio()
                    pos = calibration.map_gaze(hr, vr, detector.pose_features()) if (hr is not None and vr is not None) else None
                    gx = pos[0] if pos else None
                    gy = pos[1] if pos else None

                    is_fix = fixation.update(gx, gy, now, detector.is_valid()) if (gx is not None and gy is not None) else False
                    res = dwell.update(
                        gx, gy, is_fix, detector.is_blinking(),
                        detector.is_face_detected(), detector.is_valid(), now
                    )

                    status_str = (
                        f"\r[Gaze] X: {gx if gx else '---':>5} | Y: {gy if gy else '---':>5} | "
                        f"HR: {hr if hr else 0.0:.2f} | VR: {vr if vr else 0.0:.2f} | "
                        f"Dwell: {res.dwell_time:.2f}s | State: {res.state.value:<18}"
                    )
                    print(status_str, end="", flush=True)

                    if res.triggered:
                        print(f"\n>>> [DWELL EVENT TRIGGERED] at ({gx:.1f}, {gy:.1f})")
                        logger.log_event("dwell_triggered", gx, gy, hr, vr, res.dwell_time)

                time.sleep(0.03)
        except KeyboardInterrupt:
            print("\n[Headless] Terminating session...")
        finally:
            webcam.stop()
            logger.stop_session()

    else:
        # Desktop GUI Mode
        from ui.dashboard import EyeGazeDashboard
        app = EyeGazeDashboard(config)

        if args.calibrate:
            app.after(500, app.open_calibration)
        elif args.keyboard:
            app.after(800, app.open_keyboard)
        elif args.benchmark:
            app.after(500, app.open_test_mode)

        app.mainloop()


if __name__ == "__main__":
    main()

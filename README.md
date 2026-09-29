# Real-Time Webcam Eye-Gaze Tracking and Gaze Keyboard
GazeType is a low-cost, webcam-based eye-gaze communication system for people with severe speech and motor impairments.
It converts eye movements into text using a virtual keyboard and word prediction, then transforms messages into speech for faster, more independent communication.

# Real-Time Webcam Eye-Gaze Tracking, Dwell Detection & Gaze Keyboard Demo
https://github.com/user-attachments/assets/cc3c787e-92a9-42b2-85d6-44e7e979eed9

### Gaze keyboard
```
python main.py --keyboard          # opens dashboard + full-screen keyboard
```
- Top row: WATER, FOOD, WASHROOM, HELP, PAIN, MEDICINE, YES, NO - spoken aloud (pyttsx3) and typed.
  HELP also plays an alert beep. Edit `config/keyboard_phrases.json` to change them (2-10 phrases).
- Letters A-Z, `.`, `?`, `,`, SPACE, DEL (undo), CLEAR. SPEAK is in the suggestion row (v4).
- PAUSE (top-right) blocks all selections so the user can rest; look at it again to resume.
- Keys cover the whole screen with no gaps: a gaze point at or beyond a screen edge selects the edge key.
- Dwell: 1.0 s letters, 1.2 s phrases/controls (`keyboard_letter_dwell`, `keyboard_phrase_dwell` in config).
- F2 toggles the gaze cursor, ESC closes (operator).

### Recommended workflow per user
1. `python main.py --calibrate` - 16 grid targets + 8 edge targets. Accept only if leave-one-out error < ~100 px.
2. `python main.py --keyboard`.
3. Recalibrate whenever the user or the laptop/camera position changes noticeably.

## Description

A robust, high-performance desktop application that tracks eye gaze in real time via an integrated or external webcam and detects continuous dwell time on screen targets. Designed as an accessibility foundation for eye-gaze typing keyboards and assistive communication devices for individuals with severe motor impairments.

Runs 100% locally on standard Windows hardware without sending camera frames to any external service or cloud API.

---

## 1. System Architecture

```
┌──────────────────────┐
│     Webcam Input     │ ── (OpenCV DirectShow Threaded Capture @ 30 FPS)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Face & Eye Detector │ ── (dlib 68-point Facial Landmark Predictor)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│    Pupil Isolation   │ ── (Bilateral Filter, Adaptive Threshold, Iris Moments)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Gaze Estimation    │ ── (Continuous Horizontal/Vertical Ratios, EAR Blink Detector)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Calibration Mapping  │ ── (Linear / 2nd-Order Polynomial / Affine / Homography)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Smoothing & Filtering│ ── (1€ One Euro Low-Latency Filter + Saccade Reset Gate)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Fixation Detector   │ ── (Spatial Dispersion Clustering within Tolerance Radius)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Dwell-Time Engine   │ ── (Continuous Timer, Blink-Pause Grace Period, State Machine)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Region Manager & API │ ── (Free-Space Dwell & Rectangular Accessibility Keyboard UI)
└──────┬─────────┬─────┘
       │         │
       ▼         ▼
┌─────────────┐ ┌────────────────────┐
│ GUI Monitor │ │ CSV / JSON Logger  │
└─────────────┘ └────────────────────┘
```

---

## 2. Key Features

- **Continuous Gaze Estimation**: Calculates precise decimal gaze coordinates (e.g. `X: 742 px, Y: 421 px`, `Horizontal Ratio: 0.47`, `Vertical Ratio: 0.52`) instead of coarse discrete buckets.
- **Adaptive Screen-Space Calibration**: 9-point fullscreen target sequence supporting 4 mathematical mapping models:
  - *Polynomial*: 2nd-degree bivariate regression capturing non-linear eye curvature.
  - *Affine*: 2D affine transformation matrix.
  - *Homography*: 8-DOF perspective transformation matrix.
  - *Linear*: Axis-aligned bounding box interpolation.
  - Outlier rejection removes sample noise during calibration.
- **Low-Latency One Euro Smoothing**: Automatically balances jitter suppression when staring steadily against instantaneous response during rapid eye saccades.
- **Blink-Tolerant Dwell Engine**: Normal short blinks (< 0.5s) pause the dwell timer rather than aborting progress; prolonged eye closures or face losses gracefully reset the timer.
- **Circular Dwell Ring Indicator**: Real-time visual progress ring (0% to 100%) circling the gaze reticle with color-coded tracking states (`TRACKING`, `FIXATING`, `DWELLING`, `TRIGGERED`, `PAUSED_BLINK`, `INVALID`).
- **Assistive Keyboard Keypad**: Integrated 3x3 interactive keypad demo (`A`, `B`, `C`, `YES`, `SPACE`, `NO`, `HELP`, `BACK`, `CLEAR`) that types characters into an output field upon gaze dwell.
- **Accuracy Benchmark Mode**: Evaluates pixel error, dwell latency, and success rates against predefined screen targets.
- **Gaze Heatmap & CSV Logging**: Exports session trajectory and generates 2D Gaussian density heatmaps.

---

## 3. Technology Stack & Prerequisites

- **Python**: 3.10+ (tested on Python 3.11.9)
- **OpenCV (`opencv-python`)**: Camera frame acquisition and image morphology.
- **dlib**: Facial detection and 68-landmark geometry.
- **NumPy & SciPy**: Matrix fitting and spatial operations.
- **Pillow (`PIL`)**: Frame conversion for Tkinter rendering.
- **Tkinter**: Native Windows GUI framework (fast, zero DLL conflicts).
- **Pandas & Matplotlib**: Structured event export and heatmap generation.

---

## 4. Installation & Windows Setup

### Step 1: Install Dependencies
Open PowerShell or Command Prompt in this folder:
```powershell
pip install -r requirements.txt
```

### Step 2: Automatic Model Setup
Only needed for the legacy `--backend dlib`. The default MediaPipe backend ships its models inside the pip wheel (no download). For dlib, the built-in downloader fetches `shape_predictor_68_face_landmarks.dat` on first launch:
```powershell
python gaze/model_downloader.py
```
*(The model will be stored in `models/shape_predictor_68_face_landmarks.dat`).*

---

## 5. Usage Guide

### Launching the Desktop Dashboard
```powershell
python main.py
```

### Command Line Options
```powershell
# Launch directly into 16-point (4x4) screen calibration
python main.py --calibrate

# Launch directly into benchmark accuracy test mode
python main.py --benchmark

# Custom camera index and dwell threshold (e.g. 1.2s dwell, 45px radius)
python main.py --camera 0 --dwell 1.20 --radius 45.0 --model polynomial

# Run headless in console without GUI (low resource consumption)
python main.py --headless
```

---

## 6. How Calibration Works

1. Click **🎯 Calibrate (16-pt)** in the top bar.
2. A fullscreen window will display 16 sequential targets (Top-Left $\rightarrow$ Bottom-Right).
3. Look steadily at the pulsing target. The progress circle will fill as valid samples are collected.
4. The system rejects outlier samples using median distance filtering and fits your selected mapping model (Polynomial by default).
5. Calibration profiles are automatically saved to `data/calibration_profile.json` and persist across sessions.

---

## 7. Eye-Gaze Keyboard Developer API

You can easily integrate your own accessible keyboard or assistive interface with the gaze engine:

```python
from regions.region_manager import RegionManager, Region

# 1. Initialize Region Manager
manager = RegionManager(default_dwell_time=1.50)

# 2. Register UI Regions (x, y, width, height in screen coordinates)
def on_key_pressed(key_id):
    print(f"Key Activated by Eye Dwell: {key_id}")

manager.register_region(Region(
    id="KEY_ENTER",
    x=800, y=500,
    width=200, height=100,
    dwell_time=1.20,
    callback=on_key_pressed
))

# 3. Feed gaze coordinates inside your frame loop
triggered_key = manager.update(gaze_x=850, gaze_y=550, is_valid=True, timestamp=time.time())
```

---

## 8. Data Logging & Example CSV Output

When session recording is enabled, meaningful events are stored in `data/sessions/session_<timestamp>.csv`:

```csv
timestamp,gaze_x,gaze_y,horizontal_ratio,vertical_ratio,dwell_time,event,details
12:30:01.231,742.0,421.0,0.470,0.520,0.00,gaze_start,Fixation initiated
12:30:01.731,748.0,423.0,0.480,0.520,0.50,fixating,Fixation stable
12:30:02.731,744.0,419.0,0.470,0.510,1.50,dwell_triggered,Free-space dwell
```

---

## 9. Authors 
Parth Verma, Nupur Talathi, Shroojan Dhok, Anushka Wankhede, Ayush Taske & Vedika Kulkarni
## 9. Running Automated Unit Tests

To run the full unit test suite verifying fixation, dwell engine, blink pausing, and calibration models:
```powershell
python -m unittest discover tests -v
```

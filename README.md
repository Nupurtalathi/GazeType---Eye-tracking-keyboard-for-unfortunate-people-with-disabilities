# Real-Time Webcam Eye-Gaze Tracking and Gaze Keyboard
GazeType is a low-cost, webcam-based eye-gaze communication system for people with severe speech and motor impairments.
It converts eye movements into text using a virtual keyboard and word prediction, then transforms messages into speech for faster, more independent communication.

# Real-Time Webcam Eye-Gaze Tracking, Dwell Detection & Gaze Keyboard (v7)

## v7 - stable continuous typing (refinement, nothing rebuilt)

- **Fixation-locked stabiliser** (`gaze/stabilizer.py`): stronger One Euro filtering (min_cutoff 0.25,
  vertical 0.18, beta 0.006). While you look at a key, the position is held on a running median, so
  micro-jitter cannot move it. A deliberate jump of more than half a key, seen on 2 frames, is treated
  as a move: history is dropped and the cursor snaps to the new key in about 100 ms. The jump threshold
  rises automatically when the tracking is noisier.
- **Movement-aware hysteresis** (`keyboard/selector.py`): while you look at a key it is sticky (leave by
  15% of its width or 25% of its height). Right after a real eye jump the stickiness is switched off and
  switching is fast, so a small intentional move to the next key is enough.
- **Stable-gaze dwell:** dwell counts only during a stable fixation and after a 0.12 s settle. Unstable
  frames pause it instead of resetting it. Returning within 0.6 s after a quick glance away keeps 80%
  of the progress.
- **No accidental repeats:** a key that has just typed is locked. Typing it again needs either a look
  away and back, or holding for 1.8x the dwell (for double letters like EE / LL, shown with an amber
  progress bar). Set `key_repeat_factor = 0` to always require looking away.
- **Vertical accuracy:** calibration now also learns how much this user's eyelids close when looking
  down (per-user, kept only if it lowers the leave-one-out error).
- **Bottom row:** `Y Z . ? SPACE BACKSPACE CLEAR ENTER`, 20% taller than the letter rows. ENTER speaks
  the line, writes it to `data/typed_log.txt` and starts a new line (BACKSPACE undoes).

All settings are in `config/config.py` under "v7".

---

## v6 - whole keyboard reachable without extreme eye angles (14" laptop)

**The physics:** at ~50 cm from a 14" screen the whole keyboard spans only ~34 x 20 degrees.
Each letter key is ~4.4 deg wide and only ~2.6-2.9 deg tall. Webcam iris tracking is
compressed towards the centre at the edges (iris model, eyelids, head turning), so after
calibration the cursor stopped short of the outer keys and users had to roll their eyes to
extreme angles to reach them.

**What changed (letters and layout unchanged):**
- **Reach tuning** (`gaze/reach.py`, `ui/reach_ui.py`), about 15 s, runs automatically after
  calibration (or with the "Reach Tuning" button). The user looks at 9 real key centres. Separate
  left/right/up/down edge gains are fitted, so outer keys are reached at a comfortable angle; the centre
  is untouched. It reports how far the gaze got before tuning ("right 76%") and a corner check error.
- **Sticky keys:** once a key is active, gaze must leave it by 20% of its width or 30% of its height
  before a neighbour takes over. This stops jitter across key borders, especially vertically.
- Vertical axis gets extra smoothing (`one_euro_min_cutoff_y`).
- Camera now 1280x720 by default (twice the eye pixels of 640x480).
- The keyboard never runs uncalibrated: it starts calibration, then reach tuning, then opens.
- Calibration error is also shown in degrees (`screen_diagonal_in`, `viewing_distance_cm`).

**Setup for a 14" laptop:** sit 45-55 cm away, eyes level with the top third of the screen, face
evenly lit (no window behind you). Run `python main.py --keyboard`: calibration (24 dots) runs first,
then reach tuning (9 keys), then typing starts. Redo both if you move the laptop or change seat.

---

## v5 - speech fix: everything typed is spoken

**Why v4 was silent:** the voice ran pyttsx3 in a background thread. On macOS that driver needs
the main thread, and on Windows it needs COM set up in the thread, so it either failed or
hung. After a hang the keyboard thought it was *still speaking*, so SPEAK acted as STOP and every
later sentence was dropped. Errors only went to the console.

**Now:**
- Speech engine per OS: macOS `say`, Windows SAPI voice (PowerShell fallback), Linux espeak-ng.
  pyttsx3 is last resort only. A watchdog guarantees the voice can never stay stuck.
- **Speak-as-you-type** (`speech_echo` in config/config.py):
  `word` (default) = each word is spoken as soon as it is finished (SPACE, suggestion, punctuation),
  and the whole sentence is spoken at `.` / `?`; `letter` = also every letter; `sentence`; `off`.
- SPEAK reads everything typed; looking at it while talking = STOP.
- The text bar shows which voice is active, or the error in red if there is none.
- **If you hear nothing, run `python tools/test_speech.py` first** - it names the engine and the error.

---

## What changed in v4

**Bottom rows / right-hand columns could not be selected - three causes fixed:**
1. *Blink filter treated looking down as a blink.* Lowered lids (normal when looking at the
   bottom of the screen) were flagged as a blink indefinitely. Now a blink must be short
   (<= 0.35 s); a sustained low lid position is accepted as downward gaze (`gaze/blink.py`).
   Eye openness was also removed from the MediaPipe confidence score for the same reason.
2. *Head movement stole eye movement.* Users turn/tilt the head towards the edges, so the eyes
   rotate less than during calibration and the mapping under-reached the borders. Calibration now
   records head yaw/pitch and adds head-pose terms when they improve the leave-one-out error.
3. *Edges were extrapolated.* Calibration adds 8 edge targets (corners + mid-edges, 3% margin) to
   the 16-point grid, fits a cubic model, rejects a calibration with no data on any edge, and never
   hangs on a target the tracker cannot see (4 s timeout per target).
   `edge_gain_x/y` in config is a manual last-resort stretch (default 1.0).

**Word suggestions**: a row of 4 predicted words above the phrases. Completes the current word or
predicts the next one (care vocabulary + ~20k-word English list + words the user types, learned
into `data/user_vocab.json`). Offline. Offensive words are never suggested.

**Text-to-speech**: SPEAK key (top-right of the suggestion row) reads the typed text; looking at it
again while talking = STOP. Finished sentences (`.` `?`) are spoken automatically
(`keyboard_speak_on_sentence_end`). Backends: pyttsx3 -> Windows System.Speech (built in) ->
macOS `say` -> Linux espeak. The dashboard text box also has a "Speak" button.

```
+--------------------------------------------------------------+-------+
|  typed text                                                  | PAUSE |
+------------+------------+------------+------------+------------------+
| suggestion | suggestion | suggestion | suggestion |   SPEAK / STOP   |
+-------+-------+----------+------+------+----------+-----+------------+
| WATER | FOOD  | WASHROOM | HELP | PAIN | MEDICINE | YES |  NO        |
+-------+-------+-------+-------+-------+-------+-------+------------+
|   A   |   B   |   C   |   D   |   E   |   F   |   G   |   H        |
|   I   |   J   |   K   |   L   |   M   |   N   |   O   |   P        |
|   Q   |   R   |   S   |   T   |   U   |   V   |   W   |   X        |
|   Y   |   Z   |   .   |   ?   | SPACE | BACKSPACE | CLEAR | ENTER  |
+--------------------------------------------------------------------+
```
Recalibrate after upgrading: v3 calibration profiles are rejected (different model).

---

## What changed in v3

| Area | v1 (original) | v3 (this build) |
|---|---|---|
| Gaze backend | dlib 68 landmarks + pupil thresholding | **MediaPipe Face Mesh + Iris** (offline, default). dlib kept as `--backend dlib` |
| Horizontal precision* | ~76 px jitter | **~48 px** |
| Vertical precision* | ~404 px jitter (lid-normalised ratio, effectively noise) | **~39 px** (iris offset from eye-corner line fused with lid aperture) |
| Blink handling | fixed EAR threshold (missed most blinks) | adaptive per user + 3-frame post-blink hold-off |
| Calibration | 3x3, 12% margins, 15 samples, training error shown | **4x4, 5% margins (edges measured)**, 30 samples, median, leave-one-out error shown, rejects bad fits |
| Long-session drift | none | **online drift correction** learned from every confirmed key (undone by DEL) |
| Keyboard | 3x3 demo keypad | **full-screen keyboard: 8 quick phrases + A-Z + controls, tiled edge-to-edge** |

*Per-frame jitter measured on the candidate video, assuming the recorded eye movements span 90% of a
1920x1080 screen. This is precision (steadiness), not accuracy (bias): accuracy must be measured with
`tools/record_session.py` + `tools/replay_video.py` on your real webcam.

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

### Measuring accuracy (do this before claiming a number)
```
python tools/record_session.py --out data/recordings/p01 --grid 4x4 --hold 2 --shuffle
python tools/replay_video.py data/recordings/p01.mp4 --targets data/recordings/p01_targets.csv --mirror
```

---

## Original documentation (v1)

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

1. Click **🎯 Calibrate (9-pt)** in the top bar.
2. A fullscreen window will display 9 sequential targets (Top-Left $\rightarrow$ Bottom-Right).
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

## 9. Running Automated Unit Tests

To run the full unit test suite verifying fixation, dwell engine, blink pausing, and calibration models:
```powershell
python -m unittest discover tests -v
```

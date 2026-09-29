"""
Offline replay / tuning harness.

Runs the SAME GazeDetector used live on a recorded video, so feature-extraction
changes can be measured repeatably instead of by eye.

  # 1) Just inspect the signal (no ground truth):
  python tools/replay_video.py recording.mp4 --plot

  # 2) Full accuracy test with ground truth written by tools/record_session.py:
  python tools/replay_video.py recording.mp4 --targets recording_targets.csv --screen 1920x1080

Outputs <video>_features.csv (one row per frame) and, with --targets, a per-target
accuracy table using leave-one-target-out calibration (honest error, in px).

Tip: set --mirror if the live app flips the frame (dashboard does cv2.flip(frame,1)).
"""
import argparse, csv, os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gaze.mediapipe_detector import create_detector   # noqa: E402

SETTLE_S = 0.35   # ignore this long after each target appears (saccade + correction)


def extract(video, model, width, mirror, backend="mediapipe"):
    det = create_detector(backend, model, 0.15, 0.45)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if width and f.shape[1] != width:
            s = width / f.shape[1]
            f = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if mirror:
            f = cv2.flip(f, 1)
        det.refresh(f, i / fps)
        el, er = det.eye_left, det.eye_right
        rows.append(dict(t=i / fps, face=int(det.is_face_detected()), valid=int(det.is_valid()),
                         blink=int(det.is_blinking()), conf=det.confidence(),
                         hr=det.horizontal_ratio(), vr=det.vertical_ratio(),
                         ear=((el.ear + er.ear) / 2) if el and er else None,
                         head_yaw=(det.head_pose() or (None, None))[0],
                         head_pitch=(det.head_pose() or (None, None))[1]))
        i += 1
    return rows, fps


def arr(rows, k):
    return np.array([np.nan if r[k] is None else r[k] for r in rows], float)


def load_targets(path):
    with open(path) as f:
        return [dict(t0=float(r["t_start"]), t1=float(r["t_end"]), x=float(r["screen_x"]), y=float(r["screen_y"]))
                for r in csv.DictReader(f)]


def poly(u, v):
    return np.column_stack([np.ones_like(u), u, v, u**2, u * v, v**2])


def evaluate(rows, targets, screen):
    t, hr, vr = arr(rows, "t"), arr(rows, "hr"), arr(rows, "vr")
    ok = ~np.isnan(hr) & ~np.isnan(vr)
    per = []
    for tg in targets:
        m = ok & (t >= tg["t0"] + SETTLE_S) & (t <= tg["t1"])
        if m.sum() < 5:
            continue
        per.append(dict(tg, idx=np.nonzero(m)[0], hr=np.median(hr[m]), vr=np.median(vr[m])))
    if len(per) < 7:
        print("Need >= 7 usable targets for polynomial leave-one-out."); return
    U = np.array([p["hr"] for p in per]); V = np.array([p["vr"] for p in per])
    X = np.array([p["x"] for p in per]); Y = np.array([p["y"] for p in per])
    print(f"\nFeature span across targets: hr {np.ptp(U):.3f}  vr {np.ptp(V):.3f}")
    print(f"{'target':>12} {'bias px':>8} {'SD px':>7} {'valid%':>7}")
    errs, sds = [], []
    for i, p in enumerate(per):
        m = np.ones(len(per), bool); m[i] = False
        cx = np.linalg.lstsq(poly(U[m], V[m]), X[m], rcond=None)[0]
        cy = np.linalg.lstsq(poly(U[m], V[m]), Y[m], rcond=None)[0]
        B = poly(hr[p["idx"]], vr[p["idx"]])
        px, py = B @ cx, B @ cy
        bias = np.hypot(np.median(px) - p["x"], np.median(py) - p["y"])
        sd = np.sqrt(np.var(px) + np.var(py))
        n_all = ((t >= p["t0"] + SETTLE_S) & (t <= p["t1"])).sum()
        errs.append(bias); sds.append(sd)
        print(f"({p['x']:4.0f},{p['y']:4.0f}) {bias:8.0f} {sd:7.0f} {100*len(p['idx'])/max(1,n_all):6.0f}%")
    W, H = screen
    print(f"\nACCURACY (median leave-one-out bias): {np.median(errs):.0f} px "
          f"= {100*np.median(errs)/W:.1f}% of screen width")
    print(f"PRECISION (median per-frame SD, before smoothing): {np.median(sds):.0f} px")
    print("Rule of thumb: dwell targets should be >= 2 x (bias + SD) wide.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--targets")
    ap.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "..", "models",
                                                     "shape_predictor_68_face_landmarks.dat"))
    ap.add_argument("--width", type=int, default=640, help="resize to live camera width (default 640)")
    ap.add_argument("--screen", default="1920x1080")
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--backend", default="mediapipe", choices=["mediapipe", "dlib"])
    a = ap.parse_args()

    rows, fps = extract(a.video, a.model, a.width, a.mirror, a.backend)
    out = os.path.splitext(a.video)[0] + "_features.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    hr, vr = arr(rows, "hr"), arr(rows, "vr")
    print(f"{len(rows)} frames @ {fps:.1f} fps -> {out}")
    print(f"face {100*np.mean(arr(rows,'face')):.0f}%  valid {100*np.mean(arr(rows,'valid')):.0f}%  "
          f"blink/holdoff {100*np.mean(arr(rows,'blink')):.1f}%")
    print(f"hr 5-95%: {np.nanpercentile(hr,5):.3f}-{np.nanpercentile(hr,95):.3f}  "
          f"frame jitter {np.nanmedian(np.abs(np.diff(hr))):.4f}")
    print(f"vr 5-95%: {np.nanpercentile(vr,5):.3f}-{np.nanpercentile(vr,95):.3f}  "
          f"frame jitter {np.nanmedian(np.abs(np.diff(vr))):.4f}")
    if a.targets:
        evaluate(rows, load_targets(a.targets), tuple(int(v) for v in a.screen.split("x")))
    if a.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        t = arr(rows, "t")
        fig, ax = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
        ax[0].plot(t, hr); ax[0].set_ylabel("hr"); ax[1].plot(t, vr, c="C1"); ax[1].set_ylabel("vr")
        ax[2].plot(t, arr(rows, "ear"), c="C2"); ax[2].set_ylabel("EAR"); ax[2].set_xlabel("s")
        png = os.path.splitext(a.video)[0] + "_signals.png"; plt.tight_layout(); plt.savefig(png, dpi=90)
        print("plot ->", png)


if __name__ == "__main__":
    main()

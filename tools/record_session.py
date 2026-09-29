"""
Ground-truth recorder: shows fullscreen targets AND records the webcam, so every
frame has a known "where the person was looking". This is what the candidate video
was missing - without it only precision (jitter) can be measured, not accuracy.

  python tools/record_session.py --out data/recordings/rahul_01 --grid 4x4 --hold 2.0

Writes <out>.mp4 (webcam, un-mirrored) and <out>_targets.csv (t_start,t_end,screen_x,screen_y).
Then:  python tools/replay_video.py <out>.mp4 --targets <out>_targets.csv --mirror
Keys: ESC aborts.
"""
import argparse, csv, os, random, time
import numpy as np
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--grid", default="4x4", help="cols x rows (4x4 = 16 targets; >=9 needed)")
    ap.add_argument("--hold", type=float, default=2.0, help="seconds per target")
    ap.add_argument("--margin", type=float, default=0.08)
    ap.add_argument("--screen", default="1920x1080")
    ap.add_argument("--shuffle", action="store_true", help="random order (reduces anticipation bias)")
    a = ap.parse_args()
    W, H = (int(v) for v in a.screen.split("x"))
    cols, rows = (int(v) for v in a.grid.split("x"))
    xs = np.linspace(W * a.margin, W * (1 - a.margin), cols)
    ys = np.linspace(H * a.margin, H * (1 - a.margin), rows)
    pts = [(int(x), int(y)) for y in ys for x in xs]
    if a.shuffle:
        random.shuffle(pts)

    cap = cv2.VideoCapture(a.camera, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(a.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480); cap.set(cv2.CAP_PROP_FPS, 30)
    ok, fr = cap.read()
    if not ok:
        raise SystemExit("camera not available")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    vw = cv2.VideoWriter(a.out + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (fr.shape[1], fr.shape[0]))
    cv2.namedWindow("rec", cv2.WINDOW_NORMAL); cv2.setWindowProperty("rec", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    log, n_frames = [], 0
    for k, (px, py) in enumerate([None] + pts):          # first slot = 3 s "get ready"
        dur = 3.0 if px is None else a.hold
        t_start = n_frames / 30.0
        t0 = time.time()
        while time.time() - t0 < dur:
            ok, fr = cap.read()
            if not ok:
                continue
            vw.write(fr); n_frames += 1       # time base = frame count -> matches the video exactly
            canvas = np.zeros((H, W, 3), np.uint8)
            if px is None:
                cv2.putText(canvas, "Keep head still. Follow the dot with your eyes.", (W // 2 - 420, H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (220, 220, 220), 2)
            else:
                r = int(18 - 10 * min(1, (time.time() - t0) / 0.6))
                cv2.circle(canvas, (px, py), r, (0, 200, 255), -1); cv2.circle(canvas, (px, py), 3, (0, 0, 0), -1)
            cv2.imshow("rec", canvas)
            if cv2.waitKey(1) == 27:
                pts = []; break
        if px is not None:
            log.append(dict(t_start=round(t_start, 3), t_end=round(n_frames / 30.0, 3), screen_x=px, screen_y=py))
    vw.release(); cap.release(); cv2.destroyAllWindows()
    with open(a.out + "_targets.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t_start", "t_end", "screen_x", "screen_y"]); w.writeheader(); w.writerows(log)
    print(f"saved {a.out}.mp4 and {a.out}_targets.csv ({len(log)} targets)")
    # Target times are frame-count based, so they stay aligned with the saved video
    # even if the camera delivers fewer than 30 fps.


if __name__ == "__main__":
    main()

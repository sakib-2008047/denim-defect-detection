"""
record_trial.py - record a trial on the laptop (no GPU, no model needed).

Records the capture-card stream at 640 x 480 (same setting as the original trials) to a video file,
with an accurate timestamp for every frame and the moments you press a key.
Detection is run afterwards on Google Colab (GPU) with NB6_trials_colab.py, so every frame is analysed.

Keys while recording:
  SPACE = "defect passing now" (press when a defect is in the middle of the view)
  m     = general marker (e.g. "speed changed", "stain applied")
  q     = stop and save

Usage (Windows command prompt):
  pip install opencv-python
  python record_trial.py --label clean --speed 1.90
  python record_trial.py --label defects --speed 3.55 --note "order: H MY AM OS H MY AM OS H MY AM OS"
  python record_trial.py --label oil_age05 --speed 4.58 --note "stains applied 14:02"

Output (next to this script): trials/<label>_<speed>_<date-time>/  video.avi, frames.csv, markers.csv, meta.json
Copy the trial folders to Google Drive (fabric/trials/) and run NB6_trials_colab.py.
"""
import argparse, csv, json, os, time
from datetime import datetime
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--label")
ap.add_argument("--speed", type=float, help="measured fabric speed in cm/s (0 if static)")
ap.add_argument("--note", default="")
ap.add_argument("--camera", type=int, default=0, help="try 1 or 2 if 0 opens the laptop webcam")
args = ap.parse_args()
# If started without arguments (e.g. Run button in an editor), ask for them
if not args.label:
    args.label = input("Label (e.g. test, clean, defects, oil_age05): ").strip() or "test"
if args.speed is None:
    args.speed = float(input("Measured speed in cm/s (0 if not moving): ").strip() or 0)
if not args.note:
    args.note = input("Note (optional, press Enter to skip): ").strip()
cam = input(f"Camera number [Enter = {args.camera}; try 1 if the laptop webcam opens]: ").strip()
if cam:
    args.camera = int(cam)

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE = os.path.dirname(os.path.abspath(__file__))   # always next to this script
out = os.path.join(BASE, "trials", f"{args.label}_{str(args.speed).replace('.', 'p')}_{stamp}")
print("Files will be saved in:", out)
os.makedirs(out, exist_ok=True)

cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)          # same as the deployed script
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
ok, frame = cap.read()
if not ok:
    raise SystemExit("Camera not found - try --camera 1")
h, w = frame.shape[:2]
print(f"Capture size: {w} x {h}")
writer = cv2.VideoWriter(os.path.join(out, "video.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (w, h))
fcsv = open(os.path.join(out, "frames.csv"), "w", newline=""); fw = csv.writer(fcsv); fw.writerow(["frame", "t_s"])
mcsv = open(os.path.join(out, "markers.csv"), "w", newline=""); mw = csv.writer(mcsv); mw.writerow(["frame", "t_s", "key"])

cv2.namedWindow("record_trial", cv2.WINDOW_NORMAL)
t0 = time.perf_counter(); i = 0
while True:
    if i > 0:
        ok, frame = cap.read()
        if not ok:
            break
    t = time.perf_counter() - t0
    writer.write(frame)
    fw.writerow([i, f"{t:.4f}"])
    disp = frame.copy()
    cv2.putText(disp, f"REC {args.label}  {args.speed} cm/s  t={t:6.1f}s  [SPACE=defect m=marker q=stop]",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 1)
    cv2.imshow("record_trial", disp)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    if k in (ord(" "), ord("m")):
        mw.writerow([i, f"{t:.4f}", "defect" if k == ord(" ") else "marker"]); mcsv.flush()
    i += 1

dur = time.perf_counter() - t0
meta = {"label": args.label, "speed_cm_s": args.speed, "note": args.note, "started": stamp, "frames": i,
        "duration_s": round(dur, 2), "mean_fps": round(i / dur, 2) if dur else None, "frame_size": [w, h]}
json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)
fcsv.close(); mcsv.close(); writer.release(); cap.release(); cv2.destroyAllWindows()
print("\nSaved to:", out)
print(meta)
try:
    os.startfile(out)   # opens the folder in Windows Explorer
except Exception:
    pass

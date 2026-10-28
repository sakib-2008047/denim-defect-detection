"""
analyze_trials.py - reproduce the supplementary-trial results (Section 4.6, Table 7).

Input: the folder produced by NB6_trials_colab.py ("nb6"), containing one sub-folder per run
       (E1_clean_S1_..., E2_defects_S1_..., defect_free_images) with detections.csv, frames.csv, meta.json.
Usage: python analyze_trials.py path/to/nb6
Output: printed tables + trial_results.json

Definitions (Section 3.6):
  * a frame is flagged when any box has confidence >= threshold
    (0.60, or the class-specific thresholds selected out of sample in each cross-validation fold);
  * a stop needs k consecutive flagged frames (k = 1, 2, 3);
  * E1 (defect-free fabric): number of stop events;
  * E2 (defects): a defect counts as detected if a run of k flagged frames occurs in its time window,
    which starts 0.5 s before the labelled time and ends at the next defect's window.
    Labelled times are playback seconds in the video; the AVI declares 30 fps, so frame = seconds x 30,
    and the real time of that frame is read from frames.csv.
"""
import csv, glob, json, os, sys
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "nb6"
HERE = os.path.dirname(os.path.abspath(__file__))
FOLD_THR = json.load(open(os.path.join(HERE, "class_thresholds_per_fold.json")))
SETTINGS = [("0.60", 0.60)] + [(f"class_fold{i}", t) for i, t in enumerate(FOLD_THR)]
CLS = {"H": "Holes", "MY": "Missing_Yarn", "AM": "Abrasion_Mark", "OS": "Oil_Stain"}


def load(run):
    d = glob.glob(os.path.join(ROOT, run + "*"))[0]
    det = list(csv.DictReader(open(os.path.join(d, "detections.csv"))))
    t = [float(r["t_s"]) for r in csv.DictReader(open(os.path.join(d, "frames.csv")))]
    assert len(t) == len(det)
    boxes = [json.loads(r["boxes_json"]) for r in det]
    sharp = [float(r["sharpness"]) for r in det]
    meta = json.load(open(os.path.join(d, "meta.json")))
    return np.array(t), boxes, sharp, meta


def flagged(bs, thr):
    return any(b["conf"] >= (thr[b["cls"]] if isinstance(thr, dict) else thr) for b in bs)


def stop_events(flags, k):
    run = ev = 0; stopped = False
    for f in flags:
        run = run + 1 if f else 0
        if not stopped and run >= k: stopped = True; ev += 1
        if stopped and not f: stopped = False
    return ev


def longest_run(flags):
    m = r = 0
    for f in flags:
        r = r + 1 if f else 0; m = max(m, r)
    return m


def wilson(x, n, z=1.96):
    p = x / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / d, (c + r) / d


out = {}
# ---------- defect-free still images ----------
rows = list(csv.DictReader(open(os.path.join(ROOT, "defect_free_images", "images.csv"))))
out["images"] = {}
for name, thr in [("0.25", 0.25)] + SETTINGS:
    x = sum(flagged(json.loads(r["boxes_json"]), thr) for r in rows)
    out["images"][name] = {"flagged": x, "n": len(rows), "wilson95": [round(v, 4) for v in wilson(x, len(rows))]}
print("Defect-free images:", {k: v["flagged"] for k, v in out["images"].items()}, "of", len(rows))

# ---------- E1: defect-free moving fabric ----------
out["E1"] = {}
for s in ("S1", "S2", "S3"):
    t, boxes, sharp, m = load(f"E1_clean_{s}")
    r = {"speed_cm_s": m["speed_cm_s"], "frames": len(t), "duration_s": round(float(t[-1]), 1),
         "fabric_m": round(float(t[-1]) * m["speed_cm_s"] / 100, 2), "median_sharpness": round(float(np.median(sharp)))}
    for name, thr in SETTINGS:
        fl = [flagged(b, thr) for b in boxes]
        r[name] = {"flagged_frames": sum(fl), **{f"stops_k{k}": stop_events(fl, k) for k in (1, 2, 3)}}
    out["E1"][s] = r
    print("E1", s, r)

# ---------- E2: defects ----------
labels = {r["run"]: (r["defect_order"].split(), [float(v) for v in r["defect_time_in_video_s"].split()])
          for r in csv.DictReader(open(os.path.join(HERE, "E2_defect_labels.csv")))}
out["E2"] = {}
for s in ("S1", "S2", "S3"):
    run = f"E2_defects_{s}"; order, sec = labels[run]
    t, boxes, sharp, m = load(run); n = len(t)
    start = [t[min(int(x * 30), n - 1)] - 0.5 for x in sec] + [t[-1] + 1]
    res = []
    for i, c in enumerate(order):
        w = [j for j in range(n) if start[i] <= t[j] < start[i + 1]]
        r = {"defect": c, "frames": len(w), "max_conf": float(max([b["conf"] for j in w for b in boxes[j]], default=0))}
        for name, thr in SETTINGS:
            r[name] = longest_run([flagged(boxes[j], thr) for j in w])
        res.append(r)
    out["E2"][s] = res

print("\nTable 7 - defects detected out of 3 (1.90 / 3.55 / 4.58 cm/s)")
for name, _ in SETTINGS:
    for k in (1, 2, 3):
        cells = []
        for c in ("H", "MY", "AM", "OS"):
            cells.append(c + " " + "/".join(str(sum(r[name] >= k for r in out["E2"][s] if r["defect"] == c)) for s in ("S1", "S2", "S3")))
        print(f"{name:12s} k={k}: " + "   ".join(cells))

json.dump(out, open("trial_results.json", "w"), indent=1)
print("\nSaved trial_results.json")

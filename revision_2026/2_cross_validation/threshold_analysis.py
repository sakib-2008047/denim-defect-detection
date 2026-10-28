"""
Threshold / operating-point analysis on the baseline YOLOv8n (trained 768, inferred 640)
using the per-image predictions exported by Notebook 1. Addresses R3.3, R4.10, R3.10.

Instance level: class-aware greedy matching, IoU >= 0.5 (same criterion as mAP@0.5).
Image level  : the actual stop decision = "any box above threshold in the frame".
"""
import numpy as np, pandas as pd, json

NAMES = {0: "Holes", 1: "Abrasion_Mark", 2: "Oil_Stain", 3: "Missing_Yarn"}
p = pd.read_csv("val_predictions.csv")
g = pd.read_csv("val_ground_truth.csv")


def iou(a, b):  # xywh normalised
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    return inter / (a[2] * a[3] + b[:, 2] * b[:, 3] - inter + 1e-12)


# --- mark each prediction TP/FP once (class-aware, greedy by confidence) ---
p = p.sort_values("conf", ascending=False).reset_index(drop=True)
p["tp"] = False
gt_by = {k: v[["xc", "yc", "w", "h"]].to_numpy() for k, v in g.groupby(["image", "cls"])}
used = {k: np.zeros(len(v), bool) for k, v in gt_by.items()}
for i, r in p.iterrows():
    k = (r.image, r.cls)
    if k not in gt_by:
        continue
    ious = iou(np.array([r.xc, r.yc, r.w, r.h]), gt_by[k])
    ious[used[k]] = -1
    j = int(np.argmax(ious))
    if ious[j] >= 0.5:
        used[k][j] = True
        p.at[i, "tp"] = True

n_gt = g.groupby("cls").size()
img_cls = g.groupby("image").cls.agg(lambda s: sorted(set(s)))
grid = np.round(np.arange(0.05, 0.951, 0.05), 2)

rows = []
for t in sorted(set(grid) | {0.25, 0.524, 0.60}):
    q = p[p.conf >= t]
    for c in NAMES:
        qc = q[q.cls == c]
        tp, fp = int(qc.tp.sum()), int((~qc.tp).sum())
        P = tp / (tp + fp) if tp + fp else np.nan
        R = tp / n_gt[c]
        F1 = 2 * P * R / (P + R) if P and R and not np.isnan(P) else 0.0
        # image-level: share of images containing class c that would trigger a stop (any box, any class)
        imgs_c = img_cls[img_cls.apply(lambda s: c in s)].index
        stop = q[q.image.isin(imgs_c)].image.nunique() / len(imgs_c)
        rows.append(dict(thr=t, cls=NAMES[c], gt=int(n_gt[c]), tp=tp, fp=fp, fn=int(n_gt[c] - tp),
                         P=P, R=R, F1=F1, image_stop_rate=stop))
df = pd.DataFrame(rows)
df.to_csv("threshold_table.csv", index=False)

pd.set_option("display.width", 200)
fmt = lambda d: d.to_string(index=False, float_format=lambda v: f"{v:.3f}")
for t in (0.25, 0.524, 0.60):
    d = df[df.thr == t]
    print(f"\n=== threshold {t} ===")
    print(fmt(d[["cls", "gt", "tp", "fp", "fn", "P", "R", "F1", "image_stop_rate"]]))
    tp, fp = d.tp.sum(), d.fp.sum()
    print(f"  all classes: TP {tp}  FP {fp}  FN {d.fn.sum()}  P {tp/(tp+fp):.3f}  R {tp/d["gt"].sum():.3f}"
          f"  | images that would stop: {p[p.conf>=t].image.nunique()}/300")

# per-class F1-optimal threshold
print("\n=== per-class F1-optimal threshold (grid 0.05 step) ===")
best = df[df.thr.isin(grid)].loc[lambda d: d.groupby("cls").F1.idxmax()]
print(fmt(best[["cls", "thr", "P", "R", "F1", "fp", "fn", "image_stop_rate"]]))

# class-specific scheme vs global 0.60
scheme = {r.cls: r.thr for r in best.itertuples()}
tot = dict(tp=0, fp=0, fn=0)
for c, name in NAMES.items():
    r = df[(df.cls == name) & (df.thr == scheme[name])].iloc[0]
    tot["tp"] += r.tp; tot["fp"] += r.fp; tot["fn"] += r.fn
g60 = df[df.thr == 0.60]
print(f"\nGlobal 0.60      : TP {g60.tp.sum()} FP {g60.fp.sum()} FN {g60.fn.sum()}")
print(f"Per-class optimum: TP {tot['tp']} FP {tot['fp']} FN {tot['fn']}  thresholds {scheme}")

# image-level stop rate per image class at 0.60 (single-frame, static)
json.dump({"per_class_F1_opt": scheme}, open("threshold_summary.json", "w"), indent=1)

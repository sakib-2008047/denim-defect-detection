"""
Grouped 5-fold CV analysis (EAAI-26-20775 revision). Run from analysis/cv/.
1) mean ± SD per model (640 and 768 inference)            -> R4.3
2) paired YOLOv10n vs YOLOv8n comparison across folds      -> R1.4 / R2.5
3) 768 vs 640 inference across folds                       -> R2.6
4) operating point: fixed 0.60 vs class-specific thresholds chosen
   out-of-sample (thresholds tuned on the other 4 folds, applied to the held-out fold) -> R3.3 / R4.10
Matching for (4): class-aware greedy, IoU >= 0.5, same as mAP@0.5.
"""
import json, numpy as np, pandas as pd
from scipy import stats

NAMES = {0: "Holes", 1: "Abrasion_Mark", 2: "Oil_Stain", 3: "Missing_Yarn"}
MODELS, FOLDS = ("yolov8n", "yolov10n"), range(5)
S = {(m, f): json.load(open(f"{m}_fold{f}/summary.json")) for m in MODELS for f in FOLDS}
for k, s in S.items():
    assert s["fold_check"] == "e5ff4456", k
out = {}

# ---------- 1) mean ± SD ----------
print("=== 1) Mean ± SD across 5 folds ===")
rows = []
for m in MODELS:
    for sz in ("infer640", "infer768"):
        e = [S[m, f]["eval"][sz] for f in FOLDS]
        r = {"model": m, "infer": sz[5:]}
        for key in ("mAP50", "mAP50_95", "P", "R"):
            v = np.array([x[key] for x in e]); r[key] = f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"
        for c in NAMES.values():
            v = np.array([x["per_class"][c]["AP50"] for x in e]); r[c] = f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"
        rows.append(r)
t1 = pd.DataFrame(rows); print(t1.to_string(index=False)); out["mean_sd"] = rows

# ---------- 2) paired model comparison ----------
def paired(a, b, label):
    d = np.array(b) - np.array(a)
    t = stats.ttest_rel(b, a)
    ci = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    w = stats.wilcoxon(b, a, method="exact")
    r = dict(comparison=label, diffs=[round(x, 3) for x in d], mean_diff=round(d.mean(), 4),
             ci95=[round(d.mean() - ci, 4), round(d.mean() + ci, 4)], t_p=round(t.pvalue, 3),
             wilcoxon_p=round(w.pvalue, 3), wins=f"{(d > 0).sum()}/{len(d)}")
    print(r); return r

print("\n=== 2) YOLOv10n − YOLOv8n, paired by fold ===")
out["model_comparison"] = []
for sz in ("infer640", "infer768"):
    for key in ("mAP50", "mAP50_95"):
        a = [S["yolov8n", f]["eval"][sz][key] for f in FOLDS]
        b = [S["yolov10n", f]["eval"][sz][key] for f in FOLDS]
        out["model_comparison"].append(paired(a, b, f"{key} @{sz[5:]}"))
for c in NAMES.values():
    a = [S["yolov8n", f]["eval"]["infer640"]["per_class"][c]["AP50"] for f in FOLDS]
    b = [S["yolov10n", f]["eval"]["infer640"]["per_class"][c]["AP50"] for f in FOLDS]
    out["model_comparison"].append(paired(a, b, f"AP50 {c} @640"))

# ---------- 3) 768 vs 640 inference ----------
print("\n=== 3) Inference 768 − 640, paired by fold ===")
out["resolution"] = []
for m in MODELS:
    for key in ("mAP50", "mAP50_95"):
        a = [S[m, f]["eval"]["infer640"][key] for f in FOLDS]
        b = [S[m, f]["eval"]["infer768"][key] for f in FOLDS]
        out["resolution"].append(paired(a, b, f"{m} {key}"))

# ---------- 4) operating point ----------
def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    return inter / (a[2] * a[3] + b[:, 2] * b[:, 3] - inter + 1e-12)

def load(m, f):
    p = pd.read_csv(f"{m}_fold{f}/val_predictions.csv").sort_values("conf", ascending=False).reset_index(drop=True)
    g = pd.read_csv(f"{m}_fold{f}/val_ground_truth.csv")
    gt = {k: v[["xc", "yc", "w", "h"]].to_numpy() for k, v in g.groupby(["image", "cls"])}
    used = {k: np.zeros(len(v), bool) for k, v in gt.items()}
    tp = np.zeros(len(p), bool)
    for i, r in enumerate(p.itertuples()):
        k = (r.image, r.cls)
        if k not in gt: continue
        o = iou(np.array([r.xc, r.yc, r.w, r.h]), gt[k]); o[used[k]] = -1
        j = int(np.argmax(o))
        if o[j] >= 0.5: used[k][j] = True; tp[i] = True
    p["tp"], p["fold"] = tp, f; g["fold"] = f
    return p, g

def counts(p, g, thr):  # thr: dict cls -> threshold
    keep = p[p.conf >= p.cls.map(thr)]
    res = {}
    for c in NAMES:
        q = keep[keep.cls == c]; n = int((g.cls == c).sum())
        res[c] = (int(q.tp.sum()), int((~q.tp).sum()), n)
    imgs = g.image.astype(str) + "#" + g.fold.astype(str)
    stopped = set(keep.image.astype(str) + "#" + keep.fold.astype(str))
    return res, len(set(imgs) & stopped), imgs.nunique()

def f1(tp, fp, n):
    P = tp / (tp + fp) if tp + fp else 0; R = tp / n
    return 2 * P * R / (P + R) if P + R else 0

GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)
print("\n=== 4) Operating point (pooled over 5 held-out folds) ===")
out["operating_point"] = {}
for m in MODELS:
    data = [load(m, f) for f in FOLDS]
    chosen, pooled = [], {"0.60": [], "class_specific_oos": [], "0.25": []}
    for k in FOLDS:
        tr_p = pd.concat([data[f][0] for f in FOLDS if f != k]); tr_g = pd.concat([data[f][1] for f in FOLDS if f != k])
        thr = {}
        for c in NAMES:
            best = max(GRID, key=lambda t: f1(*counts(tr_p, tr_g, {i: t for i in NAMES})[0][c]))
            thr[c] = float(best)
        chosen.append(thr)
        for name, th in (("0.60", {c: 0.60 for c in NAMES}), ("class_specific_oos", thr), ("0.25", {c: 0.25 for c in NAMES})):
            pooled[name].append(counts(data[k][0], data[k][1], th))
    print(f"\n{m}: class thresholds chosen per held-out fold (on other 4 folds):")
    for k, th in enumerate(chosen): print(f"  fold {k}: " + ", ".join(f"{NAMES[c]} {v:.2f}" for c, v in th.items()))
    res_m = {"thresholds_per_fold": [{NAMES[c]: v for c, v in th.items()} for th in chosen]}
    for name, lst in pooled.items():
        row = {}
        for c in NAMES:
            tp = sum(x[0][c][0] for x in lst); fp = sum(x[0][c][1] for x in lst); n = sum(x[0][c][2] for x in lst)
            row[NAMES[c]] = dict(TP=tp, FP=fp, FN=n - tp, P=round(tp / (tp + fp), 3) if tp + fp else None,
                                 R=round(tp / n, 3), F1=round(f1(tp, fp, n), 3))
        per_fold_R = [sum(x[0][c][0] for c in NAMES) / sum(x[0][c][2] for c in NAMES) for x in lst]
        stop = sum(x[1] for x in lst); nimg = sum(x[2] for x in lst)
        row["all"] = dict(TP=sum(v["TP"] for v in row.values()), FP=sum(v["FP"] for v in row.values()),
                          instance_recall=round(sum(x[0][c][0] for x in lst for c in NAMES) / sum(x[0][c][2] for x in lst for c in NAMES), 3),
                          instance_recall_fold_sd=round(float(np.std(per_fold_R, ddof=1)), 3),
                          image_stop_rate=f"{stop}/{nimg} = {stop / nimg:.3f}")
        res_m[name] = row
        print(f"  [{name}] " + " | ".join(f"{c}: P {v['P']} R {v['R']} F1 {v['F1']}" for c, v in row.items() if c != "all"))
        print(f"           all: {row['all']}")
    out["operating_point"][m] = res_m

json.dump(out, open("cv_results.json", "w"), indent=2, default=float)
print("\nSaved cv_results.json")

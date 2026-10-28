"""Preprocessing ablation: CLAHE and AGCWD vs no preprocessing (YOLOv8n CV control), paired by fold."""
import json, numpy as np
from scipy import stats
L=lambda p:json.load(open(p))["eval"]
G={"none":[L(f"../cv/yolov8n_fold{k}/summary.json") for k in range(5)],
   "clahe":[L(f"clahe_fold{k}/summary.json") for k in range(5)],
   "agcwd":[L(f"agcwd_fold{k}/summary.json") for k in range(5)]}
M={"mAP50":lambda e:e["mAP50"],"mAP50_95":lambda e:e["mAP50_95"]}
for c in ("Holes","Abrasion_Mark","Oil_Stain","Missing_Yarn"): M[c+"_AP50"]=lambda e,c=c:e["per_class"][c]["AP50"]
M["Abrasion_recall"]=lambda e:e["per_class"]["Abrasion_Mark"]["R"]
out={}
for sz in ("infer640","infer768"):
  print(f"\n=== {sz} ===")
  for name,f in M.items():
    v={g:np.array([f(x[sz]) for x in G[g]]) for g in G}
    row={g:f"{v[g].mean():.3f} ± {v[g].std(ddof=1):.3f}" for g in G}
    for a,b in (("clahe","none"),("agcwd","none"),("agcwd","clahe")):
      d=v[a]-v[b]; ci=stats.t.ppf(.975,4)*d.std(ddof=1)/np.sqrt(5)
      row[f"{a}-{b}"]=dict(diffs=np.round(d,3).tolist(),mean=round(float(d.mean()),3),ci=[round(float(d.mean()-ci),3),round(float(d.mean()+ci),3)],
        t_p=round(float(stats.ttest_rel(v[a],v[b]).pvalue),2),w_p=round(float(stats.wilcoxon(v[a],v[b],method="exact").pvalue),2),better=f"{int((d>0).sum())}/5")
    out[f"{sz}:{name}"]=row
    print(f"{name:16s} none {row['none']}  clahe {row['clahe']}  agcwd {row['agcwd']}")
    for k in ("clahe-none","agcwd-none","agcwd-clahe"):
      r=row[k]; print(f"   {k:12s} {r['mean']:+.3f} [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] better {r['better']} t_p {r['t_p']} w_p {r['w_p']}  {r['diffs']}")
json.dump(out,open("preproc_results.json","w"),indent=1)

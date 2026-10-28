# NB6 - run the deployed model on the new images and trial videos (Google Colab, GPU runtime).
# Paste each cell into Colab. Runtime > Change runtime type > T4 GPU.

# ---- Cell 1: setup ----
from google.colab import drive
drive.mount('/content/drive')
!pip -q install ultralytics==8.4.183

# ---- Cell 2: paths (edit if your folders differ) ----
MODEL  = '/content/drive/MyDrive/fabric/runs_clean/detect/fabric_defect_yolov8n_clean/weights/best.pt'  # the deployed model
IMAGES = '/content/drive/MyDrive/fabric/defect_free_images'      # the defect-free still images
TRIALS = '/content/drive/MyDrive/fabric/trials'                  # folders made by record_trial.py
OLDVID = '/content/drive/MyDrive/fabric/clean_videos'            # optional: other clean-fabric videos
OUT    = '/content/drive/MyDrive/fabric/revision/nb6'

# ---- Cell 3: analysis code ----
import os, glob, json, csv, cv2, torch
from ultralytics import YOLO
model = YOLO(MODEL); names = model.names
os.makedirs(OUT, exist_ok=True)

def detect(img):
    r = model.predict(source=img, conf=0.25, imgsz=640, device=0, verbose=False)[0]
    return [{"cls": names[int(b.cls)], "conf": round(float(b.conf), 4),
             "xyxy": [round(float(v), 1) for v in b.xyxy[0]]} for b in (r.boxes or [])]

def sharp(img):
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())

def run_images(folder, label):
    files = sorted(sum([glob.glob(os.path.join(folder, e)) for e in ('*.jpg','*.jpeg','*.png','*.bmp','*.JPG','*.PNG')], []))
    od = os.path.join(OUT, label); os.makedirs(od, exist_ok=True); sizes = set()
    with open(os.path.join(od, 'images.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['file','width','height','sharpness','n_boxes_ge025','max_conf','boxes_json'])
        for p in files:
            img = cv2.imread(p); hh, ww = img.shape[:2]; sizes.add((ww, hh)); bx = detect(img)
            w.writerow([os.path.basename(p), ww, hh, f'{sharp(img):.1f}', len(bx), f"{max([b['conf'] for b in bx], default=0):.4f}", json.dumps(bx)])
    json.dump({'n_images': len(files), 'sizes': sorted(sizes)}, open(os.path.join(od, 'meta.json'), 'w'), indent=2)
    print(label, len(files), 'images', sorted(sizes))

def run_video(path, label, extra=None):
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS); n = 0; size = None
    od = os.path.join(OUT, label); os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, 'detections.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['frame','sharpness','n_boxes_ge025','max_conf','boxes_json'])
        while True:
            ok, img = cap.read()
            if not ok: break
            size = (img.shape[1], img.shape[0]); bx = detect(img)
            w.writerow([n, f'{sharp(img):.1f}', len(bx), f"{max([b['conf'] for b in bx], default=0):.4f}", json.dumps(bx)])
            n += 1
    meta = {'label': label, 'frames': n, 'fps_in_file': fps, 'frame_size': size, 'source': os.path.basename(path)}
    if extra: meta.update(extra)
    json.dump(meta, open(os.path.join(od, 'meta_nb6.json'), 'w'), indent=2)
    print(label, n, 'frames', size, fps, 'fps')

# ---- Cell 4: defect-free images ----
if os.path.isdir(IMAGES): run_images(IMAGES, 'defect_free_images')

# ---- Cell 5: other clean-fabric videos (optional) ----
if os.path.isdir(OLDVID):
    for p in sorted(glob.glob(os.path.join(OLDVID, '*'))):
        run_video(p, 'old_' + os.path.splitext(os.path.basename(p))[0])

# ---- Cell 6: trials recorded with record_trial.py (copies their logs next to the detections) ----
import shutil
for d in sorted(glob.glob(os.path.join(TRIALS, '*'))):
    v = os.path.join(d, 'video.avi')
    if not os.path.exists(v): continue
    label = os.path.basename(d)
    meta = json.load(open(os.path.join(d, 'meta.json')))
    run_video(v, label, meta)
    for fn in ('frames.csv', 'markers.csv', 'meta.json'):
        shutil.copy(os.path.join(d, fn), os.path.join(OUT, label, fn))

# ---- Cell 7: zip results to send ----
!cd /content/drive/MyDrive/fabric/revision && zip -qr nb6_results.zip nb6 && ls -la nb6_results.zip

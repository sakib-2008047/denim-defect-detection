from google.colab import drive
drive.mount('/content/drive')

import torch
print("GPU available:", torch.cuda.is_available())
print("GPU name       :", torch.cuda.get_device_name(0)
      if torch.cuda.is_available() else "NONE - check Runtime type!")


import zipfile, os, glob

zip_path = '/content/drive/MyDrive/fabric/dataset_clean.zip'
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall('/content/data')


for root, dirs, files in os.walk('/content/data'):
    if files:
        print(root, '->', len(files), 'files')


ids = set()
for lbl in glob.glob('/content/data/**/labels/**/*.txt', recursive=True):
    for line in open(lbl):
        if line.strip():
            ids.add(int(line.split()[0]))
print("Class IDs present in labels:", sorted(ids))


yaml_content = """
path: /content/data/dataset_clean
train: images/train
val: images/val
names:
  0: Holes
  1: Abrasion_Mark
  2: Oil_Stain
  3: Missing_Yarn
"""

with open('/content/data.yaml', 'w') as f:
    f.write(yaml_content)

print(open('/content/data.yaml').read())


from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # pretrained YOLOv8 Nano backbone

results = model.train(
    data="/content/data.yaml",
    epochs=100,
    imgsz=768,          # high resolution for small defects
    batch=16,
    device=0,           # GPU
    workers=8,
    patience=20,        # early stopping
    pretrained=True,
    optimizer="AdamW",
    lr0=0.001,
    cos_lr=True,        # cosine LR schedule
    augment=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    flipud=0.0,         # no vertical flip (preserves fabric orientation)
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    name="fabric_defect_yolov8n_clean",
    exist_ok=True,
)

import shutil

shutil.copytree(
    '/content/runs',
    '/content/drive/MyDrive/fabric/runs_clean',
    dirs_exist_ok=True
)
print("Saved to Drive: fabric/runs_clean")


import glob, yaml
from ultralytics import YOLO

# Locate best checkpoint
best_pt = glob.glob(
    '/content/drive/MyDrive/fabric/runs_clean/**/best.pt',
    recursive=True
)[0]
m = YOLO(best_pt)
print("Class names from model:", m.names)

# Rebuild data.yaml using the model's own class order
data_cfg = {
    'path': '/content/dataset/dataset_clean',
    'train': 'images/train',
    'val': 'images/val',
    'names': {int(k): v for k, v in m.names.items()},
}
with open('/content/data.yaml', 'w') as f:
    yaml.safe_dump(data_cfg, f, sort_keys=False)

# Validate at two resolutions
DATA = "/content/data.yaml"
for res in (768, 640):
    print(f"\n===== Validation @ {res}x{res} =====")
    r = m.val(data=DATA, imgsz=res, split="val", verbose=False)
    print(f"mAP@0.5      : {r.box.map50:.4f}")
    print(f"mAP@0.5:0.95 : {r.box.map:.4f}")
    for i, ci in enumerate(r.box.ap_class_index):
        print(f"   {m.names[int(ci)]:<20} AP@0.5 = {r.box.ap50[i]:.4f}")


import numpy as np, glob
from ultralytics import YOLO

model = YOLO(best_pt)
imgs = sorted(glob.glob("/content/dataset/dataset_clean/images/val/*.jpg"))
print(f"{len(imgs)} validation images found")

# GPU warm-up (first runs are artificially slow)
for _ in range(20):
    model.predict(imgs[0], imgsz=640, device=0, verbose=False)

pre, inf, post = [], [], []
for img in imgs:
    s = model.predict(img, imgsz=640, device=0, verbose=False)[0].speed
    pre.append(s['preprocess'])
    inf.append(s['inference'])
    post.append(s['postprocess'])

pre, inf, post = map(np.array, (pre, inf, post))
total = pre + inf + post

print(f"Preprocessing  : {pre.mean():.2f} +/- {pre.std():.2f} ms")
print(f"Inference      : {inf.mean():.2f} +/- {inf.std():.2f} ms")
print(f"Postprocessing : {post.mean():.2f} +/- {post.std():.2f} ms")
print(f"Total / frame  : {total.mean():.2f} +/- {total.std():.2f} ms")
print(f"Inference FPS  : {1000/inf.mean():.1f}")
print(f"End-to-end FPS : {1000/total.mean():.1f}")

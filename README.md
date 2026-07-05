# Detect-Decide-Act: Denim Defect Detection with Closed-Loop Actuation

Code extracted from the thesis appendices (Appendix A, B, C) accompanying
the manuscript "Detect-Decide-Act: Coupling Real-Time Defect Detection to
Physical Actuation in Denim Inspection."

## Structure

```
preprocessing/       Dataset preparation (resize, rename, augment)
training/            YOLOv8n training pipeline (Google Colab)
inference/           Real-time detection scripts (windowed / fullscreen / Pi-linked)
raspberry_pi/        Raspberry Pi-side control script
latency_benchmark/   Scripts behind the estimated latency budget (Table 2) - READ ITS README FIRST
```

## Important notes

- **`raspberry_pi/pi_server_DRAFT_NEEDS_VERIFICATION.py` is a reconstruction**,
  not code from the original thesis appendix. The appendix only includes the
  host-side script that *sends* signals to the Pi; the receiving/GPIO-control
  script was described in prose (Sections 4.2.5, 4.3.3) but never included as
  code. Replace this file with your actual script, or verify every line
  against your hardware before relying on it.
- `training/train_yolov8_colab.py` was written as sequential Google Colab
  notebook cells, not a standalone script. The `!pip install` line and
  `google.colab` import only work inside Colab.
- `preprocessing/augment_images.py` has hardcoded Windows paths and a
  single defect-class name ("Oil Strain") - edit both before reuse.
- Confidence threshold for deployed inference is 0.60 throughout
  (see manuscript Section 3.5 / 4.6 for the rationale).
- The reported 11.40 ± 0.13 ms latency figure is an **estimated budget**
  combining separately benchmarked pipeline stages, not a single live
  measurement on the physically loaded system. See
  `latency_benchmark/README.md` and manuscript Sections 3.6, 4.3, and 4.9
  before citing or reusing this number.

## Requirements

```
ultralytics
opencv-python
torch
numpy
pyyaml
```

Raspberry Pi side additionally requires:
```
gpiozero
```

## Citation

If you use this code, please cite the associated dataset and manuscript
(see the paper's Data availability / Code availability statements).

## License

Add a LICENSE file (e.g. MIT or CC-BY-4.0) before publishing - Zenodo
reads this automatically when archiving via GitHub.

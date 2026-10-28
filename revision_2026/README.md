# Revision code (2026)

Code added during the revision of the manuscript. The original training and deployment code is in the
main folder of this repository and is unchanged.

All notebooks run on Google Colab (GPU runtime) with `ultralytics==8.4.183`; set the Drive paths in the first cells.

| Folder | What it does | Paper |
|---|---|---|
| `1_validation_and_training/` | Re-validates the deployed YOLOv8n at the 640 input; trains YOLOv10n, YOLO11n and the resolution variants on the original split | Sections 4.1, 4.2, 4.7 |
| `2_cross_validation/` | Grouped five-fold cross-validation (YOLOv8n, YOLOv10n), fold assignment, summary statistics, class-specific thresholds selected out of sample, threshold analysis | Sections 3.6, 4.1, 4.6, 4.7; Tables 6, 8 |
| `3_preprocessing/` | CLAHE and AGCWD under the same five folds | Section 4.5, Table 5 |
| `4_supplementary_trials/` | Recording script (laptop), detection on Colab, analysis that reproduces the supplementary-trial results | Sections 3.6, 4.4, 4.6; Table 7 |


## Supplementary trials (folder 4)

1. `record_trial.py` - run on the laptop connected to the inspection camera; records 640 × 480 video with a
   timestamp for every frame. Needs only `pip install opencv-python`.
2. `NB6_trials_colab.py` - run on Colab; applies the deployed model to every frame and every defect-free
   image and saves all boxes with confidence ≥ 0.25.
3. `analyze_trials.py` - `python analyze_trials.py path/to/nb6` prints the false-stop counts, the defect-free
   image results and Table 7, and writes `trial_results.json`.
   `E2_defect_labels.csv` gives the order and video time of each defect;
   `class_thresholds_per_fold.json` gives the class-specific thresholds from the cross-validation.

The recordings and detection logs are in the archived dataset (see the paper's Data availability statement).

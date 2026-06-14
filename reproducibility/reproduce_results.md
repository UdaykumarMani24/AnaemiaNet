# Reproduce Results

Step 1
Prepare CP-AnemiC dataset.

Step 2

python preprocessing/preprocess_images.py

Step 3

python training/train_phase1.py

Step 4

python training/train_phase2.py

Step 5

python evaluate_model.py

Step 6

python export_tflite.py

Expected outputs:

AUC ≈ 0.89
Sensitivity ≈ 93%
Specificity ≈ 81%

Hardware:
Single GPU recommended.

Random seed:
42

Notes:
Deployment APK and trained production weights are excluded.

# Training

Phase 1:
Train classification head with frozen EfficientNetB0 backbone.

Run:

python train_phase1.py

Phase 2:
Selective fine-tuning of upper layers.

Run:

python train_phase2.py

Outputs:

* trained checkpoints
* training curves
* validation metrics
* final export model

Early stopping is based on validation AUC.

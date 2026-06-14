# AnaemiaNet

AnaemiaNet: Offline Smartphone-Based Screening of Paediatric Anaemia Using Conjunctival Imaging

## Overview

AnaemiaNet is a deep learning framework for non-invasive screening of paediatric anaemia from lower palpebral conjunctival photographs.

This repository contains the code required to reproduce model training, evaluation, preprocessing, and TensorFlow Lite export described in the associated manuscript.

The deployment-oriented Android application and trained production assets are not included in this public release.

## Features

* EfficientNetB0 backbone
* Two-stage optimisation
* ROC threshold optimisation
* Grad-CAM explainability
* TensorFlow Lite export
* Offline deployment workflow

## Dataset

This study uses the publicly available CP-AnemiC dataset.

Dataset access remains governed by the original dataset providers.

No patient images are redistributed.

## Repository Structure

training/
preprocessing/
evaluate_model.py
export_tflite.py
reproducibility/

## Installation

pip install -r requirements.txt

## Training

Phase 1:

python training/train_phase1.py

Phase 2:

python training/train_phase2.py

## Evaluation

python evaluate_model.py

## Export

python export_tflite.py

## Reproducibility

See:

reproducibility/reproduce_results.md

## Citation

If using this work, please cite the associated manuscript.

## Disclaimer

Research use only.

Not intended for clinical diagnosis.

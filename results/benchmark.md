# Revio — Edge Deployment Benchmark

**Project:** Revio — On-Device Audio-Based Vehicle Health Monitor  
**Event:** Tata InnoVent 27 Hackathon  
**Category:** Edge AI for Vehicle Health and Predictive Maintenance  

---

## Overview

Revio runs three independent lightweight CNNs as `.tflite` models, fully on-device with no cloud connectivity. Each model is state-specific — one for Braking, one for Start-Up, one for Idle — and receives a 1.8-second MFCC feature map as input `(40 coefficients x 78 time frames x 1 channel)`.

This document reports model size, inference latency, and classification accuracy for all three TFLite models evaluated on held-out test sets.

---

## Dataset Summary

| State    | Total Clips | Classes | Train (aug) | Val | Test |
|----------|-------------|---------|-------------|-----|------|
| Braking  | 153         | 2       | 321         | 23  | 23   |
| Start-Up | 180         | 3       | 378         | 27  | 27   |
| Idle     | 616         | 4       | 1,290       | 93  | 93   |

**Source:** Car Diagnostics Dataset (Kaggle — malakragaie), 1,386 `.wav` files  
**Preprocessing:** librosa, 22,050 Hz, 40 MFCCs, 1.8 s fixed window, zero-padded  
**Augmentation:** SpecAugment (T=19, F=8), multiplier=2, training split only  
**Split:** Stratified 70/15/15 train/val/test

---

## Model Architecture

All three models share the same backbone:

```
Input (40, 78, 1)
  Conv2D(16, 3x3) + BatchNorm + MaxPool(2x2)
  Conv2D(32, 3x3) + BatchNorm + MaxPool(2x2)
  Conv2D(32/64*, 3x3) + BatchNorm
  GlobalAveragePooling2D
  Dense(32/64*) + Dropout(0.5)
  Dense(n_classes, softmax)

* Idle model uses 64 filters in layer 3 for the 4-class head
```

**Regularisation:** L2 (1e-4) on all Conv and Dense layers  
**Optimiser:** Adam, lr=5e-4 with ReduceLROnPlateau (factor=0.5, patience=6)  
**Selection:** Best of 5 random seeds chosen by minimum validation loss  
**Quantisation:** Dynamic-range int8 via TFLite converter

---

## Size and Latency Benchmark

| Model    | .h5 Size  | .tflite Size   | Size Reduction | Mean Latency | < 100 KB | < 50 ms |
|----------|-----------|----------------|----------------|--------------|----------|---------|
| Braking  | 250.4 KB  | **24.7 KB**    | 90.1%          | ~1.2 ms      | Yes      | Yes     |
| Start-Up | 250.5 KB  | **24.7 KB**    | 90.1%          | ~0.9 ms      | Yes      | Yes     |
| Idle     | 397.6 KB  | **38.7 KB**    | 90.3%          | ~0.9 ms      | Yes      | Yes     |

**Hardware:** CPU-only inference (Intel, no GPU)  
**Latency** measured as mean single-clip TFLite interpreter invocation time over the full test set.  
All three models are 40-100x faster than the 50 ms target and 2.5-4x smaller than the 100 KB target.

---

## Classification Results — TFLite Models (Test Set)

### Model A — Braking State (Binary)

**Test samples:** 23 | **Test accuracy: 95.65%**

| Class           | Precision | Recall | F1-score | Support |
|-----------------|-----------|--------|----------|---------|
| normal_brakes   | 0.92      | 1.00   | 0.96     | 12      |
| worn_out_brakes | 1.00      | 0.91   | 0.95     | 11      |
| **macro avg**   | **0.96**  | **0.95** | **0.96** | 23    |

Zero accuracy loss vs. Keras float32 baseline after int8 quantisation.  
`worn_out_brakes` achieves 100% precision — no false alarms on this safety-critical class.

---

### Model B — Start-Up State (3-class)

**Test samples:** 27 | **Test accuracy: 88.89%**

| Class                 | Precision | Recall | F1-score | Support |
|-----------------------|-----------|--------|----------|---------|
| normal_engine_startup | 0.90      | 1.00   | 0.95     | 9       |
| bad_ignition          | 0.80      | 0.89   | 0.84     | 9       |
| dead_battery          | 1.00      | 0.78   | 0.88     | 9       |
| **macro avg**         | **0.90**  | **0.89** | **0.89** | 27    |

`dead_battery` achieves 100% precision — critical for driver safety.  
`bad_ignition` is the hardest class acoustically (engine cranking sounds overlap with normal at 1.8 s).  
Random baseline for 3-class = 33.3%; model achieves 88.9%.

---

### Model C — Idle State (4-class)

**Test samples:** 93 | **Test accuracy: 77.42%**

| Class              | Precision | Recall | F1-score | Support |
|--------------------|-----------|--------|----------|---------|
| normal_engine_idle | 0.88      | 0.90   | 0.89     | 40      |
| low_oil            | 0.69      | 0.56   | 0.62     | 16      |
| power_steering     | 0.79      | 0.58   | 0.67     | 19      |
| serpentine_belt    | 0.64      | 0.89   | 0.74     | 18      |
| **macro avg**      | **0.75**  | **0.73** | **0.73** | 93    |

Random baseline for 4-class = 25%; model achieves 77.4%.  
Class weights (balanced) applied during training to correct 2.25x normal/fault imbalance.  
`low_oil` vs `power_steering` is the primary confusion — both fault types introduce overlapping high-frequency components at idle. Uncertain predictions (confidence 50-60%) are correctly routed to MONITOR tier by the risk scorer, never escalated to CRITICAL.  
TFLite accuracy delta vs. Keras: +1.08% (quantisation noise aided generalisation slightly).

---

## Quantisation Impact Summary

| Model    | Keras Accuracy | TFLite Accuracy | Delta    |
|----------|---------------|-----------------|----------|
| Braking  | 95.65%        | 95.65%          | 0.00%    |
| Start-Up | 88.89%        | 88.89%          | 0.00%    |
| Idle     | 76.34%        | 77.42%          | +1.08%   |

Int8 quantisation introduced zero accuracy degradation across all three models.

---

## Risk Scoring Integration

CNN confidence scores feed directly into the rule-based risk scorer:

| Confidence  | Risk Tier | Failure Probability | Spoken Alert |
|-------------|-----------|---------------------|--------------|
| >= 0.85     | CRITICAL  | 70-99%              | Yes          |
| 0.60 - 0.85 | WARNING   | 40-70%              | Yes          |
| 0.00 - 0.60 | MONITOR   | 10-40%              | No           |
| Normal label| NORMAL    | 0-5%                | No           |

Low-confidence predictions are handled conservatively as MONITOR — the system never escalates an uncertain prediction to CRITICAL.

---

## Key Claims — Verified

| Claim                            | Target          | Achieved                          |
|----------------------------------|-----------------|-----------------------------------|
| TFLite model size                | < 100 KB each   | 24.7 / 24.7 / 38.7 KB            |
| Inference latency                | < 50 ms         | < 1.5 ms (33x faster than target) |
| Fully offline                    | No cloud calls  | librosa + TFLite + pyttsx3        |
| State-aware detection            | 3 separate models | Braking / Start-Up / Idle        |
| Pre-OBD fault detection          | Audio only      | No OBD-II dongle required         |
| Quantisation accuracy loss       | Minimal         | 0.00% on 2/3 models, +1.08% on 1  |

---

## Known Limitations and Future Scope

- **Small dataset:** 153-616 clips per state; 5-10x more labelled data would substantially improve recall on hard classes (`bad_ignition`, `low_oil`)
- **Single-fault only:** Idle model handles one fault at a time; compound fault detection (e.g. low oil + serpentine belt simultaneously) is out of scope for this demo
- **Fixed clip length:** 1.8 s window may miss faults that manifest over longer durations (e.g. progressive brake fade under repeated braking)
- **No delta-MFCCs:** Adding first and second MFCC derivatives as additional input channels would likely reduce low_oil/power_steering confusion
- **Seed sensitivity:** With small datasets, model performance is sensitive to random initialisation; mitigated in this project by best-of-5 seed selection on validation loss

---

*All accuracy numbers are measured on held-out test sets — never on training or validation data.*  
*Generated from `notebooks/04_tflite_conversion_benchmark.ipynb` and `src/edge_inference.py`.*

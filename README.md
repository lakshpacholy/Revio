# Revio — On-Device Audio-Based Vehicle Health Monitor

Revio listens to your car and tells you what's wrong with it — entirely on-device, with no cloud connectivity, no OBD-II dongle, and no internet connection required.

Built for the **Tata InnoVent 27 Hackathon** (Edge AI for Vehicle Health and Predictive Maintenance).

## What it does

Revio uses three lightweight, independently-trained CNNs to classify vehicle sounds captured at three distinct driving states, then turns the result into a spoken, driver-friendly alert:

```
Audio clip → MFCC extraction → TFLite CNN → Risk Scorer → Narrative → Text-to-Speech
```

| State | Model detects |
|---|---|
| **Braking** | Normal brakes vs. worn-out brake pads |
| **Start-Up** | Normal engine startup vs. bad ignition vs. dead battery |
| **Idle** | Normal idle vs. low oil vs. power steering fault vs. serpentine belt fault |

Each prediction's confidence score feeds a rule-based risk scorer that assigns a tier — **NORMAL**, **MONITOR**, **WARNING**, or **CRITICAL** — and a failure probability. WARNING and CRITICAL tiers trigger a spoken alert via offline text-to-speech; low-confidence predictions are always routed conservatively to MONITOR and never escalated.

## Why it's edge-first

- **Fully offline:** librosa for feature extraction, TFLite for inference, pyttsx3 for speech — no API calls, ever.
- **Tiny models:** each `.tflite` model is under 40 KB (90%+ smaller than the original Keras `.h5`), with zero accuracy loss from int8 quantisation.
- **Fast:** mean inference latency of ~1 ms per clip on CPU — 30-100x faster than a 50 ms target.
- **Pre-OBD detection:** flags faults from sound alone, before they'd trip an OBD-II code.

See [`results/benchmark.md`](results/benchmark.md) for full accuracy, latency, and size numbers per model, plus known limitations.

## Repository structure

```
Revio/
├── demo/               Streamlit demo app + sample .wav clips for each fault class
├── models/             Trained Keras (.h5) and quantised TFLite (.tflite) models
├── notebooks/          EDA, preprocessing/augmentation, training, and TFLite benchmarking
├── results/            Benchmark report, confusion matrices, training curves, plots
└── src/                Core pipeline: preprocessing, augmentation, training, inference,
                         risk scoring, narrative generation, TTS alerts
```

Key modules in `src/`:

- `preprocess.py` — loads audio, extracts 40-coefficient MFCCs over a fixed 1.8 s window
- `augment.py` — SpecAugment-based augmentation for the training split
- `train_braking.py` / `train_startup.py` / `train_idle.py` — per-state CNN training scripts
- `edge_inference.py` — converts `.h5` → `.tflite` and runs quantised inference
- `risk_scorer.py` — maps (fault label, confidence) → risk tier + failure probability
- `narrative.py` — template-based, offline alert text generation (no LLM)
- `tts_alert.py` — offline speech playback via `pyttsx3`

## Dataset

Trained on the [Car Diagnostics Dataset](https://www.kaggle.com/) (Kaggle, malakragaie) — 1,386 labelled `.wav` clips across braking, start-up, and idle states. Audio is resampled to 22,050 Hz, converted to 40 MFCCs, and fixed to a 1.8-second window (zero-padded where needed). See `notebooks/01_eda.ipynb` and `notebooks/02_preprocessing_augmentation.ipynb` for the full pipeline.

## Model results (test set)

| Model | Classes | Test Accuracy | TFLite Size | Mean Latency |
|---|---|---|---|---|
| Braking | 2 | 95.65% | 24.7 KB | ~1.2 ms |
| Start-Up | 3 | 88.89% | 24.7 KB | ~0.9 ms |
| Idle | 4 | 77.42% | 38.7 KB | ~0.9 ms |

Full per-class precision/recall, confusion matrices, and quantisation impact are in [`results/benchmark.md`](results/benchmark.md).

## Running the demo

The demo is a Streamlit app that runs the full pipeline end-to-end on sample or uploaded audio clips.

```bash
pip install streamlit tensorflow librosa numpy pyttsx3 matplotlib
streamlit run demo/app.py
```

Sample clips covering every fault class are provided in `demo/demo_clips/`.

## Known limitations

- Small dataset (153-616 clips per state) — more labelled data would improve recall on the hardest classes (`bad_ignition`, `low_oil`)
- Idle model handles a single fault at a time; compound faults are out of scope
- Fixed 1.8 s clip window may miss faults that develop over longer durations

See [`results/benchmark.md`](results/benchmark.md) for the complete list and future scope.

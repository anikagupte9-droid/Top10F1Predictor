@'
# F1Predictor

### Lightweight Formula 1 Top-10 Prediction with Edge AI Deployment

F1Predictor is a lightweight machine learning system that predicts whether a Formula 1 driver will finish inside the Top 10 using practice and qualifying performance.

The project follows a complete edge-AI pipeline:

OpenF1 → Data Processing → Tiny Neural Network → INT8 Quantization → TFLite → TFLite Micro → C++ Inference

---

## Overview

The model uses four features available before the race:

- FP1 position
- FP2 position
- FP3 position
- Qualifying position

The target is binary:

- `1` → Top-10 race finish
- `0` → Outside Top 10

The project was designed around a small neural network suitable for deployment on resource-constrained devices.

---

## Architecture

```text
                    OpenF1 API
                        │
                        ▼
              Race Session Data
                        │
                        ▼
              Data Preprocessing
                        │
                        ▼
              StandardScaler
                        │
                        ▼
             ┌──────────────────┐
             │   TinyNN Model   │
             │                  │
             │  4 → 16 → 8 → 1 │
             └──────────────────┘
                        │
                        ▼
                 Full INT8
                Quantization
                        │
                        ▼
              f1_top10_full_int8
                     .tflite
                        │
                        ▼
                 TFLite Micro
                        │
                        ▼
                 C++ Inference
                        │
                        ▼
              TOP 10 / OUTSIDE TOP 10
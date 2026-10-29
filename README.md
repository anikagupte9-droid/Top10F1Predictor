# F1Predictor

### Lightweight Formula 1 Top-10 Prediction with Edge AI Deployment

F1Predictor is a lightweight machine learning system that predicts whether a Formula 1 driver will finish inside the Top 10 using practice and qualifying performance.

The project follows a complete edge-AI deployment pipeline:

**OpenF1 → Data Processing → Tiny Neural Network → INT8 Quantization → TFLite → TFLite Micro → C++ Inference → Android**

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

### Model

```text
Input: 4 features

        ↓

Dense Layer
4 → 16
ReLU

        ↓

Dense Layer
16 → 8
ReLU

        ↓

Output Layer
8 → 1
Sigmoid

        ↓

Top-10 Probability
```

The network contains **225 trainable parameters**.

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
                    │     TinyNN       │
                    │                  │
                    │   4 → 16 → 8 → 1│
                    └──────────────────┘
                             │
                             ▼
                       FP32 Model
                             │
                             ▼
                    Full INT8 Quantization
                             │
                             ▼
                  f1_top10_full_int8.tflite
                             │
                 ┌───────────┴───────────┐
                 ▼                       ▼
          TFLite Micro              Android
                 │                       │
                 ▼                       ▼
          C++ Inference          TensorFlow Lite
                 │                       │
                 ▼                       ▼
       Python/C++ Validation      On-Device Inference
                                         │
                                         ▼
                                Samsung Galaxy S24 FE
                                         │
                                         ▼
                              TOP 10 / OUTSIDE TOP 10
```

---

## Dataset

The project uses Formula 1 session data retrieved through the OpenF1 API.

### Features

| Feature | Description |
|---|---|
| `FP1_Position` | Driver's position in Free Practice 1 |
| `FP2_Position` | Driver's position in Free Practice 2 |
| `FP3_Position` | Driver's position in Free Practice 3 |
| `Qualifying_Position` | Driver's qualifying position |

### Target

```text
Top10 = 1  → Race finish ≤ 10
Top10 = 0  → Race finish > 10
```

The V1 pipeline uses standard race weekends with FP1, FP2, FP3 and qualifying data. Sprint-weekend handling is planned for a future version.

---

## Data Processing

The preprocessing pipeline:

1. Retrieves race session data from OpenF1.
2. Combines practice and qualifying results.
3. Removes incomplete samples.
4. Creates the binary Top-10 target.
5. Fits `StandardScaler` using training data only.
6. Applies the trained scaler to validation and test data.
7. Produces the final datasets used for training and deployment.

### Dataset Split

The data is split chronologically to avoid using future race information during training.

| Dataset | Samples |
|---|---:|
| Training | 491 |
| Validation | 61 |
| Test | 278 |
| **Total complete samples** | **830** |

The test set consists of the **2025 season**.

---

## Training

The model is a small fully connected neural network:

```text
4 inputs
   ↓
Dense(16, ReLU)
   ↓
Dense(8, ReLU)
   ↓
Dense(1, Sigmoid)
```

Training configuration:

- Optimizer: Adam
- Loss: Binary Cross-Entropy
- Epochs: 50
- Batch size: 16
- Validation: chronological holdout
- Trainable parameters: **225**

The best validation loss was approximately **0.2979**, with validation accuracy of **93.44%**.

---

## Model Evaluation

The final Keras model was evaluated on the complete 2025 test set.

| Metric | Result |
|---|---:|
| Accuracy | **79.50%** |
| Precision | **82.89%** |
| Recall | **80.25%** |
| F1-score | **81.55%** |

### Confusion Matrix

```text
                  Predicted
                 0       1

Actual  0       95      26
        1       31     126
```

---

## INT8 Quantization

The trained model was converted to a fully quantized INT8 TensorFlow Lite model.

```text
FP32 Model
    │
    ▼
TensorFlow Lite Conversion
    │
    ▼
Full Integer Quantization
    │
    ▼
f1_top10_full_int8.tflite
```

The resulting model is approximately **3.24 KB**.

The fully quantized model uses:

- INT8 input
- INT8 weights
- INT8 intermediate tensors
- INT8 output

This significantly reduces the memory footprint compared with conventional floating-point deployment.

---

## TFLite Micro Deployment

The INT8 model was integrated with **TensorFlow Lite Micro** for lightweight C++ inference.

The deployment includes:

```text
tflite_micro/
├── config.h
├── f1_inference.cc
├── model_data.cc
└── model_data.h
```

The TFLite model is converted into a C/C++ byte array and executed using the TensorFlow Lite Micro interpreter.

### Example

Input:

```text
FP1 = 4
FP2 = 2
FP3 = 7
Qualifying = 3
```

Output:

```text
Top-10 probability ≈ 97.27%
Prediction = TOP 10
```

---

## Python vs C++ Deployment Validation

The C++ TFLite Micro implementation was validated against the Python INT8 inference pipeline using the complete 2025 test set.

| Metric | Result |
|---|---:|
| Test samples | 278 |
| Python INT8 accuracy | 78.42% |
| C++ TFLite Micro accuracy | 78.42% |
| Python/C++ prediction agreement | **100.00%** |
| Maximum probability difference | **0.0078125** |
| Mean probability difference | **0.00009836** |

**Result: 100% deployment agreement**

This verifies that the embedded C++ inference implementation reproduces the Python INT8 model behaviour.

---

## Android Deployment

The same fully quantized INT8 model was deployed to an Android application using TensorFlow Lite.

The Android application provides four input fields:

```text
FP1 Position
FP2 Position
FP3 Position
Qualifying Position
```

The application performs:

```text
User Input
    ↓
StandardScaler Transformation
    ↓
INT8 Quantization
    ↓
TFLite Inference
    ↓
INT8 Output Dequantization
    ↓
Top-10 Probability
    ↓
TOP 10 / OUTSIDE TOP 10
```

### Android Project

The Android implementation is located in:

```text
android/F1Predictor/
```

The model is bundled directly with the application:

```text
android/F1Predictor/
└── app/
    └── src/
        └── main/
            └── assets/
                └── f1_top10_int8.tflite
```

The application performs inference **locally on the device** and does not require a server for prediction.

---

## Real Device Test

The Android application was successfully tested on a **Samsung Galaxy S24 FE**.

Example input:

```text
FP1 = 5
FP2 = 6
FP3 = 3
Qualifying = 4
```

Result:

```text
Top-10 Probability
96.09%

Prediction
TOP 10
```

This demonstrates the complete transition from a trained machine learning model to real on-device inference.

---

## Project Structure

```text
Top10F1Predictor/
│
├── android/
│   └── F1Predictor/
│       ├── app/
│       │   ├── src/
│       │   │   └── main/
│       │   │       ├── assets/
│       │   │       │   └── f1_top10_int8.tflite
│       │   │       └── java/
│       │   │           └── com/anika/f1predictor/
│       │   │               └── MainActivity.kt
│       │   └── build.gradle.kts
│       ├── gradle/
│       ├── build.gradle.kts
│       ├── gradlew
│       └── settings.gradle.kts
│
├── data/
│   ├── raw datasets
│   └── processed datasets
│
├── models/
│   ├── FP32 models
│   ├── TFLite models
│   └── INT8 models
│
├── results/
│   └── evaluation and deployment validation
│
├── src/
│   ├── data.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── export.py
│   └── verification scripts
│
├── tflite_micro/
│   ├── config.h
│   ├── f1_inference.cc
│   ├── model_data.cc
│   └── model_data.h
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Edge AI Pipeline

The complete deployment workflow is:

```text
          FORMULA 1 DATA
                │
                ▼
            OpenF1 API
                │
                ▼
        DATA PREPROCESSING
                │
                ▼
          MODEL TRAINING
                │
                ▼
            TinyNN
         4 → 16 → 8 → 1
                │
                ▼
         INT8 QUANTIZATION
                │
        ┌───────┴────────┐
        ▼                ▼
   TFLite Micro       Android
        │                │
        ▼                ▼
      C++          TensorFlow Lite
        │                │
        └───────┬────────┘
                ▼
         ON-DEVICE INFERENCE
                │
                ▼
        TOP 10 PREDICTION
```

---

## Technologies

### Machine Learning

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy

### Data

- OpenF1 API

### Edge AI

- TensorFlow Lite
- INT8 Quantization
- TensorFlow Lite Micro
- C++

### Android

- Android Studio
- Kotlin
- Jetpack Compose
- TensorFlow Lite

### Development

- Git
- GitHub
- VS Code

---

## Key Results

| Component | Result |
|---|---|
| Neural network | `4 → 16 → 8 → 1` |
| Trainable parameters | **225** |
| Validation accuracy | **93.44%** |
| 2025 test accuracy | **79.50%** |
| 2025 F1-score | **81.55%** |
| INT8 model size | **~3.24 KB** |
| TFLite Micro Python/C++ agreement | **100%** |
| Android deployment | **Successful** |
| Physical device | **Samsung Galaxy S24 FE** |

---

## Future Work

### V2 — Sprint-Aware Prediction

The current V1 pipeline focuses on conventional race weekends. Future development will add sprint-weekend-aware feature engineering so that races without the standard FP2/FP3 structure can also be incorporated.

Potential extensions include:

- Sprint-session feature engineering
- Additional qualifying and race-weekend features
- More seasons of training data
- Driver/team contextual features
- Further model compression
- Hardware-specific Edge AI benchmarking

---

## License

This project is intended for educational and research purposes.

import os
import numpy as np
import pandas as pd
import tensorflow as tf


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "models/f1_top10.keras"

FP32_PATH = "models/f1_top10_fp32.tflite"
INT8_PATH = "models/f1_top10_int8.tflite"

TRAIN_PATH = "data/processed/train.csv"

FEATURES = [
    "FP1_Position",
    "FP2_Position",
    "FP3_Position",
    "Qualifying_Position",
]


# --------------------------------------------------
# Representative dataset
# --------------------------------------------------

def representative_dataset():

    train_df = pd.read_csv(TRAIN_PATH)

    X_train = train_df[FEATURES].values.astype(np.float32)

    # A representative subset is enough for calibration.
    sample_count = min(100, len(X_train))

    for sample in X_train[:sample_count]:

        sample = sample.reshape(1, 4)

        yield [sample]


# --------------------------------------------------
# Convert FP32 TFLite
# --------------------------------------------------

def convert_fp32():

    print("\nConverting FP32 TFLite model...")

    converter = tf.lite.TFLiteConverter.from_keras_model(
        tf.keras.models.load_model(MODEL_PATH)
    )

    tflite_model = converter.convert()

    with open(FP32_PATH, "wb") as f:
        f.write(tflite_model)

    print(f"Saved: {FP32_PATH}")


# --------------------------------------------------
# Convert INT8 TFLite
# --------------------------------------------------

def convert_int8():

    print("\nConverting INT8 TFLite model...")

    model = tf.keras.models.load_model(MODEL_PATH)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    converter.representative_dataset = representative_dataset

    # Force integer operations where supported.
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    tflite_model = converter.convert()

    with open(INT8_PATH, "wb") as f:
        f.write(tflite_model)

    print(f"Saved: {INT8_PATH}")

def convert_full_int8():

    print("\nConverting FULL INT8 TFLite model...")

    model = tf.keras.models.load_model(MODEL_PATH)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    converter.representative_dataset = representative_dataset

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    # Fully integer input/output
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    path = "models/f1_top10_full_int8.tflite"

    with open(path, "wb") as f:
        f.write(tflite_model)

    print(f"Saved: {path}")

# --------------------------------------------------
# File size
# --------------------------------------------------

def print_size(path):

    size_bytes = os.path.getsize(path)

    print(
        f"{os.path.basename(path):25s}"
        f"{size_bytes:8d} bytes "
        f"({size_bytes / 1024:.2f} KB)"
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("F1 Top-10 Model Export")
    print("======================")

    convert_fp32()
    convert_int8()
    convert_full_int8()

    print("\nModel sizes")
    print("-----------")

    print_size(FP32_PATH)
    print_size(INT8_PATH)

    print("\nExport complete.")


if __name__ == "__main__":
    main()
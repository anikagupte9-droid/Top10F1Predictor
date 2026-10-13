import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TEST_PATH = "data/processed/test.csv"

KERAS_PATH = "models/f1_top10.keras"
FP32_PATH = "models/f1_top10_fp32.tflite"
INT8_PATH = "models/f1_top10_int8.tflite"

FEATURES = [
    "FP1_Position",
    "FP2_Position",
    "FP3_Position",
    "Qualifying_Position",
]

TARGET = "Top10"


# --------------------------------------------------
# TFLite inference
# --------------------------------------------------

def run_tflite(model_path, X):

    interpreter = tf.lite.Interpreter(
        model_path=model_path
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    predictions = []

    for sample in X:

        sample = sample.reshape(1, 4).astype(np.float32)

        interpreter.set_tensor(
            input_index,
            sample
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_index
        )

        predictions.append(
            float(output[0][0])
        )

    return np.array(predictions)


# --------------------------------------------------
# Compare models
# --------------------------------------------------

def compare_model(name, reference, candidate, y_test):

    max_difference = np.max(
        np.abs(reference - candidate)
    )

    mean_difference = np.mean(
        np.abs(reference - candidate)
    )

    reference_predictions = (
        reference >= 0.5
    ).astype(int)

    candidate_predictions = (
        candidate >= 0.5
    ).astype(int)

    agreement = np.mean(
        reference_predictions == candidate_predictions
    ) * 100

    accuracy = accuracy_score(
        y_test,
        candidate_predictions
    )

    print(f"\n{name}")
    print("-" * len(name))

    print(
        f"Maximum probability difference: "
        f"{max_difference:.8f}"
    )

    print(
        f"Mean probability difference:    "
        f"{mean_difference:.8f}"
    )

    print(
        f"Prediction agreement:            "
        f"{agreement:.2f}%"
    )

    print(
        f"2025 test accuracy:              "
        f"{accuracy:.4f}"
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("TFLite Model Verification")
    print("=========================")

    # --------------------------------------------------
    # Load test data
    # --------------------------------------------------

    test_df = pd.read_csv(TEST_PATH)

    X_test = test_df[
        FEATURES
    ].values.astype(np.float32)

    y_test = test_df[
        TARGET
    ].values.astype(np.int32)

    print(
        f"\nTest samples: {len(X_test)}"
    )

    # --------------------------------------------------
    # Keras reference predictions
    # --------------------------------------------------

    print("\nRunning Keras model...")

    keras_model = tf.keras.models.load_model(
        KERAS_PATH
    )

    keras_predictions = (
        keras_model.predict(
            X_test,
            verbose=0
        )
        .flatten()
    )

    keras_accuracy = accuracy_score(
        y_test,
        (keras_predictions >= 0.5).astype(int)
    )

    print(
        f"Keras 2025 accuracy: "
        f"{keras_accuracy:.4f}"
    )

    # --------------------------------------------------
    # FP32 TFLite
    # --------------------------------------------------

    print("\nRunning FP32 TFLite model...")

    fp32_predictions = run_tflite(
        FP32_PATH,
        X_test
    )

    compare_model(
        "Keras vs FP32 TFLite",
        keras_predictions,
        fp32_predictions,
        y_test
    )

    # --------------------------------------------------
    # INT8 TFLite
    # --------------------------------------------------

    print("\nRunning INT8 TFLite model...")

    int8_predictions = run_tflite(
        INT8_PATH,
        X_test
    )

    compare_model(
        "Keras vs INT8 TFLite",
        keras_predictions,
        int8_predictions,
        y_test
    )

    print("\n=========================")
    print("Verification complete.")
    print("=========================")


if __name__ == "__main__":
    main()
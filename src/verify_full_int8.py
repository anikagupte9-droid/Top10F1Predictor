import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score


TEST_PATH = "data/processed/test.csv"
KERAS_PATH = "models/f1_top10.keras"
INT8_PATH = "models/f1_top10_full_int8.tflite"

FEATURES = [
    "FP1_Position",
    "FP2_Position",
    "FP3_Position",
    "Qualifying_Position",
]

TARGET = "Top10"


def main():

    test_df = pd.read_csv(TEST_PATH)

    X_test = test_df[FEATURES].values.astype(np.float32)
    y_test = test_df[TARGET].values.astype(np.int32)

    # ---------------------------------------------
    # Keras reference
    # ---------------------------------------------

    model = tf.keras.models.load_model(KERAS_PATH)

    keras_probabilities = (
        model.predict(
            X_test,
            verbose=0
        )
        .flatten()
    )

    keras_predictions = (
        keras_probabilities >= 0.5
    ).astype(int)

    # ---------------------------------------------
    # Load INT8 TFLite
    # ---------------------------------------------

    interpreter = tf.lite.Interpreter(
        model_path=INT8_PATH
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_scale, input_zero = input_details["quantization"]
    output_scale, output_zero = output_details["quantization"]

    print("Full INT8 Verification")
    print("======================")

    print("\nInput quantization")
    print("------------------")
    print(f"Scale:      {input_scale}")
    print(f"Zero point: {input_zero}")

    print("\nOutput quantization")
    print("-------------------")
    print(f"Scale:      {output_scale}")
    print(f"Zero point: {output_zero}")

    int8_probabilities = []

    # ---------------------------------------------
    # INT8 inference
    # ---------------------------------------------

    for sample in X_test:

        # Quantize float input
        quantized_input = np.round(
            sample / input_scale
        ) + input_zero

        quantized_input = np.clip(
            quantized_input,
            -128,
            127
        ).astype(np.int8)

        interpreter.set_tensor(
            input_details["index"],
            quantized_input.reshape(1, 4)
        )

        interpreter.invoke()

        quantized_output = interpreter.get_tensor(
            output_details["index"]
        )

        # Dequantize output
        probability = (
            quantized_output.astype(np.float32)
            - output_zero
        ) * output_scale

        int8_probabilities.append(
            float(probability[0][0])
        )

    int8_probabilities = np.array(
        int8_probabilities
    )

    int8_predictions = (
        int8_probabilities >= 0.5
    ).astype(int)

    # ---------------------------------------------
    # Comparison
    # ---------------------------------------------

    max_difference = np.max(
        np.abs(
            keras_probabilities -
            int8_probabilities
        )
    )

    mean_difference = np.mean(
        np.abs(
            keras_probabilities -
            int8_probabilities
        )
    )

    agreement = np.mean(
        keras_predictions ==
        int8_predictions
    ) * 100

    keras_accuracy = accuracy_score(
        y_test,
        keras_predictions
    )

    int8_accuracy = accuracy_score(
        y_test,
        int8_predictions
    )

    print("\nResults")
    print("-------")

    print(
        f"Keras accuracy:              "
        f"{keras_accuracy:.4f}"
    )

    print(
        f"Full INT8 accuracy:          "
        f"{int8_accuracy:.4f}"
    )

    print(
        f"Maximum probability diff:    "
        f"{max_difference:.6f}"
    )

    print(
        f"Mean probability diff:       "
        f"{mean_difference:.6f}"
    )

    print(
        f"Prediction agreement:        "
        f"{agreement:.2f}%"
    )


if __name__ == "__main__":
    main()
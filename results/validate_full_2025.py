import pandas as pd
import subprocess
import numpy as np
import tensorflow as tf
import json

CSV = "results/cpp_verification_inputs_full.csv"
TEST = "data/processed/test.csv"
MODEL = "models/f1_top10_full_int8.tflite"
EXE = "tflite_micro/f1_inference.exe"

# Load the processed test set
df = pd.read_csv(TEST)

# Recover original race positions
with open("models/scaler.json") as f:
    scaler = json.load(f)

features = scaler["features"]
mean = np.array(scaler["mean"], dtype=np.float32)
scale = np.array(scaler["scale"], dtype=np.float32)

raw = df.copy()

for i, feature in enumerate(features):
    raw[feature] = df[feature] * scale[i] + mean[i]

raw.to_csv(CSV, index=False)

# Python TFLite
interpreter = tf.lite.Interpreter(model_path=MODEL)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_scale, input_zp = input_details["quantization"]
output_scale, output_zp = output_details["quantization"]

cpp_correct = 0
python_correct = 0
prediction_matches = 0

max_diff = 0.0
sum_diff = 0.0

for _, row in raw.iterrows():

    values = [
        int(row["FP1_Position"]),
        int(row["FP2_Position"]),
        int(row["FP3_Position"]),
        int(row["Qualifying_Position"])
    ]

    # ---------- Python TFLite ----------
    raw_input = np.array([values], dtype=np.float32)
    scaled = (raw_input - mean) / scale

    quantized = np.round(
        scaled / input_scale + input_zp
    ).clip(-128, 127).astype(np.int8)

    interpreter.set_tensor(
        input_details["index"],
        quantized
    )

    interpreter.invoke()

    q_output = interpreter.get_tensor(
        output_details["index"]
    )[0][0]

    python_probability = (
        float(q_output) - output_zp
    ) * output_scale

    python_prediction = int(python_probability >= 0.5)

    # ---------- C++ TFLite Micro ----------
    args = [
        EXE,
        str(values[0]),
        str(values[1]),
        str(values[2]),
        str(values[3])
    ]

    result = subprocess.check_output(
        args,
        text=True
    ).strip().split()

    cpp_probability = float(result[0])
    cpp_prediction = int(result[1])

    actual = int(row["Top10"])

    # Metrics
    python_correct += python_prediction == actual
    cpp_correct += cpp_prediction == actual

    if python_prediction == cpp_prediction:
        prediction_matches += 1

    diff = abs(python_probability - cpp_probability)

    max_diff = max(max_diff, diff)
    sum_diff += diff

print()
print("=" * 65)
print("FULL 2025 TFLITE MICRO VALIDATION")
print("=" * 65)

print(f"Test samples:              {len(raw)}")
print()
print(f"Python INT8 accuracy:      {python_correct / len(raw):.4f}")
print(f"C++ TFLite Micro accuracy: {cpp_correct / len(raw):.4f}")
print()
print(f"Python/C++ agreement:      {prediction_matches / len(raw):.4%}")
print(f"Maximum probability diff:  {max_diff:.8f}")
print(f"Mean probability diff:     {sum_diff / len(raw):.8f}")

print("=" * 65)

if prediction_matches == len(raw):
    print("RESULT: 100% DEPLOYMENT AGREEMENT")
else:
    print(
        f"RESULT: {prediction_matches}/{len(raw)} "
        "predictions agree"
    )

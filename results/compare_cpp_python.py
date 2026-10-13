import pandas as pd
import numpy as np
import subprocess
import tensorflow as tf
import json

CSV = "results/cpp_verification_inputs.csv"
MODEL = "models/f1_top10_full_int8.tflite"
EXE = "tflite_micro/f1_inference.exe"

df = pd.read_csv(CSV)

# Load scaler
with open("models/scaler.json") as f:
    scaler = json.load(f)

features = scaler["features"]
mean = np.array(scaler["mean"], dtype=np.float32)
scale = np.array(scaler["scale"], dtype=np.float32)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_scale, input_zp = input_details["quantization"]
output_scale, output_zp = output_details["quantization"]

print("Python TFLite vs TFLite Micro")
print("=" * 85)
print("Input                    Python Prob    C++ Prob      Diff       Match")
print("-" * 85)

max_diff = 0.0
all_match = True

for _, row in df.iterrows():

    raw = np.array(
        [[row[f] for f in features]],
        dtype=np.float32
    )

    # StandardScaler
    scaled = (raw - mean) / scale

    # INT8 quantization
    quantized = np.round(
        scaled / input_scale + input_zp
    ).clip(-128, 127).astype(np.int8)

    interpreter.set_tensor(
        input_details["index"],
        quantized
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details["index"]
    )[0][0]

    python_probability = (
        float(output) - output_zp
    ) * output_scale

    python_prediction = python_probability >= 0.5

    # Run C++ TFLite Micro
    args = [
        EXE,
        str(int(row["FP1_Position"])),
        str(int(row["FP2_Position"])),
        str(int(row["FP3_Position"])),
        str(int(row["Qualifying_Position"]))
    ]

    result = subprocess.check_output(
        args,
        text=True
    ).strip().split()

    cpp_probability = float(result[0])
    cpp_prediction = bool(int(result[1]))

    diff = abs(python_probability - cpp_probability)
    max_diff = max(max_diff, diff)

    match = python_prediction == cpp_prediction

    if not match:
        all_match = False

    print(
        f"({args[1]},{args[2]},{args[3]},{args[4]})"
        f"{python_probability:>15.6f}"
        f"{cpp_probability:>14.6f}"
        f"{diff:>12.6f}"
        f"{'YES' if match else 'NO':>10}"
    )

print("-" * 85)
print(f"Maximum probability difference: {max_diff:.8f}")
print(f"Prediction agreement: {'100%' if all_match else 'NOT 100%'}")

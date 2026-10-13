import pandas as pd
import subprocess

df = pd.read_csv("results/cpp_verification_inputs.csv")
exe = "tflite_micro/f1_inference.exe"

print("Input              Actual   C++ Probability   C++ Prediction")
print("-" * 70)

for _, row in df.iterrows():
    args = [
        exe,
        str(int(row["FP1_Position"])),
        str(int(row["FP2_Position"])),
        str(int(row["FP3_Position"])),
        str(int(row["Qualifying_Position"]))
    ]

    output = subprocess.check_output(args, text=True).strip().split()

    probability = float(output[0])
    prediction = int(output[1])

    actual = int(row["Top10"])

    print(
        f"({args[1]}, {args[2]}, {args[3]}, {args[4]})"
        f"   {actual:^6}"
        f"   {probability:^17.6f}"
        f"   {'TOP 10' if prediction else 'OUTSIDE TOP 10'}"
    )

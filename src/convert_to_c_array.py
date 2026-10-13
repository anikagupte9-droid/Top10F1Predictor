import os


MODEL_PATH = "models/f1_top10_full_int8.tflite"

OUTPUT_DIR = "tflite_micro"

HEADER_PATH = os.path.join(
    OUTPUT_DIR,
    "model_data.h"
)

SOURCE_PATH = os.path.join(
    OUTPUT_DIR,
    "model_data.cc"
)

ARRAY_NAME = "f1_model"


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(MODEL_PATH, "rb") as f:
        model_data = f.read()

    print("TFLite → C++ Model Conversion")
    print("=============================")

    print(f"Model: {MODEL_PATH}")
    print(f"Model size: {len(model_data)} bytes")

    # --------------------------------------------------
    # Header
    # --------------------------------------------------

    with open(HEADER_PATH, "w") as f:

        f.write(
            "#ifndef F1_MODEL_DATA_H\n"
            "#define F1_MODEL_DATA_H\n\n"
            "#include <cstdint>\n"
            "\n"
            f"extern const unsigned char {ARRAY_NAME}[];\n"
            f"extern const unsigned int {ARRAY_NAME}_len;\n"
            "\n"
            "#endif\n"
        )

    # --------------------------------------------------
    # Source
    # --------------------------------------------------

    with open(SOURCE_PATH, "w") as f:

        f.write(
            '#include "model_data.h"\n\n'
        )

        f.write(
            f"const unsigned char {ARRAY_NAME}[] = {{\n"
        )

        for i in range(0, len(model_data), 12):

            chunk = model_data[i:i + 12]

            values = ", ".join(
                f"0x{byte:02x}"
                for byte in chunk
            )

            f.write(
                f"    {values},\n"
            )

        f.write("};\n\n")

        f.write(
            f"const unsigned int {ARRAY_NAME}_len = "
            f"{len(model_data)};\n"
        )

    print("\nGenerated:")
    print(HEADER_PATH)
    print(SOURCE_PATH)

    print("\nConversion complete.")


if __name__ == "__main__":
    main()
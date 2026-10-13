import tensorflow as tf

MODEL_PATH = "models/f1_top10_full_int8.tflite"

def main():

    interpreter = tf.lite.Interpreter(
        model_path=MODEL_PATH
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("TFLite Model Inspection")
    print("=======================")

    print("\nINPUT")
    print("-----")

    for tensor in input_details:
        print("Name:       ", tensor["name"])
        print("Shape:      ", tensor["shape"])
        print("dtype:      ", tensor["dtype"])
        print("quantization:",
              tensor["quantization"])

    print("\nOUTPUT")
    print("------")

    for tensor in output_details:
        print("Name:       ", tensor["name"])
        print("Shape:      ", tensor["shape"])
        print("dtype:      ", tensor["dtype"])
        print("quantization:",
              tensor["quantization"])

    print("\nINTERMEDIATE TENSORS")
    print("--------------------")

    tensors = interpreter.get_tensor_details()

    for tensor in tensors:

        scale, zero_point = tensor["quantization"]

        if scale != 0:

            print(
                f'{tensor["name"]}: '
                f'dtype={tensor["dtype"]}, '
                f'scale={scale}, '
                f'zero_point={zero_point}'
            )


if __name__ == "__main__":
    main()
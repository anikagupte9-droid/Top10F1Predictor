#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

#include "config.h"
#include "model_data.h"

static constexpr float INPUT_SCALE = 0.01327837910503149f;
static constexpr int INPUT_ZERO_POINT = -8;

static constexpr float OUTPUT_SCALE = 0.00390625f;
static constexpr int OUTPUT_ZERO_POINT = -128;

static constexpr int TENSOR_ARENA_SIZE = 8192;

static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

int main(int argc, char* argv[]) {

    float raw_input[F1_INPUT_FEATURES];

    // Automated mode:
    // ./f1_inference.exe FP1 FP2 FP3 Qualifying
    if (argc == 5) {
        for (int i = 0; i < F1_INPUT_FEATURES; ++i) {
            raw_input[i] = std::strtof(argv[i + 1], nullptr);
        }
    }
    else {
        std::printf("=== F1 TinyNN / TFLite Micro ===\n\n");

        std::printf("Enter FP1 position: ");
        if (std::scanf("%f", &raw_input[0]) != 1) return 1;

        std::printf("Enter FP2 position: ");
        if (std::scanf("%f", &raw_input[1]) != 1) return 1;

        std::printf("Enter FP3 position: ");
        if (std::scanf("%f", &raw_input[2]) != 1) return 1;

        std::printf("Enter Qualifying position: ");
        if (std::scanf("%f", &raw_input[3]) != 1) return 1;
    }

    // StandardScaler preprocessing.
    float scaled_input[F1_INPUT_FEATURES];

    for (int i = 0; i < F1_INPUT_FEATURES; ++i) {
        scaled_input[i] =
            (raw_input[i] - F1_MEAN[i]) / F1_SCALE[i];
    }

    // Quantize to INT8.
    int8_t input_data[F1_INPUT_FEATURES];

    for (int i = 0; i < F1_INPUT_FEATURES; ++i) {
        float quantized =
            scaled_input[i] / INPUT_SCALE + INPUT_ZERO_POINT;

        int value = static_cast<int>(std::round(quantized));

        if (value < -128) value = -128;
        if (value > 127) value = 127;

        input_data[i] = static_cast<int8_t>(value);
    }

    const tflite::Model* model =
        tflite::GetModel(f1_model);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        std::printf("Model schema mismatch.\n");
        return 1;
    }

    tflite::MicroMutableOpResolver<4> resolver;

    if (resolver.AddFullyConnected() != kTfLiteOk) {
        return 1;
    }

    if (resolver.AddLogistic() != kTfLiteOk) {
        return 1;
    }

    tflite::MicroInterpreter interpreter(
        model,
        resolver,
        tensor_arena,
        TENSOR_ARENA_SIZE);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        return 1;
    }

    TfLiteTensor* input = interpreter.input(0);

    if (input->type != kTfLiteInt8) {
        return 1;
    }

    for (int i = 0; i < F1_INPUT_FEATURES; ++i) {
        input->data.int8[i] = input_data[i];
    }

    if (interpreter.Invoke() != kTfLiteOk) {
        return 1;
    }

    TfLiteTensor* output = interpreter.output(0);

    if (output->type != kTfLiteInt8) {
        return 1;
    }

    const int8_t output_value = output->data.int8[0];

    const float probability =
        (static_cast<int>(output_value) - OUTPUT_ZERO_POINT)
        * OUTPUT_SCALE;

    const bool top10 = probability >= F1_THRESHOLD;

    // Machine-readable output for automated verification.
    if (argc == 5) {
        std::printf("%.8f %d\n",
                    probability,
                    top10 ? 1 : 0);
        return 0;
    }

    // Human-readable interactive output.
    std::printf("\n------------------------------\n");
    std::printf("FP1:          %.0f\n", raw_input[0]);
    std::printf("FP2:          %.0f\n", raw_input[1]);
    std::printf("FP3:          %.0f\n", raw_input[2]);
    std::printf("Qualifying:   %.0f\n", raw_input[3]);
    std::printf("------------------------------\n");

    std::printf(
        "Top-10 probability: %.2f%%\n",
        probability * 100.0f);

    std::printf(
        "Prediction: %s\n",
        top10 ? "TOP 10" : "OUTSIDE TOP 10");

    return 0;
}

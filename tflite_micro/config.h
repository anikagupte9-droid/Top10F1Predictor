#ifndef F1_MODEL_CONFIG_H
#define F1_MODEL_CONFIG_H

// --------------------------------------------------
// F1 Top-10 TinyNN
// --------------------------------------------------

#define F1_INPUT_FEATURES 4

// Feature order:
// 0 = FP1_Position
// 1 = FP2_Position
// 2 = FP3_Position
// 3 = Qualifying_Position

// StandardScaler parameters used during training

static constexpr float F1_MEAN[F1_INPUT_FEATURES] = {
    9.885947046843178f,
    10.081466395112017f,
    10.228105906313646f,
    9.963340122199593f
};

static constexpr float F1_SCALE[F1_INPUT_FEATURES] = {
    5.6611034012999815f,
    5.729251299830547f,
    5.775228394720603f,
    5.61300136851328f
};

// Classification threshold

static constexpr float F1_THRESHOLD = 0.5f;

#endif
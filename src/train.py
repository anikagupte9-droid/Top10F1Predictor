import os
import numpy as np
import pandas as pd
import tensorflow as tf

# --------------------------------------------------
# Configuration
# --------------------------------------------------

TRAIN_PATH = "data/processed/train.csv"
VAL_PATH = "data/processed/validation.csv"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "f1_top10.keras")

FEATURES = [
    "FP1_Position",
    "FP2_Position",
    "FP3_Position",
    "Qualifying_Position",
]

TARGET = "Top10"

EPOCHS = 50
BATCH_SIZE = 16
SEED = 42


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

np.random.seed(SEED)
tf.random.set_seed(SEED)


# --------------------------------------------------
# Load data
# --------------------------------------------------

def load_data():

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    X_train = train_df[FEATURES].values.astype(np.float32)
    y_train = train_df[TARGET].values.astype(np.float32)

    X_val = val_df[FEATURES].values.astype(np.float32)
    y_val = val_df[TARGET].values.astype(np.float32)

    return X_train, y_train, X_val, y_val


# --------------------------------------------------
# Dataset diagnostics
# --------------------------------------------------

def print_class_balance(name, y):

    values, counts = np.unique(y.astype(int), return_counts=True)

    print(f"\n{name} class balance")
    print("-------------------")

    for value, count in zip(values, counts):

        label = "Outside Top 10" if value == 0 else "Top 10"

        percentage = count / len(y) * 100

        print(
            f"{label:15s}: "
            f"{count:3d} ({percentage:.2f}%)"
        )


# --------------------------------------------------
# Build Tiny Neural Network
# --------------------------------------------------

def build_model():

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),

        tf.keras.layers.Dense(
            16,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            8,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# --------------------------------------------------
# Training
# --------------------------------------------------

def main():

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading processed datasets...")

    X_train, y_train, X_val, y_val = load_data()

    print(f"\nTraining samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Input features:     {len(FEATURES)}")

    print_class_balance("Training", y_train)
    print_class_balance("Validation", y_val)

    print("\nBuilding Tiny Neural Network...")

    model = build_model()

    model.summary()

    print("\nTraining...")
    print("----------------------------------------")

    # Save the model with the lowest validation loss
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1
    )

    history = model.fit(
        X_train,
        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=EPOCHS,
        batch_size=BATCH_SIZE,

        callbacks=[
            checkpoint
        ],

        verbose=1
    )

    print("\n----------------------------------------")
    print("Training complete.")

    print(f"\nBest model saved to:")
    print(MODEL_PATH)

    # --------------------------------------------------
    # Parameter count
    # --------------------------------------------------

    print("\nModel parameters:")
    print("-----------------")
    print(f"Trainable parameters: {model.count_params()}")

    parameter_memory = model.count_params() * 4

    print(
        f"Approx. FP32 parameter memory: "
        f"{parameter_memory} bytes "
        f"({parameter_memory / 1024:.2f} KB)"
    )


if __name__ == "__main__":
    main()
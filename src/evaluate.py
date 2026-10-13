import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

TEST_PATH = "data/processed/test.csv"
MODEL_PATH = "models/f1_top10.keras"

RESULTS_DIR = "results"
PREDICTIONS_PATH = os.path.join(
    RESULTS_DIR,
    "test_predictions.csv"
)

FEATURES = [
    "FP1_Position",
    "FP2_Position",
    "FP3_Position",
    "Qualifying_Position",
]

TARGET = "Top10"

THRESHOLD = 0.5


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading test data...")
    test_df = pd.read_csv(TEST_PATH)

    X_test = test_df[FEATURES].values.astype(np.float32)
    y_test = test_df[TARGET].values.astype(np.int32)

    print(f"Test samples: {len(test_df)}")

    # --------------------------------------------------
    # Load trained model
    # --------------------------------------------------

    print("\nLoading trained model...")

    model = tf.keras.models.load_model(MODEL_PATH)

    print(f"Model loaded: {MODEL_PATH}")

    # --------------------------------------------------
    # Generate predictions
    # --------------------------------------------------

    probabilities = model.predict(
        X_test,
        verbose=0
    ).flatten()

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print("\n========================================")
    print("FINAL TEST RESULTS — 2025")
    print("========================================")

    print(f"Accuracy :  {accuracy:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall   :  {recall:.4f}")
    print(f"F1-score :  {f1:.4f}")

    print("\nConfusion Matrix")
    print("----------------")
    print(cm)

    print("\nClassification Report")
    print("---------------------")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Outside Top 10",
                "Top 10"
            ],
            zero_division=0
        )
    )

    # --------------------------------------------------
    # Save predictions
    # --------------------------------------------------

    output_df = test_df.copy()

    output_df["Prediction_Probability"] = probabilities
    output_df["Predicted_Top10"] = predictions

    output_df.to_csv(
        PREDICTIONS_PATH,
        index=False
    )

    print("Predictions saved to:")
    print(PREDICTIONS_PATH)


if __name__ == "__main__":
    main()
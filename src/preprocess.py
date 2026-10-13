import pandas as pd
import joblib
import json

from pathlib import Path
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/f1_training_data.csv")
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")

FEATURES = [
    "FP1_Position",
    "FP2_Position",
    "FP3_Position",
    "Qualifying_Position",
]

TARGET = "Top10"


def load_data():
    df = pd.read_csv(DATA_PATH)

    print(f"Raw dataset: {len(df)} rows")

    # Remove rows where any required ML field is missing.
    df = df.dropna(
        subset=FEATURES + ["Race_Position", TARGET]
    ).copy()

    print(
        f"Complete samples: {len(df)} rows"
    )

    return df


def chronological_split(df):

    # -------------------------
    # 2023 = training
    # -------------------------

    train_2023 = df[
        df["Season"] == 2023
    ].copy()

    # -------------------------
    # 2024 = train + validation
    # -------------------------

    races_2024 = (
        df[df["Season"] == 2024]["Race"]
        .drop_duplicates()
        .tolist()
    )

    split_index = 14

    train_2024_races = races_2024[:split_index]
    val_2024_races = races_2024[split_index:]

    train_2024 = df[
        (df["Season"] == 2024)
        & (df["Race"].isin(train_2024_races))
    ].copy()

    val_2024 = df[
        (df["Season"] == 2024)
        & (df["Race"].isin(val_2024_races))
    ].copy()

    # -------------------------
    # 2025 = final test
    # -------------------------

    test_2025 = df[
        df["Season"] == 2025
    ].copy()

    train_df = pd.concat(
        [train_2023, train_2024],
        ignore_index=True
    )

    return (
        train_df,
        val_2024,
        test_2025,
        train_2024_races,
        val_2024_races,
    )


def preprocess():

    df = load_data()

    (
        train_df,
        val_df,
        test_df,
        train_2024_races,
        val_2024_races,
    ) = chronological_split(df)

    print("\nDataset split")
    print("-------------")

    print(
        f"Training:   {len(train_df)} rows"
    )

    print(
        f"Validation: {len(val_df)} rows"
    )

    print(
        f"Test:       {len(test_df)} rows"
    )

    print("\nTraining:")
    print("2023 + first 14 races of 2024")

    print("\n2024 training races:")
    print(", ".join(train_2024_races))

    print("\n2024 validation races:")
    print(", ".join(val_2024_races))

    print("\nTest:")
    print("Entire 2025 season")

    # -------------------------
    # Features / targets
    # -------------------------

    X_train = train_df[FEATURES].values
    y_train = train_df[TARGET].values

    X_val = val_df[FEATURES].values
    y_val = val_df[TARGET].values

    X_test = test_df[FEATURES].values
    y_test = test_df[TARGET].values

    # -------------------------
    # Normalize
    # -------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_val_scaled = scaler.transform(
        X_val
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # -------------------------
    # Save processed datasets
    # -------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_processed = pd.DataFrame(
        X_train_scaled,
        columns=FEATURES
    )

    train_processed[TARGET] = y_train

    val_processed = pd.DataFrame(
        X_val_scaled,
        columns=FEATURES
    )

    val_processed[TARGET] = y_val

    test_processed = pd.DataFrame(
        X_test_scaled,
        columns=FEATURES
    )

    test_processed[TARGET] = y_test

    train_processed.to_csv(
        PROCESSED_DIR / "train.csv",
        index=False
    )

    val_processed.to_csv(
        PROCESSED_DIR / "validation.csv",
        index=False
    )

    test_processed.to_csv(
        PROCESSED_DIR / "test.csv",
        index=False
    )

    joblib.dump(
        scaler,
        MODELS_DIR / "scaler.pkl"
    )

    # Save scaler parameters for deployment
    scaler_params = {
        "features": FEATURES,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist()
    }

    with open("models/scaler.json", "w") as f:
        json.dump(scaler_params, f, indent=4)

    print("Saved: models/scaler.json")

    print("\nPreprocessing complete.")

    print(
        f"Saved: {PROCESSED_DIR / 'train.csv'}"
    )

    print(
        f"Saved: {PROCESSED_DIR / 'validation.csv'}"
    )

    print(
        f"Saved: {PROCESSED_DIR / 'test.csv'}"
    )

    print(
        f"Saved: {MODELS_DIR / 'scaler.pkl'}"
    )


if __name__ == "__main__":
    preprocess()
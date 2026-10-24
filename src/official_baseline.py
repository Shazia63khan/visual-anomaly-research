
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# Project root: parent folder of src/
ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "split_csv" / "1cls.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OBJECT_NAME = "capsules"
IMAGE_SIZE = (128, 128)


def extract_features(image_path):
    """Extract simple colour and texture features from one image."""
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize(IMAGE_SIZE)
        rgb = np.asarray(image, dtype=np.float32) / 255.0

    gray = rgb.mean(axis=2)

    features = []
    for channel in range(3):
        features.extend([
            rgb[:, :, channel].mean(),
            rgb[:, :, channel].std(),
        ])

    features.extend([
        gray.mean(),
        gray.std(),
        np.abs(np.diff(gray, axis=0)).mean(),
        np.abs(np.diff(gray, axis=1)).mean(),
    ])

    return np.asarray(features, dtype=np.float32)


def resolve_image_path(relative_path):
    """Resolve a dataset path and verify that the image exists."""
    path = ROOT / "data" / relative_path

    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    return path


def main():
    df = pd.read_csv(CSV_PATH)
    df = df[df["object"] == OBJECT_NAME].copy()

    train_df = df[
        (df["split"] == "train") & (df["label"] == "normal")
    ].copy()

    test_df = df[df["split"] == "test"].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("The official train/test split is empty.")

    print(f"Category: {OBJECT_NAME}")
    print(f"Training normal images: {len(train_df)}")
    print(f"Test images: {len(test_df)}")
    print("\nTest label counts:")
    print(test_df["label"].value_counts().to_string())

    print("\nExtracting training features...")
    X_train = np.stack([
        extract_features(resolve_image_path(path))
        for path in train_df["image"]
    ])

    print("Extracting test features...")
    X_test = np.stack([
        extract_features(resolve_image_path(path))
        for path in test_df["image"]
    ])

    # Fit preprocessing only on the normal training images.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Use only normal training images to establish normality.
    neighbors = NearestNeighbors(n_neighbors=5, metric="euclidean")
    neighbors.fit(X_train)

    distances, _ = neighbors.kneighbors(X_test)
    anomaly_scores = distances.mean(axis=1)

    # Anomaly = 1, normal = 0.
    y_true = (test_df["label"].to_numpy() == "anomaly").astype(int)

    roc_auc = roc_auc_score(y_true, anomaly_scores)
    average_precision = average_precision_score(y_true, anomaly_scores)

    print("\n--- Official VisA 1-class baseline results ---")
    print(f"Category: {OBJECT_NAME}")
    print(f"Feature dimensions: {X_train.shape[1]}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Average Precision: {average_precision:.4f}")

    results = test_df[["object", "split", "label", "image"]].copy()
    results["anomaly_score"] = anomaly_scores
    results["anomaly_score"] = results["anomaly_score"].astype(float)
    results["is_anomaly"] = y_true

    output_csv = RESULTS_DIR / f"{OBJECT_NAME}_official_baseline.csv"
    results.to_csv(output_csv, index=False)

    np.savez(
        RESULTS_DIR / f"{OBJECT_NAME}_official_baseline.npz",
        y_true=y_true,
        anomaly_scores=anomaly_scores,
        roc_auc=roc_auc,
        average_precision=average_precision,
    )

    print(f"\nSaved per-image results to: {output_csv}")
    print("Done.")


if __name__ == "__main__":
    main()

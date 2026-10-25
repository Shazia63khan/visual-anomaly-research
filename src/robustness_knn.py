
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "split_csv" / "1cls.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CATEGORIES = ["candle", "capsules", "cashew"]
K_VALUES = [1, 3, 5, 9, 15]
IMAGE_SIZE = (128, 128)


def extract_features(image_path):
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
    path = ROOT / "data" / relative_path

    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    return path


def evaluate_category(df, category):
    category_df = df[df["object"] == category].copy()

    train_df = category_df[
        (category_df["split"] == "train")
        & (category_df["label"] == "normal")
    ].copy()

    test_df = category_df[
        category_df["split"] == "test"
    ].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(f"Missing train or test data for {category}")

    print(f"\nCategory: {category}")
    print(f"Normal training images: {len(train_df)}")
    print(f"Test images: {len(test_df)}")

    X_train = np.stack([
        extract_features(resolve_image_path(path))
        for path in train_df["image"]
    ])

    X_test = np.stack([
        extract_features(resolve_image_path(path))
        for path in test_df["image"]
    ])

    # Fit the scaler only on normal training data.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_true = (test_df["label"].to_numpy() == "anomaly").astype(int)

    rows = []

    for k in K_VALUES:
        neighbors = NearestNeighbors(
            n_neighbors=k,
            metric="euclidean"
        )
        neighbors.fit(X_train)

        distances, _ = neighbors.kneighbors(X_test)
        scores = distances.mean(axis=1)

        auc = roc_auc_score(y_true, scores)
        ap = average_precision_score(y_true, scores)

        rows.append({
            "Category": category,
            "k": k,
            "ROC-AUC": auc,
            "Average Precision": ap,
        })

        print(f"k={k:2} | ROC-AUC={auc:.4f} | AP={ap:.4f}")

    return rows


def main():
    df = pd.read_csv(CSV_PATH)

    all_rows = []

    for category in CATEGORIES:
        all_rows.extend(evaluate_category(df, category))

    results = pd.DataFrame(all_rows)
    output_path = RESULTS_DIR / "knn_robustness.csv"
    results.to_csv(output_path, index=False)

    print("\n--- k-NN Robustness Results ---")
    print(results.round(4).to_string(index=False))
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()

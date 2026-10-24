
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = ROOT / "data" / "candle" / "Data" / "Images"
RESULTS = ROOT / "results"

IMAGE_SIZE = (64, 64)
RANDOM_SEED = 42


def image_features(path):
    """Extract simple, interpretable colour and texture features."""
    image = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
    pixels = np.asarray(image, dtype=np.float32) / 255.0

    features = []

    # Per-channel colour statistics
    for channel in range(3):
        values = pixels[:, :, channel]
        features.extend([
            values.mean(),
            values.std(),
        ])

    # Grayscale intensity and simple horizontal/vertical gradients
    gray = pixels.mean(axis=2)
    features.extend([
        gray.mean(),
        gray.std(),
        np.abs(np.diff(gray, axis=0)).mean(),
        np.abs(np.diff(gray, axis=1)).mean(),
    ])

    return np.asarray(features, dtype=np.float32)


def collect_images(folder, label):
    paths = sorted(folder.glob("*"))
    rows = []

    for path in paths:
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue

        rows.append((path, label))

    return rows


def main():
    normal = collect_images(IMAGE_ROOT / "Normal", 0)
    anomaly = collect_images(IMAGE_ROOT / "Anomaly", 1)

    if not normal or not anomaly:
        raise RuntimeError(
            f"Images not found. Check dataset path: {IMAGE_ROOT}"
        )

    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(normal)

    # Train only on normal images; reserve unseen normal images for testing.
    split = int(len(normal) * 0.8)
    train_rows = normal[:split]
    test_rows = normal[split:] + anomaly
    rng.shuffle(test_rows)

    X_train = np.stack([
        image_features(path) for path, _ in train_rows
    ])
    X_test = np.stack([
        image_features(path) for path, _ in test_rows
    ])
    y_test = np.array([label for _, label in test_rows])

    # Standardize using training data only.
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    model = NearestNeighbors(n_neighbors=5)
    model.fit(X_train)

    distances, _ = model.kneighbors(X_test)
    scores = distances.mean(axis=1)

    auc = roc_auc_score(y_test, scores)

    RESULTS.mkdir(exist_ok=True)
    np.savez(
        RESULTS / "candle_baseline.npz",
        labels=y_test,
        scores=scores,
    )

    print("=== VisA Candle: Real-Image Baseline ===")
    print(f"Normal training images: {len(train_rows)}")
    print(f"Normal test images: {sum(y_test == 0)}")
    print(f"Anomaly test images: {sum(y_test == 1)}")
    print(f"Feature dimensions: {X_train.shape[1]}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Saved scores: {RESULTS / 'candle_baseline.npz'}")
    print("Method: colour statistics + grayscale texture features")
    print("Note: this is a simple baseline, not a deep-learning model.")


if __name__ == "__main__":
    main()

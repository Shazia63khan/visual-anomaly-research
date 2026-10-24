
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, average_precision_score


ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "split_csv" / "1cls.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OBJECT_NAME = "cashew"
MODEL_NAME = "facebook/dinov2-small"
BATCH_SIZE = 8

DEVICE = torch.device("cpu")


def load_images(paths):
    images = []

    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))

    return images


def resolve_image_path(relative_path):
    path = ROOT / "data" / relative_path

    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    return path


def extract_features(paths, processor, model):
    all_features = []

    for start in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[start:start + BATCH_SIZE]
        images = load_images(batch_paths)

        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)

            # Use the CLS token as a global image representation.
            features = outputs.last_hidden_state[:, 0, :]

        features = torch.nn.functional.normalize(features, p=2, dim=1)
        all_features.append(features.cpu().numpy())

        processed = min(start + len(batch_paths), len(paths))
        print(f"  Processed {processed}/{len(paths)} images", flush=True)

    return np.concatenate(all_features, axis=0)


def main():
    torch.set_num_threads(4)

    df = pd.read_csv(CSV_PATH)
    df = df[df["object"] == OBJECT_NAME].copy()

    train_df = df[
        (df["split"] == "train") & (df["label"] == "normal")
    ].copy()

    test_df = df[df["split"] == "test"].copy()

    print(f"Category: {OBJECT_NAME}")
    print(f"Normal training images: {len(train_df)}")
    print(f"Test images: {len(test_df)}")
    print(f"Device: {DEVICE}")
    print(f"Loading pretrained model: {MODEL_NAME}", flush=True)

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    train_paths = [
        resolve_image_path(path) for path in train_df["image"]
    ]
    test_paths = [
        resolve_image_path(path) for path in test_df["image"]
    ]

    print("\nExtracting DINOv2 training features...", flush=True)
    X_train = extract_features(train_paths, processor, model)

    print("\nExtracting DINOv2 test features...", flush=True)
    X_test = extract_features(test_paths, processor, model)

    print(f"\nFeature dimensions: {X_train.shape[1]}")
    print("Fitting nearest-neighbour model...", flush=True)

    neighbors = NearestNeighbors(
        n_neighbors=5,
        metric="euclidean",
    )
    neighbors.fit(X_train)

    distances, _ = neighbors.kneighbors(X_test)
    anomaly_scores = distances.mean(axis=1)

    y_true = (test_df["label"].to_numpy() == "anomaly").astype(int)

    roc_auc = roc_auc_score(y_true, anomaly_scores)
    average_precision = average_precision_score(y_true, anomaly_scores)

    print("\n--- DINOv2 official VisA baseline ---")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Average Precision: {average_precision:.4f}")

    results = test_df[["object", "split", "label", "image"]].copy()
    results["anomaly_score"] = anomaly_scores
    results["is_anomaly"] = y_true

    output_csv = RESULTS_DIR / f"{OBJECT_NAME}_dinov2_baseline.csv"
    results.to_csv(output_csv, index=False)

    np.savez(
        RESULTS_DIR / f"{OBJECT_NAME}_dinov2_baseline.npz",
        y_true=y_true,
        anomaly_scores=anomaly_scores,
        roc_auc=roc_auc,
        average_precision=average_precision,
    )

    print(f"\nSaved per-image results to: {output_csv}")
    print("Done.")


if __name__ == "__main__":
    main()

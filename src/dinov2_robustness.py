
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

CATEGORIES = ["candle", "capsules", "cashew"]
K_VALUES = [1, 3, 5, 9, 15]

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
        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = model(**inputs)
            features = outputs.last_hidden_state[:, 0, :]

        features = torch.nn.functional.normalize(
            features, p=2, dim=1
        )
        all_features.append(features.cpu().numpy())

        processed = min(start + len(batch_paths), len(paths))
        print(f"  Processed {processed}/{len(paths)} images",
              flush=True)

    return np.concatenate(all_features, axis=0)


def main():
    torch.set_num_threads(4)

    df = pd.read_csv(CSV_PATH)
    df = df[df["object"].isin(CATEGORIES)].copy()

    print(f"Loading pretrained model: {MODEL_NAME}", flush=True)
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    all_results = []

    for category in CATEGORIES:
        category_df = df[df["object"] == category].copy()

        train_df = category_df[
            (category_df["split"] == "train")
            & (category_df["label"] == "normal")
        ].copy()

        test_df = category_df[
            category_df["split"] == "test"
        ].copy()

        print(f"\n{'=' * 45}")
        print(f"Category: {category}")
        print(f"Normal training images: {len(train_df)}")
        print(f"Test images: {len(test_df)}")

        train_paths = [
            resolve_image_path(path) for path in train_df["image"]
        ]
        test_paths = [
            resolve_image_path(path) for path in test_df["image"]
        ]

        print("Extracting training embeddings...", flush=True)
        X_train = extract_features(train_paths, processor, model)

        print("Extracting test embeddings...", flush=True)
        X_test = extract_features(test_paths, processor, model)

        y_true = (
            test_df["label"].to_numpy() == "anomaly"
        ).astype(int)

        for k in K_VALUES:
            neighbors = NearestNeighbors(
                n_neighbors=k,
                metric="euclidean",
            )
            neighbors.fit(X_train)

            distances, _ = neighbors.kneighbors(X_test)
            anomaly_scores = distances.mean(axis=1)

            roc_auc = roc_auc_score(y_true, anomaly_scores)
            average_precision = average_precision_score(
                y_true, anomaly_scores
            )

            print(
                f"k={k:2d} | ROC-AUC={roc_auc:.4f} "
                f"| AP={average_precision:.4f}",
                flush=True,
            )

            all_results.append({
                "category": category,
                "k": k,
                "roc_auc": roc_auc,
                "average_precision": average_precision,
                "train_normal_images": len(train_df),
                "test_images": len(test_df),
            })

    results_df = pd.DataFrame(all_results)
    output_path = RESULTS_DIR / "dinov2_robustness.csv"
    results_df.to_csv(output_path, index=False)

    print(f"\nSaved results to: {output_path}")
    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()

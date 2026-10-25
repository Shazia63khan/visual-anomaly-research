
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLIT_CSV = PROJECT_ROOT / "data" / "split_csv" / "1cls.csv"
IMAGE_ROOT = PROJECT_ROOT / "data"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["candle", "capsules", "cashew"]
VALIDATION_FRACTION = 0.20
RANDOM_SEED = 42
K_VALUES = [1, 3, 5, 9, 15]


def get_image_path(relative_path):
    path = IMAGE_ROOT / str(relative_path)
    if path.exists():
        return path

    # Some split files store paths relative to the dataset directory.
    matches = list(IMAGE_ROOT.rglob(Path(str(relative_path)).name))
    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        f"Could not uniquely locate image: {relative_path}"
    )


def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((128, 128))
    array = np.asarray(image, dtype=np.float32) / 255.0

    features = []

    for channel in range(3):
        plane = array[:, :, channel]
        features.extend([plane.mean(), plane.std()])

    gray = array.mean(axis=2)
    features.extend([gray.mean(), gray.std()])
    features.append(np.abs(np.diff(gray, axis=0)).mean())
    features.append(np.abs(np.diff(gray, axis=1)).mean())

    return features


def main():
    split = pd.read_csv(SPLIT_CSV)
    all_results = []

    for category in CATEGORIES:
        normal_train = split[
            (split["object"] == category)
            & (split["split"] == "train")
            & (split["label"] == "normal")
        ].copy()

        if len(normal_train) < 10:
            print(f"Skipping {category}: too few normal training images.")
            continue

        rng = np.random.default_rng(RANDOM_SEED)
        indices = rng.permutation(len(normal_train))

        validation_size = max(
            1, int(round(len(normal_train) * VALIDATION_FRACTION))
        )

        validation_indices = indices[:validation_size]
        fit_indices = indices[validation_size:]

        fit_df = normal_train.iloc[fit_indices]
        validation_df = normal_train.iloc[validation_indices]

        fit_features = np.array([
            extract_features(get_image_path(path))
            for path in fit_df["image"]
        ])

        validation_features = np.array([
            extract_features(get_image_path(path))
            for path in validation_df["image"]
        ])

        scaler = StandardScaler()
        fit_features = scaler.fit_transform(fit_features)
        validation_features = scaler.transform(validation_features)

        print(f"\nCategory: {category}")
        print(f"Fit normal images: {len(fit_df)}")
        print(f"Validation normal images: {len(validation_df)}")

        for k in K_VALUES:
            if k > len(fit_features):
                continue

            model = NearestNeighbors(n_neighbors=k)
            model.fit(fit_features)

            distances, _ = model.kneighbors(validation_features)
            scores = distances.mean(axis=1)

            # A normal-only percentile threshold for a nominal 5% FPR.
            threshold = float(np.percentile(scores, 95))
            observed_fpr = float(np.mean(scores > threshold))

            row = {
                "category": category,
                "k": k,
                "fit_normal_count": len(fit_df),
                "validation_normal_count": len(validation_df),
                "validation_score_mean": float(scores.mean()),
                "validation_score_std": float(scores.std()),
                "normal_score_95th_percentile": threshold,
                "observed_validation_fpr": observed_fpr,
            }
            all_results.append(row)

            print(
                f"k={k:2d} | "
                f"mean score={scores.mean():.4f} | "
                f"95th percentile={threshold:.4f} | "
                f"observed FPR={observed_fpr:.3f}"
            )

    output_path = RESULTS_DIR / "normal_only_validation.csv"
    pd.DataFrame(all_results).to_csv(output_path, index=False)

    print(f"\nSaved validation results to: {output_path}")
    print(
        "\nNote: This is normal-only calibration, not anomaly validation. "
        "It does not provide ROC-AUC or Average Precision."
    )


if __name__ == "__main__":
    main()

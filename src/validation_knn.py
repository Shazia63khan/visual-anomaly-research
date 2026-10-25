from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLIT_CSV = PROJECT_ROOT / "data" / "split_csv" / "1cls.csv"
IMAGE_ROOT = PROJECT_ROOT / "data"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["candle", "capsules", "cashew"]

CALIBRATION_FRACTION = 0.20
RANDOM_SEED = 42
K = 5
TARGET_FPR = 0.05


def get_image_path(relative_path):
    path = IMAGE_ROOT / str(relative_path)

    if path.exists():
        return path

    matches = list(
        IMAGE_ROOT.rglob(Path(str(relative_path)).name)
    )

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


def load_features(frame):
    return np.asarray([
        extract_features(get_image_path(path))
        for path in frame["image"]
    ], dtype=np.float32)


def main():
    split = pd.read_csv(SPLIT_CSV)
    all_results = []

    for category in CATEGORIES:
        normal_train = split[
            (split["object"] == category)
            & (split["split"] == "train")
            & (split["label"] == "normal")
        ].copy().reset_index(drop=True)

        test_df = split[
            (split["object"] == category)
            & (split["split"] == "test")
        ].copy().reset_index(drop=True)

        if len(normal_train) < 10 or len(test_df) == 0:
            print(f"Skipping {category}: insufficient data.")
            continue

        rng = np.random.default_rng(RANDOM_SEED)
        indices = rng.permutation(len(normal_train))

        calibration_size = max(
            1,
            int(round(len(normal_train) * CALIBRATION_FRACTION))
        )

        calibration_indices = indices[:calibration_size]
        fit_indices = indices[calibration_size:]

        fit_df = normal_train.iloc[fit_indices]
        calibration_df = normal_train.iloc[calibration_indices]

        fit_features = load_features(fit_df)
        calibration_features = load_features(calibration_df)
        test_features = load_features(test_df)

        # Fit preprocessing only on the fit subset.
        scaler = StandardScaler()
        fit_features = scaler.fit_transform(fit_features)
        calibration_features = scaler.transform(
            calibration_features
        )
        test_features = scaler.transform(test_features)

        # Fit nearest-neighbour model only on normal fit images.
        model = NearestNeighbors(n_neighbors=K)
        model.fit(fit_features)

        calibration_distances, _ = model.kneighbors(
            calibration_features
        )
        calibration_scores = calibration_distances.mean(axis=1)

        # Calibrate the threshold using normal calibration images only.
        threshold = float(
            np.quantile(
                calibration_scores,
                1.0 - TARGET_FPR,
                method="higher"
            )
        )

        # Evaluate the frozen model and threshold on official test data.
        test_distances, _ = model.kneighbors(test_features)
        test_scores = test_distances.mean(axis=1)

        test_labels = (
            test_df["label"].astype(str).str.lower()
            == "anomaly"
        ).astype(int).to_numpy()

        predictions = (test_scores > threshold).astype(int)

        normal_mask = test_labels == 0
        anomaly_mask = test_labels == 1

        test_fpr = (
            float(predictions[normal_mask].mean())
            if normal_mask.any()
            else np.nan
        )

        test_tpr = (
            float(predictions[anomaly_mask].mean())
            if anomaly_mask.any()
            else np.nan
        )

        roc_auc = (
            float(roc_auc_score(test_labels, test_scores))
            if len(np.unique(test_labels)) == 2
            else np.nan
        )

        average_precision = (
            float(average_precision_score(test_labels, test_scores))
            if anomaly_mask.any()
            else np.nan
        )

        result = {
            "category": category,
            "k": K,
            "fit_normal_count": len(fit_df),
            "calibration_normal_count": len(calibration_df),
            "test_count": len(test_df),
            "test_normal_count": int(normal_mask.sum()),
            "test_anomaly_count": int(anomaly_mask.sum()),
            "target_fpr": TARGET_FPR,
            "calibration_threshold": threshold,
            "calibration_score_95th_percentile": float(
                np.quantile(calibration_scores, 0.95)
            ),
            "test_fpr": test_fpr,
            "test_tpr": test_tpr,
            "test_roc_auc": roc_auc,
            "test_average_precision": average_precision,
        }

        all_results.append(result)

        print(f"\nCategory: {category}")
        print(f"Fit normal images: {len(fit_df)}")
        print(f"Calibration normal images: {len(calibration_df)}")
        print(f"Test images: {len(test_df)}")
        print(f"Calibrated threshold: {threshold:.6f}")
        print(f"Test FPR: {test_fpr:.4f}")
        print(f"Test TPR: {test_tpr:.4f}")
        print(f"Test ROC-AUC: {roc_auc:.4f}")
        print(f"Test Average Precision: {average_precision:.4f}")

    output_path = RESULTS_DIR / "normal_only_validation.csv"
    pd.DataFrame(all_results).to_csv(output_path, index=False)

    print(f"\nSaved results to: {output_path}")
    print(
        "\nImportant: test metrics are independent of threshold "
        "calibration, but repeated experiments or model selection "
        "based on these test results can still introduce bias."
    )


if __name__ == "__main__":
    main()
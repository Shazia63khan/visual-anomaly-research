
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

combined = pd.read_csv(RESULTS / "combined_results.csv")

categories = ["candle", "capsules", "cashew"]
methods = {
    "Handcrafted": "_official_baseline.csv",
    "DINOv2": "_dinov2_baseline.csv",
}

for category in categories:
    for method, suffix in methods.items():
        path = RESULTS / f"{category}{suffix}"
        df = pd.read_csv(path)

        y_true = (df["label"] == "anomaly").astype(int)
        scores = df["anomaly_score"]

        auc = roc_auc_score(y_true, scores)
        ap = average_precision_score(y_true, scores)

        row = combined[
            (combined["Category"] == category)
            & (combined["Method"] == method)
        ]

        matches = (
            not row.empty
            and abs(float(row.iloc[0]["ROC-AUC"]) - auc) < 0.0001
            and abs(
                float(row.iloc[0]["Average Precision"]) - ap
            ) < 0.0001
        )

        print(
            f"{category:8} | {method:12} | "
            f"ROC-AUC={auc:.4f} | AP={ap:.4f} | "
            f"CSV match: {matches}"
        )

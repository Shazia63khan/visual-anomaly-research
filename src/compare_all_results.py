
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

categories = ["candle", "capsules","cashew"]
methods = {
    "Handcrafted": "_official_baseline.csv",
    "DINOv2": "_dinov2_baseline.csv",
}

rows = []

for category in categories:
    for method, suffix in methods.items():
        path = RESULTS / f"{category}{suffix}"

        if not path.exists():
            raise FileNotFoundError(f"Missing results file: {path}")

        df = pd.read_csv(path)
        y_true = (df["label"] == "anomaly").astype(int)
        scores = df["anomaly_score"]

        rows.append({
            "Category": category,
            "Method": method,
            "ROC-AUC": roc_auc_score(y_true, scores),
            "Average Precision": average_precision_score(y_true, scores),
        })

summary = pd.DataFrame(rows)

print("\nCombined results:")
print(summary.round(4).to_string(index=False))

summary.to_csv(RESULTS / "combined_results.csv", index=False)

for metric in ["ROC-AUC", "Average Precision"]:
    pivot = summary.pivot(
        index="Category",
        columns="Method",
        values=metric,
    )

    ax = pivot.plot(kind="bar", figsize=(8, 5), ylim=(0, 1.05))
    ax.set_title(f"VisA Anomaly Detection: {metric}")
    ax.set_ylabel(metric)
    ax.set_xlabel("Object category")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Feature method")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output = RESULTS / f"comparison_{metric.lower().replace(' ', '_')}.png"
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved graph: {output}")

print("\nDone.")

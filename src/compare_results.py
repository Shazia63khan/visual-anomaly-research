
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

methods = {
    "Handcrafted\nFeatures": "candle_official_baseline.csv",
    "DINOv2\nFeatures": "candle_dinov2_baseline.csv",
}

metrics = {}

for method, filename in methods.items():
    path = RESULTS / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")

    df = pd.read_csv(path)

    labels = (df["label"] == "anomaly").astype(int)
    scores = df["anomaly_score"]

    from sklearn.metrics import roc_auc_score, average_precision_score

    metrics[method] = {
        "ROC-AUC": roc_auc_score(labels, scores),
        "Average Precision": average_precision_score(labels, scores),
    }

df_metrics = pd.DataFrame(metrics)

print("\nComparison:")
print(df_metrics.round(4).to_string())

ax = df_metrics.plot(kind="bar", figsize=(9, 5), ylim=(0, 1.05))
ax.set_title("VisA Candle: Anomaly Detection Baseline Comparison")
ax.set_ylabel("Score")
ax.set_xlabel("Evaluation metric")
ax.set_xticklabels(df_metrics.index, rotation=0)
ax.legend(title="Method")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

output_path = RESULTS / "candle_method_comparison.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"\nGraph saved to: {output_path}")

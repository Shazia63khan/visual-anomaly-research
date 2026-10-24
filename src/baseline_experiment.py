
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

# Reproducible synthetic experiment.
# This checks our evaluation pipeline before using real images.

rng = np.random.default_rng(42)

n_normal = 200
n_anomaly = 50
n_features = 16

normal_train = rng.normal(0, 1, (n_normal, n_features))
normal_test = rng.normal(0, 1, (100, n_features))
anomaly_test = rng.normal(2.5, 1, (n_anomaly, n_features))

X_test = np.vstack([normal_test, anomaly_test])
y_test = np.concatenate([
    np.zeros(len(normal_test)),
    np.ones(len(anomaly_test)),
])

# Fit only on normal training samples.
model = NearestNeighbors(n_neighbors=5)
model.fit(normal_train)

distances, _ = model.kneighbors(X_test)
anomaly_scores = distances.mean(axis=1)

auc = roc_auc_score(y_test, anomaly_scores)

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

np.savez(
    output_dir / "synthetic_baseline.npz",
    labels=y_test,
    scores=anomaly_scores,
)

print("=== Synthetic Baseline Experiment ===")
print(f"Normal training samples: {len(normal_train)}")
print(f"Normal test samples: {len(normal_test)}")
print(f"Anomalous test samples: {len(anomaly_test)}")
print(f"ROC-AUC: {auc:.4f}")
print("Saved scores to results/synthetic_baseline.npz")
print(
    "\nNote: This uses synthetic data to validate the pipeline, "
    "not a real-world anomaly-detection result."
)

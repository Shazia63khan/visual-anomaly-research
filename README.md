# Evaluating Self-Supervised Visual Representations for Image Anomaly Detection

## About the Project

This project investigates how pretrained visual representations perform in industrial image anomaly detection. Using the VisA dataset, I compare simple handcrafted image features with features extracted from a pretrained DINOv2 model.

The main research question is:

**Do pretrained visual representations improve image-level anomaly detection compared with basic handcrafted colour and texture features when using the same nearest-neighbour scoring method?**

The project focuses on evaluating existing feature representations rather than developing a new anomaly detection algorithm.

## Dataset

The experiments use the official one-class split provided with the VisA dataset.

Three object categories have been evaluated:

- **Candle**
- **Capsules**
- **Cashew**

For each category, the official training split contains normal images, while the test split contains normal and anomalous images. This setup supports one-class anomaly detection, where models learn from normal training examples and are evaluated on both classes at test time.

Dataset: [VisA — A Large-Scale Benchmark for Visual Anomaly Detection](https://github.com/amazon-science/spot-diff)

The dataset is not included in this repository. Please obtain it from the original source and follow its usage terms.

## Methods

### 1. Handcrafted Features

I extracted 10-dimensional image features comprising:

- RGB channel means and standard deviations
- Grayscale mean and standard deviation
- Mean absolute horizontal and vertical image gradients

The features were standardised using statistics fitted on normal training data. A k-nearest neighbours (k-NN) model with **k = 5** was used to calculate anomaly scores based on distances to normal training samples.

This method serves as a simple, interpretable baseline.

### 2. DINOv2 Features

For the second approach, I used the pretrained `facebook/dinov2-small` model to extract 384-dimensional image representations from its CLS token.

The extracted representations were L2-normalised. The same k-NN setting (**k = 5**) was used to calculate anomaly scores, allowing a comparison between handcrafted and pretrained features while keeping the scoring method consistent.

The experiments were run on CPU. No separate classifier was trained, and the pretrained DINOv2 model was not fine-tuned for this project.

## Evaluation Metrics

Two image-level metrics were used:

- **ROC-AUC:** Measures how well anomaly scores rank anomalous images above normal images.
- **Average Precision (AP):** Summarises precision-recall performance and is influenced by the proportion of anomalous examples.

Higher values indicate better performance for both metrics.

## Main Results

The following table reports the results obtained on the three evaluated VisA categories using the fixed setting **k = 5**.

| Category | Feature Method | ROC-AUC | Average Precision |
|---|---|---:|---:|
| Candle | Handcrafted | 0.8237 | 0.8565 |
| Candle | DINOv2 | **0.9269** | **0.9348** |
| Capsules | Handcrafted | 0.7267 | 0.8599 |
| Capsules | DINOv2 | **0.8213** | **0.8818** |
| Cashew | Handcrafted | **0.9488** | **0.9714** |
| Cashew | DINOv2 | 0.9248 | 0.9643 |

### Observations

- **Candle:** DINOv2 achieved a higher ROC-AUC by 0.1032 and a higher AP by 0.0783.
- **Capsules:** DINOv2 achieved a higher ROC-AUC by 0.0946 and a higher AP by 0.0219.
- **Cashew:** Handcrafted features achieved a higher ROC-AUC by 0.0240 and a higher AP by 0.0071.

In these experiments, DINOv2 performed better on candle and capsules, while handcrafted features performed better on cashew. This suggests that the relative performance of feature representations can vary across object categories.

These are preliminary results from three categories, not a complete evaluation of VisA. The observed differences have not been tested for statistical significance.

## Robustness Analysis: k-NN Neighbour Count

To investigate sensitivity to the neighbour count, both feature methods were evaluated using **k = 1, 3, 5, 9, and 15** on the same three categories.

This analysis uses the official test images and is therefore reported as an **exploratory test-set sensitivity analysis**. The results are not used to select a new best-performing k for the main comparison. The main results above retain the fixed setting k = 5.

The results show category-dependent behaviour:

- **Candle:** DINOv2 achieved higher ROC-AUC and AP than the handcrafted baseline across all tested k values.
- **Capsules:** DINOv2 achieved higher ROC-AUC and AP across all tested k values.
- **Cashew:** The handcrafted baseline achieved higher ROC-AUC for k values of 3, 5, 9, and 15. Its AP was also higher at these values.

These findings indicate that the relative performance of the representations depends on the object category and scoring configuration. Since the same test set is used across the k sweep, these observations should not be interpreted as an independent model-selection result.

The robustness results are saved in:

- `results/knn_robustness.csv` — handcrafted feature sensitivity analysis.
- `results/dinov2_robustness.csv` — DINOv2 feature sensitivity analysis.

## Normal-Only Validation and Threshold Calibration

A separate calibration experiment was conducted using only normal training images for candle, capsules, and cashew.

For each category, the normal training images were divided into a fitting subset and a validation subset using a fixed random seed (42). The feature standardisation statistics were fitted on the fitting subset only.

| Category | Fitting Images | Validation Images |
|---|---:|---:|
| Candle | 720 | 180 |
| Capsules | 434 | 108 |
| Cashew | 360 | 90 |

Handcrafted-feature k-NN scores were calculated for validation images using k values of 1, 3, 5, 9, and 15. The 95th percentile of the validation normal scores was recorded as a candidate threshold for each setting.

The resulting validation score distributions and thresholds are saved in `results/normal_only_validation.csv`.

This experiment characterises anomaly scores on normal images and explores threshold calibration. It does **not** measure anomaly-detection accuracy because the validation subset contains no anomalous images. ROC-AUC and AP cannot be calculated from this normal-only subset.

The observed validation false-positive proportions were approximately 5–5.6%. These proportions are calculated on the same normal validation scores used to derive the thresholds, so they are not independent estimates of the false-positive rate on unseen data. Further evaluation on separate labelled validation data would be needed to assess threshold performance on both normal and anomalous examples.

## Result Visualisations

The following figures are generated by `src/compare_all_results.py`:

- `results/comparison_roc-auc.png` — ROC-AUC comparison across categories.
- `results/comparison_average_precision.png` — Average Precision comparison across categories.

The combined metrics are saved to `results/combined_results.csv`.

## Project Structure

```text
VisualAnomalyResearch/
├── data/                          # Local dataset; not tracked by Git
│   └── split_csv/
├── src/
│   ├── baseline_experiment.py
│   ├── real_baseline.py
│   ├── official_baseline.py
│   ├── dinov2_baseline.py
│   ├── compare_results.py
│   ├── compare_all_results.py
│   ├── verify_results.py
│   ├── robustness_knn.py
│   ├── dinov2_robustness.py
│   └── validation_knn.py
├── results/
│   ├── combined_results.csv
│   ├── comparison_roc-auc.png
│   ├── comparison_average_precision.png
│   ├── knn_robustness.csv
│   ├── dinov2_robustness.csv
│   └── normal_only_validation.csv
├── check_environment.py
├── requirements.txt
└── README.md
```

Each category's baseline result files contain per-image predictions and anomaly scores. Corresponding `.npz` files are also generated by the experiment scripts. The dataset itself is stored locally and is not included in the repository.

## Running the Experiments

Run commands from the project root directory.

**Environment note:** The experiments were developed using Python 3.13.14 on CPU. Package versions are documented in `requirements.txt`. Installing the CPU-specific PyTorch build may require the official PyTorch CPU package index.

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure the Category

In `src/official_baseline.py` or `src/dinov2_baseline.py`, set `OBJECT_NAME` to one of the supported categories:

```python
OBJECT_NAME = "candle"
```

For example, use `"capsules"` or `"cashew"` to evaluate another category.

Ensure the VisA dataset and official split CSV are available in the expected `data/` directory, with paths matching those configured in the scripts.

### 3. Run the Handcrafted Baseline

```bash
python src/official_baseline.py
```

### 4. Run the DINOv2 Baseline

```bash
python src/dinov2_baseline.py
```

Run both methods for each category you want to evaluate.

### 5. Generate Combined Results and Graphs

```bash
python src/compare_all_results.py
```

The comparison script expects the per-category CSV files from both methods to exist in the `results/` directory.

### 6. Run the Robustness Experiments

Handcrafted features:

```bash
python src/robustness_knn.py
```

DINOv2 features:

```bash
python src/dinov2_robustness.py
```

These scripts evaluate the specified k values on the official test split. Interpret these results as exploratory sensitivity analysis rather than independent model selection.

### 7. Run Normal-Only Calibration

```bash
python src/validation_knn.py
```

This script splits normal training images into fitting and validation subsets and saves candidate thresholds and normal-score statistics to `results/normal_only_validation.csv`. It does not calculate anomaly-detection metrics.

## Limitations and Future Work

- The current evaluation covers only three of the VisA object categories.
- The experiments report image-level anomaly-detection metrics; pixel-level anomaly localisation is not evaluated.
- The handcrafted baseline uses a small set of basic image statistics and gradients, so it does not represent all possible traditional feature methods.
- The evaluation uses a simple k-NN scoring procedure and one fixed setting for the main comparison.
- The k-sweep is exploratory and uses the official test set; it should not be used to claim independently selected optimal hyperparameters.
- The normal-only calibration experiment cannot measure detection performance on anomalous images.
- Statistical significance and run-to-run variability have not been established.
- Results may depend on image preprocessing, feature extraction, and the selected scoring procedure.

Future work could extend the evaluation to more VisA categories, investigate additional pretrained representations and baselines, evaluate pixel-level localisation, introduce a separate labelled validation set for threshold and hyperparameter selection, and assess robustness across experimental settings.

## Acknowledgements

This project uses the VisA dataset and the pretrained DINOv2 model. Credit goes to the original authors and maintainers of these resources.

- [VisA dataset and benchmark](https://github.com/amazon-science/spot-diff)
- [DINOv2](https://github.com/facebookresearch/dinov2)

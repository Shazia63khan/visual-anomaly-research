# Evaluating Self-Supervised Visual Representations for Image Anomaly Detection

## About the Project

This project looks at how well pretrained visual features can identify unusual images in an industrial inspection setting. I used the VisA dataset to compare two approaches: simple handcrafted image features and features extracted using a pretrained DINOv2 model.

The main goal was to understand whether pretrained visual representations can detect anomalies more effectively than basic colour and texture features, even without training a separate classifier.

## Dataset

The experiments use the official one-class split provided with the VisA dataset.

I started with two object categories:

- **Candle**
- **Capsules**

For each category, the training set contains normal images, while the test set contains both normal and anomalous images. The official split was used to keep the evaluation consistent.

Dataset: [VisA – A Large-Scale Benchmark for Visual Anomaly Detection](https://github.com/amazon-science/spot-diff)

The dataset is not included in this repository. Please obtain it from the original source and follow its usage terms.

## Methods

### 1. Handcrafted Features

I extracted basic colour and texture-related features from each image, including RGB channel statistics, grayscale statistics, and horizontal and vertical image gradients.

These features were used as a simple baseline. After standardising the features using the normal training data, I applied a k-nearest neighbours approach to assign anomaly scores to test images.

### 2. DINOv2 Features

For the second approach, I used the pretrained `facebook/dinov2-small` model to extract image representations.

The extracted features were L2-normalised, and the same nearest-neighbour approach was used to calculate anomaly scores. This made it possible to compare the two feature methods while keeping the scoring procedure similar.

The experiments were run on CPU.

## Results

The following table shows the results obtained on the two categories tested so far.

| Category | Feature Method | ROC-AUC | Average Precision |
|---|---|---:|---:|
| Candle | Handcrafted features | 0.8237 | 0.8565 |
| Candle | DINOv2 features | 0.9269 | 0.9348 |
| Capsules | Handcrafted features | 0.7267 | 0.8599 |
| Capsules | DINOv2 features | 0.8213 | 0.8818 |

DINOv2 features performed better than the handcrafted features on both categories in these experiments. The difference was particularly noticeable in ROC-AUC.

These results are encouraging, but they cover only two categories and should not be treated as a complete evaluation of the VisA dataset.

## Project Structure

```text
VisualAnomalyResearch/
├── data/
│   └── split_csv/
├── src/
│   ├── baseline_experiment.py
│   ├── real_baseline.py
│   ├── official_baseline.py
│   ├── dinov2_baseline.py
│   ├── compare_results.py
│   └── compare_all_results.py
├── results/
│   ├── candle_official_baseline.csv
│   ├── candle_dinov2_baseline.csv
│   ├── capsules_official_baseline.csv
│   ├── capsules_dinov2_baseline.csv
│   ├── comparison_roc-auc.png
│   └── comparison_average_precision.png
├── check_environment.py
└── README.md
```

## Running the Experiments

Run these commands from the project root directory.

Install the required libraries if they are not already installed:

```bash
pip install torch torchvision transformers numpy scikit-learn matplotlib pandas pillow
```

Run the handcrafted baseline:

```bash
python src/official_baseline.py
```

Run the DINOv2 baseline:

```bash
python src/dinov2_baseline.py
```

Generate the combined comparison table and graphs:

```bash
python src/compare_all_results.py
```

The scripts expect the VisA dataset and its official split CSV to be available in the project’s `data/` directory. Check the paths configured in the scripts if your dataset is stored elsewhere.

## Limitations and Next Steps

This is an ongoing experimental project, not a new anomaly detection algorithm. The current results are based on only two object categories, and the experiments use image-level anomaly scores rather than pixel-level anomaly localisation.

The current setup also uses a simple nearest-neighbour scoring method. More categories, repeated runs where applicable, and additional baselines would be useful before drawing broader conclusions.

Next, I plan to extend the evaluation to more VisA categories, review the results in more detail, and document the experimental setup and limitations more thoroughly.

## Acknowledgements

This project uses the VisA dataset and the pretrained DINOv2 model. Credit goes to the original authors and maintainers of these resources.
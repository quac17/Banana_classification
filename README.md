# Banana Classification

## Project Overview
This repository implements a computer‑vision pipeline to automatically classify bananas into four ripeness categories: **unripe (green), ripe (yellow), overripe (brown‑spotted),** and **rotten**. The core steps are image preprocessing, segmentation, object detection & cropping, feature extraction, and classification using a Decision Tree (main model) with K‑Nearest Neighbors and SVM for comparison.

## Requirements
```bash
# Install the required Python packages
pip install -r requirements.txt
```
The `requirements.txt` pins `numpy<2`, `opencv-python`, `scikit-learn`, and `matplotlib`.

## Setup
1. Clone the repository (or download the source).
2. Ensure the folder structure:
   ```
   Banana_classification/
   ├─ data/          # contains train_data/ and test_data/ subfolders
   ├─ src/           # preprocessing, segmentation, detection, feature_extraction
   ├─ output/        # will be created automatically
   └─ process.py
   ```
3. Place your banana images inside `data/train_data/` and `data/test_data/` following the label sub‑folders (`unripe`, `ripe`, `overripe`, `rotten`).

## Run the pipeline
```bash
python process.py
```
The script will:
- Preprocess and segment each image.
- Detect and crop individual bananas.
- Extract five features (Mean Hue, Mean Saturation, Mean Value, Brown Spot Ratio, Edge Density).
- Train a Decision Tree, KNN and SVM model.
- Save cropped images, feature CSVs, classification reports, and accuracy reports in `output/` (including `KNN_output/` and `SVM_output/`).

## Visualise results
After the pipeline finishes, generate confusion‑matrix plots for all three models:
```bash
python plot.py
```
The script reads the CSV reports and saves PNG files under `output/` and the respective model output folders.

## Quick reference
- **Main model:** Decision Tree (fast, explainable).
- **Alternative models:** KNN (k=10) and SVM (RBF kernel) – useful for benchmarking.
- **Key parameters:**
  - Gaussian Blur kernel `(5,5)`
  - Mean‑Shift `sp=15`, `sr=30`
  - HSV thresholds `Hue 15‑90`, `Sat 40‑255`, `Val 40‑255`
  - Morphology opening/closing with a `5x5` kernel (2 iterations)
  - Contour area filter `< 2000 px`, aspect‑ratio `< 1.2`, saturation filters `< 20` (stone) and `< 90` (dry leaf).

## License
This project is provided for educational purposes. Feel free to adapt and extend it for your own applications.


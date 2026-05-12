# Bristol Regional Food Network — Advanced AI

AI solution for the Bristol Regional Food Network digital marketplace, implementing product recommendations (Task 1A), demand forecasting (Task 1B), and produce quality grading (Task 2).

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Notebooks](#2-notebooks)
3. [Models and Weights](#3-models-and-weights)
4. [Outputs and Evaluation Results](#4-outputs-and-evaluation-results)
5. [Dataset Download and Placement](#5-dataset-download-and-placement)
6. [Environment Setup](#6-environment-setup)
7. [Running the Notebooks](#7-running-the-notebooks)
8. [AI Pipeline Overview](#8-ai-pipeline-overview)

---

## 1. Project Structure

```
advanced-ai/
│
├── notebooks/                      # All Jupyter notebooks (development + evaluation)
│   ├── task1_dataset_analysis.ipynb
│   ├── task1a_preprocessing.ipynb
│   ├── task1a_modelling.ipynb
│   ├── task1b_preprocessing.ipynb
│   ├── task1b_training.ipynb
│   ├── dataset_analysis.ipynb
│   ├── preprocessing.ipynb
│   ├── resnet50_training.ipynb
│   ├── mobilenetv2_training.ipynb
│   └── ensemble_evaluation.ipynb
│
├── data/
│   ├── Task 1/                     # Task 1A — recommendations
│   │   ├── raw/                    # ← place downloaded Instacart dataset here
│   │   ├── processed/              # preprocessed matrices, model artefacts
│   │   └── modelling/              # evaluation outputs, charts, pre-computed recommendations
│   │
│   ├── Task 1B/                    # Task 1B — demand forecasting
│   │   ├── processed/              # time-series tensors, scalers
│   │   └── models/                 # trained LSTM weights and evaluation
│   │
│   └── Task 2/                     # Task 2 — produce quality grading
│       ├── raw/                     # ← place downloaded Fruit and Vegetables dataset here
│       ├── processed/               # split lists, class info, augmentation config
│       ├── checkpoints/             # all training checkpoints (ResNet50 + MobileNetV2)
│       │   ├── resnet50/
│       │   └── mobilenetv2/
│       ├── ensemble/                # ensemble evaluation outputs and grading module
│       └── tuner/                   # Keras Tuner hyperparameter search trials
│           ├── resnet50/
│           └── mobilenetv2/
│
├── models/                          # Production model weights (served by the API)
│   ├── resnet50_brfn.h5
│   └── mobilenetv2_brfn.h5
│
├── src/                             # FastAPI service source code
│   ├── api/
│   ├── recommendation/
│   ├── forecasting/
│   ├── quality_assessment/
│   ├── model_management/
│   └── utils/
│
├── tests/                           # Pytest test suite
├── requirements.txt
├── .env.example
└── contract.json
```

> **Note:** All `raw/` directories are empty in the repository (excluded by `.gitignore`). Everything under `processed/`, `modelling/`, `models/`, `checkpoints/`, `ensemble/`, and `tuner/` is committed and present after cloning — only the original raw datasets need to be downloaded separately (see [Section 5](#5-dataset-download-and-placement)).

---

## 2. Notebooks

All notebooks live in [`notebooks/`](notebooks/). They were developed and executed on **Google Colab** using a Google Drive mount. Each notebook contains a setup cell at the top that mounts Drive and sets `DRIVE_ROOT` — when running locally you only need to update that path (or remove the mount block and set paths to absolute local paths).

The notebooks are ordered as follows:

### Task 1A — Product Recommendations (KNN)

| Notebook | Description |
|---|---|
| [task1_dataset_analysis.ipynb](notebooks/task1_dataset_analysis.ipynb) | Exploratory analysis of the Instacart dataset: schema, scale, sparsity, product frequency, reorder behaviour |
| [task1a_preprocessing.ipynb](notebooks/task1a_preprocessing.ipynb) | Data cleaning and feature engineering: builds the user-item matrix, TF-IDF product profiles, and user content profiles |
| [task1a_modelling.ipynb](notebooks/task1a_modelling.ipynb) | Trains and evaluates the hybrid KNN model (collaborative + content-based); grid search over `k` and blend weights; SHAP explanations; fairness audit |

### Task 1B — Demand Forecasting (LSTM)

| Notebook | Description |
|---|---|
| [task1b_preprocessing.ipynb](notebooks/task1b_preprocessing.ipynb) | Reconstructs order dates, builds daily purchase time series, applies seasonal augmentation, creates 90-day sliding-window tensors |
| [task1b_training.ipynb](notebooks/task1b_training.ipynb) | Trains the global LSTM (90-day input → 7-day forecast); compares against a seasonal-average baseline; SHAP attribution per product |

### Task 2 — Produce Quality Grading (ResNet50 + MobileNetV2 Ensemble)

| Notebook | Description |
|---|---|
| [dataset_analysis.ipynb](notebooks/dataset_analysis.ipynb) | Exploratory analysis of the image dataset: class distribution, image resolution, colour characteristics |
| [preprocessing.ipynb](notebooks/preprocessing.ipynb) | Splits dataset into train/val/test; configures augmentation pipeline (rotation, flip, colour jitter, zoom) |
| [resnet50_training.ipynb](notebooks/resnet50_training.ipynb) | Two-phase training of ResNet50 — Phase 1: frozen backbone with tuned head; Phase 2: fine-tuning with low learning rate; 30-trial Keras Tuner hyperparameter search |
| [mobilenetv2_training.ipynb](notebooks/mobilenetv2_training.ipynb) | Same two-phase process for MobileNetV2 |
| [ensemble_evaluation.ipynb](notebooks/ensemble_evaluation.ipynb) | Combines both models with optimised validation-F1 weights; evaluates ensemble on the held-out test set; generates confusion matrices, per-class F1, and grading demo |

---

## 3. Models and Weights

### Task 1A — KNN Recommendation Model

| File | Description |
|---|---|
| [`data/Task 1/processed/knn_model.joblib`](data/Task%201/processed/knn_model.joblib) | Fitted KNN model (`k=10`, cosine similarity) |
| [`data/Task 1/processed/tfidf_vectoriser.joblib`](data/Task%201/processed/tfidf_vectoriser.joblib) | Fitted TF-IDF vectoriser for product content features |
| [`data/Task 1/processed/user_item_matrix_normalised.npz`](data/Task%201/processed/user_item_matrix_normalised.npz) | Normalised user-item interaction matrix (162,544 users × 30,939 products) |
| [`data/Task 1/processed/user_content_profiles_normalised.npz`](data/Task%201/processed/user_content_profiles_normalised.npz) | Normalised user content preference profiles |
| [`data/Task 1/modelling/neighbour_indices.npy`](data/Task%201/modelling/neighbour_indices.npy) | Pre-computed KNN neighbour indices |
| [`data/Task 1/modelling/neighbour_sims.npy`](data/Task%201/modelling/neighbour_sims.npy) | Pre-computed KNN similarity scores |

**Best hyperparameters:** `k=10`, blend weights `w_freq=0.4`, `w_knn=0.3`, `w_content=0.3`

**Test-set performance:** Precision@10 = 0.271, Recall@10 = 0.329, NDCG@10 = 0.650

---

### Task 1B — LSTM Demand Forecasting

| File | Description |
|---|---|
| [`data/Task 1B/models/lstm_forecast.keras`](data/Task%201B/models/lstm_forecast.keras) | **Production model** — global LSTM trained across all 28,552 products |
| [`data/Task 1B/models/lstm_best.keras`](data/Task%201B/models/lstm_best.keras) | Best checkpoint saved during training (by validation MAE) |
| [`data/Task 1B/models/lstm_mid_training.weights.h5`](data/Task%201B/models/lstm_mid_training.weights.h5) | Mid-training weight checkpoint |
| [`data/Task 1B/models/baseline_model.joblib`](data/Task%201B/models/baseline_model.joblib) | Seasonal-average baseline (used as comparison benchmark) |
| [`data/Task 1B/processed/scalers.joblib`](data/Task%201B/processed/scalers.joblib) | Per-product MinMax scalers (required for inference) |

**Architecture:** 2-layer LSTM (64 → 32 units), dropout 0.3/0.2, input window 90 days, output horizon 7 days, 3 features per timestep

**Test-set performance:** Normalised MAE = 0.121, Normalised RMSE = 0.192

---

### Task 2 — Produce Quality Grading

#### Production Weights (used by the API)

| File | Description |
|---|---|
| [`models/resnet50_brfn.h5`](models/resnet50_brfn.h5) | ResNet50 production weights — ensemble weight 0.503 |
| [`models/mobilenetv2_brfn.h5`](models/mobilenetv2_brfn.h5) | MobileNetV2 production weights — ensemble weight 0.497 |

#### Training Checkpoints

ResNet50 checkpoints — [`data/Task 2/checkpoints/resnet50/`](data/Task%202/checkpoints/resnet50/):

| File | Description |
|---|---|
| `keras_weights/phase1_best.weights.h5` | Best Phase 1 weights (frozen backbone) |
| `keras_weights/phase2_best.weights.h5` | Best Phase 2 weights (fine-tuned) |
| `periodic_weights/phase1_epoch005.weights.h5` … `phase1_epoch050.weights.h5` | Epoch checkpoints every 5 epochs, Phase 1 |
| `periodic_weights/phase2_epoch005.weights.h5` … `phase2_epoch025.weights.h5` | Epoch checkpoints every 5 epochs, Phase 2 |

MobileNetV2 checkpoints — [`data/Task 2/checkpoints/mobilenetv2/`](data/Task%202/checkpoints/mobilenetv2/) — same structure as ResNet50.

#### Hyperparameter Search Trials

[`data/Task 2/tuner/`](data/Task%202/tuner/) contains 30 Keras Tuner trials for each model. Each trial folder (e.g. `trial_0000/`) holds:
- `trial.json` — hyperparameter configuration tried
- `build_config.json` — model build config
- `checkpoint.weights.h5` — best weights found during that trial

**Ensemble test-set performance (28 classes):** Accuracy = 0.982, Macro F1 = 0.976

---

## 4. Outputs and Evaluation Results

All charts, metrics files, and pre-computed results are committed to the repository and can be viewed without re-running any notebook.

### Task 1A — Recommendations

Location: [`data/Task 1/modelling/`](data/Task%201/modelling/)

| File | Description |
|---|---|
| `evaluation_metrics.png` | Precision/Recall curves across k values |
| `grid_search_heatmap.png` | Grid search heatmap (k vs blend weight) |
| `grid_search_results.json` | Full hyperparameter search results |
| `fairness_summary.json` | Demographic parity metrics |
| `fairness_chart.png` | Fairness visualisation across user segments |
| `shap_examples.png` | SHAP feature importance examples |
| `reorder_vs_new.png` | Distribution of reorder vs new product recommendations |
| `recommendations.json` | Pre-computed top-50 recommendations for all 162,544 users |
| `explanations.json` | Plain-text explanations for each recommendation |
| `test_set_metrics.json` | Final test-set precision, recall, NDCG |

### Task 1B — Forecasting

Location: [`data/Task 1B/models/`](data/Task%201B/models/)

| File | Description |
|---|---|
| `training_history.json` | Train/val loss curves (MAE, RMSE per epoch) |
| `training_history_plot.png` | Training loss visualisation |
| `evaluation_results.json` | Final LSTM vs baseline metrics on test set |
| `model_summary.json` | Full model configuration and evaluation summary |
| `charts/forecast_*.png` | 10 example 7-day forecast charts (one per selected product) |
| `charts/shap_*.png` | 10 SHAP attribution plots showing which input days drove each forecast |

### Task 2 — Quality Grading

Individual model outputs — [`data/Task 2/checkpoints/resnet50/`](data/Task%202/checkpoints/resnet50/) and [`data/Task 2/checkpoints/mobilenetv2/`](data/Task%202/checkpoints/mobilenetv2/):

| File | Description |
|---|---|
| `confusion_matrix.png` | 28×28 class confusion matrix |
| `per_class_f1.png` | Per-class F1 score bar chart |
| `roc_curves.png` | Multi-class ROC curves |
| `confidence_distribution.png` | Prediction confidence histogram |
| `phase1_training_curves.png` | Phase 1 loss and accuracy curves |
| `phase2_training_curves.png` | Phase 2 loss and accuracy curves |
| `evaluation_metrics.json` | Full per-class precision, recall, F1, AUC |
| `best_hyperparameters.json` | Optimal hyperparameters from Keras Tuner |

Ensemble outputs — [`data/Task 2/ensemble/`](data/Task%202/ensemble/):

| File | Description |
|---|---|
| `ensemble_confusion_matrix.png` | Ensemble 28-class confusion matrix |
| `per_class_f1_comparison.png` | Side-by-side per-class F1 for ResNet50, MobileNetV2, and ensemble |
| `ensemble_vs_resnet_f1_delta.png` | Per-class F1 improvement from ensemble over ResNet50 alone |
| `ensemble_metrics.json` | Final test-set accuracy, macro F1, per-class metrics for all model variants |
| `grading_demo.png` | Example grading output with colour/size/ripeness breakdown |

---

## 5. Dataset Download and Placement

Only the **raw source datasets** need to be downloaded. All processed data, model weights, and evaluation outputs are already in the repository.

---

### Task 1A and 1B — Instacart Market Basket Analysis

**Download:** [https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset)

After downloading, extract the archive and place the CSV files at:

```
data/Task 1/raw/InstaCart Dataset/
    ├── aisles.csv
    ├── departments.csv
    ├── order_products__prior.csv
    ├── order_products__train.csv
    ├── orders.csv
    └── products.csv
```

The Task 1B preprocessing notebook reads from the same raw directory.

---

### Task 2 — Fruit and Vegetable Disease Dataset

**Download:** [https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten](https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten)

After downloading, extract so that the class folders sit at:

```
data/Task 2/raw/
    └── Fruit And Vegetable Diseases Dataset/
        ├── Apple__Healthy/
        ├── Apple__Rotten/
        ├── Banana__Healthy/
        ├── Banana__Rotten/
        └── ... (28 class folders total)
```

The notebooks auto-discover this path via `data/Task 2/raw` — no configuration change is needed as long as the folder is placed there.

---

## 6. Environment Setup

### Option A — Local virtualenv

```bash
# Python 3.11 required
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Option B — Google Colab (how notebooks were developed)

The notebooks already contain a Google Drive mount block at the top. Upload the repository to Google Drive, then update `DRIVE_ROOT` in the first cell to your Drive path. No additional installation is needed as Colab includes most dependencies; the notebooks install any extras with `!pip install` cells where required.

### Environment variables (for the API only)

```bash
cp .env.example .env
```

Edit `.env` and set `AI_DATA_ROOT` to the absolute path of this repository root. This is only required to run the FastAPI service — the notebooks handle their own paths independently.

---

## 7. Running the Notebooks

Run notebooks in the order shown in [Section 2](#2-notebooks) within each task. Preprocessing must complete before modelling/training notebooks are run, as later notebooks load artefacts produced by earlier ones.

Because all processed data and model weights are committed to the repository, you **do not need to re-run any notebook** to inspect results — all outputs are already present. Re-running preprocessing or training notebooks from scratch requires the raw datasets (see Section 5) and, for the deep learning notebooks, a GPU.

---

## 8. AI Pipeline Overview

This section describes how the three AI models are integrated into the system at inference time. Infrastructure details (Docker, API framework, etc.) are omitted.

### Overall Flow

When the web application needs an AI prediction, it calls the relevant service module inside `src/`. Each service loads its model once on startup and holds it in memory for the lifetime of the process. Predictions are synchronous — the service receives a request, runs inference, and returns a structured response.

```
Web application → src/api/routes/ → src/<task>/service.py → model → response
```

---

### Task 1A — Product Recommendations

**File:** [`src/recommendation/service.py`](src/recommendation/service.py)

At startup the service loads:
- The fitted KNN model (`knn_model.joblib`)
- The normalised user-item matrix (`user_item_matrix_normalised.npz`)
- The normalised user content profiles (`user_content_profiles_normalised.npz`)
- The pre-computed recommendations and explanations JSON files
- Product and user index mappings

At inference time, given a `user_id`, the service looks up that user's pre-computed top-50 recommendations and returns them split into **reorder** and **new product** buckets. For users not seen during training (cold start), it falls back to global popularity rankings (`global_popularity.json`).

The recommendation score for each product is a weighted blend of three signals:

```
score = 0.4 × frequency_score  +  0.3 × knn_collaborative_score  +  0.3 × content_similarity_score
```

Each returned recommendation includes a plain-text explanation (e.g. "frequently reordered by similar users") derived from whichever signal dominated the score.

**Explainability:** SHAP values were computed offline to identify which user-item interaction features most influenced the KNN similarity scores. These are stored in `shap_examples.png` and `model_summary.json`.

---

### Task 1B — Demand Forecasting

**File:** [`src/forecasting/service.py`](src/forecasting/service.py)

At startup the service loads:
- The production LSTM model (`lstm_forecast.keras`)
- Per-product MinMax scalers (`scalers.joblib`)
- Qualifying product list and seasonal configuration

At inference time the service receives:
- An Instacart `product_id`
- A sequence of 90 daily purchase counts (the input window)
- The window end date (used to derive seasonal features)

The service scales the input using the product's stored scaler, prepends the three engineered features (count, day-of-week sine/cosine encoding), runs a forward pass through the LSTM, then inverse-transforms the 7-element output back to order-count units.

**SHAP attribution:** After inference, SHAP DeepExplainer identifies the top-10 input days that most influenced the 7-day forecast. These are returned in the API response alongside the forecast values, and are also visualised in the pre-generated `charts/shap_*.png` outputs.

**Model architecture:** Global single model — one LSTM trained across all 28,552 products simultaneously (not per-product). Products are distinguished through their individual scalers and the seasonal features injected as input.

---

### Task 2 — Produce Quality Grading

**Files:** [`src/quality_assessment/service.py`](src/quality_assessment/service.py), [`src/quality_assessment/gradcam.py`](src/quality_assessment/gradcam.py)

At startup the service loads both production models:
- ResNet50 (`models/resnet50_brfn.h5`) with ensemble weight 0.503
- MobileNetV2 (`models/mobilenetv2_brfn.h5`) with ensemble weight 0.497

Both models were trained in two phases:
1. **Phase 1 (feature extraction):** ImageNet backbone frozen; only the classification head trained
2. **Phase 2 (fine-tuning):** Top layers of the backbone unfrozen and trained at a low learning rate

At inference time, an uploaded image is:
1. Resized to 224×224 and normalised using each backbone's native preprocessing
2. Passed through each model independently to obtain a 28-class probability vector
3. Linearly combined using the optimised weights: `ensemble = 0.503 × resnet50 + 0.497 × mobilenetv2`
4. The winning class (e.g. `Apple__Healthy`) is mapped to a colour score, size score, and ripeness score using the HSV and size reference tables (`colour_references.json`, `size_references.json`)
5. The three component scores are combined into an overall **grade**:
   - **Grade A:** Colour ≥ 75%, Size ≥ 80%, Ripeness ≥ 70%
   - **Grade B:** Colour ≥ 65%, Size ≥ 70%, Ripeness ≥ 60%
   - **Grade C:** Below Grade B thresholds

**Explainability (Grad-CAM):** After grading, [`src/quality_assessment/gradcam.py`](src/quality_assessment/gradcam.py) generates a Grad-CAM heatmap from the final convolutional layer of the ResNet50 model, overlaid on the original image. This highlights which regions of the produce image drove the classification decision, providing visual transparency to producers.

The standalone grading module [`data/Task 2/ensemble/grading.py`](data/Task%202/ensemble/grading.py) implements the same logic and can be run independently of the API for batch evaluation or testing.

# Credit Scoring Model — Predicting Serious Delinquency

A machine learning project that predicts which borrowers are likely to default, covering data analysis, feature engineering, model comparison and interpretation with SHAP.

## Problem

The goal is to predict whether a borrower will experience serious delinquency (90+ days past due) within two years — the binary target `SeriousDlqin2yrs` (referred to as default below).

The data come from the Kaggle competition "Give Me Some Credit" and contain 150,000 borrowers with 10 features.

The classes are imbalanced: about 7% of borrowers default.

## Approach

The project consists of 3 notebooks, each focusing on a specific stage of the machine learning pipeline: Exploratory Data Analysis (EDA), Feature Engineering, and Modeling. Imputation and percentile capping are part of the model pipeline and are fitted on the training folds only, to prevent data leakage.

## Results

### Model comparison (5-fold cross-validation on the training set)

| Model | Train AUC | CV AUC | Gini |
|---|---|---|---|
| Logistic regression (baseline) | 0.857 | 0.857 | 0.714 |
| XGBoost – default parameters | 0.941 | 0.842 | 0.684 |
| XGBoost – manual parameters | 0.875 | 0.865 | 0.729 |
| **XGBoost – Optuna-tuned (final)** | **0.876** | **0.865** | **0.730** |

XGBoost with default parameters overfits (train AUC 0.941 vs CV 0.842). The final model is better than logistic regression in every fold, but the gain is small. Optuna tuning (30 trials) brings only a marginal improvement over the manual configuration.

### Final model on the test set

| AUC | Gini |
|---|---|
| 0.870 | 0.740 |

### Decision threshold on the test set (0.8, selected by F1 on out-of-fold training predictions)

| Precision | Recall | Rejected applications |
|---|---|---|
| 0.42 | 0.48 | 7.5% |

## Key findings

- **Special codes 96/98 in the delinquency counters.** 269 records have values 96 or 98 in all three counters. These are far outside the normal range and distort the correlations. The default rate in this group is much higher than in the rest of the data, so instead of removing these records, the codes were replaced with -1 and a flag `IsSpecialStatus` was added.

- **`DebtRatio` is not a ratio when income is missing.** 94% of records without income data have `DebtRatio` above 1, so for these records the value is not debt divided by income. `DebtRatio` was set to missing for records with missing or zero income, and a flag `IsIncomeMissing` was added.

- **Most important features.** `RevolvingUtilizationOfUnsecuredLines` and past-due history (`TotalNumberPastDue`, the sum of the delinquency counters) have the strongest impact on predictions, both in the EDA and in SHAP. The default rate rises from 2.5% in the lowest utilization decile to 23% in the highest, and a single 90+ days past-due event increases it from 4.6% to 33.7%.

## Project structure

```
credit-scoring-model/
├── data/
│   └── README.md            # how to download the dataset (data files are not tracked)
├── notebooks/
│   ├── 01_eda.ipynb         # exploratory data analysis and data quality checks
│   ├── 02_features.ipynb    # data cleaning and feature engineering
│   └── 03_modeling.ipynb    # model training, evaluation and SHAP analysis
├── src/
│   ├── preprocessing.py     # capping transformer and preprocessing pipeline
│   └── evaluation.py        # cross-validation helper
├── requirements.txt
└── README.md
```

## How to run

Tested with Python 3.12.

1. Install dependencies: `pip install -r requirements.txt` (a virtual environment is recommended).
2. Download the data as described in [`data/README.md`](data/README.md).
3. Run the notebooks in order: `01_eda.ipynb`, `02_features.ipynb` (creates `data/processed.parquet`), `03_modeling.ipynb`.

On macOS, XGBoost also requires OpenMP: `brew install libomp`.

## Limitations / Future work

**Limitations**
- The model's scores are not calibrated, mainly because of `scale_pos_weight`, so they work as a ranking score rather than a direct probability of default.
- The threshold is chosen by F1, which gives equal weight to wrongly rejected good clients and missed defaults. In practice, the cost of these two errors may differ.
- The data come from a single public dataset without a time dimension, so the stability of the model over time cannot be checked.

**Future work**
- Add unit tests for `CappingTransformer`.
- Move feature engineering to a function and add a `train.py` script that trains and saves the model without notebooks.
- Serve the model through a REST API (FastAPI) in a Docker container.
- Calibrate the model's scores so they can be used directly as probability of default.

## Author
[Barbara Russ](https://www.linkedin.com/in/barbara-russ)
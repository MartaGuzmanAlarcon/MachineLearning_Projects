
# 🏦 Loan Approval Classification Prediction

Classify loan approval from applicant, income, loan and credit-history attributes.

## Overview

This project is a focused, reproducible classification case study, split across two notebooks:

- **`loan_approval_classification.ipynb`** — the core pipeline: data checks, a first pass at EDA,
  leakage-safe preprocessing, a dummy baseline, cross-validated model comparison, and holdout evaluation
  with interpretation.
- **`ClaudeLoanApproval.ipynb`** — a supplementary notebook that adds a deeper EDA, scores the unlabeled
  test set, and adds an error analysis.

## Problem Statement

| | |
|---|---|
| **Task** | Classification |
| **Target** | `Loan_Status` |
| **Primary metric** | Macro F1 |
| **Goal** | Classify loan approval from applicant, income, loan and credit-history attributes |

## Dataset

- **Availability:** Included — `loan-train.csv`, `loan-test.csv`
- **Recorded source:** Training and scoring files are included in this project
- **Target:** `Loan_Status` (present in `loan-train.csv` only; `loan-test.csv` is unlabeled)

Dataset licensing and usage conditions remain with the original publisher. Large or externally hosted data
is intentionally not duplicated here.

## 📓 Notebooks

### `loan_approval_classification.ipynb`
The main notebook. Loads and cleans `loan-train.csv`, does a first pass of EDA, splits off an untouched
holdout set, builds a leakage-safe preprocessing pipeline, compares a dummy baseline against three models
with cross-validation, evaluates the selected model on the holdout set, and interprets it with permutation
importance.

### `ClaudeLoanApproval.ipynb`
A supplementary notebook that fills three gaps left open by the main notebook:

- **Deeper EDA** — numeric feature distributions, numeric features vs. `Loan_Status`, a correlation check,
  and approval rate by category for every categorical feature (`Credit_History` stands out as the strongest
  single predictor).
- **Scoring `loan-test.csv`** — this file is described as an optional unlabeled scoring set but was not
  used anywhere in the main notebook. This notebook loads it, cleans it the same way as the training data,
  generates predictions with the selected model, and saves them to `loan-test-predictions.csv`.
- **Error analysis** — breaks down holdout mistakes into false positives and false negatives, profiles
  their feature values, and checks whether prediction accuracy holds steady across subgroups such as
  `Credit_History`, `Gender`, `Married`, `Property_Area` and `Education`.

It briefly re-runs the main notebook's loading, cleaning, splitting, preprocessing and model-selection
steps first, using the same method and models, so it has a fitted model to build the new analysis on.

## 🔄 Project Workflow

```
Data validation
      ↓
Focused EDA (main notebook) + Extended EDA (supplementary notebook)
      ↓
Train / holdout split
      ↓
Pipeline-based preprocessing
      ↓
Baseline and cross-validation
      ↓
Holdout evaluation
      ↓
Error analysis and interpretation (supplementary notebook)
      ↓
Scoring the unlabeled test set (supplementary notebook)
```

## 🛠️ Modelling Decisions

- The Kaggle-only absolute path was replaced with a portable project-relative path.
- `loan-test.csv` is an unlabeled scoring set; `ClaudeLoanApproval.ipynb` loads it, generates predictions,
  and saves them, since it has no target column to evaluate against.

## Models Compared

- Logistic Regression
- Decision Tree
- Random Forest

## Evaluation

Both notebooks evaluate macro F1 and accuracy, with class-level precision/recall and a confusion matrix.
`ClaudeLoanApproval.ipynb` adds a breakdown of holdout errors by type (false positive vs. false negative)
and by subgroup.

## Verified Results

The refurbished notebook was executed successfully against the dataset currently committed in this
project. The untouched holdout produced:

| Selected model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | 0.8618 | 0.8147 |

`ClaudeLoanApproval.ipynb` selects the same model using the same method, and its predictions on
`loan-test.csv` are saved to `loan-test-predictions.csv`.

> These values are a reproducibility record for the current data and dependency range, not a production
> benchmark.

## Repository Structure

```
├── loan_approval_classification.ipynb
├── ClaudeLoanApproval.ipynb
├── README.md
├── loan-train.csv
└── loan-test.csv
```


## Tools & Libraries

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Jupyter

## Future Improvements

- Validate on a newer or independently collected dataset.
- Add domain-specific error costs and decision thresholds.
- Track data drift and subgroup performance before deployment.
- Package the fitted pipeline only after data and licensing checks.

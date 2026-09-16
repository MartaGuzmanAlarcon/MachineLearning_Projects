# Loan Approval Classification

Predicting loan approval from applicant, income, and credit-history data — with a focus on leakage-safe evaluation and subgroup fairness testing, not just a headline accuracy number.

## Overview

End-to-end, reproducible classification case study: data validation → EDA → leakage-safe preprocessing → baseline + cross-validated model comparison → holdout evaluation → scoring an unlabeled test set → subgroup error analysis with statistical testing.

| | |
|---|---|
| **Task** | Binary classification |
| **Target** | `Loan_Status` (Y / N) |
| **Primary metric** | Macro F1 |
| **Selected model** | Logistic Regression |
| **Holdout result** | 0.86 accuracy · 0.81 macro F1 |

## Dataset

- `loan-train.csv` — labeled, used for training/holdout (614 rows)
- `loan-test.csv` — unlabeled, scored only (no ground truth available)
- Features: applicant demographics, income, loan amount/term, and credit history

## Project Workflow

```
Data validation → Focused EDA → Train/holdout split → Pipeline-based
preprocessing → Baseline + cross-validation → Holdout evaluation →
Scoring the unlabeled test set → Error analysis & subgroup testing
```

## Modelling Decisions

- **Median imputation for numeric features** — `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount` are right-skewed with outliers, so the median stays representative where the mean wouldn't.
- **Most-frequent imputation for categoricals** — avoids inventing a near-empty "Missing" category.
- **`Credit_History` note** — stored as numeric (0/1), so it's routed through the numeric branch rather than the categorical one. In practice this barely matters: its distribution is heavily skewed toward 1.0, so median and mode coincide.
- **`StandardScaler` on numeric features** — needed for Logistic Regression, otherwise `ApplicantIncome` (thousands) would dominate `Credit_History` (0/1) purely from scale.
- **One-hot, not ordinal, encoding** — none of the categorical features have a natural order.
- **No leakage** — all imputation values, scaling parameters, and encoded categories are fit on the training split only, inside the pipeline. Holdout and scoring data never influence fitting.

## Models Compared

| Model | Notes |
|---|---|
| Dummy (majority class) | Baseline any real model must beat |
| Logistic Regression | **Selected** — best CV macro F1 |
| Decision Tree | `max_depth=8`, `min_samples_leaf=4` |
| Random Forest | Fixed hyperparameters, not tuned |

Hyperparameters were fixed by hand, not grid-searched (see Future Improvements).

## Evaluation

- **Model selection:** 5-fold stratified cross-validation on macro F1
- **Holdout evaluation:** accuracy, macro F1, classification report, confusion matrix — the only point where predictions are checked against ground truth the model never trained on
- **Subgroup check:** each subgroup's error count tested with a two-sided binomial test against the overall holdout error rate (14%), rather than judged by eye

## Verified Results

| Selected model | CV macro F1 | Holdout accuracy | Holdout macro F1 |
|---|---|---|---|
| Logistic Regression | 0.707 ± 0.077 | 0.8618 | 0.8147 |

- **Strongest predictor:** `Credit_History` — by far the biggest driver of approval; income and loan amount barely separate approved from rejected applicants on their own.
- **By class:** "Y" (approved) — 0.84 precision / 0.99 recall, rarely misses a true approval. "N" (rejected) — 0.96 precision / 0.58 recall: when the model predicts rejection it's almost always right, but it only catches ~half of true rejections (22 of 38), defaulting to approval when uncertain.
- **Generalization:** holdout macro F1 (0.815) landed above CV mean + 1 std (up to 0.784) — a favorable split, not a sign of overfitting, but a small holdout and high CV variance mean this is a reasonable result rather than a guarantee.
- **Fairness finding:** `Property_Area = Rural` shows a statistically significant accuracy gap (0.676 vs. 0.86 overall, p = 0.005). No other subgroup (`Credit_History`, `Gender`, `Married`, `Education`) showed a significant gap.
- Predictions for the unlabeled `loan-test.csv` were generated and saved to `loan-test-predictions.csv`.

*These are a reproducibility record for the current data/dependency range, not a production benchmark.*

## Repository Structure

```
├── LoanApproval_Classification.ipynb
├── README.md
├── loan-train.csv
├── loan-test.csv
└── loan-test-predictions.csv
```

## How to Run

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn jupyter
jupyter notebook LoanApproval_Classification.ipynb
```

Run all cells top to bottom — the notebook is self-contained and regenerates `loan-test-predictions.csv` in Section 6.

## Technologies

Python · pandas · NumPy · scikit-learn · SciPy (`binomtest`) · Matplotlib · Seaborn · Jupyter

## Future Improvements

- Validate on a newer or independently collected dataset
- Address the confirmed `Rural` accuracy gap — more rural holdout data or a rural-specific error-cost review — before real-world use
- Add domain-specific error costs and decision thresholds (model currently defaults toward approving when uncertain)
- Add multiple-testing correction (e.g. Bonferroni) if subgroup testing is extended to more features
- Tune hyperparameters via grid/randomized search
- Track data drift and subgroup performance before deployment
- Package the fitted pipeline only after data and licensing checks

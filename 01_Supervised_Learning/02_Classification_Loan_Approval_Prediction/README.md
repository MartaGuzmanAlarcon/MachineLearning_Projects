# Loan Approval Classification

Predicting loan approval from applicant, income, and credit-history data.

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

| CV macro F1 | Holdout accuracy | Holdout macro F1 |
|---|---|---|
| 0.707 ± 0.077 | **0.862** | **0.815** |

| Class | Precision | Recall |
|---|---|---|
| Y — approved | 0.84 | 0.99 |
| N — rejected | 0.96 | 0.58 |

**🔑 Key takeaways**
- `Credit_History` >> income or loan amount as a predictor.
- Model defaults to "approve when unsure" — misses 16/38 true rejections, almost never rejects a real approval.
- Holdout score beat CV mean + 1 std → favorable split, not proof of generalization (small data, high CV variance).

** Fairness gap** — `Property_Area = Rural`: 0.676 accuracy vs. 0.86 overall (p = 0.005 ✅ real gap). No other subgroup significant.

**🔍 Why it fails** — of 16 false positives, 100% had `Credit_History = 1`: the model over-trusts good credit history. Income runs a bit higher too, but that's n=16 and untested — not a confirmed pattern.

## Repository Structure

```
├── LoanApproval_Classification.ipynb
├── README.md
├── loan-train.csv
├── loan-test.csv
└── loan-test-predictions.csv
```


## Technologies

Python · pandas · NumPy · scikit-learn · SciPy (`binomtest`) · Matplotlib · Seaborn · Jupyter

## Future Improvements

- Validate on a newer or independently collected dataset
- Address the confirmed `Rural` accuracy gap — more rural holdout data or a rural-specific error-cost review — before real-world use
- Add domain-specific error costs and decision thresholds (model currently defaults toward approving when uncertain)
- Add multiple-testing correction (e.g. Bonferroni) if subgroup testing is extended to more features
- Tune hyperparameters via grid/randomized search
- Track data drift and subgroup performance before deployment

# Breast Cancer Classification

Classifying breast tumours as benign or malignant from 30 cell-nucleus measurements, using seven machine learning models and a stacking ensemble. Recall on malignant tumours gets the most attention, since missing a cancer is the costliest mistake.

## Project Overview

The data is the Breast Cancer Wisconsin (Diagnostic) dataset: 569 patients, each described by 30 numeric measurements computed from a digitised image of a fine-needle aspirate of the breast mass, plus the final diagnosis. Ten characteristics of the cell nuclei (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension) are each summarised as a mean, a standard error and a "worst" value, which gives the 30 columns. 357 patients (62.7%) have a benign tumour and 212 (37.3%) a malignant one.

The notebook goes from the raw CSV to a model comparison:

- clean and explore the data
- add six derived features
- split the patients into training and test sets, keeping the class balance
- fit all preprocessing on the training set only
- tune seven models with cross-validation and combine them in a stacking ensemble
- score every model once on the same held-out patients
- check which measurements drive the predictions

The last section describes what would still be needed before a model like this could be used with real patients. This is a learning project and not a diagnostic tool.

## Libraries & Tools

- **Python 3** and **Jupyter** (JupyterLab)
- **pandas** and **NumPy** for data handling
- **Matplotlib** and **seaborn** for plots
- **scikit-learn** for pipelines, scaling, PCA, cross-validated grid search, most of the models and all the metrics
- **XGBoost** for gradient boosting

To run the notebook, put `Breast Cancer dataset.csv` in the same folder, install the packages above and run all cells in order. A single random seed (42) is used everywhere, so the same setup should give the same numbers. Results can still shift a little between environments: XGBoost, and with it the stacking ensemble, differed by about one patient when the notebook was run on two different setups.

## Project Structure

breast-cancer-classification/
├── README.md
├── BreastCancerPredictionClassification.ipynb   # full analysis
├── Breast Cancer dataset.csv                    # data
└── requirements.txt                             # packages

## Workflow

### Data collection and cleaning

The CSV has 569 rows and 33 columns. The `id` column carries no medical information and `Unnamed: 32` is completely empty, so both are dropped. The diagnosis is converted to a number (malignant = 1, benign = 0). There are no missing values and no duplicated rows, so no imputation is needed.

### Exploratory data analysis

- **Class balance.** The split is 63% benign and 37% malignant, which is mild enough to work without resampling. It does mean accuracy alone would be misleading, so recall and ROC-AUC are reported too.
- **Benign vs malignant.** Malignant tumours differ most in border shape. The average `concavity_mean` is about 3.5 times higher and `concave points_mean` about 3.4 times higher, `area_mean` about 2.1 times, and radius and perimeter only about 1.4 to 1.5 times. `fractal_dimension_mean` is practically the same in both groups.
- **Skewness.** 22 of the 30 features are strongly skewed, mostly the standard-error columns.
- **Outliers.** The IQR rule flags 608 values across all features, concentrated in the standard-error columns (`area_se` is the highest, with 11.4% of rows). These look like genuinely large tumours and not data-entry errors (an assumption, since the source images can't be checked), so the plan is to cap them and keep every patient.
- **Correlation.** 21 feature pairs correlate above 0.9, mostly radius, perimeter and area, which all describe tumour size. The features most correlated with malignancy are `concave points_worst` (0.79), `perimeter_worst` (0.78), `concave points_mean` (0.78) and `radius_worst` (0.78).

### Feature engineering

Six features are added to describe shape and variability and not only size:

- perimeter squared over area, for the mean and worst values (how ragged the outline is)
- worst over mean ratio for radius and for texture (how much more extreme the worst region is than the average)
- `radius_mean` times `radius_se`
- `concavity_mean` times `concave points_mean`

The most correlated with the diagnosis are `radius_worst_over_mean` (0.64) and the concavity product (0.63). All six are row-by-row calculations, so they can be created before the split without leaking anything.

### Split and preprocessing

- **Stratified 70/30 split.** 398 patients for training and 171 for testing, with 37.2% and 37.4% malignant. A 30% test set leaves about 64 malignant cases to evaluate on, so one missed tumour weighs less than it would in a smaller test set.
- **Capping.** Every feature is clipped at its 1st and 99th percentile. The limits are learned from the training set and applied unchanged to the test set (260 training values, 1.8%, were modified).
- **Scaling and PCA.** Both live inside each model's pipeline, so they are fitted only on the training folds. Two components hold 63% of the variance and 11 components keep 95% of the information in the 36 features.
- **Tuning.** Every model uses `GridSearchCV` with 5-fold stratified cross-validation and ROC-AUC as the selection metric.

## Models

| Model | Pipeline | Best settings | CV ROC-AUC |
|---|---|---|---|
| Logistic Regression | scale, PCA, model | 15 components, C ≈ 0.32 | 0.9966 |
| Decision Tree | scale, model | entropy, max depth 5, min leaf 10 | 0.9582 |
| Random Forest | scale, model | 600 trees, max depth 8, √features per split | 0.9887 |
| KNN | scale, PCA, model | 13 components, 25 neighbours, distance weights | 0.9923 |
| Gaussian Naive Bayes | scale, PCA, model | 5 components | 0.9786 |
| SVM (RBF) | scale, PCA, model | 15 components, C ≈ 158, gamma ≈ 0.0006 | 0.9976 |
| XGBoost | scale, model | learning rate 0.1, max depth 2, subsample 0.8, lambda 0.1 | 0.9939 |
| Stacking ensemble | out-of-fold probabilities of the seven models, then logistic regression | C ≈ 31.6 | 0.9970 |

PCA is used where redundancy causes trouble: distances in KNN, the independence assumption in Naive Bayes, and unstable coefficients in logistic regression. Trees and boosting work with the original features. For the stacking ensemble, the training input comes from cross-validated predictions, so every probability was produced by a model that had not seen that patient. The seven base models are then refitted on the full training set to predict on the test set.

## Evaluations & Predictions

All models are scored once on the same 171 held-out patients (107 benign and 64 malignant).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Missed cancers |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9766 | 1.0000 | 0.9375 | 0.9677 | 0.9990 | 4 |
| XGBoost | 0.9766 | 1.0000 | 0.9375 | 0.9677 | 0.9987 | 4 |
| KNN | 0.9474 | 1.0000 | 0.8594 | 0.9244 | 0.9978 | 9 |
| Stacking ensemble | 0.9825 | 1.0000 | 0.9531 | 0.9760 | 0.9978 | 3 |
| SVM (RBF) | 0.9708 | 0.9836 | 0.9375 | 0.9600 | 0.9974 | 4 |
| Random Forest | 0.9708 | 1.0000 | 0.9219 | 0.9593 | 0.9974 | 5 |
| Gaussian Naive Bayes | 0.9298 | 0.9643 | 0.8438 | 0.9000 | 0.9864 | 10 |
| Decision Tree | 0.9474 | 0.9661 | 0.8906 | 0.9268 | 0.9854 | 7 |

How to read it:

- Six of the eight models have a ROC-AUC within 0.002 of each other, so ROC-AUC alone cannot tell them apart. The differences show up in recall, which goes from 84% (Naive Bayes) to 95% (stacking).
- KNN has a high ROC-AUC but the second-lowest recall. It ranks patients well, but at the default 0.5 cut-off it misses 9 of the 64 malignant tumours.
- Five models raise no false alarms. Logistic regression, the simplest model, ties XGBoost in accuracy and recall.
- ROC-AUC picks logistic regression as the best model. If recall is the priority, the stacking ensemble is the better choice with 3 missed cancers against 4, but that is a single patient and not conclusive.

**Feature importance.** Both the random forest's built-in importance and permutation importance (computed on the test set) are led by worst-case size and border measurements. `perimeter_worst`, `radius_worst`, `area_worst`, `concave points_worst` and `concave points_mean` are in the top 10 of both. Among the engineered features, the concavity product ranks 3rd in both rankings and `radius_worst_over_mean` ranks 1st by permutation, while the two perimeter-to-area ratios rank near the bottom (26th to 34th out of 36).

**Predictions on new patients.** The notebook does not include a deployed predictor. Each model is a full pipeline, so a new patient would go through the same steps as the training data (derived features, capping with the training limits, scaling, PCA) and come out with a probability of malignancy. The last section lists what would come before real use: validation on other hospitals' data, saving the whole pipeline as one file, checking every new input, choosing the threshold together with clinicians, keeping a person in the loop, and monitoring performance after launch.

## Key Findings

1. **Size and border irregularity carry the signal.** The worst-case measurements of radius, perimeter, area and concave points separate the two classes best, and the highest ratios between classes are in concavity and concave points.
2. **The data is very redundant.** 21 pairs of features correlate above 0.9 and 11 components keep 95% of the information, so PCA works well and logistic regression on components reaches a ROC-AUC of 0.999.
3. **Model choice matters less than expected.** Logistic regression, random forest, SVM and XGBoost land within about one percentage point of accuracy, and the simple model is among the best.
4. **Accuracy and ROC-AUC hide the differences that matter.** The models mainly differ in how many malignant tumours they miss at a fixed threshold, which is why recall is reported next to them.
5. **Some engineered features helped and others did not.** The concavity product and the worst-over-mean radius ratio ranked among the most important features. The perimeter-to-area ratios add little and could be dropped.
6. **The test set is small.** With 64 malignant patients, one missed tumour changes recall by 1.6 points, so the gaps between the best models are within noise.

## Future Improvements

- **Validate on external data.** The training and test sets come from the same source, so the results are probably optimistic. Data from other hospitals and equipment would give a more honest estimate.
- **Quantify the uncertainty.** Repeated or nested cross-validation, or bootstrapped confidence intervals, would show how much of the difference between models is real.
- **Choose the decision threshold properly.** Pick it with cross-validated predictions on the training set and the clinical cost of each error, and check probability calibration and the precision-recall curve as well.
- **Widen the search ranges.** Several best settings sit at the edge of the grid (15 components for logistic regression and SVM, 25 neighbours for KNN), so larger ranges could change the result.
- **Try feature selection instead of PCA.** Keeping a smaller set of original measurements would be easier to explain, and the importance results suggest a compact panel could work.
- **Explain individual predictions.** Tools such as SHAP would show why a specific patient gets a given probability.
- **Package the pipeline.** Save the fitted pipeline with pinned library versions and wrap it in a small script or API, with monitoring and a retraining plan.

# ML Concepts Cheat Sheets

My own theory notes on the main machine learning algorithms, with the aim to keep the concepts clear while I work on the projects in this repository. I'm sharing them in case they help someone else.

## What's inside

One page per algorithm, eight in total, in the PDF [`ML_Concepts_Cheat_Sheets.pdf`](./ML_Concepts_Cheat_Sheets.pdf):

| # | Algorithm | Type | Typical uses on the page |
|---|---|---|---|
| 1 | Linear Regression | Regression | House prices, salary prediction, sales forecasting |
| 2 | Logistic Regression | Binary classification | Spam detection, disease prediction |
| 3 | Decision Trees | Classification and regression | Customer segmentation, credit risk |
| 4 | K-Nearest Neighbors (KNN) | Classification and regression | Recommender systems, pattern recognition |
| 5 | Random Forest | Ensemble (bagging) | Fraud detection, financial risk, disease prediction |
| 6 | Naive Bayes | Probabilistic classification | Text classification, spam filtering, sentiment analysis |
| 7 | Support Vector Machine (SVM) | Classification | Face detection, image classification |
| 8 | Gradient Boosting | Ensemble (boosting) | Spam detection, credit scoring, healthcare diagnosis, fraud detection |

## How each page is organised

Every algorithm follows the same layout, so the pages are easy to compare:

1. Definition and intuition, with a one-line pipeline of the steps
2. Mathematical equations
3. Geometrical interpretation
4. When to use it, with advantages and disadvantages
5. A short Python implementation with scikit-learn
6. Assumptions and key hyperparameters

Each page ends with a common pitfall, a tip, a practice question and a one-line summary.

## My annotations

The pages are covered in colour highlights and handwritten notes. I used them to mark the key points and to remember details that are easy to forget: for example, that L1 regularisation can remove features while L2 only shrinks them, that PCA helps when features are highly correlated, or that scaling gives regularisation a fair start. I also added worked examples for entropy and Gini, and answered the practice questions by hand. Some of the handwritten notes are in Spanish.

## Notes on the code

The Python snippets use tiny toy datasets. They show the workflow (split, fit, predict, evaluate) and are not meant as real analyses. To see these algorithms on a real dataset, look at [`03_BreastCancerPrediction_Classification`](../03_BreastCancerPrediction_Classification), where logistic regression, decision tree, random forest, KNN, Naive Bayes, SVM and XGBoost (a gradient boosting library) are compared, and at [`01_Linear_Regression_Diamond_Price`](../01_Linear_Regression_Diamond_Price) for linear regression.

These are study notes and not a textbook. If you spot a mistake, feel free to open an issue.

## Files

```
00_ML_Concepts_Cheat_Sheets/
├── README.md
└── ML_Concepts_Cheat_Sheets.pdf
```

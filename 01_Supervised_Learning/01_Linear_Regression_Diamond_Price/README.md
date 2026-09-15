# Diamond Price Prediction 💎

A machine learning project focused on predicting diamond market prices using regression models. This repository explores how different features impact financial valuation.

## Project Overview 

The core objective is to build an end-to-end predictive pipeline that estimates diamond prices and to study which feature matter the most. The workflow covers data collection, thorough exploratory analysis, feature engineering, model training, and performance evaluation.

## Tools and Libraries

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-Learn
* Jupyter Notebook

## Workflow Summary 

1. **Data Collection:** Includes diamond attributes, such as dimensions, physical grades, and target pricing, sourced from standard tabular datasets.
2. **Exploratory Data Analysis:** Uncovers underlying distributions and correlations through visual inspection, examining how carat weight and categorical qualities interact with price.
3. **Feature Engineering:** Prepares categorical attributes through encoding techniques, creates custom features, and applies transformations to address skewed distributions before splitting data into training and testing sets.
4. **Modelling:** Developed a machine learning pipeline incorporating the following regression algorithms:
   * **Linear Regression:** Initial baseline model.
   * **Random Forest Regressor:** Delivered strong predictive performance.
   * **Gradient Boosting Regressor / XGBoost:** Implemented to achieve further performance gains.
   * **Hyperparameter Tuning:** Performed via cross-validation, optimizing parameters such as `n_estimators`, `max_depth`, and `learning_rate`.
5. **Evaluation:** Assessed model performance using standard evaluation metrics:
   * **Mean Absolute Error (MAE)**
   * **Root Mean Squared Error (RMSE)**
   * **R² (Coefficient of Determination)**
   * **Result:**  The tuned Gradient Boosting model was the top performer, with an RMSE of about $517 and R² of about 0.983, highlighting robust predictive power and valuable insights into feature importance. 
6. **Predictions & Business Insights:** 
   * **Price Estimation:** Generated individual price estimates using the trained model and compared predicted values against actual prices. 
   * **Feature Importance Analysis:**  Confirmed through feature importance metrics that carat weight (and the engineered features derived from it, like volume and carat squared) acts as the primary price driver, while cut, clarity, and color contribute meaningfully but far less.
   * **Market Insights:** : * **Market Insights:** Feature importance confirms that quality grades (clarity, color, cut) do add real value to a diamond's price, but carat size remains by far the dominant factor. This suggests size plays the biggest role in a diamond's market value, with quality contributing a smaller, secondary effect.

## Project Structure 

```text
01_Linear_Regression_Diamond_Price/
├── DiamondPricePredictionDataSet.csv
├── Diamond_Price_Prediction.ipynb
└── README.md
```
## Key Findings 

 
* Carat weight serves as the main driver behind market pricing, while quality characteristics like cut, color, and clarity contribute meaningful secondary value.
* Applying a logarithmic transformation to the target variable effectively normalized residuals and improved overall model fit.
* Non-linear tree-based regressors outperformed standard linear models by successfully capturing complex relationships between the features and price. In this run, the untuned Linear Regression baseline performed far worse than every tree-based model (RMSE in the hundreds of thousands vs. around $500-720 for tree-based models), underlining how poorly a straight-line model fits this data.
* Engineering custom interaction variables, such as combining carat and depth, noticeably enhanced the overall accuracy of the models.
* The best-performing model overall was the tuned Gradient Boosting Regressor (MAE ≈ $262, RMSE ≈ $517, R² ≈ 0.983), narrowly ahead of the tuned Random Forest Regressor.

## Future Improvements 

* Incorporate external market factors, such as regional pricing shifts or dealer premiums, to enrich the feature space and capture broader economic trends.
* Build an interactive web application to deploy the trained model, allowing users to calculate real-time diamond valuations effortlessly.
* Experiment with advanced model stacking and blending techniques to drive prediction errors even lower.
* Integrate explainability tools like SHAP values to provide transparent and intuitive breakdowns for individual price estimates.
* Implement ongoing tracking practices to monitor model reliability and performance as real-world market dynamics shift over time.
* Add a carat-controlled price analysis (grouping diamonds into similar-size buckets and comparing
  price-per-carat within each group) to properly validate and quantify the quality premium claim, rather
  than relying on feature importance alone.

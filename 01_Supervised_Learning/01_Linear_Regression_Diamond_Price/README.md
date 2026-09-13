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
   * **Result:** The top-performing model yielded a low RMSE and a high R², highlighting robust predictive power and valuable insights into feature importance.
6. **Predictions & Business Insights:** VOLVER
   * **Price Estimation:** Generated individual price estimates using the trained model and compared predicted values against actual prices. 
   * **Feature Importance Analysis:** Confirmed through feature importance metrics that carat weight acts as the primary price driver, while cut, clarity, and color play substantial secondary roles.
   * **Market Insights:** : Analyzed how much extra market value (or price premium) buyers pay for better clarity and cut quality, keeping the diamond's size (carat) constant. VOLVER

## Project Structure 
01_Linear_Regression_Diamond_Price/
├── Dataset.txt
├── Diamond_Price_Prediction.ipynb
└── README.md
├── data/
├── notebooks/
├── src/
├── README.md
└── requirements.txt

## Key Findings VOLVER

* Carat weight serves as the main driver behind market pricing, while quality characteristics like cut, color, and clarity contribute meaningful secondary value.
* Applying a logarithmic transformation to the target variable effectively normalized residuals and improved overall model fit.
* Non-linear tree-based regressors outperformed standard linear models by successfully capturing complex relationships between the features and price.
* Engineering custom interaction variables, such as combining carat and depth, noticeably enhanced the overall accuracy of the models.

## Future Improvements VOLVER

* Incorporate external market factors, such as regional pricing shifts or dealer premiums, to enrich the feature space and capture broader economic trends.
* Build an interactive web application to deploy the trained model, allowing users to calculate real-time diamond valuations effortlessly.
* Experiment with advanced model stacking and blending techniques to drive prediction errors even lower.
* Integrate explainability tools like SHAP values to provide transparent and intuitive breakdowns for individual price estimates.
* Implement ongoing tracking practices to monitor model reliability and performance as real-world market dynamics shift over time.

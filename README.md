# LINEAR REGRESSION – HOUSE PRICE (TASK 3)

This project implements both Simple and Multiple Linear Regression on a house price dataset.
It includes data loading, preprocessing, model training, evaluation, coefficient interpretation, and output saving.

## PROJECT FILES

- linear_regression_house.py – Main script
- lr_outputs/ – Auto-generated output folder
  - cleaned_numeric_sample.csv
  - simple_reg_<feature>.png
  - lr_coefficients.csv
  - residuals_plot.png
  - linear_regression_pipeline.joblib

## REQUIREMENTS

- Python 3.8+
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, joblib

Install dependencies:
pip install pandas numpy matplotlib seaborn scikit-learn joblib

## HOW THE SCRIPT WORKS

1. Automatically detects and loads a house price dataset (CSV/XLSX).
2. Identifies the target column: SalePrice or Price.
3. Selects numeric features and imputes missing values using median.

### Simple Linear Regression
- Selects the most correlated numeric feature.
- Builds a simple regression model.
- Generates scatter plot and regression line.
- Saves: simple_reg_<feature>.png
- Computes MAE, MSE, and R².

### Multiple Linear Regression (Pipeline)
- Uses StandardScaler + LinearRegression.
- Evaluates: MAE, MSE, R².
- Performs 5-fold cross-validation (R²).
- Saves trained model as: linear_regression_pipeline.joblib

### Coefficient Interpretation
- Extracts model coefficients.
- Sorts features by absolute coefficient value.
- Saves results to lr_coefficients.csv.

### Residual Plot
- Generates residuals vs predicted values plot.
- Saves: residuals_plot.png.

### Additional Outputs
- Saves cleaned numeric dataset sample (first 500 rows) as cleaned_numeric_sample.csv.
- All outputs are stored in the lr_outputs directory.

## HOW TO RUN

1. Place your dataset in the project folder or set RAW_PATH inside the script.

2. Run the script:
python linear_regression_house.py

3. Outputs will be created in the lr_outputs folder.

## OUTPUT FILES EXPLAINED

- cleaned_numeric_sample.csv – Cleaned numeric rows (first 500).
- simple_reg_<feature>.png – Simple regression visualization.
- lr_coefficients.csv – Sorted coefficients from multiple regression.
- residuals_plot.png – Residuals vs predicted values.
- linear_regression_pipeline.joblib – Saved regression model.

## NOTES ON INTERPRETATION

- Coefficients are based on standardized features.
- Positive coefficient increases predicted price; negative decreases it.
- Absolute coefficient value indicates strength of influence.
- R² shows percentage variance explained by the model.
- MAE and MSE measure prediction errors.

## EXTENSIONS / NEXT STEPS

- Add categorical encoding via ColumnTransformer.
- Try Ridge or Lasso regression for regularization.
- Add SHAP or permutation importance for feature interpretation.
- Build a complete end-to-end prediction pipeline.

## TROUBLESHOOTING

- If dataset not found: set RAW_PATH manually in the script.
- If sort_values error occurs: older pandas version; fallback is included.
- If ModuleNotFoundError: install missing packages in the correct virtual environment.

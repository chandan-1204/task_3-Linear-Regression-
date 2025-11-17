# linear_regression_house.py
import os
import fnmatch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# -----------------------
# CONFIG - set RAW_PATH if you know it, else leave None to auto-search
# -----------------------
RAW_PATH = "C:\\Users\\chand\\Downloads\\house_prices.csv"
SEARCH_DIRS = ['.', '/mnt/data', os.path.expanduser('~'), os.path.join(os.path.expanduser('~'),'OneDrive','Documents')]
OUT_DIR = './lr_outputs'
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# helper: auto-find likely house dataset
# -----------------------
def find_house_file():
    if RAW_PATH:
        return RAW_PATH if os.path.exists(RAW_PATH) else None
    candidates = []
    for d in SEARCH_DIRS:
        try:
            for root, dirs, files in os.walk(d):
                for name in files:
                    low = name.lower()
                    if ('house' in low or 'sale' in low or 'price' in low) and low.endswith(('.csv','.xlsx','.xls')):
                        candidates.append(os.path.join(root, name))
        except Exception:
            pass
    return candidates[0] if candidates else None

path = find_house_file()
if not path:
    raise FileNotFoundError("No dataset found. Set RAW_PATH to your CSV/XLSX path or place file in project folder.")
print("Loading dataset:", path)

# read csv or excel
if path.lower().endswith(('.xls','.xlsx')):
    df = pd.read_excel(path)
else:
    df = pd.read_csv(path)

print("Initial shape:", df.shape)
print("Columns:", df.columns.tolist())

# -----------------------
# Identify target column (common names)
# -----------------------
target_candidates = [c for c in df.columns if c.lower() in ('saleprice','sale_price','price','target')]
if not target_candidates:
    # try heuristic: numeric column with 'price' in name
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    price_like = [c for c in numeric_cols if 'price' in c.lower() or 'sale' in c.lower()]
    if price_like:
        target = price_like[0]
    else:
        raise ValueError("No target column detected (common names: SalePrice, price, Sale_Price). Please set target manually.")
else:
    target = target_candidates[0]

print("Using target column:", target)

# -----------------------
# Basic cleaning & imputation
# -----------------------
# Keep numeric columns and a few categorical if needed
numeric = df.select_dtypes(include=[np.number]).columns.tolist()
# ensure target in numeric
if target not in numeric:
    # try to coerce
    df[target] = pd.to_numeric(df[target], errors='coerce')
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()

# Work on a copy
data = df[numeric].copy()

# report missing
missing = data.isnull().sum().sort_values(ascending=False)
print("\nMissing (top 10):\n", missing.head(10))

# Impute numeric with median
imp = SimpleImputer(strategy='median')
data_imputed = pd.DataFrame(imp.fit_transform(data), columns=data.columns)

# Save cleaned sample
data_imputed.head(500).to_csv(os.path.join(OUT_DIR, 'cleaned_numeric_sample.csv'), index=False)

# -----------------------
# Quick correlation check to pick best single feature
# -----------------------
corr = data_imputed.corr()[target].abs().sort_values(ascending=False)
print("\nTop correlations with target:\n", corr.head(10))

# Choose best feature for simple linear regression (exclude target itself)
best_feature = corr.index[corr.index != target][0]
print("Best single feature for simple regression:", best_feature)

# -----------------------
# Train-test split
# -----------------------
X = data_imputed.drop(columns=[target])
y = data_imputed[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train/test sizes:", X_train.shape, X_test.shape)

# -----------------------
# Simple Linear Regression (single feature)
# -----------------------
plt.figure(figsize=(8,5))
plt.scatter(X_train[best_feature], y_train, alpha=0.6, label='train points')
plt.scatter(X_test[best_feature], y_test, alpha=0.6, label='test points', marker='x')
plt.xlabel(best_feature)
plt.ylabel(target)
plt.title(f'Simple Linear Regression: {target} ~ {best_feature}')

# fit model on single feature
lr_simple = LinearRegression()
lr_simple.fit(X_train[[best_feature]], y_train)

# line
xs = np.linspace(X_train[best_feature].min(), X_train[best_feature].max(), 100)
ys = lr_simple.predict(xs.reshape(-1,1))
plt.plot(xs, ys, color='red', linewidth=2, label='fitted line')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'simple_reg_{best_feature}.png'))
plt.close()
print(f"Saved simple regression plot: simple_reg_{best_feature}.png")

# Simple metrics
y_pred_simple = lr_simple.predict(X_test[[best_feature]])
print("\nSimple Regression metrics (single feature):")
print("MAE:", mean_absolute_error(y_test, y_pred_simple))
print("MSE:", mean_squared_error(y_test, y_pred_simple))
print("R2:", r2_score(y_test, y_pred_simple))

# -----------------------
# Multiple Linear Regression (all numeric features)
# We'll use a pipeline that scales numeric features
# -----------------------
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nMultiple Regression metrics (all numeric features):")
print("MAE:", mae)
print("MSE:", mse)
print("R2:", r2)

# Cross-validated R2 estimate on full X
cv_r2 = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
print("CV R2 (5-fold):", cv_r2, "mean:", cv_r2.mean())

# -----------------------
# Coefficients interpretation
# -----------------------

model = pipeline.named_steps['model']        # LinearRegression()
scaler = pipeline.named_steps['scaler']      # StandardScaler()

# coefficients correspond to scaled features
coefs = model.coef_
feat_names = X.columns.tolist()

# Build dataframe mapping feature -> coefficient
coef_df = pd.DataFrame({'feature': feat_names, 'coefficient': coefs})

# Preferred: sorted by absolute value
try:
    coef_df = coef_df.sort_values(
        by='coefficient',
        key=lambda s: s.abs(),
        ascending=False
    )
except TypeError:
    # Fallback for older pandas
    coef_df['abs_coef'] = coef_df['coefficient'].abs()
    coef_df = coef_df.sort_values(by='abs_coef', ascending=False).drop(columns=['abs_coef'])

coef_df = coef_df.reset_index(drop=True)

print("\nTop coefficients (by absolute value):")
print(coef_df.head(15))

# Save coefficients
coef_df.to_csv(os.path.join(OUT_DIR, 'lr_coefficients.csv'), index=False)
print("Saved coefficients to lr_coefficients.csv")
# -----------------------
# Residual plot
# -----------------------
residuals = y_test - y_pred
plt.figure(figsize=(6,4))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted (Multiple Regression)')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'residuals_plot.png'))
plt.close()
print("Saved residuals plot")

# -----------------------
# Save model
# -----------------------
MODEL_OUT = os.path.join(OUT_DIR, 'linear_regression_pipeline.joblib')
joblib.dump(pipeline, MODEL_OUT)
print("Saved trained pipeline model to", MODEL_OUT)

# -----------------------
# Show a short summary and saved files
# -----------------------
print("\nDONE. Outputs saved in:", OUT_DIR)
print("Files:")
for f in sorted(os.listdir(OUT_DIR)):
    print(" -", f)

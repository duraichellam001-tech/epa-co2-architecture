# scripts/train_residual_models.py

import os
import sys
import joblib
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# -----------------------------
# Path setup
# -----------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# -----------------------------
# Load Path-B dataset
# -----------------------------

DATA_PATH = "artifacts/epa_co2_architecture_pathB_residuals_v1.parquet"

df = pd.read_parquet(DATA_PATH)

# -----------------------------
# Train / Test split (year-based)
# -----------------------------

TRAIN_END_YEAR = 2018

train_df = df[df["Model Year"] <= TRAIN_END_YEAR].copy()
test_df  = df[df["Model Year"] > TRAIN_END_YEAR].copy()

# -----------------------------
# Feature definitions
# -----------------------------

num_features_city = [
    "Model Year",
    "Test Veh Displacement (L)",
    "Equivalent Test Weight (lbs.)",
    "CO2_physics_city",
]

num_features_hwy = [
    "Model Year",
    "Test Veh Displacement (L)",
    "Equivalent Test Weight (lbs.)",
    "CO2_physics_hwy",
]

cat_features = [
    "transmission_bucket",
    "drive_bucket",
]

# -----------------------------
# Targets
# -----------------------------

target_city = "Residual_city"
target_hwy  = "Residual_hwy"

# -----------------------------
# Preprocessors
# -----------------------------

def make_pipeline(num_features):
    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )

    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        random_state=42
    )

    return Pipeline([
        ("preprocess", preprocess),
        ("model", model),
    ])

# -----------------------------
# Train City residual model
# -----------------------------

pipe_city = make_pipeline(num_features_city)

X_train_city = train_df[num_features_city + cat_features]
y_train_city = train_df[target_city]

pipe_city.fit(X_train_city, y_train_city)

# -----------------------------
# Train Highway residual model
# -----------------------------

pipe_hwy = make_pipeline(num_features_hwy)

X_train_hwy = train_df[num_features_hwy + cat_features]
y_train_hwy = train_df[target_hwy]

pipe_hwy.fit(X_train_hwy, y_train_hwy)

# -----------------------------
# Save models
# -----------------------------

os.makedirs("models/pathB", exist_ok=True)

joblib.dump(
    pipe_city,
    "models/pathB/city_residual_gbr_v1.joblib"
)

joblib.dump(
    pipe_hwy,
    "models/pathB/hwy_residual_gbr_v1.joblib"
)

print("✅ Residual models trained and saved locally")
print("   - City  model saved")
print("   - Hwy   model saved")

# scripts/train_ml_only_model.py

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
# Load Path-A dataset
# -----------------------------

DATA_PATH = "artifacts/epa_co2_architecture_pathA_v1.parquet"

df = pd.read_parquet(DATA_PATH)

# -----------------------------
# Train / Test split (year-based)
# -----------------------------

TRAIN_END_YEAR = 2018

train_df = df[df["Model Year"] <= TRAIN_END_YEAR].copy()
test_df  = df[df["Model Year"] > TRAIN_END_YEAR].copy()

# -----------------------------
# Feature definitions (Path-A)
# -----------------------------

num_features = [
    "Model Year",
    "Test Veh Displacement (L)",
    "Equivalent Test Weight (lbs.)",
]

cat_features = [
    "transmission_bucket",
    "drive_bucket",
]

target_col = "CO2_combined"

# -----------------------------
# Pipeline
# -----------------------------

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

pipe_ml_only = Pipeline([
    ("preprocess", preprocess),
    ("model", model),
])

# -----------------------------
# Train
# -----------------------------

X_train = train_df[num_features + cat_features]
y_train = train_df[target_col]

pipe_ml_only.fit(X_train, y_train)

# -----------------------------
# Save model (LOCAL ENV)
# -----------------------------

os.makedirs("models/pathA", exist_ok=True)

joblib.dump(
    pipe_ml_only,
    "models/pathA/co2_combined_gbr_v1.joblib"
)

print("✅ ML-only GBR model retrained and saved locally")

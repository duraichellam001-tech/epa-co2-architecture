"""
build_dataset_v1.py

Purpose:
--------
Builds the Version 1 ML-ready dataset for early-phase CO₂
architecture modeling using EPA certification data.

Scope (v1):
-----------
- Gasoline ICE vehicles only
- Model years: 2010–2025
- FTP (city) and HWY (highway) cycles only
- Vehicle-level aggregation using EPA 55/45 weighting
- Output: Parquet dataset + metadata

This script is the SINGLE source of truth for dataset v1.
All downstream modeling must depend on its output.
"""

import os
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# -----------------------------
# Required schema (v1)
# -----------------------------
REQUIRED_COLUMNS = [
    "Model Year",
    "Test Veh Displacement (L)",
    "Equivalent Test Weight (lbs.)",
    "Test Fuel Type Description",
    "Tested Transmission Type",
    "Drive System Code",
    "Test Category",
    "CO2 (g/mi)",
]

# -----------------------------
# Vehicle identity (Path-A)
# -----------------------------
VEHICLE_ID_COLS = [
    "Model Year",
    "Test Veh Displacement (L)",
    "Equivalent Test Weight (lbs.)",
    "Tested Transmission Type",
    "Drive System Code",
]

def list_raw_data_files():
    """
    List all raw EPA data files (CSV/XLSX) available for processing.
    """
    if not os.path.exists(RAW_DATA_DIR):
        raise FileNotFoundError(f"Raw data directory not found: {RAW_DATA_DIR}")

    files = [
        os.path.join(RAW_DATA_DIR, f)
        for f in os.listdir(RAW_DATA_DIR)
        if f.lower().endswith((".csv", ".xlsx"))
    ]

    if len(files) == 0:
        raise RuntimeError(
            f"No raw EPA files found in {RAW_DATA_DIR}. "
            "Expected CSV or XLSX files."
        )

    return sorted(files)

def normalize_column_names(df):
    """
    Normalize column names for consistency across EPA years.
    """
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    return df

def validate_required_columns(df, file_path):
    """
    Ensure all required columns exist in the raw EPA file.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        raise RuntimeError(
            f"Missing required columns in file {os.path.basename(file_path)}: {missing}"
        )

def bucket_transmission(x):
    """
    Map EPA transmission descriptions to architecture buckets.
    """
    x = str(x).lower()

    if "manual" in x and "automated" not in x:
        return "MT"
    if "cvt" in x or "variable" in x:
        return "CVT"
    return "AT"


def bucket_drive_system(x):
    """
    Map EPA drive system codes to architecture buckets.
    """
    x = str(x).upper()

    if x in ["F", "FRONT"]:
        return "FWD"
    if x in ["R", "REAR"]:
        return "RWD"
    return "AWD"


def main():
    """
    Main pipeline entry point.
    """
    print("Dataset build pipeline v1 — START")

    raw_files = list_raw_data_files()
    print(f"Found {len(raw_files)} raw EPA files.")

    all_dfs = []

    for file_path in raw_files:
        print(f"Loading file: {os.path.basename(file_path)}")

        if file_path.lower().endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        df = normalize_column_names(df)
        validate_required_columns(df, file_path)

        all_dfs.append(df)

    df_all = pd.concat(all_dfs, ignore_index=True)
    print(f"Combined dataset shape: {df_all.shape}")
    
    # -----------------------------
    # Select required columns only
    # -----------------------------
    df_model = df_all[REQUIRED_COLUMNS].copy()
    print(f"After selecting required columns: {df_model.shape}")
    
    # -----------------------------
    # Fuel scope (Gasoline ICE)
    # -----------------------------
    GASOLINE_FUEL_KEYWORDS = [
       "gasoline",
       "e10",
       "indolene",
       "cold co",
    ]

    # -----------------------------
    # Filter gasoline ICE vehicles
    # -----------------------------
    fuel_col = df_model["Test Fuel Type Description"].str.lower()

    is_gasoline = (
        fuel_col.str.contains("gasoline", na=False) |
        fuel_col.str.contains("cold co", na=False) |
        fuel_col.str.contains("indolene", na=False)
    )

    is_excluded = (
        fuel_col.str.contains("electric", na=False) |
        fuel_col.str.contains("diesel", na=False) |
        fuel_col.str.contains("e85", na=False) |
        fuel_col.str.contains("ethanol", na=False) |
        fuel_col.str.contains("cng", na=False) |
        fuel_col.str.contains("lpg", na=False) |
        fuel_col.str.contains("hydrogen", na=False)
    )

    df_model = df_model[is_gasoline & ~is_excluded].copy()
    print(f"After gasoline ICE filtering: {df_model.shape}")

    ALLOWED_TEST_CATEGORIES = ["FTP", "HWY"]

    # -----------------------------
    # Filter EPA test cycles
    # -----------------------------
    df_model = df_model[df_model["Test Category"].isin(ALLOWED_TEST_CATEGORIES)].copy()
    print(f"After test cycle filtering (FTP/HWY): {df_model.shape}")

    # -----------------------------
    # Pair FTP & HWY per vehicle
    # -----------------------------
    df_pivot = (
        df_model
        .pivot_table(
            index=VEHICLE_ID_COLS,
            columns="Test Category",
            values="CO2 (g/mi)",
            aggfunc="mean"
        )
        .reset_index()
    )

    # Keep only vehicles with both FTP and HWY
    df_pivot = df_pivot.dropna(subset=["FTP", "HWY"]).copy()
    print(f"After FTP/HWY pairing: {df_pivot.shape}")

    # -----------------------------
    # Rename FTP / HWY to explicit targets
    # -----------------------------
    df_pivot = df_pivot.rename(
        columns={
            "FTP": "CO2_city_gpm",
            "HWY": "CO2_hwy_gpm",
        }
    )


    # -----------------------------
    # EPA combined CO2 (55/45)
    # -----------------------------
    df_pivot["CO2_combined"] = (
        0.55 * df_pivot["CO2_city_gpm"] +
        0.45 * df_pivot["CO2_hwy_gpm"]
    )

    # -----------------------------
    # Architecture abstraction
    # -----------------------------
    df_pivot["transmission_bucket"] = df_pivot["Tested Transmission Type"].apply(
        bucket_transmission
    )

    df_pivot["drive_bucket"] = df_pivot["Drive System Code"].apply(
        bucket_drive_system
    )

    print(
        "Transmission buckets:",
        df_pivot["transmission_bucket"].value_counts().to_dict()
    )
    print(
        "Drive buckets:",
        df_pivot["drive_bucket"].value_counts().to_dict()
    )

    # -----------------------------
    # Final modeling table (Path-A)
    # -----------------------------
    MODEL_FEATURES = [
        "Model Year",
        "Test Veh Displacement (L)",
        "Equivalent Test Weight (lbs.)",
        "transmission_bucket",
        "drive_bucket",
    ]

    TARGET_COL = [
         "CO2_combined", 
         "CO2_city_gpm",
          "CO2_hwy_gpm",
    ]

    df_final = df_pivot[MODEL_FEATURES + TARGET_COL].copy()
    print(f"Final modeling table shape: {df_final.shape}")

    print("Final columns:")
    print(df_final.columns.tolist())

    # -----------------------------
    # Save dataset artifact (Path-A)
    # -----------------------------
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    dataset_path = os.path.join(
        ARTIFACTS_DIR,
        "epa_co2_architecture_pathB_v1.parquet"
    )

    df_final.to_parquet(dataset_path, index=False)
    print(f"Saved dataset to: {dataset_path}")

    # -----------------------------
    # Save metadata
    # -----------------------------
    metadata = {
        "dataset_name": "epa_co2_architecture_pathB",
        "version": "v1.0",
        "rows": len(df_final),
        "features": MODEL_FEATURES,
        "target": TARGET_COL,
        "fuel_scope": "Gasoline ICE only",
        "test_cycles": ["FTP", "HWY"],
        "aggregation": "EPA 55/45 FTP/HWY",
        "years": f"{df_final['Model Year'].min()}–{df_final['Model Year'].max()}",
        "notes": [
            "Vehicle-level aggregation",
            "Architecture-based features only",
            "Brand/model names excluded",
            "Ready for Path-B residual modeling"
        ]
    }

    metadata_path = os.path.join(
        ARTIFACTS_DIR,
        "epa_co2_architecture_pathB_v1_metadata.json"
    )

    import json
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to: {metadata_path}")

if __name__ == "__main__":
    main()




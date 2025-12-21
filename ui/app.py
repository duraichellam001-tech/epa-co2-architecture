	
import sys
import os

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)


import streamlit as st
import joblib
import numpy as np

from physics.physics_co2 import compute_city_hwy_co2_g_per_mi

# -----------------------------
# Load ML models (cached)
# -----------------------------

@st.cache_resource
def load_models():
    city_model = joblib.load(
        "models/pathB/city_residual_gbr_v1.joblib"
    )
    hwy_model = joblib.load(
        "models/pathB/hwy_residual_gbr_v1.joblib"
    )
    ml_only_model = joblib.load(
        "models/pathA/co2_combined_gbr_v1.joblib"
    )
    return city_model, hwy_model, ml_only_model


city_model, hwy_model, ml_only_model = load_models()


# -----------------------------
# UI Header
# -----------------------------

st.title("Physics-Guided CO₂ Prediction Tool for Architecture Screening")
st.caption("Early-phase vehicle CO₂ estimation using Physics + ML")

# -----------------------------
# Sidebar — Inputs
# -----------------------------

st.sidebar.header("Vehicle Architecture Inputs")

analysis_mode = st.radio(
    "Select Mode",
    ["Single Architecture Prediction", "Transmission Sensitivity Analysis"]
)


st.sidebar.header("Prediction Mode")

prediction_mode = st.sidebar.radio(
    "Select Model",
    ["Physics + ML", "ML Only"]
)


mass_lb = st.sidebar.number_input(
    "Vehicle Mass (lb)",
    min_value=2100,
    max_value=8500,
    value=4000,
    step=50
)

engine_disp = st.sidebar.number_input(
    "Engine Displacement (L)",
    min_value=0.5,
    max_value=8.5,
    value=2.0,
    step=0.1
)

trans_type = st.sidebar.selectbox(
    "Transmission",
    ["MT", "AT", "CVT"]
)

drive_layout = st.sidebar.selectbox(
    "Drive Layout",
    ["FWD", "RWD", "AWD"]
)

model_year = st.sidebar.slider(
    "Target Model Year",
    min_value=2015,
    max_value=2026,
    value=2025,
    step=1
)

operating_mode = st.sidebar.radio(
    "Operating Mode",
    ["City", "Highway", "Combined"]
)


# -----------------------------
# Physics Baseline
# -----------------------------

physics = compute_city_hwy_co2_g_per_mi(
    mass_lb=mass_lb,
    trans_type=trans_type,
    drive_layout=drive_layout
)

physics_city = physics["city"]
physics_hwy = physics["hwy"]

# -----------------------------
# ML Residual Correction
# -----------------------------

import pandas as pd

X_city = pd.DataFrame([{
    "Model Year": 2025,
    "Test Veh Displacement (L)": engine_disp,
    "Equivalent Test Weight (lbs.)": mass_lb,
    "CO2_physics_city": physics_city,
    "transmission_bucket": trans_type,
    "drive_bucket": drive_layout,
}])


X_hwy = pd.DataFrame([{
    "Model Year": 2025,
    "Test Veh Displacement (L)": engine_disp,
    "Equivalent Test Weight (lbs.)": mass_lb,
    "CO2_physics_hwy": physics_hwy,
    "transmission_bucket": trans_type,
    "drive_bucket": drive_layout,
}])


residual_city = city_model.predict(X_city)[0]
residual_hwy = hwy_model.predict(X_hwy)[0]

final_city = physics_city + residual_city
final_hwy = physics_hwy + residual_hwy
final_combined = 0.55 * final_city + 0.45 * final_hwy

X_ml_only = pd.DataFrame([{
    "Model Year": 2025,
    "Test Veh Displacement (L)": engine_disp,
    "Equivalent Test Weight (lbs.)": mass_lb,
    "transmission_bucket": trans_type,
    "drive_bucket": drive_layout,
}])

ml_only_co2 = ml_only_model.predict(X_ml_only)[0]

# -----------------------------
# Output Display
# -----------------------------

st.subheader("CO₂ Prediction Results (g/mi)")

if prediction_mode == "Physics + ML":
    if operating_mode == "City":
        st.metric("Physics CO₂ (City)", f"{physics_city:.1f}")
        st.metric("ML Correction", f"{residual_city:+.1f}")
        st.metric("Final CO₂ (City)", f"{final_city:.1f}")

    elif operating_mode == "Highway":
        st.metric("Physics CO₂ (Highway)", f"{physics_hwy:.1f}")
        st.metric("ML Correction", f"{residual_hwy:+.1f}")
        st.metric("Final CO₂ (Highway)", f"{final_hwy:.1f}")

    else:
        st.metric("Final CO₂ (Combined)", f"{final_combined:.1f}")

else:
    st.metric("ML-only CO₂ (Combined)", f"{ml_only_co2:.1f}")

st.divider()
st.subheader("Mass Extrapolation Study")

run_sweep = st.button("Run Mass Sweep (3000–7000 lb)")

if run_sweep:
    masses = np.arange(3000, 7001, 250)

    ml_only_vals = []
    phys_ml_vals = []

    for m in masses:
        # Physics
        phys = compute_city_hwy_co2_g_per_mi(
            mass_lb=m,
            trans_type=trans_type,
            drive_layout=drive_layout
        )

        phys_city = phys["city"]
        phys_hwy  = phys["hwy"]

        # Physics + ML
        X_city = pd.DataFrame([{
            "Model Year": 2025,
            "Test Veh Displacement (L)": engine_disp,
            "Equivalent Test Weight (lbs.)": m,
            "CO2_physics_city": phys_city,
            "transmission_bucket": trans_type,
            "drive_bucket": drive_layout,
        }])

        X_hwy = pd.DataFrame([{
            "Model Year": 2025,
            "Test Veh Displacement (L)": engine_disp,
            "Equivalent Test Weight (lbs.)": m,
            "CO2_physics_hwy": phys_hwy,
            "transmission_bucket": trans_type,
            "drive_bucket": drive_layout,
        }])

        res_city = city_model.predict(X_city)[0]
        res_hwy  = hwy_model.predict(X_hwy)[0]

        final_city = phys_city + res_city
        final_hwy  = phys_hwy + res_hwy

        final_combined = 0.55 * final_city + 0.45 * final_hwy
        phys_ml_vals.append(final_combined)

        # ML-only
        X_ml = pd.DataFrame([{
            "Model Year": 2025,
            "Test Veh Displacement (L)": engine_disp,
            "Equivalent Test Weight (lbs.)": m,
            "transmission_bucket": trans_type,
            "drive_bucket": drive_layout,
        }])

        ml_only_vals.append(
            ml_only_model.predict(X_ml)[0]
        )

    # -----------------------------
    # Plot
    # -----------------------------
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(masses, ml_only_vals, "o--", label="ML-only")
    ax.plot(masses, phys_ml_vals, "o-", label="Physics + ML")

    ax.set_xlabel("Vehicle Mass (lb)")
    ax.set_ylabel("Combined CO₂ (g/mi)")
    ax.set_title("Mass Extrapolation: ML-only vs Physics + ML")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

if analysis_mode == "Transmission Sensitivity Analysis":

    st.subheader("Transmission Sensitivity Analysis")
    st.caption("Mass, engine, and drive fixed. Transmission swept.")

    transmissions = ["MT", "CVT", "AT"]
    rows = []

    for trans in transmissions:

        # ---------- Physics ----------
        phys = compute_city_hwy_co2_g_per_mi(
            mass_lb=mass_lb,
            trans_type=trans,
            drive_layout=drive_layout
        )

        phys_city = phys["city"]
        phys_hwy  = phys["hwy"]

        # ---------- Physics + ML ----------
        X_city = pd.DataFrame([{
            "Model Year": 2025,
            "Test Veh Displacement (L)": engine_disp,
            "Equivalent Test Weight (lbs.)": mass_lb,
            "CO2_physics_city": phys_city,
            "transmission_bucket": trans,
            "drive_bucket": drive_layout,
        }])

        X_hwy = pd.DataFrame([{
            "Model Year": 2025,
            "Test Veh Displacement (L)": engine_disp,
            "Equivalent Test Weight (lbs.)": mass_lb,
            "CO2_physics_hwy": phys_hwy,
            "transmission_bucket": trans,
            "drive_bucket": drive_layout,
        }])

        res_city = city_model.predict(X_city)[0]
        res_hwy  = hwy_model.predict(X_hwy)[0]

        final_city = phys_city + res_city
        final_hwy  = phys_hwy + res_hwy
        final_comb = 0.55 * final_city + 0.45 * final_hwy

        # ---------- ML-only ----------
        X_ml = pd.DataFrame([{
            "Model Year": 2025,
            "Test Veh Displacement (L)": engine_disp,
            "Equivalent Test Weight (lbs.)": mass_lb,
            "transmission_bucket": trans,
            "drive_bucket": drive_layout,
        }])

        ml_only = ml_only_model.predict(X_ml)[0]

        rows.append({
            "Transmission": trans,
            "ML-only CO₂ (g/mi)": ml_only,
            "Physics + ML CO₂ (g/mi)": final_comb
        })

    df_trans = pd.DataFrame(rows)

    st.dataframe(df_trans, use_container_width=True)

    # ---------- Plot ----------
    st.line_chart(
        df_trans.set_index("Transmission")
    )


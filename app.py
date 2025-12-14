import joblib
import pandas as pd
import gradio as gr

# Load frozen models
tree_model = joblib.load("models/co2_architecture_tree_model.pkl")
linear_model = joblib.load("models/co2_architecture_linear_model.pkl")

def predict_co2(
    model_year,
    displacement,
    weight,
    transmission,
    drive,
    model_choice
):
    df = pd.DataFrame([{
        "Model Year": model_year,
        "Test Veh Displacement (L)": displacement,
        "Equivalent Test Weight (lbs.)": weight,
        "transmission_bucket": transmission,
        "drive_bucket": drive
    }])

    if model_choice == "Tree-based (GBDT)":
        pred = tree_model.predict(df)[0]
    else:
        pred = linear_model.predict(df)[0]

    return round(pred, 1)

interface = gr.Interface(
    fn=predict_co2,
    inputs=[
        gr.Number(label="Model Year", value=2023),
        gr.Number(label="Engine Displacement (L)", value=2.0),
        gr.Number(label="Equivalent Test Weight (lbs)", value=3500),
        gr.Dropdown(["MT", "AT", "CVT"], label="Transmission"),
        gr.Dropdown(["FWD", "RWD", "AWD"], label="Drive System"),
        gr.Radio(
            ["Tree-based (GBDT)", "Linear"],
            label="Model",
            value="Tree-based (GBDT)"
        ),
    ],
    outputs=gr.Number(label="Predicted CO₂ (g/mi)"),
    title="EPA CO₂ Architecture Screening Tool (Phase-1)",
    description=(
        "Early-phase vehicle CO₂ estimation using EPA certification data. "
        "Inputs represent architecture-level design choices."
    )
)

if __name__ == "__main__":
    interface.queue().launch(server_name="0.0.0.0", server_port=7860)
#   interface.launch() ( For local hosting )


import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="AEPL AI Costing Engine", layout="wide", page_icon="⚙️")

# 2. Header and Introduction
st.title("🏭 AEPL Automated Time & Cost Estimation")
st.markdown("""
This application demonstrates a fully functional Machine Learning architecture for predicting fabrication operation times and calculating standardized manufacturing costs. 
**Built by: [Your Name] | Supervised by: [Professor's Name]**
""")
st.divider()

# 3. Load Model and Data
@st.cache_resource
def load_ml_model():
    model_path = os.path.join(os.getcwd(), "models", "best_time_model.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_ml_model()

if model is None:
    st.error("Model not found! Please run the training scripts in Phase 11 first.")
    st.stop()

# 4. Sidebar - Financial Configuration
st.sidebar.header("💰 Financial Rates (₹/hr)")
labor_rate = st.sidebar.number_input("Labor Rate", value=250.0, step=10.0)
machine_rate = st.sidebar.number_input("Machine Rate", value=800.0, step=50.0)
consumable_rate = st.sidebar.number_input("Consumables Rate", value=150.0, step=10.0)

st.sidebar.divider()
st.sidebar.markdown("### Model Diagnostics")
st.sidebar.success("Gradient Boosting Regressor")
st.sidebar.info("Data Leakage Prevented: Yes")

# 5. Main Layout - Two Columns
col1, col2 = st.columns([1, 1])

with col1:
    st.header("⚙️ Planned Process Parameters")
    st.markdown("Enter the expected physical parameters for the new component:")
    
    # Input fields for the features
    col_a, col_b = st.columns(2)
    with col_a:
        spindle_max = st.number_input("Spindle Velocity (Max)", value=7200000.0, step=100000.0)
        x_feed = st.number_input("X-Axis Feed (Mean)", value=150.0)
        z_feed = st.number_input("Z-Axis Feed (Mean)", value=50.0)
    with col_b:
        spindle_mean = st.number_input("Spindle Velocity (Mean)", value=3500000.0, step=100000.0)
        y_feed = st.number_input("Y-Axis Feed (Mean)", value=200.0)
        
    # Automatically calculate the engineered feature
    vector_3d = np.sqrt(x_feed**2 + y_feed**2 + z_feed**2)
    st.info(f"Engineered 3D Feed Vector: **{vector_3d:.2f}**")
    
    predict_button = st.button("Calculate Time & Cost", type="primary", use_container_width=True)

with col2:
    st.header("📊 AI Estimation Results")
    
    if predict_button:
        # Create input dataframe
        input_data = pd.DataFrame([{
            'Spindle_Velocity_Mean': spindle_mean,
            'Spindle_Velocity_Max': spindle_max,
            'X_Feed_Mean': x_feed,
            'Y_Feed_Mean': y_feed,
            'Z_Feed_Mean': z_feed,
            '3D_Feed_Vector': vector_3d
        }])
        
        # Predict Time
        pred_seconds = model.predict(input_data)[0]
        # Ensure prediction is positive
        pred_seconds = max(1.0, pred_seconds)
        pred_hours = pred_seconds / 3600.0
        
        # Display Time Metrics
        met1, met2 = st.columns(2)
        met1.metric("Predicted Time (Seconds)", f"{pred_seconds:.0f} s")
        met2.metric("Predicted Time (Hours)", f"{pred_hours:.3f} hr")
        
        # Calculate Costs
        labor_cost = pred_hours * labor_rate
        machine_cost = pred_hours * machine_rate
        consumable_cost = pred_hours * consumable_rate
        total_cost = labor_cost + machine_cost + consumable_cost
        
        # Display Cost Metrics
        st.subheader("Financial Breakdown")
        st.write(f"**Labor Cost:** ₹{labor_cost:.2f}")
        st.write(f"**Machine Cost:** ₹{machine_cost:.2f}")
        st.write(f"**Consumables Cost:** ₹{consumable_cost:.2f}")
        st.divider()
        st.metric("TOTAL ESTIMATED COST", f"₹{total_cost:.2f}")

st.divider()

# 6. AI Explainability Section
st.header("🧠 AI Explainability (SHAP & Diagnostics)")
st.markdown("Visual proof that the model is interpreting the physics correctly, not just guessing.")

img_col1, img_col2 = st.columns(2)

with img_col1:
    shap_path = os.path.join(os.getcwd(), "visualizations", "shap_summary.png")
    if os.path.exists(shap_path):
        st.image(Image.open(shap_path), caption="SHAP Summary (Feature Impact)")
    else:
        st.warning("SHAP visualization not found. Run Phase 12 first.")

with img_col2:
    actual_path = os.path.join(os.getcwd(), "visualizations", "actual_vs_predicted.png")
    if os.path.exists(actual_path):
        st.image(Image.open(actual_path), caption="Actual vs Predicted Error Plot")
    else:
        st.warning("Error visualization not found. Run Phase 12 first.")
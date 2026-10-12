import pandas as pd
import numpy as np
import os
import joblib

def load_rates():
    # In a real factory, these would be loaded from an SQL database or HR spreadsheet.
    # We define standard hypothetical industrial rates here.
    return {
        "labor_rate_per_hour": 250.0,      # e.g., ₹250 per hour for operator
        "machine_rate_per_hour": 800.0,    # e.g., ₹800 per hour for CNC overhead/power
        "consumable_rate_per_hour": 150.0  # e.g., ₹150 per hour for coolant/tool wear
    }

def predict_new_operation():
    print("--- AEPL AUTOMATED COSTING ENGINE ---\n")
    
    # 1. Load the trained AI Model
    model_path = os.path.join("models", "best_time_model.pkl")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Did you run Phase 11?")
        return
        
    model = joblib.load(model_path)
    
    # 2. Define a brand new, unseen component (Synthetic Input for Testing)
    # Imagine an engineer has just programmed a new part and inputs these planned parameters:
    new_component_features = pd.DataFrame([{
        'Spindle_Velocity_Mean': 3500000.0,
        'Spindle_Velocity_Max': 7200000.0,
        'X_Feed_Mean': 150.0,
        'Y_Feed_Mean': 200.0,
        'Z_Feed_Mean': 50.0,
        '3D_Feed_Vector': np.sqrt(150.0**2 + 200.0**2 + 50.0**2) 
    }])
    
    print("1. Receiving Planned Component Parameters:")
    print(new_component_features.iloc[0].to_string())
    print("-" * 40)
    
    # 3. AI Time Prediction
    predicted_seconds = model.predict(new_component_features)[0]
    predicted_hours = predicted_seconds / 3600.0
    
    print(f"2. AI Prediction:")
    print(f"Predicted Operation Time: {predicted_seconds:.0f} seconds ({predicted_hours:.3f} hours)")
    print("-" * 40)
    
    # 4. Cost Calculation
    rates = load_rates()
    labor_cost = predicted_hours * rates["labor_rate_per_hour"]
    machine_cost = predicted_hours * rates["machine_rate_per_hour"]
    consumable_cost = predicted_hours * rates["consumable_rate_per_hour"]
    total_cost = labor_cost + machine_cost + consumable_cost
    
    print("3. Financial Cost Breakdown:")
    print(f"Labor Cost (@ ₹{rates['labor_rate_per_hour']}/hr):     ₹{labor_cost:.2f}")
    print(f"Machine Cost (@ ₹{rates['machine_rate_per_hour']}/hr):   ₹{machine_cost:.2f}")
    print(f"Consumable Cost (@ ₹{rates['consumable_rate_per_hour']}/hr):₹{consumable_cost:.2f}")
    print(f">>> TOTAL ESTIMATED COST:    ₹{total_cost:.2f}")
    print("----------------------------------------")

if __name__ == "__main__":
    predict_new_operation()
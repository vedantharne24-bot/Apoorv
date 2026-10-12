import pandas as pd
import os

def engineer_features():
    print("Starting Feature Engineering and Leakage Removal...")
    
    input_file = os.path.join("data", "processed", "operation_features.csv")
    output_file = os.path.join("data", "processed", "ml_ready_dataset.csv")
    
    # Load the aggregated operation data
    df = pd.read_csv(input_file)
    initial_rows = len(df)
    
    # 1. REMOVE DATA LEAKAGE
    print("Removing data leakage columns (e.g., Door_Interventions)...")
    if 'Door_Interventions' in df.columns:
        df = df.drop(columns=['Door_Interventions'])
        
    # 2. FILTER INVALID OPERATIONS
    # Operations where the spindle never spun are not valid cutting operations
    print("Filtering out non-cutting operations (Spindle_Velocity_Max == 0)...")
    df = df[df['Spindle_Velocity_Max'] > 0]
    
    # 3. FEATURE ENGINEERING
    # We will create a new feature representing the overall 3D Feed rate
    # Mathematically: sqrt(X^2 + Y^2 + Z^2)
    print("Engineering 3D Vector Feed feature...")
    df['3D_Feed_Vector'] = (df['X_Feed_Mean']**2 + df['Y_Feed_Mean']**2 + df['Z_Feed_Mean']**2)**0.5
    
    final_rows = len(df)
    print(f"Removed {initial_rows - final_rows} invalid operations.")
    
    # Save the final ML-ready dataset
    df.to_csv(output_file, index=False)
    
    print("\n--- FEATURE ENGINEERING COMPLETE ---")
    print(f"Total ML-Ready Operations: {final_rows}")
    print(f"Target Column: Operation_Time_Seconds")
    print(f"Feature Columns: {list(df.columns.drop('Operation_Time_Seconds'))}")
    print(f"Dataset saved to: {output_file}")

if __name__ == "__main__":
    engineer_features()
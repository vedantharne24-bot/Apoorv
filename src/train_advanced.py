import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_advanced_models():
    print("Starting Advanced Model Training...\n")
    
    # Load the split data
    X_train = pd.read_csv(os.path.join("data", "processed", "X_train.csv"))
    X_test = pd.read_csv(os.path.join("data", "processed", "X_test.csv"))
    y_train = pd.read_csv(os.path.join("data", "processed", "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join("data", "processed", "y_test.csv")).values.ravel()
    
    # Initialize the advanced models with random_state for reproducibility
    models = {
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42)
    }
    
    results = []
    
    # Train and evaluate each model
    for name, model in models.items():
        # Train the model (it learns the patterns here)
        model.fit(X_train, y_train)
        
        # Test the model on unseen data
        preds = model.predict(X_test)
        
        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        results.append({
            "Model": name,
            "MAE (Seconds)": mae,
            "RMSE": rmse,
            "R² Score": r2
        })
        
    # Print the comparison table
    print(f"{'Model':<25} | {'MAE (Seconds)':<15} | {'RMSE':<10} | {'R² Score'}")
    print("-" * 70)
    for res in results:
        print(f"{res['Model']:<25} | {res['MAE (Seconds)']:<15.2f} | {res['RMSE']:<10.2f} | {res['R² Score']:.4f}")
        
    print("\n--- ADVANCED TRAINING COMPLETE ---")

if __name__ == "__main__":
    train_advanced_models()
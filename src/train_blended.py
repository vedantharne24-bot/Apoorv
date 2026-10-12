import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_blended_model():
    print("Starting Blended Model Training for Micro-Precision...\n")
    
    # Load data
    X_train = pd.read_csv(os.path.join("data", "processed", "X_train.csv"))
    X_test = pd.read_csv(os.path.join("data", "processed", "X_test.csv"))
    y_train = pd.read_csv(os.path.join("data", "processed", "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join("data", "processed", "y_test.csv")).values.ravel()
    
    # 1. The Continuous Model (Ridge Regression)
    # Ridge needs scaled data to work properly. This model guarantees that EVERY tiny 
    # change in input creates a proportional tiny change in output time.
    ridge_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0, random_state=42))
    ])
    
    # 2. The Step-Function Model (Gradient Boosting)
    # This keeps our model grounded in the complex physics.
    gb = GradientBoostingRegressor(
        n_estimators=100, 
        max_depth=3, 
        learning_rate=0.05, 
        random_state=42
    )
    
    # 3. The Blend (Voting Regressor)
    # This averages the predictions of both models.
    blended_model = VotingRegressor(
        estimators=[('gb', gb), ('ridge', ridge_pipeline)]
    )
    
    # Train the blended model
    print("Training blended model (Gradient Boosting + Ridge)...")
    blended_model.fit(X_train, y_train)
    
    # Evaluate
    preds = blended_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    print(f"\nBlended Model Performance:")
    print(f"MAE (Seconds): {mae:.2f}")
    print(f"RMSE:          {rmse:.2f}")
    print(f"R² Score:      {r2:.4f}")
    
    # Overwrite our previous best model with this new, highly sensitive one
    model_path = os.path.join("models", "best_time_model.pkl")
    joblib.dump(blended_model, model_path)
    print(f"\nMicro-sensitive Blended Model saved to: {model_path}")
    print("Your Streamlit app will now automatically use this continuous model.")

if __name__ == "__main__":
    train_blended_model()
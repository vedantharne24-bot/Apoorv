import pandas as pd
import numpy as np
import os
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_baselines():
    print("Starting Baseline Model Training...\n")
    
    # Define file paths
    x_train_path = os.path.join("data", "processed", "X_train.csv")
    x_test_path = os.path.join("data", "processed", "X_test.csv")
    y_train_path = os.path.join("data", "processed", "y_train.csv")
    y_test_path = os.path.join("data", "processed", "y_test.csv")
    
    # Load the split data
    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)
    # .values.ravel() converts a 1D dataframe into a flat array, which scikit-learn prefers for targets
    y_train = pd.read_csv(y_train_path).values.ravel()
    y_test = pd.read_csv(y_test_path).values.ravel()
    
    # 1. DUMMY REGRESSOR (The "Mean" Guess)
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    dummy_preds = dummy.predict(X_test)
    
    dummy_mae = mean_absolute_error(y_test, dummy_preds)
    dummy_rmse = np.sqrt(mean_squared_error(y_test, dummy_preds))
    dummy_r2 = r2_score(y_test, dummy_preds)
    
    # 2. LINEAR REGRESSION (Simple Straight Line)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    
    lr_mae = mean_absolute_error(y_test, lr_preds)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
    lr_r2 = r2_score(y_test, lr_preds)
    
    # 3. DISPLAY RESULTS
    print(f"{'Model':<25} | {'MAE (Seconds)':<15} | {'RMSE':<10} | {'R² Score'}")
    print("-" * 70)
    print(f"{'Dummy (Mean Baseline)':<25} | {dummy_mae:<15.2f} | {dummy_rmse:<10.2f} | {dummy_r2:.4f}")
    print(f"{'Linear Regression':<25} | {lr_mae:<15.2f} | {lr_rmse:<10.2f} | {lr_r2:.4f}")
    print("\n--- BASELINES COMPLETE ---")

if __name__ == "__main__":
    train_baselines()
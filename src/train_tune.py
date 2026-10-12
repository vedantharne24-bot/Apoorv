import pandas as pd
import numpy as np
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def tune_model():
    print("Starting Hyperparameter Tuning for Gradient Boosting...")
    print("Goal: Apply regularization to prevent overfitting on our small dataset.\n")
    
    # Load the split data
    X_train = pd.read_csv(os.path.join("data", "processed", "X_train.csv"))
    X_test = pd.read_csv(os.path.join("data", "processed", "X_test.csv"))
    y_train = pd.read_csv(os.path.join("data", "processed", "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join("data", "processed", "y_test.csv")).values.ravel()
    
    # Initialize base model
    gb = GradientBoostingRegressor(random_state=42)
    
    # Define the Hyperparameter Grid
    # Because our dataset is tiny, we keep the trees very shallow (max_depth 2 or 3)
    param_grid = {
        'n_estimators': [30, 50, 100],          # Number of trees
        'max_depth': [2, 3],                    # Maximum depth of each tree (shallow = less overfitting)
        'learning_rate': [0.01, 0.05, 0.1],     # How aggressively the model learns
        'min_samples_split': [2, 4],            # Minimum samples required to split an internal node
        'min_samples_leaf': [2, 4]              # Minimum samples required to be at a leaf node
    }
    
    # Initialize GridSearchCV
    # cv=3 means 3-fold cross-validation (appropriate for small datasets)
    print("Testing 72 different model configurations using 3-Fold Cross Validation...")
    grid_search = GridSearchCV(
        estimator=gb,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=-1, # Uses all CPU cores to speed up training
        verbose=1
    )
    
    # Run the grid search
    grid_search.fit(X_train, y_train)
    
    # Extract the best model
    best_model = grid_search.best_estimator_
    
    # Test the tuned model
    tuned_preds = best_model.predict(X_test)
    tuned_mae = mean_absolute_error(y_test, tuned_preds)
    tuned_rmse = np.sqrt(mean_squared_error(y_test, tuned_preds))
    tuned_r2 = r2_score(y_test, tuned_preds)
    
    # Print Results
    print("\n--- TUNING COMPLETE ---")
    print("Best Hyperparameters Found:")
    for param, value in grid_search.best_params_.items():
        print(f" - {param}: {value}")
        
    print(f"\nTuned Model Performance on Unseen Test Data:")
    print(f"MAE (Seconds): {tuned_mae:.2f}")
    print(f"RMSE:          {tuned_rmse:.2f}")
    print(f"R² Score:      {tuned_r2:.4f}")
    
    # Save the best model for future use
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "best_time_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"\nModel successfully saved to: {model_path}")

if __name__ == "__main__":
    tune_model()
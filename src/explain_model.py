import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import shap

def generate_explanations():
    print("Starting Error Analysis and SHAP Explainability...")
    
    # Define paths
    x_test_path = os.path.join("data", "processed", "X_test.csv")
    y_test_path = os.path.join("data", "processed", "y_test.csv")
    model_path = os.path.join("models", "best_time_model.pkl")
    vis_dir = "visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    # Load data and model
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()
    model = joblib.load(model_path)
    
    # Generate predictions
    preds = model.predict(X_test)
    
    # ---------------------------------------------------------
    # 1. ACTUAL VS PREDICTED PLOT (Error Analysis)
    # ---------------------------------------------------------
    print("Generating Actual vs Predicted plot...")
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, preds, alpha=0.7, color='blue', edgecolor='black')
    
    # Plot the perfect prediction diagonal line
    max_val = max(max(y_test), max(preds))
    min_val = min(min(y_test), min(preds))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.title("Actual vs Predicted Operation Time")
    plt.xlabel("Actual Time (Seconds)")
    plt.ylabel("Predicted Time (Seconds)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "actual_vs_predicted.png"))
    plt.close()
    
    # ---------------------------------------------------------
    # 2. STANDARD FEATURE IMPORTANCE
    # ---------------------------------------------------------
    print("Generating standard Feature Importance chart...")
    importance = model.feature_importances_
    features = X_test.columns
    
    # Sort features by importance
    indices = np.argsort(importance)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importance[indices], color='steelblue', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.title("Model Feature Importance (Gradient Boosting)")
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "feature_importance.png"))
    plt.close()
    
    # ---------------------------------------------------------
    # 3. SHAP EXPLAINABILITY
    # ---------------------------------------------------------
    print("Calculating SHAP values (this may take a moment)...")
    # TreeExplainer is specifically mathematically optimized for tree-based models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    print("Generating SHAP Summary plot...")
    plt.figure(figsize=(10, 6))
    # show=False allows us to save the plot to a file instead of halting the script
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "shap_summary.png"))
    plt.close()
    
    print("\n--- EXPLAINABILITY COMPLETE ---")
    print(f"Visualizations successfully saved to the '{vis_dir}' folder:")
    print(" 1. actual_vs_predicted.png")
    print(" 2. feature_importance.png")
    print(" 3. shap_summary.png")

if __name__ == "__main__":
    generate_explanations()
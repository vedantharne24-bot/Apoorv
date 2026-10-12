import pandas as pd
import os
from sklearn.model_selection import train_test_split

def split_dataset():
    print("Starting Train/Test Split...")
    
    input_file = os.path.join("data", "processed", "ml_ready_dataset.csv")
    
    # Load the ML-ready data
    df = pd.read_csv(input_file)
    
    # 1. SEPARATE FEATURES (X) AND TARGET (y)
    # The target is what we want to predict
    y = df['Operation_Time_Seconds']
    
    # The features are everything else (so we drop the target column)
    X = df.drop(columns=['Operation_Time_Seconds'])
    
    # 2. PERFORM THE SPLIT
    # test_size=0.2 means 20% of data goes to testing, 80% to training
    # random_state=42 ensures we get the exact same split every time we run this code
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. SAVE THE SPLITS
    # Saving them physically ensures our later training scripts don't accidentally mix them up
    X_train.to_csv(os.path.join("data", "processed", "X_train.csv"), index=False)
    X_test.to_csv(os.path.join("data", "processed", "X_test.csv"), index=False)
    y_train.to_csv(os.path.join("data", "processed", "y_train.csv"), index=False)
    y_test.to_csv(os.path.join("data", "processed", "y_test.csv"), index=False)
    
    print("\n--- SPLIT COMPLETE ---")
    print(f"Total Operations: {len(df)}")
    print(f"Training Set: {len(X_train)} operations")
    print(f"Testing Set: {len(X_test)} operations")
    print("Files successfully saved to data/processed/")

if __name__ == "__main__":
    split_dataset()
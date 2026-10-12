import pandas as pd
import os

def audit_dataset():
    print("Starting dataset audit...")
    
    # Define file paths
    input_file = os.path.join("data", "raw", "data.csv")
    report_file = os.path.join("reports", "column_audit.csv")
    summary_file = os.path.join("reports", "dataset_summary.txt")
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"ERROR: Cannot find {input_file}. Please ensure data.csv is placed in the data/raw/ folder.")
        return

    # Load dataset
    print("Loading data... (This might take a moment for 50,000+ rows)")
    df = pd.read_csv(input_file, low_memory=False)
    
    # Calculate dataset dimensions
    num_rows, num_cols = df.shape
    
    # Perform column-level audit
    audit_data = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing_count = df[col].isnull().sum()
        missing_percentage = round((missing_count / num_rows) * 100, 2)
        unique_count = df[col].nunique()
        
        # Infer column role based on unique values
        role_hint = "Feature"
        if unique_count == 1:
            role_hint = "Constant (Drop)"
        elif unique_count == num_rows:
            role_hint = "Unique ID (Drop or Group)"
        
        audit_data.append({
            "Column_Name": col,
            "Data_Type": dtype,
            "Missing_Count": missing_count,
            "Missing_Percentage": missing_percentage,
            "Unique_Values": unique_count,
            "Possible_Role": role_hint
        })
        
    # Convert audit data to a DataFrame and save
    audit_df = pd.DataFrame(audit_data)
    audit_df.to_csv(report_file, index=False)
    
    # Generate overall summary text
    with open(summary_file, "w") as f:
        f.write("=== DATASET SUMMARY ===\n")
        f.write(f"Total Rows: {num_rows}\n")
        f.write(f"Total Columns: {num_cols}\n")
        f.write(f"Memory Usage: {df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB\n\n")
        f.write("=== PREVIEW OF FIRST 5 ROWS ===\n")
        f.write(df.head().to_string())
    
    print("\n--- AUDIT COMPLETE ---")
    print(f"Total Rows: {num_rows}")
    print(f"Total Columns: {num_cols}")
    print(f"Detailed column audit saved to: {report_file}")
    print(f"Dataset summary saved to: {summary_file}")
    print("Please open 'reports/column_audit.csv' in Excel or VS Code to review the columns.")

if __name__ == "__main__":
    audit_dataset()
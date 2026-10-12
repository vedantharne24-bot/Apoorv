import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda():
    print("Starting EDA to identify grouping keys...")
    
    input_file = os.path.join("data", "raw", "data.csv")
    df = pd.read_csv(input_file, low_memory=False)
    
    # 1. Convert time column to actual datetime objects
    # We slice the first 10,000 rows to make plotting faster and clearer
    df['time'] = pd.to_datetime(df['time'])
    df_sample = df.head(10000).copy()
    
    # 2. Look for variables that indicate the start/stop of a cycle.
    # We will plot Program_Idle, Door_open, and Spindle_angular_velocity over time.
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Spindle Velocity (Shows when the machine is physically cutting)
    plt.subplot(3, 1, 1)
    plt.plot(df_sample['time'], df_sample['Spindle_angular_velocity'], color='blue', label='Spindle Velocity')
    plt.title('Machine Activity Over Time (First 10,000 seconds)')
    plt.ylabel('Velocity')
    plt.grid(True)
    
    # Plot 2: Program Idle State (Boolean, usually True when waiting for new part)
    plt.subplot(3, 1, 2)
    # Convert boolean to integer (0 or 1) for plotting
    plt.plot(df_sample['time'], df_sample['Program_Idle'].astype(int), color='red', label='Program Idle')
    plt.ylabel('Idle State (1=Idle)')
    plt.grid(True)
    
    # Plot 3: Door Open State (Usually True when operator is changing the part)
    plt.subplot(3, 1, 3)
    plt.plot(df_sample['time'], df_sample['Door_open'].astype(int), color='green', label='Door Open')
    plt.ylabel('Door Open (1=Open)')
    plt.xlabel('Time')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join("visualizations", "cycle_identification.png")
    plt.savefig(output_path)
    print(f"EDA plot saved to {output_path}")
    
    # 3. Look for explicit counter columns
    print("\nSearching for potential cycle/program counter columns...")
    potential_counters = [col for col in df.columns if 'count' in col.lower() or 'cycle' in col.lower() or 'program' in col.lower()]
    for col in potential_counters:
        print(f"- {col}: {df[col].nunique()} unique values")

if __name__ == "__main__":
    run_eda()
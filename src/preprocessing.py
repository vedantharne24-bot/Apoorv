import pandas as pd
import os

def create_operation_dataset():
    print("Starting sensor-to-operation aggregation...")
    
    input_file = os.path.join("data", "raw", "data.csv")
    output_file = os.path.join("data", "processed", "operation_features.csv")
    
    # Load raw data
    print("Loading raw data...")
    df = pd.read_csv(input_file, low_memory=False)
    
    # 1. State Machine Logic to Identify Operations
    print("Applying state-machine logic to group operations...")
    # Convert 'Program_Idle' to boolean if it isn't already
    df['is_idle'] = df['Program_Idle'].astype(bool)
    
    # Identify exactly where the machine switches between idle and active
    df['state_change'] = df['is_idle'] != df['is_idle'].shift(1)
    
    # Cumulative sum creates a unique block ID for every contiguous state
    df['block_id'] = df['state_change'].cumsum()
    
    # We ONLY want the blocks where the machine is actively cutting (is_idle == False)
    active_blocks = df[df['is_idle'] == False]
    
    # 2. Aggregation / Feature Engineering
    print("Calculating operation-level features and total time...")
    # For every unique active block, calculate statistics
    ops_df = active_blocks.groupby('block_id').agg(
        # Since each row is 1 second, counting the rows gives us TOTAL DURATION IN SECONDS!
        Operation_Time_Seconds=('time', 'count'),
        
        # Process Features (What happened during this operation?)
        Spindle_Velocity_Mean=('Spindle_angular_velocity', 'mean'),
        Spindle_Velocity_Max=('Spindle_angular_velocity', 'max'),
        X_Feed_Mean=('X-axis_feed', 'mean'),
        Y_Feed_Mean=('Y-axis_feed', 'mean'),
        Z_Feed_Mean=('Z-axis_feed', 'mean'),
        
        # How many times did the operator open the door DURING the cycle? (Should be ~0)
        Door_Interventions=('Door_open', 'sum') 
    ).reset_index(drop=True)
    
    # 3. Clean up the data
    # Remove "micro-glitches" (e.g., machine briefly switches on for 2 seconds)
    # We will assume a valid manufacturing operation takes at least 15 seconds
    initial_ops = len(ops_df)
    ops_df = ops_df[ops_df['Operation_Time_Seconds'] >= 15]
    final_ops = len(ops_df)
    
    print(f"Removed {initial_ops - final_ops} micro-glitches (< 15 seconds).")
    
    # Save the processed dataset
    ops_df.to_csv(output_file, index=False)
    
    print("\n--- AGGREGATION COMPLETE ---")
    print(f"Total Discrete Operations Found: {final_ops}")
    print(f"Total Columns (Features + Target): {ops_df.shape[1]}")
    print(f"Processed ML dataset saved to: {output_file}")
    
    # Display preview
    print("\nPreview of the first 5 operations:")
    print(ops_df.head())

if __name__ == "__main__":
    create_operation_dataset()
import os
import pickle
import pandas as pd

# Directory containing .pkl files
pkl_dir = "datasets_MVC"
csv_dir = "datasets_MVC"
os.makedirs(csv_dir, exist_ok=True)  # Create the directory for .csv files if it doesn't exist

# Helper function to normalize data
def normalize_data(data):
    """
    Normalize the data structure to ensure consistency for DataFrame conversion.
    """
    if isinstance(data, dict):
        # Ensure all keys have values of the same length
        max_len = max(len(v) if isinstance(v, (list, tuple)) else 1 for v in data.values())
        normalized_data = {}
        for key, value in data.items():
            if isinstance(value, (list, tuple)):
                # Pad shorter lists with None
                normalized_data[key] = value + [None] * (max_len - len(value))
            else:
                # Repeat scalar values to match the max length
                normalized_data[key] = [value] * max_len
        return normalized_data
    else:
        raise ValueError("Unsupported data format. Expected a dictionary.")

# Convert each .pkl file in the directory to .csv
for file_name in os.listdir(pkl_dir):
    if file_name.endswith(".pkl"):
        pkl_path = os.path.join(pkl_dir, file_name)
        csv_path = os.path.join(csv_dir, file_name.replace(".pkl", ".csv"))
        
        # Load the .pkl file
        with open(pkl_path, "rb") as pkl_file:
            data = pickle.load(pkl_file)
        
        # Normalize the data for DataFrame conversion
        try:
            normalized_data = normalize_data(data)
            df = pd.DataFrame(normalized_data)
            
            # Rename 'labels' column to 'label' if it exists
            if 'labels' in df.columns:
                df.rename(columns={'labels': 'label'}, inplace=True)
            
            # Set escape character for special characters
            df.to_csv(csv_path, index=False, escapechar='\\')
            print(f"Converted {file_name} to {csv_path}")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

print(f"All .pkl files have been processed and converted (if possible) to .csv.")

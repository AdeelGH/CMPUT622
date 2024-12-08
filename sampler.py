import pandas as pd

def keep_random_10k_rows(input_csv, num_rows=5000):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_csv)
    
    # Check if the number of rows is less than the desired number
    if len(df) <= num_rows:
        print(f"Input file has only {len(df)} rows or less. No rows will be removed.")
        return
    
    # Randomly sample 10k rows
    df_sample = df.sample(n=num_rows, random_state=42)
    
    # Save the sampled rows back into the original file (overwrite it)
    df_sample.to_csv(input_csv, index=False)
    print(f"Retained {num_rows} rows in {input_csv}. Other rows were removed.")

# Example usage
input_file = "datasets_MST_MVC_epsilon_0.1/sst2_train.csv"  # Replace with your file path
keep_random_10k_rows(input_file)

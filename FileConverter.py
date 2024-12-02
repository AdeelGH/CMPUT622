import pickle
import pandas as pd


# Convert .pkl file to .csv
def convert_pkl_to_csv(input_file, output_file):
    # Load the .pkl file
    with open(input_file, "rb") as f:
        data = pickle.load(f)

    # Convert to DataFrame
    df = pd.DataFrame({
        "sentence1": data["perturbed_sentence1"],
        "sentence2": data.get("sentence2"),  # Handle dual-sentence datasets
        "label": data["labels"],
    })

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")


# Example usage
convert_pkl_to_csv("./processed_datasets/rte_train.pkl", "./processed_datasets/rte_train.csv")
convert_pkl_to_csv("./processed_datasets/rte_validation.pkl", "./processed_datasets/rte_validation.csv")
import pandas as pd


def fill_missing_ahp_index(input_file, output_file):
    # Load the CSV file into a DataFrame
    data = pd.read_csv(input_file)

    # Check if the 'ahp_index' column exists
    if "ahp_index" in data.columns:
        # Replace null or missing values in 'ahp_index' column with 0
        data["ahp_index"] = data["ahp_index"].fillna(0)
    else:
        print("Error: 'ahp_index' column not found in the file.")
        return

    # Save the updated DataFrame back to a new CSV file
    data.to_csv(output_file, index=False)
    print(
        f"Missing values in 'ahp_index' column have been replaced with 0. Updated file saved as {output_file}."
    )


# Specify the input and output file names
input_file = "../data/raw/ahp_label0.csv"
output_file = "../data/raw/ahp_label0_filtered.csv"

# Perform the task
fill_missing_ahp_index(input_file, output_file)

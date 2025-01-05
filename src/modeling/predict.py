import pandas as pd
import numpy as np
from keras._tf_keras.keras.models import load_model

# Load the model
model = load_model("../../models/landslide_model.h5")

# Load the CSV file
input_csv = "../../data/raw/more_predict.csv"
data = pd.read_csv(input_csv)

# Extract the first two columns
output_data = data.iloc[:, :2].copy()

# Extract the feature columns (from column index 3 onwards)
features = data.iloc[:, 2:].values

# Make predictions
predictions = model.predict(features)

# Add the predictions as the third column
output_data["Predicted_Value"] = predictions

# Save to a new CSV file
output_csv = "../../data/raw/more_output.csv"
output_data.to_csv(output_csv, index=False)

print(f"Predictions saved to {output_csv}")
